
from modelling_utils import set_seed,test_loop, train_loop, print_info, normalise_counts
from typing import Optional, Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import BaseEstimator, TransformerMixin
import copy, logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_dataloader(X, y=None, batch_size=64, shuffle=False):
    # converts input data into a Torch Dataloader
    return DataLoader(ClientDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def get_embeddings(encoder_model, X):
    """
    Get Autoencoder embeddings
    
    :param encoder_model: Autoencoder model
    :param X: Input data to obtain embeddings
    :returns Embeddings in Numpy arrays
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model.to(device)
    encoder_model.eval()
    
    X_loader = create_dataloader(X, batch_size=128, shuffle=False) if not isinstance(X, torch.utils.data.dataloader.DataLoader) else X
    
    embeddings = []
    with torch.no_grad():
        for inputs, _ in X_loader:
            _, embedding = encoder_model(inputs)
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()
    return embeddings


class ClientDataset(Dataset):
    """Creates a Pytorch Dataset"""
    def __init__(self, X, y=None): 
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is None:
            self.y = torch.zeros(len(X), dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class PreprocessClientData(BaseEstimator, TransformerMixin):
    """Preprocesses Datasets and creates datasets for clients"""
    def __init__(self, encoder_scaler, client_datasets:list, subject_info):
        """
        :param encoder_scaler: Scaler object used during Autoencoder development to scale input data
        :param class_data: List of normalised and scaled client data
        :param clients_names: List of client names
        :param subject_ids: Subjects from which samples will be split to ensure federated learning privacy
        """
        self.encoder_scaler = encoder_scaler
        self.client_datasets = client_datasets
        self.client_splits = None
        self.client_embeddings = None
        self.subject_info = subject_info # dataframe
        self.label_names = None

    def split_data(self, label_id='label', unique_id='subject_id', 
                   test_fraction=0.1, random_state=None, num_clients=4):
        """
        Randomly Split data in clients

        :param label_id: ID on subject info that represents the target class
        :param test_fraction: Fraction to set aside for test
        :param unique_id: Unique Identifier to ensure privacy
        :param random_state: Pseudo random seed
        :param num_clients: Number of clients
        """
        logger.info(f'Splitting data into {num_clients} clients...')
        client_splits = {}
        all_data = self.client_datasets
        self.subject_info = self.subject_info.reindex(all_data.index) # set data to match subject info
        
        # create client groups
        subjects = np.unique(self.subject_info[unique_id].values)
        _, groups = zip(*list(KFold(num_clients).split(subjects))) # get groups
        labels = self.subject_info[label_id].values
        self.label_names = np.unique(labels)

        ids = self.subject_info.index # ids in index
        
        # # normalise client data (CLR)
        logger.info('Normalising and scaling kmer counts...')
        all_data = self.encoder_scaler.transform(normalise_counts(all_data))
        
        for cid, idx in enumerate(groups):
            client_subjects = subjects[idx]
            subject_idx = self.subject_info['subject_id'].isin(client_subjects).values
            client_df = all_data[subject_idx]
            client_labs = labels[subject_idx]
            client_ids = ids[subject_idx]

            if test_fraction is not None:
                train_df, test_df, train_labs, test_labs, train_ids, test_ids = train_test_split(
                    client_df, client_labs, client_ids, test_size=test_fraction, random_state=random_state)
            else:
                train_ids = client_ids; train_df = client_df; train_labs = client_labs
                test_df = None; test_ids = None;  test_labs = None
            
            # append train, val, labels, ids, client name
            client_splits[str(cid)] = [train_df, test_df, train_labs, test_labs, train_ids, test_ids, self.label_names]
            self.client_splits = client_splits
        return client_splits
    
    def extract_embeddings(self, enc_model):
        """
        Extract encoder embeddings

        :param enc_model: Encoder model
        :returns: Dict[list] of client embeddings for train and validation sets
        """
        logger.info('Extracting client data embeddings....')
        self.client_embeddings = copy.copy(self.client_splits)

        for cid, val in self.client_splits.items():
            if val[1] is not None:
                train_test = [get_embeddings(enc_model, x) for x in val[:2]]
                self.client_embeddings[cid][:2] = train_test
            else:
                self.client_embeddings[cid][0] = get_embeddings(enc_model, val[0])
        return self.client_embeddings
    
    def preprocess_client_data(self, enc_model, label_id='label', unique_id='subject_id', 
                               test_fraction=0.1, random_state=None, num_clients=4):
        """
        Splits data and extracts encoder embeddings
        
        :param enc_model: Encoder model
        :param label_id: None | str | pd.DataFrame | pd.Series | np.ndarray
        :param test_fraction: Fraction for validation
        :param random_state: Pseudorandom generator for reproducibility
        :param num_clients: Number of clients
        """
        logger.info('Data Preprocessing\n=========================')
        X = self.split_data(label_id, unique_id, test_fraction, random_state, num_clients)
        X = self.extract_embeddings(enc_model)
        logger.info('Feature Extraction completed.\n')
        return X


class Client:
    """
    Client Object class
    """
    def __init__(self, model, data, config, scale=True, scale_type:Optional[Literal['ss', 'mm']]='ss'):
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.train_ids, self.test_ids, self.label_names = data
        self.model = copy.deepcopy(model)
        self.scale = scale
        self.scale_type = scale_type
        self.scaler_ = None
        self.config = self.check_client_config(config)
        self.optimiser = None
        self.label_ids = dict(zip(self.label_names, range(len(self.label_names))))
        self.val_data = None
        self.val_labels = None
    
    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())
    
    def reset_optimiser(self):
        if self.optimiser is not None:
            self.optimiser.state.clear()
    
    def set_parameters(self, params:list):
        state_dict = self.model.state_dict()
        if len(params) != len(state_dict):
            raise ValueError("Parameter length mismatch")
        new_state = {k: p for (k, _), p in zip(state_dict.items(), params)}
        self.model.load_state_dict(new_state)
    
    def check_client_config(self, config):
        valid = set(['loss_fn', 'optimiser', 'local_epochs', 'lr', 'weight_decay', 'validation_fraction',
                     'random_state', 'n_epoch_print', 'verbose', 'batch_size', 'shuffle'])
        invalid = valid - set(config)
        if invalid:
            raise ValueError(f'Invalid config settings: {invalid}.\nRecognised configs are {valid}')
        
        if not callable(config['optimiser']):
            raise TypeError ('optimiser must be a callable')
        return config
        
    def fit(self, X):
        if self.scale:
            if self.scaler_ is None:
                self.scaler_ = MinMaxScaler() if self.scale_type == 'mm' else StandardScaler()
                self.scaler_.fit(X)
        return self
    
    def transform(self, X):
        X = self.scaler_.transform(X) if self.scaler_ is not None else X
        return X
    
    def label_tonumeric(self, y):
        dtype_ = np.array(y).dtype
        if dtype_ != 'int'  and dtype_ != 'float':
            y = list(map(lambda x: self.label_ids[x], y)) # convert to numeric
            y = np.array(y).astype(np.int32)
        return y
    
    def train_model(self, loss_fn, optimiser, local_epochs=5, random_state=None, 
                    verbose=False, n_epoch_print=None, batch_size=64, shuffle=True):
        # set seed
        if random_state is not None:
            set_seed(random_state)
        
        # preprocess data
        self.fit(self.train_data)
        Xtrain = self.transform(self.train_data)
        ytrain = np.array(self.train_labels)
        ytrain = self.label_tonumeric(ytrain)
        train_loader = create_dataloader(Xtrain, ytrain, batch_size=128, shuffle=shuffle)

        self.optimiser = optimiser(self.model.parameters(), 
                                   lr=self.config['lr'], 
                                   weight_decay=self.config['weight_decay'])
        self.loss_fn = loss_fn
        
        # loop
        best_loss = float('inf')
        for epoch in range(1, 1+local_epochs):
            train_loss = train_loop(self.model, self.loss_fn, self.optimiser, train_loader)
            if self.val_data is not None:
                xval = self.transform(self.val_data)
                yval = self.label_tonumeric(self.val_labels)
                val_loader = create_dataloader(xval, yval, batch_size=64, shuffle=True)
                test_loss = test_loop(self.model, self.loss_fn, val_loader)['loss']
            if verbose and self.val_data is not None:
                if print_info(epoch, local_epochs, n_epoch_print) and val_loader is not None:
                    logger.info(f'Epoch: {epoch}/{local_epochs}, Train Loss: {train_loss:.5f}, Val Loss: {test_loss:.5f}')
                else:
                    logger.info(f'Epoch: {epoch}/{local_epochs}, Train Loss: {train_loss:.5f}')
        return self.get_parameters(), len(Xtrain) # return client's model parameters and sample size
    
    def create_test_split(self, validation_fraction, random_state=None):
        if validation_fraction is None: 
            trainx, trainy = self.train_data.copy(), self.train_labels.copy()
            valx, valy = None, None
        else:
            trainx, valx = train_test_split(self.train_data,
                                            test_size=validation_fraction, 
                                            random_state=random_state)
            trainy, valy = train_test_split(self.train_labels, 
                                            test_size=validation_fraction, 
                                            random_state=random_state)
        return trainx, trainy, valx, valy
    
    def get_eval_data(self, eval_type, raw=False):
        if eval_type == 'val':
            if self.val_data is not None and self.val_labels is not None:
                Xeval = self.val_data if raw else self.transform(self.val_data)
                yeval = np.array(self.val_labels)
                yeval = self.label_tonumeric(yeval)
                return Xeval, yeval
        elif eval_type == 'test':
            if self.test_data is not None and self.test_data is not None:
                Xeval = self.test_data if raw else self.transform(self.test_data)
                yeval = np.array(self.test_labels)
                yeval = self.label_tonumeric(yeval)
                return Xeval, yeval
        elif eval_type == 'train': # there must be train data
            Xeval = self.train_data if raw else self.transform(self.train_data)
            yeval = np.array(self.train_labels)
            yeval = self.label_tonumeric(yeval)
            return Xeval, yeval
        return None, None
    
    def get_scaling_stats(self):
        if self.scale_type == 'ss':
            means_ = np.mean(self.train_data, axis=0)
            vars_ = np.var(self.train_data, axis=0)
            counts = len(self.train_data)
            return {'mean': means_, 'var':vars_, 'nsamples':counts}
        elif self.scale_type == 'mm':
            mins_ = np.min(self.train_data, axis=0)
            maxs_ = np.max(self.train_data, axis=0)
            return {'min': mins_, 'max':maxs_}
    
    def evaluate_model(self, model_obj, eval_type='val'):
        model = model_obj.model
        X, y = model_obj.get_eval_data(eval_type, raw=False)
        if X is None or y is None:
            print(f"[Warning] Skipping evaluation for eval_type='{eval_type}' due to missing data.")
            return {'loss': None, 'acc': None}
        val_loader = create_dataloader(X, y, batch_size=32)
        res = test_loop(model, self.config['loss_fn'], val_loader)
        return res


class Server:
    """
    Server for monitoring training of client model
    During each training round, clients are selected for training and evaluation. 
    Selection depends on the fit_fraction and evaluate_fit keys in the server configuration settings

    Global model updates are done after training clients and parameters updated by federated averaging.
    Weighted averaging is done based on the proportion of samples in each client used for training

    If scaling is done by client, global model gets its client mean and variance values from the first model. 
    This is updated at every training round by weighted averaging the training samples mean and variance for all clients used for training
    Global model redistributes this to all the clients. 
    
    Also, after each training round, all client parameters are updated by the new parameters of the global model after federated averaging
    For validation and test performance evaluation, performance is done at the client and server levels. Only test performance on by the global model.
    All validation metrics from each client are aggregated by applying weights from their individual validation samples, while the global uses its learned parameters
    to evaluate the validation data of clients selected for evaluation. Last client parameter update is done prior to the last number of training rounds.
    After that, global model evaluates the test data from all clients in the server.
    """
    def __init__(self, global_model, client_models, strategy_config):
        self.client_models = client_models
        self.global_model = global_model # gets the global model from which other
        self.strategy_config = self.check_server_config(strategy_config)
        self.global_scaler = None

    def fedavg(self, updates, sample_sizes):
        """Aggregates parameters of client models and updates global parameters"""
        sample_wts = np.array(sample_sizes)/np.sum(sample_sizes)
        avg = [np.zeros_like(p, dtype=np.float32) for p in updates[0]] # instantiate parameter shape in first client
        for i, client_wts in enumerate(updates):
            sample_wt = sample_wts[i]
            for j, param in enumerate(client_wts):
                avg[j] += sample_wt * param
        avg = list(map(torch.tensor, avg))
        return avg
    
    def check_server_config(self, config):
        valid = set(['num_rounds', 'early_stopping', 'verbose', 'fit_fraction', 
                     'fraction_evaluate', 'patience', 'random_state', 'eval_metric', 
                     'n_epoch_print'])
        invalid = set(valid) - set(config.keys())
        if invalid:
            raise ValueError(f'Invalid config settings: {invalid}.\nRecognised configs are {valid}')
        return config
    
    def set_global_params(self, params):
        """Set parameter of global model"""
        state_dict = self.global_model.state_dict()
        new_state = {k: p for (k, _), p in zip(state_dict.items(), params)}
        self.global_model.load_state_dict(new_state)
        
    def compute_global_scaler(self):
        """
        Compute scaler of global model
        """
        # temporarily fit on first client's training samples and update later
        scaler_type = self.client_models[0].scale_type
        self.global_scaler = StandardScaler() if scaler_type == 'ss' else MinMaxScaler()
        self.global_scaler.fit(self.client_models[0].train_data)  # temp fit for attributes
        
        # update here by getting all clients scaling stats
        if scaler_type == 'ss':
            all_means, all_vars, all_counts = [],[],[]
            for client in self.client_models:
                res = client.get_scaling_stats() # returns a dictionary
                all_means.append(res['mean'])
                all_vars.append(res['var'])
                all_counts.append(res['nsamples'])

            total_count = np.sum(all_counts)
            weights = np.array(all_counts) / total_count

            global_mean = np.sum(np.stack(all_means, axis=1) * weights, axis=1)
            global_var = np.sum(
                (np.stack(all_vars, axis=1) + (np.stack(all_means, axis=1) - global_mean[:, None])**2) * weights, axis=1
            )
            self.global_scaler.mean_ = global_mean
            self.global_scaler.var_ = global_var
            self.global_scaler.scale_ = np.sqrt(global_var)

        elif scaler_type == 'mm':
            all_mins = []; all_maxs = []; 
            for client in self.client_models:
                res = client.get_scaling_stats()
                all_mins.append(res['min'])
                all_maxs.append(res['max'])
            mins = np.stack(mins, axis=1)  # (d, num_clients)
            maxs = np.stack(maxs, axis=1)  # (d, num_clients)

            global_min = np.min(mins, axis=1)
            global_max = np.max(maxs, axis=1)
            global_range = global_max - global_min

            self.global_scaler.data_min_ = global_min
            self.global_scaler.data_max_ = global_max
            self.global_scaler.data_range_ = global_range
            self.global_scaler.scale_ = 1.0 / global_range
            self.global_scaler.min_ = 0.0
    
    def distribute_global_scaler(self):
        """Send the fitted global scaler to all clients."""
        for client in self.client_models:
            client.scaler_ = copy.deepcopy(self.global_scaler)
    
    def evaluate_global(self, criterion, selected_client_idx=None, eval_type='val'):
        """Evaluate performance of global model on test/validation data"""
        all_X, all_y = [], []
        if selected_client_idx is None:
            selected_client_idx = range(len(self.client_models))

        for idx in selected_client_idx:
            if eval_type == 'val':
                eval_data = self.client_models[idx].get_eval_data(raw=True, eval_type=eval_type)
            elif eval_type == 'test':
                eval_data = self.client_models[idx].get_eval_data(raw=True, eval_type=eval_type)
            elif eval_type == 'train':
                eval_data = self.client_models[idx].get_eval_data(raw=True, eval_type=eval_type)
            if eval_data:
                X, y = eval_data
                all_X.append(X); all_y.append(y)
        if not all_X: return None
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # preprocess
        if self.global_scaler is not None:
            X = self.global_scaler.transform(X)

        val_loader = create_dataloader(X, y, batch_size=64, shuffle=False)
        res = test_loop(self.global_model, criterion, val_loader)
        return res

    def train_rounds(self, num_rounds=20, fit_fraction=1., fraction_evaluate=1., 
                     early_stopping=False, verbose=False, patience=5, 
                     random_state=None, eval_metric='loss', n_epoch_print=5):
        """
        Train clients at each number of rounds
        :param num_rounds: Number of training rounds
        :param fit_fraction: The fraction of clients for training
        :param fraction_evaluate: Fraction of clients for validation evaluation
        :param early_stopping: Boolean. Activat Early stopping
        :param verbose: To print validation at client and server level
        :param patience: Number of rounds if model has not improved
        :param random_state: Pseudorandom generator
        :param eval_metric: Evaluation metric to be returned. loss (loss) or acc (accuracy)
        :param n_epoch_print: Number of printing rounds
        :returns None
        """
        
        # create validation data for all client if validation fraction is not None in client configuration
        if self.client_models[0].config.get('validation_fraction'):
            for client in self.client_models:
                trainx, trainy, valx, valy = client.create_test_split(
                    validation_fraction = client.config.get('validation_fraction'), 
                    random_state = client.config.get('random_state')
                )
                client.train_data, client.train_labels = trainx, trainy
                client.val_data, client.val_labels = valx, valy
        
        if random_state is not None:
            set_seed(random_state)

        # compute and redistribute global scaler to all clients
        self.compute_global_scaler()
        self.distribute_global_scaler()

        num_clients = len(self.client_models)
        clients_to_select = int(np.ceil(fit_fraction * num_clients))
        
        if self.client_models[0].config.get('validation_fraction'): # if no validation, none is seleceted for evaluation
            eval_clients = int(np.ceil(fraction_evaluate*num_clients)) # get fraction of clients for evaluation
        else:
            eval_clients = 0
        
        logger.info(f'Total Clients: {num_clients}\n{clients_to_select} clients selected for training and {eval_clients} for evaluation\n')
        
        # Sample clients for training and evalaution
        np.random.seed(random_state)
        selected_clients_idx = np.random.choice(range(num_clients), size=clients_to_select)
        # Select for evaluation (if validation fraction is None or client has no val data)
        eval_idx = np.random.choice(range(num_clients), size=eval_clients) if eval_clients > 0 else None

        best_server_metric= {'loss':None, 'acc': None}
        best_epoch = None # epoch/round with best val score
        eval_metric = self.strategy_config.get('eval_metric', 'loss')
        patience_counter = 0
        iter_counter = 1
        logger.info('Training started in server\n========================')
        for rnd in range(1, num_rounds+1):
            param_updates, sample_size, test_metrics = [], [], []
            for idx in selected_clients_idx:
                # client_seed = int(random_state + idx) if random_state is not None else None
                client_config = self.client_models[idx].config
                # select keys that are arguments in client's train_model function
                model_config = {k:v for k, v in client_config.items() if k not in ['lr', 'weight_decay', 'validation_fraction']}
                # model_config['random_state'] = int(client_seed + random_state) if random_state is not None else None # update client seed
                param, nsamples = self.client_models[idx].train_model(**model_config) # returns client parameters and number of training samples
                param = [p.detach().clone().numpy() for p in param.values()]
                
                param_updates.append(param)
                sample_size.append(nsamples)

                # get train metrics
                
            
            # aggregate using fedavg
            new_global_wts = self.fedavg(param_updates, sample_size)
            self.set_global_params(new_global_wts) # set global param

            # Only calculate train metrics if no validation data for evaluation
            # get train metrics after all clients have been trained and parameters of global model have been updated
            if eval_clients  == 0:
                train_metrics = self.evaluate_global(self.client_models[0].config['loss_fn'], eval_type='train') # get train of all clients and evaluate using global model
                if train_metrics:
                    if verbose:
                        if print_info(rnd, num_rounds, n_epoch_print):
                            logger.info(f'Round {rnd} Train: Loss: {train_metrics['loss']}, Accuracy: {train_metrics['acc']}')
                    if eval_metric == 'loss':
                        criteria = best_server_metric['loss'] is None or train_metrics['loss'] < best_server_metric['loss']
                    else:
                        criteria = best_server_metric['acc'] is None or train_metrics['acc'] > best_server_metric['acc']
                    if criteria:
                        best_server_metric[eval_metric] = train_metrics[eval_metric]
                        other_metric = list(set(['loss', 'acc']) - set([eval_metric]))
                        other_metric = ''.join(map(str, other_metric))
                        best_server_metric[other_metric] = train_metrics[other_metric]
                        best_epoch = rnd
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    # early stopping
                    if early_stopping and patience_counter >= patience:
                        print(f'Early Stopping around round {rnd}')
                        break
            # client-side evaluation (validation data) if validation_fraction in client configuration settings
            if eval_idx is not None:
                client_metrics = []
                client_val_samples = []
                for index in eval_idx:
                    # get metrics from each client selected for evaluation and get their combined values
                    client_obj= self.client_models[index]
                    metrics = client_obj.evaluate_model(client_obj, 'val')
                    test_size = len(client_obj.val_data)
                    client_metrics.append(metrics)
                    client_val_samples.append(test_size)
                if client_metrics:
                    total_samples = sum(client_val_samples)
                    sample_wts = np.array(client_val_samples)/total_samples
                    avg_loss = sum([m['loss']*wt for m, wt in zip(client_metrics, sample_wts)])
                    avg_acc = sum([m['acc']*wt for m, wt in zip(client_metrics, sample_wts)])
                    if verbose:
                        if print_info(rnd, num_rounds, n_epoch_print):
                            logger.info(f'Round {rnd}\n===========')
                            logger.info(f'Client: Loss: {avg_loss}, Accuracy: {avg_acc}')
            
                # server-side evaluation
                server_eval = self.evaluate_global(self.client_models[0].config['loss_fn'], eval_idx, 'val')
                if server_eval:
                    if verbose:
                        if print_info(rnd, num_rounds, n_epoch_print):
                            logger.info(f"Server: Loss: {server_eval['loss']}, Accuracy: {server_eval['acc']}")
                            print('+=+'*15, '\n')
                    
                    # monitor performance
                    if eval_metric == 'loss':
                        criteria = best_server_metric['loss'] is None or server_eval['loss'] < best_server_metric['loss']
                    else:
                        criteria = best_server_metric['acc'] is None or server_eval['acc'] > best_server_metric['acc']
                    if criteria:
                        best_server_metric[eval_metric] = server_eval[eval_metric]
                        other_metric = list(set(['loss', 'acc']) - set([eval_metric]))
                        other_metric = ''.join(map(str, other_metric))
                        best_server_metric[other_metric] = server_eval[other_metric]
                        best_epoch = rnd
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # early stopping
                    if early_stopping and patience_counter >= patience:
                        print(f'Early Stopping around round {rnd}')
                        break
            
            # set client parameter with that of the new_global weights
            while iter_counter < num_rounds:
                for client in self.client_models:
                    client.set_parameters(new_global_wts)
                    client.reset_optimiser() # reset optimiser
                iter_counter += 1

        if self.client_models[0].val_data is None:
            logger.info(f"\nTrain data: Best epoch: {best_epoch}, Loss: {best_server_metric['loss']}, Accuracy: {best_server_metric['acc']}")
        if self.client_models[0].val_data is not None:
            logger.info(f"\nVal data: Best epoch: {best_epoch}, Loss: {best_server_metric['loss']}, Accuracy: {best_server_metric['acc']}")
        # evaluate global model on all client test data
        if self.client_models[0].test_data is not None:
            # selects test data from all clients for final evaluation, if present
            server_eval = self.evaluate_global(self.client_models[0].config['loss_fn'],  eval_type='test')
            logger.info(f"Test Evaluation: Loss: {server_eval['loss']}, Accuracy: {server_eval['acc']}")
        
        logger.info('\nTraining Completed!')
