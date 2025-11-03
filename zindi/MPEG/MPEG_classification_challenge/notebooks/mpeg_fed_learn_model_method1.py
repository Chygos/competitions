
# load packages
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name="federated_learning_model_emissions1")
tracker.start()

import numpy as np
import pandas as pd
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, re, gc
from glob import glob
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal, Optional
from torch.utils.data import Dataset, DataLoader
import copy, logging
import joblib
import matplotlib.pyplot as plt
from modelling_utils import KmerAutoEncoder, KmerClassifier, AutoEncoderScaler, normalise_counts, LogisticRegression
from modelling_utils import load_encoder_model, test_loop, train_loop, print_info, set_seed
from fl_utils import Client, Server, PreprocessClientData, create_dataloader, get_embeddings, ClientDataset
import visual_utils
import time


start = time.time()


set_seed(42)


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')


# ## **Helper Functions**

def load_datasets(base_dir): 
    """
    Loads dataset

    :param base_dir: Base directory where client data are stored
    :returns pd.DataFrame. All dataset in folder path (in kmers x sampleIDs)
    """
    try:
        logger.info(f'Loading Dataset from {base_dir} directory')
        files = glob(f'{base_dir}/**/*.parquet')
        all_data = []
        if files:
            clients = list(map(lambda x: re.search('Mouth|Nasal|Stool|Skin', x, flags=re.IGNORECASE), files))
            clients = [client.group() if client else None for client in clients]
        
        for _, file in enumerate(files):
            all_data.append(pd.read_parquet(file))
    except Exception as err:
        logger.exception(f'Error loading datasets.\n{err}')
    all_data = pd.concat(all_data, axis=1)
    logger.info('Datasets loaded')
    return all_data

# Define the Global model for use for prediction on heldout test data from each client and test data (without class labels)
class GlobalModel:
    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.model = None
        self.scaler = None
    
    def fit(self, X):
        if self.model_fn.global_scaler:
            self.scaler = self.model_fn.global_scaler
        self.scaler.fit(X)
        self.model = self.model_fn.global_model.to(device)
        return self
    
    def transform(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        X = create_dataloader(X, batch_size=128, shuffle=False)
        return X
    
    def predict_proba(self, X):
        if self.model is None:
            self.fit(X)
        X_loader = self.transform(X)
        
        self.model.eval()
        probs = []
        with torch.no_grad():
            for inputs, _ in X_loader:
                outputs = self.model(inputs.to(device))
                prob = F.softmax(outputs, dim=1)
                probs.append(prob)
        probs = torch.cat(probs).cpu().numpy()
        return probs
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# load train labels (information)
train_labels = pd.read_csv('../data/Train.csv')
subjects = train_labels.groupby('SubjectID').size().sort_values(ascending=False).index
subject_ids = {k:j for k, j in zip(train_labels.filename.str.replace('.mgb', '').str.strip(), train_labels.SubjectID)}
subject_info = train_labels[['filename', 'SampleType', 'SubjectID']]
subject_info.loc[:, 'filename'] = subject_info.loc[:, 'filename'].str.replace('.mgb', '').str.strip()
subject_info =  subject_info.rename(columns={'filename': 'ID', 'SampleType': 'label', 'SubjectID': 'subject_id'}).set_index('ID')


# load client datasets
client_data = load_datasets('../data/fl_data/')
print()
client_data.shape


# transpose to ID vs kmer
client_data = client_data.T


# __Load client datasets and extract their embeddings__
# autoencoder model
encoder_scaler = joblib.load('../data/models/encoder_scaler.pkl')
encoder_model = load_encoder_model('../data/models/clr_std_latent64_epoch112_model_k8.pth')


# ### Preprocessing

# prepare Client Data and extract their emebddings
prep = PreprocessClientData(encoder_scaler, client_data, subject_info)
client_embeddings = prep.preprocess_client_data(encoder_model, num_clients=4, random_state=42, test_fraction=None)


# get input shape
input_dim = client_embeddings['0'][0].shape[1]
num_classes = 4
input_dim


# __Configuration settings for both client and server__

# configuration setings for client and server (global model)
class ClientConfig:
    config = {
        'local_epochs' : 5,
        'loss_fn' : nn.CrossEntropyLoss(),
        'optimiser' : optim.RMSprop, # optim.AdamW,
        'lr' : 5e-4, 
        'weight_decay' : 0,
        'random_state' : 42,
        'n_epoch_print' : 250,
        'verbose' : True,
        'validation_fraction': None,
        'batch_size' : 128,
        'shuffle' : True
    }

class StrategyConfig:
    config = {
        'fit_fraction' : 1., # train using 70% of the clients
        'num_rounds' : 200,
        'fraction_evaluate': 1., # evaluate on half
        'early_stopping': True,
        'patience' : 50, 
        'verbose' : True,
        'eval_metric' : 'loss',
        'random_state':42,
        'n_epoch_print': 20
    }

# Instantiate client and server class
client_config = ClientConfig.config
strategy_config = StrategyConfig.config


# - Instantiating global model
# - Assigning clients with global model parameters and their respective client configuratons

# instantiate global model and clients
global_model = KmerClassifier(input_dim=64, num_classes=num_classes, layer_mult=3)

# instantiate clients models
clients = [Client(global_model, client_embeddings.get(key), config=client_config) for key in client_embeddings.keys()]


# ## Training clients and global models
def server_fn(global_model, clients_models, strategy_config):
    """
    Server App
    
    :param global_model: Global model class
    :param client_models: Client models
    :param strategy_config: Confugration settings for server
    :returns: Server class object
    """
    server = Server(global_model, clients_models, strategy_config)
    config = server.check_server_config(server.strategy_config)
    server.train_rounds(**config)
    return server


# return for use
logger.info('Performing Federated Learning\n===============================')
fc_server = server_fn(global_model, clients, strategy_config)

model = GlobalModel(fc_server) # define global model

lab_ids = clients[0].label_ids
lab_names = list(lab_ids.keys())

# ### __Model Performance__

# Visualising global model's performance on the test data from all clients
if clients[0].test_data is not None:
    all_test = [(model.test_data, model.test_labels) for model in clients]
    X, y = zip(*all_test)
    X = np.vstack(X)
    y = np.concatenate(y)
    X.shape, y.shape
    y = list(map(lambda x: lab_ids.get(x), y))

    logger.info('Classification Report')
    visual_utils.print_classification_report(model, X, y, lab_names)
    print()

    logger.info('Classification Evaluation Metrics')
    print(visual_utils.classification_eval_metrics(model, X, y))
    print()


    logger.info('Saving Classification Performance Chart Report')
    visual_utils.classification_performance_chart_report(model, X, y, display_names=lab_names)
    plt.savefig('classification_performance_chart_report1.png')
    print()






# ### __Test Predictions (Unseen data)__

# Load test data
logger.info('Obtaining Test Predictions for submission\n')

test_data = pd.read_parquet('../data/test_8kmer.parquet')
test_data.shape
# transpose to samples x features
test_data = test_data.T

# get their sample IDs
test_idx = test_data.index.tolist()


# prepare test data
def prepare_test(X, encoder_model, encoder_scaler):
    # normalise and scale using the encoder scaler to get embeddings
    norm_X = normalise_counts(X)
    test_scaled = encoder_scaler.transform(norm_X)
    test_embeddings = get_embeddings(encoder_model, test_scaled)
    return test_embeddings


def save_file(probs, ids, filename):
    cols = lab_names
    path = 'preds'
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame()
    df['ID'] = test_idx
    df[cols] = probs
    print(df)
    filepath = os.path.join(path, f'{filename}.csv')
    df.to_csv(filepath, index=False)

# test embeddings
logger.info('Preparing test embeddings')
test_embs = prepare_test(test_data, encoder_model, encoder_scaler)

# test probabilities
logger.info('Predicting test probabilities')
test_probs = model.predict_proba(test_embs)

# save file
logger.info('Saving test predictions')
save_file(test_probs, test_idx, 'federated_learnin_preds1')

# total time
end = time.time()
mins = (end - start)/60
logger.info(f'\nTotal Time taken : {mins:.4f} Mins')

tracker.stop()