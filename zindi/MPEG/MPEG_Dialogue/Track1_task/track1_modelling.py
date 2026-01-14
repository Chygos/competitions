
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import random, os, logging, copy

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn import metrics
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def normalise_counts(X):
    """
    Normalises kmer counts using Centered Log Ratio (CLR)
    
    :param X: Pandas DataFrame | Numpy ndarray
    :returns CLR-normalised counts
    """
    logX = np.log1p(np.array(X))
    gm = np.mean(logX, axis=1, keepdims=True)
    norm_counts = logX - gm
    return norm_counts.astype(np.float32)


def tensor_to_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()


def print_info(epoch, epochs, n_epoch_print):
    """
    Print epoch info during training
    """
    if (epoch % n_epoch_print) == 0 or epoch == 1 or epoch == epochs:
        return True

# Train and test loop
def train_loop(model, criterion, optimiser, train_loader, device=device, alpha=0, l1_ratio=0):
    model.to(device) # move to device
    n_batch = len(train_loader) # number of batches in train loader
    total_loss = 0.
    model.train()
    for inputs, labels in train_loader:
        optimiser.zero_grad()
        outputs = model(inputs.to(device))
        base_loss = criterion(outputs, labels.to(device))
        loss = elastic_net_loss(base_loss, model, alpha, l1_ratio)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / n_batch

def test_loop(model, criterion, val_loader, yscaler, device=device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    
    n = len(val_loader.dataset)
    nbatch = len(val_loader)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = tensor_to_numpy(outputs)
            outputs = yscaler.inverse_transform(outputs)
            outputs = torch.from_numpy(outputs).to(device)
            labels = tensor_to_numpy(labels)
            labels = yscaler.inverse_transform(labels)
            labels = torch.from_numpy(labels).to(device)
            loss = np.sum((tensor_to_numpy(labels)- tensor_to_numpy(outputs))**2)
            test_loss += loss
            # loss = criterion(outputs, labels)
            # test_loss += loss.item()  # sum loss over samples
    avg_loss = test_loss / nbatch
    return np.sqrt(avg_loss)


def elastic_net_loss(base_loss, model, alpha=1.0, l1_ratio=0.5):
    weights = [param for name, param in model.named_parameters() if 'bias' not in name]
    n_params = sum(p.numel() for p in weights)
    l1_penalty = sum(torch.norm(param, 1) for name, param in model.named_parameters() if 'bias' not in name)/n_params
    l2_penalty = sum(torch.norm(param, 2)**2 for name, param in model.named_parameters() if 'bias' not in name)/n_params
    
    elastic_penalty = alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) / 2 * l2_penalty)
    return base_loss + elastic_penalty


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc1(x)


class CytokineModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.LayerNorm(128)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.LayerNorm(64)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        return x        

class CytokineModel1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.LayerNorm(128)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = self.fc2(x)
        return x        


train = pd.read_parquet('../data/train_8kmer_all.parquet')
test = pd.read_parquet('../data/test_8kmer_all.parquet')
cytokine_df = pd.read_csv('../data/cytokine_profiles.csv')
train_info = pd.read_csv('../data/Train_Subjects.csv')
train_info2 = pd.read_csv('../data/Train.csv')

selected_kmers = pd.read_csv('../data/selected_kmers_unscaled_var.csv', index_col=0).index

coef_scores = np.load('../data/fs_by_multitask.npy')



train.shape, cytokine_df.shape, train_info.shape, train_info2.shape


train.head()

# Data cleaning
# - **Merging cytokines data with metadata**


merged_data = (
    cytokine_df.loc[:, 'SampleID':'CollectionDate'].assign(CollectionDate = pd.to_datetime(cytokine_df.CollectionDate))
    .drop(columns=['Plate'])
    .merge(train_info2
           .rename(columns={'filename': 'ID', 'SampleType': 'site'}), on=['SampleID'])
    .merge(train_info[['SubjectID', 'FPG_class', 'Class', 'Gender', 'Adj.age', 'BMI']], on='SubjectID')
)


# remove .mgb from ID
merged_data.ID = merged_data['ID'].str.replace('.mgb', '').str.strip()

# - **Selecting samples with cytokine data from train data**
# 
# - **Obtaining the cytokines for use later**


# intersection IDs (selecting samples in train and merged data ie cytokines)
samples = train.columns.intersection(merged_data.ID)
target = cytokine_df.loc[:, 'IL17F':'CHEX4'].columns
len(samples), len(target)


# Align SampleID in merged data to match those in train 
train = train.reindex(merged_data.ID, axis=1) # axis=1 because they are as columns
train.shape


# ## Feature Selection
# 
# - **Selecting kmers with variance above a cutoff**
# 
# - In a previous notebook we had performed variance thresholding to select kmers with their CLR-normalised variance above a cutoff. The cutoff was selected based on the elbow method. Kmers were saved.
# 
# - Because train data wass saved as kmers as rows and sample IDs as features, we will need to transpose it so that kmers become columns while samples, rows. Thereafter, we will select our kmers


# select kmers and transpose
train = train.loc[selected_kmers].T


# realigning samples to match the rows in train
samples = samples.reindex(train.index)[0]


merged_data.head()

merged_data.FPG_class.unique(), merged_data.Class.unique()


merged_data.FPG_class.value_counts()


merged_data.Class.value_counts()


merged_data.groupby('SubjectID').BMI.mean().describe()


train.head()


def bin_age(age):
    if age < 18:
        return 'Child'
    elif age < 30:
        return 'Young Adult'
    elif age < 45:
        return 'Adult'
    elif age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'


merged_data['BMI_group'] = pd.cut(merged_data['BMI'], bins=[0, 18.5, 25, 30, np.inf],
                                  labels=['underweight', 'normal', 'overweight', 'obese']).astype(str)

merged_data['site_metabolic_obesity'] = merged_data['site'] + '_' + merged_data['FPG_class'] + '_' + merged_data['BMI_group']
merged_data['site_FPG'] = merged_data['site'] + '_' + merged_data['FPG_class']
merged_data['BMI_FPG'] = merged_data['BMI_group'] + '_' + merged_data['FPG_class']
merged_data['site_BMI'] = merged_data['site'] + '_' + merged_data['BMI_group']
merged_data['age_gp'] = merged_data['Adj.age'].apply(bin_age)

merged_data['site_age'] = merged_data['site'] + '_' + merged_data['age_gp']
merged_data['FPG_age'] = merged_data['age_gp'] + '_' + merged_data['FPG_class']
merged_data['age_BMI'] = merged_data['age_gp'] + '_' + merged_data['BMI_group']


# one hot encoding covariates
covariates = merged_data[['site_BMI', 'site_FPG', 'BMI_FPG', 'Gender', 
                          'site_metabolic_obesity', 'site_age', 'FPG_age', 'age_BMI']]
covariates = pd.get_dummies(covariates, drop_first=True, dtype=np.uint32)


covariates.shape, train.shape, merged_data.shape


class ModelPipeline:
    def __init__(self, model, scale=True, scale_columns=None, scale_type='std'):
        self.scale = scale
        self.model = model
        self.scaler_ = None
        self.scale_type = scale_type
        self.scale_columns = scale_columns
        self.scale_columns_idx = None
    def fit(self, X):
        if isinstance(X, pd.DataFrame) and self.scale_columns is not None:
            self.scale_columns_idx = X.columns.get_indexer(self.scale_columns).tolist()
        X = np.array(X).astype(np.float32)
        if self.scale:
            self.scaler_ = StandardScaler() if self.scale_type == 'std' else MinMaxScaler()
            if self.scale_columns is None:
                self.scaler_.fit(X)
            elif self.scale_columns_idx is not None:
                self.scaler_.fit(X[:, self.scale_columns_idx])
            else:
                self.scaler_.fit(X)
        return self
    def transform(self, X):
        X = np.array(X).astype(np.float32)
        if self.scaler_ is not None and self.scale_columns is None:
            X = self.scaler_.transform(X)
        elif self.scale_columns_idx:
            X[:, self.scale_columns_idx] = self.scaler_.transform(X[:, self.scale_columns_idx])
        return X
    def create_dataloader(self, X, y, batch_size=32, shuffle=False):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y) if y is not None else torch.ones(len(X))
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
        
    def train_model(self, optimiser_fn, criterion,ytransformer,
                    xtrain, ytrain, xval=None, yval=None, 
                    epochs=50, early_stopping=False, n_epoch_print=10, 
                    patience=10, device=device, batch_size=32, shuffle=True, 
                    seed=42, alpha=0, l1_ratio=0,return_state_dict=True):
        
        optimiser = optimiser_fn(self.model.parameters())
        
        self.fit(xtrain)
        xtrain = self.transform(xtrain)
        train_loader = self.create_dataloader(xtrain, ytrain, batch_size=batch_size, shuffle=shuffle)
        
        if xval is not None:
            xval = self.transform(xval)
            val_loader = self.create_dataloader(xval, yval, batch_size=100, shuffle=False)
        
        best_epoch = None; wait = 0; best_train_rmse = float('inf')
        best_loss = float('inf'); best_val_trainloss = float('inf')
        best_model_state = None
        
        if seed is not None:
            set_seed(seed)
        for epoch in range(1, epochs+1):
            train_loss = train_loop(self.model, criterion, optimiser, train_loader, 
                                    device=device, alpha=alpha, l1_ratio=l1_ratio)
            train_rmse = np.sqrt(test_loop(self.model, criterion, train_loader, ytransformer, device=device))

            if val_loader is not None:
                val_loss = np.sqrt(test_loop(self.model, criterion, val_loader, ytransformer, device=device))
                if n_epoch_print is not None and print_info(epoch, epochs, n_epoch_print):
                    print(f'Epoch: {epoch}, Train Loss: {train_loss:.5f}, Train RMSE: {train_rmse:.5f}, Val RMSE: {val_loss}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_val_trainloss = train_loss
                    best_train_rmse = train_rmse
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
            else:
                if n_epoch_print is not None and print_info(epoch, epochs, n_epoch_print):
                        print(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Train RMSE: {train_rmse:.5f}')
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_train_rmse = train_rmse
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
            if early_stopping and wait >= patience:
                break
        if val_loader is not None:
            print(f'Best Epoch: {best_epoch}, Train Loss: {best_val_trainloss}, Train RMSE: {best_train_rmse}, Val RMSE: {best_loss}')
        else:
            print(f'Best Epoch: {best_epoch}, Train Loss: {best_loss}, Train RMSE: {best_train_rmse}')
        if return_state_dict:
            self.model.load_state_dict(best_model_state)
    def predict(self, X):
        self.model.eval()
        xtest = self.transform(X)
        test_loader = self.create_dataloader(xtest, None, batch_size=200, shuffle=False)
        preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                output = self.model(inputs.to(device))
                preds.append(output)
        preds = torch.cat(preds).detach().cpu().numpy()
        return preds


# ### Normalising and merging our data and covariates


norm_counts  = normalise_counts(train)

norm_counts = pd.DataFrame(norm_counts, columns=train.columns, index=train.index)

all_df = pd.concat([norm_counts.reset_index(drop=True), covariates], axis=1)
all_df.index= train.index

all_df.shape

all_df.head()


yscaler = StandardScaler() #PowerTransformer(method='yeo-johnson')

yscaler.fit(merged_data[target].values)
yscaled = yscaler.transform(merged_data[target].values).astype(np.float32)


# ## Model Training

# In a different notebook, we selected features using multitask elastic net using 5-cv score. Out of the 9155 features, about 1520 were selected by the model. Therefore, we will use these selected features to fit our final model

coef_scores.shape

(coef_scores != 0).any(axis=0).shape

# selecting non-zero features across all cytokines
selected_features = all_df.columns[(coef_scores != 0).any(axis=0)]

xtrain, xval, ytrain, yval = train_test_split(all_df[selected_features], yscaled, 
                                              random_state=123, test_size=0.2)
xtrain.shape, xval.shape, ytrain.shape, yval.shape

input_dim = xtrain.shape[1] #- covariates.shape[1]
output_dim = len(target)
input_dim, output_dim


categorical_features = selected_features.difference(selected_kmers)
scaled_features = selected_features.intersection(selected_kmers)


len(scaled_features), len(categorical_features)


# ### __Penalised Linear Regression (Pytorch)__


criterion = nn.MSELoss()


model1 = LinearRegression(input_dim, output_dim)
model1_pipe = ModelPipeline(model1, scale_columns = scaled_features, scale_type='minmax')
optimiser1 = partial(optim.SGD, lr=0.1, weight_decay=0)


model1_pipe.train_model(optimiser1, criterion, yscaler, xtrain, ytrain, xval, yval, 
                        epochs=2000, n_epoch_print=100, early_stopping=True, 
                        patience=100, batch_size=128, alpha=0.1, l1_ratio=0.5) #128 minmax


os.makedirs('models', exist_ok=True)

torch.save(model1.state_dict(), 'models/linear_regression.pth')


# ### __Deep Learning__


model2 = CytokineModel(input_dim, output_dim)
model2_pipe = ModelPipeline(model2, scale_columns = scaled_features, scale_type='minmax')
optimiser2 = partial(optim.AdamW, lr=0.0005, weight_decay=0)


model4 = CytokineModel1(input_dim, output_dim)
model4_pipe = ModelPipeline(model4, scale_columns = scaled_features, scale_type='minmax')
optimiser2 = partial(optim.AdamW, lr=0.0005, weight_decay=0)


model4_pipe.train_model(optimiser2, criterion, yscaler, xtrain, ytrain, xval, yval, 
                        epochs=1000, n_epoch_print=100, alpha=1, l1_ratio=0.5,
                        early_stopping=True, patience=100)


model2_pipe.train_model(optimiser2, criterion, yscaler, xtrain, ytrain, xval, yval, 
                        epochs=1000, alpha=1, l1_ratio=0.5, n_epoch_print=100, 
                        early_stopping=True, patience=100)


torch.save(model4.state_dict(), 'models/model2_one_layer.pth')
torch.save(model2.state_dict(), 'models/model2_two_layer.pth')


# ## Cross validation


merged_data.shape,all_df.shape
groups = merged_data.SubjectID

# splitting by subject ID


gkfold = GroupKFold()
kfold = KFold(random_state=123, shuffle=True)


def cross_validation(model, X, y, cv_fold, groups=None, scale_type='minmax'):
    scores = []
    for i, (tr_idx, val_idx) in enumerate(cv_fold.split(X, y, groups=groups)):
        xtrain, ytrain = X.iloc[tr_idx, :], y[tr_idx, :]
        xval, yval = X.iloc[val_idx, :], y[val_idx, :]
        if callable(model):
            model_pipe = ModelPipeline(model(), scale_columns = scaled_features, scale_type=scale_type)
            optimiser_fn = optimiser1 if 'LinearRegression' == model().__class__.__name__ else optimiser2
            
            model_pipe.train_model(optimiser_fn, criterion, yscaler, xtrain, ytrain, xval, yval, 
                                   early_stopping=True, epochs=2000, n_epoch_print=500, 
                                   patience=100, alpha=1, l1_ratio=0.5, return_state_dict=False)
            ypreds = model_pipe.predict(xval)
            ypreds_orig = yscaler.inverse_transform(ypreds)
            yval_orig = yscaler.inverse_transform(yval)
            rmse = metrics.mean_squared_error(yval_orig, ypreds_orig, squared=False)
            scores.append(rmse)
        else:
            model.fit(xtrain, ytrain)
            ypreds = model.predict(xval)
            ypreds_orig = yscaler.inverse_transform(ypreds)
            yval_orig = yscaler.inverse_transform(yval)
            rmse = metrics.mean_squared_error(yval_orig, ypreds_orig, squared=False)
            scores.append(rmse)
        print(f'\nFold {i+1}: RMSE: {rmse:.7f}\n')
    
    ci_95 = np.quantile(scores, [0.025, 0.975])
    avg_rmse = np.mean(scores); std_rmse = np.std(scores)
    print(f'\nAvg RMSE: {avg_rmse:.5f} +- {std_rmse:.3f}\nCI (95%): [{ci_95[0]:.5f}, {ci_95[1]:.5f}]')
        

X = all_df[selected_features]
y = yscaled

X.shape, y.shape

optimiser1 = partial(optim.SGD, lr=0.01, weight_decay=0)
optimiser2 = partial(optim.AdamW, lr=0.0005, weight_decay=0)

model1 = lambda: LinearRegression(input_dim, output_dim)
model2 = lambda: CytokineModel(input_dim, output_dim)
model4 = lambda: CytokineModel1(input_dim, output_dim)


cross_validation(model1, X, y, kfold, groups)


cross_validation(model2, X, y, kfold, groups)

cross_validation(model4, X, y, kfold, groups)


