
# ## Loading Libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
import os, warnings
from sklearn.model_selection import train_test_split
import torch.nn as nn
from typing import Literal
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


## Notebook run on Kaggle (GPU: T4 X 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')


# ## Helper functions


# for reproducibility
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


set_seed(42)

class KmerAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout_rate=0.25, neuron_size=512):
        super(KmerAutoEncoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, neuron_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(neuron_size, latent_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, neuron_size),
            nn.ReLU(),
            nn.Linear(neuron_size, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# change to dataset path
data_path = '../data'
train = pd.read_parquet(f'{data_path}/train_8kmer.parquet')
test = pd.read_parquet(f'{data_path}/test_8kmer.parquet')
train_labels = pd.read_csv(f'{data_path}/Train.csv')


train.head()


train = train.T
test = test.T


# get train and test IDs
train_idx = train.index
test_idx = test.index


train_labels = train_labels.assign(ID = train_labels.filename.str.replace('.mgb', '').str.strip())
# rename and select ID and target
train_labels = train_labels.rename(columns={'SampleType': 'target'})[['ID', 'target']]

# reindexing to match columns in train data
train_labels = train_labels.set_index('ID').reindex(train.index)
train_labels.shape


def convert_to_Tensor(X):    
    X_tensor = np.array(X)
    X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
    return TensorDataset(X_tensor)

def to_Dataloader(df, batch_size=32, shuffle=False):
    return DataLoader(df, batch_size=batch_size, shuffle=shuffle)

def preprocess_X(X, batch_size=128, shuffle=False):
    if not isinstance(X, torch.utils.data.dataloader.DataLoader):
        if not isinstance(X, torch.Tensor):
            X_tensor = convert_to_Tensor(X)
            X_loader = to_Dataloader(X_tensor, batch_size=batch_size, shuffle=shuffle)
        
    else:
        X_loader = X
    return X_loader

def extract_embeddings(model, new_data, **kwargs):
    model.eval()
    embeddings = []
    if not isinstance(new_data, torch.utils.data.dataloader.DataLoader):
        if not isinstance(new_data, torch.Tensor):
            new_data_tensor = convert_to_Tensor(new_data, **kwargs)
            new_data_loader = to_Dataloader(new_data_tensor, batch_size=128, shuffle=False)
        
    else:
        new_data_loader = new_data
    
    with torch.no_grad():
        for batch_x, in new_data_loader:
            batch_x = batch_x.to(device)
            _, encoded = model(batch_x)
            embeddings.append(encoded)
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()
    return embeddings


# ## Feature Extraction

# __Training an Autoencoder__

def val_performance(model, loss_func, val_data):
    model.eval()
    # loss for clean and noisy data if noise was added validation data
    val_loss = 0
    with torch.no_grad():
        for batch_x, in val_data:
            batch_x = batch_x.to(device)
            batch_decoded, batch_encoded = model(batch_x)
            loss = loss_func(batch_decoded, batch_x).item()
            val_loss += loss
    
    n = len(val_data)
    return val_loss/n

def print_performance(epoch, num_epochs, epoch_print=10):
    if epoch % epoch_print == 0 or epoch == 1 or epoch == num_epochs:
        return True

def training_loop(model, optimizer, criterion, train_dataloader, val_dataloader=None, epochs=50, 
                  patience=10, epoch_print=5, return_best_epoch=False):
    
    best_epoch = None
    wait = 0  # patience counter
    best_model_state = None

    if val_dataloader is not None and isinstance(val_dataloader, torch.utils.data.DataLoader):
        best_val_loss = float('inf')
        train_loss_at_best_val = None
    else:
        best_train_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        total_loss = 0
        for batch_x, in train_dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            batch_decoded, batch_encoded = model(batch_x)
            loss = criterion(batch_decoded, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        # ---- Validation ----
        if val_dataloader is not None and isinstance(val_dataloader, torch.utils.data.DataLoader):
            # returns validation data loss
            val_loss = val_performance(model, criterion, val_dataloader)

            if print_performance(epoch, epochs, epoch_print):
                print(f"Epoch {epoch}: Train Loss: {avg_loss:.7f}, Val Loss: {val_loss:.7f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                train_loss_at_best_val = avg_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered at epoch {epoch}. "
                          f"No improvement in val loss for {patience} epochs.")
                    break
        else:
            # ---- No validation: track best training loss ----
            if print_performance(epoch, epochs, epoch_print):
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.7f}")

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break

    # ---- Final report ----
    if val_dataloader is not None and isinstance(val_dataloader, torch.utils.data.DataLoader):
        print(f"\nBest Epoch = {best_epoch}, Best Val Loss = {best_val_loss:.7f}, Train Loss at Best Val = {train_loss_at_best_val:.7f}")
    else:
        print(f"\nBest Epoch = {best_epoch}, Best Train Loss = {best_train_loss:.7f}")

    # set model weights with best model's state
    model.load_state_dict(best_model_state)
    
    if return_best_epoch:
        return best_epoch


class_map = dict(zip(np.sort(train_labels['target'].unique()), range(train_labels['target'].nunique())))
train_labels['class_int'] = train_labels['target'].map(class_map)


def split_data(X, y=train_labels['class_int'], test_size=0.2):
    Xtrain, Xval, y_train, y_val = train_test_split(X, y, stratify=y, random_state=123, test_size=test_size)
    return Xtrain, Xval, y_train, y_val


Xtrain, Xval, y_train, y_val = split_data(train)

Xtrain.shape, Xval.shape

input_dim = train.shape[1]; input_dim


## Automencoder (64 embeddings)
# for saving models
os.makedirs('data/models', exist_ok=True)

## Centered Log ratio

batch_size = 128
neuron_size = 1024
dropout_rate = 0.15


preprocessor = StandardScaler()
preprocessor.fit(Xtrain)

Xtrain_clr = preprocessor.transform(Xtrain)
Xval_clr = preprocessor.transform(Xval)
Xtest_clr = preprocessor.transform(test)

# convert to data loaders
xtrain_dataloader = preprocess_X(Xtrain_clr, shuffle=True, batch_size=batch_size)
xval_dataloader = preprocess_X(Xval_clr, shuffle=False, batch_size=batch_size)
xtest_dataloader = preprocess_X(Xtest_clr, shuffle=False, batch_size=batch_size)

# fit on whole data and transform test using train info (for use in final model)
preprocessor.fit(train)
train_clr = preprocessor.transform(train)
test_clr = preprocessor.transform(test)


model = KmerAutoEncoder(input_dim, 64, dropout_rate, neuron_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


best_epoch = training_loop(model, optimizer, criterion, xtrain_dataloader, xval_dataloader, 
                            patience=20, epochs=500, epoch_print=100, 
                            return_best_epoch=True)
print('='*20)

# test on test data
test_loss = val_performance(model, criterion, xtest_dataloader)
print(f'Test Loss: {test_loss:.7f}')
print()

# train on all train data at best epoch
print('Fitting whole data on all train and best epoch\n==============================')
final_model = KmerAutoEncoder(input_dim, 64, dropout_rate, neuron_size).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

training_loop(final_model, optimizer, criterion, 
              preprocess_X(train_clr, shuffle=True, batch_size=batch_size), 
              epochs=best_epoch+50, epoch_print=100)

# test on test data
test_loss = val_performance(final_model, criterion, 
                            preprocess_X(test_clr, shuffle=True, batch_size=batch_size))
print(f'Test Loss: {test_loss:.7f}')
print('=+='*25)
print()

joblib.dump(preprocessor, 'data/models/encoder_scaler.pkl')
torch.save(final_model.state_dict(), f'data/models/clr_std_latent64_epoch{best_epoch}_model_k8.pth')