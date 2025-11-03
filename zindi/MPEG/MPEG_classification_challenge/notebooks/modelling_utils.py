
import numpy as np
import pandas as pd
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, re, gc
from glob import glob
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
import joblib, copy, logging
from typing import Literal
from sklearn.cross_decomposition import PLSRegression

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


## Helper Functions
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


class KmerAutoEncoder(nn.Module):
    """Feature Extraction Encoder-Decoder Class"""
    def __init__(self, input_dim, latent_dim, dropout_rate=0.15, neuron_size=1024):
        super().__init__()
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
    

class PLSLatentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self._pls = PLSRegression(n_components=self.n_components, scale=False)

    def fit(self, X, y):
        y_cat = LabelBinarizer().fit_transform(y)
        self._pls.fit(X, y_cat)
        return self

    def transform(self, X):
        return self._pls.transform(X)  # Return latent features
    
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
        

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=True, scale_type:Literal['ss', 'mm']='ss'):
        self.scale = scale
        self.scale_type = scale_type
        self.scaler_ = None
    
    def fit(self, X):
        if self.scale:
            self.scaler_ = StandardScaler() if self.scale_type == 'ss' else MinMaxScaler()
            self.scaler_.fit(X)
        return self
    def transform(self, X):
        if self.scaler_:
            X = self.scaler_.transform(X)
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class AutoEncoderScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=True, scale_type:Literal['ss', 'mm']='ss'):
        self.scale = scale
        self.scale_type = scale_type
        self.scaler_ = None
    
    def fit(self, X):
        if self.scale:
            self.scaler_ = StandardScaler() if self.scale_type == 'ss' else MinMaxScaler()
            self.scaler_.fit(X)
        return self
    def transform(self, X):
        if self.scaler_:
            X = self.scaler_.transform(X)
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

# Classifer
class KmerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, layer_mult=3):
        super().__init__()
        self.input_dim = input_dim
        self.layer_mult = layer_mult
        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, 64*self.layer_mult)
        self.bn1 = nn.BatchNorm1d(64*self.layer_mult)
        self.fc2 = nn.Linear(64*self.layer_mult, 16*self.layer_mult)
        self.bn2 = nn.BatchNorm1d(16*self.layer_mult)
        self.fc3 = nn.Linear(16*self.layer_mult, 8*self.layer_mult)
        self.bn3 = nn.BatchNorm1d(8*self.layer_mult)
        self.output = nn.Linear(8*self.layer_mult, num_classes)
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.output(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc1(x)
    

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


def load_encoder_model(encoder_path = "clr_std_latent64_epoch112_model_k8.pth"):
    """Loads weight of autoencoder model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_wts = torch.load(encoder_path, map_location='cpu')
    encoder_model = KmerAutoEncoder(input_dim=4**8, latent_dim=64).to(device)
    encoder_model.load_state_dict(encoder_wts)
    return encoder_model

def print_info(epoch, epochs, n_epoch_print):
    """
    Print epoch info during training
    """
    if (epoch % n_epoch_print) == 0 or epoch == 1 or epoch == epochs:
        return True
    else:
        return False

# Train and test loop
def train_loop(model, criterion, optimiser, train_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device) # move to device
    n_batch = len(train_loader) # number of batches in train loader
    total_loss = 0.
    model.train()
    for inputs, labels in train_loader:
        optimiser.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / n_batch

def test_loop(model, criterion, val_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    test_loss = 0.0
    tot_correct = 0
    
    n = len(val_loader.dataset)
    nbatch = len(val_loader)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()  # sum loss over samples
            preds = outputs.argmax(dim=1)
            tot_correct += (preds == labels).sum().item()

    avg_loss = test_loss / nbatch
    acc = tot_correct / n
    return {'loss': avg_loss, 'acc': acc}