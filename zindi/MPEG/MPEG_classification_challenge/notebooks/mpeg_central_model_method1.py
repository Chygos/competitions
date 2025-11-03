
# load libraries
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name="central_model_emissions1")
tracker.start()

import numpy as np
import pandas as pd
import torch, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import os, re, gc
from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal, Optional, Union
import copy, logging, warnings
import joblib
from functools import partial
from modelling_utils import KmerAutoEncoder, KmerClassifier, AutoEncoderScaler, normalise_counts, load_encoder_model
from fl_utils import create_dataloader, get_embeddings, train_loop, test_loop, print_info
import visual_utils
import time


start = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

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

# load (files in kmer x sampleIDs)
train = pd.read_parquet('../data/train_8kmer.parquet')
test = pd.read_parquet('../data/test_8kmer.parquet')
train_labels = pd.read_csv('../data/Train.csv')

train_labels = train_labels.assign(ID = train_labels.filename.str.replace('.mgb', '').str.strip())

# rename and select ID and target
train_labels = train_labels.rename(columns={'SampleType': 'target'})[['ID', 'target']]

# set train labels to match with train columns arrangement
train_labels = train_labels.set_index('ID').reindex(train.columns)
train_labels.shape


class_map = dict(zip(
    np.sort(train_labels['target'].unique()), 
    range(train_labels['target'].nunique())
))

train_labels['class_int'] = train_labels['target'].map(class_map)

target = train_labels.class_int


# ## Normalising counts
# 
# Normalising kmer counts using centered-log ratio (CLR) to prevent sample bias due to read length and depth
# 
# The `normalise_counts` works with the assumption that samples are as rows and kmers as features. But the data was saved in kmer as rows and samples as features to save memory and fast loading. Hence, we will transpose our counts before normalising.
print('Data Preprocessing\n====================')
print('Normalising kmer counts')
train_norm = normalise_counts(train.T)
test_norm = normalise_counts(test.T)

train_norm.shape, test_norm.shape


# ## Extract Embeddings
# 
# We will extract autoencoder embeddings from saved autoencoder. The trained autoencoder was developed such that the input data were standardised using standardscaler. Hence, we will first scale and extract their embeddings. After that, we will load our saved autoencoder model. This saved model depends on the loaded `KmerAutoEncoder` class. We will use the load_encoder_model function to do that by passing the file path. 
# 
# This was done as a dimensionality reduction strategy for fast compute and modelling as against the 66k kmers from ($4^8$ possible 8-kmer sequences). The about 66k kmer features were reduced to 64 embeddings.

encoder_scaler = AutoEncoderScaler(scale=True, scale_type='ss')
encoder_scaler.fit(train_norm)

# joblib.dump(encoder_scaler, 'data/models/encoder_scaler.pkl') # saving for use

# scale and transform
print('Scaling and transforming train and test data')
scaled_train_norm = encoder_scaler.transform(train_norm)
scaled_test_norm = encoder_scaler.transform(test_norm)

# load autoencoder model
encoder_path = glob('../data/models/*latent64*.pth')[0]
encoder_path

encoder_model = load_encoder_model(encoder_path)

# extract embeddings
print('Extracting embeddings for train and test data')
train_embeddings = get_embeddings(encoder_model, scaled_train_norm)
test_embeddings = get_embeddings(encoder_model, scaled_test_norm)


train_embeddings.shape, test_embeddings.shape


# ## __Modelling__
class KmerPipeline:
    def __init__(self, model_fn, criterion, optimiser_fn, preprocessor):
        super().__init__()
        self.model_fn = model_fn # callable
        self.optimiser_fn = optimiser_fn # callable
        self.model = None
        self.preprocessor = preprocessor
        self.criterion = criterion
        self.optimiser = None
        self.best_model_state = None
    
    def train_model(self, X, y, Xval=None, yval=None, epochs=50, batch_size=32, 
                    shuffle=True, early_stopping_rounds=None, print_rounds=5, 
                    seed=None, verbose=False):
        set_seed(seed)
        y = np.array(y)
        # preprocessor
        self.preprocessor.fit(X, y)
        X = self.preprocessor.transform(X) 

        # build model
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = self.model_fn(input_dim, num_classes).to(device)
        self.optimiser = self.optimiser_fn(self.model)
        
        # convert to tensor and dataloader
        train_loader = create_dataloader(X, y, batch_size=128, shuffle=shuffle)
        
        if Xval is not None and yval is not None:
            yval = np.array(yval)
            Xval = self.preprocessor.transform(Xval)
            val_loader = create_dataloader(Xval, yval, batch_size=64, shuffle=True)
        else:
            val_loader = None
        # fit model
        best_loss = float('inf'); wait = 0; best_epoch = None; best_model_train_loss = float('inf')
        for epoch in range(1, epochs+1):
            train_loss = train_loop(self.model, self.criterion, self.optimiser, train_loader)
            if val_loader is not None:
                val_loss = test_loop(self.model, self.criterion, val_loader)['loss']
                if verbose:
                    if print_info(epoch, epochs, print_rounds):
                        print(f"Epoch {epoch}: Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_model_train_loss = train_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
            else:
                if verbose:
                    if print_info(epoch, epochs, print_rounds):
                        print(f"Epoch {epoch}: Train Loss: {train_loss:.7f}")
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
            # early stopping
            if early_stopping_rounds and wait >= early_stopping_rounds:
                # print(f"\nEarly stopping triggered at epoch {epoch}. No improvement after {early_stopping_rounds} epochs.")
                break
        # print best model
        if val_loader is not None:
            print(f'Best Model: Epoch: {best_epoch}, Train Loss: {best_model_train_loss:.7f}, Val Loss: {best_loss:.7f}')
        else:
            print(f'Best Model: Epoch: {best_epoch}, Train Loss: {best_loss:.7f}')
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
    
    def predict_proba(self, X):
        self.model.eval()
        X = self.preprocessor.transform(X)
        test_loader = create_dataloader(X, batch_size=64, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = self.model(inputs.to(device))
                probs = F.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy())
        return np.vstack(probabilities)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def save_file(probs, ids, filename):
    cols = list(class_map.keys())
    path = './'
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame()
    df['ID'] = ids
    df[cols] = probs
    print(df)
    filepath = os.path.join(path, f'{filename}.csv')
    df.to_csv(filepath, index=False)


preprocessor = StandardScaler()

# optimiser
rms_optimiser = lambda model, lr=0.0005: torch.optim.RMSprop(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()



# instantiate model
model = partial(KmerClassifier, layer_mult=3)

X = train_embeddings
y = train_labels.class_int.values

# define model pipeline and train model
print('Modelling\n===============')
print('Training started..')
model_pipe = KmerPipeline(model, criterion, rms_optimiser, preprocessor)
model_pipe.train_model(X, y, epochs=2000, early_stopping_rounds=100, 
                       print_rounds=150, batch_size=256, seed=42, verbose=True)

print('Training Completed!\n')
# get test IDs
test_idx = test.columns
len(test_idx)

# model_predictions
print('Predicting test probabilities..')
test_probs = model_pipe.predict_proba(test_embeddings)

# 
os.makedirs('preds', exist_ok=True)
#
print('Saving test predictions\n')
save_file(test_probs, test_idx, 'preds/centralised_LB_score1')

tracker.stop()


# ### **Cross-Validation**

print('\nPerforming cross-validation\n==============================')
folds = np.zeros(len(train_norm))
skfold = StratifiedKFold(shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(skfold.split(train_norm, train_labels.class_int)):
    folds[val_idx] = i

def cross_validation(clf, optimiser, criterion, train_emb, y=target, cv_folds:int=5, verbose=False):
    lloss = []
    
    skfold = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
    for i in range(len(np.unique(folds))):
        val_idx = folds == i
        xtrain, ytrain = train_emb[~val_idx], y.loc[~val_idx]
        xval, yval = train_emb[val_idx], y.loc[val_idx]
        
        model_pipe = KmerPipeline(model, criterion, rms_optimiser, preprocessor)
        model_pipe.train_model(xtrain, ytrain, xval, yval, epochs=2000, early_stopping_rounds=100, 
                               print_rounds=500, batch_size=256, seed=42, verbose=False)
        
        # evaluate
        res = visual_utils.classification_eval_metrics(model_pipe, xval, yval)
        if verbose:
            print(f'\nFold {i+1}\tLogLoss: {np.array(res.LLoss).squeeze()}')
            print('=='*30)
        lloss.append(np.array(res.LLoss).squeeze())
    avg_lloss = np.mean(lloss)
    ci95_l, ci95_h = np.quantile(lloss, [0.025, 0.975])
    print(f'\nAvg LLoss: {avg_lloss:.8f}')
    print(f'95th CI: [{ci95_l:.8f}, {ci95_h:.8f}]\n')
    

cross_validation(KmerClassifier, rms_optimiser, criterion, train_embeddings, target, verbose=True)

# 
end = time.time()
mins = (end - start)/60
print(f'Total Time taken : {mins:.4f} Mins')
