
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os, random



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

set_seed(42)


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
cytokines = pd.read_csv('../data/cytokine_profiles.csv')
train_info = pd.read_csv('../data/Train_Subjects.csv')
train_info2 = pd.read_csv('../data/Train.csv')


coef_scores = np.load('../data/fs_by_multitask.npy')


selected_kmers = pd.read_csv('../data/selected_kmers_unscaled_var.csv', index_col=0).index


# ## __Data Cleaning__


# - **Merging cytokines data with metadata**


merged_data = (
    cytokines.loc[:, 'SampleID':'CollectionDate'].assign(CollectionDate = pd.to_datetime(cytokines.CollectionDate))
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
target = cytokines.loc[:, 'IL17F':'CHEX4'].columns
len(samples), len(target)


# sorting SampleID in merged data to match those in train 
train = train.reindex(merged_data.ID, axis=1)
train.shape


# - **Selecting kmers with variance above a cutoff**
# 
# - In a previous notebook we had performed variance thresholding to select kmers with their CLR-normalised variance above a cutoff. The cutoff was selected based on the elbow method. Kmers were saved.
# 
# - Because train data wass saved as kmers as rows and sample IDs as features, we will need to transpose it so that kmers become columns while samples, rows. Thereafter, we will select our kmers


# select kmers and transpose
train = train.loc[selected_kmers].T


# - **Normalising kmer counts**
# 
# Here, we will normalise kmer counts to prevent both read sequence depth and size. We will do this using the centered log-ratio (CLR) method.


# resorting samples to match the rows in train
samples = samples.reindex(train.index)[0]


# normalise train
norm_counts = normalise_counts(train)


norm_counts = pd.DataFrame(norm_counts, columns=train.columns, index=train.index)


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


# adding interaction terms to our covariates table
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


# get features used to fit a Multitask elastic net model for featre selection
kmers_covariates = pd.concat([norm_counts.reset_index(), covariates], axis=1).set_index('ID').columns


all_cols = kmers_covariates
selected_features = all_cols[(coef_scores != 0).any(axis=0)]
len(selected_features)


# updating selected kmers
selected_kmers = selected_kmers.intersection(selected_features)


# ### Preprocessing dataset to match the state the model was trained


categorical_features = selected_features.difference(selected_kmers)
scaled_features = selected_features.intersection(selected_kmers)


len(scaled_features), len(categorical_features)


all_df = pd.concat([norm_counts.reset_index(drop=True), covariates], axis=1)
all_df.index= train.index

all_df.shape


all_df[selected_features].shape


# - Scaling only numerical features and leaving categorical features as is


# preprocessor for input data
preprocessor = ColumnTransformer([('scale', MinMaxScaler(), scaled_features)], remainder='passthrough')


X = preprocessor.fit_transform(all_df[selected_features]).astype(np.float32)


# scaling dependent variables to be on the same range
yscaler = StandardScaler()

yscaler.fit(merged_data[target].values)
yscaled = yscaler.transform(merged_data[target].values).astype(np.float32)

X.shape, yscaled.shape

xtrain, xval, ytrain, yval = train_test_split(X, yscaled, random_state=123, test_size=0.3)
xtrain.shape, xval.shape, ytrain.shape, yval.shape


input_dim = xtrain.shape[1] #- covariates.shape[1]
output_dim = len(target)
input_dim, output_dim


model1 = LinearRegression(input_dim, output_dim)
model2 = CytokineModel(input_dim, output_dim)
model3 = CytokineModel1(input_dim, output_dim)

# load model weights
model1.load_state_dict(torch.load('models/linear_regression.pth', map_location='cpu', weights_only=False))
model3.load_state_dict(torch.load('models/model2_one_layer.pth', map_location='cpu', weights_only=False))
model2.load_state_dict(torch.load('models/model2_two_layer.pth', map_location='cpu', weights_only=False))

# shap values
explainer = shap.DeepExplainer(model1, torch.tensor(xval))


shap_values = explainer.shap_values(torch.tensor(xval))


shap_values

# visualise
plt.title('Top 20 Features (Shap values)', fontsize=12, fontweight='bold', loc='left')
plt.yticks(fontsize=6)
# take absolute average across cytokine levels
shap.summary_plot(np.abs(shap_values).mean(2), selected_features, plot_type='bar', max_display=20, plot_size=(12,9))


# model based importance
coefficients = model1.state_dict()['fc1.weight'].cpu().numpy()


avg_vals = np.abs(coefficients).mean(0)

# sorting by absolute average values
sorted_vals = np.argsort(-abs(avg_vals)) 

# What direction do they drive cytokine levels?
# negative directions
neg_direct_counts = np.sum(coefficients < 0, axis=0)

pd.DataFrame(np.c_[avg_vals, neg_direct_counts/66], index=selected_features, columns=['magnitude', 'direction']).iloc[sorted_vals[:20]]

pd.Series(avg_vals, index=selected_features, name='score').iloc[sorted_vals[:20]][::-1].plot.barh(
    figsize=(12,9), width=0.7, color='indianred')
plt.title('Top 20 features (Absolute coefficient values)', fontsize=12, fontweight='bold', loc='left')
