from sklearn.linear_model import MultiTaskElasticNetCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random, os, gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging, warnings
from sklearn.model_selection import train_test_split



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


def variance_threshold(X):
    X_scaled = MinMaxScaler().fit_transform(X)
    x_var = np.var(X_scaled, axis=0)
    return x_var

def find_elbow_threshold(X):
    """
    Find elbow point in variance CDF using a simple curvature method.
    variances: 1D array of feature variances (unscaled data)
    Returns: variance threshold at elbow
    """
    def compute_elbow(vals):
        sorted_val = np.sort(vals)
        cdf = np.arange(1, len(sorted_val) + 1) / len(sorted_val)
        # Normalise both axes to [0,1]
        x_norm = (sorted_val - sorted_val.min()) / (sorted_val.max() - sorted_val.min())
        y_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min())

        distances = y_norm - x_norm
        idx_elbow = np.argmax(distances)
        elbow_var = sorted_val[idx_elbow]
        return sorted_val, elbow_var, cdf
    
    scaled_var = np.var(MinMaxScaler().fit_transform(X), axis=0)
    unscaled_var = np.var(X, axis=0)

    u_sorted_val, u_elbow_var, u_cdf = compute_elbow(unscaled_var)
    s_sorted_val, s_elbow_var, s_cdf = compute_elbow(scaled_var)
    
    # Plot for visual check
    fig, ax = plt.subplots(1,2, figsize=(10,4.5))
    ax[0].plot(u_sorted_val, u_cdf, label='CDF (unscaled var)')
    ax[0].axvline(u_elbow_var, color='r', ls='--', label=f'Elbow ~ {u_elbow_var:.4f}')
    ax[0].set(xlabel='Variance', ylabel='CDF')

    ax[1].plot(s_sorted_val, u_cdf, label='CDF (scaled var)')
    ax[1].axvline(s_elbow_var, color='r', ls='--', label=f'Elbow ~ {s_elbow_var:.4f}')
    ax[1].set(xlabel='Variance', ylabel='CDF')
    fig.suptitle('Cumulative Distribution function for scaled and unscaled variance', 
                 fontweight='bold', fontsize=12)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    plt.show()

    return u_elbow_var, s_elbow_var

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
    

# load datasets

train = pd.read_parquet('../data/train_8kmer_all.parquet')
test = pd.read_parquet('../data/test_8kmer_all.parquet')
cytokine_df = pd.read_csv('../data/cytokine_profiles.csv')
train_info = pd.read_csv('../data/Train_Subjects.csv')
train_info2 = pd.read_csv('../data/Train.csv')


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

# Selecting samples with cytokine data from train data**
# Obtaining the cytokines for use later**


# intersection IDs (selecting samples in train and merged data ie cytokines)
samples = train.columns.intersection(merged_data.ID)
target = cytokine_df.loc[:, 'IL17F':'CHEX4'].columns
len(samples), len(target)


# Align SampleID in merged data to match those in train 
train = train.reindex(merged_data.ID, axis=1) # axis=1 because they are as columns
train.shape

# realigning samples to match the rows in train
samples = samples.reindex(train.index)[0]

# create interaction terms
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


# Feature selection (Variance thresholding)
# - Removing kmers with very low variance across samples
norm_counts = normalise_counts(train.T)

# variance of normalised counts
unscaled_var = np.var(norm_counts, axis=0)
scaled_var = variance_threshold(norm_counts)

# return scaled and unscaled variance
un_elbow_var, sc_elbow_var = find_elbow_threshold(norm_counts)

kmers = train.index

selected_kmers = pd.Series(kmers[unscaled_var >= un_elbow_var])

selected_kmers.to_csv('selected_kmers_unscaled_var.csv', index=False)


# Feature selection using MultitaskElasticNetCV
norm_counts = pd.DataFrame(norm_counts, columns=kmers, index=train.columns)

all_df = pd.concat([norm_counts.reset_index(drop=True), covariates], axis=1)
all_df.index= train.index

print(all_df.shape)

yscaler = StandardScaler() # scale y to be on similar levels
yscaler.fit(merged_data[target].values)
yscaled = yscaler.transform(merged_data[target].values).astype(np.float32)
xtrain, xval, ytrain, yval = train_test_split(all_df, yscaled, random_state=123, test_size=0.2)
xtrain.shape, xval.shape, ytrain.shape, yval.shape


enet = MultiTaskElasticNetCV(
    l1_ratio=0.5,   # balance L1 vs L2
    alphas=[0.1, 0.5, 1.0, 5, 10.0],  # search grid
    cv=5,
    random_state = 42,
    max_iter=5000,
    n_jobs=3
)


model = Pipeline([
    ('transformer', ColumnTransformer(
        [('scale', StandardScaler(), selected_kmers)], remainder='passthrough')),
    ('enet', enet)])

model.fit(xtrain, ytrain)
np.save('fs_by_multitask.npy', enet.coef_)

(enet.coef_ != 0).any(axis=0).shape, enet.alpha_