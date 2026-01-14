from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os, gc, warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.linear_model import Ridge
import scipy.stats as st
from tqdm import tqdm
from joblib import delayed, Parallel

warnings.filterwarnings('ignore')

def fit_regression_model(model, 
                         df:pd.DataFrame, 
                         cytokine:str|list, 
                         kmer:str|list, 
                         covariates:list=None,
                         scale_y:bool=True, 
                         scale_x:bool=False) -> pd.DataFrame:
    """
    Fits a Linear Regression Model

    :param model: Regression model
    :param df: Pandas DataFrame
    :param cytokine: str. Cytokine outcome. Target variable
    :param kmer: str|list of kmers. Independent variable
    :param covariates: Covariates to include in the model
    :param scale_y: Boolean. Scale target variable
    """
    X = df.copy()
    y = df[cytokine]
    
    if covariates is not None:
        cat_cols = [i for i in covariates if X[i].dtype == object]
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        ohe.fit(X[cat_cols])
        output_features = ohe.get_feature_names_out().tolist()
        X_catcols = pd.DataFrame(ohe.transform(X[cat_cols]).astype(np.int32), 
                                 columns=output_features, 
                                 index=X.index)
        # merge 
        X = pd.concat([X, X_catcols], axis=1).drop(columns=cat_cols)
        if scale_x: # scale numerical features not cat cols
            num_features = [i for i in covariates if i not in cat_cols]
            X[num_features]  = X[num_features].apply(st.zscore)
            
        features = kmer+output_features+num_features if isinstance(kmer, list) else kmer.tolist()+output_features+num_features
        X = X[features]

    kmer_idx = X.columns.get_indexer(kmer) # get indexes for kmers
    if scale_x:
        X[kmer] = X[kmer].apply(st.zscore)
    
    if scale_y:
        y = st.zscore(y)
    
    model.fit(X, y)
    coefs = model.coef_
    
    return coefs[:, kmer_idx] if np.ndim(coefs) == 2 else coefs[kmer_idx]


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


def compute_topN_stability(coef_array, N=20):
    """
    Compute Top-N stability for each kmer-cytokine pair.
    
    :paramcoef_array: np.ndarray of shape [n_bootstraps, n_kmers, n_cytokines]
    :param N: int, number of top features to consider
    :returns topN_stability: np.ndarray of shape [n_kmers, n_cytokines]
    """
    if isinstance(coef_array, list):
        coef_array = np.stack(coef_array)
    n_bootstraps, n_kmers, n_cytokines = coef_array.shape
    topN_stability = np.zeros((n_kmers, n_cytokines))

    for j in range(n_cytokines):
        # Get absolute coefficients for cytokine j across bootstraps
        abs_coefs = np.abs(coef_array[:, :, j])  # shape: [n_bootstraps, n_kmers]

        # Rank features for each bootstrap
        ranks = np.argsort(-abs_coefs, axis=1)  # descending order

        # Create a mask where each cell is True if the kmer is in top N
        top_mask = np.zeros_like(abs_coefs, dtype=bool)
        for b in range(n_bootstraps):
            top_mask[b, ranks[b, :N]] = True

        # Average across bootstraps to get stability score
        topN_stability[:, j] = np.mean(top_mask, axis=0)

    return topN_stability


def bootstrap_fit(model, seed):
    """
    Create Bootstrap fits and returns list of results
    
    :param model: Regression Model
    :param seed: Pseudo-random seed for reproducibility
    :returns res: np.ndarray of shape [n_kmers, n_cytokines]
    """
    np.random.seed(seed)
    df = resample(kmer_data)
    res = fit_regression_model(model, df, target_vars, sig_kmers, 
                               ['age', 'BMI', 'FPG_class', 'Gender', 'site'], 
                               scale_x=True, scale_y=True)
    return res.T


def compute_sign_stability(coef_array):
    """
    Computes the number of times the coefficient was positive and negative

    :param coef_array: list[np.array]|np.array
    :returns npos, nneg, maximum(npos, nneg)
    """
    if isinstance(coef_array, list):
        coef_array = np.stack(coef_array)
    
    neg = np.mean(coef_array < 0, axis=0)
    pos = np.mean(coef_array > 0, axis=0)
    sign_stability = np.maximum(neg, pos)
    return neg, pos, sign_stability


def compute_coefficient_stability(coef_array):
    """
    Computes stability of coefficients
    cv = std(coef_array)/ |mean(coef_array)|

    :param coef_array: List[np.ndarray], np.ndarray
    :returns normalised cv score. Higher the better stable
    """
    if isinstance(coef_array, list):
        coef_array = np.stack(coef_array)
    cv = np.std(coef_array, axis=0) / np.abs(np.mean(coef_array, axis=0))
    return 1/(1+cv)


def compute_confidence_scores(coef_array, alpha:float|int=0.95):
    """
    Computes upper and lower limits of the confidence interval
    :param coef_array: List[np.ndarray] | np.ndarray
    :param alpha: float | int, confidence level
    :returns lower, upper: np.ndarray of shape [n_kmers, n_cytokines]
    """
    if isinstance(alpha, int):
        alpha /= 100
    q = np.round([(1-alpha)/2, alpha + (1-alpha)/2], 4)

    if isinstance(coef_array, list):
        coef_array = np.stack(coef_array)
    lower, upper = np.quantile(coef_array, q=q, axis=0)
    return lower, upper


def compute_mean_values(coef_array):
    """
    Computes mean value of bootstrapped coefficients
    :param coef_array: List[np.ndarray] | np.ndarray
    :returns mean_vals: np.ndarray of shape [n_kmers, n_cytokines]
    """
    return np.mean(coef_array, axis=0)



def main(coef_array):
    """
    Computes all stability scores

    :param coef_array: List[np.ndarray] | np.ndarray
    :returns all_vals: np.ndarray of shape [n_kmers, n_cytokines, 10]
    
    Columns:
    [0]: mean values
    [1]: lower confidence interval
    [2]: upper confidence interval
    [3]: topn stability
    [4]: cv stability
    [5]: number of negative coefficients
    [6]: number of positive coefficients
    [7]: sign stability
    [8]: mean scale
    [9]: composite score
    """
    mean_vals = compute_mean_values(coef_array)
    lower, upper = compute_confidence_scores(coef_array, alpha=0.95)
    topn_stability = compute_topN_stability(coef_array, N=100)
    cv_stability = compute_coefficient_stability(coef_array)
    nneg, npos, sign_stability = compute_sign_stability(coef_array)
    mean_scale = MinMaxScaler().fit_transform(np.abs(mean_vals))
    composite_score = 0.6*cv_stability + 0.4*sign_stability

    all_vals = np.column_stack([mean_vals.ravel(), lower.ravel(), upper.ravel(), 
                                topn_stability.ravel(), cv_stability.ravel(), 
                                nneg.ravel(), npos.ravel(), sign_stability.ravel(), 
                                mean_scale.ravel(), composite_score.ravel()])
    return all_vals


train = pd.read_parquet('../data/train_8kmer_all.parquet').T # transpose to sample ID x kmers
cytokine_df = pd.read_csv('../data/cytokine_profiles.csv')
train_info = pd.read_csv('../data/Train_Subjects.csv')
train_info2 = pd.read_csv('../data/Train.csv')

train.shape, cytokine_df.shape, train_info.shape, train_info2.shape

# merge subjectID to cytokine_df for easy merging
cytokine_df = train_info2[['SubjectID', 'SampleID']].drop_duplicates().merge(cytokine_df, on=['SampleID'])
cytokine_df = cytokine_df.merge(train_info[['SubjectID', 'FPG_class', 'Class', 'Gender', 'Adj.age', 'BMI']], on='SubjectID')
cytokine_df['CollectionDate'] = pd.to_datetime(cytokine_df.CollectionDate, errors='coerce')

# load significant kmers from track 5
sig_kmers = pd.read_csv('../data/sig_kmers.csv', index_col=0).index.tolist()


train.head(2)

# filtering train data using significant kmers
train = train[sig_kmers]


# normalise using significant kmers
norm_counts = normalise_counts(train)

# convert to dataframe
norm_counts = pd.DataFrame(norm_counts, columns=sig_kmers, index=train.index)

target_vars = cytokine_df.loc[:, 'IL17F':'CHEX4'].columns


# prepare kmer data by merging metadata
kmer_data = (
    train_info2.assign(filename = train_info2.filename.str.replace('.mgb', '').str.strip())
    .rename(columns={'filename': 'ID', 'SampleType':'site'})
    .merge(norm_counts.reset_index(names='ID'), on='ID')
    .merge(cytokine_df
           .rename(columns={'Adj.age': 'age'})[['SampleID', 'Gender', 'BMI', 'age', 'FPG_class']+target_vars.tolist()], 
           on='SampleID')
    .set_index('ID')
)

# Generate Boostraps
model = Ridge(alpha=0.1) # add regularisation to prevent gradient explosion due to multicolinearity

# fit baseline model
baseline_res = fit_regression_model(model, kmer_data, target_vars, sig_kmers, 
                                    ['age', 'BMI', 'FPG_class', 'Gender', 'site'], 
                                    scale_x=True, scale_y=True).T.ravel()



np.random.seed(10)
nrounds = 1000
seeds = np.random.randint(0, 1e4, size=nrounds)
coef_res = Parallel(n_jobs=3)(delayed(bootstrap_fit)(model, seed) for seed in tqdm(seeds, desc='Bootstrapping Method'))

# fit bootstraps
res = main(coef_res)


# create a combination of kmer and cytokine
metadata = (
    pd.DataFrame(target_vars, columns=['cyt'])
    .assign(key=1)
    .merge(pd.DataFrame(sig_kmers, columns=['kmer'])
           .assign(key=1), 
           on='key', how='right').drop(columns='key')
    )

metadata['baseline_coef'] = baseline_res # add baseline coefficient

# Add to metadata
metadata[['coef_mean', 'coef_lower', 'coef_upper', 'topn_stability', 
         'cv_stability', 'perc_neg', 'perc_pos', 'sign_stability', 
         'mean_scale', 'composite_score']] = res

metadata = metadata.sort_values('composite_score', ascending=False).reset_index(drop=True)

metadata.to_csv('../data/track2_resample_results.csv', index=False)

print(metadata.head(10))