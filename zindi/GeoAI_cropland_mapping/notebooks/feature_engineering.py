# load libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import os, warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re, time
from typing import Literal
from scipy.spatial import cKDTree
from functools import partial
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

os.makedirs('imgs', exist_ok=True)

## Helper functions
start  = time.time()
## Data clustering
def get_optimal_clusters(df, n_clusters=13, scale=False, 
                         scorer:Literal['silhouette', 'elbow']='elbow'):
    """
    Gets the optimal number of clusters using the Silhouette or Elbow methods

    :param df: Input Data
    :param n_clusters: Number of clusters to check
    :param scale: Boolean. To standardize input data
    :param scorer: Method to use for selecting optimal clusters (silhouette or elbow)
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    X = df.copy()
    scores = []
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    for i in tqdm(range(2, n_clusters+1)):
        res = KMeans(i, max_iter=500, n_init=10)
        res.fit(X)
        if scorer == 'silhouette':
            scores.append(silhouette_score(X, res.labels_))
        elif scorer == 'elbow':
            scores.append(res.inertia_)

    # visualise
    fig, ax = plt.subplots(1, figsize=(8,4.5))
    ax.plot(list(range(2,n_clusters+1)), scores, 'o-', linewidth=1.4, markersize=3)
    ax.set_xticks(range(2, n_clusters+1, 2), range(2, n_clusters+1, 2))
    ax.set_xlabel('Number of clusters')
    ax.set_title(f'{scorer.title()} Method', loc='left', fontweight='bold', fontsize=10)
    ax.set_ylabel(f'{scorer.title()} scores', fontweight='bold')
    fig.tight_layout()


def cluster_data(df, n_cluster, scale=False):
    """
    Fits a KMeans clustering algorithm based on a defined number of clusers

    :param df: Pandas DataFrame
    :param n_cluster: Number of clusters to group data
    :param scale: Boolean. Standardize the input values

    :returns cluster labels and centers for clustered data
    """
    X = df.copy()

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=34, max_iter=500)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

def harmonic_regression(group, col='ndvi'):
    """Fits a simple harmonic regression model to capture seasonal cycle"""
    months = group["month"].values
    values = group[col].values

    def harmonic(x, a0, a1, b1):
        return a0 + a1 * np.cos(2 * np.pi * x / 12) + b1 * np.sin(2 * np.pi * x / 12)

    try:
        invalid = np.isnan(values)
        popt, _ = curve_fit(harmonic, months[~invalid], values[~invalid], maxfev=10000)
        a0, a1, b1 = popt
        amplitude = np.sqrt(a1**2 + b1**2)
        phase = np.arctan2(b1, a1)
        phase = (phase + 2*np.pi) % (2*np.pi)
        peak_month = (phase / (2*np.pi)) * 12 
    except:
        a0, a1, b1, amplitude, phase, peak_month = [np.nan] * 6
    res = np.array([amplitude, phase, peak_month])
    return res

def bin_coordinates(df, loc_bounds, grid_size=0.03, distance=None):
    df_copy = df.copy()
    if 'location' in df.columns:
        for region in df['location'].unique():
            location_coords = df.query(f'location == "{region}"').geometry.get_coordinates()
            region_bounds = loc_bounds[region]
            if grid_size and distance:
                raise Exception('One of grid size or distance must be set')
            elif grid_size:
                cell_grid_size = grid_size
                xmin, ymin, xmax, ymax = list(region_bounds.values())
            elif distance:
                utm_dist = {'Fergana' : 32642, 'Orenburg':32640}
                location_coords = df.query(f'location == "{region}"').to_crs(utm_dist[region]).geometry.get_coordinates()
                cell_grid_size = distance*1000 # km -> metres
                xmin = location_coords.x.min(); ymin = location_coords.y.min()
                xmax = location_coords.x.max(); ymax = location_coords.y.max()
            
            eps = 1e-12
            xbin_size = np.arange(xmin, xmax+cell_grid_size+eps, cell_grid_size)
            ybin_size = np.arange(ymin, ymax+cell_grid_size+eps, cell_grid_size)
            xbins = pd.cut(location_coords.x, bins=xbin_size, right=False, include_lowest=True)
            ybins = pd.cut(location_coords.y, bins=ybin_size, right=False, include_lowest=True)                
            df_copy.loc[df_copy.location == region, 'grid_id'] = region[0] + \
                pd.Categorical(xbins).codes.astype(str) + \
                    pd.Categorical(ybins).codes.astype(str)
    return df_copy


def linear_regression(group, ind_col=None, col='ndvi', **kwargs):
    """
    Fit a simple linear regression model to capture seasonal cycle:
    y = a0 + a1*t
    Returns coefficients (a0, a1, b1)

    :param group: Grouped data for each ID
    :param ind_col: An independent variable
    :param col: Dependent variable
    :param **kwargs: Keyword arguments. Accepted arguments include add_diff, add_pct, add_rolling, n_roll and roll_func
    
    :returns coefficients for each input variable and intercept
    """
    add_diff = kwargs.get('add_diff', False)
    add_pct = kwargs.get('add_pct', False)
    add_rolling = kwargs.get('add_rolling', False)
    n_roll = kwargs.get('n_roll', 1)
    roll_func = kwargs.get('roll_func', 'mean')

    if add_diff and add_pct:
        raise Exception('One of add_diff of add_pct must be provided')
    if add_diff:
        values = group[col].diff()
    elif add_pct:
        values = group[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    else: 
        values = group[col]

    if add_rolling: # rolling summary
        values = values.rolling(n_roll).agg(roll_func)
    if ind_col is None:
        X = np.column_stack([np.ones(len(values)), np.arange(len(values))])
    else:
        X = np.column_stack([np.ones(len(values)), group[ind_col]])

    add_seasonal = kwargs.get('add_seasonal', False)
    if add_seasonal:
        seasons = group['month'].map(encode_season)
        X_seasonal, _ = one_hot_encode(seasons) 
        X = np.c_[X, X_seasonal][:, 1:] # fit regression without intercept
    missing = np.isnan(values)
    coeffs, _, _, _ = np.linalg.lstsq(X[~missing], values[~missing], rcond=None)
    return coeffs

def get_nearest_neighbors(ref_locs, query_locs, n_neighbors=1, threshold=None):
    """
    Find the nearest neighbors in query_locs for each point in ref_locs.
    :param ref_locs: Reference location
    :param query_locs: Query locations (coordinates)
    :param n_neighbours: Int or List: Number of nearest neighbours to return. 
                         If int, nearest neighbours up to set value is returned. if list, only nearest neighbours in the list are returned
    :param threshold: Distance threshold to return (Distance in metres)
    """
     # convert to UTM for accurate distance
    tree = cKDTree(ref_locs.to_crs(32633).get_coordinates())
    # Get the nearest neighbor's ID and Cropland status
    distances, nearest = tree.query(query_locs.to_crs(32633).get_coordinates(), k=n_neighbors)
    distances = distances.ravel()
    nearest = nearest.ravel()
    
    closest_loc = (
        ref_locs[['ID']].iloc[nearest]
        .rename(columns={'ID': 'nn_PID'})
        .reset_index(drop=True)
        .assign(distance = distances)
    )

    if isinstance(n_neighbors, (list, tuple)):
        locs = pd.DataFrame(np.repeat(query_locs['ID'].values, len(n_neighbors)), columns=['ID']).reset_index(drop=True)
    elif isinstance(n_neighbors, int):
        locs = pd.DataFrame(np.repeat(query_locs['ID'].values, n_neighbors), columns=['ID']).reset_index(drop=True)
    
    closest_loc = pd.concat([locs, closest_loc], axis=1)
    # add query locs
    closest_loc = pd.merge(query_locs[['ID']], closest_loc, on='ID', how='left')
    # return values with distances less than cutoff
    return closest_loc.query(f'distance <= {threshold}') if threshold is not None else closest_loc 


def extract_features(dem, s1, s2, clim):
    """
    Feature engineering
    :param dem: Elevation model dataframe
    :param s1: Sentinel-1 data
    :param s2: Sentinel-2 data
    :param clim: Climate data

    :returns Pandas DataFrame
    """
    # create vegetative indices and polarisation ratios
    locations = dem[['ID', 'location', 'Cropland', 'elevation', 'slope', 'x', 'y']].sort_values('ID').reset_index(drop=True)

    s1_temp = s1.copy().sort_values(['date', 'ID']).assign(month=s1.date.dt.month)
    s2_temp = s2.copy().sort_values(['date', 'ID']).assign(month=s2.date.dt.month)
    clim_temp = clim.copy().sort_values(['date', 'ID']).assign(month=clim.date.dt.month)
    
    # compute sentinel-1 and -2 indices
    s1_temp = compute_s1_indices(s1_temp)
    s2_temp = compute_s2_indices(s2_temp)
    
    # calculate sentinel data (monthly and yearly summary stat)
    s1_year = s1_temp.groupby('ID')[s1_cols+['vh_vv_ratio']].agg(['min', 'mean', 'std', 'max'])
    s1_year.columns = [f'{col}_{i}' for col, i in s1_year.columns]
    
    s2_year = s2_temp.groupby('ID')[['ndvi', 'ndwi', 'savi', 'evi', 'bsi', 'msi', 'ndre', 'ndwi_8a']].agg(['min','mean', 'std', 'max'])
    s2_year.columns = [f'{col}_{i}' for col, i in s2_year.columns]

    clim_year = clim.groupby('ID')[['LST_celsius', 'temps', 'ro', 'pr', 'evapotrans', 'wsi', 'soil']].agg(['mean', 'std'])
    clim_year.columns = [f'{col}_{i}' for col, i in clim_year.columns]

    locations = locations.merge(s1_year, on='ID', how='left').merge(s2_year, on='ID', how='left').merge(clim_year, on='ID', how='left')

    temp = locations[['ID']].set_index('ID').copy() #pd.DataFrame(index=np.sort(s2_temp['ID'].unique()))
    m = 6 # month smooth
    temp.index.name = 'ID'

    for col in ['VV', 'VH']:
        # calc harmonic regression (seasonal changes)
        # group by mean over the years and find seasonal changes
        harm_result = s1_temp.groupby('ID').apply(partial(harmonic_regression, col=col), include_groups=False)
        harm_coeffs = harm_result.pipe(np.stack)
        temp[[f'{col}_ampli', f'{col}_phase', f'{col}_peak']] = harm_coeffs

        # monthly changes over time
        lin_coeffs = s1_temp.groupby('ID').apply(linear_regression, col=col, add_diff=False, include_groups=False).pipe(np.stack)
        temp[f'{col}_month_chg'] = lin_coeffs[:, 1] # take slope
        
        # adding trend acceleration (by adding monthly change)
        lin_coeffs = s1_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, include_groups=False).pipe(np.stack)
        temp[f'{col}_month_accel'] = lin_coeffs[:, 1] # take slope
    
    for col in ['ndvi', 'ndwi', 'bsi', 'msi', 'ndre', 'savi']:
        # calc harmonic regression (seasonal changes)
        # group by mean over the years and find seasonal changes
        if col in ['ndvi', 'ndwi', 'bsi', 'msi']:
            harm_result = s2_temp.groupby('ID').apply(partial(harmonic_regression, col=col), include_groups=False)
            harm_coeffs = harm_result.pipe(np.stack)
            temp[[f'{col}_ampli', f'{col}_phase', f'{col}_peak']] = harm_coeffs
        
        # monthly changes over time
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, include_groups=False).pipe(np.stack)
        temp[f'{col}_month_chg'] = lin_coeffs[:, 1] # take slope
        
        # adding trend acceleration (by adding monthly change)
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, include_groups=False).pipe(np.stack)
        temp[f'{col}_month_accel'] = lin_coeffs[:, 1] # take slope
        
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, include_groups=False).pipe(np.stack)
        temp[f'{col}_month_pct'] = lin_coeffs[:, 1] # take slope
        
        # every 6 month
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, add_rolling=True, 
                                                    n_roll=m, include_groups=False).pipe(np.stack)
        temp[f'{col}_{m}month_accel_ave'] = lin_coeffs[:, 1] # take slope

        # adding trend acceleration (by adding monthly pct change)
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, add_rolling=True, 
                                                    n_roll=m, include_groups=False).pipe(np.stack)
        temp[f'{col}_{m}month_pct_ave'] = lin_coeffs[:, 1] # take slope

        # volatility
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, add_rolling=True, 
                                                    n_roll=m, include_groups=False, roll_func='std').pipe(np.stack)
        temp[f'{col}_{m}month_accel_std'] = lin_coeffs[:, 1] # take slope

        # adding trend acceleration (by adding monthly pct change)
        lin_coeffs = s2_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, add_rolling=True, 
                                                    n_roll=m, include_groups=False, roll_func='std').pipe(np.stack)
        temp[f'{col}_{m}month_pct_std'] = lin_coeffs[:, 1] # take slope
        
        # auc- green-up rate
        if col == 'ndvi':
            aucs = s2_temp.groupby('ID').apply(lambda x: np.trapezoid(
                x[col][~np.isnan(x[col])], np.arange(len(x['month']))[~np.isnan(x[col])]), include_groups=False)
            temp[f'{col}_auc'] = aucs

    for col in ['LST_celsius', 'pr', 'ro', 'temps', 'evapotrans', 'soil', 'wsi']: # climate
        # monthly change
        lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, include_groups=False).pipe(np.stack))
        temp[f'{col}_month_chg'] = lin_coeffs[:, 1]

        # accel
        lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, include_groups=False).pipe(np.stack))
        temp[f'{col}_month_accel'] = lin_coeffs[:, 1]
            
        # pct change
        lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, include_groups=False).pipe(np.stack))
        temp[f'{col}_month_pct'] = lin_coeffs[:, 1]
        
        # accel (6 months changes) rolling mean, sum or variation
        roll_func = 'sum' if col == 'pr' else 'mean'
        lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, add_rolling=True, n_roll=m, 
                                                    include_groups=False, roll_func=roll_func).pipe(np.stack))
        temp[f'{col}_{m}month_accel_{roll_func}'] = lin_coeffs[:, 1]
            
        # pct change
        lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, add_rolling=True, n_roll=m, 
                                                    include_groups=False, roll_func=roll_func).pipe(np.stack))
        temp[f'{col}_{m}month_pct_{roll_func}'] = lin_coeffs[:, 1]

        # volatility
        if col == 'LST_celsius':
            # accel
            lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_diff=True, add_rolling=True, n_roll=m, 
                                                        include_groups=False, roll_func='std').pipe(np.stack))
            temp[f'{col}_{m}month_accel_std'] = lin_coeffs[:, 1]
            # pct change
            lin_coeffs = (clim_temp.groupby('ID').apply(linear_regression, col=col, add_pct=True, add_rolling=True, n_roll=m, 
                                                        include_groups=False, roll_func='std').pipe(np.stack))
            temp[f'{col}_{m}month_pct_std'] = lin_coeffs[:, 1]

    locations[temp.columns.tolist()] = temp.values
    # locations = pd.concat([locations.set_index('ID'), temp], axis=1).reset_index()
    return locations

def encode_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'

def one_hot_encode(values):
    values = np.array(values)
    categories, inverse = np.unique(values, return_inverse=True)
    # drops the last category
    one_hot = np.eye(len(categories))[inverse]
    return one_hot, categories.tolist()

def compute_s2_indices(s2):
    s2['ndvi'] = list(map(lambda x: calc_ndvi(*x), s2[['B4', 'B8']].values))
    s2['ndwi'] = list(map(lambda x: calc_ndwi(*x), s2[['B8', 'B11']].values))
    s2['ndwi_8a'] = list(map(lambda x: calc_ndwi2(*x), s2[['B8A', 'B11']].values))
    s2['msi'] = list(map(lambda x: calc_msi(*x), s2[['B8', 'B11']].values))
    s2['evi'] = list(map(lambda x: calc_evi(*x), s2[['B2', 'B4', 'B8']].values))
    s2['bsi'] = list(map(lambda x: calc_bsi(*x), s2[['B2', 'B4', 'B8', 'B11']].values))
    s2['savi'] = list(map(lambda x: calc_savi(*x), s2[['B4', 'B8']].values))
    s2['ndre'] = list(map(lambda x: calc_ndre(*x), s2[['B5', 'B8']].values))
    return s2

def compute_s1_indices(s1):
    s1['vh_vv_ratio'] = sar_ratios(s1.VV, s1.VH)
    return s1

# Helper functions to get sentinel indices
def calc_ndvi(b4, b8):
    denom = b4 + b8
    ndvi = 0 if denom == 0 else (b8-b4)/denom
    return np.clip(ndvi, -1, 1)
def calc_savi(b4, b8):
    L= 0.5
    savi = (b8-b4)/(b4+b8+L) * (1+L)
    return np.clip(savi, -1, 1)
def calc_ndwi(b8, b11):
    denom = b8+b11
    ndwi = 0 if denom == 0 else (b8-b11)/denom
    return np.clip(ndwi, -1, 1)
def calc_ndwi2(b8a, b11):
    denom = b8a+b11
    ndwi = 0 if denom == 0 else (b8a-b11)/denom
    return np.clip(ndwi, -1, 1)
def calc_evi(b2, b4, b8):
    evi = 2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1)
    return np.clip(evi, -1, 1)
def calc_msi(b8, b11):
    return b11/b8 if b8 != 0 else 0
def calc_bsi(b2, b4, b8, b11):
    numer = (b11+b4) - (b8+b2)
    return numer/ ((b11 + b4) + (b8+b2))
def calc_ndre(b5, b8):
    ndre = (b8-b5)/(b8+b5)
    return np.clip(ndre, -1, 1)
def sar_ratios(vv_db, vh_db):
    return vh_db - vv_db



# load datasets
print('Loading Datasets\n=======================================')

test = pd.read_csv('../data/test.csv')

# sentinel
train_s1 = pd.concat([
    pd.read_csv('../data/fergana_s1.csv'),  
    pd.read_csv('../data/orenburg_s1.csv')
], ignore_index=True)

train_s2 = pd.concat([
    pd.read_csv('../data/fergana_s2.csv'),  
    pd.read_csv('../data/orenburg_s2.csv')
], ignore_index=True)

test_s1 = pd.read_csv('../data/Sentinel1.csv')
test_s2 = pd.read_csv('../data/Sentinel2.csv')


# elevation models
train_dem = pd.concat([
    pd.read_csv('../data/fergana_dem.csv'),
    pd.read_csv('../data/orenburg_dem.csv')
], ignore_index=True)

test_dem = pd.read_csv('../data/test_dem.csv')


# climate
train_clim = pd.concat([
    pd.read_parquet('../data/fergana_climate.parquet'),
    pd.read_parquet('../data/orenburg_climate.parquet')
], ignore_index=True)

test_clim = pd.read_parquet('../data/test_climate.parquet')

print('Dataset Loading completed\n')


# Aggregating test sentinel 1 and 2 into monthly aggregates

# rounding to the month
test_s1['date'] = test_s1['date'].astype('datetime64[ns]').dt.to_period('M').dt.start_time
test_s2['date'] = test_s2['date'].astype('datetime64[ns]').dt.to_period('M').dt.start_time

train_s1['date'] = train_s1['date'].astype('datetime64[ns]')
train_s2['date'] = train_s2['date'].astype('datetime64[ns]')

train_clim['date'] = train_clim['date'].astype('datetime64[ns]')
test_clim['date'] = test_clim['date'].astype('datetime64[ns]')


# Generate train data
train = train_s2[['ID', 'x', 'y', 'location', 'Cropland']].drop_duplicates().reset_index(drop=True)

# merge location and ID to get unique values
train['ID'] = 'ID_' + train.ID.astype(str) + '_' + train.location
train_s1['ID'] = 'ID_' + train_s1.ID.astype(str) + '_' + train_s1.location
train_s2['ID'] = 'ID_' + train_s2.ID.astype(str) + '_' + train_s2.location
train_dem['ID'] = 'ID_' + train_dem.ID.astype(str) + '_' + train_dem.location
train_clim['ID'] = 'ID_' + train_clim.ID.astype(str) + '_' + train_clim.location


# sentinel 1 and 2 column names
s1_cols = ['VV', 'VH']
s2_cols = train_s2.filter(regex=r'B\w+$').columns.tolist()


# Average to monthly data to match those extracted from train data
test_s2 = test_s2.groupby(['ID', 'date'])[s2_cols+['translated_lon', 'translated_lat']].mean().reset_index()
test_s1 = test_s1.groupby(['ID', 'date'])[s1_cols+['translated_lon', 'translated_lat']].mean().reset_index()

# normalise sentinel2 data to 0-1 range
test_s2[s2_cols] = test_s2[s2_cols]/1e4


# Create GeoDataFrame for train and test locations
train_locs = gpd.GeoDataFrame(train[['ID', 'location']], crs=4326, 
                              geometry=gpd.points_from_xy(train.x, train.y))

test_locs = gpd.GeoDataFrame(test[['ID', 'location']], crs=4326, 
                             geometry=gpd.points_from_xy(test.translated_lon, test.translated_lat))

# Aggregate test s1 and s2 coordinates for each ID to get centroid (mean values)
test_s1_locs = (
    test_s1[['ID', 'translated_lat', 'translated_lon']]
    .groupby('ID').mean().reset_index()
    .pipe(lambda df: gpd.GeoDataFrame(
        df[['ID']], crs=4326, geometry=gpd.points_from_xy(df.translated_lon, df.translated_lat))
    )
)

test_s2_locs = (
    test_s2[['ID', 'translated_lat', 'translated_lon']]
    .groupby('ID').mean().reset_index()
    .pipe(lambda df: gpd.GeoDataFrame(
        df[['ID']], crs=4326, geometry=gpd.points_from_xy(df.translated_lon, df.translated_lat))
    )
)



# All location lookup table
location_info = pd.concat([
    test_locs.merge(test_dem[['ID', 'location', 'elevation', 'slope']], on=['ID', 'location']),
    train_locs.merge(train_dem[['ID', 'location', 'elevation', 'slope']], on=['ID', 'location'])], ignore_index=True)



# Clustering locations into regions
# 
# - Here, we will cluster locations into groups (regions). Because there are two regions: Fergana and Orenburg, we will cluster sites in each region separately.
# - Next, we will bin coordinates into grids. We will use the 0.15 grid size, indicating locations within about 16km latitude. 
#  These will be used to calculate summary statistics of sites within this radius.


# get location bounds (upper and lower bounds based on 
location_bounds = location_info.groupby('location').apply(
    lambda x: x.get_coordinates(), include_groups=False).groupby('location').agg(
        minx= pd.NamedAgg('x', 'min'), miny= pd.NamedAgg('y', 'min'), 
        maxx= pd.NamedAgg('x', 'max'), maxy= pd.NamedAgg('y', 'max')
).T.to_dict()


# regions within 50km distance grid cell
location_info = bin_coordinates(location_info, location_bounds, grid_size=None, distance=50)


# get optimal clusters
print('Clustering locations into groups\n=================================')
get_optimal_clusters(location_info.query('location == "Fergana"').get_coordinates().map(np.radians), scale=False, scorer='elbow')
plt.savefig('imgs/fergana_optimal_clusters.png')

get_optimal_clusters(location_info.query('location == "Fergana"').get_coordinates().map(np.radians), scale=False, scorer='elbow')
plt.savefig('imgs/orenburg_optimal_clusters.png')
print('Clustering completed!\n')

# Grouping in 4 clusters for each region

location_info.loc[location_info.location == "Fergana", 'region'] = 'F' + cluster_data(
    location_info.query('location == "Fergana"').geometry.get_coordinates().map(np.radians), 4)[0].astype(str)

location_info.loc[location_info.location != "Fergana", 'region'] = 'O' + cluster_data(
    location_info.query('location != "Fergana"').geometry.get_coordinates().map(np.radians), 4)[0].astype(str)


# add water stress index (wsi)
train_clim = train_clim.assign(wsi = 1-(train_clim.evapotrans/train_clim.p_evapotrans))
test_clim = test_clim.assign(wsi = 1-(test_clim.evapotrans/test_clim.p_evapotrans))

##################################################################################################################################
# Feature Engineering
print('Extracting Features\n=====================================')
print('Extracting for Test data')
test_agg = extract_features(test_dem, test_s1, test_s2, test_clim)
print('Extracting for Train data')
train_agg = extract_features(train_dem, train_s1, train_s2, train_clim)
print('Feature extraction completed!\n')

# reset index to be like in train
train_agg = train_agg.set_index('ID').reindex(train.ID).reset_index()
test_agg = test_agg.set_index('ID').reindex(test.ID).reset_index()
###########################################################################################################

# Adding Location information
# - We will add distance to nearest site
# - Number of sites within a certain radius, their average distance and standard deviation
print('Adding site Location information\n')
nearest_site = get_nearest_neighbors(location_info, location_info, n_neighbors=[2])
sites_10km = get_nearest_neighbors(location_info, location_info, n_neighbors=list(range(2,30)), threshold=10000)

# aggregate site specific information
sites_10km_agg = sites_10km.groupby('ID').agg(mean_dist_10km = pd.NamedAgg('distance', 'mean'),
                                              n_sites_10km = pd.NamedAgg('nn_PID', 'size'), 
                                              std_dist_10km = pd.NamedAgg('distance', 'std'))

dist_df = (nearest_site
           .merge(sites_10km_agg, on='ID', how='left')
           .rename(columns={'distance':'nearest_site_dist'})
           .drop('nn_PID', axis=1)
)

# fill sites with no site within 10km with 0
dist_df.n_sites_10km = dist_df.n_sites_10km.fillna(0)


# grid based S2-information
grid_s2 = (
    pd.concat([compute_s2_indices(train_s2), compute_s2_indices(test_s2)], ignore_index=True)
    .merge(location_info[['ID', 'region', 'grid_id']], on='ID')
    .groupby('grid_id')[['ndvi', 'bsi', 'msi', 'ndwi']]
    .agg(['min', 'mean', 'std', 'max'])
    )
grid_s2.columns = [f'grid_{i}_{j}' for i, j in grid_s2.columns]



# grid based climatic-information
grid_clim = (
    pd.concat([train_clim, test_clim], ignore_index=True)
    .merge(location_info[['ID', 'region', 'grid_id']], on='ID')
    .groupby('grid_id')[['wsi', 'evapotrans', 'LST_celsius', 'pr', 'temps']]
    .agg({
        'wsi' : ['min', 'mean', 'std', 'max'],
        'evapotrans': ['min', 'mean', 'std', 'max'],
        'LST_celsius': ['min', 'mean', 'std', 'max'],
        'temps': ['min', 'mean', 'std', 'max'],
        'pr': ['mean', 'std', 'max']
        })
    )
grid_clim.columns = [f'grid_{i}_{j}' for i, j in grid_clim.columns]


test_agg = (test_agg
            .merge(grid_clim
                   .merge(grid_s2, on='grid_id')
                   .merge(location_info[['ID', 'grid_id', 'region']], on='grid_id')
                   .merge(dist_df, on='ID', how='left'), 
            on='ID', how='left'))

train_agg = (train_agg
             .merge(grid_clim
                    .merge(grid_s2, on='grid_id')
                    .merge(location_info[['ID', 'grid_id', 'region']], on='grid_id')
                    .merge(dist_df, on='ID', how='left'), 
            on='ID', how='left'))



test_agg.isna().sum().nlargest(5)

train_agg.isna().sum().nlargest(5)


# extract ID again
train_agg['ID'] = train_agg.ID.str.extract(r'([0-9]+)').map(np.int32)

# save data
print('\nSaving Datasets\n=============================')
test_agg.to_parquet('../data/test_agg3.parquet')
train_agg.to_parquet('../data/train_agg3.parquet')
print('Completed!\n')

end  = time.time()
print(f'Total Time taken: {(end-start)/60:.6f} mins')