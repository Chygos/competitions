# Importing Python packages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings, os
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from typing import Union, Literal
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# loading datasets
train_gap = pd.read_csv('data/Gap_Train.csv')
test_gap = pd.read_csv('data/Gap_Test.csv')
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')


# band data
lsat8 = pd.read_parquet('data/landsat_band.parquet')
surf_temp = pd.read_parquet('data/land_surf.parquet')
sl1 = pd.read_parquet('data/sl1_bands.parquet')
evap_trans = pd.read_parquet('data/evapotrans.parquet')


print('Number of columns with missing rows:\nTrain: {}\tTest: {}'.format(
    train.isna().any().sum(), test.isna().any().sum()))

# %%
# missing columns
missing_cols = train.columns[train.isna().any()].tolist()
print('Missing columns: {}'.format(missing_cols))


# df with PID, their bulk density, and available_ppm (soil nutrient)
missing = train.loc[train.isna().any(axis=1)] # get missing rows
t = (
    pd.concat([train.loc[missing.index, 'PID'], 
               missing.loc[:, 'BulkDensity':'B']], axis=1)
        .melt(id_vars=['PID', 'BulkDensity'], var_name='Nutrient', value_name='Available_ppm')
    )
# merge train gap to it
t = train_gap[['PID', 'Nutrient', 'Available', 'Required']].merge(t, on=['PID', 'Nutrient'], how='right')

# get PID and their calculated bulk densities
bulk_missing_map = (
    t.assign(bulk_density = np.round(t['Available'] / (t['Available_ppm'] * 20 * 0.1), 2))
     .query('BulkDensity.isna()')
     .filter(regex = 'PID|bulk_density')
     .drop_duplicates()
     .set_index('PID').to_dict()['bulk_density']
)

# Merging all data
df = pd.concat([train, test], ignore_index=True)

def get_nearest_PID(data, query_data, k=1, threshold=None):
    """
    Returns PIDs in data closest to query data
    :param data: Reference data containing coordinates
    :param query_data: Data to query (containing coordinates)
    :param k: Number of n-nearest neighbours to return
    """
    k = k+1 # add 1 so that the PIDs that are the same will be dropped

    # convert to Geopandas and convert to a UTM projection (for calculation in metres and KM)
    data_copy = gpd.GeoDataFrame(data[['PID', 'lon', 'lat']], 
                                 geometry=gpd.points_from_xy(data.lon, data.lat), crs=4326).to_crs(32633)
    query_data_copy = gpd.GeoDataFrame(query_data[['PID', 'lon', 'lat']], 
                                       geometry=gpd.points_from_xy(query_data.lon, query_data.lat), crs=4326).to_crs(32633)
    nn = cKDTree(data_copy.get_coordinates())
    # index locations of closest locations
    distances, nearest  = nn.query(query_data_copy.get_coordinates(), k=k)
    distances, nearest = distances.reshape(-1), nearest.reshape(-1)
    # get distance and nearest soil site (PID)
    closest_loc = (
        data[['PID', 'site']].iloc[nearest]
        .rename(columns={'PID': 'nn_PID', 'site':'nn_site'})
        .reset_index(drop=True)
        .assign(distance = distances)
    )

    PIDs = np.repeat(query_data['PID'], k) # repeat to the number of k-nearest neighbours and merge to closest Dataframe
    closest_loc = pd.concat([closest_loc, pd.DataFrame(PIDs).reset_index(drop=True)], axis=1)
    closest_loc = closest_loc.merge(query_data[['site', 'PID']], on='PID')
    # drop PIDs that are the same (distance  = 0) and those whose site ID's are the same
    closest_loc = closest_loc.query('distance > 0 & nn_site  == site').reset_index(drop=True).drop('nn_site', axis=1)
    # # return values with distances less than cutoff
    return closest_loc.query(f'distance <= {threshold}') if threshold is not None else closest_loc 

# target variables
target_variables = train.loc[:, 'N':'B'].columns.tolist()
target_variables_str = '^'+ '$|^'.join(target_variables) + '$'

# Filling missing values
def fill_missing_values(
        data:pd.DataFrame, 
        group_level:Union[None, Literal['nn']]=None, 
        stat_type='mean'):
    """
    Fills missing values
    :param df: DataFrame to fill missing values
    :param group_level: None|str. If None, then missing values are filled by taking the summary statistics chosen in stat_type
                        If 'nn', then missing values are filled using the closest soil sample site. 
                        If missing values exist, then fill using the farm site information
    :param stat_type: Summary statistics type to fill missing values
    """
    df_copy = data.copy()
    is_missing = df_copy.filter(regex= f'^(?!({target_variables_str}))').isna().any()
    # drop target variables
    missing_cols = df_copy.filter(regex= f'^(?!({target_variables_str}))').columns[is_missing]
    
    # filling missing values for bulk_density
    df_copy.loc[df_copy.BulkDensity.isna(), 'BulkDensity'] = df_copy.loc[df_copy.BulkDensity.isna(), 'PID'].map(bulk_missing_map)

    if group_level is None:
        agg_values = df_copy[missing_cols].agg(stat_type)
        df_copy = df_copy.fillna(agg_values)
    elif group_level == 'nn':
        all_df = df_copy.copy()
        closest_PID = get_nearest_PID(all_df, df_copy, k=1)
        # get information about closest PID
        closest_PID = all_df.rename(columns={'PID': 'nn_PID'}).merge(closest_PID, on=['nn_PID', 'site'], how='right')
        agg_values = closest_PID.groupby('PID')[missing_cols].agg(stat_type).to_dict()
        for col in missing_cols:
            missing_rows = df[col].isna()
            # replace with information of closest PID
            df_copy.loc[missing_rows, col] = df_copy.loc[missing_rows, 'PID'].map(agg_values[col]).values
            # replace with site-level information if no closest PID is found
            if df_copy[col].isna().any():
                site_values = all_df.groupby('site')[col].agg(stat_type).to_dict()
                df_copy.loc[df_copy[col].isna(), col] = df_copy.loc[df_copy[col].isna(), 'site'].map(site_values)
            # if still missing, fill in with regional data
            if df_copy[col].isna().any(): 
                regional_values = all_df.assign(reg_id = all_df.site.str.extract(r'(site_id_\w)')).groupby('reg_id')[col].agg(stat_type).to_dict()
                df_copy.loc[df_copy[col].isna(), col] = df_copy.assign(reg_id = df_copy.site.str.extract(r'(site_id_\w)')).loc[df_copy[col].isna(), 'reg_id'].map(regional_values)
    return df_copy


# Feature engineering
def harmonic_regression(months, values):
    """
    Fit a simple harmonic regression model to capture seasonal cycle:
    y = a0 + a1*cos(2*pi*t/12) + b1*sin(2*pi*t/12)
    Returns coefficients (a0, b1, b2)
    """
    t = 2 * np.pi * months / 12
    X = np.column_stack([np.ones(len(t)), np.cos(t), np.sin(t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    return coeffs

def aggregate_farm_data(df, site_type:Literal['PID', 'site'], 
                        return_harmonic_regression=False,
                        harmonic_features='|'.join(['evi', 'ndvi', 'savi']), 
                        return_monthly_changes=False,
                        monthly_change_features = '|'.join(['ndvi', 'nbr', 'ndmi'])):
    """
    Aggregate monthly data per field with summary statistics, phenological metrics, and temporal dynamics.
    :param df: Time series DataFrame
    :param site_type: Level of granularity
    :param return_harmonic_regression: Boolean. To return harmonic regression coefficients
    :param harmonic_features: Str. Regular Expression pattern of columns to calculate harmonic regression parameters
    :param return_monthly_change: Boolean. To return monthly change of band indices for each site type
    :param monthly_change_features: Str. Regular Expression pattern of variables to calculate monthly changes (linear regression)
    
    Returns:
    sites: DataFrame with one row per field and aggregated features
    """
    sites = df[[site_type]].drop_duplicates().reset_index(drop=True)
    colnames = df.select_dtypes('number').columns
    
    # aggregate month-year data to months since they don't show any obvious trend from the visualisation above
    month_data = df.assign(month = df.mdate.dt.month).groupby([site_type, 'month'])[colnames].mean().reset_index()

    # get annual min, mean, max, and std values for all fields
    # exclude columns with index (composite index)
    annual_stats = df.groupby(f'{site_type}')[colnames].agg(['mean','std']).filter(regex='^(?!.*_index).*')
    annual_stats.columns = [f'{i}_{j}' for i, j in annual_stats.columns]
    mean_vals = annual_stats.filter(regex='mean').filter(regex='^(?!.*_index).*')
    std_vals = annual_stats.filter(regex='std').filter(regex='^(?!.*_index).*')
    coef_var = pd.DataFrame(std_vals.values / mean_vals.values, columns=std_vals.columns)
    coef_var.columns = coef_var.columns.str.replace('std', 'coef_var', regex=True)

    annual_stats = pd.concat([annual_stats.reset_index(), coef_var.reset_index(drop=True)], axis=1)
    sites = sites.merge(annual_stats, on=site_type, how='left') # merge aggregated features to site type dataframe
    
    # harmonic regression coefficients
    if return_harmonic_regression:
        harmonic_features = df.filter(regex=f'{harmonic_features}').columns
        for col in harmonic_features:
            # harmonic regression 
            hr_result = month_data.groupby([site_type]).apply(lambda x: harmonic_regression(x['month'], x[col]))
            hr_result = pd.DataFrame(np.stack(hr_result), 
                                        index=hr_result.keys(), 
                                        columns=[f'{col}_harmonic_b0', f'{col}_harmonic_b1', f'{col}_harmonic_b2'])
            b1 = hr_result[f'{col}_harmonic_b1'] # cos phase
            b2 = hr_result[f'{col}_harmonic_b2'] # sin phase
            hr_result[f'{col}_amplitude'] = np.sqrt(b1**2 + b2**2)
            hr_result[f'{col}_phase'] = np.arctan2(-b2, b1)
            sites = sites.merge(hr_result.filter(regex='ampli|phase').reset_index(), on=f'{site_type}')
            
            # AUC (Area under the curve) - Extent of vegetative growth in a year
            auc_res = month_data.groupby(site_type).apply(lambda x: np.trapz(x[col], x['month'].astype(int)))
            # auc_res = df.assign(month = df.mdate.dt.month).groupby(site_type).apply(lambda x: np.trapz(x[col], x['month'].astype(int)))
            auc_res = auc_res.reset_index(name=f"{col.replace('mean', '')}_auc")
            sites = sites.merge(auc_res, on=site_type)
    
    # slope trend and intercept (rate of change per month per year)
    if return_monthly_changes:
        monthly_change_features = df.filter(regex=f'{monthly_change_features}').columns
        for col in monthly_change_features:
            # Temporal dynamics - linear trend (monthly change over time)
            lr_result = month_data.groupby(site_type).apply(lambda x: np.polyfit(x['month'], x[col], 1))

            lr_result = pd.DataFrame(np.stack(lr_result), 
                                index=lr_result.keys(), 
                                columns=[f'{col}_slope', f'{col}_intercept'])
            sites = sites.merge(lr_result.filter(regex='slope').reset_index(), on=site_type)
    return sites


def obtain_composite_index(df, features:Union[str, list], 
                           return_pcs=False, composite_name=None,
                           weight_type:Literal['equal', 'weighted']='weighted',
                           weights:Union[Literal['pca'], list, tuple, None]=None):
    """
    Combines spectral band indices into a composite index value.
    First columns are scaled in a 0-1 range and combined based on a equal or given weight type
    :param features: A list of column names or regular expression representing columns to be selected
    :param return_pcs: Boolean. To return Principal components
    :param composite_name: Name to assign to the created composite index
    :param weight_type: Type of weight to apply. Equal or weighted
    :param weights: Asigned weights. Weigths obtained via PCA loadings on first PC component, list of weights or None
                    If None, weights are obtained by dividing each column's scaled contribution by their total values

    """
    df_extract = df.filter(regex=features) if isinstance(features, str) else df[features]
    scaler = MinMaxScaler()
    pca = PCA(random_state=0)
    # scale data and apply PCA to get weights
    df_extract_scaled = pd.DataFrame(scaler.fit_transform(df_extract), columns=df_extract.columns)

    if weight_type == 'equal':
        df_extract_scaled[composite_name] = df_extract_scaled.mean(1)
    elif weight_type == 'weighted':
        if weights is None:
            weights = df_extract_scaled.to_numpy() / df_extract_scaled.to_numpy().sum(axis=1, keepdims=True)
        elif weights == 'pca':
            pca.fit(df_extract_scaled) # use pca to obtain weights
            loadings = pca.components_[0] # loadings for PC1
            pca_weights =  loadings * np.sqrt(pca.explained_variance_[0]) # normalise loadings by standard deviation
            weights = pca_weights/(pca_weights).sum() # obtain normalised weights
        elif isinstance(weights, (list, tuple, dict)):
            assert len(weights) == df_extract.shape[1], "Length of weights must equal the number of columns"
            if isinstance(weights, dict): # if a dictionary, map feature names to weight values
                feature_names = df_extract.columns.tolist()
                weights = list(weights.get(col, None) for col in feature_names)
        df_extract_scaled[composite_name] = np.average(df_extract_scaled[df_extract.columns], weights= weights, axis=1)

    if return_pcs:
        pca.fit(df_extract_scaled[df_extract.columns])
        df_extract_scaled[f'{composite_name}_PC1'] = pca.transform(df_extract_scaled[df_extract.columns])[:, 0]
    new_cols = df_extract_scaled.columns.symmetric_difference(df_extract.columns)
    df_extract[new_cols] = df_extract_scaled[new_cols]
    return df_extract

# Aggregating vegetative and soil moisture band indices to obtain a composite index
veg_idx = 'ndvi|gndvi|savi|evi'
sm_idx = 'ndmi|msi'

# Extract composite index and assign to dataframe
print('===> Computing composite indices')
lsat8 = lsat8.assign(
    land_surf_index = obtain_composite_index(
        df=lsat8, features=f'{veg_idx}|{sm_idx}|nbr|bsi', 
        composite_name='land_surf_index', weight_type='weighted', 
        weights=None).iloc[:, [-1]].values,
    veg_index = obtain_composite_index(
        df=lsat8, features=veg_idx, 
        composite_name='veg_index', weight_type='equal', 
        weights=None).iloc[:, [-1]].values,
    moisture_index = obtain_composite_index(
        df=lsat8, features=sm_idx, 
        composite_name='moisture_index', 
        weight_type='equal', weights=None).iloc[:, [-1]].values,
    veg_sm_index = obtain_composite_index(
        df=lsat8, features=f'{sm_idx}|{veg_idx}', 
        composite_name='veg_sm_index', 
        weight_type='weighted', weights='pca').iloc[:, [-1]].values,
    baresoil_burn_index = obtain_composite_index(
        df=lsat8, features='nbr|bsi', 
        composite_name='baresoil_burn_index', 
        weight_type='weighted', weights=None).iloc[:, [-1]].values,
    sm_bb_index = obtain_composite_index(
        df=lsat8, features=f'{sm_idx}|nbr|bsi', 
        composite_name='sm_bb_index', 
        weight_type='weighted', weights=None).iloc[:, [-1]].values
)

# %%
# constants for calculating land surface temperature
print('Performing data aggregation\n=============================')
K1 = 774.8853  
K2 = 1321.0789
print('===> Aggregating Landsat-8')
lsat8_agg = (
    lsat8
    .assign(brightness_temp = K2/ np.log((K1/lsat8.ST_B10)+1),
            clay_index_ratio = lsat8.swir1/lsat8.swir2,
            nbr2 = (lsat8.swir2-lsat8.swir1)/(lsat8.swir2+lsat8.swir1),
            red_blue_ratio = lsat8.red/lsat8.blue,
            swir1_3_ratio = lsat8.swir3/lsat8.swir1)
    .pipe(aggregate_farm_data, site_type='PID', return_harmonic_regression=True, 
                                harmonic_features='ndvi|savi|evi|temp|..+index$', 
                                return_monthly_changes=True, 
                                monthly_change_features = 'ndvi|savi|evi|..+index$')
)

# %%
print('===> Aggregating Sentinel-1')
sl1_agg = (
    sl1
    .assign(vv_vh = 10**(sl1.VV/10)/10**(sl1.VH/10), 
            vv_vh_diff = sl1.VV - sl1.VH, 
            ndpi = (sl1.VV - sl1.VH)/(sl1.VV + sl1.VH))
    .pipe(aggregate_farm_data, site_type='PID', return_harmonic_regression=True, 
                               harmonic_features='vv_vh',
                               return_monthly_changes=True, 
                               monthly_change_features='vv_vh')
)

# %%
print('===> Aggregating Evapotranspiration')
et_agg = (
    evap_trans
    .assign(wsi = evap_trans.ET/evap_trans.PET)
    .pipe(aggregate_farm_data, site_type='PID', return_harmonic_regression=True, 
                               harmonic_features='wsi', 
                               return_monthly_changes=True, 
                               monthly_change_features='wsi')
)

# %%
print('===> Aggregating Surface temperature')
surf_temp_agg = (
    surf_temp
    .assign(daily_LST = 0.5*(surf_temp.LST_Day_1km+surf_temp.LST_Night_1km),
            LST_diff = surf_temp.LST_Day_1km - surf_temp.LST_Night_1km)
    .pipe(aggregate_farm_data, site_type='PID', return_harmonic_regression=True, 
                               harmonic_features='LST_diff|daily_LST', 
                               return_monthly_changes=True, 
                               monthly_change_features='daily_LST|LST_diff')
)

# %%
# Merging datasets
print('Merging datasets')
df = (
    df.merge(et_agg, on='PID', how='left')
      .merge(surf_temp_agg, on='PID', how='left')
      .merge(lsat8_agg, on='PID', how='left')
      .merge(sl1_agg, on='PID', how='left')     
)#.filter(regex='^(?!(swir\d|nir|blue|red|green|ST))') # drop reflectance values

# %%
# filling missing values
print('===> Filling missing values')
df = fill_missing_values(data=df, group_level='nn')
df.head()

# %%
# generating more features
print('===> Creating feature interaction terms')
df = df.assign(
    dryness_index= df.bio1/df.bio12,
    diurnal_temp_range = df.lstd - df.lstn,
    rainfall_precip_ratio = df.bio15/df.bio12,
    org_density = df.soc20 / df.BulkDensity,
    sand_org_carbon_ratio = np.where(df.soc20 == 0, 0 ,df.snd20 / df.soc20),
    cec_efficiency = df.cec20 - df.ecec20,
    cec_comb = (df.cec20 + df.ecec20)/2,
    ph_diff = df.pH - df.ph20,
    ph_comb = (df.pH + df.ph20)/2,
    acidity_ratio = np.where(df.xhp20 == 0, 0, df.hp20 / df.xhp20),
    tim_slope_ratio = np.where(df.slope == 0, 0, df.tim /(df.slope/100)),
    tim_slope = df.tim *(df.slope/100),
    slope_elev = df.mdem * df.slope/100,
    slope_elev_ratio = np.where(df.slope==0, 0, df.mdem / (df.slope/100)),
    alb_para = df.alb *df.para/100,
    # added newly
    sand_organic_carbon = df.snd20*df.soc20,
    rainfall_elev = (df.bio12/1000*df.mdem/1000),
    ph_cec = df.cec20*df.pH,
    ph_soc20 = df.soc20*df.pH
)
print(df.shape)
# split into train and test sets
train = df[~df[target_variables].isna().any(axis=1)].replace([np.inf, -np.inf], 0)
test = df[df[target_variables].isna().any(axis=1)].drop(columns=target_variables).replace([np.inf, -np.inf], 0)

print(f'Test: {test.shape}\tTrain: {train.shape}')

print('===> Saving datasets')
train.to_parquet('data/train_preprocessed.parquet', index=False)
test.to_parquet('data/test_preprocessed.parquet', index=False)
print('===> Datasets saved')