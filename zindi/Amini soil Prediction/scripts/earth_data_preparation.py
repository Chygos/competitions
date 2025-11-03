
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import warnings
from pathlib import os
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# Loading train and test data with soil parameters

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')

# Loading earth observation datasets (Land surface, soil temperature, Sentinel-1 and Landsat-8)
modis_a1 = pd.read_csv('data/MODIS_MOD11A1_data.csv')
modis_a2 = pd.read_csv('data/MODIS_MOD16A2_data.csv')
lsat = pd.read_csv('data/LANDSAT8_data_updated.csv')
sl1 = pd.read_csv('data/Sentinel1_data.csv')

# converting string dates to date type
for df in [modis_a1, modis_a2, lsat, sl1]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['mdate'] = pd.to_datetime(df['date'].astype(str).str.replace(r'\d+$', '01', regex=True))


# Checking if datasets are in their 0-1 range
# Reflectance values are usually in the 0-1 range. If they exist as DN values, they're normalised by dividing by 10,000 which is the maximum DN unit, resulting in 0-1 reflectance range.
# So, in the next cell, we will find satellite data with bands (represented as B or band), and get their minimum and maximum values

for df, name in zip([modis_a1, modis_a2, lsat, sl1], ['a1', 'a2', 'l8', 's1']):
    cols = df.filter(regex=r'.?[bB](.+)?\d+').columns
    if len(cols) > 0:
        print(f'{name}\n')
        print(df[cols].agg(['min', 'max']))
        print()

# - Some satellite data are in beyond the 0-1 range, while some others (s2) are still in their DN units (with above 10k max unit). 
# - For dataframes beyond the 0-1 range, we'd clip to 0 and 1, while s2 will be normalised by dividing by 10,000 which is the max range of a DN unit
for df in [modis_a1, modis_a2, lsat, sl1]:
    cols = df.filter(regex=r'.?[bB](.+)?\d+').columns
    if len(cols) == 0:
        continue
    if df[cols].dtypes.unique() in ['int64', 'int32']: # if satellite data is integer then it is still in their DN values
        df[cols] = df[cols].clip(0, 1e4) / 1e4 # clip to 0 and 1e4 and normalise by 1e4
    elif df[cols].dtypes.unique() in ['float64', 'float32']:
        df[cols] = df[cols].clip(0, 1)

# checking if we've solved the problem
for df, name in zip([modis_a1, modis_a2, lsat, sl1], 
                    ['a1', 'a2', 'a4', 'ga', 'q1', 'l8', 's1', 's2']):
    cols = df.filter(regex=r'.?[bB](.+)?\d+').columns
    if len(cols) > 0:
        print(f'{name}\n')
        print(df[cols].agg(['min', 'max']))
        print()


# Monthly land surface temperature
modis_a1_mean = modis_a1.drop(['lat', 'lon', 'date'], axis=1).groupby(['PID', 'mdate']).mean().reset_index()
# Evapotranspiration
modis_a2_mean = modis_a2.groupby(['PID', 'mdate'])[['ET', 'PET']].mean().reset_index()



# Helper functions to calculate band indices

def calc_ndvi(red, nir):
    eps = 1e-6
    red = np.clip(red, 0, None); nir = np.clip(nir, 0, None)
    ndvi = (nir-red)/(nir+red+eps)
    return np.clip(ndvi, -1, 1)
def calc_evi(blue, red, nir):
    eps = 1e-6;red = np.clip(red, 0, None); nir = np.clip(nir,0, None); blue = np.clip(blue, 0, None)
    evi = 2.5 * (nir-red)/ (nir + 6*red - 7.5*blue + 1 + eps)
    return np.clip(evi, -1, 1)
def calc_ndre(red_edge1, nir):
    eps = 1e-6; red_edge1 = np.clip(red_edge1, 0, None); nir = np.clip(nir, 0, None)
    ndre = (nir - red_edge1)/ (nir+red_edge1+eps)
    return np.clip(ndre, -1, 1)
def calc_ndmi(nir, swir1):
    eps = 1e-6; swir1 = np.clip(swir1, 0, None); nir = np.clip(nir, 0, None)
    ndwi = (nir - swir1)/(nir + swir1 + eps)
    return np.clip(ndwi, -1, 1)
def calc_sipi(blue, red, nir):
    eps = 1e-6; red = np.clip(red, 0, None); nir = np.clip(nir, 0, None); blue = np.clip(blue, 0, None)
    sipi = (nir-blue)/(nir-red+eps)
    return sipi
def calc_gndvi(nir, green):
    eps = 1e-6; green = np.clip(green, 0, None); nir = np.clip(nir,0, None)
    gndvi = (nir - green) / (nir + green+eps)
    return np.clip(gndvi, -1, 1)
def calc_reci(nir, red_edge1):
    eps = 1e-6; red_edge1 = np.clip(red_edge1, 0.1, None); nir = np.clip(nir, 0, None)
    reci = (nir / (red_edge1+eps)) - 1
    return reci
def calc_renvi(nir, red_edge1):
    eps = 1e-6; red_edge1 = np.clip(red_edge1, 0, None); nir = np.clip(nir, 0, None)
    return nir / (red_edge1+eps)
def calc_swirr(swir1, swir2):
    eps = 1e-6; swir1 = np.clip(swir1, 0, None); swir2=np.clip(swir2, 0, None)
    return (swir1-swir2) / (swir1+swir2+eps)
def calc_savi(nir, red, L=0.5):
    eps = 1e-6; red = np.clip(red, 0, None); nir = np.clip(nir, 0, None)
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return np.clip(savi, -1, 1)
def calc_bsi(swir, red, blue, nir):
    # bare soil index
    eps = 1e-6; red = np.clip(red, 0, None); nir = np.clip(nir, 0, None); blue = np.clip(blue, 0, None); swir=np.clip(swir,0, None)
    bsi = ((swir-red) - (nir + blue))/((swir+red) + (nir + blue) + eps)
    return np.clip(bsi, -1, 1)
    # return np.clip(ndwi, -1, 1)
def calc_nbr(nir, swir2):
    eps = 1e-6; swir2 = np.clip(swir2, 0, None); nir = np.clip(nir, 0, None)
    nbr = (nir - swir2) / (nir+swir2+eps)
    return np.clip(nbr, -1, 1)
def calc_msi(nir, swir2):
    # moisture stress index
    eps = 1e-6; swir2 = np.clip(swir2, 0, None); nir = np.clip(nir, 0.1, None) # if if nir is extremely low to avoid large values
    return swir2 / (nir+eps)


# Landsat and sentinel models
lsat_mean = lsat.filter(regex=r'^S[TR]_|PID|mdate').groupby(['PID', 'mdate']).mean().reset_index()

lsat_mean = lsat_mean.assign(
    evi = lsat_mean.apply(lambda x: calc_evi(x['SR_B1'], x['SR_B3'], x['SR_B4']), axis=1).values,
    ndvi= lsat_mean.apply(lambda x: calc_ndvi(x['SR_B3'], x['SR_B4']), axis=1).values,
    ndmi= lsat_mean.apply(lambda x: calc_ndmi(x['SR_B4'], x['SR_B5']), axis=1).values, # had formerly used B6
    bsi = lsat_mean.apply(lambda x: calc_bsi(x['SR_B6'], x['SR_B3'], x['SR_B1'], x['SR_B4']), axis=1).values,
    msi = lsat_mean.apply(lambda x: calc_msi(x['SR_B4'], x['SR_B6']), axis=1).values,
    gndvi= lsat_mean.apply(lambda x: calc_gndvi(x['SR_B4'], x['SR_B2']), axis=1).values,
    nbr = lsat_mean.apply(lambda x: calc_nbr(x['SR_B4'], x['SR_B6']), axis=1).values,
    savi = lsat_mean.apply(lambda x: calc_savi(x['SR_B4'], x['SR_B3']), axis=1)
)


# Sentinel
sl1_mean = sl1.groupby(['PID', 'mdate'])[['VH', 'VV']].mean(numeric_only=True).reset_index()


# > __Notes__
# - Sentinel-2 data contains PIDs that are not found in both the train or test data
# - Sentinel-1 and MODIS datasets have all PIDs in train and not test data
# - Solve this by getting PIDs of those in earth observation features closet to the PID locations in either train or test sets 
# - Almost similar values for bands and their indices in `modis_ga` and `modis_q1` mean dataframes
#     - Find similar columns in both datasets
#     - Take their mean values
#     - Take this final data as band and indices dataframe
#     - Attach columns not in either datasets to this final dataframe
# - Merge other data to these band and indices dataframe and save for future purposes (Sentinel-2 data may not be used again)


# assigning actual band names as in satellite datasets
band_maps = {
    'red' : ['sur_refl_b01', 'SR_B3', 'B4', 'Nadir_Reflectance_Band1'], 
    'blue' : ['sur_refl_b03', 'SR_B1', 'B2', 'Nadir_Reflectance_Band3'],
    'green' : ['sur_refl_b04', 'SR_B2', 'B3', 'Nadir_Reflectance_Band4'],
    'nir' : ['sur_refl_b02', 'SR_B4', 'B8', 'Nadir_Reflectance_Band2'],
    'swir1' : ['sur_refl_b05', 'SR_B5', 'B11'],
    'swir2' : ['sur_refl_b06', 'SR_B6', 'B12'],
    'swir3' : ['sur_refl_b07', 'SR_B7']
}

# all with their colnames as keys and actual band name as values
band_ids = {}
for i in band_maps.keys():
    band_name = band_maps[i]
    for j in band_name:
        band_ids[j] = i


# print(band_ids)
# get locations in each Earth observation data (satellite data)
a1_locs = modis_a1.groupby(['PID'])[['lat', 'lon']].mean().reset_index()
a2_locs = modis_a2.groupby(['PID'])[['lat', 'lon']].mean().reset_index()
s1_locs = sl1.groupby(['PID'])[['lat', 'lon']].mean().reset_index()
l8_locs = lsat.groupby(['PID'])[['lat', 'lon']].mean().reset_index()


def get_nearest_PID(data, query_data, k=1, threshold=800):
    """
    Returns PIDs in data closest to query data
    :param data: Reference data containing coordinates
    :param query_data: Data to query (containing coordinates)
    :param k: Number of n-nearest neighbours to return
    """
    # convert to Geopandas and convert to a UTM projection (for calculation in metres and KM)
    data_copy = gpd.GeoDataFrame(data[['PID', 'lon', 'lat']], 
                                 geometry=gpd.points_from_xy(data.lon, data.lat), crs=4326).to_crs(32633)
    query_data_copy = gpd.GeoDataFrame(query_data[['PID', 'lon', 'lat']], 
                                       geometry=gpd.points_from_xy(query_data.lon, query_data.lat), crs=4326).to_crs(32633)
    nn = cKDTree(data_copy.get_coordinates())
    # index locations of closest locations
    distances, nearest  = nn.query(query_data_copy.get_coordinates(), k=k)
    closest_loc = data[['PID']].iloc[nearest].rename(columns={'PID': 'nn_PID'}).reset_index(drop=True).assign(distance = distances)
    closest_loc = pd.concat([closest_loc, query_data[['site', 'PID']]], axis=1)
    # return values with distances less than cutoff
    return closest_loc.query(f'distance <= {threshold}') if threshold is not None else closest_loc 

# locations of PIDs in both train and test data
all_df = pd.concat([train, test],ignore_index=True)[['site', 'PID', 'lat', 'lon']]

# get nearest PID
a1_locs_nn = get_nearest_PID(a1_locs, all_df, threshold=500)
a2_locs_nn = get_nearest_PID(a2_locs, all_df, threshold=500)
s1_locs_nn = get_nearest_PID(s1_locs, all_df, threshold=500)
l8_locs_nn = get_nearest_PID(l8_locs, all_df, threshold=500)


def merge_band_data(satellite_df, nn_df):
    """
    Merges PIDs in satellite data to PIDs based on nearest PIDs
    :param satellite_df: Pandas dataframe containing satellite data
    :param nn_df: Dataframe containing all PIDs with their nearest PIDs in satellite data
    :returns merged data
    """
    temp_copy = (
        satellite_df #.filter(regex='.?[bB](.+)?\d+|PID|site|mdate')
        .merge(nn_df, left_on='PID', right_on='nn_PID', suffixes=('_sat', '_main'), how='right')
        .drop(columns=['PID_sat', 'nn_PID', 'distance'])
        .rename(columns={'PID_main':'PID'})
    )
    return temp_copy.set_index(['site', 'PID', 'mdate']) if 'mdate' in temp_copy.columns else temp_copy.set_index(['site', 'PID'])


# LandSat-8 bands
# - Because Landsat-8 satellite offers a higher resolution and accuracy than the MODIS data, we will check for the number of PIDs in both train and test sets that are in Landsat-8

# get nearest PIDs of PIDs not in L8
get_nearest_PID(l8_locs, all_df[~all_df.PID.isin(l8_locs.PID.unique())].reset_index(), threshold=None)


# - Only three PIDs are not found in landsat-8, however, their closest PIDs have distances more than 2 KM. 
# - We can assign these PIDs with the values of their closest PID or take a summary statistics of PIDs in their respective sites.


# get PIDs in Landsat-8 that are closer with PIDs in train and test sets (without setting a threshold since we see that the distance apart is 3.4KM)
print(get_nearest_PID(l8_locs, all_df, threshold=500).describe().T)

l8_all_nn = get_nearest_PID(l8_locs, all_df, threshold=None)

# lsat bands with their temporal characteristics and rename band names
lsat_bands = merge_band_data(lsat_mean, l8_all_nn).reset_index().rename(columns=band_ids)

lsat_bands.isna().any().sum()

# plot Landsat-8 over the years
# lsat_bands.groupby('mdate').mean(numeric_only=True).plot(subplots=True, figsize=(14,8), layout=(4,4), fontsize=9)
# plt.tight_layout()
# plt.suptitle('Monthly Trend for Landsat Bands and indices', y=1.03, fontsize=12, fontweight='bold')
# plt.show()

# Sentinel 1
s1_bands = merge_band_data(sl1_mean, s1_locs_nn).reset_index()

print('Columns with missing data in Sentinel-1 data: ', s1_bands.isna().any().sum())

# Surface temperature and evapotranspiration
# land surface temperature 
land_surf_temp = merge_band_data(modis_a1_mean, a1_locs_nn).reset_index()

# Evapotranspiration
evap_df = merge_band_data(modis_a2_mean, a2_locs_nn).reset_index()
print('Landsat: {}, Land Surface: {}, Evapotranspiration: {}, Sentinel-1: {}'.format(
    lsat_bands.shape, land_surf_temp.shape, evap_df.shape, s1_bands.shape))

# Saving datasets
print('Saving Datasets')
lsat_bands.to_parquet('data/landsat_band.parquet', index=False)
land_surf_temp.to_parquet('data/land_surf.parquet', index=False)
evap_df.to_parquet('data/evapotrans.parquet', index=False)
s1_bands.to_parquet('data/sl1_bands.parquet', index=False)
print('Datasets saved')


