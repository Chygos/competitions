# load libraries

import geopandas as gpd
from tqdm import tqdm
import datetime, os, logging, time
import ee
import pandas as pd
from typing import Literal
import numpy as np
from functools import partial
from shapely.geometry import mapping


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeoAI_Cropland_Mapping')

# read datasets (shapefiles)
fergana = gpd.read_file('../data/Fergana_training_samples.shp')
orenburg = gpd.read_file('../data/Orenburg_training_samples.shp')
test = pd.read_csv('../data/test.csv')

print(fergana.shape, orenburg.shape, test.shape)

# convert test DataFrame to GeoDataFrame
test['geometry'] = gpd.points_from_xy(test['translated_lon'], test['translated_lat'])
test = gpd.GeoDataFrame(test, geometry='geometry', crs='EPSG:4326')



# authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='gmap-project-259019')

# area of interest (convert coordinates to Point coordinates)
fergana_aoi = fergana.geometry.map(lambda geom: ee.Geometry.Point(geom.x, geom.y)).tolist()
orenburg_aoi = orenburg.geometry.map(lambda geom: ee.Geometry.Point(geom.x, geom.y)).tolist()
test_aoi = test.geometry.map(lambda geom: ee.Geometry.Point(geom.x, geom.y)).tolist()

# Helper function
## Functions for cloud masking and monitor task download
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)

def start_and_monitor_task(task, label=''):
    import time
    task.start()
    task_id = task.id
    print(f"Started task {label}...")

    while task.active():
        print(f"{label} is still running...")
        time.sleep(300)  # Check every 5 minutes

    # Final status
    final_status = task.status()['state']
    print(f"Task {label} finished with status: {final_status}")


def download_elevation_models(aoi_points, aoi_info, location_name='Fergana', data='train', save_to_drive=False):
    """
    Get Structural Elevation Models

    :param aoi_points: List of Points for Area of interests
    :param aoi_info: Metadata of area of interest
    :param location_name: Location to extract (Fergana or Orenburg)
    :param data: Type of data (train or test)
    :param save_to_drive: Boolean. Whether to save to Google Drive

    :returns FeatureCollection if save_to_drive is False else nothing is returned
    """
    logger.info(f'Downloading elevation models for {location_name}')

    features = []
    try:
        for i in tqdm(range(len(aoi_points)), desc=f'Downloading {location_name} elevation and slope data'):
            pt = aoi_points[i]
            site_id = str(aoi_info.ID[i])
            crop_id = str(aoi_info.Cropland[i]) if 'Cropland' in aoi_info.columns else 'Unknown'
            location = aoi_info.location[i] if 'location' in aoi_info.columns else location_name
            lon = aoi_info.geometry.x[i]
            lat = aoi_info.geometry.y[i]

            # Get elevation data
            dem = ee.Image('USGS/SRTMGL1_003')
            slope = ee.Terrain.slope(dem)
            
            elevation = dem.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=pt.buffer(100),  # buffer to avoid edge effects
                scale=30,
                maxPixels=1e13
            )
            slope_value = slope.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=pt.buffer(100),  # buffer to avoid edge effects
                scale=30,
                maxPixels=1e13
            )

            combined = elevation.combine(slope_value)
            metadata = ee.Dictionary({
                'ID': site_id,
                'Cropland': crop_id,
                'location': location,
                'x': lon,
                'y': lat
            })
            # Append metadata as properties
            properties = combined.combine(metadata)
            res_feature = ee.Feature(pt, properties)
            features.append(res_feature)
    except Exception as err:
        logger.exception(f'Error processing elevation data for {location_name}\n{err}')
    
    fc = ee.FeatureCollection(features)
    if save_to_drive:
        task = ee.batch.Export.table.toDrive(
                collection=fc,
                description=f'{data}_elevation_{location_name}',
                folder = 'ZindiCropland',
                fileFormat='CSV'
            )
        start_and_monitor_task(task, f'{data}_{str(location_name)}_elevation')
    else:
        return fc


def download_sentinel2(monthly_intervals, aoi_points, aoi_info, location_name='Fergana', 
                       image_dates= ['2021-07-01', '2025-07-01'], save_to_drive=False, 
                       verbose=False):
    """
    Extract Sentinel-2 data

    :param monthly_intervals: List of monthly dates (start and end dates) to extract data for
    :param aoi_info: Metadata of area of interest
    :param location_name: Location to extract (Fergana or Orenburg)
    :param image_dates: Date range to filter from Earth Image dataset
    :param save_to_drive: Boolean. Whether to save to Google Drive
    :param verbose: Boolean. To print download progress

    :returns FeatureCollection if save_to_drive is False else nothing is returned
    """
    # Define bands to extract
    bands_to_extract = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

    # Sentinel-2 (Surface reflectance)
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(image_dates[0], image_dates[1])
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                    #  .map(mask_s2_clouds)
                     .select(bands_to_extract)
                    )    

    features = []
    if verbose:
        iters = tqdm(monthly_intervals, total=len(monthly_intervals), desc=f'Downloading {location_name} S2 data')
    else:
        iters = monthly_intervals
    for start, end in iters:
        # Sentinel-2 (Surface reflectance)
        s2_range = (s2_collection.filterDate(start, end))
        
        image = (
            s2_range
            .map(lambda img: img.divide(1e4)) # normalise image and take mean
            .mean()
        )

        for i in range(len(aoi_points)):
            try:
                pt = aoi_points[i]
                site_id = str(aoi_info.ID[i])
                crop_id = str(aoi_info.Cropland[i]) if 'Cropland' in aoi_info.columns else 'Unknown'
                location = aoi_info.location[i] if 'location' in aoi_info.columns else location_name
                lon = aoi_info.geometry.x[i]
                lat = aoi_info.geometry.y[i]

                # Reduce spectral bands
                reduced = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=pt.buffer(100),  # buffer to avoid edge effects
                    scale=10,
                    maxPixels=1e13
                )
                metadata = {
                    'date': start,
                    'ID': site_id,
                    'Cropland': crop_id,
                    'location': location,
                    'x': lon,
                    'y': lat
                }
                
                # Append metadata as properties
                properties = reduced.combine(metadata)
                feature = ee.Feature(pt, properties)
                features.append(feature)
            except Exception as err:
                logger.exception(f'Error processing image for {start}\n{err}')

    fc = ee.FeatureCollection(features)
    if save_to_drive:
        task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f'train_sentinel2_{location_name}_{start[:4]}',
        folder = 'ZindiCropland',
        fileFormat='CSV'
        )
        start_and_monitor_task(task, f'{location_name}_{start[:4]}')
    else:
        return fc


def download_sentinel1(monthly_intervals, aoi_points, aoi_info, location_name='Fergana', 
                       image_dates= ['2021-07-01', '2025-07-01'], save_to_drive=False,
                       verbose=False):
    """
    Extract Sentinel-1 data

    :param monthly_intervals: List of monthly dates (start and end dates) to extract data for
    :param aoi_info: Metadata of area of interest
    :param location_name: Location to extract (Fergana or Orenburg)
    :param image_dates: Date range to filter from Earth Image dataset
    :param save_to_drive: Boolean. Whether to save to Google Drive
    :param verbose: Boolean. To print download progress

    :returns FeatureCollection if save_to_drive is False else nothing is returned
    """
    # Apply filtering to Sentinel-1 collection
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterDate(image_dates[0], image_dates[1])
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                     .select(['VV', 'VH'])  # Optional: restrict to just VV/VH
                )

    features = []
    if verbose:
        iters = tqdm(monthly_intervals, total=len(monthly_intervals), desc=f'Downloading {location_name} S2 data')
    else:
        iters = monthly_intervals
    for start, end in iters:
        # Sentinel-1 (VV and VH polarisations)
        s1_range = (s1_collection.filterDate(start, end))   

        image = (s1_range.mean())

        for i in range(len(aoi_points)):
            try:
                pt = aoi_points[i]
                site_id = str(aoi_info.ID[i])
                crop_id = str(aoi_info.Cropland[i]) if 'Cropland' in aoi_info.columns else 'Unknown'
                location = aoi_info.location[i] if 'location' in aoi_info.columns else location_name
                lon = aoi_info.geometry.x[i]
                lat = aoi_info.geometry.y[i]

                # Reduce spectral bands
                reduced = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=pt.buffer(100),  # buffer to avoid edge effects
                    scale=10,
                    maxPixels=1e13
                )
                metadata = {
                    'date': start,
                    'ID': site_id,
                    'Cropland': crop_id,
                    'location': location,
                    'x': lon,
                    'y': lat
                }
                
                # Append metadata as properties
                properties = reduced.combine(metadata)
                feature = ee.Feature(pt, properties)
                features.append(feature)
            except Exception as err:
                logger.exception(f'Error processing image for {start}\n{err}')

    fc = ee.FeatureCollection(features)
    if save_to_drive:
        task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f'train_sentinel1_{location_name}_{start[:4]}',
        folder = 'ZindiCropland',
        fileFormat='CSV'
        )
        start_and_monitor_task(task, f'{location_name}_{start[:4]}')
    else:
        return fc


# climate data
lst = ee.ImageCollection('MODIS/061/MOD11A1')
temps = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") # temperature
evapo = ee.ImageCollection("MODIS/061/MOD16A2")

def download_climate_data(img_collection, bands, monthly_intervals, 
                          aoi_points, aoi_info, location_name='Fergana', 
                          feature_name=None, save_to_drive=False, verbose=False):
    """
    Extract Climate data

    :param img_collection: Climate image collection dataset
    :param bands: Band(s) to filter from image collection
    :param monthly_intervals: List of monthly dates (start and end dates) to extract data for
    :param aoi_info: Metadata of area of interest
    :param location_name: Location to extract (Fergana or Orenburg)
    :param feature_name: Name to rename newly extracted climate data
    :param image_dates: Date range to filter from Earth Image dataset
    :param save_to_drive: Boolean. Whether to save to Google Drive
    :param verbose: Boolean. To print download progress

    :returns FeatureCollection if save_to_drive is False else nothing is returned
    """

    features = []
    if verbose:
        iters = tqdm(monthly_intervals, total=len(monthly_intervals), desc=f'Downloading {location_name} {feature_name} climate data')
    else:
        iters = monthly_intervals
    
    for start, end in iters:
        s1_range = (img_collection.select(bands).filterDate(start, end))   

        if 'lst' in feature_name.lower():
            reducer = ee.Reducer.mean()
            image = (s1_range
                     .map(lambda img: img.multiply(0.02).subtract(273.15).rename(feature_name))
                     .mean())
        elif 'evapo' in feature_name.lower():
            reducer = ee.Reducer.sum()
            image = (s1_range
                     .map(lambda img: img.multiply(0.1).rename(['evapotrans', 'p_evapotrans']).copyProperties(img, img.propertyNames()))
                     .sum()
                     )
        elif 'temp' in feature_name.lower():
            reducer = ee.Reducer.sum()
            image = (s1_range
                     .map(lambda img: img.addBands(
                         img.select('tmmn').add(img.select('tmmx')).divide(2).rename('temps').copyProperties(img, img.propertyNames())))
                     .map(lambda img: img.addBands(
                         img.select(['temps', 'soil']).multiply(0.1).copyProperties(img, img.propertyNames()), overwrite=True))
                     .map(lambda img: img.select(['temps', 'ro', 'soil', 'pr']))
                     .sum()
                     )

        for i in range(len(aoi_points)):
            try:
                pt = aoi_points[i]
                site_id = str(aoi_info.ID[i])
                crop_id = str(aoi_info.Cropland[i]) if 'Cropland' in aoi_info.columns else 'Unknown'
                location = aoi_info.location[i] if 'location' in aoi_info.columns else location_name
                lon = aoi_info.geometry.x[i]
                lat = aoi_info.geometry.y[i]

                # Reduce spectral bands
                reduced = image.reduceRegion(reducer=reducer, geometry=pt)
                metadata = {
                    'date': start, 'ID': site_id, 'Cropland': crop_id, 
                    'location': location, 'x': lon, 'y': lat
                }
                
                # Append metadata as properties
                properties = reduced.combine(metadata)
                feature = ee.Feature(pt, properties)
                features.append(feature)
            except Exception as err:
                logger.exception(f'Error processing image for {start}\n{err}')

    fc = ee.FeatureCollection(features)
    if save_to_drive:
        task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f'train_{feature_name}_{location_name}_{start[:4]}',
        folder = 'ZindiCropland',
        fileFormat='CSV'
        )
        start_and_monitor_task(task, f'{location_name}_{start[:4]}')
    else:
        return fc


def create_monthly_intervals(start, end):
    """Generates monthly intervals from start and end dates"""
    intervals = []
    current = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    while current < end:
        next_month = current + datetime.timedelta(days=32)
        next_month = datetime.datetime(next_month.year, next_month.month, 1)
        intervals.append((current.strftime("%Y-%m-%d"), next_month.strftime("%Y-%m-%d")))
        current = next_month
    return intervals

def get_intervals_by_year(monthly_intervals):
    """Groups monthly date intervals by year"""
    from collections import defaultdict
    year_chunks = defaultdict(list)
    for start, end in monthly_intervals:
        year = start[:4]
        year_chunks[year].append((start, end))
    return year_chunks


def preprocess_data(res, data_type:Literal['s1', 's2', 'dem', 'clim']='s1', climate_type=None):
    """
    Processess Feature collection returned from extracted datasets

    :param res: FeatureCollection result
    :param data_type: Type of feature collection data
    :param climate_type: Type of climate data (if data type is clim)

    :returns Pandas DataFrame of Extracted data
    """
    s1_cols = ['VV', 'VH']
    s2_cols = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
    dem_cols = ['elevation', 'slope']
    temp_cols = ['pr', 'ro', 'temps']

    if not isinstance(res, ee.FeatureCollection):
        raise ValueError("Input must be an ee.FeatureCollection")
    if data_type not in ['s1', 's2', 'dem', 'clim', 'water']:
        raise ValueError(f'{data_type} not recognised')
    
    try:
        features = []
        res_info = res.getInfo()['features'] # get features information from the FeatureCollection
        for i in range(len(res_info)):
            res = res_info[i]['properties'] # returns a dictionary
            if data_type == 's1':
                for col in s1_cols:
                    if col not in res.keys():
                        res[col] = np.nan
            elif data_type == 's2':
                for col in s2_cols:
                    if col not in res.keys():
                        res[col] = np.nan
            elif data_type == 'dem':
                for col in dem_cols:
                    if col not in res.keys():
                        res[col] = np.nan
            elif data_type == 'clim':
                if climate_type is None:
                    raise ValueError('climate_type must not be None')
                if climate_type == 'temps':
                    for col in temp_cols:
                        if col not in res.keys():
                            res[col] = np.nan
                elif climate_type == 'evapotrans':
                    for col in ['evapotrans', 'p_evapotrans']:
                        if col not in res.keys():
                            res[climate_type] = np.nan
                elif climate_type == 'LST_celsius':
                    if climate_type not in res.keys():
                        res[climate_type] = np.nan
            features.append(res)
        return pd.DataFrame(features)
    except Exception as err:
        logger.exception(f'Error processing feature {i}\n{err}')



start = '2021-07-01'
end = '2025-06-30'
monthly_intervals = create_monthly_intervals(start, end)
print(len(monthly_intervals))


# Downloading DEM models

# Fergana
fergana_dem = download_elevation_models(fergana_aoi, fergana, 'Fergana', 'train')
fergana_dem = preprocess_data(fergana_dem)

# Orenburg
orenburg_dem = download_elevation_models(orenburg_aoi, orenburg, 'Orenburg', 'train')
orenburg_dem = preprocess_data(orenburg_dem, 'dem')

# Test data
test_dem = download_elevation_models(test_aoi, test, 'test data', 'test')
test_dem = preprocess_data(test_dem, 'dem')

# save datasets
os.makedirs('../data', exist_ok=True)

fergana_dem.to_csv('../data/fergana_dem.csv', index=False)
orenburg_dem.to_csv('../data/orenburg_dem.csv', index=False)
test_dem.to_csv('../data/test_dem.csv', index=False)

# Downloading Sentinel 1
# Orenburg has no Sentinel-1 image from 2022 and above, hence we will download sentinel 1 data four years from 2021 (2018-2021)
def batch_download_sentinel(monthly_intervals):
    """
    Downloads Sentinel data in batches. 
    This is to solve data limit in Earth Engine
    """

    yearly_intervals = get_intervals_by_year(monthly_intervals)
    for year, intervals in yearly_intervals.items():
        yield year, intervals

# create month intervals for orenburg for sentinel1
orenburg_intervals = create_monthly_intervals('2018-01-01', '2022-01-01')
print(len(orenburg_intervals))

orenburg_s1 = []
for year, intervals in batch_download_sentinel(orenburg_intervals):
    logger.info(f"Downloading Sentinel-1 for ({year})")
    for month_interval in intervals:
        # get image year
        image_year = [month_interval[0], month_interval[1]]
        res = download_sentinel1([month_interval], orenburg_aoi, orenburg, 'Orenburg', image_year)
        orenburg_s1.append(res)


fergana_s1 = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Sentinel-1 for ({year})")
    for month_interval in intervals:
        # get image year
        image_year = [month_interval[0], month_interval[1]]
        res = download_sentinel1([month_interval], fergana_aoi, fergana, 'Fergana', image_year)
        fergana_s1.append(res)



# Downloading Sentinel-2 images
orenburg_s2 = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Sentinel-2 for ({year})")
    for month_interval in intervals:
        # get image year
        image_year = [month_interval[0], month_interval[1]]
        res = download_sentinel2([month_interval], orenburg_aoi, orenburg, 'Orenburg', image_year)
        orenburg_s2.append(res)

fergana_s2 = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Sentinel-2 for ({year})")
    for month_interval in intervals:
        # get image year
        image_year = [month_interval[0], month_interval[1]]
        res = download_sentinel2([month_interval], fergana_aoi, fergana, 'Fergana', image_year)
        fergana_s2.append(res)


# Downloading Climate Data (Evapotranspiration and Land Surface temperature)

# Land surface data
fergana_lst = [] 
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Land surface temperature for ({year})")
    for month_interval in intervals:
        res = download_climate_data(lst, 'LST_Day_1km', [month_interval], fergana_aoi, fergana, 'Fergana', 'LST_celsius')
        fergana_lst.append(res)

# Evaporation
fergana_evapo = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Evapotranspiration for ({year})")
    for month_interval in intervals:
        res = download_climate_data(evapo, ['ET', 'PET'], [month_interval], fergana_aoi, fergana, 'Fergana', 'evapotrans')
        fergana_evapo.append(res)

# Temperaturem runoff, precipitation and soil temperature
fergana_temps = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Climate data for ({year})")
    for month_interval in intervals:
        res = download_climate_data(temps, ['ro', 'pr', 'soil', 'tmmn', 'tmmx'], [month_interval], fergana_aoi, fergana, 'Fergana', 'temps')
        fergana_temps.append(res)


# Orenburg
orenburg_lst = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Land surface temperature for ({year})")
    for month_interval in intervals:
        res = download_climate_data(lst, 'LST_Day_1km', [month_interval], orenburg_aoi, orenburg, 'Orenburg', 'LST_celsius')
        orenburg_lst.append(res)


orenburg_evapo = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Evapotranspiration for ({year})")
    for month_interval in intervals:
        res = download_climate_data(evapo, ['ET', 'PET'], [month_interval], orenburg_aoi, orenburg, 'Orenburg', 'evapotrans')
        orenburg_evapo.append(res)


orenburg_temps = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Climate for ({year})")
    for month_interval in intervals:
        res = download_climate_data(temps, ['ro', 'pr', 'soil', 'tmmn', 'tmmx'], [month_interval], orenburg_aoi, orenburg, 'Orenburg', 'temps')
        orenburg_temps.append(res)

# Test data
test_evapo = []

for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Evapotranspiration for ({year})")
    for month_interval in intervals:
        res = download_climate_data(evapo, ['ET', 'PET'], [month_interval], test_aoi, test, 'test', 'evapotrans')
        test_evapo.append(res)


test_temps = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Climate for ({year})")
    for month_interval in intervals:
        res = download_climate_data(temps, ['ro', 'pr', 'soil', 'tmmn', 'tmmx'], 
                                    [month_interval], test_aoi, test, 'test', 'temps')
        test_temps.append(res)

test_lst = []
for year, intervals in batch_download_sentinel(monthly_intervals):
    logger.info(f"Downloading Land surface temperature for ({year})")
    for month_interval in intervals:
        res = download_climate_data(lst, 'LST_Day_1km', [month_interval], test_aoi, test, 'test', 'LST_celsius')
        test_lst.append(res)


def extract_sentinel_data(sentinel_data:list, **kwargs):
    """Extracts sentinel data by processing in year batches"""
    results = pd.DataFrame()
    if not isinstance(sentinel_data, (list, tuple)):
        raise TypeError('Input data must be a list or tuple')
    data_type = kwargs.get('data_type', 's1')
    climate_type = kwargs.get('climate_type', None)

    for i in tqdm(range(len(sentinel_data))):
        res = preprocess_data(sentinel_data[i], data_type=data_type, climate_type=climate_type)
        time.sleep(2) # sleep for 2 seconds
        results = pd.concat([results, res], ignore_index=True)
    return results


# Aggregating sentinel 1 and 2
oren_s1 = extract_sentinel_data(orenburg_s1, data_type='s1')
oren_s2 = extract_sentinel_data(orenburg_s2, data_type='s2')

ferg_s1 = extract_sentinel_data(fergana_s1, data_type='s1')
ferg_s2 = extract_sentinel_data(fergana_s2, data_type='s2')

# Save data
oren_s1.to_csv('../data/orenburg_s1.csv', index=False)
oren_s2.to_csv('../data/orenburg_s2.csv', index=False)
ferg_s1.to_csv('../data/fergana_s1.csv', index=False)
ferg_s2.to_csv('../data/fergana_s2.csv', index=False)



# Aggregating climate data
ferg_evapo = extract_sentinel_data(fergana_evapo, data_type='clim', climate_type='evapotrans')
ferg_lst = extract_sentinel_data(fergana_lst, data_type='clim', climate_type='LST_celsius')
ferg_temps = extract_sentinel_data(fergana_temps, data_type='clim', climate_type='temps')


oren_evapo = extract_sentinel_data(orenburg_evapo, data_type='clim', climate_type='evapotrans')
oren_lst = extract_sentinel_data(orenburg_lst, data_type='clim', climate_type='LST_celsius')
oren_temps = extract_sentinel_data(orenburg_temps, data_type='clim', climate_type='temps')

# test
test_evapo_agg = extract_sentinel_data(test_evapo, data_type='clim', climate_type='evapotrans')
test_temps_agg = extract_sentinel_data(test_temps, data_type='clim', climate_type='temps')
test_lst_agg = extract_sentinel_data(test_lst, data_type='clim', climate_type='LST_celsius')

# Merge climate datasets
# get common columns in train
common_cols = test_temps_agg.columns.union(test_evapo_agg.columns).intersection(test_lst_agg.columns).tolist()
common_cols


def merge_climate(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = pd.merge(res, args[i], on=common_cols)
    return res

test_climate = merge_climate(test_evapo_agg, test_lst_agg, test_temps_agg)
oren_climate = merge_climate(oren_evapo, oren_lst, oren_temps)
ferg_climate = merge_climate(ferg_evapo, ferg_lst, ferg_temps)

print(test_climate.shape, oren_climate.shape, ferg_climate.shape)

# save
test_climate.to_parquet('../data/test_climate.parquet', index=False)
ferg_climate.to_parquet('../data/fergana_climate.parquet', index=False)
oren_climate.to_parquet('../data/orenburg_climate.parquet', index=False)

print("Missing_values", oren_s1.isna().any().sum(), ferg_s1.isna().any().sum())

print("Missing_values", oren_s2.isna().any().sum(), ferg_s2.isna().any().sum())

