import time
import os, sys, json
from itertools import compress
import _pickle as cPickle
from multiprocessing import Pool
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import num2date, Dataset
import xarray as xr
import rioxarray
from ..tools import RasterResampling_tif, RasterizeAdminShape_tif, save_hdf
import rasterio
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

def dekad2date_d3(idk):
    '''Returns month and day of dekad number.
    '''
    i, j = int(idk[:2]), int(idk[-1])
    cmat = np.vstack([np.arange(1,13).repeat(3), np.tile(np.arange(1,22,10), 12)]).T
    k = (i-1)*3+(j-1)
    return cmat[k,0], cmat[k,1]

def ExtractAdmNDVI_eviirs(year, fnid_dict):
    # Get filenames
    fn_data = '/home/chc-sandbox/people/dlee/ndvi_eviirs/ndvi.eviirs.{:04d}*.tif'
    infile = sorted(glob.glob(fn_data.format(year)))
    if len(infile) == 0: return

    # Get datetimeindex
    infile_short = [os.path.split(filn)[1] for filn in infile]
    dekad = [filn[17:20] for filn in infile_short]
    date = ['%04d-%02d-%02d' % (year, dekad2date_d3(dkd)[0], dekad2date_d3(dkd)[1]) for dkd in dekad]
    date = pd.to_datetime(date)

    # Remove existing files
    filn_out_crop = '/home/dlee/chafs/data/eodata/ndvi_eviirs/adm.ndvi.eviirs.crop.{:04d}.{:02d}.{:02d}.hdf'
    filn_out_all = '/home/dlee/chafs/data/eodata/ndvi_eviirs/adm.ndvi.eviirs.all.{:04d}.{:02d}.{:02d}.hdf'
    fn_out_crop = [filn_out_crop.format(dt.year, dt.month, dt.day) for dt in date]
    fn_out_all = [filn_out_all.format(dt.year, dt.month, dt.day) for dt in date]
    date_retain_crop = [not os.path.exists(filn) for filn in fn_out_crop]
    date_retain_all = [not os.path.exists(filn) for filn in fn_out_all]
    date_retain = list(np.array(date_retain_crop) | np.array(date_retain_all))
    infile = list(compress(infile, date_retain))
    date = date[date_retain]
    fn_out_crop = list(compress(fn_out_crop, date_retain))
    fn_out_all = list(compress(fn_out_all, date_retain))
    if len(fn_out_crop) == 0: return
    
    # Load unique and reduced RasterIndex (rdx)
    with open('/home/dlee/chafs/data/eodata/rdxu.adm.ndvi.eviirs.pickle', 'rb') as f:
        rdx_unique = cPickle.load(f)    
    with open('/home/dlee/chafs/data/eodata/rdxr.adm.ndvi.eviirs.pickle', 'rb') as f:
        rdx_reduced = cPickle.load(f)
    
    # Resample cropland data
    fn_cropland = '/home/dlee/chafs/data/cropland/Hybrid_10042015v9.img'
    fn_sample = '/home/chc-sandbox/people/dlee/ndvi_eviirs/ndvi.eviirs.2012.021.tif'
    cropland = RasterResampling_tif(fn_cropland, fn_sample).flatten()
    # - Reducing rasters
    cropland = cropland[rdx_unique]
    
    # Load data from GeoTiff files
    tim = date.copy()
    data = np.zeros([len(tim), len(rdx_unique)], np.uint8)
    for i, filn in enumerate(infile):
        with rasterio.open(filn) as src:
            data[i,:] = src.read(1).flatten()[rdx_unique]

    # Aggregate to administrative units
    data_adm_crop = pd.DataFrame(index=tim, columns=fnid_dict.keys())
    data_adm_crop.index.name = 'time'
    data_adm_all = data_adm_crop.copy()
    for fnid in fnid_dict.keys():
        # Grid IDs in the current unit
        subID = rdx_reduced[fnid]
        # Choose Cropland (percent > 0) grids
        # - if no cropland found, use entire spatial average (no cropland)
        if sum(cropland[subID] > 0) > 0:
            subID_crop = subID[cropland[subID] > 0]
        else:
            subID_crop = subID
        subData_crop = data[:, subID_crop]
        subData_all = data[:, subID]
        # Two SubData (cropland and all grids)
        for i, subData in enumerate([subData_crop, subData_all]):
            # Define additional invalid values (201 - 255)
            subData = subData.astype(float)
            subData[subData > 200] = np.nan
            # Ingnore grids with NaN more than 50% of record
            rdx = np.isnan(subData).sum(0) > subData.shape[0]/2
            subData = subData[:, ~rdx]
            # Ignore time with NaN more than 50% of grids
            rdx = np.isnan(subData).sum(1) > subData.shape[1]/2
            subData[rdx,:] = np.nan
            # Aggregated values
            if i == 0:
                data_adm_crop[fnid] = np.nanmean(subData, 1)
            else:
                data_adm_all[fnid] = np.nanmean(subData, 1)
            # If final averages are all NaN, use entire grids values
            if data_adm_crop[fnid].isna().sum() == data_adm_crop[fnid].shape[0]:
                data_adm_crop[fnid] = data_adm_all[fnid]

    # Replace ID to FNID
    data_adm_crop = data_adm_crop.rename(fnid_dict,axis=1)
    data_adm_all = data_adm_all.rename(fnid_dict,axis=1)
    # Rescaled to NDVI
    data_adm_crop = (data_adm_crop - 100)/100
    data_adm_all = (data_adm_all - 100)/100
    # Save files
    for i, t in enumerate(tim):
        save_hdf(fn_out_crop[i], data_adm_crop.loc[t], set_print=True)
        save_hdf(fn_out_all[i], data_adm_all.loc[t], set_print=True)
        
    return


def ndvi_eviirs():
    # Load both admin1 and admin2 boundaries
    adm1 = gpd.read_file('/home/dlee/chafs/data/shapefile/adm1_glob.shp')
    adm2 = gpd.read_file('/home/dlee/chafs/data/shapefile/adm2_glob.shp')
    adm = pd.concat([adm1, adm2], axis=0).reset_index(drop=True)
    # Select African countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = world[world['continent'] == 'Africa']
    africa = africa.dissolve(by='continent').reset_index()
    overlaid = gpd.overlay(adm1, africa, how='intersection')
    africa_countries = overlaid.loc[overlaid.continent == 'Africa', 'ADMIN0'].unique()
    shape_adm = adm[adm.ADMIN0.isin(africa_countries)].reset_index(drop=True)
    # Exchange FNID to numeric ID
    shape_adm.loc[:,'ID'] = np.arange(1,shape_adm.shape[0]+1)
    fnid_dict = shape_adm[['ID','FNID']].set_index('ID').to_dict()['FNID']
    shape_adm1 = shape_adm[shape_adm.FNID.isin(adm1.FNID)]
    shape_adm2 = shape_adm[shape_adm.FNID.isin(adm2.FNID)]

    fn_rdx_unique = '/home/dlee/chafs/data/eodata/rdxu.adm.ndvi.eviirs.pickle'
    fn_rdx_reduced = '/home/dlee/chafs/data/eodata/rdxr.adm.ndvi.eviirs.pickle'
    if False:
        fn_sample = '/home/chc-sandbox/people/dlee/ndvi_emodis/ndvi.eviirs.2012.021.tif'
        # Rasterize administrative boudaries
        admid1 = RasterizeAdminShape_tif(shape_adm1, 'ID', fn_sample).flatten().astype(np.uint16)
        admid2 = RasterizeAdminShape_tif(shape_adm2, 'ID', fn_sample).flatten().astype(np.uint16)
        # - Some units are less than eMODIS gird...
        admid1_unique = np.unique(admid1)
        admid2_unique = np.unique(admid2)
        fnid_all = np.array(list(fnid_dict.keys()))
        fnid_small = fnid_all[~np.isin(fnid_all, [*admid1_unique, *admid2_unique])]
        admid3 = RasterizeAdminShape_tif(shape_adm2[shape_adm2.ID.isin(fnid_small)], 'ID', fn_sample).flatten().astype(np.uint16)
        admid3_unique = np.unique(admid3)
        assert np.all(np.isin(list(fnid_dict.keys()), [*admid1_unique, 
                                                       *admid2_unique, 
                                                       *admid3_unique])) == True
        rdx_reduced1 = {}
        for fnid in admid1_unique[admid1_unique != 0]:
            rdx_reduced1[fnid] = np.flatnonzero(admid1 == fnid).astype(np.uint32)
        rdx_reduced2 = {}
        for fnid in admid2_unique[admid2_unique != 0]:
            rdx_reduced2[fnid] = np.flatnonzero(admid2 == fnid).astype(np.uint32)
        rdx_reduced3 = {}
        for fnid in admid3_unique[admid3_unique != 0]:
            rdx_reduced3[fnid] = np.flatnonzero(admid3 == fnid).astype(np.uint32)
        rdx_reduced = {**rdx_reduced1, **rdx_reduced2, **rdx_reduced3}
        # Reduced rdx
        rdx_unique = np.unique(np.concatenate( [v for k, v in rdx_reduced.items()] ))
        rdx_reduced = {k: np.searchsorted(rdx_unique, v).astype(np.uint32) for k,v in rdx_reduced.items()}
        # Saving files
        with open(fn_rdx_unique, 'wb') as f:
            cPickle.dump(rdx_unique, f, protocol=-1)
            print('%s is saved..' % fn_rdx_unique)

        with open(fn_rdx_reduced, 'wb') as f:
            cPickle.dump(rdx_reduced, f, protocol=-1)
            print('%s is saved..' % fn_rdx_reduced)

    # with open(fn_rdx_unique, 'rb') as f:
    #     rdx_unique = cPickle.load(f)
    # with open(fn_rdx_reduced, 'rb') as f:
    #     rdx_reduced = cPickle.load(f)

    
    # Remove and Re-extract recent data
    files_rm = sorted(glob.glob('/home/dlee/chafs/data/eodata/ndvi_eviirs/adm.ndvi.eviirs.crop.????.*.hdf'))[-5:]
    if len(files_rm) > 0: 
        for file in files_rm: 
            os.remove(file)

    stime = time.time()
    args = ((year, fnid_dict) for year in np.arange(2012, 2023))
    with Pool(processes=8) as pool:
        pool.starmap(ExtractAdmNDVI_eviirs, args)
        pool.close()
        pool.join()
    print(time.time() - stime)

    return