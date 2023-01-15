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
from ..tools import RasterResampling, RasterizeAdminIndex, save_hdf
import rasterio
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

def ExtractAdmSTMP_fldas(year, fnid_dict):
    # Get filenames
    fn_data = '/home/chc-sandbox/people/dlee/fldas/africa/SoilTemp00_10cm_tavg_{:04d}??.nc'
    infile = sorted(glob.glob(fn_data.format(year)))
    if len(infile) == 0: return

    # Get datetimeindex
    date = pd.to_datetime([os.path.split(filn)[1][-9:-3] for filn in infile], format='%Y%m')

    # Remove existing files
    filn_out_crop = '/home/dlee/chafs/data/eodata/stmp_fldas/adm.stmp.fldas.crop.{:04d}.{:02d}.{:02d}.hdf'
    filn_out_all = '/home/dlee/chafs/data/eodata/stmp_fldas/adm.stmp.fldas.all.{:04d}.{:02d}.{:02d}.hdf'
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

    # Resample cropland data
    fn_cropland = '/home/dlee/chafs/data/cropland/Hybrid_10042015v9.img'
    fn_sample ='/home/chc-sandbox/people/dlee/fldas/africa/SoilTemp00_10cm_tavg_198201.nc'
    cropland = RasterResampling(fn_cropland, fn_sample).flatten()

    # Load the reduced raster indicies
    with open('/home/dlee/chafs/data/eodata/rdx.adm.smos.fldas.pickle', 'rb') as f:
        rdx_reduced = cPickle.load(f)

    # Load data from NetCDF files
    data = xr.open_mfdataset(infile, combine='by_coords', parallel=True)
    nlat, nlon, ntim = data.dims['Y'], data.dims['X'], data.dims['time']
    tim = data.time.values
    data = data.variables['SoilTemp00_10cm_tavg'].values
    data = data.reshape([ntim,nlat*nlon])

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

    # Save files
    for i, t in enumerate(tim):
        save_hdf(fn_out_crop[i], data_adm_crop.loc[t], set_print=True)
        save_hdf(fn_out_all[i], data_adm_all.loc[t], set_print=True)
        
    return

def stmp_fldas():
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

    # Rasterize administrative boudaries
    fn_rdx_out = '/home/dlee/chafs/data/eodata/rdx.adm.stmp.fldas.pickle'
    if False:
        fn_sample ='/home/chc-sandbox/people/dlee/fldas/africa/SoilTemp00_10cm_tavg_198201.nc'
        data_sample = xr.open_dataset(fn_sample)
        rdx_reduced = RasterizeAdminIndex(shape_adm, 'ID', data_sample['SoilTemp00_10cm_tavg'])
        assert len([k for k,v in rdx_reduced.items() if len(v) == 0]) == 0
        with open(fn_rdx_out, 'wb') as f:
            cPickle.dump(rdx_reduced, f, protocol=-1)
            print('%s is saved..' % fn_rdx_out)
    # with open(fn_rdx_out, 'rb') as f:
    #     rdx_reduced = cPickle.load(f)


#     # Remove and Re-extract recent data
#     files_rm = sorted(glob.glob('/home/dlee/chafs/data/eodata/stmp_fldas/adm.stmp.fldas.crop.????.??.??.hdf'))
#     if len(files_rm) > 0: 
#         for file in files_rm: 
#             os.remove(file)
            
    # Running
    stime = time.time()
    args = ((year, fnid_dict) for year in np.arange(1982, 2023))
    with Pool(processes=8) as pool:
        pool.starmap(ExtractAdmSTMP_fldas, args)
        pool.close()
        pool.join()
    print(time.time() - stime)

    return