import os, shutil
from os import path
import subprocess
import shlex
from re import search
import datetime
import glob
import urllib
from itertools import product, compress
from multiprocessing import Pool
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import xarray as xr
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import rioxarray
from ..tools import DownloadFromURL, LinkFromURL, LinkFromFTP
import warnings
from datetime import date
from dateutil.relativedelta import relativedelta
today = date.today()

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        line = process.stdout.readline()
        if not line: break
        print(line.rstrip())
    return

def stream_eodata():
    # NDVI-eMODIS =====================================
    # a) Mirror 2020-current GeoTiff files from USGS http server to "ndvi_emodis_mirror"
    command = 'wget -m -nd --reject="data_202???01_202???28.tiff, data_202???01_202???29.tiff, data_202???01_202???30.tiff, data_202???01_202???31.tiff" -A "data_202?*.tiff" https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/viewer_G5/emodisndvic6v2_africa_dekad_data -P /home/chc-sandbox/people/dlee/ndvi_emodis_mirror/ -q --show-progress'
    print('='*50)
    print('Mirroring NDVI-eMODIS')
    print(command)
    run_command(command)

    # b) Generate symbolic links from "ndvi_emodis_mirror" to "ndvi_emodis"
    def date2dekad3(dt):
        cmat = np.vstack([np.arange(1,13).repeat(3), 
                          np.tile(np.arange(1,22,10), 12), 
                          np.tile(np.array([1,2,3]), 12),
                          np.arange(1,37)]).T
        year, mon, day = dt.year, dt.month, dt.day
        idx = (cmat[:,0] == mon) & (cmat[:,1] == day)
        dkd = cmat[idx,2][0]
        return dkd
    print('-'*50)

    # Load filenames
    src_dir = '/home/chc-sandbox/people/dlee/ndvi_emodis_mirror'
    src_filns = sorted(glob.glob(os.path.join(src_dir, '*.tiff')))
    src_short = [os.path.split(filn)[1] for filn in src_filns]
    # Create symbolic links
    date_cal = pd.to_datetime([short[5:13] for short in src_short])
    date_dkd = [date2dekad3(t) for t in date_cal]
    loc_filns = ['/home/chc-sandbox/people/dlee/ndvi_emodis/ndvi.emodis.%04d.%02d%1d.tif' % (year, month, dkd) for year, month, dkd in zip(date_cal.year, date_cal.month, date_dkd)]
    for filn_in, filn_out in zip(src_filns, loc_filns):
        if os.path.exists(filn_out): os.remove(filn_out)
        os.symlink(filn_in, filn_out)
        print('%s is saved.' % filn_out)
    # =================================================



    # 2. NOAA-CPC Global Daily Temperature ============
    # a) Mirror annual NetCDF files
    command = 'wget -m -nd -A "tmax.????.nc" ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp/ -P /home/chc-sandbox/people/dlee/temp_noaa-cpc/ -q --show-progress'
    print('='*50)
    print('MAX TEMP-NOAA-CPC')
    print(command)
    run_command(command)
    print('-'*50)
    command = 'wget -m -nd -A "tmin.????.nc" ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp/ -P /home/chc-sandbox/people/dlee/temp_noaa-cpc/ -q --show-progress'
    print('MIN TEMP-NOAA-CPC')
    print(command)
    run_command(command)
    print('-'*50)

    # b) Extract all daily NetCDF files of each annual file.
    # - The recent 45 days of the latest year is forced to be re-extracted.

    # Date setting
    file_format_max = '/home/chc-sandbox/people/dlee/temp_noaa-cpc/tmax.noaa-cpc.{:04d}.{:02d}.{:02d}.nc'
    file_format_min = '/home/chc-sandbox/people/dlee/temp_noaa-cpc/tmin.noaa-cpc.{:04d}.{:02d}.{:02d}.nc'
    file_format_gdd = '/home/chc-sandbox/people/dlee/gdd_noaa-cpc/gdd.noaa-cpc.{:04d}.{:02d}.{:02d}.nc'
    url = 'ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp/'
    tmp = '/home/dlee/data/tmp'
    # Generate all dates in the period
    datePeriod = pd.date_range('1979-01-01', date.today())
    # Remove missing dates noted by NOAA-CPC
    date_remove = ['19810101','19810102','19830426','19830427','19830428','19830429','19830430','19840107',
                   '19850107','19850108','19850206','19850112','19850113','19850116','19850119','19850101',
                   '19850810','19850717','19860104','19860328','19860920','19861114','19861121','19920731']
    date_remove = pd.to_datetime(date_remove)
    datePeriod = datePeriod[~datePeriod.isin(date_remove)]

    # Maximum Temperature
    for year in datePeriod.year.unique():
        dateYear = datePeriod[datePeriod.year == year]
        # Generate file names in the directory
        file_list = [file_format_max.format(dateDay.year, dateDay.month, dateDay.day) for dateDay in dateYear]
        # Remove exsiting files in the directory
        index_none = [not os.path.exists(file) for file in file_list]
        file_list = list(compress(file_list, index_none))
        dateYear = dateYear[index_none]
        if len(dateYear) == 0: continue
        # Get dates in the NetCDF file
        fn_dir = '/home/chc-sandbox/people/dlee/temp_noaa-cpc/tmax.{:04d}.nc'.format(year)
        data = xr.open_dataset(fn_dir)
        date_nc = pd.to_datetime(np.array(data.time))
        # Generate a list of missing and available files
        if year < datePeriod.year.unique()[-1]:
            dateYear = dateYear[dateYear.isin(date_nc)]
        else:
            dateYear = date_nc[-45:]
        # Extract daily data
        for dateDay in dateYear:
            fn_out = file_format_max.format(dateDay.year, dateDay.month, dateDay.day)
            data_date = data.sel(time=slice(dateDay, dateDay))
            # Move the east half to the west (0 - 365 to -180 - 180)
            data_date = data_date.assign_coords(lon=(((data_date.lon + 180) % 360) - 180)).sortby('lon')
            data_date.to_netcdf(fn_out)
            print('%s is saved..' % fn_out)

    # Minimum Temperature
    for year in datePeriod.year.unique():
        dateYear = datePeriod[datePeriod.year == year]
        # Generate file names in the directory
        file_list = [file_format_min.format(dateDay.year, dateDay.month, dateDay.day) for dateDay in dateYear]
        # Remove exsiting files in the directory
        index_none = [not os.path.exists(file) for file in file_list]
        file_list = list(compress(file_list, index_none))
        dateYear = dateYear[index_none]
        if len(dateYear) == 0: continue
        # Get dates in the NetCDF file
        fn_dir = '/home/chc-sandbox/people/dlee/temp_noaa-cpc/tmin.{:04d}.nc'.format(year)
        data = xr.open_dataset(fn_dir)
        date_nc = pd.to_datetime(np.array(data.time))
        # Generate a list of missing and available files
        if year < datePeriod.year.unique()[-1]:
            dateYear = dateYear[dateYear.isin(date_nc)]
        else:
            dateYear = date_nc[-45:]
        # Extract daily data
        for dateDay in dateYear:
            fn_out = file_format_min.format(dateDay.year, dateDay.month, dateDay.day)
            data_date = data.sel(time=slice(dateDay, dateDay))
            # Move the east half to the west (0 - 365 to -180 - 180)
            data_date = data_date.assign_coords(lon=(((data_date.lon + 180) % 360) - 180)).sortby('lon')
            data_date.to_netcdf(fn_out)
            print('%s is saved..' % fn_out)

    # Growing Degree Days (GDD)
    for year in datePeriod.year.unique():
        dateYear = datePeriod[datePeriod.year == year]
        # Generate file names in the directory
        file_list = [file_format_gdd.format(dateDay.year, dateDay.month, dateDay.day) for dateDay in dateYear]
        # Remove exsiting files in the directory
        index_none = [not os.path.exists(file) for file in file_list]
        file_list = list(compress(file_list, index_none))
        if year < datePeriod.year.unique()[-1]:
            dateYear = dateYear[index_none]
        else:
            dateYear = dateYear[-45:]
        if len(dateYear) == 0: continue
        # Calculate and Save daily GDD
        for dateDay in dateYear:
            fn_out = file_format_gdd.format(dateDay.year, dateDay.month, dateDay.day)
            # Load TMAX and TMIN
            fn_tmax = file_format_max.format(dateDay.year, dateDay.month, dateDay.day)
            if not os.path.exists(fn_tmax): continue
            da_tmax = xr.open_dataset(file_format_max.format(dateDay.year, dateDay.month, dateDay.day))
            da_tmin = xr.open_dataset(file_format_min.format(dateDay.year, dateDay.month, dateDay.day))
            tmax = da_tmax.tmax.values
            tmin = da_tmin.tmin.values
            lat, lon, time = da_tmax.coords['lat'], da_tmax.coords['lon'], da_tmax.coords['time']
            # Calculate Growing Degree Days (GDD)
            tmin[tmin < 10] = 10
            tmax[tmax < 10] = 10
            tmax[tmax > 30] = 30
            tmean = (tmax + tmin)/2
            gdd = tmean - 10
            gdd[gdd < 0] = 0
            # Save as Xarray Dataset
            ds_gdd = xr.Dataset({'gdd': (['time','lat','lon'], gdd)}, coords={'lat':lat, 'lon':lon, 'time':time})
            ds_gdd.to_netcdf(fn_out)
            print('%s is saved..' % fn_out)


    # ETOS-NOAA =======================================
    # a) Mirror NetCDF files from NOAA FTP server to "ndvi_emodis"
    print('='*50)
    print('MAX TEMP-NOAA-CPC')
    command = 'wget -m -nd -A "ETos_fine_*.nc" -R ".listing" ftp://ftp.cdc.noaa.gov/Projects/RefET/global/Gen-0/fine_resolution/data_v2/ -P /home/chc-sandbox/people/dlee/etos_noaa/ -q --show-progress'
    print(command)
    run_command(command)
    print('-'*50)

    # b) Ignore preQAQC files. These files will be automatically mirrored.
    filn_src = sorted(glob.glob(os.path.join('/home/chc-sandbox/people/dlee/etos_noaa/', '*preQAQC.nc')))
    filn_dst = [t.replace('_preQAQC', '') for t in filn_src]
    for fn_src, fn_dst in zip(filn_src, filn_dst):
        if os.path.exists(fn_dst): continue
        shutil.copyfile(fn_src, fn_dst)
        print('%s is saved.' % fn_dst)
    # =================================================
    
    return
