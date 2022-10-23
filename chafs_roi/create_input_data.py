import os, sys, glob, json, time, warnings
from distutils import util
from itertools import compress, product
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from .tools import split, save_hdf
import scipy
import scipy.signal
from scipy.stats import zscore
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def dekad2date(dkd, string=False):
    '''Returns month and day of dekad number.
    '''
    dkd = int(dkd)
    assert 0 < dkd <= 36
    cmat = np.vstack([np.arange(1,13).repeat(3), np.tile(np.arange(1,22,10), 12)]).T
    if string:
        return '%02d-%02d' % (cmat[dkd-1,0], cmat[dkd-1,1])
    else:
        return cmat[dkd-1,0], cmat[dkd-1,1]
    
    
def MergeUpdateData(name):
    path_chafs = '/home/dlee/chafs'
    if name == 'etos':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/etos_noaa/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/etos_noaa/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.etos.noaa.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.etos.noaa.crop.hdf'
    elif name == 'tmax':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/tmax_noaa-cpc/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/tmax_noaa-cpc/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.tmax.noaa-cpc.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.tmax.noaa-cpc.crop.hdf'
    elif name == 'tmin':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/tmin_noaa-cpc/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/tmin_noaa-cpc/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.tmin.noaa-cpc.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.tmin.noaa-cpc.crop.hdf'  
    elif name == 'gdd':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/gdd_noaa-cpc/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/gdd_noaa-cpc/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.gdd.noaa-cpc.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.gdd.noaa-cpc.crop.hdf'
    elif name == 'prcp_chrips':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirps-v2/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirps-v2/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.prcp.chirps-v2.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.prcp.chirps-v2.crop.hdf'
    elif name == 'prcp_chrips_prelim':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirps-v2p/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirps-v2p/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.prcp.chirps-v2p.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.prcp.chirps-v2p.crop.hdf'
    elif name == 'prcp_chrip':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirp-v2/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/prcp_chirp-v2/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.prcp.chirp-v2.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.prcp.chirp-v2.crop.hdf'
    elif name == 'ndvi_emodis':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/ndvi_emodis/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/ndvi_emodis/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.ndvi.emodis.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.ndvi.emodis.crop.hdf'
    elif name == 'ndvi_avhrr':
        filns_agg_all = sorted(glob.glob(path_chafs + '/data/eodata/ndvi_avhrr-v5/*.all.*.hdf'))
        filns_agg_crop = sorted(glob.glob(path_chafs + '/data/eodata/ndvi_avhrr-v5/*.crop.*.hdf'))
        filn_all = path_chafs + '/data/eodata/adm.ndvi.avhrr-v5.all.hdf'
        filn_crop = path_chafs + '/data/eodata/adm.ndvi.avhrr-v5.crop.hdf'
    else:
        raise ValueError("Wrong variable name.")

    # Get filenames and dates of the aggregated data
    date_agg_all = pd.to_datetime([filn[-14:-4] for filn in filns_agg_all])
    date_agg_crop = pd.to_datetime([filn[-14:-4] for filn in filns_agg_crop])
    assert np.all(date_agg_all == date_agg_crop)
    date_agg = date_agg_all.copy()

    # Create merged files if not exist
    df_adm = pd.concat([pd.read_hdf(filn) for filn in filns_agg_all], axis=1)
    df_adm = df_adm[sorted(df_adm.columns)].T
    save_hdf(filn_all, df_adm)
    df_adm = pd.concat([pd.read_hdf(filn) for filn in filns_agg_crop], axis=1)
    df_adm = df_adm[sorted(df_adm.columns)].T
    save_hdf(filn_crop, df_adm)

    # Create merged files if not exist
    if not os.path.exists(filn_all):
        df_adm = pd.concat([pd.read_hdf(filn) for filn in filns_agg_all], axis=1)
        df_adm = df_adm[sorted(df_adm.columns)].T
        save_hdf(filn_all, df_adm)
    if not os.path.exists(filn_crop):
        df_adm = pd.concat([pd.read_hdf(filn) for filn in filns_agg_crop], axis=1)
        df_adm = df_adm[sorted(df_adm.columns)].T
        save_hdf(filn_crop, df_adm)

    # Retreive dates of merged files
    df_all = pd.read_hdf(filn_all, mode='r').copy()
    df_crop = pd.read_hdf(filn_crop, mode='r').copy()
    date_all = df_all.index
    date_crop = df_crop.index
    # Find new files
    filn_new_all = list(compress(filns_agg_all, date_agg > date_all.max()))
    filn_new_crop = list(compress(filns_agg_crop, date_agg > date_crop.max()))

    # Append new files
    for (df_old, filn_new, filn_out) in zip([df_all, df_crop], [filn_new_all, filn_new_crop], [filn_all, filn_crop]):
        if len(filn_new) == 0: continue
        df_new = pd.concat([pd.read_hdf(filn) for filn in filn_new], axis=1)
        df_new = df_new[sorted(df_new.columns)].T
        merged = pd.concat([df_old, df_new], axis=0).sort_index()
        merged.to_hdf(filn_out, key='df', complib='blosc:zstd', complevel=9)
        print('%s is updated.' % filn_out)
    return

def ResampleDay2Dekad(df, method):
    d = df.index.day - np.clip((df.index.day-1) // 10, 0, 2)*10 - 1
    date = df.index.values - np.array(d, dtype="timedelta64[D]")
    return df.groupby(date).agg(method)

def FillMissDateMean(df):
    df = df.reindex(pd.date_range(df.index[0], df.index[-1]))
    df = df.groupby(df.index.strftime('%m-%d')).transform(lambda x: x.fillna(x.mean()))
    return df

def LongTermDaily(df, years, method='mean'):
    df_long = df[df.index.year.isin(years)]
    df_long = df_long.reindex(df.index)
    if type(method) == float:
        df_long = df_long.groupby(df_long.index.strftime('%m-%d')).transform(lambda x: x.quantile(method))
    elif method == 'mean':
        df_long = df_long.groupby(df_long.index.strftime('%m-%d')).transform('mean')
    else:
        raise NameError('Unsupported method')
    return df_long

    
def create_input_data():
    
    path_chafs = '/home/dlee/chafs'

    # Load crop yield data ------------------- #
    # CPS (Country-Product-Season)
    cps = [
        ['Somalia','Sorghum','Gu'],
        ['Somalia','Sorghum','Deyr'],
        ['Somalia','Maize','Gu'],
        ['Somalia','Maize','Deyr'],
        ['Malawi','Maize','Main'],
        ['Kenya','Maize','Long'],
        ['Kenya','Maize','Short'],
        ['Burkina Faso','Maize','Main'],
        ['Burkina Faso','Sorghum','Main']
    ]
    # Load crop area, production, yield data
    df = pd.read_csv('https://raw.githubusercontent.com/chc-ucsb/gscd/main/public/gscd_data_stable.csv', index_col=0)
    # Reduce data
    container = []
    for country_name, product_name, season_name in cps:
        sub = df[
                (df['country'] == country_name) &
                (df['product'] == product_name) &
                (df['season_name'] == season_name)
        ]
        container.append(sub)
    df = pd.concat(container, axis=0)
    # Pivot table format
    table = df.pivot_table(
        index='year',          
        columns=['fnid','country','name','product','season_name','harvest_end','indicator'],         
        values='value'
    )
    # Record years
    record = table.loc[:,pd.IndexSlice[:,:,:,:,:,:,'yield']].notna().sum().reset_index().rename(columns={0:'record'})

    # FNID information
    fnids_info = table.columns.droplevel(-1).to_frame().drop_duplicates().reset_index(drop=True)
    fnids_info = fnids_info.merge(
        record[['fnid','product','season_name','record']], 
        left_on=['fnid','product','season_name'], 
        right_on=['fnid','product','season_name']
    )
    fnids_info.insert(2, 'country_iso', fnids_info['country'])
    fnids_info['country_iso'].replace({
        'Kenya': 'KE',
        'Somalia': 'SO',
        'Malawi': 'MW',
        'Burkina Faso': 'BF'
    },inplace=True)
    # Load FEWSNET admin boundaries
    shape = gpd.read_file('https://raw.githubusercontent.com/chc-ucsb/gscd/main/public/gscd_shape_stable.json').drop(columns='id')
    fnids_eo = shape.FNID.reset_index(drop=True)
    # ---------------------------------------- #

    # Export crop data ----------------------- #
    # Remove existing CROP data files
    for file in glob.glob('./data_in/*_crop_*'): 
        os.remove(file)

    # Export crop production data
    for fnid, country_name, product_name, season_name in fnids_info[['fnid','country','product','season_name']].values.tolist():
        dfCrop = table.loc[:,pd.IndexSlice[fnid,country_name,:,product_name,season_name]].astype(np.float32)
        save_hdf('./data_in/{:s}_crop_{:s}_{:s}.hdf'.format(fnid, product_name, season_name), dfCrop, set_print=False)

    # Export FNID infomation
    save_hdf('./data_in/fnids_info.hdf', fnids_info, set_print=False)
    print('Exporting CROP data is completed.')
    # ---------------------------------------- #


    # Merge & Update the latest admin-level aggregated data
    '''
    Previously, we "updated" the aggregated file with newly added daily files, but we downgraded it to aggregate all files everytime.
    '''
    # Merge & Update the latest admin-level aggregated data 
    for name in ['etos', 'tmax', 'tmin', 'gdd', 'prcp_chrips', 'prcp_chrips_prelim', 'ndvi_emodis']:
        MergeUpdateData(name)
    # ---------------------------------------- #


    # Define EO data variables --------------- #
    # Load completely updated admin-level EO data
    cropland = 'crop'
    etos = pd.read_hdf(path_chafs + '/data/eodata/adm.etos.noaa.%s.hdf' % cropland)[fnids_eo]
    tmax = pd.read_hdf(path_chafs + '/data/eodata/adm.tmax.noaa-cpc.%s.hdf' % cropland)[fnids_eo]
    tmin = pd.read_hdf(path_chafs + '/data/eodata/adm.tmin.noaa-cpc.%s.hdf' % cropland)[fnids_eo]
    gdd = pd.read_hdf(path_chafs + '/data/eodata/adm.gdd.noaa-cpc.%s.hdf' % cropland)[fnids_eo]

    # Merge the latest CHIRPS and CHIRPS-Prelim data
    prcp_chirps = pd.read_hdf(path_chafs + '/data/eodata/adm.prcp.chirps-v2.%s.hdf' % cropland)[fnids_eo]
    prcp_chirps_p = pd.read_hdf(path_chafs + '/data/eodata/adm.prcp.chirps-v2p.%s.hdf' % cropland)[fnids_eo]
    prcp_chirps_p = prcp_chirps_p.loc[prcp_chirps_p.index[prcp_chirps_p.index > prcp_chirps.index.max()]]
    prcp = pd.concat([prcp_chirps, prcp_chirps_p], axis=0)
    # Link the rescaled AVHRR-v5 (1981-06 - 2002-07) to eMODIS (2002-07 - present)
    ndvi_emodis = pd.read_hdf(path_chafs + '/data/eodata/adm.ndvi.emodis.%s.hdf' % cropland)[fnids_eo]
    ndvi_avhrr_rcon = pd.read_hdf(path_chafs + '/data/eodata/adm.ndvi.avhrr-v5_rcon.%s.hdf' % cropland)[fnids_eo]
    ndvi_raw = pd.concat([ndvi_avhrr_rcon, ndvi_emodis], axis=0)

    # Define start and end date of dekadal data
    # - prcp starts at 1981-01-01
    # - etos starts at 1981-01-01
    # - ndvi starts at 1981-06-21
    # - tmax starts at 1979-01-01
    prcp_start, prcp_end = prcp.index[0], prcp.index[-1]
    etos_start, etos_end = etos.index[0], etos.index[-1]
    tmax_start, tmax_end = tmax.index[0], tmax.index[-1]
    ndvi_start, ndvi_end = ndvi_raw.index[0], ndvi_raw.index[-1]
    if ndvi_end.day <= 10:
        ext = 10
    elif ndvi_end.day <= 20:
        ext = 20
    else:
        ext = ndvi_end.days_in_month
    ndvi_end = pd.Timestamp(year=ndvi_end.year, month=ndvi_end.month, day=ext)

    # Extension of EO data time period
    dekad_start = ndvi_start
    dekad_end = max([prcp_end, etos_end, tmax_end, ndvi_end])
    if dekad_end.day <= 10:
        ext = 10
    elif dekad_end.day <=20:
        ext = 20
    else:
        ext = dekad_end.days_in_month
    dekad_end = pd.Timestamp(year=dekad_end.year, month=dekad_end.month, day=ext)

    print('='*40)
    print('Current EO data temporal coverage')
    print('-'*40)
    print('PRCP:\t%s to %s' % (prcp_start.strftime('%Y-%m-%d'), prcp_end.strftime('%Y-%m-%d')))
    print('ETOS:\t%s to %s' % (etos_start.strftime('%Y-%m-%d'), etos_end.strftime('%Y-%m-%d')))
    print('TEMP:\t%s to %s' % (tmax_start.strftime('%Y-%m-%d'), tmax_end.strftime('%Y-%m-%d')))
    print('NDVI:\t%s to %s' % (ndvi_start.strftime('%Y-%m-%d'), ndvi_end.strftime('%Y-%m-%d')))
    print('='*40)
    print('Data extension')
    print('-'*40)
    print('Start:\t%s' % (dekad_start.strftime('%Y-%m-%d')))
    print('End:\t%s' % (dekad_end.strftime('%Y-%m-%d')))
    print('='*40)

    # EO data extension
    prcp = prcp.reindex(index=pd.date_range(dekad_start, dekad_end))
    etos = etos.reindex(index=pd.date_range(dekad_start, dekad_end))
    tmax = tmax.reindex(index=pd.date_range(dekad_start, dekad_end))
    tmin = tmin.reindex(index=pd.date_range(dekad_start, dekad_end))
    gdd = gdd.reindex(index=pd.date_range(dekad_start, dekad_end))
    ndvi_raw = ndvi_raw.reindex(index=pd.date_range(dekad_start, dekad_end))

    # Fill missing values with long-term daily means
    prcp = FillMissDateMean(prcp)
    etos = FillMissDateMean(etos)
    tmax = FillMissDateMean(tmax)
    tmin = FillMissDateMean(tmin)
    gdd = FillMissDateMean(gdd)

    # Fill missing or negative NDVI with long-term daily means
    ndvi_raw[ndvi_raw < 0] = np.nan
    ndvi_raw = ndvi_raw.reindex(index=pd.date_range(dekad_start, dekad_end))
    ndvi_raw = FillMissDateMean(ndvi_raw)
    ndvi_raw = ResampleDay2Dekad(ndvi_raw, 'max')

    # Save compelte EO data
    save_hdf(path_chafs + '/data/eodata/adm.etos.noaa.%s.extended.hdf' % cropland, etos)
    save_hdf(path_chafs + '/data/eodata/adm.tmax.noaa-cpc.%s.extended.hdf' % cropland, tmax)
    save_hdf(path_chafs + '/data/eodata/adm.tmin.noaa-cpc.%s.extended.hdf' % cropland, tmin)
    save_hdf(path_chafs + '/data/eodata/adm.gdd.noaa-cpc.%s.extended.hdf' % cropland, gdd)
    save_hdf(path_chafs + '/data/eodata/adm.prcp.merged.%s.extended.hdf' % cropland, prcp)
    save_hdf(path_chafs + '/data/eodata/adm.ndvi.merged.%s.extended.hdf' % cropland, ndvi_raw)
    # Select FNIDs
    prcp = prcp[fnids_eo]
    etos = etos[fnids_eo]
    tmax = tmax[fnids_eo]
    tmin = tmin[fnids_eo]
    gdd = gdd[fnids_eo]
    ndvi_raw = ndvi_raw[fnids_eo]

    # Create specific variables -------------- #
    year_long = np.arange(1986,2015)
    # Variables related to Precipitation (PRCP)
    # - Accumulated precipitation
    pacu = ResampleDay2Dekad(prcp, np.nansum)
    # - Number of dry days
    pdry = ResampleDay2Dekad(prcp == 0, np.nansum)
    # - Number of days above 95% percentile of daily precipitation
    pa95 = prcp - LongTermDaily(prcp, year_long, 0.95)
    pa95 = ResampleDay2Dekad(pa95 > 0, np.nansum)
    # - Number of days above 90% percentile of daily precipitation
    pa90 = prcp - LongTermDaily(prcp, year_long, 0.90)
    pa90 = ResampleDay2Dekad(pa90 > 0, np.nansum)
    # - Number of days above 80% percentile of daily precipitation
    pa80 = prcp - LongTermDaily(prcp, year_long, 0.80)
    pa80 = ResampleDay2Dekad(pa80 > 0, np.nansum)
    # - Number of days above 50% percentile of daily precipitation
    pa50 = prcp - LongTermDaily(prcp, year_long, 0.50)
    pa50 = ResampleDay2Dekad(pa50 > 0, np.nansum)
    # - Number of days below 5% percentile of daily precipitation
    pb05 = prcp - LongTermDaily(prcp, year_long, 0.05)
    pb05 = ResampleDay2Dekad(pb05 < 0, np.nansum)
    # - Number of days below 10% percentile of daily precipitation
    pb10 = prcp - LongTermDaily(prcp, year_long, 0.10)
    pb10 = ResampleDay2Dekad(pb10 < 0, np.nansum)
    # - Number of days below 20% percentile of daily precipitation
    pb20 = prcp - LongTermDaily(prcp, year_long, 0.20)
    pb20 = ResampleDay2Dekad(pb20 < 0, np.nansum)

    # Variables related to Evapotranspiration (ET)
    # - Accumulated ET
    eacu = ResampleDay2Dekad(etos, np.nansum)
    # - Number of days above 95% percentile of daily ET
    ea95 = etos - LongTermDaily(etos, year_long, 0.95)
    ea95 = ResampleDay2Dekad(ea95 > 0, np.nansum)
    # - Number of days above 90% percentile of daily ET
    ea90 = etos - LongTermDaily(etos, year_long, 0.90)
    ea90 = ResampleDay2Dekad(ea90 > 0, np.nansum)
    # - Number of days above 80% percentile of daily ET
    ea80 = etos - LongTermDaily(etos, year_long, 0.80)
    ea80 = ResampleDay2Dekad(ea80 > 0, np.nansum)
    # - Number of days above 80% percentile of daily ET
    ea50 = etos - LongTermDaily(etos, year_long, 0.50)
    ea50 = ResampleDay2Dekad(ea50 > 0, np.nansum)
    # - Number of days below 5% percentile of daily ET
    eb05 = etos - LongTermDaily(etos, year_long, 0.05)
    eb05 = ResampleDay2Dekad(eb05 < 0, np.nansum)
    # - Number of days below 10% percentile of daily ET
    eb10 = etos - LongTermDaily(etos, year_long, 0.10)
    eb10 = ResampleDay2Dekad(eb10 < 0, np.nansum)
    # - Number of days below 20% percentile of daily ET
    eb20 = etos - LongTermDaily(etos, year_long, 0.20)
    eb20 = ResampleDay2Dekad(eb20 < 0, np.nansum)

    # Variables related to Available Water (AWTR = PRECIP - ET)
    awtr = prcp - etos
    # - Accumulated AWTR
    aacu = ResampleDay2Dekad(awtr, np.nansum)
    # - Number of days above 95% percentile of daily AWTR
    aa95 = awtr - LongTermDaily(awtr, year_long, 0.95)
    aa95 = ResampleDay2Dekad(aa95 > 0, np.nansum)
    # - Number of days above 90% percentile of daily AWTR
    aa90 = awtr - LongTermDaily(awtr, year_long, 0.90)
    aa90 = ResampleDay2Dekad(aa90 > 0, np.nansum)
    # - Number of days above 80% percentile of daily AWTR
    aa80 = awtr - LongTermDaily(awtr, year_long, 0.80)
    aa80 = ResampleDay2Dekad(aa80 > 0, np.nansum)
    # - Number of days above 80% percentile of daily AWTR
    aa50 = awtr - LongTermDaily(awtr, year_long, 0.50)
    aa50 = ResampleDay2Dekad(aa50 > 0, np.nansum)
    # - Number of days below 5% percentile of daily AWTR
    ab05 = awtr - LongTermDaily(awtr, year_long, 0.05)
    ab05 = ResampleDay2Dekad(ab05 < 0, np.nansum)
    # - Number of days below 10% percentile of daily AWTR
    ab10 = awtr - LongTermDaily(awtr, year_long, 0.10)
    ab10 = ResampleDay2Dekad(ab10 < 0, np.nansum)
    # - Number of days below 20% percentile of daily AWTR
    ab20 = awtr - LongTermDaily(awtr, year_long, 0.20)
    ab20 = ResampleDay2Dekad(ab20 < 0, np.nansum)

    # Variables related to Temperature (TEMP)
    # - Mean of Maximum temperature
    tmax = ResampleDay2Dekad(tmax, np.nanmean)
    # - Mean of Minimum temperature
    tmin = ResampleDay2Dekad(tmin, np.nanmean)
    # - Mean of Mean temperature
    tavg = ResampleDay2Dekad((tmax+tmin)/2, np.nanmean)
    # - Number of days above 95% percentile of daily maximum temperature
    ta95 = tmax - LongTermDaily(tmax, year_long, 0.95)
    ta95 = ResampleDay2Dekad(ta95 > 0, np.nansum)
    # - Number of days below 5% percentile of daily minimum temperature
    tb05 = tmin - LongTermDaily(tmin, year_long, 0.05)
    tb05 = ResampleDay2Dekad(tb05 < 0, np.nansum)
    # - Accumulated GDD 
    gacu = ResampleDay2Dekad(gdd, np.nansum)
    # - Accumulated GDD anomaly to 30-year average (1986-2015)
    ganm = LongTermDaily(gdd, year_long, 'mean')
    ganm = ResampleDay2Dekad(gdd - ganm, np.nansum)

    # Variables related to NDVI
    ndvi = ndvi_raw.copy()

    # Aggregate all variables to a single dataframe
    pacu = pd.concat({'pacu': pacu}, names=['name'],axis=1)
    pdry = pd.concat({'pdry': pdry}, names=['name'],axis=1)
    pa95 = pd.concat({'pa95': pa95}, names=['name'],axis=1)
    pa90 = pd.concat({'pa90': pa90}, names=['name'],axis=1)
    pa80 = pd.concat({'pa80': pa80}, names=['name'],axis=1)
    pa50 = pd.concat({'pa50': pa50}, names=['name'],axis=1)
    pb05 = pd.concat({'pb05': pb05}, names=['name'],axis=1)
    pb10 = pd.concat({'pb10': pb10}, names=['name'],axis=1)
    pb20 = pd.concat({'pb20': pb20}, names=['name'],axis=1)
    eacu = pd.concat({'eacu': eacu}, names=['name'],axis=1)
    ea95 = pd.concat({'ea95': ea95}, names=['name'],axis=1)
    ea90 = pd.concat({'ea90': ea90}, names=['name'],axis=1)
    ea80 = pd.concat({'ea80': ea80}, names=['name'],axis=1)
    ea50 = pd.concat({'ea50': ea50}, names=['name'],axis=1)
    eb05 = pd.concat({'eb05': eb05}, names=['name'],axis=1)
    eb10 = pd.concat({'eb10': eb10}, names=['name'],axis=1)
    eb20 = pd.concat({'eb20': eb20}, names=['name'],axis=1)
    aacu = pd.concat({'aacu': aacu}, names=['name'],axis=1)
    aa95 = pd.concat({'aa95': aa95}, names=['name'],axis=1)
    aa90 = pd.concat({'aa90': aa90}, names=['name'],axis=1)
    aa80 = pd.concat({'aa80': aa80}, names=['name'],axis=1)
    aa50 = pd.concat({'aa50': aa50}, names=['name'],axis=1)
    ab05 = pd.concat({'ab05': ab05}, names=['name'],axis=1)
    ab10 = pd.concat({'ab10': ab10}, names=['name'],axis=1)
    ab20 = pd.concat({'ab20': ab20}, names=['name'],axis=1)
    tavg = pd.concat({'tavg': tavg}, names=['name'],axis=1)
    tmax = pd.concat({'tmax': tmax}, names=['name'],axis=1)
    ta95 = pd.concat({'ta95': ta95}, names=['name'],axis=1)
    tmin = pd.concat({'tmin': tmin}, names=['name'],axis=1)
    tb05 = pd.concat({'tb05': tb05}, names=['name'],axis=1)
    gacu = pd.concat({'gacu': gacu}, names=['name'],axis=1)
    ganm = pd.concat({'ganm': ganm}, names=['name'],axis=1)
    ndvi = pd.concat({'ndvi': ndvi}, names=['name'],axis=1)
    PRED_ALL = pd.concat([
        pacu, pdry, pa95, pa90, pa80, pa50, pb05, pb10, pb20,
        eacu, ea95, ea90, ea80, ea50, eb05, eb10, eb20,
        aacu, aa95, aa90, aa80, aa50, ab05, ab10, ab20, 
        tavg, tmax, ta95, tmin, tb05,
        gacu, ganm, ndvi
    ],axis=1)
    
    # EO data variable table
    variable_table = [['pacu', 'Accumulated precipitation (mm)', 'sum'],
                      ['pdry', 'Number of dry days (zero precipitation)', 'sum'],
                      ['pa95', 'Number of precipitation events above 95 percentile', 'sum'],
                      ['pa90', 'Number of precipitation events above 90 percentile', 'sum'],
                      ['pa80', 'Number of precipitation events above 80 percentile', 'sum'],
                      ['pa50', 'Number of precipitation events above 50 percentile', 'sum'],
                      ['pb05', 'Number of precipitation events below 5 percentile', 'sum'],
                      ['pb10', 'Number of precipitation events below 10 percentile', 'sum'],
                      ['pb20', 'Number of precipitation events below 20 percentile', 'sum'],
                      ['eacu', 'Accumulated evapotranspiration (mm)', 'sum'],
                      ['ea95', 'Number of evapotranspiration events above 95 percentile', 'sum'],
                      ['ea90', 'Number of evapotranspiration events above 90 percentile', 'sum'],
                      ['ea80', 'Number of evapotranspiration events above 80 percentile', 'sum'],
                      ['ea50', 'Number of evapotranspiration events above 50 percentile', 'sum'],
                      ['eb05', 'Number of evapotranspiration events below 5 percentile', 'sum'],
                      ['eb10', 'Number of evapotranspiration events below 10 percentile', 'sum'],
                      ['eb20', 'Number of evapotranspiration events below 20 percentile', 'sum'],
                      ['aacu', 'Accumulated available water (precipitation - evapotranspiration) (mm)', 'sum'],
                      ['aa95', 'Number of available water events above 95 percentile', 'sum'],
                      ['aa90', 'Number of available water events above 90 percentile', 'sum'],
                      ['aa80', 'Number of available water events above 80 percentile', 'sum'],
                      ['aa50', 'Number of available water events above 50 percentile', 'sum'],
                      ['ab05', 'Number of available water events below 5 percentile', 'sum'],
                      ['ab10', 'Number of available water events below 10 percentile', 'sum'],
                      ['ab20', 'Number of available water events below 20 percentile', 'sum'],
                      ['tavg', 'Mean of average temperature (°C)', 'mean'],
                      ['tmax', 'Mean of maximum temperature (°C)', 'mean'],
                      ['ta95', 'Number of days of maximum temperature above 95 percentile', 'sum'],
                      ['tmin', 'Mean of minimum temperature (°C)', 'mean'],
                      ['tb05', 'Number of days of minimum temperature below 5 percentile', 'sum'],
                      ['gacu', 'Accumulated growing degree days', 'sum'],
                      ['ganm', 'Accumulated growing degree days anomalies', 'sum'],
                      ['ndvi', 'Maximum NDVI', 'mean']
                     ]
    variable_table = pd.DataFrame(variable_table, columns=['Name','Definition','Aggregation_method'])
    # ---------------------------------------- #


    # Exporting EO data ---------------------- #
    # Remove existing EO data files
    for file in glob.glob('./data_in/*_pred.hdf'): 
        os.remove(file)
    # EO data
    for fnid in np.unique(fnids_eo):
        dfPred = PRED_ALL.loc[:,pd.IndexSlice[:,fnid]].droplevel(1,axis=1)
        dfPred.index.name = 'time'
        dfPred = dfPred.loc[(dfPred.index >= dekad_start) & (dfPred.index <= dekad_end)]
        # Save as HDF format
        save_hdf('./data_in/{:s}_pred.hdf'.format(fnid), dfPred, set_print=False)
    # EO feature table
    filn = './data_in/pred_table.hdf'
    if os.path.exists(filn):
        save_hdf(filn, variable_table, set_print=False)
    # Temporal data coverage
    with open('./data_in/eodata_coverage.txt', 'w') as txt:
        print('='*40, file=txt)
        print('Current EO data temporal coverage',file=txt)
        print('-'*40,file=txt)
        print('PRCP:\t%s to %s' % (prcp_start.strftime('%Y-%m-%d'), prcp_end.strftime('%Y-%m-%d')),file=txt)
        print('ETOS:\t%s to %s' % (etos_start.strftime('%Y-%m-%d'), etos_end.strftime('%Y-%m-%d')),file=txt)
        print('TEMP:\t%s to %s' % (tmax_start.strftime('%Y-%m-%d'), tmax_end.strftime('%Y-%m-%d')),file=txt)
        print('NDVI:\t%s to %s' % (ndvi_start.strftime('%Y-%m-%d'), ndvi_end.strftime('%Y-%m-%d')),file=txt)
        print('='*40,file=txt)
        print('Data extension',file=txt)
        print('-'*40,file=txt)
        print('Start:\t%s' % (dekad_start.strftime('%Y-%m-%d')),file=txt)
        print('End:\t%s' % (dekad_end.strftime('%Y-%m-%d')),file=txt)
        print('='*40,file=txt)
    print('Exporting EO data is completed.')
    # ---------------------------------------- #
    
    return
    
    
    
    
