import os, sys, glob, json
from itertools import product
from functools import reduce
import numpy as np
import pandas as pd
import geopandas as gpd
import json
np.seterr(divide='ignore', invalid='ignore')

# Long-term averages
def LongTermDaily(df, years, duration='dekad', method='mean'):
    if duration == 'dekad':
        duration = '%m-%d'
    elif duration == 'month':
        duration = '%m'
    df_long = df[df.index.year.isin(years)]
    df_long = df_long.reindex(df.index)
    if type(method) == float:
        df_long = df_long.groupby(df_long.index.strftime(duration)).transform(lambda x: x.quantile(method))
    elif method == 'mean':
        df_long = df_long.groupby(df_long.index.strftime(duration)).transform('mean')
    elif method == 'std':
        df_long = df_long.groupby(df_long.index.strftime(duration)).transform('std')
    else:
        raise NameError('Unsupported method')
    return df_long


def generate_viewer_eo():
    # Load FEWSNET admin boundaries
    shape = gpd.read_file('/home/donghoonlee/chafs_roi/data/gscd_shape_stable.json').drop(columns='id')
    shape = shape[shape['ADMIN0'].isin(['Somalia','Kenya','Malawi','Burkina Faso'])]
    shape.geometry = shape.geometry.simplify(0.01)
    geojson = json.loads(shape[['FNID','geometry']].to_json())
    fnids_info = shape[['FNID','ADMIN0','ADMIN1','ADMIN2']]
    fnids_info.columns = ['fnid','country','admin1','admin2']
    fnids = fnids_info['fnid'].unique()

    # Load EO data per station
    names = ['pacu','pdry','eacu','aacu','tavg','tmax','tmin','gacu','ndvi']
    methods = ['sum', 'sum', 'sum', 'sum','mean','mean','mean','sum','max']
    container = []
    for fnid in fnids:
        df = pd.read_hdf('./data_in/%s_pred.hdf' % fnid)
        df = df.stack().reset_index(drop=False)
        df.columns = ['time', 'name', fnid]
        container.append(df)
    merged = reduce(lambda  left,right: pd.merge(left,right,on=['time','name'],how='outer'), container)
    merged = merged.set_index('time')
    merged = merged[merged.name.isin(names)]

    # pacu
    pacu_dekad = merged[merged.name == 'pacu'].drop("name", axis=1)
    pacu_month = pacu_dekad.resample("1M").sum()
    mean_long = LongTermDaily(pacu_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(pacu_month, np.arange(1986,2015), duration="month", method="std")
    pacu_month_anom_long = (pacu_month - mean_long)/std_long
    pacu_month_pnom_long = (pacu_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(pacu_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(pacu_month, np.arange(2009,2019), duration="month", method="std")
    pacu_month_anom_last10 = (pacu_month - mean_last10)/std_last10
    pacu_month_pnom_last10 = (pacu_month - mean_last10)/mean_last10*100
    # pdry
    pdry_dekad = merged[merged.name == 'pdry'].drop("name", axis=1)
    pdry_month = pdry_dekad.resample("1M").sum()
    mean_long = LongTermDaily(pdry_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(pdry_month, np.arange(1986,2015), duration="month", method="std")
    pdry_month_anom_long = (pdry_month - mean_long)/std_long
    pdry_month_pnom_long = (pdry_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(pdry_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(pdry_month, np.arange(2009,2019), duration="month", method="std")
    pdry_month_anom_last10 = (pdry_month - mean_last10)/std_last10
    pdry_month_pnom_last10 = (pdry_month - mean_last10)/mean_last10*100
    # eacu
    eacu_dekad = merged[merged.name == 'eacu'].drop("name", axis=1)
    eacu_month = eacu_dekad.resample("1M").sum()
    mean_long = LongTermDaily(eacu_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(eacu_month, np.arange(1986,2015), duration="month", method="std")
    eacu_month_anom_long = (eacu_month - mean_long)/std_long
    eacu_month_pnom_long = (eacu_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(eacu_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(eacu_month, np.arange(2009,2019), duration="month", method="std")
    eacu_month_anom_last10 = (eacu_month - mean_last10)/std_last10
    eacu_month_pnom_last10 = (eacu_month - mean_last10)/mean_last10*100
    # aacu
    aacu_dekad = merged[merged.name == 'aacu'].drop("name", axis=1)
    aacu_month = aacu_dekad.resample("1M").sum()
    mean_long = LongTermDaily(aacu_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(aacu_month, np.arange(1986,2015), duration="month", method="std")
    aacu_month_anom_long = (aacu_month - mean_long)/std_long
    aacu_month_pnom_long = (aacu_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(aacu_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(aacu_month, np.arange(2009,2019), duration="month", method="std")
    aacu_month_anom_last10 = (aacu_month - mean_last10)/std_last10
    aacu_month_pnom_last10 = (aacu_month - mean_last10)/mean_last10*100
    # tavg
    tavg_dekad = merged[merged.name == 'tavg'].drop("name", axis=1)
    tavg_month = tavg_dekad.resample("1M").mean()
    mean_long = LongTermDaily(tavg_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(tavg_month, np.arange(1986,2015), duration="month", method="std")
    tavg_month_anom_long = (tavg_month - mean_long)/std_long
    tavg_month_pnom_long = (tavg_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(tavg_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(tavg_month, np.arange(2009,2019), duration="month", method="std")
    tavg_month_anom_last10 = (tavg_month - mean_last10)/std_last10
    tavg_month_pnom_last10 = (tavg_month - mean_last10)/mean_last10*100
    # tmax
    tmax_dekad = merged[merged.name == 'tmax'].drop("name", axis=1)
    tmax_month = tmax_dekad.resample("1M").mean()
    mean_long = LongTermDaily(tmax_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(tmax_month, np.arange(1986,2015), duration="month", method="std")
    tmax_month_anom_long = (tmax_month - mean_long)/std_long
    tmax_month_pnom_long = (tmax_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(tmax_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(tmax_month, np.arange(2009,2019), duration="month", method="std")
    tmax_month_anom_last10 = (tmax_month - mean_last10)/std_last10
    tmax_month_pnom_last10 = (tmax_month - mean_last10)/mean_last10*100
    # tmin
    tmin_dekad = merged[merged.name == 'tmin'].drop("name", axis=1)
    tmin_month = tmin_dekad.resample("1M").mean()
    mean_long = LongTermDaily(tmin_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(tmin_month, np.arange(1986,2015), duration="month", method="std")
    tmin_month_anom_long = (tmin_month - mean_long)/std_long
    tmin_month_pnom_long = (tmin_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(tmin_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(tmin_month, np.arange(2009,2019), duration="month", method="std")
    tmin_month_anom_last10 = (tmin_month - mean_last10)/std_last10
    tmin_month_pnom_last10 = (tmin_month - mean_last10)/mean_last10*100
    # gacu
    gacu_dekad = merged[merged.name == 'gacu'].drop("name", axis=1)
    gacu_month = gacu_dekad.resample("1M").sum()
    mean_long = LongTermDaily(gacu_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(gacu_month, np.arange(1986,2015), duration="month", method="std")
    gacu_month_anom_long = (gacu_month - mean_long)/std_long
    gacu_month_pnom_long = (gacu_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(gacu_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(gacu_month, np.arange(2009,2019), duration="month", method="std")
    gacu_month_anom_last10 = (gacu_month - mean_last10)/std_last10
    gacu_month_pnom_last10 = (gacu_month - mean_last10)/mean_last10*100
    # ndvi
    ndvi_dekad = merged[merged.name == 'ndvi'].drop("name", axis=1)
    ndvi_month = ndvi_dekad.resample("1M").mean()
    mean_long = LongTermDaily(ndvi_month, np.arange(1986,2015), duration="month", method="mean")
    std_long = LongTermDaily(ndvi_month, np.arange(1986,2015), duration="month", method="std")
    ndvi_month_anom_long = (ndvi_month - mean_long)/std_long
    ndvi_month_pnom_long = (ndvi_month - mean_long)/mean_long*100
    mean_last10 = LongTermDaily(ndvi_month, np.arange(2009,2019), duration="month", method="mean")
    std_last10 = LongTermDaily(ndvi_month, np.arange(2009,2019), duration="month", method="std")
    ndvi_month_anom_last10 = (ndvi_month - mean_last10)/std_last10
    ndvi_month_pnom_last10 = (ndvi_month - mean_last10)/mean_last10*100

    # Stacking all variables
    variable = [
        pacu_month, pacu_month_anom_long, pacu_month_pnom_long,
        pdry_month, pdry_month_anom_long, pdry_month_pnom_long,
        eacu_month, eacu_month_anom_long, eacu_month_pnom_long,
        aacu_month, aacu_month_anom_long, aacu_month_pnom_long,
        tavg_month, tavg_month_anom_long, tavg_month_pnom_long,
        tmax_month, tmax_month_anom_long, tmax_month_pnom_long,
        tmin_month, tmin_month_anom_long, tmin_month_pnom_long,
        gacu_month, gacu_month_anom_long, gacu_month_pnom_long,
        ndvi_month, ndvi_month_anom_long, ndvi_month_pnom_long
    ]
    variable_name = [
        'pacu_month', 'pacu_month_anom_long', 'pacu_month_pnom_long',
        'pdry_month', 'pdry_month_anom_long', 'pdry_month_pnom_long',
        'eacu_month', 'eacu_month_anom_long', 'eacu_month_pnom_long',
        'aacu_month', 'aacu_month_anom_long', 'aacu_month_pnom_long',
        'tavg_month', 'tavg_month_anom_long', 'tavg_month_pnom_long',
        'tmax_month', 'tmax_month_anom_long', 'tmax_month_pnom_long',
        'tmin_month', 'tmin_month_anom_long', 'tmin_month_pnom_long',
        'gacu_month', 'gacu_month_anom_long', 'gacu_month_pnom_long',
        'ndvi_month', 'ndvi_month_anom_long', 'ndvi_month_pnom_long'
    ]
    container = []
    for var, name in zip(variable, variable_name):
        df = var.stack().reset_index(drop=False)
        df.columns = ['time', 'fnid', 'value']
        df['variable'] = name
        df['year'] = df.time.dt.year 
        df['month'] = df.time.dt.month
        vtype = '_'.join(name.split('_')[1:])
        if vtype == 'month':
            df['type'] = 'value'
        elif vtype == 'month_anom_long':
            df['type'] = 'anom_long'
        elif vtype == 'month_pnom_long':
            df['type'] = 'pnom_long'
        else:
            raise NameError('Unsupported type!')
        df['duration'] = 'month'
        df['day'] = 1
        container.append(df)
    df = pd.concat(container, axis=0).reset_index(drop=True)
    df = df.merge(fnids_info[['fnid','country','admin1','admin2']].drop_duplicates(), left_on='fnid', right_on='fnid', how='inner')
    df = df[['fnid','country','admin1','admin2','duration','time','year','month','day','variable','type','value']]

    # Save a file
    filn = './viewer/viewer_eo_data.csv'
    df.to_csv(filn)
    print('%s is saved.' % filn)
    fn_out = '/home/chc-data-out/people/dlee/viewer/viewer_data_eo.csv'
    df.to_csv(fn_out)
    print('%s is saved.' % fn_out)
    
    return