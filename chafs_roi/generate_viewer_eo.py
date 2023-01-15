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
        df_long = df_long.groupby(df_long.index.strftime(duraztion)).transform('std')
    else:
        raise NameError('Unsupported method')
    return df_long


def generate_viewer_eo():
    
    # Load FEWSNET admin boundaries
    shape = gpd.read_file('https://raw.githubusercontent.com/chc-ucsb/gscd/main/public/gscd_shape_stable.json').drop(columns='id')
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


    # Calculate anomalies
    for name, method in zip(names, methods):
        # Dekadal
        exec('%s_dekad = merged[merged.name == "%s"].drop("name", axis=1)' % (name, name))
        exec('mean_long = LongTermDaily(%s_dekad, np.arange(1986,2015), duration="dekad", method="mean")' % (name))
        exec('std_long = LongTermDaily(%s_dekad, np.arange(1986,2015), duration="dekad", method="std")' % (name))
        exec('%s_dekad_anom_long = (%s_dekad - mean_long)/std_long' % (name, name))
        exec('%s_dekad_pnom_long = (%s_dekad - mean_long)/mean_long*100' % (name, name))
        exec('mean_last10 = LongTermDaily(%s_dekad, np.arange(2009,2019), duration="dekad", method="mean")' % (name))
        exec('std_last10 = LongTermDaily(%s_dekad, np.arange(2009,2019), duration="dekad", method="std")' % (name))
        exec('%s_dekad_anom_last10 = (%s_dekad - mean_last10)/std_last10' % (name, name))
        exec('%s_dekad_pnom_last10 = (%s_dekad - mean_last10)/mean_last10*100' % (name, name))
        # Monthly
        exec('%s_month = %s_dekad.resample("1M").%s()' % (name, name, method))
        exec('mean_long = LongTermDaily(%s_month, np.arange(1986,2015), duration="month", method="mean")' % (name))
        exec('std_long = LongTermDaily(%s_month, np.arange(1986,2015), duration="month", method="std")' % (name))
        exec('%s_month_anom_long = (%s_month - mean_long)/std_long' % (name, name))
        exec('%s_month_pnom_long = (%s_month - mean_long)/mean_long*100' % (name, name))
        exec('mean_last10 = LongTermDaily(%s_month, np.arange(2009,2019), duration="month", method="mean")' % (name))
        exec('std_last10 = LongTermDaily(%s_month, np.arange(2009,2019), duration="month", method="std")' % (name))
        exec('%s_month_anom_last10 = (%s_month - mean_last10)/std_last10' % (name, name))
        exec('%s_month_pnom_last10 = (%s_month - mean_last10)/mean_last10*100' % (name, name))

    # Stacking all variables
    container = []
    comb = product(names, [
        # 'dekad', 
        # 'dekad_anom_long', 'dekad_pnom_long', 
        # 'dekad_pnom_last10', 'dekad_anom_last10', 
        'month', 
        'month_anom_long', 'month_pnom_long', 
        # 'month_pnom_last10', 'month_anom_last10', 
    ])
    for name, vtype in comb:
        vname = '%s_%s' % (name, vtype)
        exec('df = %s.stack().reset_index(drop=False)' % (vname))
        df.columns = ['time', 'fnid', 'value']
        df['variable'] = name
        df['year'] = df.time.dt.year 
        df['month'] = df.time.dt.month
        if (vtype == 'dekad') | (vtype == 'month'):
            df['type'] = 'value'
        elif vtype.endswith('pnom_long'):
            df['type'] = 'pnom_long'
        elif vtype.endswith('anom_long'):
            df['type'] = 'anom_long'
        elif vtype.endswith('pnom_last10'):
            df['type'] = 'pnom_last10'
        elif vtype.endswith('anom_last10'):
            df['type'] = 'anom_last10'
        else:
            raise NameError('Unsupported type!')

        if vtype.startswith('month'):
            df['duration'] = 'month'
            df['day'] = np.nan
        else:
            df['duration'] = 'dekad'
            df['day'] = df.time.dt.day
        # df = df.drop('time', axis=1)
        container.append(df)
    df = pd.concat(container, axis=0).reset_index(drop=True)
    df = df.merge(fnids_info[['fnid','country','admin1','admin2']].drop_duplicates(), left_on='fnid', right_on='fnid', how='inner')
    df = df[['fnid','country','admin1','admin2','duration','time','year','month','day','variable','type','value']]

    # Save a file
    filn = './viewer/viewer_eo_data.csv'
    df.to_csv(filn)
    print('%s is saved.' % filn)
    
    return