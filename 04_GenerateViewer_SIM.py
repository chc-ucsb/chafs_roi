import os, json, glob, time
from datetime import datetime
from functools import reduce
from itertools import product, combinations, compress
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import geopandas as gpd
from chafs_tools import CHAFS_Aggregate_CPSM

def month_date(year, forecast_end, lead):
    # No year (scores)
    if pd.isna(year): year = 2000
    fe = pd.to_datetime(str(int(year))+'-'+forecast_end)
    # No lead (observation)
    if pd.isna(lead): lead = 0
    ld = pd.DateOffset(months=lead)
    return fe-ld

def main():
    # Export a specific cpsme to "viewer_data_sim.csv" - # 
    cpsme_list = [
        ['Somalia','Maize','Gu','XGB','YFT_ACUM_ALL'],
        ['Somalia','Sorghum','Gu','XGB','YFT_ACUM_ALL'],
        ['Somalia','Maize','Deyr','XGB','YFT_ACUM_ALL'],
        ['Somalia','Sorghum','Deyr','XGB','YFT_ACUM_ALL'],
        ['Kenya','Maize','Long','XGB','YFT_ACUM_ALL'],
        ['Kenya','Maize','Short','XGB','YFT_ACUM_ALL'],
        ['Malawi','Maize','Main','XGB','YFT_ACUM_ALL'],
        ['Burkina Faso','Maize','Main','XGB','YFT_ACUM_ALL'],
    ]
    # Printing a specific lead
    container_score = []
    container_value = []
    container_importance = []
    for (country_name, product_name, season_name, model_name, exp_name) in cpsme_list:
        cpsme = (country_name, product_name, season_name, model_name, exp_name)
        cpsme_string = '%s_%s_%s_%s_%s' % (cpsme)
        print(cpsme_string)
        cpsme_short = '%s%s%s_%s_%s' % (country_name[0].upper(),product_name[0].upper(),season_name[0].upper(),model_name,exp_name)
        box = CHAFS_Aggregate_CPSM(cpsme)
        # Skill score
        stack_score = box['stack_score']
        stack_score['model'] = model_name
        stack_score['type'] = 'score'
        container_score.append(stack_score)
        # Forecast value
        stack_value = box['stack_value']
        stack_value['model'] = model_name
        stack_value['type'] = 'value'
        container_value.append(stack_value)
        # Importance
        stack_importance = box['stack_importance']
        stack_importance['model'] = model_name
        stack_importance['type'] = 'importance'
        container_importance.append(stack_importance)
    stack_score = pd.concat(container_score, axis=0).reset_index(drop=True)
    stack_value = pd.concat(container_value, axis=0).reset_index(drop=True)
    stack_importance = pd.concat(container_importance, axis=0).reset_index(drop=True)

    # # NaN Errors in Hyperparameter tunings 
    # stack_importance.loc[stack_importance['value'].isna(),'value'] = 0
    # -------------------------------------------------- #
    
    
    
    # Crop Production Data ----------------------------- #
    # Load FEWSNET admin boundaries
    shape = pd.read_csv('https://raw.githubusercontent.com/chc-ucsb/GlobalCropData/main/public/gscd_shape_stable.csv', index_col=0)
    # shape = shape[shape.ADMIN0.isin(country_to_use)].reset_index(drop=True)
    dist_info = shape[['FNID','ADMIN0','ADMIN1','ADMIN2']]
    dist_info.columns = ['fnid','country','admin1','admin2']
    column_order = ['fnid','country','admin1','admin2','year','product','season','month','dekad','day','out-of-sample','variable','value']

    # Load crop area, production, yield data
    df = pd.read_csv('https://raw.githubusercontent.com/chc-ucsb/GlobalCropData/main/public/gscd_data_stable.csv', index_col=0)
    # Reduce data according to CPS
    container = []
    for (country_name, product_name, season_name, model_name, exp_name) in cpsme_list:
        sub = df[
                (df['country'] == country_name) &
                (df['product'] == product_name) &
                (df['season_name'] == season_name)
        ]
        container.append(sub)
    data = pd.concat(container, axis=0).reset_index(drop=True)
    data.rename(columns={'season_name':'season', 'indicator':'variable'}, inplace=True)
    data = data[['fnid','product','season','year','variable','value']]
    container0 = data.copy()
    container0.replace({'variable': {'area':'area_obs', 
                                     'production':'prod_obs',
                                     'yield':'yield_obs'}},
                       inplace=True)

    # (1) Long-term mean
    temp = data[data.year < 2019]
    long = temp.groupby(['fnid','product','season','variable'])['value'].mean().reset_index()
    long.replace({'variable':{'area':'area_mean_all',
                              'production':'prod_mean_all',
                              'yield':'yield_mean_all'
                             }},
                 inplace=True
                )


    # (2) Recent 10 years mean (2009-2018)
    temp = data[data.year.isin(np.arange(2009,2019))]
    last = temp.groupby(['fnid','product','season','variable'])['value'].mean().reset_index()
    last.replace({'variable':{'area':'area_mean_last10',
                              'production':'prod_mean_last10',
                              'yield':'yield_mean_last10'
                             }},
                 inplace=True
                )
    container1 = pd.concat([long,last],axis=0)
    container1.insert(3, 'year', np.nan)

    # Create a table for calculating % of errors to long and recent 10 years
    crop_mean_long = long.pivot_table(index=['fnid','product','season'], columns='variable',values='value')
    crop_mean_last10 = last.pivot_table(index=['fnid','product','season'], columns='variable',values='value')

    # Merge crop data
    container = pd.concat([container0, container1], axis=0)
    container[['month','dekad','day','out-of-sample']] = np.nan
    container = pd.merge(container, dist_info, left_on='fnid', right_on='fnid')
    container_obs = container[column_order]

    # Cropped area
    crop_area = pd.read_hdf('./data/cropmask/adm_cropland_area.hdf')['area00']*100
    crop_area = crop_area.reset_index()
    crop_area.columns = ['fnid', 'value']
    crop_area['variable'] = 'crop_area_percent'
    crop_area[['year','product','season','month','dekad','day','out-of-sample']] = np.nan
    crop_area = pd.merge(crop_area, dist_info, left_on='fnid', right_on='fnid')
    crop_area = crop_area[column_order]
    crop_area['type'] = 'crop'

    # Merge all
    stack_crop = pd.concat([container_obs, crop_area], axis=0).reset_index(drop=True)
    stack_crop['type'] = 'crop'
    # -------------------------------------------------- #
    
    
    # Generate a CSV file with specific variables ------ #
    # Load stacked values and scores
    stack_value['variable'] = 'yield_' + stack_value['variable']
    eoname_list = stack_importance['eoname'].unique()
    stack_importance['variable'] = 'FI_'+stack_importance['eoname']
    stack_score = stack_score.rename(columns={'score':'variable'})
    stack_score = stack_score[stack_score['variable'].isin(['mape','nse'])]   # Remain only MAPE and NSE
    stack_score['variable'] = 'yield_' + stack_score['variable']
    stack_forecast = pd.concat([stack_score, stack_value, stack_importance])
    stack_forecast = stack_forecast.merge(shape[['FNID','ADMIN1','ADMIN2']], left_on='fnid', right_on='FNID')
    stack_forecast = stack_forecast.rename(columns={
        'ADMIN1':'admin1',
        'ADMIN2':'admin2',
        'season_name':'season',
        'pred':'out-of-sample'
    })
    stack_forecast['out-of-sample'].replace({'obs':np.nan,'hcst':0,'fcst':1,'rcst':2},inplace=True)

    # Merging "stack_crop" and "stack_forecast"
    stack = pd.concat([stack_crop, stack_forecast], axis=0)

    # Forecast start, end, and season start month
    cols = ['country','product','season','forecast_start_month','season_start_month','forecast_end']
    start = [
        ['Somalia','Sorghum','Gu',2,3,'08-01'],
        ['Somalia','Sorghum','Deyr',9,10,'03-01'],
        ['Somalia','Maize','Gu',2,3,'08-01'],
        ['Somalia','Maize','Deyr',9,10,'03-01'],
        ['Kenya','Maize','Long',2,3,'08-01'],
        ['Kenya','Maize','Short',9,10,'03-01'],
        ['Malawi','Maize','Main',10,11,'04-01'],
        ['Burkina Faso','Maize','Main',4,5,'10-01'],
        ['Burkina Faso','Sorghum','Main',4,5,'10-01'],
    ]
    start = pd.DataFrame(start, columns=cols)
    stack = stack.merge(start, on=['country','product','season'], how='inner')

    # Update "year" and "month" with "forecast_end" and "lead"
    stack['month'] = np.nan
    stack['date'] = np.vectorize(month_date)(stack['year'],stack['forecast_end'],stack['lead'])
    target = stack['lead'].notna()
    stack.loc[target, 'year'] = stack.loc[target, 'date'].dt.year
    stack.loc[target, 'month'] = stack.loc[target, 'date'].dt.month
    stack.loc[stack['variable'].isin(eoname_list),'year'] = np.nan
    # Remove "year" of score values
    stack.loc[stack['type'] == 'score','year'] = np.nan
    # - (deprecated) Dekad and day
    stack.loc[target, 'dekad'] = 3
    stack.loc[target, 'day'] = 21

    # "crop_area_percent"
    stack = pd.concat([stack, crop_area])

    # Remove rows having missing values
    stack = stack[stack['value'].notna()].reset_index(drop=True)

    # Save to CSV file.
    stack1 = stack[[
        'fnid','country','admin1','admin2','product','season',
        'forecast_start_month','season_start_month','model',
        'year','lead','month','dekad','day','out-of-sample',
        'variable','value'
    ]]
    fn_out = './viewer/viewer_data_sim.csv'
    stack1.to_csv(fn_out)
    print('%s is saved.' % fn_out)
    # -------------------------------------------------- #
    
    
    
    # Merge "viewer_data_sim.csv" and "viewer_data_com.csv"
    # (1) Complicated approach
    df_com = pd.read_csv('./viewer/viewer_data_com.csv', low_memory=False).drop(['Unnamed: 0'],axis=1)
    # - Remove 1st and 2nd dekadal data and leave only 3rd dekadal data
    df_com = df_com[~df_com['day'].isin([1,11])]
    # print(df_com['variable'].unique())
    # print(df_com['out-of-sample'].unique())

    # (2) Simplified approach
    df_sim = pd.read_csv('./viewer/viewer_data_sim.csv', low_memory=False).drop(['Unnamed: 0'],axis=1)
    # - Rename XGB to GB
    df_sim['model'].replace({'XGB':'GB'}, inplace=True)
    # print(df_sim['variable'].unique())
    # print(df_sim['out-of-sample'].unique())

    # (3) Merging two CSV files
    assert all(df_com.columns.isin(df_sim.columns))
    df_merged = pd.concat([df_com, df_sim], axis=0)
    # - Remove duplicates (crop, croparea, etc.)
    df_merged = df_merged.drop_duplicates().reset_index(drop=True).drop(['lead'],axis=1)
    # - Retain only the end of month forecasts
    df_date = pd.to_datetime(df_merged[['year','month','day']])
    today = pd.to_datetime('today')
    dayfirst = pd.to_datetime('%04d-%02d' % (today.year, today.month), dayfirst=True)
    df_merged = df_merged[~(df_date >= dayfirst)].reset_index(drop=True)
    print(df_merged.shape)
    fn_out = './viewer/viewer_data.csv'
    df_merged.to_csv(fn_out)
    print('%s is saved.' % fn_out)
    fn_out = '/home/chc-data-out/people/dlee/viewer/viewer_data.csv'
    df_merged.to_csv(fn_out)
    print('%s is saved.' % fn_out)
    # -------------------------------------------------- #
    
    
    # -------------------------------------------------- #
    df = pd.read_csv('./viewer/viewer_data.csv', low_memory=False).drop(['Unnamed: 0'],axis=1)
    df['date'] = pd.to_datetime(df[['year','month','day']])
    cpsm = df.loc[df['model'].notna(),['country','product','season','model']].drop_duplicates()
    cpsm = cpsm.reset_index(drop=True)
    for i, (country_name, product_name, season_name, model_name) in cpsm.iterrows():
        sub = df[
            (df['country'] == country_name) &
            (df['product'] == product_name) &
            (df['season'] == season_name) &
            (df['model'] == model_name) &
            (df['year'] == 2022) &
            (df['out-of-sample'] == 2) &
            (df['variable'] == 'yield_fcst')
        ]
        cpsm.loc[i,'latest'] = sub['date'].max()
    cpsm = cpsm.sort_values(by=['country','product','season','model']).reset_index(drop=True)
    print(cpsm)
    print(df['variable'].unique())
    # -------------------------------------------------- #
    
    
    
if __name__ == "__main__":
    main()