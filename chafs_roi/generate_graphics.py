import os, json, glob, time
from datetime import datetime
from functools import reduce
from itertools import product
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from .tools import month2lead
from .chafs_graphic import PlotForecastMap, PlotScoreHeatmap, PlotScoreMap, PlotImportanceHeatmap, PlotImportanceMap
import subprocess
import shlex


def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        line = process.stdout.readline()
        if not line: break
        print(line.rstrip())
    return


def generate_graphics():
    
    # Load FEWSNET admin boundaries
    shape = gpd.read_file('./data/shapefile/gscd_shape_stable.shp')
    shape['ADMINX'] = shape['ADMIN2'].fillna(shape['ADMIN1'])
    shape.geometry = shape.geometry.simplify(0.01)
    geojson = json.loads(shape[['FNID','geometry']].to_json())

    # Load "viewer_data.csv"
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


    ####################
    fig_scale = 1.3
    ####################
    for i, (country_name, product_name, season_name, model_name, _) in cpsm.iterrows():
        cpsm_string = '%s-%s-%s-%s' % (country_name, product_name, season_name, model_name)
        cps = (country_name, product_name, season_name)
        cps_string = '%s-%s-%s' % (cps)
        cps_string = cps_string.replace(' ', '-')
        if not os.path.exists('./viewer/figures/'+cps_string):
            os.mkdir('./viewer/figures/'+cps_string)
            os.mkdir('./viewer/figures/'+cps_string+'/ET')
            os.mkdir('./viewer/figures/'+cps_string+'/ET/feature_importance')
            os.mkdir('./viewer/figures/'+cps_string+'/ET/forecast')
            os.mkdir('./viewer/figures/'+cps_string+'/ET/hindcast')
            os.mkdir('./viewer/figures/'+cps_string+'/ET/skill_scores')
            os.mkdir('./viewer/figures/'+cps_string+'/GB')
            os.mkdir('./viewer/figures/'+cps_string+'/GB/feature_importance')
            os.mkdir('./viewer/figures/'+cps_string+'/GB/forecast')
            os.mkdir('./viewer/figures/'+cps_string+'/GB/hindcast')
            os.mkdir('./viewer/figures/'+cps_string+'/GB/skill_scores')

        # Observed data
        obs = df[
            (df['country'] == country_name) &
            (df['product'] == product_name) &
            (df['season'] == season_name) &
            (df['variable'] == 'yield_obs')
        ]
        obs = obs.pivot_table(index='fnid', columns=['year'], values='value')
        obs_year = obs.columns.astype(int)

        # Simulation data
        sim = df[
            (df['country'] == country_name) &
            (df['product'] == product_name) &
            (df['season'] == season_name) &
            (df['model'] == model_name)
        ]
        # Adjust year (same year for the same harvest_end)
        sim['lead'] = sim['month'].apply(lambda x: month2lead(cps,x)).values
        sim['year_adj'] = sim['year']
        sim.loc[sim['month'] + sim['lead'] > 12, 'year_adj'] += 1

        # Skill Score ------------------------------ #
        score = sim[
            (sim['out-of-sample'] == 0) &
            (sim['lead'].isin([5,4,3,2,1])) &
            (sim['variable'].isin(['yield_nse', 'yield_mape'])) 
        ].merge(shape, left_on='fnid', right_on='FNID')
        lead_month = score[['lead','month']].drop_duplicates()
        lead_month['monthL'] = lead_month['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
        months = lead_month['monthL']
        # Score Heatmap
        fn_out = './viewer/figures/%s/%s/skill_scores/score_heatmap_%s' % (cps_string, model_name, model_name)
        footnote = '%s' % (cpsm_string)
        fig = PlotScoreHeatmap(score, footnote)
        fig.write_image(fn_out+'.png', scale=fig_scale)
        fig.write_html(fn_out+'.html')
        # Score Maps
        fn_out = './viewer/figures/%s/%s/skill_scores/score_map_%s' % (cps_string, model_name, model_name)
        footnote = '%s' % (cpsm_string)
        fig = PlotScoreMap(score, geojson, country_name, footnote)
        fig.write_image(fn_out+'.png', scale=fig_scale)
        fig.write_html(fn_out+'.html')

        # Feature Importance ----------------------- # 
        importance = sim[sim['variable'].apply(lambda x: x.startswith('FI'))].merge(shape, left_on='fnid', right_on='FNID')
        importance['eoname'] = importance['variable'].apply(lambda x: x.split('.')[0][3:])
        eoname_replace = {'pacu':'PRCP', 'eacu':'ET', 'tavg':'TEMP', 'ndvi':'NDVI'}
        importance['eoname'] = importance['eoname'].replace(eoname_replace)
        if len(importance) > 0:
            # Feature importance Heatmap
            fn_out = './viewer/figures/%s/%s/feature_importance/importance_heatmap_%s' % (cps_string, model_name, model_name)
            footnote = '%s' % (cpsm_string)
            fig = PlotImportanceHeatmap(importance, footnote)
            fig.write_image(fn_out+'.png', scale=fig_scale)
            fig.write_html(fn_out+'.html')
            # Feature importance Maps
            fn_out = './viewer/figures/%s/%s/feature_importance/importance_map_%s' % (cps_string, model_name, model_name)
            footnote = '%s' % (cpsm_string)
            fig = PlotImportanceMap(importance, geojson, country_name, footnote)
            fig.write_image(fn_out+'.png', scale=fig_scale)
            fig.write_html(fn_out+'.html')

        # Hindcast Results ------------------------- #
        hcst = sim[
            (sim['out-of-sample'] == 0) &
            (sim['variable'] == 'yield_fcst')
        ]
        lead_table = hcst[['lead','month']].drop_duplicates().reset_index(drop=True)
        hcst = hcst.pivot_table(index='fnid', columns=['lead','month','year_adj'], values='value')
        error = pd.DataFrame().reindex_like(hcst)
        perror = pd.DataFrame().reindex_like(hcst)
        for i, (lead, month) in lead_table.iterrows():
            error[lead, month] = hcst[lead, month] - obs
            perror[lead, month] = (hcst[lead, month] - obs)/obs*100
        hcst = hcst.T.stack().rename('value').reset_index()
        error = error.T.stack().rename('value').reset_index()
        perror = perror.T.stack().rename('value').reset_index()
        hcst['variable'] = 'yield_fcst'
        error['variable'] = 'yield_error'
        perror['variable'] = 'yield_perror'
        hcst = pd.concat([hcst, error,perror],axis=0)
        hcst['monthL'] = hcst['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
        hcst['year'] = hcst['year_adj']
        hcst.loc[hcst['month'] + hcst['lead'] > 12, 'year'] -= 1
        hcst = hcst.merge(shape, left_on='fnid', right_on='FNID')
        year_lead_table = hcst[['year','lead','month','monthL']].drop_duplicates()
        year_lead_table = year_lead_table.sort_values(by=['year','month'], ascending=True).reset_index(drop=True)
        for i, (year, lead, month, monthL) in year_lead_table.iterrows():
            temp = hcst[(hcst['year'] == year) & (hcst['lead'] == lead)]
            # Forecast Map (percent to recent 10-year mean)
            fn_out = './viewer/figures/%s/%s/hindcast/hcst_perror_map_%s_%04d_%02d' % (cps_string, model_name, model_name, year, month)
            footnote = "%s-%s at %04d.%02d" % (cps_string, model_name, year, month)
            fig = PlotForecastMap(temp[temp['variable'] == 'yield_perror'], geojson, country_name, footnote, ftype='hindcast')
            fig.write_image(fn_out+'.png', scale=fig_scale)

        # Forecast Results ------------------------- #
        year = 2022
        fcst = sim[
            (sim['out-of-sample'] == 2) &
            (sim['variable'].isin(['yield_fcst', 'yield_fcst_p10', 'yield_fcst_p10_dt']))
        ]
        fcst = fcst[fcst['year_adj'] == year]
        fcst['monthL'] = fcst['month'].apply(lambda x: pd.to_datetime('2000-%02d-01' % x).strftime('%b'))
        fcst = fcst.merge(shape, left_on='fnid', right_on='FNID')
        year_lead_table = fcst[['year','lead','month','monthL']].drop_duplicates()
        year_lead_table = year_lead_table.sort_values(by=['year','month'], ascending=True).reset_index(drop=True)
        todate = datetime.today() 
        for i, (year, lead, month, monthL) in year_lead_table.iterrows():
            # continue if future forecasts
            if pd.to_datetime('%04d-%02d-15'%(year, month)) > todate: continue
            temp = fcst[(fcst['year'] == year) & (fcst['lead'] == lead)]
            # Forecast Map (percent to recent 10-year mean)
            fn_out = './viewer/figures/%s/%s/forecast/fcst_p10_map_%s_%04d_%02d' % (cps_string, model_name, model_name, year, month)
            footnote = "%s-%s at %04d.%02d" % (cps_string, model_name, year, month)
            fig = PlotForecastMap(temp[temp['variable'] == 'yield_fcst_p10'], geojson, country_name, footnote, ftype='forecast')
            fig.write_image(fn_out+'.png', scale=fig_scale)
            # Forecast Map (percent to recent 10-year mean) (detrend)
            fn_out = './viewer/figures/%s/%s/forecast/fcst_p10_dt_map_%s_%04d_%02d' % (cps_string, model_name, model_name, year, month)
            footnote = "%s-%s at %04d.%02d" % (cps_string, model_name, year, month)
            fig = PlotForecastMap(temp[temp['variable'] == 'yield_fcst_p10_dt'], geojson, country_name, footnote, ftype='forecast')
            fig.write_image(fn_out+'.png', scale=fig_scale)
        print('%s is processed.' % cpsm_string)

    # Figures and Tables
    command = "rsync -auzv --exclude=.DS_Store --exclude=.ipynb_checkpoints/ ./viewer/figures/ /home/chc-data-out/people/dlee/viewer/figures/"
    print('='*50)
    print('Copy figures to /home/chc-data-out/people/dlee/viewer/figures/')
    print(command)
    run_command(command)
    print('='*50)

    return