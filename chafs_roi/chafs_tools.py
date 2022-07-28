"""
Climate Hazards Center Agricultural Forecast System (CHAFS) Time-Series approach

File name: chafs_tools.py
Date revised: 06/05/2022
"""
__version__ = "0.0.1"
__author__ = "Donghoon Lee"
__maintainer__ = "Donghoon Lee"
__email__ = "donghoonlee@ucsb.edu"
from copy import deepcopy
import os, json, glob
from functools import reduce
from itertools import product, combinations, compress
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import xgboost as xgb
from xgboost import XGBRegressor
from scipy import stats, signal
from scipy.stats import norm, gaussian_kde, zscore, pearsonr
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict, TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from . import metrics as mt
from boruta import BorutaPy
import skopt
from skopt import BayesSearchCV 
from skopt.space import Real, Categorical, Integer
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
warnings.filterwarnings("ignore", category=UserWarning)


def month_date(year, forecast_end, lead):
    # No year (scores)
    if pd.isna(year): year = 2000
    fe = pd.to_datetime(str(int(year))+'-'+forecast_end)
    # No lead (observation)
    if pd.isna(lead): lead = 0
    ld = pd.DateOffset(months=lead)
    return fe-ld


def CPS_info(country_name, product_name, season_name):
    cps_name = country_name+'_'+product_name+'_'+season_name
    if cps_name == 'Somalia_Sorghum_Gu': 
        month = ['Feb','Mar','Apr','May','Jun','Jul']; 
    elif cps_name == 'Somalia_Sorghum_Deyr': 
        month = ['Sep','Oct','Nov','Dec','Jan','Feb']; 
    elif cps_name == 'Somalia_Maize_Gu': 
        month = ['Feb','Mar','Apr','May','Jun','Jul']; 
    elif cps_name == 'Somalia_Maize_Deyr': 
        month = ['Sep','Oct','Nov','Dec','Jan','Feb']; 
    elif cps_name == 'Malawi_Maize_Main': 
        month = ['Oct','Nov','Dec','Jan','Feb','Mar']; 
    elif cps_name == 'Kenya_Maize_Long': 
        month = ['Feb','Mar','Apr','May','Jun','Jul']; 
    elif cps_name == 'Kenya_Maize_Short': 
        month = ['Sep','Oct','Nov','Dec','Jan','Feb']; 
    elif cps_name == 'Burkina Faso_Maize_Main': 
        month = ['Apr','May','Jun','Jul','Aug','Sep']; 
    elif cps_name == 'Burkina Faso_Sorghum_Main': 
        month = ['Apr','May','Jun','Jul','Aug','Sep']; 
    else: raise ValueError('CPS is not defined.')
    return month


def nash_sutcliffe_efficiency(y_true, y_pred, y_mean):
    mse_pred = mean_squared_error(y_true, y_pred)
    mse_clim = mean_squared_error(y_true, np.ones(y_true.shape)*y_mean)
    if mse_clim == 0: mse_clim = 0.001
    return (1 - mse_pred/mse_clim)

def CategorizeLabel(x, boundaries, descending=False):
    # add min and max values of your data
    boundaries = sorted({-np.Inf, np.Inf} | set(boundaries))
    # "descending" means the lowest value category gets "zero"
    if descending:
        # For example, the drier binary category is "0".
        labels = range(len(boundaries) - 1)
    else:
        # For example, the drier binary category is "1".
        labels = range(len(boundaries) -2,-1,-1)
    # discretization
    discretized = pd.cut(x, bins=boundaries, labels=labels, 
                         right=False, include_lowest=True, ordered=False)
    return discretized.astype(int)

def CalSkillScores(obs, obs_mean_long, obs_mean_recent, sim):
    # CM4EW Classification
    # - Exceptional: >10% above-average 
    # - Favourable: 10% above-average to 10% below-average 
    # - Poor: 10-25% below-average
    # - Failure: >25% below-average
    boundaries3 = obs_mean_recent * np.array([0.75, 0.90])
    obs3 = CategorizeLabel(obs, boundaries3)
    obs3_names = ['Favourable', 'Poor', 'Failure']
    boundaries2 = obs_mean_recent * np.array([0.90])
    obs2 = CategorizeLabel(obs, boundaries2)
    obs2_names = ['Favourable', 'Poor']
    # Classification labeling
    sim3 = sim.apply(lambda x: CategorizeLabel(x, boundaries3), axis=0)
    sim2 = sim.apply(lambda x: CategorizeLabel(x, boundaries2), axis=0)
    # # Contingency tables
    # table2 = 

    # Skill scores
    # names = ['nse','mape','f1_score3','f1_score2','precision2','recall2','accuracy2']
    score = pd.DataFrame(index=sim.columns, columns=[]).rename_axis(index='lead', columns="score")
    # - Deterministic metrics
    score['nse'] = sim.apply(lambda x: nash_sutcliffe_efficiency(obs, x, obs_mean_long), axis=0)
    score['mape'] = sim.apply(lambda x: mean_absolute_percentage_error(obs, x)*100, axis=0)
    # - Categorical metrics
    score['f1_score3'] = sim3.apply(lambda x: f1_score(obs3, x, average='weighted'), axis=0)
    score['f1_score2'] = sim2.apply(lambda x: f1_score(obs2, x, average='weighted'), axis=0)
    score['precision2'] = sim2.apply(lambda x: precision_score(obs2, x, pos_label=0, average='binary'), axis=0)
    score['recall2'] = sim2.apply(lambda x: recall_score(obs2, x, pos_label=0, average='binary'), axis=0)
    score['accuracy2'] = sim2.apply(lambda x: accuracy_score(obs2, x), axis=0)
    return score


def CHAFS_Aggregate_CPSM(cpsme):
    # Parameters
    [country_name, product_name, season_name, model_name, exp_name] = cpsme
    leadmat = [6,5,4,3,2,1]
    recent_10yr = np.arange(2009,2019)
    recent_05yr = np.arange(2014,2019)
    obox = dict()

    # Check the numer of the simulated districsts
    fnids_info = pd.read_hdf('./data_in/fnids_info.hdf')
    fnids_info_sub = fnids_info[
        (fnids_info['country'] == country_name) & 
        (fnids_info['product'] == product_name) & 
        (fnids_info['season_name'] == season_name)
    ]
    fnids_orig = fnids_info_sub['fnid'].values
    fn_format = './data_out/chafs/chafs_{}_{}_{}_{}_{}.npz'
    exist = [os.path.exists(fn_format.format(fnid, product_name, season_name, model_name, exp_name)) for fnid in fnids_orig]
    fnids_modeled = fnids_orig[np.array(exist)]
    fnids_not_modeled = fnids_orig[~np.array(exist)]
    cpsme_string = '%s_%s_%s_%s_%s' % (country_name, product_name, season_name, model_name, exp_name)
    # print('%s: %d/%d districts.' % (cpsme_string, len(fnids_modeled), len(fnids_orig)))

    ### THIS WILL BE REPLACED BY SEPARATIVE REFORECAST SCRIPT ###
    # Initialize containers
    box_obs = {fnid: [] for fnid in fnids_modeled}
    box_y_dt = {fnid: [] for fnid in fnids_modeled}
    box_hcst = {fnid: [] for fnid in fnids_modeled}
    box_hcst_error = box_hcst.copy()
    box_hcst_perror = box_hcst.copy()
    box_hcst_low = box_hcst.copy()
    box_hcst_high = box_hcst.copy()
    box_fcst = box_hcst.copy()
    box_fcst_error = box_hcst.copy()
    box_fcst_perror = box_hcst.copy()
    box_fcst_low = box_hcst.copy()
    box_fcst_high = box_hcst.copy()
    box_rcst = box_hcst.copy()
    box_rcst_dt = box_hcst.copy()
    box_rcst_low = box_hcst.copy()
    box_rcst_high = box_hcst.copy()
    box_importance = box_hcst.copy()
    box_score = box_hcst.copy()

    # Read leadmat
    fn = fn_format.format(fnids_modeled[0], product_name, season_name, model_name, exp_name)
    box = np.load(fn, allow_pickle=True)['obox'].tolist()
    leadmat = box['leadmat']
    forecast_end = box['forecast_end']
    fe = pd.to_datetime('2000'+'-'+forecast_end)
    leadmat_month = [fe - pd.DateOffset(months=l) for l in leadmat]
    leadmat_month = [m.month for m in leadmat_month]
    
    # Retrieve forecast results
    for fnid in fnids_modeled:
        fn = fn_format.format(fnid, product_name, season_name, model_name, exp_name)
        if not os.path.exists(fn): continue

        # Load result file
        box = np.load(fn, allow_pickle=True)['obox'].tolist()
        box_obs[fnid] = obs = box['y']
        box_y_dt[fnid] = box['y_dt']
        box_hcst[fnid] = hcst = box['hcst']
        box_hcst_error[fnid] = hcst.subtract(obs[hcst.index], axis=0)
        box_hcst_perror[fnid] = hcst.apply(lambda x: (x-obs)/obs*100)
        # box_hcst_perror[fnid] = hcst.subtract(obs[hcst.index], axis=0)
        # box_hcst_low[fnid] = box['hcst_low']
        # box_hcst_high[fnid] = box['hcst_high']
        box_fcst[fnid] = fcst = box['fcst']
        box_fcst_error[fnid] = fcst.subtract(obs[fcst.index], axis=0)
        box_fcst_perror[fnid] = fcst.apply(lambda x: (x-obs[fcst.index])/obs[fcst.index]*100)
        # box_fcst_low[fnid] = box['fcst_low']
        # box_fcst_high[fnid] = box['fcst_high']
        box_rcst[fnid] = box['rcst']
        box_rcst_dt[fnid] = box['rcst_dt']
        # box_rcst_low[fnid] = box['rcst_low']
        # box_rcst_high[fnid] = box['rcst_high']
        box_importance[fnid] = box['importance']
    #############################################################

    stack_obs = []
    stack_value = []
    stack_score = []
    stack_importance = []
    for fnid in fnids_modeled:
        fn = fn_format.format(fnid, product_name, season_name, model_name, exp_name)
        if not os.path.exists(fn): continue

        obs = box_obs[fnid]
        y_dt = box_y_dt[fnid]
        hcst = box_hcst[fnid]
        hcst_error = box_hcst_error[fnid]
        hcst_perror = box_hcst_perror[fnid]
        # hcst_low = box_hcst_low[fnid]
        # hcst_high = box_hcst_high[fnid]
        fcst = box_fcst[fnid]
        fcst_error = box_fcst_error[fnid]
        fcst_perror = box_fcst_perror[fnid]
        # fcst_low = box_fcst_low[fnid]
        # fcst_high = box_fcst_high[fnid]
        rcst = box_rcst[fnid]
        rcst_dt = box_rcst_dt[fnid]
        # rcst_low = box_rcst_low[fnid]
        # rcst_high = box_rcst_high[fnid]
        importance = box_importance[fnid]

        # Observation
        df_obs = obs.reset_index()
        df_obs['year'] = df_obs['date'].dt.year
        df_obs[['fnid','lead','pred']] = fnid, np.nan, 'obs'
        stack_obs.append(df_obs)
        mean_long = df_obs.loc[df_obs['year'].isin(range(0,2019)), 'value'].mean()
        mean_last10 = df_obs.loc[df_obs['year'].isin(range(2009,2019)), 'value'].mean()
        mean_last10_dt = y_dt[y_dt.index.year.isin(range(2009,2019))].mean()

        # Forecast values
        # - Hindcast
        df_hcst = hcst.melt(ignore_index=False).reset_index()
        df_hcst['year'] = df_hcst['date'].dt.year
        df_hcst[['fnid', 'pred']] = fnid, 'hcst'
        df_hcst['variable'] = 'fcst'
        # - Forecast
        df_fcst = fcst.melt(ignore_index=False).reset_index()
        df_fcst['year'] = df_fcst['date'].dt.year
        df_fcst[['fnid', 'pred']] = fnid, 'fcst'
        df_fcst['variable'] = 'fcst'
        # - Reconstruct
        df_rcst = rcst.melt(ignore_index=False).reset_index()
        df_rcst['year'] = df_rcst['date'].dt.year
        df_rcst[['fnid', 'pred']] = fnid, 'rcst'
        df_rcst['variable'] = 'fcst'
        # Merge and Append
        df = pd.concat([df_rcst, df_hcst, df_fcst], axis=0)
        stack_value.append(df)
        # - Percentage to long-term mean 
        df_prct_long = df.copy()
        df_prct_long['value'] = df_prct_long['value']/mean_long*100
        df_prct_long['variable'] = 'fcst_p30'
        # - Percentage to last 10-year mean
        df_prct_last10 = df.copy()
        df_prct_last10['value'] = df_prct_last10['value']/mean_last10*100
        df_prct_last10['variable'] = 'fcst_p10'
        # - Percentage to last 10-year mean (detrended)
        df_prct_last10_dt = rcst_dt.melt(ignore_index=False).reset_index()
        df_prct_last10_dt['year'] = df_prct_last10_dt['date'].dt.year
        df_prct_last10_dt[['fnid', 'pred']] = fnid, 'rcst'
        df_prct_last10_dt['value'] = df_prct_last10_dt['value']/mean_last10_dt*100
        df_prct_last10_dt['variable'] = 'fcst_p10_dt'
        # Merge and Append
        stack_value.append(pd.concat([df_prct_long, df_prct_last10, df_prct_last10_dt], axis=0))
        
        
        ### Forecast confidence intervals


        # Forecast errors
        # - Hindcast
        df_hcst = hcst_error.melt(ignore_index=False).reset_index()
        df_hcst['year'] = df_hcst['date'].dt.year
        df_hcst[['fnid', 'pred']] = fnid, 'hcst'
        df_hcst['variable'] = 'fcst_error'
        # - Forecast
        df_fcst = fcst_error.melt(ignore_index=False).reset_index()
        df_fcst['year'] = df_fcst['date'].dt.year
        df_fcst[['fnid', 'pred']] = fnid, 'fcst'
        df_fcst['variable'] = 'fcst_error'
        # Merge and Append
        stack_value.append(pd.concat([df_hcst, df_fcst], axis=0))
        
        # Forecast percentage errors
        # - Hindcast
        df_hcst = hcst_perror.melt(ignore_index=False).reset_index()
        df_hcst['year'] = df_hcst['date'].dt.year
        df_hcst[['fnid', 'pred']] = fnid, 'hcst'
        df_hcst['variable'] = 'fcst_perror'
        # - Forecast
        df_fcst = fcst_perror.melt(ignore_index=False).reset_index()
        df_fcst['year'] = df_fcst['date'].dt.year
        df_fcst[['fnid', 'pred']] = fnid, 'fcst'
        df_fcst['variable'] = 'fcst_perror'
        # Merge and Append
        stack_value.append(pd.concat([df_hcst, df_fcst], axis=0))

        # Skill scores
        score = CalSkillScores(obs[hcst.index], obs.mean(), obs[fcst.index].mean(), hcst)
        temp = score.T.melt(ignore_index=False).reset_index()
        temp[['fnid', 'pred']] = fnid, 'hcst'
        stack_score.append(temp)
        score = CalSkillScores(obs[fcst.index], obs.mean(), obs[fcst.index].mean(), fcst)
        temp = score.T.melt(ignore_index=False).reset_index()
        temp[['fnid', 'pred']] = fnid, 'fcst'
        stack_score.append(temp)

        # Stacking importance
        temp = importance.T.melt(ignore_index=False).reset_index()
        temp['fnid'] = fnid
        stack_importance.append(temp)

    # Merge containers
    stack_value = pd.concat(stack_value, axis=0).reset_index(drop=True)
    stack_value = stack_value.merge(fnids_info_sub, left_on='fnid', right_on='fnid')
    stack_value = stack_value[['fnid','country','name','product','season_name','harvest_end','year','lead','pred','value','variable']]
    stack_score = pd.concat(stack_score, axis=0).reset_index(drop=True)
    stack_score = stack_score.merge(fnids_info_sub, left_on='fnid', right_on='fnid')
    stack_score = stack_score[['fnid','country','name','product','season_name','harvest_end','lead','score','pred','value']]
    stack_importance = pd.concat(stack_importance, axis=0).reset_index(drop=True)
    stack_importance = stack_importance.merge(fnids_info_sub, left_on='fnid', right_on='fnid')
    stack_importance = stack_importance[['fnid','country','name','product','season_name','harvest_end','lead','eoname','value']]

    # Skill calculation by time and district ----------------- #
    # Tables for skill calculation
    obs_all = pd.concat(stack_obs, axis=0).pivot_table(index='fnid', columns='year', values='value')
    year_all = np.sort(obs_all.columns)
    obs_long_avg = obs_all.mean(1)
    obs_recent_10yr = obs_all.loc[:,obs_all.columns.isin(recent_10yr)]
    obs_recent_05yr = obs_all.loc[:,obs_all.columns.isin(recent_05yr)]
    assert obs_recent_10yr.isna().all(axis=1).sum() == 0
    assert obs_recent_05yr.isna().all(axis=1).sum() == 0
    obs_recent_10yr_avg = obs_recent_10yr.mean(1)
    obs_recent_05yr_avg = obs_recent_05yr.mean(1)
    sub = stack_value.copy()
    hcst_all = sub[(sub['variable']=='fcst') & (sub['pred']=='hcst')].pivot_table(index='fnid',columns=['lead','year'],values='value')
    fcst_all = sub[(sub['variable']=='fcst') & (sub['pred']=='fcst')].pivot_table(index='fnid',columns=['lead','year'],values='value')
    rcst_all = sub[(sub['variable']=='fcst') & (sub['pred']=='rcst')].pivot_table(index='fnid',columns=['lead','year'],values='value')

    # Current forecast
    current_value = rcst_all.loc[:,pd.IndexSlice[:,2022]]
    current_last10 = current_value.divide(obs_recent_10yr_avg, axis=0)*100
    current_last05 = current_value.divide(obs_recent_05yr_avg, axis=0)*100
    current_value = current_value.T.stack().rename('value').reset_index()
    current_value['type'] = 'current_value'
    current_last10 = current_last10.T.stack().rename('value').reset_index()
    current_last10['type'] = 'current_last10'
    current_last05 = current_last05.T.stack().rename('value').reset_index()
    current_last05['type'] = 'current_last05'
    current = pd.concat([current_value, current_last10, current_last05], axis=0).reset_index(drop=True)
    current['date'] = np.vectorize(month_date)(current['year'], forecast_end, current['lead'])
    current['month'] = current['date'].dt.month

    # Skill Scores per Time
    mdx = pd.MultiIndex.from_product([leadmat,['nse','mape']], names=['lead','variable'])
    table_score_time = pd.DataFrame(index=year_all, columns=mdx, dtype=np.float32).rename_axis(index='year')
    error_all = hcst_all.copy()
    perror_all = hcst_all.copy()
    for lead in leadmat:
        hcst_lead = hcst_all[lead]
        hcst_error = (hcst_lead - obs_all)
        hcst_perror = (hcst_lead - obs_all)/obs_all*100
        error_all.loc[:,pd.IndexSlice[lead,:]] = hcst_error.values
        perror_all.loc[:,pd.IndexSlice[lead,:]] = hcst_perror.values
        table_score_time.loc[:,pd.IndexSlice[lead,'mape']] =  np.abs(hcst_perror).mean()
        for year in year_all:
            hcst_year = hcst_all.loc[:,pd.IndexSlice[lead,year]].dropna()
            obs_year = obs_all[year].dropna()
            table_score_time.loc[year,pd.IndexSlice[lead,'nse']] = nash_sutcliffe_efficiency(obs_year, hcst_year, obs_year.mean())
    stack_score_time = table_score_time.T.stack().rename('value').reset_index()

    # Skill Scores per District
    mdx = pd.MultiIndex.from_product([leadmat,['nse','mape'],['hcst','fcst']], names=['lead','variable','pred'])
    table_score_dist = pd.DataFrame(index=fnids_modeled, columns=mdx, dtype=np.float32).rename_axis(index='fnid')
    for fnid, lead in product(fnids_modeled, leadmat):
        # fnid, lead = 'SO1990A21201', 6
        obs = obs_all.loc[fnid,:].dropna()
        hcst = hcst_all.loc[fnid,lead].dropna()
        fcst = fcst_all.loc[fnid,lead].dropna()
        rcst = rcst_all.loc[fnid,lead].dropna()
        assert all(obs.index == hcst.index)
        obs_hcst = obs.copy()
        obs_fcst = obs[fcst.index]
        hcst_error = hcst - obs_hcst
        fcst_error = fcst - obs_fcst
        hcst_perror = hcst_error/obs_hcst*100
        fcst_perror = fcst_error/obs_fcst*100
        # Calculating district Skill Scores
        table_score_dist.loc[fnid,pd.IndexSlice[lead,'mape','hcst']] = np.abs(hcst_perror).mean()
        table_score_dist.loc[fnid,pd.IndexSlice[lead,'mape','fcst']] = np.abs(fcst_perror).mean()
        table_score_dist.loc[fnid,pd.IndexSlice[lead,'nse','hcst']] = nash_sutcliffe_efficiency(obs, hcst, obs.mean())
        table_score_dist.loc[fnid,pd.IndexSlice[lead,'nse','fcst']] = nash_sutcliffe_efficiency(obs[fcst.index], fcst, obs.mean())
    stack_score_dist = table_score_dist.T.stack().rename('value').reset_index()

    # Quick Score
    temp = stack_score_dist
    temp1 = temp[temp['pred'] == 'hcst'].groupby(['variable','lead'])['value'].mean().rename('value').reset_index()
    temp1['type'] = 'dist'
    temp = stack_score_time
    temp2 = temp.groupby(['variable','lead'])['value'].mean().rename('value').reset_index()
    temp2['type'] = 'time'
    score_quick = pd.concat([temp1,temp2], axis=0).reset_index().pivot_table(index='lead',columns=['type','variable'],values='value')
    score_quick = score_quick.sort_index(ascending=False)
    
    
    # Forecast results
    obox['forecast_end'] = forecast_end
    obox['leadmat'] = leadmat
    obox['leadmat_month'] = leadmat_month
    obox['box_obs'] = box_obs
    obox['box_y_dt'] = box_y_dt
    obox['box_hcst'] = box_hcst
    obox['box_hcst_error'] = box_hcst_error
    obox['box_hcst_low'] = box_hcst_low
    obox['box_hcst_high'] = box_hcst_high
    obox['box_fcst'] = box_fcst
    obox['box_fcst_error'] = box_fcst_error
    obox['box_fcst_low'] = box_fcst_low
    obox['box_fcst_high'] = box_fcst_high
    obox['box_rcst'] = box_rcst
    obox['box_rcst_dt'] = box_rcst_dt
    obox['box_rcst_low'] = box_rcst_low
    obox['box_rcst_high'] = box_rcst_high
    obox['box_importance'] = box_importance
    obox['stack_score'] = stack_score
    obox['stack_value'] = stack_value
    obox['stack_importance'] = stack_importance
    obox['stack_score_dist'] = stack_score_dist
    obox['stack_score_time'] = stack_score_time
    obox['score_quick'] = score_quick
    obox['current'] = current
    
    return obox