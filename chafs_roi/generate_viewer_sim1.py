"""
Climate Hazards Center Agricultural Forecast System (CHAFS) Time-Series approach

File name: chafs_time_v2.py
Date revised: 01/11/2023
"""
__version__ = "0.2.0"
__author__ = "Donghoon Lee"
__maintainer__ = "Donghoon Lee"
__email__ = "donghoonlee@ucsb.edu"
from copy import deepcopy
import os, json, glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import reduce
from itertools import product
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from xgboost import XGBRegressor
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
warnings.filterwarnings("ignore", category=UserWarning)


def CropDataControl(fnid, product_name, season_name, indicator_name):
    # Select crop data ------------------------------------ #
    crop = pd.read_hdf('./data_in/%s_crop_%s_%s.hdf' % (fnid, product_name, season_name))
    crop = crop.loc[:,pd.IndexSlice[:,:,:,:,:,:,:,indicator_name]].droplevel(-1,axis=1)
    # Get basic information of admin unit
    fnid, country, name, product_name, season_name, growing_month, harvest_month = list(crop.columns.values[0])
    # ----------------------------------------------------- #

    # Forecasting ending month (ending month) ------------- #
    cps_name = country+'_'+product_name+'_'+season_name
    if cps_name == 'Somalia_Maize_Deyr': forecast_end = '03-01'
    elif cps_name == 'Somalia_Maize_Gu': forecast_end = '08-01'
    elif cps_name == 'Somalia_Sorghum_Deyr': forecast_end = '03-01'
    elif cps_name == 'Somalia_Sorghum_Gu': forecast_end = '08-01'
    elif cps_name == 'Kenya_Maize_Long' : forecast_end = '08-01'
    elif cps_name == 'Kenya_Maize_Short' : forecast_end = '03-01'
    elif cps_name == 'Malawi_Maize_Main' : forecast_end = '04-01'
    elif cps_name == 'Burkina Faso_Maize_Main': forecast_end = '10-01'
    elif cps_name == 'Burkina Faso_Sorghum_Main': forecast_end = '10-01'
    else: raise ValueError('country, product_name, or season_name is not defined.')
    tdx_forecast_end = pd.to_datetime(['%4d-%02s-01' % (year, forecast_end[:2]) for year in crop.index])
    info = (fnid, country, name, product_name, season_name, growing_month, harvest_month, forecast_end)
    # ----------------------------------------------------- #
    
    # Initial Cleaning ------------------------------------ #
    y = crop.copy()
    y[y == 0] = np.nan   # Remove zero value
    # Convert DataFrame to Series
    y = pd.Series(index = tdx_forecast_end, data = y.values.flatten()).rename_axis(index='date').rename('value')
    # Drop missing years
    tdx = y[y.notna()].index
    y = y[tdx]
    # ----------------------------------------------------- #
    return info, y

def _Detrend(sr):
    sr = sr.copy()
    sr.index = pd.PeriodIndex(sr.index, freq='Y')
    sr = sr.reindex(pd.period_range(sr.index[0], sr.index[-1], freq='Y'), fill_value=np.nan)
    sr.index = sr.index.strftime('%Y').astype(int)
    nni = sr.notna()
    return LinearRegression().fit(sr[nni].index.values[:,None],sr[nni])

def EODataControl(fnid, method_resample):
    df = pd.read_hdf('./data_in/%s_pred.hdf' % fnid)
    # Fill missing values of predictors with climatological means
    df = df.groupby(df.index.strftime('%m-%d')).transform(lambda x: x.fillna(x.mean()))
    # Monthly aggregation
    df = df.resample('1M').agg(method_resample)
    df.index = df.index.map(lambda x: x.replace(day=1))
    return df

def CheckLeadPred(lead_pred):
    for k,v in lead_pred.items():
        list_lead = [v.split('.')[1] for v in v]
        # The lead digit should be four or two.
        list_lead_len = [len(l) for l in list_lead]
        assert np.isin(list_lead_len, [2,4]).all()
        list_lead_all = []
        for l in list_lead:
            if len(l) == 4:
                list_lead_all.append(l[:2])
                assert int(l[:2]) > int(l[-2:])
            list_lead_all.append(l[-2:])
        list_lead_all = np.array([int(l) for l in list_lead_all])
        assert (list_lead_all >= k).all()  # Should be greater than lead time
    return True

def subsequences(iterable, length):
    return [iterable[i: i + length] for i in range(len(iterable) - length + 1)]

def CombSerialLead(lead_srl):
    comb = [subsequences(lead_srl,l) for l in range(1,len(lead_srl)+1)]
    comb = [item for sublist in comb for item in sublist]
    return comb

def GenerateSeriesLeadPredTable(df, tdx, leadcomb, flag_reduce=True):
    var_agg_mean = ['tavg','tmax','tmin','ndvi']
    if flag_reduce:
        # Reduce features
        df = df.loc[:, df.columns.isin(['pacu','pdry','eacu','aacu','tavg','tmax','tmin','ndvi'])]
    table = []
    for l in leadcomb:   
        table_sub = []
        for j in l:
            temp_sub = df.iloc[df.index.get_indexer(tdx) - j]
            temp_sub.index = tdx
            table_sub.append(temp_sub)
        temp = reduce(lambda x, y: x.add(y, fill_value=0), table_sub)
        temp.loc[:,temp.columns.isin(var_agg_mean)] = temp.loc[:,temp.columns.isin(var_agg_mean)]/len(table_sub)
        if len(l) == 1:
            temp.columns = ['%s.%02d' % (name,l[0]) for name in temp.columns]
        else:
            temp.columns = ['%s.%02d%02d' % (name,l[0],l[-1]) for name in temp.columns]
        table.append(temp)
    table = pd.concat(table,axis=1)
    return table

def GeneratePredTable(df, tdx, lead_srl, method_resample):
    table = []
    for l in lead_srl:
        temp = df.iloc[df.index.get_indexer(tdx)-l]
        temp.index = tdx
        temp.columns = ['%s.%02d' % (name,l) for name in temp.columns]
        table.append(temp)
    X = pd.concat(table, axis=1).rename_axis(index='date', columns="eoname")
    # Accumulate monthly values over lead-time 
    for name in method_resample.keys():
        target = [col for col in X.columns if name in col]
        X[str(name)] = X[target].aggregate(method_resample[name],axis=1)
    X = X[method_resample.keys()]
    return X

def HyperparameterTuning(X, y, regr, model_name):
    if model_name == 'XGB':
        search_spaces={
            'max_depth': Integer(1, 20),
            # 'min_child_weight': Integer(0, 20),
            'max_depth': Integer(0, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'colsample_bylevel': Real(0.5, 1.0),
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01,0.2, 'log-uniform'),
            'reg_lambda': Real(1e-5,100,'log-uniform'),
            'reg_alpha': Real(1e-5,100,'log-uniform'),
        }
        bayes_cv_tuner = BayesSearchCV(
            estimator = regr,
            search_spaces = search_spaces,
            scoring='neg_mean_squared_error',
            n_iter = 10,
            # n_iter = 30,
            cv = 5,
            n_jobs = 8,
            verbose = 0,
            refit = True,
            random_state=42
        )
        # Multi-steps
        # https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
        # Fit the model
        # Callback: https://www.kaggle.com/code/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769/notebook
        result = bayes_cv_tuner.fit(X, y)
        all_models = pd.DataFrame(result.cv_results_)    
        best_params = pd.Series(result.best_params_)
    
    return result

def CHAFS_TIME(bp):
    # Validation of parameters
    fnid = bp['fnid']
    product_name = bp['product_name']
    season_name = bp['season_name']
    indicator_name = bp['indicator_name']
    model_name = bp['model_name']
    lead_pred = bp['lead_pred']
    lead_pred = dict(sorted(lead_pred.items(),reverse=True))
    assert CheckLeadPred(lead_pred) == True
    pred_name = [v[0] for k,v in lead_pred.items()]
    pred_name_unique = list(np.unique([t.split('.')[0] for t in pred_name]))
    isPower = bp['isPower']
    isTrend = bp['isTrend']
    note = bp['note']

    # Initialization
    leadmat = list(lead_pred.keys())
    obox = dict()

    # Crop data control
    info, y = CropDataControl(fnid, product_name, season_name, indicator_name)

    # y = y[y.index.year >= 2001]

    y_raw, tdx = y.copy(), y.index
    if indicator_name == 'production': 
        # Avoid error for box-cox transformation
        y = np.ceil(y).astype(int)
    elif indicator_name == 'yield':
        y = y.round(3)
    fnid, country, name, product_name, season_name, growing_month, harvest_month, forecast_end = info
    obox = dict({'FNID': fnid, 'country':country, 'name':name, 'product_name': product_name,
                 'season_name':season_name, 'indicator_name':indicator_name, 
                 'growing_month':growing_month ,'harvest_month':harvest_month, 'forecast_end':forecast_end, 
                 'status':0, 'isTrend': isTrend, 'isPower': isPower,'model':model_name, 'note':note
                })
    # - Month of leadmat
    fe = pd.to_datetime('2000'+'-'+forecast_end)
    leadmat_month = [fe - pd.DateOffset(months=l) for l in leadmat]
    leadmat_month = [m.month for m in leadmat_month]

    # Power Transformation
    if isPower:
        pt = PowerTransformer(method='box-cox', standardize=False, copy=True)
        pt.fit(y.values[:,None])
        y = pd.Series(index=y.index, data=pt.transform(y.values[:,None]).squeeze())
        # y = np.log(y)

    # First Detrending
    if isTrend:
        trend = _Detrend(y)
        y = y - trend.predict(y.index.year.values[:,None]) + y.mean()
        obox.update({'trend':(trend.coef_[0], trend.intercept_)})
    # EO data control
    # - Load the variables only included in "method_resample"
    method_resample = {'pacu':'sum', 'pdry':'sum', 'eacu':'sum', 'tavg':'mean', 'ndvi':'max'}
    df = EODataControl(fnid, method_resample)

    # Forecast Time Table (FTT)
    predyear = np.array(df.index.year.unique())
    # - Add a future buffer year. This will ignored if irrelavant.
    predyear = np.append(predyear, predyear[-1]+1)
    datelead = df.index[df.index.get_loc(y.index[0]) - np.array(leadmat)]
    datelead_month = datelead.strftime('%m-%d')
    temp = [(datelead + pd.DateOffset(years=diff)).values for diff in (predyear - datelead.year.max())]
    temp = pd.to_datetime(np.concatenate(temp))
    avaldata = df.iloc[:,0].to_frame().reindex(temp).values.reshape([len(predyear), len(datelead_month)])
    ftt = pd.DataFrame(index=pd.to_datetime(['%d-%s-01'%(yr, forecast_end[:2]) for yr in predyear]), 
                       columns=leadmat, data=avaldata).notna()
    # - Remove years that forecasts are entirely unavailable
    ftt = ftt[ftt.sum(1) > 0]

    # Split data for model devlopment and testing
    # Here we do out-of-sample prediction for each year in model validation period.
    # tdx_train, tdx_test = tdx[tdx.year < 2011], tdx[tdx.year >= 2011]
    tdx_train, _ = train_test_split(tdx[tdx.year < 2019], test_size=0.30, shuffle=False, random_state=0)
    tdx_test = tdx[~tdx.isin(tdx_train)]
    tdx_test_start = tdx.get_loc(tdx_test[0])

    # Regression models
    regressors = {
        'ET': Pipeline([('regr', ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            criterion='squared_error',
            random_state=42
        ))]),
        'XGB': Pipeline([('regr', XGBRegressor(
            objective='reg:squarederror',
            booster='gbtree',
            random_state=42,
            seed=42,
            n_jobs = 8
        ))]),
        'MLR': Pipeline([('scaler', StandardScaler()),
                         ('regr', LinearRegression())])
    }

    # Containers of result tables
    lead_hcst = pd.DataFrame(index=tdx, columns=leadmat, dtype=np.float32).rename_axis(index='date', columns="lead")
    lead_hcst_esm = {}
    lead_fcst = pd.DataFrame(index=tdx_test, columns=leadmat, dtype=np.float32).rename_axis(index='date', columns="lead")
    lead_fcst_esm = {}
    lead_rcst = pd.DataFrame(index=ftt.index, columns=leadmat, dtype=np.float32).rename_axis(index='date', columns="lead")
    lead_rcst_esm = {}
    lead_model_fitted = {}
    importance = []

    # Main forecasting algorithms
    # Start Loop --------------------------- #
    for lead in leadmat:
        # Predictor contorl -------------------- #
        # Generate monthly lagged features
        lead_srl = leadmat[:leadmat.index(lead)+1]
        tdx_all = ftt.index[ftt[lead]]
        leadcomb = CombSerialLead(lead_srl)
        X_candidate = GenerateSeriesLeadPredTable(df, tdx_all, leadcomb, flag_reduce=True)
        X_ftt = X_candidate.loc[:,lead_pred[lead]]
        assert np.isin(lead_pred[lead], X_candidate.columns).all() # Validation
        X = X_ftt.loc[tdx]
        if model_name in ['XGB']:
            # Model setup -------------------------- #
            regr = sklearn.base.clone(regressors[model_name]['regr'])

            # Hyperparameter tuning ---------------- #
            tuned = HyperparameterTuning(X, y, regr, model_name)
            feature_importance = tuned.best_estimator_.feature_importances_
            importance.append(pd.DataFrame(index=[lead], columns=X.columns, data=feature_importance[None,:]))
            params = tuned.best_params_

            # Hindcast available years ------------- #
            # regr = sklearn.base.clone(tuned.best_estimator_)
            regr = sklearn.base.clone(regressors[model_name]['regr'])
            regr = regr.set_params(**params)
            hcst = np.zeros(tdx.shape)
            hcst_esm = np.zeros([len(tdx), regr.n_estimators])
            for it, (train_index, test_index) in enumerate(LeaveOneOut().split(tdx)):
                X_train_loo, X_test_loo = X.iloc[train_index], X.iloc[test_index]
                y_train_loo, y_test_loo = y.iloc[train_index], y.iloc[test_index]
                regr.fit(X_train_loo, y_train_loo)
                hcst[it] = regr.predict(X_test_loo)
            lead_hcst.loc[:, lead] = hcst

            # Forecast out-of-sample years --------- #
            # regr = sklearn.base.clone(tuned.best_estimator_)
            regr = sklearn.base.clone(regressors[model_name]['regr'])
            regr = regr.set_params(**params)
            fcst = np.zeros(tdx_test.shape)
            tscv = TimeSeriesSplit(n_splits=len(tdx)-1)
            for it, (train_index, test_index) in enumerate(tscv.split(tdx)):
                if test_index < tdx_test_start: continue
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                regr.fit(X_train, y_train)
                fcst[it-tdx_test_start+1] = regr.predict(X_test)
            lead_fcst.loc[:, lead] = fcst

            # Reconstruct all years ---------------- #
            # regr = sklearn.base.clone(tuned.best_estimator_)
            regr = sklearn.base.clone(regressors[model_name]['regr'])
            regr = regr.set_params(**params)
            regr.fit(X,y)
            # Reconstruct all available time
            rcst = regr.predict(X_ftt)
            lead_rcst.loc[tdx_all, lead] = rcst
            # Store the trained model
            lead_model_fitted[lead] = dict(regr=deepcopy(regr))

        elif model_name in ['MLR']:
            # Model setup -------------------------- #
            regr = LinearRegression()
            # Standardization for a reference period (1985-2014)
            X_ftt_ref = X_ftt.loc[X_ftt.index.year.isin(np.arange(1985,2015)),:]
            scaler_X = StandardScaler().fit(X_ftt_ref)
            scaler_y = StandardScaler().fit(y.values[:,None])

            # Hindcast available years ------------- #
            hcst = np.zeros(tdx.shape)
            for it, (train_index, test_index) in enumerate(LeaveOneOut().split(tdx)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                regr.fit(scaler_X.transform(X_train), scaler_y.transform(y_train.values[:,None]))
                hcst[it] = regr.predict(scaler_X.transform(X_test))
            lead_hcst.loc[:, lead] = scaler_y.inverse_transform(hcst[None,:]).squeeze()

            # Forecast out-of-sample years --------- #
            fcst = np.zeros(tdx_test.shape)
            tscv = TimeSeriesSplit(n_splits=len(tdx)-1)
            for it, (train_index, test_index) in enumerate(tscv.split(tdx)):
                if test_index < tdx_test_start: continue
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                regr.fit(scaler_X.transform(X_train), scaler_y.transform(y_train.values[:,None]))
                fcst[it-tdx_test_start+1] = regr.predict(scaler_X.transform(X_test))
            lead_fcst.loc[:, lead] = scaler_y.inverse_transform(fcst[None,:]).squeeze()

            # Reconstruct all years ---------------- #
            regr.fit(scaler_X.transform(X), scaler_y.transform(y.values[:,None]))
            importance.append(pd.DataFrame(index=[lead], columns=X.columns, data=regr.coef_))
            # Reconstruct all available time
            rcst = regr.predict(scaler_X.transform(X_ftt))
            lead_rcst.loc[tdx_all, lead] = scaler_y.inverse_transform(rcst).squeeze()
            # Store the trained model
            lead_model_fitted[lead] = dict(
                scaler_X = scaler_X, 
                scaler_y = scaler_y, 
                regr = deepcopy(regr)
            )

        else:
            raise ValueError('%s is not correct as a "model_name".' % model_name)

    # Out-Of-Loop -------------------------- #
    lead_rcst_dt = lead_rcst.copy()
    if isTrend:
        # Retrending
        lead_hcst = lead_hcst + trend.predict(np.array(tdx.year)[:,None])[:,None] - y.mean()
        lead_fcst = lead_fcst + trend.predict(np.array(tdx_test.year)[:,None])[:,None] - y.mean()
        lead_rcst = lead_rcst + trend.predict(np.array(ftt.index.year)[:,None])[:,None] - y.mean()

    if isPower:
        lead_hcst[lead_hcst > y.max()], lead_hcst[lead_hcst < y.min()] = y.max(), y.min()
        lead_fcst[lead_fcst > y.max()], lead_fcst[lead_fcst < y.min()] = y.max(), y.min()
        lead_rcst[lead_rcst > y.max()], lead_rcst[lead_rcst < y.min()] = y.max(), y.min()
        lead_hcst = lead_hcst.apply(lambda x: np.squeeze(pt.inverse_transform(x.values[:,None])))
        lead_fcst = lead_fcst.apply(lambda x: np.squeeze(pt.inverse_transform(x.values[:,None])))
        lead_rcst = lead_rcst.apply(lambda x: np.squeeze(pt.inverse_transform(x.values[:,None])))
        # lead_hcst = np.exp(lead_hcst)
        # lead_fcst = np.exp(lead_fcst)
        # lead_rcst = np.exp(lead_rcst)

    ###### TEMP ########
    lead_hcst[lead_hcst.isna()] = y_raw.mean()
    lead_fcst[lead_fcst.isna()] = y_raw.mean()
    ####################

    # Aggregation outputs
    importance = pd.concat(importance,axis=0).rename_axis(index='lead', columns="eoname")

    # Scores for internal evaluation. A detailed skill assessment will be performed later.
    score = pd.DataFrame(index=leadmat, columns=[]).rename_axis(index='lead', columns="score")
    score['r2_hcst'] = lead_hcst.apply(lambda x: r2_score(y_raw.loc[tdx], x), axis=0)
    score['r2_fcst'] = lead_fcst.apply(lambda x: r2_score(y_raw.loc[tdx_test], x), axis=0)

    # Save to output container
    obox.update({
        'leadmat': leadmat,
        'leadmat_month': leadmat_month,
        'y': y_raw,
        'y_dt': y,
        'hcst': lead_hcst,
        'fcst': lead_fcst,
        'rcst': lead_rcst,
        'rcst_dt': lead_rcst_dt,
        'importance': importance,
        'model_fitted': lead_model_fitted,
        'score': score
    })

    return obox


def LeadPredPocket(name_lead_pred):
    if name_lead_pred == 'ACUM_P':
        lead_pred = {5:['pacu.05'],4:['pacu.0504'],3:['pacu.0503'],2:['pacu.0502'],1:['pacu.0501']}
    elif name_lead_pred == 'ACUM_ALL':
        lead_pred = {
            5:['pacu.05','eacu.05','tavg.05','ndvi.05'],
            4:['pacu.0504','eacu.0504','tavg.0504','ndvi.0504'],
            3:['pacu.0503','eacu.0503','tavg.0503','ndvi.0503'],
            2:['pacu.0502','eacu.0502','tavg.0502','ndvi.0502'],
            1:['pacu.0501','eacu.0501','tavg.0501','ndvi.0501'],
        }
    elif name_lead_pred == 'INDV_P':
        lead_pred = {5:['pacu.05'],4:['pacu.05','pacu.04'],3:['pacu.05','pacu.04','pacu.03'],2:['pacu.05','pacu.04','pacu.03','pacu.02'],1:['pacu.05','pacu.04','pacu.03','pacu.02','pacu.01']}
    elif name_lead_pred == 'MIX1_P':
        lead_pred = {5:['pacu.05'],4:['pacu.0504'],3:['pacu.0503'],2:['pacu.0503','pacu.02'],1:['pacu.0503','pacu.02','pacu.01']}
    else: 
        lead_pred = {}
        ValueError('Invalid name for name_lead_pred.')
    return lead_pred

def ExperimentSettingPocket(name):
    isPower, isTrend = False, False
    # Crop Indicator
    if name[0] == 'P': indicator_name = 'production'
    elif name[0] == 'Y': indicator_name = 'yield'
    else: ValueError('Invalid crop_indicator.')
    # Power Transformation
    if name[1] == 'T': isPower = True
    elif name[1] == 'F': isPower = False
    else: ValueError('Invalid isPower.')
    # Trending
    if name[2] == 'T': isTrend = True
    elif name[2] == 'F': isTrend = False
    else: ValueError('Invalid isTrend.')
    # Lead Predictors
    lead_pred = LeadPredPocket(name.split('_',maxsplit=1)[-1])
    exp_setting = dict(
        indicator_name=indicator_name,
        isPower=isPower,
        isTrend=isTrend,
        lead_pred=lead_pred
    )
    return exp_setting

def generate_viewer_sim1():
    # Load FNID information ---------------------------- #
    isReplace = True
    min_records = 12
    cps = [
        ['Somalia', 'Sorghum', 'Deyr'],
        ['Somalia', 'Maize', 'Deyr'],
        ['Somalia', 'Sorghum', 'Gu'],
        ['Somalia', 'Maize', 'Gu'],
        ['Kenya', 'Maize', 'Long'],
        ['Kenya', 'Maize', 'Short'],
        ['Malawi', 'Maize', 'Main'],
        ['Burkina Faso', 'Maize', 'Main']
    ]
    list_model = ['XGB']

    combination = [
        ['Y'],
        ['F'],  # isPower
        ['T'],  # isTrend
        ['ACUM_ALL']
        # ['ACUM_P','INDV_P','MIX1_P']
    ]
    exp_setting_name = []
    for t1,t2,t3,t4 in product(*combination):
        exp_setting_name.append(t1+t2+t3+'_'+t4)
    exp_setting_dict = {exp_name: ExperimentSettingPocket(exp_name) for exp_name in exp_setting_name}
    comb = product(cps, list_model, exp_setting_name)
    for (country_name, product_name, season_name), model_name, exp_name in comb:
        psme_string = '%s_%s_%s_%s' % (product_name, season_name, model_name, exp_name)
        print("--------"*6)
        print("Processing %s_%s." % (country_name, psme_string))
        df = pd.read_hdf('./data_in/fnids_info.hdf')
        fnids_info = df[
            (df['country'] == country_name) &
            (df['product'] == product_name) &
            (df['season_name'] == season_name) &
            (df['record_yield'] >= min_records)
        ].reset_index(drop=True)
        country_iso = fnids_info['country_iso'].unique()[0]
        
        # Experiment setting ------------------------------- #
        exp_setting = exp_setting_dict[exp_name]
        
        # Remove existing files ---------------------------- #
        if isReplace:
            remove_files = sorted(glob.glob('./data_out/chafs/chafs_%s*_%s.npz' % (country_iso, psme_string)))
            num_rm_files = len(remove_files)
            for fn in remove_files:
                os.remove(fn)
            print('Removing %d existing files.' % num_rm_files)
        
        # Forecasting -------------------------------------- #
        stime = time.time()
        for i, row in fnids_info.iterrows():
            fnid, product_name, season_name = row['fnid'], row['product'], row['season_name']
            fn_out = './data_out/chafs/chafs_%s_%s' % (fnid, psme_string)
            if os.path.exists(fn_out+'.npz'): continue
            
            # Model development
            bp = dict(
                fnid = fnid,
                product_name = product_name,
                season_name = season_name,
                indicator_name = exp_setting['indicator_name'],
                model_name = model_name,
                lead_pred = exp_setting['lead_pred'],
                isPower = exp_setting['isPower'],
                isTrend = exp_setting['isTrend'],
                note = exp_name
            )
            obox = CHAFS_TIME(bp)
            np.savez_compressed(fn_out, obox=obox)
            # print('%s_%s is done.' % (fnid, psme_string))
        print('%d districts took %ds.' % (fnids_info.shape[0], time.time()-stime))
    print("--------"*6)