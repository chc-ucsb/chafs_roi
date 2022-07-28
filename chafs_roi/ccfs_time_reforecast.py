import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os, json
from functools import reduce
from itertools import product, combinations, compress
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error, r2_score, roc_curve, auc, mean_absolute_error
from . import metrics as mt
from boruta import BorutaPy
import xgboost as xgb
from xgboost import XGBRegressor
pd.options.mode.chained_assignment = None


# def Reforecast_by_FNID(bp):
#     # --------------------------
#     # Validation of parameters
#     fnid = bp['fnid']
#     product_name = bp['product_name']
#     season_name = bp['season_name']
#     indicator_name = bp['indicator_name']
#     model_name = bp['model_name']
#     lead_dkd = bp['lead_dkd']
#     flag_dekad = bp['flag_dekad']
#     flag_ext = bp['flag_ext']
#     flag_serial = bp['flag_serial']
#     isPower = bp['isPower']
#     isTrend = bp['isTrend']
#     exp_name = bp['exp_name']
#     note = bp['note']


def Reforecast_by_FNID(fnid, product_name, season_name, indicator_name, model_name, lead_dkd, flag_dekad, flag_ext, flag_serial, isPower, isTrend, exp_name, note):
    # --------------------------
    # Load individual output
    fn = './data_out/ccfs/ccfs_{:s}_{:s}_{:s}_{:s}_{:s}.npz'.format(fnid, product_name, season_name, model_name, exp_name)
    leadmat = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    obox = dict()

    # Load forecast results and input files
    box = np.load(fn, allow_pickle=True)['obox'].tolist()
    y = box['y']
    y_raw = y.copy()
    tdx = y.index
    tdx_test = box['fcst'].index
    y_test = y[tdx_test]
    harvest_end = box['harvest_end']
    datemat = box['datemat']
    # - Forecast results
    lead_hcst = box['hcst']
    lead_fcst = box['fcst']
    lead_hcst_esm = box['hcst_esm']
    lead_fcst_esm = box['fcst_esm']
    lead_features = box['feature_sel']
    model_fitted = box['model_fitted']

    # Crop data control
    info, _ = CropDataControl(fnid, product_name, season_name, indicator=indicator_name)

    # EO data control
    df = EODataControl(fnid)
    pname = df.columns
    npred = pname.shape[0]

    fnid, country, name, product_name, season_name, harvest_end, forecast_end = info
    obox = dict({
        'FNID': fnid, 'country':country, 'name':name, 'product_name': product_name,
        'season_name':season_name, 'indicator_name':indicator_name,  
        'harvest_end':harvest_end, 'forecast_end':forecast_end, 
        'status':0, 'isTrend': isTrend,
        'flag_dekad':flag_dekad, 'flag_ext':flag_ext, 'flag_serial':flag_serial,
    })

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

    # Forecast Time Table (FTT)
    predyear = np.array(df.index.year.unique())
    # - Add a future buffer year. This will ignored if irrelavant.
    predyear = np.append(predyear, predyear[-1]+1)
    datelead = df.index[df.index.get_loc(y.index[0]) - np.array(leadmat)]
    datelead_dekad = datelead.strftime('%m-%d')
    temp = [(datelead + pd.DateOffset(years=diff)).values for diff in (predyear - datelead.year.max())]
    temp = pd.to_datetime(np.concatenate(temp))
    avaldata = df.iloc[:,0].to_frame().reindex(temp).values.reshape([len(predyear), len(datelead_dekad)])
    ftt = pd.DataFrame(index=pd.to_datetime(['%d-%s-01'%(yr, forecast_end[:2]) for yr in predyear]), 
                       columns=leadmat, data=avaldata).notna()
    # - Remove years that forecasts are entirely unavailable
    ftt = ftt[ftt.sum(1) > 0]

    # Containers of result tables
    lead_rcst = pd.DataFrame(index=ftt.index, columns=ftt.columns, dtype=np.float32)
    lead_rcst_esm = {}
    # Main forecasting algorithms
    # Start Loop --------- #
    for lead in leadmat:
        lead_srl = leadmat[:leadmat.index(lead)+1]
        tdx_all = ftt.index[ftt[lead]]
        features_sel = lead_features[lead]

        # Generate lead series predictors
        leadcomb = CombSerialLead(lead_srl, time_res='dekad', flag_serial=True)
        xTRAN = GenerateSeriesLeadPredTable(df, tdx, leadcomb, ['tavg','tmax','tmin','ndvi'], flag_ext=True)
        xRCST = GenerateSeriesLeadPredTable(df, tdx_all, leadcomb, ['tavg','tmax','tmin','ndvi'], flag_ext=True)

        # ---------- Reconstructing ---------- #
        y_train, y_mean = y.values, y.mean()
        X_train_sel = xTRAN[features_sel].values
        regr = sklearn.base.clone(model_fitted[lead])
        regr.random_state = 42

        # Reconstruct with a single model and entire records
        rcst = np.zeros(tdx_all.shape)
        rcst_esm = np.zeros([len(tdx_all), regr.n_estimators])
        regr.fit(X_train_sel, y_train)
        for it, index in enumerate(tdx_all):
            # - Get all ensembles
            X_test_rcst = xRCST.loc[index, features_sel].values[None,:]
            rcst[it] = regr.predict(X_test_rcst)
            if model_name in ['RF','ET'] :
                estimators = regr.estimators_
            elif model_name == 'GB':
                estimators = [est[0] for est in regr.estimators_.tolist()]
            esm = np.array(list(map(lambda x: x.predict(X_test_rcst)[0], estimators)))
            rcst_esm[it] = esm
        lead_rcst.loc[tdx_all, lead] = rcst
        # Retrend ensembles
        rcst_esm = rcst_esm + trend.predict(np.array(tdx_all.year)[:,None])[:,None] - y.mean()
        lead_rcst_esm[lead] = rcst_esm

    # Out-Of-Loop --------- #
    lead_rcst_dt = lead_rcst.copy()
    if isTrend:
        # Retrending
        lead_rcst = lead_rcst + trend.predict(np.array(ftt.index.year)[:,None])[:,None] - y.mean()

    # Low and High prediction interavals
    lead_hcst_low = lead_hcst.copy()
    lead_hcst_high = lead_hcst.copy()
    lead_fcst_low = lead_fcst.copy()
    lead_fcst_high = lead_fcst.copy()
    lead_rcst_low = lead_rcst.copy()
    lead_rcst_high = lead_rcst.copy()
    for lead in leadmat:
        # Hindcast
        lead_hcst_low.loc[tdx,lead] = lead_hcst[lead] - lead_hcst_esm[lead].std(1)
        lead_hcst_high.loc[tdx,lead] = lead_hcst[lead] + lead_hcst_esm[lead].std(1)
        # Forecast
        lead_fcst_low.loc[tdx_test,lead] = lead_fcst[lead] - lead_fcst_esm[lead].std(1)
        lead_fcst_high.loc[tdx_test,lead] = lead_fcst[lead] + lead_fcst_esm[lead].std(1)
        # Reconstruct
        tdx_all = ftt.index[ftt[lead]]
        lead_rcst_low.loc[tdx_all,lead] = lead_rcst.loc[tdx_all,lead] - lead_rcst_esm[lead].std(1)
    lead_rcst_high.loc[tdx_all,lead] = lead_rcst.loc[tdx_all,lead] + lead_rcst_esm[lead].std(1)

    # Forecast errors (observed - forecasted)
    lead_hcst_error = y.values[:,None] - lead_hcst
    lead_fcst_error = y_test.values[:,None] - lead_fcst

    # Reorder forecasts by better & earlier lead-time
    nse_hcst = lead_hcst.apply(lambda x: mt.msess(y, x, y.mean()), axis=0)/100
    nse_fcst = lead_fcst.apply(lambda x: mt.msess(y_test, x, y.mean()), axis=0)/100
    mape_hcst = lead_hcst.apply(lambda x: mean_absolute_percentage_error(y, x), axis=0)
    mape_fcst = lead_fcst.apply(lambda x: mean_absolute_percentage_error(y_test, x), axis=0)
    score = pd.concat([nse_hcst, nse_fcst, 
                       mape_hcst, mape_fcst
                      ], axis=1, 
                      keys=['nse_hcst', 'nse_fcst', 
                            'mape_hcst', 'mape_fcst'
                           ]).T
    criteria_nse = np.dot(score.loc[['nse_hcst', 'nse_fcst']].values.T, [0.8,0])
    criteria_mape = np.dot(score.loc[['mape_hcst', 'mape_fcst']].values.T, [0.2,0])
    criteria = criteria_nse - criteria_mape
    # criteria = criteria_nse
    ssorder_month_end = 18 - (criteria.reshape(6,3).argmax(axis=1) + np.arange(6)*3)
    ssorder = np.array(leadmat.copy())
    ssorder[[ 2,  5,  8, 11, 14, 17]] = ssorder_month_end

    # Re-order forecast results
    score = score[ssorder]
    score.columns = leadmat
    for j, lead in enumerate(leadmat):
        # Hindcast
        lead_hcst[lead] = lead_hcst[ssorder[j]]
        lead_hcst_error[lead] = lead_hcst_error[ssorder[j]]
        lead_hcst_high[lead] = lead_hcst_high[ssorder[j]]
        lead_hcst_low[lead] = lead_hcst_low[ssorder[j]]
        # Forecast
        lead_fcst[lead] = lead_fcst[ssorder[j]]
        lead_fcst_error[lead] = lead_fcst_error[ssorder[j]]
        lead_fcst_high[lead] = lead_fcst_high[ssorder[j]]
        lead_fcst_low[lead] = lead_fcst_low[ssorder[j]]
        # Reconstruct
        lead_rcst[lead] = lead_rcst[ssorder[j]]
        lead_rcst_dt[lead] = lead_rcst_dt[ssorder[j]]
        lead_rcst_high[lead] = lead_rcst_high[ssorder[j]]
        lead_rcst_low[lead] = lead_rcst_low[ssorder[j]]
        
    # Control forecast time table
    lead_rcst = lead_rcst[ftt]
    lead_rcst_dt = lead_rcst_dt[ftt]
    lead_rcst_high = lead_rcst_high[ftt]
    lead_rcst_low = lead_rcst_low[ftt]

    # Save to output container
    obox['fnid'] = fnid
    obox['y'] = y_raw
    obox['y_dt'] = y
    obox['score'] = score
    obox['ssorder'] = ssorder
    obox['hcst'] = lead_hcst
    obox['hcst_error'] = lead_hcst_error
    obox['hcst_low'] = lead_hcst_low
    obox['hcst_high'] = lead_hcst_high
    obox['fcst'] = lead_fcst
    obox['fcst_error'] = lead_fcst_error
    obox['fcst_low'] = lead_fcst_low
    obox['fcst_high'] = lead_fcst_high
    obox['rcst'] = lead_rcst
    obox['rcst_dt'] = lead_rcst_dt
    obox['rcst_low'] = lead_rcst_low
    obox['rcst_high'] = lead_rcst_high
    print('%s is finished.' % fnid)

    return obox


def CropDataControl(fnid, product_name, season_name, indicator):
    # Select crop data
    crop = pd.read_hdf('./data_in/%s_crop_%s_%s.hdf' % (fnid, product_name, season_name))
    crop = crop.loc[:,pd.IndexSlice[:,:,:,:,:,:,indicator]].droplevel(-1,axis=1)
    # Get basic information of admin unit
    fnid, country, name, product_name, season_name, harvest_end = list(crop.columns.values[0])

    # Forecasting ending month (ending month)
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
    info = (fnid, country, name, product_name, season_name, harvest_end, forecast_end)
    
    # Crop data control -------------- #
    y = crop.copy()
    y[y == 0] = np.nan   # Remove zero value

    # Convert DataFrame to Series
    y = pd.Series(index = tdx_forecast_end, data = y.values.flatten()).rename_axis(index='date').rename('value')
    # Drop missing years
    tdx = y[y.notna()].index
    y = y[tdx]
    return info, y

def EODataControl(fnid):
    df = pd.read_hdf('./data_in/%s_pred.hdf' % fnid)
    # Fill missing values of predictors with climatological means
    df = df.groupby(df.index.strftime('%m-%d')).transform(lambda x: x.fillna(x.mean()))
    return df

def _Detrend(sr):
    sr = sr.copy()
    sr.index = pd.PeriodIndex(sr.index, freq='Y')
    sr = sr.reindex(pd.period_range(sr.index[0], sr.index[-1], freq='Y'), fill_value=np.nan)
    sr.index = sr.index.strftime('%Y').astype(int)
    nni = sr.notna()
    return LinearRegression().fit(sr[nni].index.values[:,None],sr[nni])

def subsequences(iterable, length):
    return [iterable[i: i + length] for i in range(len(iterable) - length + 1)]


def CombSerialLead(lead_srl, time_res='dekad', flag_serial=True):
    # Dekadal & Serial
    comb = [subsequences(lead_srl,l) for l in range(1,len(lead_srl)+1)]
    comb = [item for sublist in comb for item in sublist]
    # Monthly & Serial
    if time_res == 'month':
        length = np.array([len(sub) for sub in comb])
        sel1 = np.mod(length, 3) == 0    # only 3,6,9,12,15,18 dekads
        sel2 = np.isin(np.array([sub[0] for sub in comb]), [18,15,12,9,6,3])
        sel3 = np.isin(np.array([sub[-1] for sub in comb]), [16,13,10,7,4,1])
        comb = list(compress(comb, (sel1 & sel2 & sel3)))
    # Dekadal & Unserial
    if (time_res=='dekad') & (flag_serial==False):
        comb = [[ld] for ld in lead_srl]
    # Monthly & Unserial
    elif (time_res=='month') & (flag_serial==False):
        comb = [comb for comb in comb if len(comb) == 3]
    return comb

def GenerateSeriesLeadPredTable(df, time_index, leadcomb, var_agg_mean, flag_ext=True):
    if not flag_ext:
        # Only original features
        df = df.loc[:, df.columns.isin(['pacu','eacu','aacu','tavg', 'tmax', 'tmin','ndvi'])]
    table = []
    for l in leadcomb:   
        table_sub = []
        for j in l:
            temp_sub = df.iloc[df.index.get_indexer(time_index) - j]
            temp_sub.index = time_index
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