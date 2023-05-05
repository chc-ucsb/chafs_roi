import os, sys, glob, json, time
from itertools import product
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from .ccfs_time_reforecast import _Detrend, CombSerialLead, GenerateSeriesLeadPredTable
from .ccfs_time_reforecast import Reforecast_by_FNID
import multiprocessing as mp
np.seterr(divide='ignore', invalid='ignore')

def FindLatestDeakad(feature_sel):
    return np.array([int(ft[-2:]) for ft in feature_sel]).min()


def generate_viewer_com():
    
    # Load FNID information ---------------------------- #
    fnids_info = pd.read_hdf('./data_in/fnids_info.hdf')
    fnids = fnids_info['fnid'].unique().tolist()
    country_code = fnids_info[['country','country_iso']].drop_duplicates().set_index('country').to_dict()['country_iso']
    country_to_use = country_code.keys()
    fnids_dict = fnids_info.groupby('country_iso')['fnid'].apply(lambda x: x.unique().tolist()).to_dict()
    # -------------------------------------------------- #
    
    # (1) Reforecast all years ------------------------- #
    # In order to run multiprocessing codes on Ipython, we need to make a function of main work
    # (Source: https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac)
    list_model = ['ET']
    cps = [
        ['Somalia','Sorghum','Deyr'],
        ['Somalia','Sorghum','Gu'],
        ['Somalia','Maize','Deyr'],
        ['Somalia','Maize','Gu'],
        ['Malawi','Maize','Main'],
        ['Kenya','Maize','Long'],
        ['Kenya','Maize','Short'],
        ['Burkina Faso','Maize','Main'],
        # ['Burkina Faso','Sorghum','Main']
    ]
    comb = product(cps, list_model)
    stime = time.time()
    for (country_name, product_name, season_name), model_name in comb:
        country_iso = country_code[country_name]
        leadmat = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        print('%s-%s-%s-%s' % (country_name, product_name, season_name, model_name))

        # Basic parameters
        indicator_name = 'yield'
        lead_dkd = 'all'
        flag_dekad = True
        flag_ext = True
        flag_serial = True
        isPower = False
        isTrend = True
        exp_name = 'YFT_INDV_ALL'
        note = ''

        # Remove the missing FNIDS
        fn_format = './data_out/ccfs/ccfs_{:s}_{:s}_{:s}_{:s}_{:s}.npz'
        fnids_exists = [fnid for fnid in fnids_dict[country_iso] if 
                        os.path.exists(fn_format.format(fnid, product_name, season_name, model_name, exp_name))]

        # Initial parameters
        obox = dict()
        box_y = {fnid: [] for fnid in fnids_exists}
        box_y_dt = {fnid: [] for fnid in fnids_exists}
        box_ssorder = pd.DataFrame(index=fnids_exists, columns=leadmat, dtype=np.float32).rename_axis(index='fnid', columns="lead")
        box_nse_hcst = box_ssorder.copy()
        box_nse_fcst = box_ssorder.copy()
        box_mape_hcst = box_ssorder.copy()
        box_mape_fcst = box_ssorder.copy()
        box_hcst = {fnid: [] for fnid in fnids_exists}
        box_hcst_error = {fnid: [] for fnid in fnids_exists}
        box_hcst_low = {fnid: [] for fnid in fnids_exists}
        box_hcst_high = {fnid: [] for fnid in fnids_exists}
        box_fcst = {fnid: [] for fnid in fnids_exists}
        box_fcst_error = {fnid: [] for fnid in fnids_exists}
        box_fcst_low = {fnid: [] for fnid in fnids_exists}
        box_fcst_high = {fnid: [] for fnid in fnids_exists}
        box_rcst = {fnid: [] for fnid in fnids_exists}
        box_rcst_dt = {fnid: [] for fnid in fnids_exists}
        box_rcst_low = {fnid: [] for fnid in fnids_exists}
        box_rcst_high = {fnid: [] for fnid in fnids_exists}

        # Multiprocessing per district
        pool = mp.Pool(4)
        jobs = []
        for fnid in fnids_exists:
            bp = (
                fnid, product_name, season_name, indicator_name, 
                model_name, lead_dkd, flag_dekad, flag_ext, flag_serial, 
                isPower, isTrend, exp_name, note
            )
            jobs.append(pool.apply_async(Reforecast_by_FNID, bp))
        pool.close()
        pool.join()
        output = [job.get() for job in jobs]

        # Store predicted values, errors, confidence intervals
        for ibox in output:
            fnid = ibox['fnid']
            box_y[fnid] = ibox['y']
            box_y_dt[fnid] = ibox['y_dt']
            box_ssorder.loc[fnid,:] = ibox['ssorder']
            box_nse_hcst.loc[fnid,:] = ibox['score'].loc['nse_hcst']
            box_nse_fcst.loc[fnid,:] = ibox['score'].loc['nse_fcst']
            box_mape_hcst.loc[fnid,:] = ibox['score'].loc['mape_hcst']
            box_mape_fcst.loc[fnid,:] = ibox['score'].loc['mape_fcst']
            box_hcst[fnid] = ibox['hcst']
            box_hcst_error[fnid] = ibox['hcst_error']
            box_hcst_low[fnid] = ibox['hcst_low']
            box_hcst_high[fnid] = ibox['hcst_high']
            box_fcst[fnid] = ibox['fcst']
            box_fcst_error[fnid] = ibox['fcst_error']
            box_fcst_low[fnid] = ibox['fcst_low']
            box_fcst_high[fnid] = ibox['fcst_high']
            box_rcst[fnid] = ibox['rcst']
            box_rcst_dt[fnid] = ibox['rcst_dt']
            box_rcst_low[fnid] = ibox['rcst_low']
            box_rcst_high[fnid] = ibox['rcst_high']

        # Save values
        obox['box_fnids'] = fnids_exists
        obox['box_y'] = box_y
        obox['box_y_dt'] = box_y_dt
        obox['box_ssorder'] = box_ssorder
        obox['box_nse_hcst'] = box_nse_hcst
        obox['box_nse_fcst'] = box_nse_fcst
        obox['box_mape_hcst'] = box_mape_hcst
        obox['box_mape_fcst'] = box_mape_fcst
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
        fn_out = './result/ccfs_reforecast_%s_%s_%s_%s_%s.npz' % (country_iso, product_name, season_name, model_name, exp_name)
        np.savez_compressed(fn_out, obox=obox)
        print('%s is saved.' % fn_out)
    print('%.1fs took' % (time.time() - stime))
    # -------------------------------------------------- #
    
    
    # (2) Extract crop data ---------------------------- #
    # CPS (Country-Product-Season)
    cps = [
        ['Somalia','Sorghum','Deyr'],
        ['Somalia','Sorghum','Gu'],
        ['Somalia','Maize','Deyr'],
        ['Somalia','Maize','Gu'],
        ['Malawi','Maize','Main'],
        ['Kenya','Maize','Long'],
        ['Kenya','Maize','Short'],
        ['Burkina Faso','Maize','Main'],
        # ['Burkina Faso','Sorghum','Main']
    ]
    country_to_use = list(np.unique(np.array(cps)[:,0]))

    # Load FEWSNET admin boundaries
    shape = gpd.read_file('/home/donghoonlee/chafs_roi/data/gscd_shape_stable.json').drop(columns='id')
    # shape = shape[shape.ADMIN0.isin(country_to_use)].reset_index(drop=True)
    dist_info = shape[['FNID','ADMIN0','ADMIN1','ADMIN2']]
    dist_info.columns = ['fnid','country','admin1','admin2']
    column_order = ['fnid','country','admin1','admin2','year','product','season','month','dekad','day','out-of-sample','variable','value']

    # Load crop area, production, yield data
    df = pd.read_csv('/home/donghoonlee/chafs_roi/data/gscd_data_stable.csv', index_col=0)
    # Reduce data according to CPS
    container = []
    for country_name, product_name, season_name in cps:
        sub = df[
                (df['country'] == country_name) &
                (df['product'] == product_name) &
                (df['season_name'] == season_name) &
                (df['gscd_code'] == 'calibrated')
        ]
        container.append(sub)
    df = pd.concat(container, axis=0).reset_index(drop=True)

    # Pivot table format --------------------------------- #
    area = df[df['indicator'] == 'area'].pivot_table(
        index='harvest_year',
        columns=['fnid','country','name','product','season_name','harvest_month'],         
        values='value', aggfunc='sum'
    )
    prod = df[df['indicator'] == 'production'].pivot_table(
        index='harvest_year',
        columns=['fnid','country','name','product','season_name','harvest_month'],         
        values='value', aggfunc='sum'
    )
    crop = df[df['indicator'] == 'yield'].pivot_table(
        index='harvest_year',
        columns=['fnid','country','name','product','season_name','harvest_month'],         
        values='value', aggfunc='sum'
    )
    # Extend columns
    area_cols = area.columns.to_frame().reset_index(drop=True)
    prod_cols = prod.columns.to_frame().reset_index(drop=True)
    crop_cols = crop.columns.to_frame().reset_index(drop=True)
    cols_extend = pd.concat([area_cols, prod_cols, crop_cols],axis=0).drop_duplicates().reset_index(drop=True)
    cols_extend = pd.MultiIndex.from_frame(cols_extend)
    area = area.reindex(columns = cols_extend)
    prod = prod.reindex(columns = cols_extend)
    crop = crop.reindex(columns = cols_extend)

    # Recalculate crop yields
    crop_recal = prod/area
    crop_recal[crop.isna()] = np.nan

    # Round off to the third decimal point
    crop = crop.round(3)
    crop_recal = crop_recal.round(3)

    # Replace with the recalculated crop yields
    number_replacement = sum(~np.isnan(crop.values[crop != crop_recal]))
    crop[crop != crop_recal] = crop_recal[crop != crop_recal]

    # Reform to pivot-table format
    area = area.T
    area['indicator'] = 'area'
    area = area.set_index('indicator', append=True).T
    prod = prod.T
    prod['indicator'] = 'production'
    prod = prod.set_index('indicator', append=True).T
    crop = crop.T
    crop['indicator'] = 'yield'
    crop = crop.set_index('indicator', append=True).T
    merged = pd.concat([area,prod,crop],axis=1)
    table = merged.reindex(columns = merged.columns.sortlevel([0])[0])
    data = table.T.stack().reset_index().rename(columns={0:'value'})
    # ---------------------------------------------------- #

    data['year'] = data['harvest_year']
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

    # Merge all
    container_crop = pd.concat([container_obs, crop_area], axis=0).reset_index(drop=True)
    # -------------------------------------------------- #
    
    
    # (3) Export reforecast results -------------------- #
    # Base dekads for temporal referencing
    dekad0 = pd.date_range('1980-01-01', '2023-12-31')
    d = dekad0.day - np.clip((dekad0.day-1) // 10, 0, 2)*10 - 1
    dekad0 = dekad0 - np.array(d, dtype="timedelta64[D]")
    dekad0 = dekad0.unique()
    def dekad_date(dekad, lead):
        return dekad0[dekad0.get_loc(dekad) - lead]

    # Extract forecast results
    def prettytable(df, fnid, product_name, season_name):
        df.columns = ['harvest', 'lead', 'value']
        df['date'] = np.vectorize(dekad_date)(df['harvest'],df['lead'])
        df[['fnid','product','season']] = fnid, product_name, season_name
        dt = df['date'].dt
        df[['year', 'month', 'dekad', 'day']] = pd.concat([dt.year, dt.month, 
                                                           np.ceil(dt.day/10).astype(int), 
                                                           dt.day], axis=1).values
        return df
    
    
    # Stacking forecast results
    list_model = ['ET']
    cps = [
        ['Somalia','Sorghum','Deyr'],
        ['Somalia','Sorghum','Gu'],
        ['Somalia','Maize','Deyr'],
        ['Somalia','Maize','Gu'],
        ['Malawi','Maize','Main'],
        ['Kenya','Maize','Long'],
        ['Kenya','Maize','Short'],
        ['Burkina Faso','Maize','Main'],
        # ['Burkina Faso','Sorghum','Main']
    ]
    container_out = []
    comb = product(cps, list_model)
    stime = time.time()
    for (country_name, product_name, season_name), model_name in comb:
        exp_name = 'YFT_INDV_ALL'
        country_iso = country_code[country_name]
        print('%s-%s-%s-%s' % (country_name, product_name, season_name, model_name))

        # Load forecat results
        fn_format = './result/ccfs_reforecast_{:s}_{:s}_{:s}_{:s}_{:s}.npz'
        box = np.load(fn_format.format(country_iso, product_name, season_name, model_name, exp_name), 
                      allow_pickle=True)['obox'].tolist()
        box_fnids = box['box_fnids']
        box_y_dt = box['box_y_dt']
        box_hcst = box['box_hcst']
        box_hcst_error = box['box_hcst_error']
        box_hcst_low = box['box_hcst_low']
        box_hcst_high = box['box_hcst_high']
        box_fcst = box['box_fcst']
        box_fcst_error = box['box_fcst_error']
        box_fcst_low = box['box_fcst_low']
        box_fcst_high = box['box_fcst_high']
        box_rcst = box['box_rcst']
        box_rcst_dt = box['box_rcst_dt']
        box_rcst_low = box['box_rcst_low']
        box_rcst_high = box['box_rcst_high']
        box_nse_hcst = box['box_nse_hcst']
        box_nse_fcst = box['box_nse_fcst']
        box_mape_hcst = box['box_mape_hcst']
        box_mape_fcst = box['box_mape_fcst']

        # Loop per FNID
        container = []
        for fnid in box_fnids:
            yield_long = crop_mean_long.loc[pd.IndexSlice[fnid,product_name,season_name],'yield_mean_all']
            yield_last10 = crop_mean_last10.loc[pd.IndexSlice[fnid,product_name,season_name],'yield_mean_last10']

            # (1) Hindcast + Forecast (values and percentages of last 10-year mean)
            # - Hindcast
            df_hcst = box_hcst[fnid].stack().reset_index()
            df_hcst = prettytable(df_hcst, fnid, product_name, season_name)
            df_hcst['out-of-sample'] = 0
            df_hcst['variable'] = 'yield_fcst'
            # - Forecast
            df_fcst = box_fcst[fnid].stack().reset_index()
            df_fcst = prettytable(df_fcst, fnid, product_name, season_name)
            df_fcst['out-of-sample'] = 1
            df_fcst['variable'] = 'yield_fcst'
            # - Reconstruct
            df_rcst = box_rcst[fnid].stack().reset_index()
            df_rcst = prettytable(df_rcst, fnid, product_name, season_name)
            df_rcst['out-of-sample'] = 2
            df_rcst['variable'] = 'yield_fcst'
            # - Merge them and remove the hindcast in the forecast period
            df = pd.concat([df_rcst, df_hcst, df_fcst], axis=0).reset_index(drop=True)
            # df = df.drop_duplicates(df.columns[[0,1,3,4,5,6,7,8,9,11]], keep='last').reset_index(drop=True)
            container.append(df.copy())
            # - Percentage to long-term mean 
            df_prct_long = df.copy()
            df_prct_long['value'] = df_prct_long['value']/yield_long*100
            df_prct_long['variable'] = 'yield_fcst_p30'
            container.append(df_prct_long)
            # - Percentage to last 10-year mean
            df_prct_last10 = df.copy()
            df_prct_last10['value'] = df_prct_last10['value']/yield_last10*100
            df_prct_last10['variable'] = 'yield_fcst_p10'
            container.append(df_prct_last10)
            # - Percentage to last 10-year mean (detrended)
            yield_dt = box_y_dt[fnid]
            yield_last10_dt = yield_dt[yield_dt.index.year.isin(np.arange(2009, 2019))].mean()
            df_rcst_dt = box_rcst_dt[fnid].stack().reset_index()
            df_prct_last10_dt = prettytable(df_rcst_dt, fnid, product_name, season_name)
            df_prct_last10_dt['out-of-sample'] = 2
            df_prct_last10_dt['value'] = df_prct_last10_dt['value']/yield_last10_dt*100
            df_prct_last10_dt['variable'] = 'yield_fcst_p10_dt'
            container.append(df_prct_last10_dt)

            # (2) Hindcast Low (values and percentages of last 10-year mean)
            # - Hindcast
            df_hcst = box_hcst_low[fnid].stack().reset_index()
            df_hcst = prettytable(df_hcst, fnid, product_name, season_name)
            df_hcst['out-of-sample'] = 0
            df_hcst['variable'] = 'yield_fcst_low'
            # - Forecast
            df_fcst = box_fcst_low[fnid].stack().reset_index()
            df_fcst = prettytable(df_fcst, fnid, product_name, season_name)
            df_fcst['out-of-sample'] = 1
            df_fcst['variable'] = 'yield_fcst_low'
            # - Reconstruct
            df_rcst = box_rcst_low[fnid].stack().reset_index()
            df_rcst = prettytable(df_rcst, fnid, product_name, season_name)
            df_rcst['out-of-sample'] = 2
            df_rcst['variable'] = 'yield_fcst_low'
            # - Merge them and remove the hindcast in the forecast period
            df = pd.concat([df_rcst, df_hcst, df_fcst], axis=0).reset_index(drop=True)
            # df = df.drop_duplicates(df.columns[[0,1,3,4,5,6,7,8,9,11]], keep='last').reset_index(drop=True)
            container.append(df.copy())
            # - Percentage to long-term mean 
            df_prct_long = df.copy()
            df_prct_long['value'] = df_prct_long['value']/yield_long*100
            df_prct_long['variable'] = 'yield_fcst_low_p30'
            container.append(df_prct_long)
            # - Percentage to last 10-year mean
            df_prct_last10 = df.copy()
            df_prct_last10['value'] = df_prct_last10['value']/yield_last10*100
            df_prct_last10['variable'] = 'yield_fcst_low_p10'
            container.append(df_prct_last10)

            # (2) Hindcast High (values and percentages of last 10-year mean)
            # - Hindcast
            df_hcst = box_hcst_high[fnid].stack().reset_index()
            df_hcst = prettytable(df_hcst, fnid, product_name, season_name)
            df_hcst['out-of-sample'] = 0
            df_hcst['variable'] = 'yield_fcst_high'
            # - Forecast
            df_fcst = box_fcst_high[fnid].stack().reset_index()
            df_fcst = prettytable(df_fcst, fnid, product_name, season_name)
            df_fcst['out-of-sample'] = 1
            df_fcst['variable'] = 'yield_fcst_high'
            # - Reconstruct
            df_rcst = box_rcst_high[fnid].stack().reset_index()
            df_rcst = prettytable(df_rcst, fnid, product_name, season_name)
            df_rcst['out-of-sample'] = 2
            df_rcst['variable'] = 'yield_fcst_high'
            # - Merge them and remove the hindcast in the forecast period
            df = pd.concat([df_rcst, df_hcst, df_fcst], axis=0).reset_index(drop=True)
            # df = df.drop_duplicates(df.columns[[0,1,3,4,5,6,7,8,9,11]], keep='last').reset_index(drop=True)
            container.append(df.copy())
            # - Percentage to long-term mean 
            df_prct_long = df.copy()
            df_prct_long['value'] = df_prct_long['value']/yield_long*100
            df_prct_long['variable'] = 'yield_fcst_high_p30'
            container.append(df_prct_long)
            # - Percentage to last 10-year mean
            df_prct_last10 = df.copy()
            df_prct_last10['value'] = df_prct_last10['value']/yield_last10*100
            df_prct_last10['variable'] = 'yield_fcst_high_p10'
            container.append(df_prct_last10)

            # (4) Hindcast error
            # - Hindcast
            df_hcst = box_hcst_error[fnid].stack().reset_index()
            df_hcst = prettytable(df_hcst, fnid, product_name, season_name)
            df_hcst['out-of-sample'] = 0
            df_hcst['variable'] = 'yield_fcst_error'
            # - Forecast
            df_fcst = box_fcst_error[fnid].stack().reset_index()
            df_fcst = prettytable(df_fcst, fnid, product_name, season_name)
            df_fcst['out-of-sample'] = 1
            df_fcst['variable'] = 'yield_fcst_error'
            # - Merge them and remove the hindcast in the forecast period
            df = pd.concat([df_hcst, df_fcst], axis=0).reset_index(drop=True)
            # df = df.drop_duplicates(df.columns[[0,1,3,4,5,6,7,8,9,11]], keep='last').reset_index(drop=True)
            container.append(df.copy())
        container = pd.concat(container, axis=0)
        container = pd.merge(container, dist_info, left_on='fnid', right_on='fnid')
        container_hcst = container[[*column_order, 'lead']]

        # NSE Hindcast
        temp = box_nse_hcst
        temp = temp.stack().reset_index()
        temp.columns = ['fnid','lead','value']
        temp['variable'] = 'yield_nse'
        base = container_hcst[['fnid','country','admin1','admin2','product','season','month','dekad','day','lead']].drop_duplicates()
        container_nse_hcst = pd.merge(temp, base, left_on=['fnid','lead'], right_on=['fnid','lead'], how='outer')
        container_nse_hcst['out-of-sample'] = 0
        # NSE Forecast
        temp = box_nse_fcst
        temp = temp.stack().reset_index()
        temp.columns = ['fnid','lead','value']
        temp['variable'] = 'yield_nse'
        base = container_hcst[['fnid','country','admin1','admin2','product','season','month','dekad','day','lead']].drop_duplicates()
        container_nse_fcst = pd.merge(temp, base, left_on=['fnid','lead'], right_on=['fnid','lead'], how='outer')
        container_nse_fcst['out-of-sample'] = 1

        # MAPE Hindcast
        temp = box_mape_hcst*100
        temp = temp.stack().reset_index()
        temp.columns = ['fnid','lead','value']
        temp['variable'] = 'yield_mape'
        base = container_hcst[['fnid','country','admin1','admin2','product','season','month','dekad','day','lead']].drop_duplicates()
        container_mape_hcst = pd.merge(temp, base, left_on=['fnid','lead'], right_on=['fnid','lead'], how='outer')
        container_mape_hcst['out-of-sample'] = 0
        # MAPE Forecast
        temp = box_mape_fcst*100
        temp = temp.stack().reset_index()
        temp.columns = ['fnid','lead','value']
        temp['variable'] = 'yield_mape'
        base = container_hcst[['fnid','country','admin1','admin2','product','season','month','dekad','day','lead']].drop_duplicates()
        container_mape_fcst = pd.merge(temp, base, left_on=['fnid','lead'], right_on=['fnid','lead'], how='outer')
        container_mape_fcst['out-of-sample'] = 1

        # Merge forecast results and MAPE
        container_forecast = pd.concat([
            container_hcst, 
            container_nse_hcst, container_nse_fcst,
            container_mape_hcst, container_mape_fcst,
        ], axis=0)[column_order].reset_index(drop=True)

        # # Exclude districts modeled with insufficient records
        # min_records = 15
        # sub_info = fnids_info[
        #     (fnids_info['country'] == country_name) &
        #     (fnids_info['product'] == product_name) &
        #     (fnids_info['season_name'] == season_name)
        # ]
        # fnids_valid = sub_info.loc[sub_info['record'] >= min_records, 'fnid'].values.tolist()
        # container_forecast = container_forecast[container_forecast['fnid'].isin(fnids_valid)]
        container_forecast['model'] = model_name
        container_out.append(container_forecast)
    container_forecast = pd.concat(container_out, axis=0).reset_index(drop=True)
    # -------------------------------------------------- #
    
    
    # Season start month and Forecast start month ------ #
    # Start months of crop season and forecast
    cols = ['country','product','season','forecast_start_month','season_start_month']
    start = [
        ['Somalia','Sorghum','Gu',2,3],
        ['Somalia','Sorghum','Deyr',9,10],
        ['Somalia','Maize','Gu',2,3],
        ['Somalia','Maize','Deyr',9,10],
        ['Malawi','Maize','Main',10,11],
        ['Kenya','Maize','Long',2,3],
        ['Kenya','Maize','Short',9,10],
        ['Burkina Faso','Maize','Main',4,5],
    ]
    start = pd.DataFrame(start, columns=cols)
    # -------------------------------------------------- #
    
    
    # Save the "viewer_com.csv" ------------------------ #
    # Export data
    # container = pd.concat([container_crop, container_forecast], axis=0).reset_index(drop=True)
    container = pd.concat([container_forecast], axis=0).reset_index(drop=True)
    container = container.merge(start, on=['country','product','season'], how='outer') # 'outer' for 'crop_area_percent'
    container = container[['fnid','country','admin1','admin2','year','product','season',
                           'forecast_start_month','season_start_month',
                           'month','dekad','day','model','out-of-sample','variable','value']]
    container = container[container['variable'].notna()]
    fn_out = './viewer/viewer_data_com.csv'
    container.to_csv(fn_out)
    print('%s is saved.' % fn_out)

    # # Export shapefile
    # fn_out = './viewer/viewer_data.shp'
    # shape.to_file(fn_out)
    # print('%s is saved.' % fn_out)
    
    # df = pd.read_csv('./viewer/viewer_data_com.csv', low_memory=False).drop(['Unnamed: 0'],axis=1)
    # df['date'] = pd.to_datetime(df[['year','month','day']])
    # cps = df.loc[df['season'].notna(),['country','product','season']].drop_duplicates().reset_index(drop=True)
    # for i, (country_name, product_name, season_name) in cps.iterrows():
    #     sub = df[
    #         (df['country'] == country_name) &
    #         (df['product'] == product_name) &
    #         (df['season'] == season_name) &
    #         (df['year'].isin([2022,2023])) &
    #         (df['out-of-sample'] == 2) &
    #         (df['variable'] == 'yield_fcst')
    #     ]
    #     # print(sub.groupby(['country','fnid','product','season'])['date'].max())
    #     cps.loc[i,'latest'] = sub['date'].max()
    # print(cps)
    # print(df['variable'].unique())
    # df[['out-of-sample','variable']].drop_duplicates()
    # -------------------------------------------------- #
    
    return