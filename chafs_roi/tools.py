import os
import math
import shutil
import urllib
import requests
from itertools import product, compress, chain
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from netCDF4 import Dataset
import xarray as xr
import rtree
import fiona
import shapefile as shp
from shapely.geometry import shape, mapping
import rasterio
import rioxarray
from rasterio import features
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def Load_GSCD(cps):
    # Load crop area, production, yield data
    df = pd.read_csv('https://raw.githubusercontent.com/chc-ucsb/gscd/main/public/gscd_data_stable.csv', index_col=0)
    # df = pd.read_csv('/Users/dlee/gscd/public/gscd_data_stable.csv', index_col=0)

    # Reduce data
    container = []
    for country_name, product_name, season_name in cps:
        sub = df[
                (df['country'] == country_name) &
                (df['product'] == product_name) &
                (df['season_name'] == season_name)
        ]
        container.append(sub)
    df = pd.concat(container, axis=0).reset_index(drop=True)

    # Pivot table format
    table = df.pivot_table(
        index='harvest_year',
        columns=['fnid','country','name','product','season_name','harvest_month','indicator'],         
        values='value'
    )

    # Record years
    record = table.melt().pivot_table(
        index=['fnid','country','name','product','season_name','harvest_month'],
        columns='indicator',values='value',aggfunc='count'
    )
    record = record.reset_index()
    record = record.rename(columns={'area':'record_area','production':'record_production','yield':'record_yield'})

    # FNID information
    info = table.columns.droplevel(-1).to_frame().drop_duplicates().reset_index(drop=True)
    info = info.merge(
        record[['fnid','product','season_name','record_area','record_production','record_yield']], 
        left_on=['fnid','product','season_name'], 
        right_on=['fnid','product','season_name']
    ) 
    info.insert(2, 'country_iso', info['country'])
    info['country_iso'].replace({
        'Kenya': 'KE',
        'Somalia': 'SO',
        'Malawi': 'MW',
        'Burkina Faso': 'BF'
    },inplace=True)

    # Load FEWSNET admin boundaries
    shape = gpd.read_file('https://raw.githubusercontent.com/chc-ucsb/gscd/main/public/gscd_shape_stable.json').drop(columns='id')
    # shape = gpd.read_file('/Users/dlee/gscd/public/gscd_shape_stable.json').drop(columns='id')
    shape = shape[shape['ADMIN0'].isin(info.country.unique())].reset_index(drop=True)
    
    return df, info, shape


def month2lead(cps, m):
    [country_name, product_name, season_name] = cps
    cps_name = country_name+'_'+product_name+'_'+season_name
    if cps_name == 'Somalia_Sorghum_Gu': 
        month_lead = {2:6, 3:5, 4:4, 5:3, 6:2, 7:1}
    elif cps_name == 'Somalia_Sorghum_Deyr': 
        month_lead = {9:6, 10:5, 11:4, 12:3, 1:2, 2:1}
    elif cps_name == 'Somalia_Maize_Gu': 
        month_lead = {2:6, 3:5, 4:4, 5:3, 6:2, 7:1}
    elif cps_name == 'Somalia_Maize_Deyr': 
        month_lead = {9:6, 10:5, 11:4, 12:3, 1:2, 2:1}
    elif cps_name == 'Malawi_Maize_Main': 
        month_lead = {10:6, 11:5, 12:4, 1:3, 2:2, 3:1}
    elif cps_name == 'Kenya_Maize_Long': 
        month_lead = {2:6, 3:5, 4:4, 5:3, 6:2, 7:1}
    elif cps_name == 'Kenya_Maize_Short': 
        month_lead = {9:6, 10:5, 11:4, 12:3, 1:2, 2:1}
    elif cps_name == 'Burkina Faso_Maize_Main': 
        month_lead = {4:6, 5:5, 6:4, 7:3, 8:2, 9:1}
    elif cps_name == 'Burkina Faso_Sorghum_Main': 
        month_lead = {4:6, 5:5, 6:4, 7:3, 8:2, 9:1}
    else: raise ValueError('CPS is not defined.')
    return month_lead[m]


def sort_dict(d):
    return dict(sorted(d.items()))

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = key 
            else: 
                inverse[item].append(key) 
    return inverse

def invert_dicts(d):
    inverse = {}
    for k, v in d.items():
        inverse[v] = sorted(inverse.get(v, []) + [k])
    return inverse

def NestLink(link1, link2):
    merged = {}
    for current, before in link1.items():
        if before in link2.keys():
            if isinstance(link2[before], list):
                merged[current] = [before, *link2[before]]
            else:
                merged[current] = [before, link2[before]]
        else:
            merged[current] = [before]
    return merged

def CreateNestedLinks(adm_link):
    lst = list(adm_link.items())
    # Build a directed graph and a list of all names that have no parent
    graph = {name: set() for tup in lst for name in tup}
    has_parent = {name: False for tup in lst for name in tup}
    for parent, child in lst:
        graph[parent].add(child)
        has_parent[child] = True
    roots = [name for name, parents in has_parent.items() if not parents]
    roots = sorted(roots)
    # Nested links
    nested = {}
    for root in roots:
        key = root
        links = []
        while root in adm_link.keys():
            root = adm_link[root]
            links.append(root)
        nested[key] = links
    return nested

def CreateLinkAdmin(shape_old, shape_new, old_on='ADMIN2', new_on='ADMIN2'):
    '''Algorithm to get relationships of changes in administrative boundaries
    '''
    # Find link
    over = gpd.overlay(shape_old, shape_new, how='intersection')
    over = over.to_crs('EPSG:32629')
    over['area'] = over['geometry'].area / 10**6
    link = {}
    for i in shape_new.FNID:
        temp = over.loc[over['FNID_2'] == i, ['FNID_1', 'area']]
        link[i] = temp.iloc[temp.area.argmax()]['FNID_1']
    # Find link and name
    link_name = {}
    for l2, l1 in link.items():
        name2 = shape_new.loc[shape_new.FNID == l2, new_on].to_list()
        name1 = shape_old.loc[shape_old.FNID == l1, old_on].to_list()
        link_name['%s (%s)' % (l2, *name2)] = '%s (%s)' % (l1, *name1)
    # Find newly added units as an inverse form
    inv = invert_dicts(link_name)
    add = list(compress(list(inv.keys()), np.array([len(v) for k, v in inv.items()]) > 1))
    diff_inv = {k: inv[k] for k in add}
    return sort_dict(link), sort_dict(link_name), sort_dict(diff_inv)


def ExtractCropDataArea(df, latest, fnid_area, name_area='Area Planted', adm_link=None):
    # Collect unique FNID and Name according to their Admin-level
    fnid_name = df[['fnid','admin_1','admin_2']].drop_duplicates().reset_index(drop=True)
    fdx = [fnid[6:8] == 'A1' for fnid in df.fnid.unique()]; fnid_name.loc[fdx, 'name'] = fnid_name.loc[fdx, 'admin_1']
    fdx = [fnid[6:8] == 'A2' for fnid in df.fnid.unique()]; fnid_name.loc[fdx, 'name'] = fnid_name.loc[fdx, 'admin_2']
    fnid_name = fnid_name[['fnid','name']]
    # Collect season names and dates
    season = df[['season_name','season_date']]
    season['season_date'] = pd.to_datetime(season['season_date']).dt.strftime('%m-%d')
    season = season.groupby('season_name').agg(pd.Series.mode).reset_index()
    season_name, season_date = list(season['season_name']), list(season['season_date'])
    # Loop by FNID
    frame_area = []
    frame_prod = []
    for i, (fnid, name) in fnid_name.iterrows():
        subdf = df[df.fnid == fnid]
        time = pd.to_datetime(subdf['season_date'].values)
        pivot = pd.pivot_table(subdf, values='value', index=['season_name','season_date'], columns=['indicator'], 
                               aggfunc=lambda x: x.sum(min_count=1))
        pivot = pivot.reindex(columns = [name_area,'Quantity Produced'])
        # Remove season if either area and production is missing
        pivot = pivot[pivot.isna().sum(1) == 0]
        if len(pivot) == 0: continue
        time = pd.to_datetime(pivot.index.get_level_values(1))
        pivot['year'] = time.year
        # Final area and production tables
        table_area = pd.pivot_table(pivot, index='year',columns='season_name',values=name_area).reindex(season_name, axis=1)
        table_prod = pd.pivot_table(pivot, index='year',columns='season_name',values='Quantity Produced').reindex(season_name, axis=1)
        # MultiIndex
        frame_index = season.copy()
        frame_index[['fnid', 'name']] = fnid, name
        frame_index = frame_index[['fnid', 'name', 'season_name', 'season_date']]
        mdx = pd.MultiIndex.from_frame(frame_index, names=['FNID','name','season','harvest_end'])
        table_area.columns = mdx
        table_prod.columns = mdx
        frame_area.append(table_area)
        frame_prod.append(table_prod)
    area = pd.concat(frame_area, axis=1)
    prod = pd.concat(frame_prod, axis=1)
    
    # Append the missing lastest FNIDs (This should exist for adm_link)
    if latest.iloc[0].FNID[7] == '1':
        adm_name = 'ADMIN1'
    elif latest.iloc[0].FNID[7] == '2':
        adm_name = 'ADMIN2'
    else:
        adm_name = 'ADMIN3'
    fnid_name_missing = latest.loc[~latest.FNID.isin(area.columns.get_level_values(0).unique()), ['FNID',adm_name]]
    fnid_name_missing.columns = ['fnid','name']
    if len(fnid_name_missing) > 0:
        frame_area = []
        frame_prod = []
        for i, (fnid, name) in fnid_name_missing.iterrows():
            # MultiIndex
            table_area = table_area.copy()
            frame_index = season.copy()
            frame_index[['fnid', 'name']] = fnid, name
            frame_index = frame_index[['fnid', 'name', 'season_name', 'season_date']]
            mdx = pd.MultiIndex.from_frame(frame_index, names=['FNID','name','season','harvest_end'])
            area_temp = pd.DataFrame(index=table_area.index, data=np.full_like(table_area, np.nan))
            prod_temp = pd.DataFrame(index=table_area.index, data=np.full_like(table_area, np.nan))
            area_temp.columns = mdx
            prod_temp.columns = mdx
            frame_area.append(area_temp)
            frame_prod.append(prod_temp)
        area_missing = pd.concat(frame_area, axis=1)
        prod_missing = pd.concat(frame_prod, axis=1)
        # Merging the missing districts
        area = pd.concat([area, area_missing], axis=1)
        prod = pd.concat([prod, prod_missing], axis=1)
    
    # Calibrate the divisions of districts (connect data)
    if adm_link is not None:
        for current, before in adm_link.items():
            # Find existing districts
            if np.isin(current, area.columns.get_level_values(0)).sum() == 0: continue
            exist = list(compress(before, np.isin(before, area.columns.get_level_values(0))))
            if len(exist) == 0: continue
            # Caclulate the ratios of area
            area_current = fnid_area.loc[fnid_area.FNID == current, 'area'].values[0]
            area_exist = [fnid_area.loc[fnid_area.FNID == fnid, 'area'].values[0] for fnid in exist]
            area_ratio = area_exist/area_current
            area_ratio[area_ratio > 0.95] = 1
            # Multiply existing unit with the ratios
            area_temp = [area[current]]
            prod_temp = [prod[current]]
            for fnid, ratio in zip(exist, area_ratio):
                area_temp.append(area[fnid]*ratio)
                prod_temp.append(prod[fnid]*ratio)
            area_temp = pd.concat(area_temp, axis=1).groupby(level=1, axis=1).sum(min_count=1)
            area[current] = area_temp.values
            prod_temp = pd.concat(prod_temp, axis=1).groupby(level=1, axis=1).sum(min_count=1)
            prod[current] = prod_temp.values

    # Select latest districts
    fnid_latest = area.columns.get_level_values(0).isin(latest.FNID.unique())
    year_extend = np.arange(area.index.min(), area.index.max()+1)
    area = area.loc[:,fnid_latest].reindex(year_extend)
    prod = prod.loc[:,fnid_latest].reindex(year_extend)

    return area, prod


def ExtractCropData(df, latest, name_area='Area Planted', adm_link=None):
    # Collect unique FNID and Name according to their Admin-level
    fnid_name = df[['fnid','admin_1','admin_2']].drop_duplicates().reset_index(drop=True)
    fdx = [fnid[6:8] == 'A1' for fnid in df.fnid.unique()]; fnid_name.loc[fdx, 'name'] = fnid_name.loc[fdx, 'admin_1']
    fdx = [fnid[6:8] == 'A2' for fnid in df.fnid.unique()]; fnid_name.loc[fdx, 'name'] = fnid_name.loc[fdx, 'admin_2']
    fnid_name = fnid_name[['fnid','name']]
    # Collect season names and dates
    season = df[['season_name','period_date']]
    season['period_date'] = pd.to_datetime(season['period_date']).dt.strftime('%m-%d')
    season = season.groupby('season_name').agg(pd.Series.mode).reset_index()
    season_name, season_date = list(season['season_name']), list(season['period_date'])
    # Loop by FNID
    frame_area = []
    frame_prod = []
    for i, (fnid, name) in fnid_name.iterrows():
        subdf = df[df.fnid == fnid]
        time = pd.to_datetime(subdf['period_date'].values)
        pivot = pd.pivot_table(subdf, values='value', index=['season_name','period_date'], columns=['indicator'], 
                               aggfunc=lambda x: x.sum(min_count=1))
        pivot = pivot.reindex(columns = [name_area,'Quantity Produced'])
        # Remove season if either area and production is missing
        pivot = pivot[pivot.isna().sum(1) == 0]
        if len(pivot) == 0: continue
        time = pd.to_datetime(pivot.index.get_level_values(1))
        pivot['year'] = time.year
        # Final area and production tables
        table_area = pd.pivot_table(pivot, index='year',columns='season_name',values=name_area).reindex(season_name, axis=1)
        table_prod = pd.pivot_table(pivot, index='year',columns='season_name',values='Quantity Produced').reindex(season_name, axis=1)
        # MultiIndex
        frame_index = season.copy()
        frame_index[['fnid', 'name']] = fnid, name
        frame_index = frame_index[['fnid', 'name', 'season_name', 'period_date']]
        mdx = pd.MultiIndex.from_frame(frame_index, names=['FNID','name','season','harvest_end'])
        table_area.columns = mdx
        table_prod.columns = mdx
        frame_area.append(table_area)
        frame_prod.append(table_prod)
    area = pd.concat(frame_area, axis=1)
    prod = pd.concat(frame_prod, axis=1)
    
    # Calibrate the divisions of districts (connect data)
    if adm_link is not None:
        for current, before in adm_link.items():
            if np.isin(current, area.columns.get_level_values(0)).sum() == 0: continue
            exist = list(compress(before, np.isin(before, area.columns.get_level_values(0))))
            if len(exist) == 0: continue
            area_temp = area[[current, *exist]].groupby(level=2, axis=1).sum(min_count=1)
            area[current] = area_temp.values
            prod_temp = prod[[current, *exist]].groupby(level=2, axis=1).sum(min_count=1)
            prod[current] = prod_temp.values
#             area_temp = area[[current, *exist]].groupby(level=2, axis=1).sum()
#             area_temp_na = area[[current, *exist]].notna().groupby(level=2, axis=1).sum() > 0
#             area[current] = area_temp[area_temp_na].values
#             prod_temp = prod[[current, *exist]].groupby(level=2, axis=1).sum()
#             prod_temp_na = prod[[current, *exist]].notna().groupby(level=2, axis=1).sum() > 0
#             prod[current] = prod_temp[prod_temp_na].values
        
    # Select latest districts
    fnid_latest = area.columns.get_level_values(0).isin(latest.FNID.unique())
    year_extend = np.arange(area.index.min(), area.index.max()+1)
    area = area.loc[:,fnid_latest].reindex(year_extend)
    prod = prod.loc[:,fnid_latest].reindex(year_extend)
    
    return area, prod

def DownloadShapefile(path_url, path_dir, shape_name, flag_exist=True):
    fn_url = os.path.join(path_url, shape_name)
    fn_dir = os.path.join(path_dir, shape_name)
    if not os.path.exists(fn_dir[:-3] + 'shp'):
        urllib.request.urlretrieve(fn_url, fn_dir)
        shutil.unpack_archive(fn_dir, path_dir)
        os.remove(fn_dir)
    return


def RasterizeAdminIndex_rio(shapefile, field, xr_da):
    # Flat indices of each feature
    rdx = {}
    for geom, value in zip(shapefile.geometry, shapefile[field]):
        burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
                                    fill=0, 
                                    out_shape=xr_da.shape[1:], 
                                    transform=xr_da.rio.transform(),
                                    dtype=rasterio.uint16)
        if ~np.isin(value, np.unique(burned)):
            # If no grids, use all_touched=True
            burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
                                        fill=0, 
                                        out_shape=xr_da.shape[1:], 
                                        transform=xr_da.rio.transform(),
                                        dtype=rasterio.uint16,
                                        all_touched=True)
        rdx[value] = np.where(burned.flatten() == value)[0]
    return rdx


def RasterizeAdminIndex(shapefile, field, xr_da):
    # Flat indices of each feature
    rdx = {}
    for geom, value in zip(shapefile.geometry, shapefile[field]):
        burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
                                    fill=0, 
                                    out_shape=xr_da.shape[1:], 
                                    transform=xr_da.affine,
                                    dtype=rasterio.uint16)
        if ~np.isin(value, np.unique(burned)):
            # If no grids, use all_touched=True
            burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
                                        fill=0, 
                                        out_shape=xr_da.shape[1:], 
                                        transform=xr_da.affine,
                                        dtype=rasterio.uint16,
                                        all_touched=True)
        rdx[value] = np.where(burned.flatten() == value)[0]
    return rdx


# def RasterizeAdminIndex_reduced(shapefile, field, xr_da):
#     # Flat indices of each feature
#     rdx = {}
#     for geom, value in zip(shapefile.geometry, shapefile[field]):
#         burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
#                                     fill=0, 
#                                     out_shape=xr_da.shape[1:], 
#                                     transform=xr_da.affine,
#                                     dtype=rasterio.uint16)
#         if ~np.isin(value, np.unique(burned)):
#             # If no grids, use all_touched=True
#             burned = features.rasterize(shapes=((geom,value),(geom,value)), # To be iterable..
#                                         fill=0, 
#                                         out_shape=xr_da.shape[1:], 
#                                         transform=xr_da.affine,
#                                         dtype=rasterio.uint16,
#                                         all_touched=True)
#         rdx[value] = np.where(burned.flatten() == value)[0]
    
#     # Reduced indices 
#     rdx_long = [v for k,v in rdx.items()]
#     rdx_unique = np.unique(np.concatenate(rdx_long))
#     rdx_reduced = {}
#     for k,v in rdx.items():
#         rdx_reduced[k] = np.where(np.isin(rdx_unique,v))[0]
    
#     return rdx_unique, rdx_reduced


def RasterResampling_tif(filn_src_rst, filn_ref_rst, filn_dst_rst=None):
    '''Returns ndarray of source raster resampled to extent and resolution of the reference GeoTiff file.
    '''
    # Read source raster
    with rasterio.open(filn_src_rst) as src:
        source = src.read(1)
        src_meta = src.meta.copy()
        src_transform = src.transform
        src_crs = src.crs
    # Read reference raster
    with rasterio.open(filn_ref_rst) as ref:
        meta = ref.meta
    # Resampling
    dst, dst_transform = reproject(source,
                                   np.zeros([meta['height'], meta['width']], src_meta['dtype']),
                                   src_transform=src_transform,
                                   src_crs=src_crs,
                                   dst_transform=meta['transform'],
                                   dst_crs=src_crs,
                                   resampling=Resampling.average)
    # Save destination raster file
    if filn_dst_rst is not None:
        with rasterio.open(filn_dst_rst, 'w', **_meta) as dest:
            dest.write(dst[None,:])
    
    return dst

def RasterizeAdminShape_tif(shapefile, field, fn_ref_rst, fn_dst_rst=None):
    # Create a generator of geom and field value pairs
    shapes = ((geom,value) for geom, value in zip(shapefile.geometry, shapefile[field]))
    with rasterio.open(fn_ref_rst) as src:
        src_meta = src.meta.copy()
    burned = features.rasterize(shapes=shapes, 
                                fill=0, 
                                out_shape=[src_meta['height'], src_meta['width']], 
                                transform=src_meta['transform'])
    burned = burned.astype(rasterio.int16)
    # Save as GeoTiff file
    if fn_dst_rst is not None:
        meta = {'driver': 'GTiff',
                'height': burned.shape[0],
                'width': burned.shape[1],
                'transform': meta['transform'],
                'dtype':rasterio.uint16,
                'count':1}
        with rasterio.open(fn_dst_rst, 'w', **meta) as dst:
            dst.write(burned, indexes=1)
        print('%s is saved' % fn_dst_rst)
    return burned


def RasterizeAdminShape(shapefile, field, xr_da, fn_dst_rst=None):
    # Create a generator of geom and field value pairs
    shapes = ((geom,value) for geom, value in zip(shapefile.geometry, shapefile[field]))
    burned = features.rasterize(shapes=shapes, 
                                fill=0, 
                                out_shape=xr_da.shape[1:], 
                                transform=xr_da.affine)
    burned = burned.astype(rasterio.int16)
    # Save as GeoTiff file
    if fn_dst_rst is not None:
        meta = {'driver': 'GTiff',
                'height': burned.shape[0],
                'width': burned.shape[1],
                'transform': xr_da.affine,
                'dtype':rasterio.uint16,
                'count':1}
        with rasterio.open(fn_dst_rst, 'w', **meta) as dst:
            dst.write(burned, indexes=1)
        print('%s is saved' % fn_dst_rst)
    return burned

def RasterResampling(filn_src_rst, filn_dst_nc, filn_dst_rst=None):
    '''Returns ndarray of source raster resampled to extent and resolution of the reference NetCDF file.
    '''
    # Read source raster
    with rasterio.open(filn_src_rst) as src:
        source = src.read(1)
        src_meta = src.meta.copy()
        src_transform = src.transform
        src_crs = src.crs
        
    # Read reference NetCDF file
    data = xr.open_dataset(filn_dst_nc)
    if 'latitude' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.latitude), len(data.longitude))
        lat = data.latitude.values; lon = data.longitude.values
    elif 'lat' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.lat), len(data.lon))
        lat = data.lat.values; lon = data.lon.values
    elif 'x' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.y), len(data.x))
        lat = data.y.values; lon = data.x.values
    elif 'X' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.Y), len(data.X))
        lat = data.Y.values; lon = data.X.values
        
    # Resampling
    d = np.abs(np.mean(lat[:-1] - lat[1:]))
    dst_transform = A.translation(lon.min()-d/2, lat.max()+d/2) * A.scale(d, -d)
    dst, dst_transform = reproject(source,
                                   np.zeros(dst_shape, src_meta['dtype']),
                                   src_transform=src_transform,
                                   src_crs=src_crs,
                                   dst_transform=dst_transform,
                                   dst_crs=src_crs,
                                   resampling=Resampling.average)
    # Save destination raster file
    if filn_dst_rst is not None:
        out_meta = src_meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": dst.shape[0],
                         "width": dst.shape[1],
                         "transform": dst_transform})
        with rasterio.open(filn_dst_rst, 'w', **out_meta) as dest:
            dest.write(dst[None,:])
    
    return dst

def RasterResampling_da(filn_src_rst, data):
    '''Returns ndarray of source raster resampled to extent and resolution of the reference NetCDF file.
    '''
    # Read source raster
    with rasterio.open(filn_src_rst) as src:
        source = src.read(1)
        src_meta = src.meta.copy()
        src_transform = src.transform
        src_crs = src.crs
        
    # Read reference NetCDF file
    if 'latitude' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.latitude), len(data.longitude))
        lat = data.latitude.values; lon = data.longitude.values
    elif 'lat' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.lat), len(data.lon))
        lat = data.lat.values; lon = data.lon.values
    elif 'x' in list(data.coords.keys()):
        rows, cols = dst_shape = (len(data.y), len(data.x))
        lat = data.y.values; lon = data.x.values
        
    # Resampling
    d = np.abs(np.mean(lat[:-1] - lat[1:]))
    dst_transform = A.translation(lon.min()-d/2, lat.max()+d/2) * A.scale(d, -d)
    dst, dst_transform = reproject(source,
                                   np.zeros(dst_shape, src_meta['dtype']),
                                   src_transform=src_transform,
                                   src_crs=src_crs,
                                   dst_transform=dst_transform,
                                   dst_crs=src_crs,
                                   resampling=Resampling.average)
    
    return dst

def ResampleDay2Dekad(df):
    # Source: https://stackoverflow.com/a/15409033
    d = df.index.day - np.clip((df.index.day-1) // 10, 0, 2)*10 - 1
    date = df.index.values - np.array(d, dtype="timedelta64[D]")
    method = {'prcp': np.nansum,
              'etos': np.nanmean,
              'ndvi': np.nanmax,
              'tmax': np.nanmax}
    return df.groupby(date).agg(method)


def LinkFromFTP(ftp):
    link = []
    for line in urllib.request.urlopen(ftp): 
        k = line.decode("utf-8").strip().split(' ') 
        link.append(k[-1])
    return link


def RemoveBadNetcdfFileFast(fnList):
    new=[] 
    for fn in fnList:
        try:
            Dataset(fn)
            new.append(fn)
        except:
            pass
    return new


def RemoveBadNetcdfFile(fnList):
    new=[] 
    for fn in fnList:
        try:
            xr.open_dataset(fn)
            new.append(fn)
        except:
            pass
    return new


def LinkFromURL(url):
    '''
    Returns all hyperlinks in the URL.
    '''
    # Retreive links in the URL path
    urlpath = urllib.request.urlopen(url)
    html_doc = urlpath.read().decode('utf-8')
    # BeautifulSoup object
    soup = BeautifulSoup(html_doc, 'html.parser')
    # Make a list of hyerlinks
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    links.pop(0)     # Remove the parent link        
    return links


def DownloadFromURL(fullURL, fullDIR, showLog = False):
    '''
    Downloads the inserted hyperlinks (URLs) to the inserted files n the disk
    '''
    # Make parent directories if they do not exist
    if type(fullDIR) == list:
        parentDIRS = list(np.unique([os.path.dirname(DIR) for DIR in fullDIR]))
        for parentDIR in parentDIRS:
            os.makedirs(parentDIR, exist_ok=True)
        # Download all files
        nError = 0
        nExist = 0
        nDown = 0
        for file_url, file_dir in zip(fullURL, fullDIR):
            if not os.path.exists(file_dir):
                result = requests.get(file_url)
                try:
                    result.raise_for_status()
                    f = open(file_dir,'wb')
                    f.write(result.content)
                    f.close()
                    nDown += 1
                    print(file_dir, 'is saved.')
                except:
                    nError += 1
                    pass
            else:
                nExist += 1
        if showLog:
            print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
            
    elif type(fullDIR) == str:
        parentDIRS = os.path.dirname(fullDIR)
        # Download all files
        nError = 0
        nExist = 0
        nDown = 0
        if not os.path.exists(fullDIR):
            result = requests.get(fullURL)
            try:
                result.raise_for_status()
                f = open(fullDIR,'wb')
                f.write(result.content)
                f.close()
                nDown += 1
                print(file_dir, 'is saved.')
            except:
                nError += 1
                pass
            else:
                nExist += 1
        if showLog:
            print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
    return





# def DownloadFromURL(fullURL, fullDIR, showLog = False):
#     '''
#     Downloads the inserted hyperlinks (URLs) to the inserted files n the disk
#     '''
#     # Make parent directories if they do not exist
#     if type(fullDIR) == list:
#         parentDIRS = list(np.unique([os.path.dirname(DIR) for DIR in fullDIR]))
#         for parentDIR in parentDIRS:
#             os.makedirs(parentDIR, exist_ok=True)
#         # Download all files
#         nError = 0
#         nExist = 0
#         nDown = 0
#         for file_url, file_dir in zip(fullURL, fullDIR):
#             if not os.path.exists(file_dir):
#                 try:
#                     urllib.request.urlretrieve(file_url, file_dir)
#                     nDown += 1
#                     print(file_dir, 'is saved.')
#                 except:
#                     nError += 1
#                     pass
#             else:
#                 nExist += 1
#         if showLog:
#             print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
            
#     elif type(fullDIR) == str:
#         parentDIRS = os.path.dirname(fullDIR)
#         # Download all files
#         nError = 0
#         nExist = 0
#         nDown = 0
#         if not os.path.exists(fullDIR):
#                 try:
#                     urllib.request.urlretrieve(fullURL, fullDIR)
#                     nDown += 1
#                     print(fullDIR, 'is saved.')
#                 except:
#                     nError += 1
#                     pass
#                 else:
#                     nExist += 1
#         if showLog:
#             print('%d files are tried: %d exist, %d downloads, %d errors' % (len(fullURL),nExist,nDown,nError))
#     return




def z_score(df): 
    return (df-df.mean())/df.std(ddof=0)


def ExtendWithClimMean(df, tdx):
    df = df.reindex(tdx)
    df = df.groupby([df.index.month]).transform(lambda x: x.fillna(x.mean()))
    return df


def ExtendPeriodIndex(df_in):
    df_out = df_in.copy()
    tdx = pd.to_datetime(df_out.index,format='%Y-%m-%d')
    df_out.index = tdx.to_period('M')
    df_out = df_out.reindex(pd.date_range(tdx.min(), tdx.max(),freq='M').to_period('M'), 
                            fill_value=np.nan)
    return df_out



def IntersectShapefiles(refSHP, admSHP, outSHP, set_print=True):
    '''
    
    Source: https://gis.stackexchange.com/a/119397
    Revised by Donghoon Lee @ SEP-27-2020
    '''
    with fiona.open(refSHP, 'r') as layer1:
        with fiona.open(admSHP, 'r') as layer2:
            # We copy schema and add the  new property for the new resulting shp
            schema = layer2.schema.copy()
            schema['properties']['ID'] = 'int:10'
            # We open a first empty shp to write new content from both others shp
            with fiona.open(outSHP, 'w', 'ESRI Shapefile', schema) as layer3:
                index = rtree.index.Index()
                for feat1 in layer1:
                    fid = int(feat1['id'])
                    geom1 = shape(feat1['geometry'])
                    index.insert(fid, geom1.bounds)

                for feat2 in layer2:
                    geom2 = shape(feat2['geometry'])
                    for fid in list(index.intersection(geom2.bounds)):
                        if fid != int(feat2['id']):
                            feat1 = layer1[fid]
                            geom1 = shape(feat1['geometry'])
                            if geom1.intersects(geom2):
                                # We take attributes from admSHP
                                props = feat2['properties']
                                # Then append the uid attribute we want from the other shp
                                props['ID'] = feat1['properties']['ID']
                                # Add the content to the right schema in the new shp
                                layer3.write({
                                    'properties': props,
                                    'geometry': mapping(geom1.intersection(geom2))
                                })

    # Save a projection file (filename.prj)
    filename, _ = os.path.splitext(outSHP)
    prj = open("%s.prj" % filename, "w") 
    epsg = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]' 
    prj.write(epsg)
    prj.close()
    if set_print:
        print('%s is saved.' % outSHP)

    
    
def CreateGridBox_subextent(shp_out, extent, dx, dy, sub_extent, set_print=True):
    '''Create grid with degrees of extent, dx, dy of the target box
    
    Parameters
    ----------
    extent: list
        [minx,maxx,miny,maxy]
    dx: value
        degree of x
    dy: value
        degree of y

    Returns
    -------
    shp_out file is created.
    
    
    Source: https://gis.stackexchange.com/a/81120/29546
    Revised by Donghoon Lee @ Sep-24-2020
    '''
    
    # Size of extent
    width = np.round((extent[1] - extent[0])/dx)
    height = np.round((extent[3] - extent[2])/dy)
    
    # Adjust sub_extent with base grid extent
    minx = extent[0] + np.floor((sub_extent[0] - extent[0])/dx)*dx
    maxx = extent[0] + np.ceil((sub_extent[1] - extent[0])/dx)*dx
    maxy = extent[3] - np.floor((extent[3] - sub_extent[3])/dy)*dy
    miny = extent[3] - np.ceil((extent[3] - sub_extent[2])/dy)*dy
    sub_extent = [minx, maxx, miny, maxy]
    
    # Index of sub_extent
    left = np.floor((sub_extent[0] - extent[0])/dx)
    right = np.floor((sub_extent[1] - extent[0])/dx)
    top = np.floor((extent[3] - sub_extent[3])/dy)
    bottom = np.floor((extent[3] - sub_extent[2])/dy)
    
    # # Create vertices per each grid
    nx = int(math.ceil(abs(maxx - minx)/dx))
    ny = int(math.ceil(abs(maxy - miny)/dy))
    w = shp.Writer(shp_out, shp.POLYGON)
    w.autoBalance = 1
    w.field("ID")
    id=int(top*width+left)  # Initial ID
    for i in range(ny):
        for j in range(nx):
            vertices = []
            parts = []
            vertices.append([min(minx+dx*j,maxx),max(maxy-dy*i,miny)])
            vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*i,miny)])
            vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*(i+1),miny)])
            vertices.append([min(minx+dx*j,maxx),max(maxy-dy*(i+1),miny)])
            parts.append(vertices)
            w.poly(parts)
            w.record(id)
            id+=1
        id+=int(width-nx)
    w.close()
    
    
    # Save a projection file (filename.prj)
    filename, _ = os.path.splitext(shp_out)
    prj = open("%s.prj" % filename, "w")
    epsg = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]' 
    prj.write(epsg)
    prj.close()
    if set_print:
        print('%s is saved.' % shp_out)

        
def CreateGridBox(shp_out, extent, dx, dy, set_print=True):
    '''Create grid with degrees of extent, dx, dy of the target box
    
    Parameters
    ----------
    extent: list
        [minx,maxx,miny,maxy]
    dx: value
        degree of x
    dy: value
        degree of y

    Returns
    -------
    shp_out file is created.
    
    
    Source: https://gis.stackexchange.com/a/81120/29546
    Revised by Donghoon Lee @ Aug-10-2019
    '''
    minx,maxx,miny,maxy = extent
    nx = int(math.ceil(abs(maxx - minx)/dx))
    ny = int(math.ceil(abs(maxy - miny)/dy))
    w = shp.Writer(shp_out, shp.POLYGON)
    w.autoBalance = 1
    w.field("ID")
    id=0
    for i in range(ny):
        for j in range(nx):
            vertices = []
            parts = []
            vertices.append([min(minx+dx*j,maxx),max(maxy-dy*i,miny)])
            vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*i,miny)])
            vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*(i+1),miny)])
            vertices.append([min(minx+dx*j,maxx),max(maxy-dy*(i+1),miny)])
            parts.append(vertices)
            w.poly(parts)
            w.record(id)
            id+=1
    w.close()
    
    # Save a projection file (filename.prj)
    filename, _ = os.path.splitext(shp_out)
    prj = open("%s.prj" % filename, "w")
    epsg = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]' 
    prj.write(epsg)
    prj.close()
    if set_print:
        print('%s is saved.' % shp_out)

        
def CreateAdminDataGrid(fn_out, fn_adm, extent, dx, dy, sub_extent, epsg=20538, set_print=False):
    CreateGridBox_subextent(fn_out, extent, dx, dy, sub_extent, set_print=set_print)
    IntersectShapefiles(fn_out, fn_adm, outSHP=fn_out, set_print=set_print)
    # Reproject to Somalia UTM: UTM zone 37N (EPSG:32637)
    grid_dist = gpd.read_file(fn_out)
    grid_dist = grid_dist.to_crs(epsg=32637)
    # Caculate areas of all split polygons
    grid_dist["area_km2"] = grid_dist['geometry'].area / 10**6
    # Save GeoDataFrame to Shapefile
    grid_dist.to_file(fn_out)
    print('%s is saved.' % fn_out)
        
        
        
    
def save_hdf(filn, df, set_print=True):
    df.to_hdf(filn, key='df', complib='blosc:zstd', complevel=9)
    if set_print:
        print('%s is saved.' % filn)
        
        
# Colarmap and Colorbar controller
def cbarpam(bounds, color, labloc='on', boundaries=None, extension=None):
    '''Returns parameters for colormap and colorbar objects with a specified style.

        Parameters
        ----------
        bounds: list of bounds
        color: name of colormap or list of color names

        labloc: 'on' or 'in'
        boundaries: 
        extension: 'both', 'min', 'max'

        Return
        ------
        cmap: colormap
        norm: nomalization
        vmin: vmin for plotting
        vmax: vmax for plotting
        boundaries: boundaries for plotting
        
        Donghoon Lee @ Mar-15-2020
    '''
    
    gradient = np.linspace(0, 1, len(bounds)+1)
    # Create colorlist
    if type(color) is list:
        cmap = colors.ListedColormap(color,"")
    elif type(color) is str:
        cmap = plt.get_cmap(color, len(gradient))    
        # Extension
        colorsList = list(cmap(np.arange(len(gradient))))
        if extension == 'both':
            cmap = colors.ListedColormap(colorsList[1:-1],"")
            cmap.set_under(colorsList[0])
            cmap.set_over(colorsList[-1])
        elif extension == 'max':
            cmap = colors.ListedColormap(colorsList[:-1],"")
            cmap.set_over(colorsList[-1])
        elif extension == 'min':
            cmap = colors.ListedColormap(colorsList[1:],"")
            cmap.set_under(colorsList[0])
        elif extension == None:
            gradient = np.linspace(0, 1, len(bounds)-1)
            cmap = plt.get_cmap(color, len(gradient))
        else:
            raise ValueError('Check the extension')
    else:
        raise ValueError('Check the type of color.')
    # Normalization
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # vmin and vmax
    vmin=bounds[0]
    vmax=bounds[-1]
    # Ticks
    if labloc == 'on':
        ticks = bounds
    elif labloc == 'in':
        ticks = np.array(bounds)[0:-1] + (np.array(bounds)[1:] - np.array(bounds)[0:-1])/2
    
    return cmap, norm, vmin, vmax, ticks, boundaries


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))