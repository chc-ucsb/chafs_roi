import os, time
from chafs_roi.eodata_control.stream_eodata import stream_eodata
from chafs_roi.eodata_control.etos_noaa import etos_noaa
from chafs_roi.eodata_control.tmax_noaa_cpc import tmax_noaa_cpc
from chafs_roi.eodata_control.tmin_noaa_cpc import tmin_noaa_cpc
from chafs_roi.eodata_control.gdd_noaa_cpc import gdd_noaa_cpc
from chafs_roi.eodata_control.prcp_chirps import prcp_chirps
from chafs_roi.eodata_control.prcp_chirps_prelim import prcp_chirps_prelim
from chafs_roi.eodata_control.ndvi_emodis import ndvi_emodis
from chafs_roi.create_input_data import create_input_data
from chafs_roi.generate_viewer_com import generate_viewer_com
from chafs_roi.generate_viewer_sim import generate_viewer_sim
from chafs_roi.generate_graphics import generate_graphics

def main(): 
    stime = time.time()
    
    # Earth Observation (EO) Data Control
    # 1) NDVI_eMODIS is
    # retrieved from https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/viewer_G5/emodisndvic6v2_africa_dekad_data
    # stored at /home/chc-sandbox/people/dlee/ndvi_emodis_mirror/
    # mirrored to /home/chc-sandbox/people/dlee/ndvi_emodis/
    # 2) Temperature Max and Min (Growing Degree Day is calculated from Tmax and Tmin)
    # retrieved from ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp/
    # stored at /home/chc-sandbox/people/dlee/temp_noaa-cpc/
    # 3) Reference ET is
    # retrieved from ftp://ftp.cdc.noaa.gov/Projects/RefET/global/Gen-0/fine_resolution/data_v2/
    # stored at /home/chc-sandbox/people/dlee/etos_noaa/
    stream_eodata()         # Retrieve the latest data into ROI

    # 
    etos_noaa()             # Spatial means of ETo
    tmax_noaa_cpc()         # Spatial means of Tmax
    tmin_noaa_cpc()         # Spatial means of Tmax
    gdd_noaa_cpc()          # Spatial means of Growing Degree Days
    prcp_chirps()           # Spatial means of CHIRPS
    prcp_chirps_prelim()    # Spatial means of CHIRPS Prelim
    ndvi_emodis()           # Spatial means of eMODIS NDVI

    # Create Input Data
    create_input_data()     # Create input data files for forecasting

    # Forecasting: Complicated Approach
    generate_viewer_com()   # Generate "viewer_data_com.csv"

    # Forecasting: Simplified Approach
    generate_viewer_sim()   # Generate "viewer_data_sim.csv" and "viewer_data.csv"

    # Generate Graphics
    generate_graphics()     # Generate graphics
    
    print('chafs_roi.py took %.ds' % (round(time.time() - stime)))



if __name__ == '__main__': 
    main()


