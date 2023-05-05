import os, time
from chafs_roi.eodata_control.stream_eodata import stream_eodata
from chafs_roi.eodata_control.etos_noaa import etos_noaa
from chafs_roi.eodata_control.tmax_noaa_cpc import tmax_noaa_cpc
from chafs_roi.eodata_control.tmin_noaa_cpc import tmin_noaa_cpc
from chafs_roi.eodata_control.gdd_noaa_cpc import gdd_noaa_cpc
from chafs_roi.eodata_control.prcp_chirps import prcp_chirps
from chafs_roi.eodata_control.prcp_chirps_prelim import prcp_chirps_prelim
from chafs_roi.eodata_control.ndvi_emodis import ndvi_emodis
from chafs_roi.eodata_control.ndvi_eviirs import ndvi_eviirs
from chafs_roi.eodata_control.smos_fldas import smos_fldas
from chafs_roi.eodata_control.stmp_fldas import stmp_fldas
from chafs_roi.eodata_control.atmp_fldas import atmp_fldas
from chafs_roi.create_input_data import create_input_data
from chafs_roi.generate_viewer_com import generate_viewer_com
from chafs_roi.generate_viewer_sim1 import generate_viewer_sim1
from chafs_roi.generate_viewer_sim2 import generate_viewer_sim2
from chafs_roi.generate_viewer_eo import generate_viewer_eo
from chafs_roi.generate_graphics import generate_graphics

def main(): 
    stime = time.time()
    print('chafs_roi.py started at %s.' % (time.ctime()))

    # Earth Observation Data Control
    print('stream_eodata() starts...')
    stream_eodata()         # Retrieve the latest data into ROI
    print('etos_noaa() starts...')
    etos_noaa()             # Spatial means of ETo
    print('tmax_noaa_cpc() starts...')
    tmax_noaa_cpc()         # Spatial means of Tmax
    print('tmin_noaa_cpc() starts...')
    tmin_noaa_cpc()         # Spatial means of Tmax
    print('gdd_noaa_cpc() starts...')
    gdd_noaa_cpc()          # Spatial means of Growing Degree Days
    print('prcp_chirps() starts...')
    prcp_chirps()           # Spatial means of CHIRPS
    print('prcp_chirps_prelim() starts...')
    prcp_chirps_prelim()    # Spatial means of CHIRPS Prelim
    print('ndvi_eviirs() starts...')
    ndvi_eviirs()           # Spatial means of eVIIRS NDVI
    print('smos_fldas() starts...')
    smos_fldas()            # Spatial means of FLDAS Soil Moisture
    print('stmp_fldas() starts...')
    stmp_fldas()            # Spatial means of FLDAS Soil Temperature
    print('atmp_fldas() starts...')
    atmp_fldas()            # Spatial means of FLDAS Air Temperature
    ### ndvi_emodis()           # eMODIS NDVI is deprecated

    # Create Input Data
    print('create_input_data() starts...')
    create_input_data()     # Create input data files for forecasting

    # Forecasting: Complicated Approach
    # print('generate_viewer_com() starts...')
    # generate_viewer_com()   # Generate "viewer_data_com.csv"

    # Forecasting: Simplified Approach
    # print('generate_viewer_sim1() starts...')
    # generate_viewer_sim1()
    # print('generate_viewer_sim2() starts...')
    # generate_viewer_sim2()

    # Earth Observation
    # print('generate_viewer_eo() starts...')
    # generate_viewer_eo()    # Generate "viewer_data_eo.csv"  (fixing..)

    # Generate Graphics
    # print('generate_graphics() starts...')
    # generate_graphics()     # Generate graphics

    print('chafs_roi.py finished at %s (%ds took).' % (time.ctime(), time.time()-stime))

if __name__ == '__main__': 
    main()