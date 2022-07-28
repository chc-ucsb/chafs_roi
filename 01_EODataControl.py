import os, time
if __name__ == '__main__': 
    stime = time.time()
    exec(open("P00_Stream_EOdata.py").read())
    exec(open("P11_ETOS_NOAA.py").read())
    exec(open("P12_TMAX_NOAA-CPC.py").read())
    exec(open("P13_TMIN_NOAA-CPC.py").read())
    exec(open("P14_GDD_NOAA-CPC.py").read())
    exec(open("P15_PRCP_CHIRPS.py").read())
    exec(open("P16_PRCP_CHIRPS_prelim.py").read())
    exec(open("P17_NDVI_eMODIS.py").read())
    print('01_EODataControl.py took %.0fs' % (time.time() - stime))