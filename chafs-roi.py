import os, time
if __name__ == '__main__': 
    stime = time.time()
    exec(open("01_EODataControl.py").read())
    exec(open("02_CreateDataChunk.py").read())
    exec(open("03_GenerateViewer_COM.py").read())
    exec(open("04_GenerateViewer_SIM.py").read())
    exec(open("05_GenerateGraphics.py").read())
    print('chafs-roi.py took %.0fs' % (time.time() - stime))