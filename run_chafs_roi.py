#!/home/donghoonlee/anaconda3/envs/chafs_b/bin/python
import sys, os, json
from subprocess import Popen, PIPE
from itertools import product, compress
import time
import subprocess

log = open('./run_chafs_roi.log', 'w')
log.write('======================================\n')
log.write('This is output of "run_chafs_roi.py"\n')
log.write('======================================\n')
log.flush()
command = 'conda run -n chafs_b python chafs_roi.py'
process = subprocess.Popen(['bash', '-c', command], stdout=log, stderr=log, shell=False, text=True)