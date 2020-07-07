# ERA-Interim and MISR test in isolation

import numpy as np
import os
import subprocess
import time
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import math
from pylab import *
import matplotlib.pyplot as plt
from scipy import interpolate
from netCDF4 import Dataset
from scipy import interpolate, stats
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, RegularGridInterpolator
from os import listdir
from mpl_toolkits.basemap import Basemap, shiftgrid
from time import localtime, strftime
from scipy import stats
from pandas import *

interpdir = '/Users/nickedkins/Dropbox/Input Data for RCM Interpolation/'
misr_cf_latalt_max = np.load(interpdir+'misr_cf_latalt.npy')
misr_cf_latalt_max = np.nan_to_num(misr_cf_latalt_max)
cfs = misr_cf_latalt_max
misr_alts = np.linspace(0,20,num=39)
misr_lats = np.linspace(-90,90,num=360)
latbins = np.linspace(-90,90,num=ncols)
altbin_edges = np.linspace(0,10,num=ncloudcols+1)
altbins = np.zeros(ncloudcols)
cld_taus = np.zeros(ncloudcols)

for i in range(len(altbins)+1):
	altbins[i-1] = (altbin_edges[i-1] + altbin_edges[i])/2.0

od_low = 3.0 * 0.0
od_mid = 3.0 * 0.0
od_high = 0.3 * 0.0

extra_cld_tau = 0.3 
extra_cld_frac = 0.2
extra_cld_alt = 2.0
extra_cld_latcol = 2
extra_cld_cldcol = 2

for i in range(len(altbins)):
    if (altbins[i] < 3.0):
        cld_taus[i] = od_low
    elif(altbins[i] < 7.0):
        cld_taus[i] = od_mid
    else:
        cld_taus[i] = od_high

binned_cf_lat = np.zeros((ncols,len(misr_alts)))
cf_lat_bincounts = np.zeros((ncols,len(misr_alts)))

binned_cf = np.zeros((ncols,ncloudcols))
cf_bincounts = np.zeros((ncols,ncloudcols))

xbounds = np.linspace(-1,1,ncols+1)
latbounds = np.rad2deg(np.arcsin(xbounds))
collats = np.zeros(ncols)
colwidth = 180.0 / (ncols+1)

for i in range(1,len(latbounds)):
    collats[i-1] = (latbounds[i-1] + latbounds[i])/2
    
for i in range(len(misr_lats)):
    ibin = np.argmin(abs(misr_lats[i] - collats))
    for j in range(len(misr_alts)):
        binned_cf_lat[ibin,j] = binned_cf_lat[ibin,j] + cfs[i,j]
        cf_lat_bincounts[ibin,j] = cf_lat_bincounts[ibin,j] + 1

binned_cf_lat= binned_cf_lat / cf_lat_bincounts

for ibin in range(ncols):
    for j in range(len(misr_alts)):
        jbin = np.argmin(abs(misr_alts[j] - altbins))
        binned_cf[ibin,jbin] = binned_cf[ibin,jbin] + binned_cf_lat[ibin,j]
        cf_bincounts[ibin,jbin] = cf_bincounts[ibin,jbin] + 1

clearfrac = np.zeros(ncols)

for col in range(ncols):
    tempcloudfrac = 0.0
    for cloudcol in range(ncloudcols-1):
        tempcloudfrac = tempcloudfrac + binned_cf[col,cloudcol] - tempcloudfrac * binned_cf[col,cloudcol]
    clearfrac[col] = (1.0 - tempcloudfrac) / 2.0

# for col in range(ncols):
#     filename = 'ccfracs col %2d' % (col)
#     fileloc = outdir + filename
#     file = open(fileloc,'w')
#     for cloudcol in range(ncloudcols):
#         file.write(str(binned_cf[col,cloudcol]))
#         file.write('\n')
#     file.write(str(clearfrac[col]))
#     file.write('\n')
#     file.write(str(extra_cld_frac))
#     #if(col==extra_cld_latcol):
#     #    file.write(str(extra_cld_frac))
#     #else:
#     #    file.write(str('0.0'))
#     file.close()

# for col in range(ncols):
#     filename = 'ccalts col %2d' % (col)
#     fileloc = outdir + filename
#     file = open(fileloc,'w')
#     for cloudcol in range(ncloudcols):
#         file.write(str(altbins[cloudcol]))
#         file.write('\n')
#     file.write('1.0')
#     file.write('\n')
#     file.write(str(extra_cld_alt))
#     file.close()


# for col in range(ncols):
#     filename = 'cctaus col %2d' % (col)
#     fileloc = outdir + filename
#     file = open(fileloc,'w')
#     for cloudcol in range(ncloudcols):
#         file.write(str(cld_taus[cloudcol]))
#         file.write('\n')
#     file.write('0.0')
#     file.write('\n')
#     file.write(str(extra_cld_tau))
#     file.close()

##################################################################################################################################################################################################################################################

show()