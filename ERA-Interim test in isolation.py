# ERA-Interim test in isolation

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

print 'Started'
nlatcols=10
nlays=60

def interpolate_createprrtminput_lev(shortname,latparray,ps,lats):
                    
    lats = lats
    pressures = ps

    xx,yy = np.meshgrid(lats[::-1],pressures)
    
    if(disttypelev[shortname] == 'lat'):
        
        z = latparray

        f = interpolate.RegularGridInterpolator((lats[::-1],pressures),z.T,bounds_error=False,fill_value=1000.0)

        xnew = latgrid
        ynew = pgrid
        xxnew, yynew = np.meshgrid(xnew,ynew)
        (xxnewr,yynewr) = (xxnew.ravel(),yynew.ravel())
        znew = f((xxnewr,yynewr),method='linear')
        znew=znew.reshape(nlays,nlatcols)
        znew = znew[:,::-1]
        xnew = xnew[::-1]
        ynew = ynew[::-1]
        
    elif(disttypelev[shortname]=='avg'):
        z = latparray
        f = interpolate.RegularGridInterpolator((lats[::-1],pressures),z.T,bounds_error=False,fill_value=1000.0)
        xnew = latgrid
        ynew = pgrid
        xxnew, yynew = np.meshgrid(xnew,ynew)
        (xxnewr,yynewr) = (xxnew.ravel(),yynew.ravel())
        znew = f((xxnewr,yynewr),method='linear')
        znew=znew.reshape(nlays,nlatcols)
        znew = znew[:,::-1]
        xnew = xnew[::-1]
        ynew = ynew[::-1]
        zavg = np.zeros(nlays)
        
        for col in range(0,nlatcols):
                zavg = zavg + znew[:,col] * latweights[col] 

        zavg = zavg / sum(latweights)

    return znew

def interpolate_createprrtminput_sfc(shortname,latarray,lats):

    lats = lats

    z = latarray

    f = interp1d(lats,z)
    varss_int = f(latgrid)
    
    return varss_int

interpdir = '/Users/nickedkins/Dropbox/Input Data for RCM Interpolation/'            

q_ps = np.load(interpdir+'q_ps.npy')
o3_ps = np.load(interpdir+'o3_ps.npy')
q_lats = np.load(interpdir+'q_lats.npy')
o3_lats = np.load(interpdir+'o3_lats.npy')
fal_lats = np.load(interpdir+'fal_lats.npy')

q_latp_max = np.load(interpdir+'q_latp.npy')
o3_latp_max = np.load(interpdir+'o3_latp.npy')
fal_lat_max = np.load(interpdir+'fal_lat.npy')
            
latgrid = np.linspace(-90,90,nlatcols)
pgrid = np.linspace(1000,1,nlays)

# shortnameslev = ['cc','clwc','o3','q','ciwc']
shortnameslev = ['q', 'o3']
longnameslev = {'cc':'Cloud fraction','clwc':'Cloud liquid water content (kg/kg)','o3':'Ozone mixing ratio','q':'Specific humidity (kg/kg)','ciwc':'Cloud ice water content (kg/kg)'}
disttypelev = {'cc':'lat','clwc':'lat','o3':'lat','q':'lat','ciwc':'lat'}

shortnamessfc = ['fal']
longnamessfc = {'fal':'Surface albedo'}
disttypesfc = {'fal':'lat'}

loop = 1

print "Creating input files by interpolating ERA-Interim data"

q=interpolate_createprrtminput_lev('q',q_latp_max,q_ps,q_lats)
o3=interpolate_createprrtminput_lev('o3',o3_latp_max,o3_ps,o3_lats)
fal=interpolate_createprrtminput_sfc('fal',fal_lat_max,fal_lats)

plt.plot(latgrid,fal)

print 'Finished'

show()