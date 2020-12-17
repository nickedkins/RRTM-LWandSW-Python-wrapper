# RRTM LW and SW Python wrapper

import numpy as np
import os
import subprocess
import time
from subprocess import Popen, PIPE, STDOUT
import matplotlib.pyplot as plt
from pylab import *
import datetime
from random import randint
from scipy import interpolate, stats
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, RegularGridInterpolator

tstart = datetime.datetime.now()
project_dir = '/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/'

print('testing https')

def init_plotting():
    plt.rcParams['figure.figsize'] = (10,10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    # plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3    
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1   
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['lines.linewidth'] = 2.0 
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
init_plotting()

def logpplot(x,p,xlab,ylab,color='blue'):
    plt.semilogy(x,p,'-',c=color,alpha=(float(ts)/timesteps)**1.0)
    plt.ylim(max(p),min(p))
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def loglogplot(x,p,xlab,ylab):
    plt.loglog(x,p,'-')
    plt.ylim(max(p),min(p))
    plt.xlabel(xlab)
    plt.ylabel(ylab)

# write a 1D array of params to file
def writeparams(params,f):  
    for i in params:
        f.write(str(i))
        f.write('\n')

# write a list of 1D arrays of params to file
def writeparamsarr(params,f):
    for param in params:
        for i in range(len(param)):
            f.write(str(param[i]))
            f.write('\n')

# write an input file to be read by the LW version of rrtm.f (matching the format of the example input files)
def writeformattedinputfile_sw():
    f=open('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/SW/Input RRTM SW NJE Formatted','w+')
    f.write('INPUT_RRTM_SW NJE created\n')
    f.write('0        1         2         3         4         5         6         7         8         9\n')
    f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
    f.write('$ Dynamic Formatted RRTM SW Input\n')
    f.write('                  {:2d}                             {:1d}                                {:1d} {:1d}   {:02d}    {:1d}   {:1d}{:1d}\n'.format(iaer,iatm,iscat,istrm,iout,icld,idelm,icos))
    f.write('            {:3d}   {:7.3f}    {:1d}'.format(juldat,sza,isolvar))
    for i in range(29):
        f.write('{:5.3f}'.format(solvar[i]))
    f.write('\n')
    f.write('           {:1d}  {:1d}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}\n'.format(iemis,ireflect,semiss[15],semiss[16],semiss[17],semiss[18],semiss[19],semiss[20],semiss[21],semiss[22],semiss[23],semiss[24],semiss[25],semiss[26],semiss[27],semiss[28] ))
    f.write(' {:1d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
    f.write('{:11.4f}{:17.5f}{:10s}{:3s}{:2d}{:8.3f}{:8.2f}{:7.2f}{:7.3f}{:8.2f}{:10.5f}\n'.format(pavel[0],tavel[0],secntk,cinp,ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
    f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
    for i in range(2,nlayers+1):
        if(pavel[i-1]<0.1):
            f.write('  {:13.7f}{:13.5f}{:15.0f}{:30.3f}{:7.2f}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        elif(pavel[i-1]<1.):
            f.write('  {:9.7f}{:17.5f}{:15.0f}{:30.3f}{:7.2f}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        elif(pavel[i-1]<10.):
            f.write('   {:8.6f}{:17.5f}{:15.0f}{:30.3f} {:7.2f}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        elif(pavel[i-1]<100.):
            f.write('   {:8.5f}{:17.5f}{:15.0f}{:30.3f} {:7.2f}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        else:
            f.write('   {:8.4f}{:17.5f}{:15.0f}{:30.3f} {:7.2f}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
    f.write('%%%%%\n')
    f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
    f.close()

# same as above but for SW, which has a slightly different format
def writeformattedinputfile_lw():
    f=open('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/LW/Input RRTM LW NJE Formatted','w+')
    f.write('INPUT_RRTM_SW NJE created\n')
    f.write('0        1         2         3         4         5         6         7         8         9\n')
    f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
    f.write('$ STANDARD MID-LATITUDE SUMMER ATMOSPHERE\n')
    f.write('{:50d}{:20d}{:13d}{:2d}{:5d}{:5d}\n'.format(iatm,ixsect,iscat,numangs,iout,icld))
    f.write('{:16.11f} {:1d}  {:1d}'.format(tbound,iemiss,ireflect)) #add semis read here?
    for i in semis:
        f.write('{:5.3f}'.format(i))
    f.write('\n')
    f.write('{:2d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
    f.write('{:21.14f}{:25.13f}{:15d}{:18.13f}{:18.12f}{:17.12f}{:17.13f}{:18.12f}{:17.12f}\n'.format(pavel[0],tavel[0],ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
    f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
    for i in range(2,nlayers+1):
        f.write('{:21.14f}{:25.13f}{:15.0f}{:41.14f}{:18.13f}{:17.12f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
        f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
    f.write('%%%%%\n')
    f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')

# write formatted cloud file for rrtm input
def writeformattedcloudfile():
    f=open('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/IN_CLD_RRTM NJE','w+')
    f.write('   {:2d}    {:1d}    {:1d}\n'.format(inflags_master[i_zon,i_lat].astype(int),iceflags_master[i_zon,i_lat].astype(int),liqflags_master[i_zon,i_lat].astype(int)))
    for i_cld in range(nclouds):
        f.write('{} {:3d}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n'.format(ctest,int(cld_lays_master[i_zon,i_lat,i_cld]),cld_fracs_master[i_zon,i_lat,i_cld],tauclds_master[i_zon,i_lat,i_cld],ssaclds_master[i_zon,i_lat,i_cld],radice,radliq))
    f.write('%\n')
    f.write('123456789-123456789-123456789-123456789-123456789-123456789-\n')
    f.write('\n')
    f.write('\n')

# call the compiled rrtmlw executable
def callrrtmlw():
    loc = '/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/LW/rrtmlw'
    os.chdir(project_dir+'/LW')
    p = subprocess.Popen([loc])
    stdoutdata, stderrdata = p.communicate()

# ditto for SW
def callrrtmsw():
    loc = '/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/SW/rrtmsw'
    os.chdir(project_dir+'/SW')
    p = subprocess.Popen([loc])
    stdoutdata, stderrdata = p.communicate()

# read output produced by rrtmlw executable for use in next timestep within this python wrapper
def readrrtmoutput_lw():
    f=open('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/LW/My Live Output RRTM')
    for i in range(0,nlayers+1):
        totuflux_lw[i] =  f.readline()
    for i in range(0,nlayers+1):
        totdflux_lw[i] =  f.readline()
    for i in range(0,nlayers+1):
        fnet_lw[i] =  f.readline()
    for i in range(0,nlayers+1):
        htr_lw[i] =  f.readline()
    return totuflux_lw,totdflux_lw,fnet_lw,htr_lw

def readrrtmoutput_sw():
    f=open('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/SW/My Live Output RRTM')
    for i in range(0,nlayers+1):
        totuflux_sw[i] =  f.readline()
    for i in range(0,nlayers+1):
        totdflux_sw[i] =  f.readline()
    for i in range(0,nlayers+1):
        fnet_sw[i] =  f.readline()
    for i in range(0,nlayers+1):
        htr_sw[i] =  f.readline()
    return totuflux_sw,totdflux_sw,fnet_sw,htr_sw

# plot main rrtm variables 
def plotrrtmoutput():
    plt.figure(1)
    plt.subplot(331)
    plt.semilogy(tz,pz)
    plt.ylim(max(pz),min(pz))
    plt.xlabel('tz')
    plt.ylabel('pz')
    plt.subplot(332)
    plt.semilogy(fnet,pz,c='b',label='total')
    plt.semilogy(fnet_lw,pz,c='r',label='lw')
    plt.semilogy(fnet_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('fnet')
    plt.ylabel('pz')
    plt.legend()
    plt.subplot(333)
    plt.semilogy(totuflux,pz,c='b',label='total')
    plt.semilogy(totuflux_lw,pz,c='r',label='lw')
    plt.semilogy(totuflux_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('totuflux')
    plt.ylabel('pz')
    plt.legend()
    plt.subplot(334)
    plt.semilogy(totdflux,pz,c='b',label='total')
    plt.semilogy(totdflux_lw,pz,c='r',label='lw')
    plt.semilogy(totdflux_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('totdflux')
    plt.ylabel('pz')
    plt.legend()
    plt.subplot(335)
    plt.semilogy(tavel,pavel,'o',c='b')
    plt.ylim(max(pz),min(pz))
    plt.subplot(336)
    logpplot(wbrodl,pavel,'wbrodl','pavel')
    plt.subplot(337)
    logpplot(wkl[1,:],pavel,'wkl1 (h2o)','pavel')
    plt.subplot(338)
    logpplot(wkl[2,:],pavel,'wkl2 (co2)','pavel')
    plt.subplot(339)
    logpplot(wkl[3,:],pavel,'wkl3 (o3)','pavel')

# perform a basic convective adjustment, given T and z
def convection(T,z,conv_log):
    T[0]=tbound
    for i in range(1,len(T)):
        dT = (T[i]-T[i-1])
        dz = (z[i]-z[i-1])/1000.
        if( (-1.0 * dT/dz > lapse or z[i]/1000. < 0.) and z[i]/1000. < 1000. ):
            if(conv_log==1):
                conv[i]=1.
            T[i] = T[i-1] - lapse * dz

# same but for moist adiabatic lapse rate            
def convection_moist(T,z,conv_log):
    T[0]=tbound
    for i in range(1,len(T)):
        dT = (T[i]-T[i-1])
        dz = (z[i]-z[i-1])/1000.
        if( (-1.0 * dT/dz > gamma_m[i] or z[i]/1000. < 0.) and z[i]/1000. < 1000. ):
            if(conv_log==1):
                conv[i]=1.
            T[i] = T[i-1] - gamma_m[i] * dz

# write output file containing all atmospheric variables, with timestamped title, for later analysis
def writeoutputfile_masters():
    tlabel = datetime.datetime.now()
    tlabelstr = str(tlabel.strftime('%Y_%m_%d %H_%M_%S'))
    f = open(project_dir+'_Raw Output Data/'+tlabelstr,'w+')

    for x in vars_0d:
        f.write(str(x))
        f.write('\n')

    for x in vars_master_lay_zon_lat:
        for k in range(nlatcols):
            for j in range(nzoncols):
                for i in range(nlayers):
                    f.write(str(x[i,j,k]))
                    f.write('\n')

    for x in vars_master_lev_zon_lat:
        for k in range(nlatcols):
            for j in range(nzoncols):
                for i in range(nlayers+1):
                    f.write(str(x[i,j,k]))
                    f.write('\n')

    i_lens=0
    for x in vars_misc_1d:
        for i in range(vars_misc_1d_lens[i_lens]):
            f.write(str(x[i]))
            f.write('\n')
        i_lens+=1

    for x in vars_master_lay_zon_nmol_lat:
        for l in range(nlatcols):
            for k in range(nmol+1):
                for j in range(nzoncols):
                    for i in range(nlayers):
                        f.write(str(x[i,j,k,l]))
                        f.write('\n')
                        
    for x in vars_master_zon_lat:
        for j in range(nlatcols):
            for i in range(nzoncols):
                f.write(str(x[i,j]))
                f.write('\n')

    for x in vars_master_zon_lat_cld:
        for k in range(nclouds):
            for j in range(nlatcols):
                for i in range(nzoncols):
                    f.write(str(x[i,j,k]))
                    f.write('\n')
                    
    for x in vars_master_lat:
        for i in range(nlatcols):
            f.write(str(x[i]))
            f.write('\n')

# use array of previously collated MISR data to interpolate from to get cloud fractions for a given height
# also assign cld_tau based on if statements within this function (NOT from MISR data)
def read_misr():
    interpdir = '/Users/nickedkins/Dropbox/Input Data for RCM Interpolation/'
    misr_cf_latalt_max = np.load(interpdir+'misr_cf_latalt.npy')
    misr_cf_latalt_max = np.nan_to_num(misr_cf_latalt_max)
    cfs = misr_cf_latalt_max
    misr_alts = np.linspace(0,20,num=39) # previously num=390
    misr_lats = np.linspace(-90,90,num=360)
    latbins = np.linspace(-90,90,num=nlatcols)
    # altbin_edges = np.linspace(0,10,num=nclouds+1)
    altbin_edges=altz/1000.
    altbins = np.zeros(nclouds)
    cld_taus = np.zeros(nclouds)
    for i in range(len(altbins)+1):
        altbins[i-1] = (altbin_edges[i-1] + altbin_edges[i])/2.0
    od_low = 3.0
    od_mid = 3.0
    od_high = 0.3
    for i_lat in range(nlatcols):
        for i in range(len(altbins)):
            if (altbins[i] < 3.0):
                cld_taus[i] = od_low
            elif(altbins[i] < 7.0):
                cld_taus[i] = od_mid
            else:
                cld_taus[i] = od_high
    binned_cf_lat = np.zeros((nlatcols,len(misr_alts)))
    cf_lat_bincounts = np.zeros((nlatcols,len(misr_alts)))
    binned_cf = np.zeros((nlatcols,nclouds))
    cf_bincounts = np.zeros((nlatcols,nclouds))
    xbounds = np.linspace(-1,1,nlatcols+1)
    latbounds = np.rad2deg(np.arcsin(xbounds))
    collats = np.zeros(nlatcols)
    colwidth = 180.0 / (nlatcols+1)
    for i in range(1,len(latbounds)):
        collats[i-1] = (latbounds[i-1] + latbounds[i])/2
    for i in range(len(misr_lats)):
        ibin = np.argmin(abs(misr_lats[i] - collats))
        for j in range(len(misr_alts)):
            binned_cf_lat[ibin,j] = binned_cf_lat[ibin,j] + cfs[i,j]
            cf_lat_bincounts[ibin,j] = cf_lat_bincounts[ibin,j] + 1
    binned_cf_lat= binned_cf_lat / cf_lat_bincounts
    for ibin in range(nlatcols):
        for j in range(len(misr_alts)):
            jbin = np.argmin(abs(misr_alts[j] - altbins))
            binned_cf[ibin,jbin] = binned_cf[ibin,jbin] + binned_cf_lat[ibin,j]
            cf_bincounts[ibin,jbin] = cf_bincounts[ibin,jbin] + 1
    clearfrac = np.zeros(nlatcols)
    for col in range(nlatcols):
        tempcloudfrac = 0.0
        for cloudcol in range(nclouds-1):
            tempcloudfrac = tempcloudfrac + binned_cf[col,cloudcol] - tempcloudfrac * binned_cf[col,cloudcol]
        clearfrac[col] = (1.0 - tempcloudfrac) / 2.0
    return binned_cf[i_lat,:],altbins,cld_taus

# used 2D interpolation to get MISR cloud fraction for a given latitude and altitude
def read_misr_2():
    interpdir = '/Users/nickedkins/Dropbox/Input Data for RCM Interpolation/'
    misr_cf_latalt_max = np.load(interpdir+'misr_cf_latalt.npy')
    misr_cf_latalt_max = np.nan_to_num(misr_cf_latalt_max)
    cfs = misr_cf_latalt_max
    misr_alts = np.linspace(0,20,num=39)
    misr_lats = np.linspace(-90,90,num=360)
    x=misr_lats
    y=misr_alts
    z=cfs.T
    f = interpolate.interp2d(x, y, z, kind='linear')
    xx,yy=np.meshgrid(latgrid,altz/1000.)
    zz=f(latgrid,altz/1000.)
    od_low = 1.5/50.
    od_mid = 1.5/50.
    od_high = 0.15/50.
    cld_taus = np.zeros(nclouds)
    for i in range(len(altz)-1):
        if (altz[i]/1000. < 3.0):
            cld_taus[i] = od_low
        elif(altz[i]/1000. < 7.0):
            cld_taus[i] = od_mid
        elif(altz[i]/1000.) < 15.0:
            cld_taus[i] = od_high
        else:
            cld_taus[i] = 1e-3


    return zz[1:,:].T, altz, cld_taus


# use actual cloud optical depths from MISR (as well as fractions)
def read_misr_3():
    interpdir='/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/MISR Data/'
    misr_cf_latalt_max=np.load(interpdir+'fracs_latalt.npy')
    cfs=misr_cf_latalt_max
    misr_cod_latalt_max=np.load(interpdir+'od_wghtd_latalt.npy')
    cods=misr_cod_latalt_max
    misr_alts=np.load(interpdir+'alts.npy')
    misr_lats=np.load(interpdir+'lats.npy')
    x=misr_lats[1:,0]
    y=misr_alts[:,0]
    
    z1=cfs

    f1 = interpolate.interp2d(x, y, z1, kind='linear')
    xx,yy=np.meshgrid(latgrid,altz/1000.)
    zz1=f1(latgrid,altz/1000.)

    z2=cods
    f2 = interpolate.interp2d(x, y, z2, kind='linear')
    xx,yy=np.meshgrid(latgrid,altz/1000.)
    zz2=f2(latgrid,altz/1000.)

    return zz1[1:,:].T, altz, zz2[1:,:].T

# same as misr_3 but calculate an 'effective cloud fraction' by multiplying MISR frac by MISR COD
def read_misr_4():
    interpdir='/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/MISR Data/'
    misr_cf_latalt_max=np.load(interpdir+'fracs_latalt.npy')
    cfs=misr_cf_latalt_max
    misr_cod_latalt_max=np.load(interpdir+'od_wghtd_latalt.npy')
    cods=misr_cod_latalt_max
    misr_alts=np.load(interpdir+'alts.npy')
    misr_lats=np.load(interpdir+'lats.npy')
    x=misr_lats[1:,0]
    y=misr_alts[:,0]
    
    z1=cfs

    f1 = interpolate.interp2d(x, y, z1, kind='linear')
    xx,yy=np.meshgrid(latgrid,altz/1000.)
    zz1=f1(latgrid,altz/1000.)

    z2=cods
    f2 = interpolate.interp2d(x, y, z2, kind='linear')
    xx,yy=np.meshgrid(latgrid,altz/1000.)
    zz2=f2(latgrid,altz/1000.)
    
    for i_lat in range(nlatcols):    
        h_eff=np.sum(zz1[:,i_lat]*altz)
        cld_lay_v2[i_lat] = np.argmin(np.abs(altz-h_eff))
        cf_mro[i_lat] = 0.
        for i in range(nlayers):
            cf_mro[i_lat] = cf_mro[i_lat] + zz1[i,i_lat] - (cf_mro[i_lat] * zz1[i,i_lat])
        od_eff[i_lat]=np.sum(zz1[:,i_lat]*zz2[:,i_lat])
    
    return cld_lay_v2[i_lat], cf_mro[i_lat], od_eff[i_lat]

#  use array of previously collated ERA-Interim data to interpolate from to get h2o and o3 mixing ratios and surface albedos
def read_erai():

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
            znew=znew.reshape(nlayers,nlatcols)
            # znew = znew[:,::-1]
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
            znew=znew.reshape(nlayers,nlatcols)
            znew = znew[:,::-1]
            xnew = xnew[::-1]
            ynew = ynew[::-1]
            zavg = np.zeros(nlayers)
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
    r_ps = np.load(interpdir+'r_ps.npy')
    o3_ps = np.load(interpdir+'o3_ps.npy')
    q_lats = np.load(interpdir+'q_lats.npy')
    r_lats = np.load(interpdir+'r_lats.npy')
    o3_lats = np.load(interpdir+'o3_lats.npy')
    fal_lats = np.load(interpdir+'fal_lats.npy')
    q_latp_max = np.load(interpdir+'q_latp.npy')
    r_latp_max = np.load(interpdir+'r_latp.npy')
    o3_latp_max = np.load(interpdir+'o3_latp.npy')
    fal_lat_max = np.load(interpdir+'fal_lat.npy')
    # latgrid = np.linspace(-90,90,nlatcols)
    pgrid = np.linspace(1000,1,nlayers)
    # shortnameslev = ['cc','clwc','o3','q','ciwc']
    shortnameslev = ['q', 'o3', 'r']
    longnameslev = {'cc':'Cloud fraction','clwc':'Cloud liquid water content (kg/kg)','o3':'Ozone mixing ratio','q':'Specific humidity (kg/kg)','ciwc':'Cloud ice water content (kg/kg)', 'r':'Relative Humidity'}
    disttypelev = {'cc':'lat','clwc':'lat','o3':'lat','q':'lat','ciwc':'lat', 'r':'lat'}
    shortnamessfc = ['fal']
    longnamessfc = {'fal':'Surface albedo'}
    disttypesfc = {'fal':'lat'}
    loop = 1
    q=interpolate_createprrtminput_lev('q',q_latp_max,q_ps,q_lats)
    r=interpolate_createprrtminput_lev('r',r_latp_max,r_ps,r_lats)
    o3=interpolate_createprrtminput_lev('o3',o3_latp_max,o3_ps,o3_lats)*28.964/48.
    fal=interpolate_createprrtminput_sfc('fal',fal_lat_max,fal_lats)
    return q, o3, fal,r

# read an input distribution of a variable vs latitude and interpolate to model lat grid
def createlatdistbn(filename):
    fileloc = project_dir+'/Latitudinal Distributions/'+filename+'.txt'
    file = open(fileloc,'r')
    lat = []
    var = []
    with file as f:
        for l in f:
            lat.append(float(l.split(',')[0]))
            var.append(float(l.split(',')[1]))
    # lat = data[:,0]
    # var = data[:,1]    
    f = interpolate.interp1d(lat,var)
    templats = np.linspace(-90,90,60)
    varinterp = np.zeros(len(latgrid))
    cossums=np.zeros(len(latgrid))
    for j in range(len(latgridbounds)-1):
        for i in range(len(templats)):
            if (templats[i] > latgridbounds[j] and templats[i] < latgridbounds[j+1]):
                varinterp[j] += f(templats[i]) * np.cos(np.deg2rad(templats[i]) )
                cossums[j] += np.cos(np.deg2rad(templats[i]))
    varinterp = list(varinterp / cossums)
    return varinterp

#################functions###########################################################

# set overall dimensions for model
nlayers=60 # number of vertical layers
nzoncols=2 # number of zonal columns (usually just 2: cloudy and clear)
nlatcols=1 # number of latitude columns

# latgridbounds=[-90,-66.5,-23.5,23.5,66.5,90] # 5 box poles, subtropics, tropics

# create latgrid evenly spaced in latitude
latgridbounds=np.linspace(60,90.,nlatcols+1)
xgridbounds=np.sin(np.deg2rad(latgridbounds))

# create latgrid evenly spaced in cos(lat)
# xgridbounds=np.linspace(-0.,1.,nlatcols+1)
# latgridbounds=np.rad2deg(np.arcsin(xgridbounds))

latgrid=np.zeros(nlatcols)
for i in range(nlatcols):
    latgrid[i]=(latgridbounds[i]+latgridbounds[i+1])/2.

# create array of weights to multiply any variable by to get the area-weighted version ( if latgrid evenly spaced in cos(lat), latweights_area = 1 )
latweights_area=np.zeros(nlatcols)
for i_xg in range(len(xgridbounds)-1):
    latweights_area[i_xg]=xgridbounds[i_xg+1]-xgridbounds[i_xg]
latweights_area/=np.mean(latweights_area)

# if there's only one lat column, pick its lat and set some nearby boundaries to enable interpolation over short interval
if(nlatcols==1):
    latgrid=np.array([80.])
    latgridbounds=[latgrid[0]-5.,latgrid[0]+5.]

nmol=7 # number of gas molecule species
# nclouds=10
nclouds=nlayers # number of cloud layers

lw_on=1 # 0: don't call rrtm_lw, 1: do
sw_on=1 # 0: don't call rrtm_sw, 1: do
gravity=9.79764 # RCEMIP value
avogadro=6.022e23 # avogadro's constant
iatm=0 #0 for layer values, 1 for level values
ixsect=0 #could be 1, but why?
iscat=0 #just absorption and emission
# numangs=0 #can be 0-4 for higher precision
numangs=3 #can be 0-4 for higher precision 3 for SW, 0 for LW
# iout=99 #for info in all spectral bands
iout=0 #for broadband only
# iout=-1 #for broadband, no printings
# iout=29
# icld=0 #for clear sky
icld=1  #for grey clouds
ur_min=0.6 # minimum value for under-relaxation constant
ur_max=3.0 # minimum value for under-relaxation constant
eqb_maxhtr=1e-4 # equilibrium defined as when absolute value of maximum heating rate is below this value (if not using dfnet to determine eqb)
# eqb_maxdfnet=1e-4

eqb_maxdfnet=0.1*(60./nlayers) # equilibrium defined as when absolute value of maximum layer change in net flux is below this value (if not using htr to determine eqb)
eqb_col_budgs=1.0e12 # max equilibrium value of total column energy budget at TOA
timesteps=2000 # number of timesteps until model exits
maxdfnet_tot=1.0 # maximum value of dfnet for and lat col and layer (just defining initial value here)
toa_fnet_eqb=1.0e12 # superseded now by eqb_col_budgs, but leave in for backward compatibility so I can read old files


# master switches for the basic type of input
master_input=6 #0: manual values, 1: MLS, 2: MLS RD mods, 3: RDCEMIP, 4: RD repl 'Nicks2', 5: Pierrehumbert95 radiator fins, 6: ERA-Interim
input_source=0 # 0: set inputs here, 1: use inputs from output file of previous run, 2: use outputs of previous run and run to eqb
prev_output_file='/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/baselines/nl=590,ncol=5 [NH only]'
lapse_sources=[4] # 0: manual, 1: Mason ERA-Interim values, 2: Hel82 param, 3: SC79, 4: CJ19 RAE only

adv_locs=[0] # 0: heating everywhere, 1: heating only in tropopause
nbs=[2] # power for power law scaling of z distbn of heating from horizontal transport (from XXX)
adv_on=np.zeros(nlatcols) # 0: no advective heating, 1: advective heating on
if(nlatcols==1):
  adv_on=0

zen=np.zeros((nlatcols,365,24)) # solar zenith angle
insol=np.zeros((nlatcols,365,24)) # insolation
zenlats=np.zeros(nlatcols) 
insollats=np.zeros(nlatcols)
solar_constant=1368.

# lapseloops=np.arange(4,11) # global average critical lapse rates to loop over
lapseloops=[6]

c_zonals=[0.] #zonal transport coefficient
c_merids=[0.] #meridional transport coefficient

extra_forcings=[0.] # add an extra TOA forcing to any box
fixed_sws=np.array([340.]) # for using a fixed value of total SW absorption instead of using RRTM_SW
tbounds=np.array([300.]) # initalise lower boundary temperature
wklfacs=[1.0] # multiply number of molecules of a gas species by this factor in a given lat and layer range defined later
wklfac_co2s=[1.] # ditto for co2 specifically

# location of perturbations to number of gas molecules
pertzons=[0]
pertlats=[0]
pertmols=[1] #don't do zero!
pertlays=[0]
perts=[1.0]

tbound_adds=[0.] # add a constant to tbound 

#################################################################### end of variable intialisation ##################################################################################

# calculate total number of parameter combinations (number of model runs)
i_loops=0
totloops=np.float(len(pertzons)*len(pertlats)*len(pertmols)*len(pertlays)*len(perts)*len(c_merids)*len(c_zonals)*len(adv_locs)*len(nbs)*len(lapseloops)*len(wklfac_co2s)*len(extra_forcings)*len(lapse_sources) )
looptime = 60
print('Total loops: {:4d} | Expected run time: {:4.1f} minute(s)'.format(int(totloops), totloops*looptime/60.))
print()

for tbound_add in tbound_adds:
    for lapse_source in lapse_sources:
        for adv_loc in adv_locs:
            for nb in nbs:
                for wklfac_co2 in wklfac_co2s:
                    for c_zonal in c_zonals:
                        for c_merid in c_merids:
                            for extra_forcing in extra_forcings:
                                for fixed_sw in fixed_sws:
                                    for wklfac in wklfacs:
                                        for pert in perts:
                                            for pertmol in pertmols:
                                                for pertlat in pertlats:
                                                    for pertzon in pertzons:
                                                        for pertlay in pertlays:
                                                            print('loop {} of {}, {} percent done '.format(i_loops,totloops,i_loops/totloops*100.))
                                                            i_loops+=1
    
                #########################################################################################################################################################
              
                                                            lapse_master=np.ones((nzoncols,nlatcols))*5.7
                                                            if(lapse_source==0):
                                                                lapse_master=np.ones((nzoncols,nlatcols)) * 6.5
                                                            elif(lapse_source==1):
                                                                for i_zon in range(nzoncols):
                                                                    lapse_master[i_zon,:]=np.array(createlatdistbn('Doug Mason Lapse Rate vs Latitude'))
                                                                lapse_master*=-1.
               
                                                            # calculate insolation at a given latitude, day, and hour
                                                            for i_lat in range(nlatcols):
                                                                for day in range(1,365):
                                                                    for hour in range(1,24):
                                                                        hourang = 15.0 * (np.float(hour)-12.0)
                                                                        declin = -23.5 * np.cos(np.deg2rad(360.0/365.0 + (np.float(day) + 10.0)))
                                                                        Xrad = latgrid[i_lat] * 2.0*3.14/360.0
                                                                        Yrad = declin * 2.0*3.14/360.0
                                                                        Hrad = hourang * 2.0*3.14/360.0
                                                                        if (np.sin(Xrad)*np.sin(Yrad) + np.cos(Xrad)*np.cos(Yrad)*np.cos(Hrad) > 0):
                                                                            zen[i_lat,day,hour] = np.arccos(np.sin(Xrad)*np.sin(Yrad) + np.cos(Xrad)*np.cos(Yrad)*np.cos(Hrad))
                                                                            insol[i_lat,day,hour] = solar_constant * np.cos(zen[i_lat,day,hour])
                                                                        else:
                                                                            insol[i_lat,day,hour] = 0.
                                                            # average over day and hour to give annual mean insolation at a given latitude
                                                            for i_lat in range(nlatcols):
                                                                insollats[i_lat] = np.sum(insol[i_lat,:,:]) / np.size(insol[i_lat,:,:])
                                                                zenlats[i_lat] = np.sum(zen[i_lat,:,:]) / np.size(zen[i_lat,:,:])
                                                            # calculate annual average solar zenith angle for a given latitude required to give the calculated annual average insolation at that latitude
                                                            szas = np.rad2deg(np.arccos(insollats/solar_constant))
                                                            
                                                            if(master_input==1):
                                                                nlayers=51
                                                            conv_on_lats=np.ones(nlatcols) #0: no convection, 1: convective adjustment
                                                            if(master_input==3):
                                                                conv_on_lats=np.ones(nlatcols)
                                                            surf_lowlev_coupled=1 #0: surface and lowest level temperatures independent, 1: lowest level temperature = surface temperature
                                                            lay_intp=0 #0: linear interpolation to get tavel from tz, 1: isothermal layers
                                                            # if(conv_on==1):
                                                            #     surf_lowlev_coupled=1
    
                                                            # Declare variables
                                                            cti=0
                                                            surf_rh=0.8
                                                            vol_mixh2o_min = 1e-6
                                                            vol_mixh2o_max = 1e6
                                                            esat_liq=np.zeros(nlayers)
                                                            rel_hum=np.zeros(nlayers)
                                                            maxhtr=0.
                                                            toa_fnet=0
                                                            tbound=290. #surface temperature (K)
                                                            # tbound_inits=220. + np.cos(np.deg2rad(latgrid))*80.
                                                            tbound_inits=220. + np.cos(np.deg2rad(latgrid))*80.
                                                            # undrelax_lats= (2.0 - np.cos(np.deg2rad(latgrid)))*2.
                                                            # data=np.genfromtxt('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/Latitudinal Distributions/Doug Mason Temperature vs Latitude NH.txt',delimiter=',')
                                                            # data=np.genfromtxt('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/Latitudinal Distributions/Doug Mason Temperature vs Latitude.txt',delimiter=',')
                                                            data=np.genfromtxt('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/Latitudinal Distributions/HV19 inferred tbound.txt',delimiter=',')
                                                            lat_obs=data[:,0]
                                                            tg_obs=data[:,1]
                                                            f=interp1d(lat_obs,tg_obs)
                                                            tbound_inits=f(latgrid)
                                                            undrelax_lats=np.ones(nlatcols)*4.*(60./nlayers)*4. #for nl=60
                                                            # undrelax_lats=np.ones(nlatcols)*0.5 #for nl=590
                                                            iemiss=2 #surface emissivity. Keep this fixed for now.
                                                            iemis=2
                                                            ireflect=0 #for Lambert reflection
                                                            iaer=0 #0=aerosols off, 1=on
                                                            istrm=1             # ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0). 
                                                                                    #0=4 streams
                                                                                    #1=8 streams
                                                            idelm=1             # flag for outputting downwelling fluxes computed using the delta-M scaling approximation. 0=output "true" direct and diffuse downwelling fluxes, 1=output direct and diffuse downwelling fluxes computed with delta-M approximation
                                                            icos=0              #0:there is no need to account for instrumental cosine response, 1:to account for instrumental cosine response in the computation of the direct and diffuse fluxes, 2:2 to account for instrumental cosine response in the computation of the diffuse fluxes only
                                                            semis=np.ones(16)*1.0   #all spectral bands the same as iemissm (maybe this is the surface??)
                                                            semiss=np.ones(29)*1.
                                                            if(master_input==1):
                                                                semiss[15:29] = np.array([
                                                                0.881,
                                                                0.794,
                                                                0.738,
                                                                0.727,
                                                                0.657,
                                                                0.660,
                                                                0.626,
                                                                0.655,
                                                                0.791,
                                                                0.883,
                                                                0.957,
                                                                0.958,
                                                                0.958,
                                                                0.970
                                                                ])
                                                            elif(master_input==2):
                                                                semiss[15:29] = np.array([
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                0.64,
                                                                ])
                                                            elif(master_input==3):
                                                                semiss=np.ones(29)*0.93
                                                            iform=1
                                                            psurf=1000.
                                                            if(master_input==3):
                                                                psurf=1014.8
                                                            pmin=1.
                                                            secntk='' #based on not appearing in input mls sw
                                                            cinp='' #based on not appearing in input mls sw
                                                            ipthak=3
                                                            ipthrk=3
                                                            juldat=0        #Julian day associated with calculation (1-365/366 starting January 1). Used to calculate Earth distance from sun. A value of 0 (default) indicates no scaling of solar source function using earth-sun distance.
                                                            sza=75.52
                                                            if(master_input==1):
                                                                sza=65.             #Solar zenith angle in degrees (0 deg is overhead).
                                                            elif(master_input==2):
                                                                sza=45. #(RD repl)          #Solar zenith angle in degrees (0 deg is overhead).
                                                            elif(master_input==3): # RCEMIP
                                                                sza=42.05
                                                            isolvar=0       #= 0 each band uses standard solar source function, corresponding to present day conditions. 
                                                                            #= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
                                                                            #= 2 scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))          
                                                            solvar=np.ones(29)
                                                            if(master_input==3):
                                                                isolvar=2
                                                                solvar=np.ones(29)*409.6/1015.98791896 # different interpretation of 'insolation'
                                                            lapse=5.7
                                                            if(master_input==3): # RCEMIP
                                                                lapse=6.7
                                                            elif(master_input==5):
                                                                lapse=6.2
                                                            tmin=100.
                                                            if(master_input==5):
                                                                tmin=200.
                                                            tmax=340.
                                                            rsp=287.04 # RCEMIP value
                                                            gravity=9.81
                                                            filewritten=0
                                                            sw_freq=100
                                                            plotted=1
    
                                                            pin2 = 0.79 * 1e5 #convert the input in bar to Pa
                                                            pico2 = 400e-6 * 1e5 #convert the input in bar to Pa
                                                            pio2 = 0.2 * 1e5
                                                            piar = 0.0 * 1e5 #convert the input in bar to Pa
                                                            pich4 = 0.0 * 1e5 #convert the input in bar to Pa
                                                            pih2o = 0.0 * 1e5 #convert the input in bar to Pa
                                                            pio3 = 0.0 * 1e5 #convert the input in bar to Pa
    
                                                            mmwn2 = 28.0134e-3
                                                            mmwco2 = 44.01e-3
                                                            mmwo2 = 31.9988e-3
                                                            mmwar = 39.948e-3
                                                            mmwch4 = 16.04e-3
                                                            mmwh2o = 18.01528e-3
                                                            mmwo3 = 48.0e-3
    
                                                            piair = pin2 + pio2 + piar
    
                                                            massatmo_co2 = pico2 / gravity # [kg]
                                                            massatmo_n2 = pin2 / gravity # [kg]
                                                            massatmo_o2 = pio2 / gravity # [kg]
                                                            massatmo_ar = piar / gravity # [kg]
                                                            massatmo_ch4 = pich4 / gravity # [kg]
                                                            massatmo_h2o = pih2o / gravity # [kg]
                                                            massatmo_o3 = pio3 / gravity # [kg]
    
                                                            massatmo = massatmo_co2 + massatmo_n2 + massatmo_o2 + massatmo_ar + massatmo_ch4 + massatmo_h2o + massatmo_o3
    
                                                            # # Gas mass mixing ratios 
                                                            mass_mixco2 = massatmo_co2 / massatmo
                                                            mass_mixn2 = massatmo_n2 / massatmo
                                                            mass_mixo2 = massatmo_o2 / massatmo
                                                            mass_mixar = massatmo_ar / massatmo
                                                            mass_mixch4 = massatmo_ch4 / massatmo
                                                            mass_mixh2o = massatmo_h2o / massatmo
                                                            mass_mixo3 = massatmo_o3 / massatmo
    
                                                            # # Number of molecules of each gas
                                                            molec_co2 = massatmo_co2 / mmwco2 * avogadro
                                                            molec_n2 = massatmo_n2 / mmwn2 * avogadro
                                                            molec_o2 = massatmo_o2 / mmwo2 * avogadro
                                                            molec_ar = massatmo_ar / mmwar * avogadro
                                                            molec_ch4 = massatmo_ch4 / mmwch4 * avogadro
                                                            molec_h2o = massatmo_h2o / mmwh2o * avogadro
                                                            molec_o3 = massatmo_o3 / mmwo3 * avogadro 
    
                                                            totmolec = molec_co2 + molec_n2 + molec_o2 + molec_ar + molec_ch4 + molec_h2o + molec_o3
    
                                                            # # Gas volume mixing ratios
                                                            vol_mixco2 = molec_co2 / totmolec
                                                            vol_mixn2 = molec_n2 / totmolec
                                                            vol_mixo2 = molec_o2 / totmolec
                                                            vol_mixar = molec_ar / totmolec
                                                            vol_mixch4 = molec_ch4 / totmolec
                                                            vol_mixh2o = np.ones(nlayers) * molec_h2o / totmolec
                                                            vol_mixo3 = np.ones(nlayers) * molec_o3 / totmolec
    
                                                            surf_rh=0.8
                                                            vol_mixh2o_min = 1e-6
                                                            vol_mixh2o_max = 1e6
    
                                                            dmax=1.
    
                                                            radice=90.
                                                            radliq=7.
    
                                                            maxdfnet_lat=np.zeros(nlatcols)
                                                            prev_max_dfnet_lat=np.zeros(nlatcols)
                                                            maxdfnet_ind=0
                                                            stepssinceswcalled=0
                                                            sw_called=0
                                                            
                                                            rad_eqb=np.zeros(nlatcols)
                                                            colbudg_eqb=np.zeros(nlatcols)
                                                            if(input_source != 2):
                                                                lapse_eqb=np.ones(nlatcols)
                                                            else:
                                                                lapse_eqb=np.zeros(nlatcols)
    
                                                            # initialise arrays
                                                            tz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            tavel_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                            pz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            pavel_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                            altz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            altavel_master=np.zeros((nlayers,nzoncols,nlatcols))
    
                                                            totuflux_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            totuflux_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            totuflux_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            totdflux_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            totdflux_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            totdflux_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            fnet_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            fnet_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            fnet_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            htr_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            htr_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            htr_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                            wbrodl_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                            conv_master=np.zeros((nlayers+1,nzoncols,nlatcols))
    
                                                            wkl_master=np.zeros((nlayers,nzoncols,nmol+1,nlatcols))
    
                                                            inflags_master=np.zeros((nzoncols,nlatcols))
                                                            iceflags_master=np.zeros((nzoncols,nlatcols))
                                                            liqflags_master=np.zeros((nzoncols,nlatcols))
                                                            
                                                            cld_lays_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                            cld_fracs_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                            tauclds_master=np.ones((nzoncols,nlatcols,nclouds))*1e-3
                                                            ssaclds_master=np.ones((nzoncols,nlatcols,nclouds))*0.5/nlayers
                                                            ssaclds_master[1,:,:]=np.ones((nlatcols,nclouds))*1e-3
    
                                                            tbound_master=np.zeros((nzoncols,nlatcols))
                                                            toa_fnet_master=np.zeros((nzoncols,nlatcols))
                                                            column_budgets_master=np.zeros((nzoncols,nlatcols))
                                                            zonal_transps_master=np.zeros((nzoncols,nlatcols))
                                                            merid_transps_master=np.zeros((nzoncols,nlatcols))
    
                                                            totuflux=np.zeros(nlayers+1)
                                                            totdflux=np.zeros(nlayers+1)
                                                            fnet=np.zeros(nlayers+1)
                                                            htr=np.zeros(nlayers+1)
                                                            pavel=np.zeros(nlayers)
                                                            pz=np.linspace(psurf,pmin,nlayers+1)
                                                            altz=np.zeros(nlayers+1)
                                                            # tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
                                                            tz=np.ones(nlayers+1) * tbound-6.5*altz/1000.
                                                            tz=np.clip(tz,200,tmax)
                                                            tavel=np.zeros(nlayers)
                                                            esat_liq=np.zeros(nlayers)
                                                            rel_hum=np.zeros(nlayers)
                                                            mperlayr = np.zeros(nlayers)
                                                            mperlayr_air = np.zeros(nlayers)
                                                            wbrodl = np.zeros(nlayers)
                                                            wkl = np.zeros((nmol+1,nlayers))
                                                            totuflux_lw=np.zeros(nlayers+1)
                                                            totdflux_lw=np.zeros(nlayers+1)
                                                            fnet_lw=np.zeros(nlayers+1)
                                                            htr_lw=np.zeros(nlayers+1)
                                                            totuflux_sw=np.zeros(nlayers+1)
                                                            totdflux_sw=np.zeros(nlayers+1)
                                                            fnet_sw=np.zeros(nlayers+1)
                                                            htr_sw=np.zeros(nlayers+1)
                                                            conv=np.zeros(nlayers+1)
                                                            altavel = np.zeros(nlayers)
    
                                                            ur=np.ones(nlayers)
    
                                                            inflags_master=(np.ones((nzoncols,nlatcols))*2).astype(int)
                                                            iceflags_master=(np.ones((nzoncols,nlatcols))*2).astype(int)
                                                            liqflags_master=(np.ones((nzoncols,nlatcols))*1).astype(int)
                                                            # cld_lays=np.ones(nzoncols)*nlayers/10.
                                                            # cld_fracs=np.ones(nzoncols)*1.
                                                            # tauclds=np.ones(nzoncols)*1.0
                                                            # ssaclds=np.ones(nzoncols)*0.4
    
                                                            dfnet_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                            dfnet_master_rad=np.zeros((nlayers,nzoncols,nlatcols))
                                                            dfnet_master_adv=np.zeros((nlayers,nzoncols,nlatcols))
                                                            dpz_master=np.zeros((nlayers,nzoncols,nlatcols))
    
                                                            cti_master=np.zeros((nzoncols,nlatcols))
                                                            maxdT=np.ones(nlatcols)
                                                            
                                                            dztrop=np.zeros(nlatcols)
                                                            
                                                            cld_lay_v2=np.zeros(nlatcols)
                                                            cf_mro=np.zeros(nlatcols)
                                                            od_eff=np.zeros(nlatcols)
    
                                                            ctest=' '
                                                            
                                                            if(input_source==1 or input_source==2):
    
                                                                f=open(prev_output_file,'r')
    
                                                                gravity =   float   (   f.readline().rstrip('\n')   )
                                                                avogadro    =   float   (   f.readline().rstrip('\n')   )
                                                                iatm    =   int (   f.readline().rstrip('\n')   )
                                                                ixsect  =   int (   f.readline().rstrip('\n')   )
                                                                iscat   =   int (   f.readline().rstrip('\n')   )
                                                                numangs =   int (   f.readline().rstrip('\n')   )
                                                                iout    =   int (   f.readline().rstrip('\n')   )
                                                                icld    =   int (   f.readline().rstrip('\n')   )
                                                                tbound  =   float   (   f.readline().rstrip('\n')   )
                                                                iemiss  =   int (   f.readline().rstrip('\n')   )
                                                                iemis   =   int (   f.readline().rstrip('\n')   )
                                                                ireflect    =   int (   f.readline().rstrip('\n')   )
                                                                iaer    =   int (   f.readline().rstrip('\n')   )
                                                                istrm   =   int (   f.readline().rstrip('\n')   )
                                                                idelm   =   int (   f.readline().rstrip('\n')   )
                                                                icos    =   int (   f.readline().rstrip('\n')   )
                                                                iform   =   int (   f.readline().rstrip('\n')   )
                                                                nlayers =   int (   f.readline().rstrip('\n')   )
                                                                nmol    =   int (   f.readline().rstrip('\n')   )
                                                                psurf   =   float   (   f.readline().rstrip('\n')   )
                                                                pmin    =   float   (   f.readline().rstrip('\n')   )
                                                                secntk  =   str (   f.readline().rstrip('\n')   )
                                                                cinp    =   str (   f.readline().rstrip('\n')   )
                                                                ipthak  =   int   (   f.readline().rstrip('\n')   )
                                                                ipthrk  =   float   (   f.readline().rstrip('\n')   )
                                                                juldat  =   int   (   f.readline().rstrip('\n')   )
                                                                sza =   float   (   f.readline().rstrip('\n')   )
                                                                isolvar =   int (   f.readline().rstrip('\n')   )
                                                                # lapse   =   float   (   f.readline().rstrip('\n')   )
                                                                f.readline()
                                                                tmin    =   float   (   f.readline().rstrip('\n')   )
                                                                tmax    =   float   (   f.readline().rstrip('\n')   )
                                                                rsp =   float   (   f.readline().rstrip('\n')   )
                                                                gravity =   float   (   f.readline().rstrip('\n')   )
                                                                pin2    =   float   (   f.readline().rstrip('\n')   )
                                                                pico2   =   float   (   f.readline().rstrip('\n')   )
                                                                pio2    =   float   (   f.readline().rstrip('\n')   )
                                                                piar    =   float   (   f.readline().rstrip('\n')   )
                                                                pich4   =   float   (   f.readline().rstrip('\n')   )
                                                                pih2o   =   float   (   f.readline().rstrip('\n')   )
                                                                pio3    =   float   (   f.readline().rstrip('\n')   )
                                                                mmwn2   =   float   (   f.readline().rstrip('\n')   )
                                                                mmwco2  =   float   (   f.readline().rstrip('\n')   )
                                                                mmwo2   =   float   (   f.readline().rstrip('\n')   )
                                                                mmwar   =   float   (   f.readline().rstrip('\n')   )
                                                                mmwch4  =   float   (   f.readline().rstrip('\n')   )
                                                                mmwh2o  =   float   (   f.readline().rstrip('\n')   )
                                                                mmwo3   =   float   (   f.readline().rstrip('\n')   )
                                                                piair   =   float   (   f.readline().rstrip('\n')   )
                                                                totmolec    =   float   (   f.readline().rstrip('\n')   )
                                                                surf_rh =   float   (   f.readline().rstrip('\n')   )
                                                                vol_mixh2o_min  =   float   (   f.readline().rstrip('\n')   )
                                                                vol_mixh2o_max  =   float   (   f.readline().rstrip('\n')   )
                                                                ur_min  =   float   (   f.readline().rstrip('\n')   )
                                                                ur_max  =   float   (   f.readline().rstrip('\n')   )
                                                                eqb_maxhtr  =   float   (   f.readline().rstrip('\n')   )
                                                                # timesteps   =   int (   f.readline().rstrip('\n')   )
                                                                f.readline()
                                                                if(input_source==1):
                                                                    timesteps=2
                                                                # elif(input_source==2):
                                                                #     timesteps=199
                                                                cti =   int (   f.readline().rstrip('\n')   )
                                                                maxhtr  =   float   (   f.readline().rstrip('\n')   )
                                                                cld_lay =   float   (   f.readline().rstrip('\n')   )
                                                                nzoncols  =   int (   f.readline().rstrip('\n')   )
                                                                master_input =  float   (   f.readline().rstrip('\n')   )
                                                                conv_on=float   (   f.readline().rstrip('\n')   )
                                                                surf_lowlev_coupled=float   (   f.readline().rstrip('\n')   )
                                                                lay_intp=float  (   f.readline().rstrip('\n')   )
                                                                lw_on=int   (   f.readline().rstrip('\n')   )
                                                                # sw_on=int   (   f.readline().rstrip('\n')   )
                                                                f.readline()
                                                                eqb_maxdfnet=float  (   f.readline().rstrip('\n')   )
                                                                toa_fnet_eqb=float  (   f.readline().rstrip('\n')   )
                                                                nlatcols=int    (   f.readline().rstrip('\n')   )
    
    
                                                                pz=np.linspace(psurf,pmin,nlayers+1)
                                                                totuflux=np.zeros(nlayers+1)
                                                                totdflux=np.zeros(nlayers+1)
                                                                fnet=np.zeros(nlayers+1)
                                                                htr=np.zeros(nlayers+1)
                                                                pavel=np.zeros(nlayers)
                                                                tz=np.ones(nlayers+1) * tbound
                                                                altz=np.zeros(nlayers+1)
                                                                altavel=np.zeros(nlayers)
                                                                # tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
                                                                tz=np.ones(nlayers+1) * tbound-6.5*altz/1000.
                                                                tavel=np.zeros(nlayers)
                                                                esat_liq=np.zeros(nlayers)
                                                                rel_hum=np.zeros(nlayers)
                                                                mperlayr = np.zeros(nlayers)
                                                                mperlayr_air = np.zeros(nlayers)
                                                                wbrodl = np.zeros(nlayers)
                                                                wkl = np.zeros((nmol+1,nlayers))
                                                                totuflux_lw=np.zeros(nlayers+1)
                                                                totdflux_lw=np.zeros(nlayers+1)
                                                                fnet_lw=np.zeros(nlayers+1)
                                                                htr_lw=np.zeros(nlayers+1)
                                                                totuflux_sw=np.zeros(nlayers+1)
                                                                totdflux_sw=np.zeros(nlayers+1)
                                                                fnet_sw=np.zeros(nlayers+1)
                                                                htr_sw=np.zeros(nlayers+1)
                                                                conv=np.zeros(nlayers+1)
                                                                altavel = np.zeros(nlayers)
                                                                vol_mixh2o = np.ones(nlayers) * molec_h2o / totmolec
                                                                vol_mixo3 = np.ones(nlayers) * molec_o3 / totmolec
                                                                solvar=np.zeros(29)
    
                                                                inflags_master=np.zeros((nzoncols,nlatcols))
                                                                iceflags_master=np.zeros((nzoncols,nlatcols))
                                                                liqflags_master=np.zeros((nzoncols,nlatcols))
                                                                
                                                                cld_lays_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                                cld_fracs_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                                tauclds_master=np.ones((nzoncols,nlatcols,nclouds))*1e-3
                                                                ssaclds_master=np.ones((nzoncols,nlatcols,nclouds))*0.5/nlayers
                                                                ssaclds_master[1,:,:]=np.ones((nlatcols,nclouds))*1e-3
    
                                                                tbound_master=np.ones((nzoncols,nlatcols))
                                                                toa_fnet_master=np.ones((nzoncols,nlatcols))
                                                                zonal_transps_master=np.zeros((nzoncols,nlatcols))
                                                                merid_transps_master=np.zeros((nzoncols,nlatcols))
    
                                                                tz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                tavel_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                                pz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                pavel_master=np.zeros((nlayers,nzoncols,nlatcols))
                                                                altz_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                altavel_master=np.zeros((nlayers,nzoncols,nlatcols))
    
                                                                totuflux_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                totuflux_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                totuflux_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                totdflux_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                totdflux_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                totdflux_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                fnet_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                fnet_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                fnet_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                htr_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                htr_lw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                htr_sw_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                wbrodl_master=np.zeros((nlayers+1,nzoncols,nlatcols))
                                                                conv_master=np.zeros((nlayers+1,nzoncols,nlatcols))
    
                                                                wkl_master=np.zeros((nlayers,nzoncols,nmol+1,nlatcols))
    
                                                                vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,nzoncols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb,nlatcols]
                                                                vars_master_lay_zon_lat=[tavel_master,pavel_master,altavel_master,wbrodl_master,dfnet_master,dfnet_master_rad,dfnet_master_adv,dpz_master]
                                                                vars_master_lev_zon_lat=[tz_master,pz_master,altz_master,totuflux_master,totuflux_lw_master,totuflux_sw_master,totdflux_master,totdflux_lw_master,totdflux_sw_master,fnet_master,fnet_lw_master,fnet_sw_master,htr_master,htr_lw_master,htr_sw_master,conv_master]
                                                                vars_misc_1d=[semis,semiss,solvar]
                                                                vars_misc_1d_lens=[16,29,29]
                                                                vars_master_lay_zon_nmol_lat=[wkl_master]
                                                                vars_master_zon_lat=[inflags_master,iceflags_master,liqflags_master,tbound_master,toa_fnet_master,zonal_transps_master,merid_transps_master,cti_master,lapse_master]
                                                                vars_master_zon_lat_cld=[cld_lays_master,cld_fracs_master,tauclds_master,ssaclds_master]
                                                                vars_master_lat=[latgrid]
                                                                
                                                                dztrop=np.zeros(nlatcols)
    
                                                                for x in vars_master_lay_zon_lat:
                                                                    for k in range(nlatcols):
                                                                        for j in range(nzoncols):
                                                                            for i in range(nlayers):
                                                                                x[i,j,k] = f.readline()
    
                                                                for x in vars_master_lev_zon_lat:
                                                                    for k in range(nlatcols):
                                                                        for j in range(nzoncols):
                                                                            for i in range(nlayers+1):
                                                                                x[i,j,k] = f.readline()
    
                                                                i_lens=0
                                                                for x in vars_misc_1d:
                                                                    for i in range(vars_misc_1d_lens[i_lens]):
                                                                        x[i] = f.readline()
                                                                    i_lens+=1
    
    
                                                                for x in vars_master_lay_zon_nmol_lat:
                                                                    for l in range(nlatcols):
                                                                        for k in range(nmol+1):
                                                                            for j in range(nzoncols):
                                                                                for i in range(nlayers):
                                                                                    x[i,j,k,l] = f.readline()
    
                                                                i_var=0
                                                                for x in vars_master_zon_lat:
                                                                    i_var+=1
                                                                    for j in range(nlatcols):
                                                                        for i in range(nzoncols):
                                                                            x[i,j]=f.readline()
    
                                                                for x in vars_master_zon_lat_cld:
                                                                    for k in range(nclouds):
                                                                        for j in range(nlatcols):
                                                                            for i in range(nzoncols):
                                                                                x[i,j,k]=f.readline()
                                                                                
                                                                for x in vars_master_lat:
                                                                    for i in range(nlatcols):
                                                                        x[i]=f.readline()
                                                                        
                                                                tbound_master+=tbound_add
    
                                                            elif(input_source==0):
    
                                                                if(master_input==1 or master_input==2):
                                                                    pz=np.array([
                                                                    1013.000000 ,
                                                                    891.460000  ,
                                                                    792.287000  ,
                                                                    718.704000  ,
                                                                    651.552000  ,
                                                                    589.841000  ,
                                                                    532.986000  ,
                                                                    480.526000  ,
                                                                    437.556000  ,
                                                                    398.085000  ,
                                                                    361.862000  ,
                                                                    328.507000  ,
                                                                    297.469000  ,
                                                                    269.015000  ,
                                                                    243.001000  ,
                                                                    218.668000  ,
                                                                    196.440000  ,
                                                                    162.913000  ,
                                                                    136.511000  ,
                                                                    114.564000  ,
                                                                    96.490300   ,
                                                                    81.200000   ,
                                                                    68.428600   ,
                                                                    57.693600   ,
                                                                    48.690400   ,
                                                                    40.535400   ,
                                                                    33.733000   ,
                                                                    28.120100   ,
                                                                    23.155700   ,
                                                                    18.791400   ,
                                                                    15.069300   ,
                                                                    11.800600   ,
                                                                    8.786280    ,
                                                                    6.613280    ,
                                                                    5.034690    ,
                                                                    3.853330    ,
                                                                    2.964080    ,
                                                                    2.291800    ,
                                                                    1.782270    ,
                                                                    1.339000    ,
                                                                    0.589399    ,
                                                                    0.430705    ,
                                                                    0.333645    ,
                                                                    0.261262    ,
                                                                    0.216491    ,
                                                                    0.179393    ,
                                                                    0.148652    ,
                                                                    0.125500    ,
                                                                    0.106885    ,
                                                                    0.091031    ,
                                                                    0.077529    ,
                                                                    0.067000    ,
                                                                        ])
    
                                                                    wkl[1,:] = np.array([
                                                                    1.5946558E-02   ,
                                                                    1.1230157E-02   ,
                                                                    7.6751928E-03   ,
                                                                    5.2688639E-03   ,
                                                                    3.6297729E-03   ,
                                                                    2.3900282E-03   ,
                                                                    1.7066754E-03   ,
                                                                    1.2718296E-03   ,
                                                                    9.5655693E-04   ,
                                                                    6.9666741E-04   ,
                                                                    5.0829613E-04   ,
                                                                    3.6584702E-04   ,
                                                                    2.4977655E-04   ,
                                                                    1.3636267E-04   ,
                                                                    6.5472166E-05   ,
                                                                    2.8419665E-05   ,
                                                                    9.6973117E-06   ,
                                                                    4.8207025E-06   ,
                                                                    3.4318521E-06   ,
                                                                    3.2663258E-06   ,
                                                                    3.1784930E-06   ,
                                                                    3.1768304E-06   ,
                                                                    3.2639416E-06   ,
                                                                    3.4095149E-06   ,
                                                                    3.5909502E-06   ,
                                                                    3.8500998E-06   ,
                                                                    4.0575464E-06   ,
                                                                    4.2513630E-06   ,
                                                                    4.3863338E-06   ,
                                                                    4.5309193E-06   ,
                                                                    4.6839027E-06   ,
                                                                    4.8067850E-06   ,
                                                                    4.9039072E-06   ,
                                                                    4.9670398E-06   ,
                                                                    5.0161370E-06   ,
                                                                    5.1013058E-06   ,
                                                                    5.2471341E-06   ,
                                                                    5.3810127E-06   ,
                                                                    5.4697343E-06   ,
                                                                    5.4735615E-06   ,
                                                                    5.3326530E-06   ,
                                                                    5.1831207E-06   ,
                                                                    5.0460312E-06   ,
                                                                    4.8780507E-06   ,
                                                                    4.7075605E-06   ,
                                                                    4.5413699E-06   ,
                                                                    4.3837813E-06   ,
                                                                    4.2189254E-06   ,
                                                                    4.0623413E-06   ,
                                                                    3.9098322E-06   ,
                                                                    3.7676771E-06   ,
                                                                        ])
    
                                                                    wkl[2,:] = np.array([
                                                                    3.5495765E-04   ,
                                                                    3.5495876E-04   ,
                                                                    3.5494528E-04   ,
                                                                    3.5497753E-04   ,
                                                                    3.5498614E-04   ,
                                                                    3.5499220E-04   ,
                                                                    3.5500020E-04   ,
                                                                    3.5500637E-04   ,
                                                                    3.5500794E-04   ,
                                                                    3.5500876E-04   ,
                                                                    3.5501100E-04   ,
                                                                    3.5501071E-04   ,
                                                                    3.5501161E-04   ,
                                                                    3.5500948E-04   ,
                                                                    3.5501219E-04   ,
                                                                    3.5501563E-04   ,
                                                                    3.5500890E-04   ,
                                                                    3.5499944E-04   ,
                                                                    3.5499918E-04   ,
                                                                    3.5499939E-04   ,
                                                                    3.5499930E-04   ,
                                                                    3.5499941E-04   ,
                                                                    3.5499973E-04   ,
                                                                    3.5499971E-04   ,
                                                                    3.5500005E-04   ,
                                                                    3.5499997E-04   ,
                                                                    3.5499965E-04   ,
                                                                    3.5500378E-04   ,
                                                                    3.5500416E-04   ,
                                                                    3.5501141E-04   ,
                                                                    3.5500969E-04   ,
                                                                    3.5500916E-04   ,
                                                                    3.5500980E-04   ,
                                                                    3.5500823E-04   ,
                                                                    3.5501088E-04   ,
                                                                    3.5501190E-04   ,
                                                                    3.5501042E-04   ,
                                                                    3.5500791E-04   ,
                                                                    3.5500259E-04   ,
                                                                    3.5500337E-04   ,
                                                                    3.5502302E-04   ,
                                                                    3.5507922E-04   ,
                                                                    3.5503952E-04   ,
                                                                    3.5511289E-04   ,
                                                                    3.5519004E-04   ,
                                                                    3.5513769E-04   ,
                                                                    3.5506955E-04   ,
                                                                    3.5528085E-04   ,
                                                                    3.5538705E-04   ,
                                                                    3.5533530E-04   ,
                                                                    3.5513588E-04   ,
                                                                        ])
    
                                                                    wkl[3,:] = np.array([
                                                                    3.1872162E-08   ,
                                                                    3.5456235E-08   ,
                                                                    3.9477314E-08   ,
                                                                    4.3921091E-08   ,
                                                                    4.8850310E-08   ,
                                                                    5.4422610E-08   ,
                                                                    6.1250461E-08   ,
                                                                    6.9855773E-08   ,
                                                                    7.9463597E-08   ,
                                                                    8.9151150E-08   ,
                                                                    1.0168034E-07   ,
                                                                    1.1558580E-07   ,
                                                                    1.3068458E-07   ,
                                                                    1.6048106E-07   ,
                                                                    1.9350828E-07   ,
                                                                    2.2751291E-07   ,
                                                                    3.0428600E-07   ,
                                                                    4.3981947E-07   ,
                                                                    5.2382995E-07   ,
                                                                    6.3216254E-07   ,
                                                                    8.2302279E-07   ,
                                                                    1.2512422E-06   ,
                                                                    1.8039109E-06   ,
                                                                    2.2908109E-06   ,
                                                                    2.8324889E-06   ,
                                                                    3.4517834E-06   ,
                                                                    4.2219772E-06   ,
                                                                    5.0326839E-06   ,
                                                                    5.6775239E-06   ,
                                                                    6.3139009E-06   ,
                                                                    6.9619100E-06   ,
                                                                    7.7728864E-06   ,
                                                                    8.5246547E-06   ,
                                                                    8.8305105E-06   ,
                                                                    8.4904723E-06   ,
                                                                    7.5621829E-06   ,
                                                                    6.2966351E-06   ,
                                                                    5.1043844E-06   ,
                                                                    4.0821087E-06   ,
                                                                    2.8155102E-06   ,
                                                                    1.8036270E-06   ,
                                                                    1.5450810E-06   ,
                                                                    1.3594723E-06   ,
                                                                    1.1832446E-06   ,
                                                                    1.0330702E-06   ,
                                                                    9.0162695E-07   ,
                                                                    7.8788491E-07   ,
                                                                    6.7509507E-07   ,
                                                                    5.7978644E-07   ,
                                                                    4.9771251E-07   ,
                                                                    4.2984522E-07   ,
                                                                        ])
    
                                                                    wkl[4,:] = np.array([
                                                                    3.2014773E-07   ,
                                                                    3.2014808E-07   ,
                                                                    3.2012952E-07   ,
                                                                    3.2017348E-07   ,
                                                                    3.2020259E-07   ,
                                                                    3.2024920E-07   ,
                                                                    3.2018053E-07   ,
                                                                    3.2015103E-07   ,
                                                                    3.2006952E-07   ,
                                                                    3.1964703E-07   ,
                                                                    3.1794278E-07   ,
                                                                    3.1485408E-07   ,
                                                                    3.0993951E-07   ,
                                                                    3.0289050E-07   ,
                                                                    2.9728139E-07   ,
                                                                    2.9330249E-07   ,
                                                                    2.8654489E-07   ,
                                                                    2.7902988E-07   ,
                                                                    2.6973828E-07   ,
                                                                    2.5467133E-07   ,
                                                                    2.3132466E-07   ,
                                                                    1.9950789E-07   ,
                                                                    1.6908091E-07   ,
                                                                    1.3991885E-07   ,
                                                                    1.1722680E-07   ,
                                                                    1.0331899E-07   ,
                                                                    9.4382699E-08   ,
                                                                    8.7561951E-08   ,
                                                                    8.2404142E-08   ,
                                                                    7.5596006E-08   ,
                                                                    6.6951600E-08   ,
                                                                    5.4150636E-08   ,
                                                                    4.2426844E-08   ,
                                                                    3.2571123E-08   ,
                                                                    2.4015852E-08   ,
                                                                    1.7783966E-08   ,
                                                                    1.2921510E-08   ,
                                                                    9.3075085E-09   ,
                                                                    6.6677854E-09   ,
                                                                    3.5912391E-09   ,
                                                                    2.0309472E-09   ,
                                                                    1.7047587E-09   ,
                                                                    1.4732259E-09   ,
                                                                    1.3152129E-09   ,
                                                                    1.2046001E-09   ,
                                                                    1.1028871E-09   ,
                                                                    1.0173566E-09   ,
                                                                    9.5524733E-10   ,
                                                                    9.0009833E-10   ,
                                                                    8.4775770E-10   ,
                                                                    8.0018175E-10   ,
                                                                        ])
    
                                                                    wkl[5,:] = np.array([
                                                                    1.4735235E-07   ,
                                                                    1.4203219E-07   ,
                                                                    1.3746356E-07   ,
                                                                    1.3388170E-07   ,
                                                                    1.3135738E-07   ,
                                                                    1.3046302E-07   ,
                                                                    1.2931390E-07   ,
                                                                    1.2701938E-07   ,
                                                                    1.2377659E-07   ,
                                                                    1.1940332E-07   ,
                                                                    1.1352941E-07   ,
                                                                    1.0700342E-07   ,
                                                                    1.0015444E-07   ,
                                                                    9.3152551E-08   ,
                                                                    8.5588468E-08   ,
                                                                    7.7191764E-08   ,
                                                                    6.3881643E-08   ,
                                                                    4.8797485E-08   ,
                                                                    3.7298612E-08   ,
                                                                    2.8723687E-08   ,
                                                                    2.2545748E-08   ,
                                                                    1.7379815E-08   ,
                                                                    1.4111547E-08   ,
                                                                    1.2622904E-08   ,
                                                                    1.2397807E-08   ,
                                                                    1.3167179E-08   ,
                                                                    1.4350868E-08   ,
                                                                    1.5625453E-08   ,
                                                                    1.6708778E-08   ,
                                                                    1.8091109E-08   ,
                                                                    1.9843396E-08   ,
                                                                    2.1874927E-08   ,
                                                                    2.3846910E-08   ,
                                                                    2.5646894E-08   ,
                                                                    2.7513584E-08   ,
                                                                    2.9431952E-08   ,
                                                                    3.0938047E-08   ,
                                                                    3.2309320E-08   ,
                                                                    3.3800561E-08   ,
                                                                    3.6464382E-08   ,
                                                                    3.9601694E-08   ,
                                                                    4.2654523E-08   ,
                                                                    4.5695458E-08   ,
                                                                    4.9774858E-08   ,
                                                                    5.4377978E-08   ,
                                                                    5.9385144E-08   ,
                                                                    6.5223382E-08   ,
                                                                    7.4618846E-08   ,
                                                                    8.5339593E-08   ,
                                                                    9.7556516E-08   ,
                                                                    1.1081534E-07   ,
                                                                        ])
    
                                                                    wkl[6,:] = np.array([
                                                                    1.7007853E-06   ,
                                                                    1.7007861E-06   ,
                                                                    1.7006882E-06   ,
                                                                    1.7000174E-06   ,
                                                                    1.6967191E-06   ,
                                                                    1.6890905E-06   ,
                                                                    1.6774702E-06   ,
                                                                    1.6625032E-06   ,
                                                                    1.6469684E-06   ,
                                                                    1.6329801E-06   ,
                                                                    1.6223285E-06   ,
                                                                    1.6071415E-06   ,
                                                                    1.5820669E-06   ,
                                                                    1.5562247E-06   ,
                                                                    1.5313253E-06   ,
                                                                    1.5080506E-06   ,
                                                                    1.4806419E-06   ,
                                                                    1.4479623E-06   ,
                                                                    1.4152675E-06   ,
                                                                    1.3795030E-06   ,
                                                                    1.3426010E-06   ,
                                                                    1.3014652E-06   ,
                                                                    1.2451943E-06   ,
                                                                    1.1722138E-06   ,
                                                                    1.0758683E-06   ,
                                                                    9.6515760E-07   ,
                                                                    8.5401462E-07   ,
                                                                    7.7107171E-07   ,
                                                                    7.2538978E-07   ,
                                                                    6.8032085E-07   ,
                                                                    6.3401592E-07   ,
                                                                    5.7941355E-07   ,
                                                                    5.2736578E-07   ,
                                                                    4.8160666E-07   ,
                                                                    4.3754815E-07   ,
                                                                    3.9457359E-07   ,
                                                                    3.5215132E-07   ,
                                                                    3.1031249E-07   ,
                                                                    2.6731394E-07   ,
                                                                    2.0088720E-07   ,
                                                                    1.5878383E-07   ,
                                                                    1.5400190E-07   ,
                                                                    1.5114806E-07   ,
                                                                    1.5015239E-07   ,
                                                                    1.5018485E-07   ,
                                                                    1.5016241E-07   ,
                                                                    1.5013467E-07   ,
                                                                    1.5023033E-07   ,
                                                                    1.5028188E-07   ,
                                                                    1.5026681E-07   ,
                                                                    1.5018884E-07   ,
                                                                        ])
    
                                                                    wkl[7,:] = np.array([
                                                                    2.0897518E-01   ,
                                                                    2.0897572E-01   ,
                                                                    2.0896780E-01   ,
                                                                    2.0898660E-01   ,
                                                                    2.0899189E-01   ,
                                                                    2.0899543E-01   ,
                                                                    2.0899996E-01   ,
                                                                    2.0900373E-01   ,
                                                                    2.0900458E-01   ,
                                                                    2.0900519E-01   ,
                                                                    2.0900649E-01   ,
                                                                    2.0900634E-01   ,
                                                                    2.0900698E-01   ,
                                                                    2.0900562E-01   ,
                                                                    2.0900711E-01   ,
                                                                    2.0900925E-01   ,
                                                                    2.0900522E-01   ,
                                                                    2.0899965E-01   ,
                                                                    2.0899954E-01   ,
                                                                    2.0899963E-01   ,
                                                                    2.0899959E-01   ,
                                                                    2.0899966E-01   ,
                                                                    2.0899986E-01   ,
                                                                    2.0899987E-01   ,
                                                                    2.0900002E-01   ,
                                                                    2.0899989E-01   ,
                                                                    2.0899986E-01   ,
                                                                    2.0900220E-01   ,
                                                                    2.0900251E-01   ,
                                                                    2.0900670E-01   ,
                                                                    2.0900570E-01   ,
                                                                    2.0900536E-01   ,
                                                                    2.0900574E-01   ,
                                                                    2.0900482E-01   ,
                                                                    2.0900646E-01   ,
                                                                    2.0900702E-01   ,
                                                                    2.0900613E-01   ,
                                                                    2.0900463E-01   ,
                                                                    2.0900150E-01   ,
                                                                    2.0900197E-01   ,
                                                                    2.0901358E-01   ,
                                                                    2.0904660E-01   ,
                                                                    2.0902328E-01   ,
                                                                    2.0906644E-01   ,
                                                                    2.0911193E-01   ,
                                                                    2.0908101E-01   ,
                                                                    2.0904104E-01   ,
                                                                    2.0916539E-01   ,
                                                                    2.0922786E-01   ,
                                                                    2.0919746E-01   ,
                                                                    2.0908001E-01   ,
                                                                        ])
    
                                                                    wbrodl = np.array([
                                                                    2.0212141E+24   ,
                                                                    1.6594377E+24   ,
                                                                    1.2158148E+24   ,
                                                                    1.1250208E+24   ,
                                                                    1.0383909E+24   ,
                                                                    9.5728074E+23   ,
                                                                    8.8085689E+23   ,
                                                                    7.1067458E+23   ,
                                                                    6.5919029E+23   ,
                                                                    6.1060294E+23   ,
                                                                    5.6462299E+23   ,
                                                                    5.2211246E+23   ,
                                                                    4.8122331E+23   ,
                                                                    4.4374356E+23   ,
                                                                    4.0814444E+23   ,
                                                                    3.7436774E+23   ,
                                                                    5.6808373E+23   ,
                                                                    4.3642727E+23   ,
                                                                    3.6534753E+23   ,
                                                                    3.0739262E+23   ,
                                                                    2.5828079E+23   ,
                                                                    2.1627581E+23   ,
                                                                    1.8117264E+23   ,
                                                                    1.5183546E+23   ,
                                                                    1.3798849E+23   ,
                                                                    1.1426864E+23   ,
                                                                    9.4440466E+22   ,
                                                                    8.4273140E+22   ,
                                                                    7.3593470E+22   ,
                                                                    6.2926780E+22   ,
                                                                    5.5637695E+22   ,
                                                                    5.1760866E+22   ,
                                                                    3.6146276E+22   ,
                                                                    2.6801360E+22   ,
                                                                    2.0049382E+22   ,
                                                                    1.5083691E+22   ,
                                                                    1.1414246E+22   ,
                                                                    8.6903496E+21   ,
                                                                    7.5064828E+21   ,
                                                                    1.2774552E+22   ,
                                                                    2.6989144E+21   ,
                                                                    1.6518584E+21   ,
                                                                    1.2455197E+21   ,
                                                                    7.5262510E+20   ,
                                                                    6.3531851E+20   ,
                                                                    5.3673077E+20   ,
                                                                    3.9289481E+20   ,
                                                                    3.1107724E+20   ,
                                                                    2.7061537E+20   ,
                                                                    2.3544404E+20   ,
                                                                    1.8746121E+20   ,
                                                                        ])
    
                                                                    pavel=np.array([
                                                                    952.1147    ,
                                                                    841.897 ,
                                                                    755.3917    ,
                                                                    685.0609    ,
                                                                    620.7571    ,
                                                                    561.5159    ,
                                                                    506.7787    ,
                                                                    458.9778    ,
                                                                    417.8179    ,
                                                                    379.9846    ,
                                                                    345.1331    ,
                                                                    313.0001    ,
                                                                    283.2681    ,
                                                                    255.9648    ,
                                                                    230.793 ,
                                                                    207.5901    ,
                                                                    179.6777    ,
                                                                    149.8259    ,
                                                                    125.467 ,
                                                                    105.5072    ,
                                                                    88.85838    ,
                                                                    74.81903    ,
                                                                    63.06029    ,
                                                                    53.19867    ,
                                                                    44.59128    ,
                                                                    37.16316    ,
                                                                    30.91292    ,
                                                                    25.6397 ,
                                                                    20.97451    ,
                                                                    16.9346 ,
                                                                    13.41941    ,
                                                                    10.30125    ,
                                                                    7.703475    ,
                                                                    5.824757    ,
                                                                    4.442682    ,
                                                                    3.407392    ,
                                                                    2.627624    ,
                                                                    2.037819    ,
                                                                    1.56118 ,
                                                                    0.9634139   ,
                                                                    0.5106084   ,
                                                                    0.3820259   ,
                                                                    0.2975729   ,
                                                                    0.2388066   ,
                                                                    0.1978831   ,
                                                                    0.1639725   ,
                                                                    0.1372726   ,
                                                                    0.1161604   ,
                                                                    0.0989301   ,
                                                                    0.0842557   ,
                                                                    0.0722468   ,
                                                                        ])
    
                                                                    tavel=np.array([
                                                                    291.77  ,
                                                                    287.03  ,
                                                                    282.23  ,
                                                                    277.43  ,
                                                                    272.63  ,
                                                                    267.83  ,
                                                                    263.03  ,
                                                                    258.3   ,
                                                                    253.75  ,
                                                                    249.2   ,
                                                                    244.65  ,
                                                                    240.13  ,
                                                                    235.64  ,
                                                                    231.1   ,
                                                                    226.55  ,
                                                                    222.01  ,
                                                                    216.81  ,
                                                                    215.71  ,
                                                                    215.7   ,
                                                                    215.7   ,
                                                                    216.18  ,
                                                                    217.39  ,
                                                                    218.72  ,
                                                                    220.08  ,
                                                                    221.46  ,
                                                                    222.88  ,
                                                                    224.24  ,
                                                                    225.81  ,
                                                                    227.61  ,
                                                                    230.17  ,
                                                                    233.52  ,
                                                                    237.51  ,
                                                                    242.34  ,
                                                                    247.27  ,
                                                                    252.17  ,
                                                                    257.13  ,
                                                                    262.09  ,
                                                                    267.05  ,
                                                                    272 ,
                                                                    274.41  ,
                                                                    268.77  ,
                                                                    263.53  ,
                                                                    258.75  ,
                                                                    253.76  ,
                                                                    249 ,
                                                                    244.24  ,
                                                                    239.61  ,
                                                                    234.65  ,
                                                                    229.81  ,
                                                                    224.97  ,
                                                                    220.34  ,
                                                                        ])
    
                                                                    altz=np.array([
                                                                    0.00    ,
                                                                    1.10    ,
                                                                    2.10    ,
                                                                    2.90    ,
                                                                    3.70    ,
                                                                    4.50    ,
                                                                    5.30    ,
                                                                    6.10    ,
                                                                    6.80    ,
                                                                    7.50    ,
                                                                    8.20    ,
                                                                    8.90    ,
                                                                    9.60    ,
                                                                    10.30   ,
                                                                    11.00   ,
                                                                    11.70   ,
                                                                    12.40   ,
                                                                    13.60   ,
                                                                    14.70   ,
                                                                    15.80   ,
                                                                    16.90   ,
                                                                    18.00   ,
                                                                    19.10   ,
                                                                    20.20   ,
                                                                    21.30   ,
                                                                    22.50   ,
                                                                    23.70   ,
                                                                    24.90   ,
                                                                    26.20   ,
                                                                    27.60   ,
                                                                    29.10   ,
                                                                    30.80   ,
                                                                    32.90   ,
                                                                    34.90   ,
                                                                    36.90   ,
                                                                    38.90   ,
                                                                    40.90   ,
                                                                    42.90   ,
                                                                    44.90   ,
                                                                    47.20   ,
                                                                    53.90   ,
                                                                    56.40   ,
                                                                    58.40   ,
                                                                    60.30   ,
                                                                    61.70   ,
                                                                    63.10   ,
                                                                    64.50   ,
                                                                    65.70   ,
                                                                    66.80   ,
                                                                    67.90   ,
                                                                    69.00   ,
                                                                    70.00   ,
                                                                        ])*1000.
    
                                                                    tz=np.array([
                                                                    294.2   ,
                                                                    289.25  ,
                                                                    284.6   ,
                                                                    279.8   ,
                                                                    275 ,
                                                                    270.2   ,
                                                                    265.4   ,
                                                                    260.55  ,
                                                                    256 ,
                                                                    251.45  ,
                                                                    246.9   ,
                                                                    242.35  ,
                                                                    237.86  ,
                                                                    233.35  ,
                                                                    228.8   ,
                                                                    224.25  ,
                                                                    219.7   ,
                                                                    215.74  ,
                                                                    215.7   ,
                                                                    215.7   ,
                                                                    215.7   ,
                                                                    216.8   ,
                                                                    218.03  ,
                                                                    219.44  ,
                                                                    220.76  ,
                                                                    222.2   ,
                                                                    223.57  ,
                                                                    224.98  ,
                                                                    226.71  ,
                                                                    228.66  ,
                                                                    231.81  ,
                                                                    235.4   ,
                                                                    239.99  ,
                                                                    244.95  ,
                                                                    249.84  ,
                                                                    254.77  ,
                                                                    259.73  ,
                                                                    264.69  ,
                                                                    269.65  ,
                                                                    274.56  ,
                                                                    270.71  ,
                                                                    265.88  ,
                                                                    261 ,
                                                                    256.08  ,
                                                                    251.32  ,
                                                                    246.56  ,
                                                                    241.8   ,
                                                                    237.02  ,
                                                                    232.18  ,
                                                                    227.34  ,
                                                                    222.5   ,
                                                                    218.1   ,
                                                                        ])
    
                                                                    tbound=tz[0]
    
                                                                if(master_input==4):
                                                                    df=pd.read_excel('/Users/nickedkins/Dropbox/Spreadsheets (Research)/Nicks2 (Roger\'s result vs mine, made by RD).xlsx', sheet_name='RCE') #read RD's data to plot against mine
                                                                    # tz=np.array(df['Tz(K)'])
                                                                    # tz=tz[::-1]
                                                                    # tavel=np.array(df['Tlayer'][:-1])
                                                                    # tavel=tavel[::-1]
                                                                    pz=np.array(df['Pz(mb)'])
                                                                    pz=pz[::-1]
                                                                    pavel=np.array(df['Player'][:-1])
                                                                    pavel=pavel[::-1]
                                                                    altz=np.array(df['Z(km)'])*1e3
                                                                    altz=altz[::-1]
                                                                    fnet_sw=np.array(df['SWnet'])
                                                                    fnet_sw=fnet_sw[::-1]
                                                                    wkl[1,:]=np.array(df['H2O'][:-1])
                                                                    wkl[1,:]=wkl[1,:][::-1]
                                                                    wkl[2,:]=np.array(df['CO2'][:-1])
                                                                    wkl[2,:]=wkl[2,:][::-1]
                                                                    wkl[3,:]=np.array(df['O3'][:-1])
                                                                    wkl[3,:]=wkl[3,:][::-1]
                                                                    wbrodl=np.array(df['dry'][:-1])
                                                                    wbrodl=wbrodl[::-1]
    
                                                                if(master_input==0 or master_input==6):
                                                                    pz=np.linspace(psurf,pmin,nlayers+1)
                                                                    for i in range(len(pavel)):
                                                                        pavel[i]=(pz[i]+pz[i+1])/2.
                                                                    tavel=np.zeros(nlayers)
                                                                    for i in range(len(pavel)):
                                                                        tavel[i]=(tz[i]+tz[i+1])/2.
                                                                    altz[0] = 0.0
                                                                    for i in range(1,nlayers):
                                                                        altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity
                                                                    altz[nlayers] = altz[nlayers-1]+(pz[nlayers-1]-pz[nlayers])*rsp*tavel[nlayers-1]/pavel[nlayers-1]/gravity
                                                                    # tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
                                                                    tz=np.ones(nlayers+1) * tbound-6.5*altz/1000.
                                                                    tz=np.clip(tz,tmin,tmax)
                                                                    for i in range(nlayers):
                                                                        tavel[i]=(tz[i]+tz[i+1])/2.
                                                                    tavel[nlayers-1] = tavel[nlayers-2]
                                                                elif(master_input==3): # RCEMIP hydrostatics
                                                                    # tbound=295.
                                                                    # tbound=280.
                                                                    # tbound=tbounds[i_tbound]
                                                                    g1=3.6478
                                                                    g2=0.83209
                                                                    g3=11.3515
                                                                    q0295=12e-3 # other values for different tbounds
                                                                    zq1=4000.
                                                                    zq2=7500.
                                                                    zt=15000.
                                                                    qt=1.e-11
                                                                    # altz=np.linspace(0.,40.,nlayers+1)
                                                                    altz=np.linspace(0.,30.,nlayers+1)
                                                                    altz*=1.e3
                                                                    trop_ind = np.argmin(abs(altz-15000.))
                                                                    q=np.zeros(nlayers+1)
                                                                    for i in range(nlayers+1):
                                                                        if(altz[i]<zt):
                                                                            q[i]=q0295*exp(-altz[i]/zq1)*exp(-(altz[i]/zq2)**2.)
                                                                        else:
                                                                            q[i]=qt
                                                                    tv0=tbound*(1.+0.608*q0295)
                                                                    tv=np.zeros(nlayers+1)
                                                                    for i in range(nlayers):
                                                                        if(altz[i]<zt):
                                                                            tv[i]=tv0-lapse*altz[i]/1000.
                                                                        else:
                                                                            tv[i]=tv0-lapse*zt/1000.
                                                                    tz=tv/(1.+0.608*q)
                                                                    for i in range(nlayers+1):
                                                                        if(altz[i]<zt):
                                                                            pz[i]=psurf*( ( tv0-lapse*altz[i]/1000. )/tv0 ) ** ( gravity/( rsp*lapse/1000. ) )
                                                                        else:
                                                                            pt=psurf*(tv[trop_ind]/tv0)**(gravity/(rsp*lapse/1000.))
                                                                            pz[i]=pt*np.exp( -( gravity*( altz[i]-altz[trop_ind] ) / ( rsp*tv[trop_ind] ) ) )
                                                                    for i in range(len(pavel)):
                                                                        pavel[i]=(pz[i]+pz[i+1])/2.
                                                                    tavel=np.zeros(nlayers)
                                                                    for i in range(len(pavel)):
                                                                        tavel[i]=(tz[i]+tz[i+1])/2.
                                                                elif(master_input==5):
                                                                    pz=np.linspace(psurf,pmin,nlayers+1)
                                                                    for i in range(len(pavel)):
                                                                        pavel[i]=(pz[i]+pz[i+1])/2.
                                                                    altz[0] = 0.0
                                                                    for k in range(3):
                                                                        for i in range(1,nlayers):
                                                                            altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity
                                                                        altz[nlayers] = altz[nlayers-1]+(pz[nlayers-1]-pz[nlayers])*rsp*tavel[nlayers-1]/pavel[nlayers-1]/gravity
                                                                        tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
                                                                        tz=np.clip(tz,tmin,tmax)
                                                                        tavel=np.zeros(nlayers)
                                                                        for i in range(nlayers):
                                                                            tavel[i]=(tz[i]+tz[i+1])/2.
                                                                        tavel[nlayers-1] = tavel[nlayers-2]
    
    
                                                                for i in range(nlayers):
                                                                    altavel[i]=(altz[i]+altz[i+1])/2.
    
                                                                for i in range(nlayers):
                                                                    # h2o (manabe mw67)
                                                                    esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
                                                                    rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
                                                                    vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
                                                                    if(i>1 and vol_mixh2o[i] > vol_mixh2o[i-1]):
                                                                        vol_mixh2o[i]=vol_mixh2o[i-1]
                                                                    vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
    
                                                                # Mean molecular weight of the atmosphere
                                                                mmwtot = mmwco2 * vol_mixco2 + mmwn2 * vol_mixn2 + mmwo2 * vol_mixo2 + mmwar*vol_mixar + mmwch4*vol_mixch4 + mmwh2o*vol_mixh2o[0]+mmwo3*vol_mixo3[0]
    
                                                                for i in range(nlayers):
                                                                    # mperlayr[i] = totmolec/nlayers #Divide the molecules equally between layers
                                                                    # mperlayr_air[i] = (molec_n2 + molec_o2)/(nlayers)
                                                                    mperlayr[i] = totmolec*(pz[i]-pz[i+1])/pz[0] #pressure weighting
                                                                    mperlayr_air[i] = (molec_n2 + molec_o2)*(pz[i]-pz[i+1])/pz[0]
    
                                                                for i in range(nlayers):
                                                                    vol_mixo3[i] = (3.6478*(pz[i]**0.83209))*np.exp(-pz[i]/11.3515)*1e-6
    
                                                                #Set up mixing ratio of broadening molecules (N2 and O2 mostly)
                                                                if(master_input==0): # manual input
                                                                    for i in range(nlayers):
                                                                        wbrodl[i] = mperlayr_air[i] * 1.0e-4
                                                                        wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]
                                                                        # wkl[1,i] = mperlayr[i] * 1.0e-4 * 123e-6
                                                                        # wkl[2,i] = mperlayr[i] * 1.0e-4 * vol_mixco2
                                                                        wkl[2,i] = mperlayr[i] * 1.0e-4 * 400e-6
                                                                        wkl[3,i] = mperlayr[i] * 1.0e-4 * vol_mixo3[i]*0.
                                                                        # wkl[3,i] = mperlayr[i] * 1.0e-4 * 456e-6
                                                                        wkl[6,i] = mperlayr[i] * 1.0e-4 * vol_mixch4*0.
                                                                        wkl[7,i] = mperlayr[i] * 1.0e-4 * vol_mixo2*0.
                                                                    # wkl = np.clip(wkl,1.,1e63) #only if wkl is molec/cm, not mixing ratio!
                                                                elif(master_input==3): # wkl RCEMIP
                                                                    g1=3.6478
                                                                    g2=0.83209
                                                                    g3=11.3515
                                                                    for i in range(nlayers):
                                                                        wbrodl[i] = mperlayr_air[i] * 1.0e-4
                                                                        # if(altz[i]/1000.<15.):
                                                                        if(q[i]>qt):
                                                                            wkl[1,i]=q[i]*1e-3
                                                                        else:
                                                                            wkl[1,i]=qt*1e-3
                                                                        wkl[2,i]=348e-6*4. # co2
                                                                        wkl[3,i]=g1*pz[i]**(g2)*np.exp(-1.0*(pz[i]/g3))*1e-6 # o3
                                                                        wkl[4,i]=306e-9 # n2o
                                                                        wkl[5,i]=0.# co
                                                                        wkl[6,i]=1650e-9 # ch4
                                                                        wkl[7,i]=0. # o2
                                                                elif(master_input==5):
                                                                    surf_rh=0.8
                                                                    for i in range(nlayers):
                                                                        esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
                                                                        rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
                                                                        vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
                                                                        if(i>1 and vol_mixh2o[i] > vol_mixh2o[i-1]):
                                                                            vol_mixh2o[i]=vol_mixh2o[i-1]
                                                                        vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
                                                                        wbrodl[i] = mperlayr_air[i] * 1.0e-4
                                                                        wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]
                                                                        # wkl[1,i] = mperlayr[i] * 1.0e-4 * 123e-6
                                                                        # wkl[2,i] = mperlayr[i] * 1.0e-4 * vol_mixco2
                                                                        wkl[2,i] = mperlayr[i] * 1.0e-4 * 400e-6
                                                                        wkl[3,i] = mperlayr[i] * 1.0e-4 * vol_mixo3[i]*0.
                                                                        # wkl[3,i] = mperlayr[i] * 1.0e-4 * 456e-6
                                                                        wkl[6,i] = mperlayr[i] * 1.0e-4 * vol_mixch4*0.
                                                                        wkl[7,i] = mperlayr[i] * 1.0e-4 * vol_mixo2*0.
                                                                    
                                                                color=[]
                                                                for i in range(nlayers+1):
                                                                    color.append('#%06X' % randint(0, 0xFFFFFF))
        
                                                                if(master_input==6): #ERA-I
                                                                    q,o3,fal,r = read_erai()
                                                            
                                                            if(input_source==2):
                                                                q,o3,fal,r = read_erai()
    
                                                            dmid=np.ones(nlatcols)*10.
                                                            dtrop=np.ones(nlatcols)*15.
                                                            ztrop_h82=np.ones(nlatcols)*12.5
                                                            ztrop_h82_ind=np.zeros(nlatcols)
                                                            ztrop=np.ones(nlatcols)*12.5
                                                            lapse_add=np.zeros(nlatcols)
                                                            
                                                            ts_rec=[]
                                                            maxdfnet_rec=[]
    
                                                            # main loop (timestepping)
                                                            for ts in range(timesteps):
    
                                                                for i_lat in range(nlatcols):
                                                                   
                                                                    conv_on=conv_on_lats[i_lat]
    
                                                                    if(ts==1):
                                                                        tbound = tbound_inits[i_lat]
                                                                        tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
                                                                        tz=np.clip(tz,200,tmax)
    
                                                                    if(input_source==0):
                                                                        
                                                                        wbrodl = mperlayr_air * 1.0e-4
                                                                        wkl[1,:]=q[:,i_lat]
                                                                        wkl[2,:]=400e-6*wklfac_co2
                                                                        wkl[3,:]=o3[:,i_lat]
                                                                        sza=szas[i_lat]
                                                                        lapse=lapse_master[0,i_lat]
                                                                        # lapse=lapseloop
                                                                        if(master_input==6):
                                                                            semiss=np.ones(29)*(1.0-fal[i_lat])
    
                                                                    if(input_source==2):
                                                                        wbrodl = mperlayr_air * 1.0e-4
                                                                        sza=szas[i_lat]
                                                                        semiss=np.ones(29)*(1.0-fal[i_lat])
                                                                        lapse=lapse_master[0,i_lat]
                                                                        # lapse=6.5
                                                                        # lapse=lapseloop
                                                                    lapse_master[0,i_lat]=lapse
                                                                    lapse_master[1,i_lat]=lapse
    
                                                                    for i_zon in range(nzoncols):
    
                                                                        if(ts>1 or input_source==1 or input_source==2):
    
                                                                            tbound=tbound_master[i_zon,i_lat]    
                                                                            tz=tz_master[:,i_zon,i_lat]
                                                                            tavel=tavel_master[:,i_zon,i_lat]    
                                                                            pz=pz_master[:,i_zon,i_lat]
                                                                            pavel=pavel_master[:,i_zon,i_lat]
                                                                            altz=altz_master[:,i_zon,i_lat]
                                                                            altavel=altavel_master[:,i_zon,i_lat]
                                                                            fnet=fnet_master[:,i_zon]
                                                                            wkl[1,:]=wkl_master[:,i_zon,0,i_lat]
                                                                            wkl[2,:]=wkl_master[:,i_zon,1,i_lat]
                                                                            wkl[3,:]=wkl_master[:,i_zon,2,i_lat]
                                                                            wkl[4,:]=wkl_master[:,i_zon,3,i_lat]
                                                                            wkl[5,:]=wkl_master[:,i_zon,4,i_lat]
                                                                            wkl[6,:]=wkl_master[:,i_zon,5,i_lat]
                                                                            wkl[7,:]=wkl_master[:,i_zon,6,i_lat]
                                                                            wbrodl=wbrodl_master[:,i_zon,i_lat]    
                                                                            toa_fnet=toa_fnet_master[i_zon,i_lat]
    
                                                                            # inflags=inflags_master[:,i_lat]
                                                                            # iceflags=iceflags_master[:,i_lat]
                                                                            # liqflags=liqflags_master[:,i_lat]
    
                                                                            # cld_lays=cld_lays_master[:,i_lat]
                                                                            # cld_fracs=cld_fracs_master[:,i_lat]
                                                                            # tauclds=tauclds_master[:,i_lat]
                                                                            # ssaclds=ssaclds_master[:,i_lat]
                                                                            # zonal_transps=zonal_transps_master[:,i_lat]
                                                                            # column_budgets=column_budgets_master[:,i_lat]
    
                                                                            totuflux_lw=totuflux_lw_master[:,i_zon,i_lat]
                                                                            totdflux_lw=totdflux_lw_master[:,i_zon,i_lat]
                                                                            totuflux_sw=totuflux_sw_master[:,i_zon,i_lat]
                                                                            totdflux_sw=totdflux_sw_master[:,i_zon,i_lat]
                                                                            fnet_lw=fnet_lw_master[:,i_zon,i_lat]
                                                                            fnet_sw=fnet_sw_master[:,i_zon,i_lat]
                                                                            
                                                                        H=7. #fix scale height here
                                                                        phi=np.deg2rad(latgrid)
                                                                        phi_edges=np.deg2rad(latgridbounds)
                                                                        
                                                                        if(lapse_source==1):
                                                                            lapse_eqb[i_lat]=1
                                                                        
                                                                        if(lapse_source==2):                                                                        
                                                                            if(abs(latgrid[i_lat])<90):
                                                                                conv_on_lats[i_lat]=1                                                                            
                                                                                if(nlatcols>1):
                                                                                    f=interp1d(latgrid,tbound_master[0,:],bounds_error=False,fill_value="extrapolate")
                                                                                    tbound_edges=f(latgridbounds)                                                                    
                                                                                    dmid[i_lat]=H * np.log( 1. - ( np.tan( phi[i_lat] ) * (tbound_edges[i_lat+1]-tbound_edges[i_lat]) ) / ( (phi_edges[i_lat+1] - phi_edges[i_lat]) * H * (9.8-lapse)) ) * 1.5            
                                                                                dtrop[i_lat]=0.8*2.26e6*(6.1094*exp(17.625*(np.mean(tbound_master[:,i_lat])-273.15)/(np.mean(tbound_master[:,i_lat])-273.15+243.04)))/pz[0]*0.622/(1e3*(9.8-lapse) ) * 1.5
                                                                                ztrop_h82[i_lat]=np.max([dmid[i_lat],dtrop[i_lat]])
                                                                                ztrop_h82_ind[i_lat]=np.argmin(np.abs(altz/1000.-ztrop_h82[i_lat]))
                                                                                
                                                                                lapse=lapse_master[0,i_lat]
                                                                                lapse=np.clip(lapse,0.,12.)
                                                                                
                                                                                lapse_eqb[i_lat]=0
                                                                                
                                                                                if(rad_eqb[i_lat]==1):
                                                                                # if(rad_eqb[i_lat]!=2):
                                                                                # if(ts%10==0):
                                                                                    
                                                                                    if(abs(latgrid[i_lat])<60):
                                                                                        ztrop[i_lat]=np.mean(altz_master[np.int(np.mean(cti_master[:,i_lat])),:,i_lat])/1000.
                                                                                    else:
                                                                                        ztrop[i_lat]=np.mean(altz_master[np.int(np.mean(np.argmin(tz_master[:,0,i_lat]))),:,i_lat])/1000.
                                                                                    
                                                                                    if( abs(ztrop[i_lat]-ztrop_h82[i_lat]) < 1.0): #H82 tightness criterion
                                                                                        lapse_eqb[i_lat]=1
                                                                                        lapse_master[0,i_lat]=lapse
                                                                                    else:
                                                                                        # lapse_master[i_lat]+=( dmid[ i_lat ] - np.mean(altz_master[ np.int( np.mean( cti_master[:,i_lat], axis=0 ) ),:,i_lat],axis=0) / 1000. )*-0.05
                                                                                        dztrop[i_lat] = ztrop_h82[i_lat] - ztrop[i_lat]
                                                                                        lapse_add[i_lat]=np.clip(dztrop[i_lat] * -0.01,-0.1,0.1)
                                                                                        lapse+=lapse_add[i_lat]
                                                                                        if(np.isnan(lapse)):
                                                                                            lapse=6.5
                                                                                        lapse=np.clip(lapse,0.,12.)
                                                                                        lapse_master[0,i_lat]=lapse
                                                                                        convection(tavel,altavel,conv_log=1)
                                                                                        convection(tz,altz,conv_log=1)
            
                                                                                    if(ts%10==0):
                                                                                        for i_lat2 in range(nlatcols):                                                                        
                                                                                            print('ts: {} lat: {: 6.0f} ztrop:  {: 6.2f} ztrop_h82: {: 6.2f} dztrop:  {: 6.2f} new lapse  {: 6.2f} lapse_add {: 6.2f} rad_eqb: {: 4.0f} lapse_eqb: {: 4.0f} '.format(ts, latgrid[i_lat2], ztrop[i_lat2],ztrop_h82[i_lat2], dztrop[i_lat2], lapse_master[0,i_lat2],lapse_add[i_lat2], rad_eqb[i_lat2], lapse_eqb[i_lat2] ) )
                                                                                        print('------------------')
            
                                                                                lapse_master[0,i_lat]=lapse
                                                                                
                                                                        elif(lapse_source==3):
                                                                            lapse=lapse_master[0,i_lat]
                                                                            midtrop_ind = np.max([np.int( np.mean( cti_master[:,i_lat] )/2 ),nlayers/10.*6.]).astype('int')
                                                                            f=interp1d(latgrid,tbound_master[0,:],bounds_error=False,fill_value="extrapolate")
                                                                            tbound_edges=f(latgridbounds)
                                                                            if(i_lat<nlatcols-1):
                                                                                gamma_c = 9.8 + np.tan( np.deg2rad( latgrid[i_lat] ) ) / (altz[midtrop_ind]/1000.) * ( tz_master[midtrop_ind,0,i_lat+1] - tz_master[midtrop_ind,0,i_lat] ) / ( np.deg2rad(latgridbounds[i_lat+1] - latgridbounds[i_lat]) )
                                                                            else:
                                                                                gamma_c = 9.8 + np.tan( np.deg2rad( latgrid[i_lat] ) ) / (altz[midtrop_ind]/1000.) * ( tz_master[midtrop_ind,0,i_lat] - tz_master[midtrop_ind,0,i_lat-1] ) / ( np.deg2rad(latgridbounds[i_lat] - latgridbounds[i_lat-1]) )
                                                                            es=np.zeros(nlayers)
                                                                            gamma_m=np.zeros(nlayers+1)
                                                                            L_h2o=np.zeros(nlayers)
                                                                            for i in range(nlayers-1):
                                                                                L_h2o[i]=2510. - 2.38 * (tz[i] - 273.)
                                                                                es[i] = 6.11 * np.exp( ( 0.622 * L_h2o[i] / 0.287 ) * ( 1. / 273. - 1./tz[i] ) )
                                                                                gamma_m[i]=9.8 * ( 1. + 0.622 * L_h2o[i] * es[i] / ( pz[i] * 0.287 * tz[i] ) ) / ( 1. + ( (0.622 * L_h2o[i]  ) / ( 1.005 * pz[i] ) ) * ( (0.622 * L_h2o[i] * es[i]  ) / ( 0.287 * tz[i]**2 ) ) )
                                                                                
                                                                            trop_mean_gamma_m = np.mean(gamma_m[:np.int(nlayers/2)])
                                                                                
                                                                            if(ts%20==0):
                                                                                lapse+=(gamma_c-lapse)*0.005
                                                                            lapse=np.clip(lapse,1,12)
                                                                            lapse_master[0,i_lat]=lapse
                                                                            if( abs(lapse-gamma_c) < 0.1 ):
                                                                                lapse_eqb[i_lat]=1
                                                                            else:
                                                                                lapse_eqb[i_lat]=0
                                                                            # lapse_eqb[i_lat]=1
                                                                            if(ts%50==0):
                                                                                print('{: 6.2f} {: 6.2f} {: 6.2f} {: 6.2f} {: 6.2f}'.format(i_lat,lapse, midtrop_ind,gamma_c, trop_mean_gamma_m))
                                                                                
                                                                            if(ts>10):
                                                                                if(gamma_c < trop_mean_gamma_m):
                                                                                    convection(tavel,altavel,conv_log=1)
                                                                                    convection(tz,altz,conv_log=1)
                                                                                else:
                                                                                    lapse=np.clip(trop_mean_gamma_m,1,12)
                                                                                    convection_moist(tavel,altavel,conv_log=1)
                                                                                    convection_moist(tz,altz,conv_log=1)
                                                                                    lapse_master[0,i_lat]=lapse
                                                                                    lapse_eqb[i_lat]=1
                                                                                    
                                                                        elif(lapse_source==4):
                                                                            adv_on=np.ones(nlatcols)
                                                                            lapse_eqb=np.ones(nlatcols)
                                                                            conv_on_lats=np.zeros(nlatcols)
                                                                            lapse=9.8
                                                                            convection(tavel,altavel,conv_log=1)
                                                                            convection(tz,altz,conv_log=1)
                                                                            lapse_master[0,i_lat]=lapse
                                                                            
                                                                            # else:
                                                                            #     conv_on_lats[i_lat]=0
                                                                            #     adv_on[i_lat]=1
                                                                            #     lapse_eqb[i_lat]=1
                                                                            #     lapse_master[0,i_lat]=lapse
    
                                                                        # cldweights,altbins,tauclds=read_misr()
                                                                        # read_misr_2()
                                                                        # cld_fracs_master[i_zon,i_lat,:],altbins,tauclds_master[0,i_lat,:]=read_misr()
    #                                                                   cld_fracs_master[i_zon,:,:],altbins,tauclds_master[0,:,:]=read_misr_2()
    #                                                                   if(ts==1):
                                                                        # cld_fracs_master[i_zon,:,:],altbins,tauclds_master[0,:,:]=read_misr_3()
                                                                        # cld_lay_v2=0
                                                                        # cf_mro=0
                                                                        # od_eff=0
                                                                        cld_lay_v2[i_lat], cf_mro[i_lat], od_eff[i_lat] = read_misr_4()
                                                                        
                                                                        cld_fracs_master[i_zon,:,np.int(cld_lay_v2[i_lat]-1)]=cf_mro[i_lat]
                                                                        tauclds_master[i_zon,:,np.int(cld_lay_v2[i_lat]-1)]=od_eff[i_lat]
                                                                        ssaclds_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                                        ssaclds_master[i_zon,:,np.int(cld_lay_v2[i_lat]-1)]=0.5
                                                                        
                                                                        # cld_fracs_master[i_zon,:,np.int(nlayers/1.5)]=0.68
                                                                        # tauclds_master[i_zon,:,np.int(nlayers/1.5)]=6.0
                                                                        # ssaclds_master=np.zeros((nzoncols,nlatcols,nclouds))
                                                                        # ssaclds_master[i_zon,:,np.int(nlayers/1.5)]=0.5
                                                                        
                                                                        # for i in range(nlayers):
                                                                        #     if(i%10!=0):
                                                                        #         cld_fracs_master[i_zon,:,i]=0.
                                                                        #         tauclds_master[0,:,i]=0.

                                                                        if(i_lat==pertlat):
                                                                            es=np.zeros(nlayers+1)
                                                                            L_h2o=np.zeros(nlayers+1)
                                                                            pert=np.zeros(nlayers)
                                                                            ws=np.zeros(nlayers+1)
                                                                            for i_lay in range(pertlay,pertlay+6):
                                                                                # tauclds_master[0,i_lat,i_lay] = tauclds_master[0,i_lat,i_lay] * pert
                                                                                t1=tavel[i_lay]-273.15 # temperature in Celsius
                                                                                pert[i_lay] = 6.1094 * np.exp( (17.625*(t1+1.)) / ( (t1+1.) + 243.04 ) ) / (6.1094 * np.exp( (17.625*t1) / ( t1 + 243.04 ) ) ) #perturbation equivalent to that of the relative increase of water vapor with a 1 K increase in temperature
                                                                                tauclds_master[0,i_lat,i_lay] = tauclds_master[0,i_lat,i_lay] * pert[i_lay]
                                                                                
                                                                        for i_cld in range(nclouds):
                                                                            # cld_lays_master[i_zon,i_lat,i_cld]=np.argmin(abs(altz/1000.-altbins[i_cld]))
                                                                            # cld_lays_master[i_zon,i_lat,i_cld]=np.argmin(abs(altz/1000.-altbins[i_cld]))
                                                                            cld_lays_master[i_zon,i_lat,i_cld]=i_cld+1
    
    
                                                                        # inflag=inflags[i_zon].astype('int')
                                                                        # iceflag=iceflags[i_zon].astype('int')
                                                                        # liqflag=liqflags[i_zon].astype('int')
                                                                        # cld_lay=cld_lays[i_zon].astype('int')
                                                                        # cld_lay=np.argmin(abs(altz/1000.-altbins[i_zon]))
                                                                        # cld_lay=np.argmin(abs(pz-700.))
                                                                        # frac=cld_fracs[i_zon]
                                                                        # taucld=tauclds[i_zon]
                                                                        # ssacld=ssaclds[i_zon]
    
                                                                        cld_lay=0 # dummy just for compatibility with old code
    
                                                                        vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,nzoncols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb,nlatcols]
                                                                        vars_master_lay_zon_lat=[tavel_master,pavel_master,altavel_master,wbrodl_master,dfnet_master,dfnet_master_rad,dfnet_master_adv,dpz_master]
                                                                        vars_master_lev_zon_lat=[tz_master,pz_master,altz_master,totuflux_master,totuflux_lw_master,totuflux_sw_master,totdflux_master,totdflux_lw_master,totdflux_sw_master,fnet_master,fnet_lw_master,fnet_sw_master,htr_master,htr_lw_master,htr_sw_master,conv_master]
                                                                        vars_misc_1d=[semis,semiss,solvar]
                                                                        vars_misc_1d_lens=[16,29,29]
                                                                        vars_master_lay_zon_nmol_lat=[wkl_master]
                                                                        vars_master_zon_lat=[inflags_master,iceflags_master,liqflags_master,tbound_master,toa_fnet_master,zonal_transps_master,merid_transps_master,cti_master,lapse_master]
                                                                        vars_master_zon_lat_cld=[cld_lays_master,cld_fracs_master,tauclds_master,ssaclds_master]
                                                                        vars_master_lat=[latgrid]
    
    
                                                                        if(ts>0):
                                                                            for i in range(1,nlayers):
                                                                                ur[i] = ur_min
                                                                            
                                                                            if(input_source==0 or input_source==2):
                                                                                conv=np.zeros(nlayers+1) #reset to zero
                                                                                conv[0]=1 # set conv of lowest layer to on, otherwise it sometimes gets misidentified 
    
                                                                                if(master_input==0 or master_input==3):
                                                                                    surf_rh=0.8
                                                                                    for i in range(nlayers):
                                                                                        esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
                                                                                        rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
                                                                                        rel_hum=np.clip(rel_hum,0.0,0.8)
                                                                                        vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
                                                                                        if(i>1 and vol_mixh2o[i] > vol_mixh2o[i-1]):
                                                                                            vol_mixh2o[i]=vol_mixh2o[i-1]
                                                                                        vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
                                                                                        if(master_input==0):
                                                                                            wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]
                                                                                        elif(master_input==3):
                                                                                            if(i_zon==0):
                                                                                                wkl[1,i] = vol_mixh2o[i]
                                                                                            elif(i_zon==1):
                                                                                                wkl[1,i] = vol_mixh2o[i]*wklfac
    
                                                                                if(master_input==6):
                                                                                    for i in range(nlayers):
                                                                                        esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
                                                                                        # if(input_source==0):
                                                                                        rel_hum[i]=r[i,i_lat]/100.
                                                                                        vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
                                                                                        # wkl[1,i] = vol_mixh2o[i]
                                                                                        wkl[1,i]=q[i,i_lat]
                                                                                    if(i_zon==0):
                                                                                        # wkl[1,:]=wkl[1,:]*2.
                                                                                        wkl[1,:]=wkl[1,:]*1.
                                                                                    elif(i_zon==1):
                                                                                        # wkl[1,:]=wkl[1,:]/2.
                                                                                        wkl[1,:]=wkl[1,:]/1.
                                                                                    wkl[1,:]=np.clip(wkl[1,:],0,0.2) # sensible bounds for H2O amount
    
                                                                            # if(i_zon==1 and ts==1):
                                                                            #   wkl[1,:]=wkl_master[:,i_zon,0]*wklfac
    
                                                                        # if(input_source==0):
                                                                        #   dtbound=toa_fnet*0.1*0.5
                                                                        #   dtbound=np.clip(dtbound,-dmax,dmax)
                                                                        #   tbound+=dtbound
    
    
                                                                        # perturb surface temperature to reduce column energy imbalance
                                                                        if((input_source==0 and ts>150) or input_source==2):
                                                                            # dtbound=toa_fnet*0.1*0.5*0.1
                                                                            dtbound=column_budgets_master[i_zon,i_lat]*0.01
                                                                            if(input_source==2):
                                                                                dtbound*=2.
                                                                            dtbound=np.clip(dtbound,-dmax,dmax)
                                                                            tbound+=dtbound*0.
                                                                        tbound=np.clip(tbound,tmin,tmax)
    
                                                                        # if(input_source==0 and master_input==5)
    
    
                                                                        # if(input_source==1):
                                                                        #   dttrop=-0.00
                                                                        #   dptrop=gravity*pz[cti]*dttrop/(rsp*lapse*1e-3*tz[cti])
                                                                        #   print(dptrop, 'dptrop')
                                                                        #   print(pz[cti])
                                                                        #   tz[cti]+=dttrop
                                                                        #   pz[cti]+=dptrop
    
                                                                        writeformattedcloudfile()
    
                                                                        if(input_source==1 or input_source==2):
                                                                            if(i_zon==pertzon and i_lat==pertlat):
                                                                                for i_lay in range(pertlay,pertlay+6):
                                                                                    wkl[pertmol,i_lay]*=pert
    
                                                                        # the actual meat! call the compiled RRTM executable for LW radiative transfer
                                                                        if(lw_on==1):
                                                                            writeformattedinputfile_lw()
                                                                            callrrtmlw()
                                                                            totuflux_lw,totdflux_lw,fnet_lw,htr_lw = readrrtmoutput_lw()
    
                                                                        # call the compiled RRTM executable for SW radiative transfer, but as infrequently as possible because it's expensive
                                                                        # if((ts==2 or (abs(maxdfnet_tot) < eqb_maxdfnet*10. and stepssinceswcalled>500)) and sw_on==1):
                                                                        # if((sw_on==1 and ts%300==1 and input_source==0) or ((input_source==1 or input_source==2) and (ts==3 and sw_on==1))):
                                                                        if(rad_eqb[i_lat]==1 and sw_called<nlatcols*2  and sw_on==1 or (sw_on==1 and (ts==5 or ts==300) )):
                                                                            sw_called+=1
                                                                        #   if(maxhtr<eqb_maxhtr):
                                                                            writeformattedinputfile_sw()
                                                                            callrrtmsw()
                                                                            print('RRTM SW Called')
                                                                            stepssinceswcalled=0
                                                                            totuflux_sw,totdflux_sw,fnet_sw,htr_sw = readrrtmoutput_sw()
                                                                        stepssinceswcalled+=1
    
                                                                        # perturb gas amounts in 6 layer blocks (should be 100 hPa, so will change)
                                                                        if(input_source==1 or input_source==2):
                                                                            if(i_zon==pertzon and i_lat==pertlat):
                                                                                for i_lay in range(pertlay,pertlay+6):
                                                                                    wkl[pertmol,i_lay]/=pert
    
                                                                        prev_htr=htr
    
                                                                        # normalise the fluxes to match DW SW of 238 Wm^-2
                                                                        if(ts>1 and master_input==2):
                                                                            totuflux_sw*=(238./fnet_sw[nlayers])
                                                                            totdflux_sw*=(238./fnet_sw[nlayers])
                                                                            htr_sw*=(238./fnet_sw[nlayers])
                                                                            fnet_sw*=(238./fnet_sw[nlayers])
                                                                            
                                                                        # add LW and SW fluxes together
                                                                        totuflux=totuflux_lw+totuflux_sw
                                                                        totdflux=totdflux_lw+totdflux_sw
                                                                        fnet=fnet_sw-fnet_lw
                                                                        htr=htr_lw+htr_sw

                                                                        # writeoutputfile_masters()
    
                                                                        # toa_fnet=totdflux[nlayers]-totuflux[nlayers] #net total downward flux at TOA
                                                                        toa_fnet=fnet_sw[nlayers]-fnet_lw[nlayers]
                                                                        # toa_fnet=totdflux[nlayers]-totuflux[nlayers]+zonal_transps[i_zon] #net total downward flux at TOA  NJE now accounting for zonal transport
    
                                                                        prev_maxhtr=maxhtr*1.0
                                                                        re_htrs = np.where(conv==0,htr,0.)
                                                                        maxhtr=max(abs(re_htrs))
                                                                        maxhtr_ind=np.argmax(abs(re_htrs))
                                                                        # dfnet=np.zeros(nlayers)
                                                                        # dpz=np.zeros(nlayers)
                                                                        # for i in range(nlayers):
                                                                        #   dfnet[i]=fnet[i+1]-fnet[i]
                                                                        #   dpz[i]=pz[i+1]-pz[i]
                                                                        # maxdfnet=max(abs(dfnet))
    
                                                                        if(input_source==0 or input_source==2):
                                                                            for i in range(1,nlayers):
                                                                                if(lay_intp==0):
                                                                                    tz[i] = (tavel[i-1] + tavel[i])/2.
                                                                                else:
                                                                                    tz[i] = tavel[i-1]*1.0
    
                                                                            if(lay_intp==0):
                                                                                tz[nlayers] = 2*tavel[nlayers-1]-tz[nlayers-1]
                                                                            else:
                                                                                tz[nlayers]=tavel[nlayers-1]
                                                                            
                                                                            altz[0] = 0.0
                                                                            for i in range(1,nlayers):
                                                                                altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity
                                                                            altz[nlayers] = altz[nlayers-1]+(pz[nlayers-1]-pz[nlayers])*rsp*tavel[nlayers-1]/pavel[nlayers-1]/gravity
                                                                            for i in range(nlayers-1):
                                                                                altavel[i]=(altz[i]+altz[i+1])/2.
    
                                                                            conv=np.zeros(nlayers+1) #reset to zero
                                                                            conv[0]=1
    
                                                                            if(surf_lowlev_coupled==1):
                                                                                tz[0]=tbound

                                                                            if(conv_on_lats[i_lat]==1 and (abs(latgrid[i_lat])<60. or lapse_source!=2) ):
                                                                                convection(tavel,altavel,1)
                                                                                convection(tz,altz,1)
    
                                                                            tavel=np.clip(tavel,tmin,tmax)
                                                                            tz=np.clip(tz,tmin,tmax)
                                                                            
                                                                            cti=0
                                                                            # for i in range(2,len(conv_master[:,i_zon,i_lat])):
                                                                            for i in range(2,nlayers):
                                                                                if(conv[i]==1):
                                                                                    cti=i
                                                                                else:
                                                                                    continue
        
                                                                            cti_master[i_zon,i_lat]=cti
    
                                                                        # treat zonal transport as a diffusion
                                                                        zonal_transps_master[0,i_lat]=(tbound_master[1,i_lat]-tbound_master[0,i_lat])*c_zonal
                                                                        zonal_transps_master[1,i_lat]=(tbound_master[0,i_lat]-tbound_master[1,i_lat])*c_zonal
    
                                                                        # treat meridional transport as diffusion
                                                                        if(nlatcols>1):
                                                                            # mti=np.int(nlayers/2) # merid transp index
                                                                            mti=1
                                                                            if(i_lat>0 and i_lat<nlatcols-1):
                                                                                # merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[0,i_zon,i_lat+1]-tz_master[0,i_zon,i_lat]) + c_merid*(tz_master[0,i_zon,i_lat-1]-tz_master[0,i_zon,i_lat]))*latweights_area[i_lat]
                                                                                merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[mti,i_zon,i_lat+1]-tz_master[mti,i_zon,i_lat]) + c_merid*(tz_master[mti,i_zon,i_lat-1]-tz_master[mti,i_zon,i_lat]))*latweights_area[i_lat]
                                                                            elif(i_lat==0):
                                                                                # merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[0,i_zon,i_lat+1]-tz_master[0,i_zon,i_lat]))*latweights_area[i_lat]
                                                                                merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[mti,i_zon,i_lat+1]-tz_master[mti,i_zon,i_lat]))*latweights_area[i_lat]
                                                                            elif(i_lat==nlatcols-1):
                                                                                # merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[0,i_zon,i_lat-1]-tz_master[0,i_zon,i_lat]))*latweights_area[i_lat]
                                                                                merid_transps_master[i_zon,i_lat]=(c_merid*(tz_master[mti,i_zon,i_lat-1]-tz_master[mti,i_zon,i_lat]))*latweights_area[i_lat]
    
                                                                        
                                                                        column_budgets_master[i_zon,i_lat]=toa_fnet+merid_transps_master[i_zon,i_lat]+zonal_transps_master[i_zon,i_lat]+extra_forcing  #nje forcing


                                                                        # add current values of variables to master arrays
                                                                        tz_master[:,i_zon,i_lat]=tz
                                                                        tavel_master[:,i_zon,i_lat]=tavel
                                                                        pz_master[:,i_zon,i_lat]=pz
                                                                        pavel_master[:,i_zon,i_lat]=pavel
                                                                        altz_master[:,i_zon,i_lat]=altz
                                                                        altavel_master[:,i_zon,i_lat]=altavel
                                                                        totuflux_master[:,i_zon,i_lat]=totuflux
                                                                        totuflux_lw_master[:,i_zon,i_lat]=totuflux_lw
                                                                        totuflux_sw_master[:,i_zon,i_lat]=totuflux_sw
                                                                        totdflux_master[:,i_zon,i_lat]=totdflux
                                                                        totdflux_lw_master[:,i_zon,i_lat]=totdflux_lw
                                                                        totdflux_sw_master[:,i_zon,i_lat]=totdflux_sw
                                                                        fnet_master[:,i_zon,i_lat]=fnet
                                                                        fnet_lw_master[:,i_zon,i_lat]=fnet_lw
                                                                        fnet_sw_master[:,i_zon,i_lat]=fnet_sw
                                                                        htr_master[:,i_zon,i_lat]=htr
                                                                        htr_lw_master[:,i_zon,i_lat]=htr_lw
                                                                        htr_sw_master[:,i_zon,i_lat]=htr_sw
    
                                                                        tbound_master[i_zon,i_lat]=tbound
                                                                        toa_fnet_master[i_zon,i_lat]=toa_fnet
    
                                                                        # inflags_master[i_zon,i_lat]=inflags[i_zon]
                                                                        # iceflags_master[i_zon,i_lat]=iceflags[i_zon]
                                                                        # liqflags_master[i_zon,i_lat]=liqflags[i_zon]
    
                                                                        # cld_lays_master[i_zon,i_lat]=cld_lays[i_zon]
                                                                        # cld_fracs_master[i_zon,i_lat]=cld_fracs[i_zon]
                                                                        # tauclds_master[i_zon,i_lat]=tauclds[i_zon]
                                                                        # ssaclds_master[i_zon,i_lat]=ssaclds[i_zon]
    
                                                                        # zonal_transps_master[i_zon,i_lat]=zonal_transps[i_zon]
                                                                        
                                                                        if(input_source==0 or input_source==2):
                                                                            conv_master[:,i_zon,i_lat]=conv
                                                                        wbrodl_master[:,i_zon,i_lat]=wbrodl
    
                                                                        # for i_mol in range(nmol+1):
                                                                        for i_mol in range(1,nmol):
                                                                            wkl_master[:,i_zon,i_mol-1,i_lat] = wkl[i_mol,:]
                                                                            # wkl_master[:,i_zon,i_mol,i_lat] = wkl[i_mol,:]

                                                                        #  end i_zon loop
    
                                                                    # end loop latcols
    
                                                                # caluclate dFnet/dp to use as eqb criterion instead of htr
                                                                for i_lat in range(nlatcols):
                                                                    for i_zon in range(nzoncols):
                                                                        for i in range(nlayers):
                                                                            dfnet_master_rad[i,i_zon,i_lat]=fnet_master[i+1,i_zon,i_lat]-fnet_master[i,i_zon,i_lat]
                                                                            dpz_master[i,i_zon,i_lat]=pz_master[i+1,i_zon,i_lat]-pz_master[i,i_zon,i_lat]
    
                                                                # mtransp_dummy_tot=extra_forcing
                                                                
                                                                Q_adv=np.zeros((nlayers,nzoncols,nlatcols))
    
                                                                # for i in range(nlayers):
                                                                #   dfnet_master_adv[i,i_zon,i_lat]=mtransp_dummy_tot*(pz[i]-pz[i+1])/(pz[0])
    
                                                                # calculate advective flux vertical profile based on total meridional transport
                                                                for i_lat in range(nlatcols):
                                                                    for i_zon in range(nzoncols):
                                                                        mtransp_dummy_tot=toa_fnet_master[i_zon,i_lat]*-1.*0.65 # factor at the end to ensure stability, but fix later
                                                                        # mtransp_dummy_tot=merid_transps_master[i_zon,i_lat]
                                                                        if(adv_loc==0):
                                                                            for i in range(nlayers):
                                                                                Q_adv[i,i_zon,i_lat] = (nb * mtransp_dummy_tot * ((pz[i]/pz[0]) **(nb-1.) ))/nlayers
                                                                        elif(adv_loc==1):
                                                                            for i in range(0,np.int(cti_master[i_zon,i_lat])):
                                                                                Q_adv[i,i_zon,i_lat] = (nb * mtransp_dummy_tot * ((pz[i]/(pz[0]-pz[np.int(cti_master[i_zon,i_lat]) ] ) ) **(nb-1.) ))/nlayers
    
                                                                        # if(ts>1 and np.sum(Q_adv) != 0):
                                                                        #     Q_adv = Q_adv * ( mtransp_dummy_tot / np.sum(Q_adv) )
    
                                                                        # if(adv_on[i_lat]==1):
                                                                        #     for i in range(nlayers):
                                                                        #         dfnet_master_adv[i,i_zon,i_lat]=Q_adv[i,i_zon,i_lat]
                                                                                
                                                                dfnet_master_adv=Q_adv                                                                            
                                                                dfnet_master=dfnet_master_adv+dfnet_master_rad
    
                                                                # perturb temperature vertical profile by heating rates or flux divergences to approach equilibrium
                                                                if(input_source==0 or input_source==2):
                                                                    for i_lat in range(nlatcols):
                                                                        for i_zon in range(nzoncols):
                                                                            for i in range(nlayers):
                                                                                cldweights=[1.,0.]
                                                                                dT=(np.mean(dfnet_master[i,:,i_lat]/dpz_master[i,:,i_lat]*cldweights))*-1.*undrelax_lats[i_lat]
                                                                                dT=np.clip(dT,-maxdT[i_lat],maxdT[i_lat])
                                                                                if(input_source==2):
                                                                                    dT*=2.
                                                                                tavel_master[i,i_zon,i_lat]+=dT
                                                                        tavel_master=np.clip(tavel_master,tmin,tmax)

                                                                # find the maximum dfnet in any layer to work out distance from equilibrium    
                                                                re_dfnets=np.where(conv_master[cti+1:-1]==0,dfnet_master[cti+1:],0.)
                                                                maxdfnet_lat=np.zeros(nlatcols)
                                                                maxdfnet_ind=0
                                                                for i_lat in range(nlatcols):
                                                                    for i_zon in range(nzoncols):
                                                                        # for i in range(np.int(cti_master[i_zon,i_lat])+1,nlayers-1):
                                                                        # radbott1=np.int(cti_master[i_zon,i_lat])+1
                                                                        radbott1=nlayers-10
                                                                        rad_bottom=np.min([radbott1,nlayers-1]) 
                                                                        for i in range(rad_bottom,nlayers):
                                                                            if(abs(np.mean(dfnet_master[i,:,i_lat])) > abs(maxdfnet_lat[i_lat])):
                                                                                # maxdfnet_lat[i_lat]=abs(np.mean(dfnet_master[i,:,i_lat]))
                                                                                maxdfnet_lat[i_lat]=abs(dfnet_master[i,0,i_lat])
                                                                                # maxdfnet_lat[i_lat]=abs(dfnet_master_rad[i,0,i_lat])
                                                                                maxdfnet_ind=i
                                                                
    
                                                                # dfnet_master_mean=np.mean(dfnet_master,axis=1)
                                                                # conv_master_min=np.min(conv_master,axis=1)
                                                                # re_dfnets=np.where(conv_master_min[:-1]==0,dfnet_master_mean,0.)
                                                                # maxdfnet=np.max(np.abs(np.mean(re_dfnets,axis=1)))
                                                                if(nlatcols>1):
                                                                    for i_lat in range(nlatcols):
                                                                        # maxdfnet_lat[i_lat]=np.max(np.abs(np.mean(re_dfnets[10:,:,i_lat],axis=1)))
                                                                        # maxdfnet_lat[i_lat]=np.max(np.abs(np.mean(re_dfnets[:,:,i_lat],axis=1)))
                                                                        if(ts%20==5):
                                                                            # if(abs(maxdfnet_lat[i_lat])>abs(prev_max_dfnet_lat[i_lat])):
                                                                                # undrelax_lats[i_lat]/=1.5
                                                                                # maxdT[i_lat]/=2.
                                                                            prev_max_dfnet_lat[i_lat]=maxdfnet_lat[i_lat]*1.0
                                                                    maxdfnet_tot=np.max(np.abs(maxdfnet_lat))
                                                                else:
                                                                    # maxdfnet_lat[0]=np.max(np.abs(np.mean(re_dfnets[10:,:,0],axis=1)))
                                                                    maxdfnet_tot=maxdfnet_lat[0]
                                                                    if(ts%20==5):
                                                                        # if(abs(maxdfnet_lat[0])>abs(prev_max_dfnet_lat[0])):
                                                                            # maxdT[i_lat]/=2.
                                                                            # print('dT in {:2d} relaxed to {:5.3f}'.format(i_lat,maxdT[i_lat]))
                                                                        # else:
                                                                        #     if(maxdfnet_tot<0.5):
                                                                        #         undrelax_lats[0]*=1.01
                                                                        prev_max_dfnet_lat[0]=maxdfnet_lat[0]*1.0
                                                                # maxdfnet_tot=np.max(np.abs(np.mean(re_dfnets,axis=1)))
    
                                                                # maxdfnet_ind=np.argmax(abs(re_dfnets),axis=0)
                                                                # maxdfnet=dfnet[maxdfnet_ind]
                                                                                                                                
                                                                for i_lat in range(nlatcols):
                                                                    rad_eqb[i_lat]=0
                                                                    if(abs(maxdfnet_lat[i_lat] ) < eqb_maxdfnet and ts>50 ):
                                                                        rad_eqb[i_lat]=1
    
                                                                # colbudg_eqb=0
                                                                # if(np.max(abs(column_budgets_master))<eqb_col_budgs):
                                                                #     colbudg_eqb=1
                                                                
                                                                for i_lat in range(nlatcols):
                                                                    colbudg_eqb[i_lat]=0
                                                                    if(abs(np.mean(column_budgets_master[:,i_lat]) ) < eqb_col_budgs and ts>50 ):
                                                                        colbudg_eqb[i_lat]=1
                                                                

                                                                # print eqbseek
                                                                if(ts%20==0):
                                                                    print( '{: 4d}|'.format(ts))
                                                                    ts_rec.append(ts)
                                                                    maxdfnet_rec.append(np.max(maxdfnet_lat))
                                                                    plt.figure(1)
                                                                    plt.plot(ts_rec,maxdfnet_rec,'-o')
                                                                    plt.ylim(0.,np.max(maxdfnet_rec[-10:])*1.1)
                                                                    plt.axhline(-eqb_maxdfnet,linestyle='--')
                                                                    plt.axhline(eqb_maxdfnet,linestyle='--')
                                                                    # plt.ylim(-abs(np.array(maxdfnet_rec[:-10])), abs(np.array(maxdfnet_rec[:-10])))
                                                                    show()
                                                                    for i_lat in range(nlatcols):
                                                                        if(i_lat<nlatcols-1):
                                                                            print( '{: 3.0f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 3d} {} {: 5.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 1.0f} {: 1.0f} {: 1.0f}|'.format(latgrid[i_lat],maxdfnet_lat[i_lat],np.mean(tbound_master[:,i_lat],axis=0),np.mean(column_budgets_master[:,i_lat],axis=0),np.mean(fnet_sw_master[nlayers,:,i_lat],axis=0),np.mean(fnet_lw_master[nlayers,:,i_lat],axis=0),np.mean(merid_transps_master[:,i_lat],axis=0), np.int(cti_master[0,i_lat]), maxdfnet_ind, altz_master[np.int(cti_master[0,i_lat]),0,i_lat]/1000., dmid[i_lat], dtrop[i_lat], lapse_master[0,i_lat], np.mean(altz_master[np.int(np.mean(cti_master[:,i_lat])),:,i_lat])/1000., rad_eqb[i_lat],colbudg_eqb[i_lat],lapse_eqb[i_lat] ))
                                                                        else:
                                                                            print( '{: 3.0f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 5.3f} {: 3d} {} {: 5.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 1.0f} {: 1.0f} {: 1.0f}|'.format(latgrid[i_lat],maxdfnet_lat[i_lat],np.mean(tbound_master[:,i_lat],axis=0),np.mean(column_budgets_master[:,i_lat],axis=0),np.mean(fnet_sw_master[nlayers,:,i_lat],axis=0),np.mean(fnet_lw_master[nlayers,:,i_lat],axis=0),np.mean(merid_transps_master[:,i_lat],axis=0), np.int(cti_master[0,i_lat]), maxdfnet_ind, altz_master[np.int(cti_master[0,i_lat]),0,i_lat]/1000., dmid[i_lat], dtrop[i_lat], lapse_master[0,i_lat], np.mean(altz_master[np.int(np.mean(cti_master[:,i_lat])),:,i_lat])/1000., rad_eqb[i_lat],colbudg_eqb[i_lat],lapse_eqb[i_lat] ))
                                                                            print('-------------------------------------------------------------------')
    
                                                                if(abs(maxdfnet_tot) < eqb_maxdfnet and (ts>100 or (input_source==2 and ts > 200)) and np.max(abs(column_budgets_master))<eqb_col_budgs and np.min(lapse_eqb)==1):
                                                                # if(abs(maxdfnet_tot) < eqb_maxdfnet and (ts>100 or (input_source==2 and ts > 10)) and np.max(abs(column_budgets_master))<eqb_col_budgs):
                                                                # if(np.max(abs(column_budgets_master))<eqb_col_budgs): # NJE temp fix for perts, remember to reset!
                                                                    if(plotted!=1):
                                                                        plotrrtmoutput()
                                                                        plotted=1
                                                                    print('Equilibrium reached!')
                                                                    os.system('say "Equilibrium reached"')
                                                                    # writeoutputfile()
                                                                    # writeoutputfile_masters()
                                                                    filewritten=1
                                                                    break
                                                                elif(ts==timesteps-1):
                                                                    print('Max timesteps')
                                                                    os.system('say "Max timesteps"')
                                                                    if(plotted!=1):
                                                                        plotrrtmoutput()
                                                                        plotted=1
                                                                    # writeoutputfile()
                                                                    # writeoutputfile_masters()
                                                                    filewritten=1
    
                                                                # end timesteps loop
                                                                
                                                            writeoutputfile_masters()
    
                                                            if(plotted==0):
                                                                plotrrtmoutput()

########################################################################################end loops########################################################################################

tend = datetime.datetime.now()
ttotal = tend-tstart
print(ttotal)

print('Done')
os.system('say "Done"')
# plt.tight_layout()
# show()