# Plot RRTM Output (with new Python wrapper)
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from os import listdir
import random
from scipy import interpolate
# import pandas as pd
# from pandas import ExcelWriter
# from pandas import ExcelFile
import datetime

datetime.datetime.now()
# print(datetime.datetime.now())
# print('Started')

plot_switch=3 # 0: T(p) and dfnet(p), 1: lapse and trops, 2: CRK, 3: water vapor perts, 4: rel hum
cti_type=0 # 0: convective, 1: top down radiative, 2: cold point, 3:WMO

directories = [
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Current Output/'
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/One layer at a time/Absolute 1e-5 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/One layer at a time/Relative 0.26 %/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/Every layer simultaneously/Relative 0.26 %/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/Every layer simultaneously/Absolute 1e-5 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Absolute 1e-5 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Absolute 2.75e-4 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Relative 7 %/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Relative 10 %/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Absolute 3.86e-4 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Absolute 2.83e-4 mol per mol/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/Every layer simultaneously/Relative 0.26 %/',
'/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/One layer at a time/Absolute 1e-5 mol per mol/',
'/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o perts/equal total perts/v2, smaller/nl=590/One layer at a time/Relative 0.26 %/',
# '/Users/nickedkins/Uni GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/h2o scale heights/renormalised/v2/'
]



c_zonals=[0.0,1.0,2.0,4.0,8.0] #zonal transport coefficient
c_merids=[2.0] #meridional transport coefficient

nlayers=590
nlatcols=1
nzoncols=2

def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r/256.,g/256.,b/256.)) 
  return ret

def init_plotting():
    plt.rcParams['figure.figsize'] = (20,10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
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
    #plt.rcParams['axes.color_cycle'] = ['b', 'r', 'g','pink','orange','darkgreen','purple']

    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5

    #plt.gca().spines['right'].set_color('None')
    #plt.gca().spines['top'].set_color('None')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
init_plotting()
colors = colors(10)

def logpplot(x,p,xlab,ylab):
    plt.semilogy(x,p,'-')
    # plt.plot(x,p,'-')
    plt.ylim(max(p),min(p))
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def plotrrtmoutput():
    plt.figure(1)
    plt.subplot(221)
    plt.semilogy(tz,pz)
    plt.plot(tbound,pz[0],'o')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('tz')
    plt.ylabel('pz')
    plt.subplot(222)
    plt.semilogy(fnet,pz,c='b',label='total')
    plt.semilogy(fnet_lw,pz,c='r',label='lw')
    plt.semilogy(fnet_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('fnet')
    plt.ylabel('pz')
    plt.legend()
    plt.subplot(223)
    plt.semilogy(totuflux,pz,c='b',label='total')
    plt.semilogy(totuflux_lw,pz,c='r',label='lw')
    plt.semilogy(totuflux_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('totuflux')
    plt.ylabel('pz')
    plt.legend()
    plt.subplot(224)
    plt.semilogy(totdflux,pz,c='b',label='total')
    plt.semilogy(totdflux_lw,pz,c='r',label='lw')
    plt.semilogy(totdflux_sw,pz,c='g',label='sw')
    plt.ylim(max(pz),min(pz))
    plt.xlabel('totdflux')
    plt.ylabel('pz')
    plt.legend()
    # plt.subplot(335)
    # logpplot(tz,pz,'tz','pz')
    # plt.plot(tbound,pz[0],'o')
    # # plt.semilogy(tz,pz,'o',c='g')
    # # plt.semilogy(tavel,pavel,'o',c='b')
    # plt.ylim(max(pz),min(pz))
    # plt.subplot(336)
    # logpplot(wbrodl,pavel,'wbrodl','pavel')
    # plt.subplot(337)
    # logpplot(wkl[1,:],pavel,'wkl1 (h2o)','pavel')
    # plt.subplot(338)
    # logpplot(wkl[2,:],pavel,'wkl2 (co2)','pavel')
    # plt.subplot(339)
    # logpplot(wkl[3,:],pavel,'wkl3 (o3)','pavel')

def plotrrtmoutput_masters():
    plt.figure(1)
    for i_lat in range(0,nlatcols):
    # for i_lat in [0]:
        for i_zon in range(nzoncols):
        # for i_zon in [0]:
            
            plt.figure(1)
            plt.subplot(121)
            plt.semilogy(tz_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat],'-',label='{}'.format(fn))
            # plt.plot(tz_master[:,i_zon,i_lat],altz_master[:,i_zon,i_lat])
            # plt.semilogy(tavel_master[:,i_zon,i_lat],pavel_master[:,i_zon,i_lat],'-o',label=str(i_zon))
            if(cti_type==0):
                cti=np.int(cti_master[i_zon,i_lat])
            elif(cti_type==1):
                cti=np.int(cti_td[i_zon,i_lat])
            elif(cti_type==2):
                cti=np.int(cti_cp[i_zon,i_lat])
            elif(cti_type==3):
                cti=np.int(cti_wmo[i_zon,i_lat])
            plt.plot(tz_master[cti,i_zon,i_lat], pz_master[ cti, i_zon, i_lat ],'o' )
            # plt.plot(tz_master[np.int(cti_td[i_zon,i_lat]),i_zon,i_lat], pz_master[np.int(cti_td[i_zon,i_lat]),i_zon,i_lat], '*' )
            # plt.plot(tz_master[cti,i_zon,i_lat], altz_master[ cti, i_zon, i_lat ],'o' )
            # plt.ylim(4000,12000)
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            plt.ylim(1000,10)
            plt.xlabel('T (K)')
            plt.ylabel('Pressure (hPa)')
            plt.grid(True,which='both')
            # plt.legend()
            
            # plt.subplot(331)
            # plt.semilogy(totuflux_lw_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat],label=fn)
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('totuflux lw')
            # plt.legend()
            
            # plt.subplot(332)
            # plt.semilogy(totdflux_lw_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat])
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('totdflux lw')
        
            # plt.subplot(333)
            # plt.semilogy(totuflux_sw_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat],label=fn)
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('totuflux sw')
            # plt.legend()
            
            # plt.subplot(334)
            # plt.semilogy(totdflux_sw_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat])
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('totdflux sw')

            # plt.subplot(335)
            # plt.semilogy(fnet_lw_master[:,i_zon],pz_master[:,i_zon],label='{}'.format(fn))
            # # plt.semilogy(totdflux_master[:,i_zon,i_lat]-totuflux_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat])
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('fnet lw')
            # plt.legend()
            
            # plt.subplot(336)
            # plt.semilogy(fnet_sw_master[:,i_zon],pz_master[:,i_zon],label='{}'.format(fn))
            # # plt.semilogy(totdflux_master[:,i_zon,i_lat]-totuflux_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat])
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('fnet sw')
            # plt.legend()
            
            # plt.subplot(335)
            # plt.figure(1)
            # plt.loglog(wkl_master[:,i_zon,0,i_lat],pavel_master[:,i_zon,i_lat],label=fn)
            # plt.ylim(np.max(pz_master[:,i_zon,i_lat]),np.min(pz_master[:,i_zon,i_lat]))
            # plt.xlabel('wkl1')
            
            # plt.subplot(336)
            # plt.semilogy(wkl_master[:,i_zon,1,i_lat],pavel_master[:,i_zon,i_lat],label=str(i_file))
            # # plt.legend()
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('wkl2')
            # plt.subplot(337)
            # plt.semilogy(wkl_master[:,i_zon,2,i_lat],pavel_master[:,i_zon,i_lat])
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('wkl3')
            # plt.subplot(338)
            # plt.semilogy(wbrodl_master[:,i_zon,i_lat],pz_master[:,i_zon,i_lat],'-o')
            # plt.ylim(np.max(pz_master[:,i_zon]),np.min(pz_master[:,i_zon]))
            # plt.xlabel('wbrodl')

            plt.subplot(122)
            plt.semilogy(np.mean(dfnet_master[:,:,i_lat],axis=1),pavel_master[:,i_zon,i_lat],'-')
            plt.axvline(-eqb_maxdfnet,linestyle='--')
            plt.axvline(eqb_maxdfnet,linestyle='--')
            plt.ylim(1000,10)
            # plt.ylim(1000,600)
            plt.xlim(-5,5)
            plt.xlabel(r'$\Delta F_{net}$ in layer (Wm$^{-2}$)')
            plt.ylabel('Pressure (hPa)')
            plt.grid(True,which='both')
            # plt.legend()
            
            # plt.subplot(133)
            # lapsedum=np.zeros(nlayers)
            # tzdum=tz_master[:,i_zon,i_lat]
            # altzdum=altz_master[:,i_zon,i_lat]
            # for i in range(nlayers):
            #     lapsedum[i]=(tzdum[i+1]-tzdum[i]) / (altzdum[i+1]-altzdum[i])*-1000.
            # plt.semilogy(lapsedum[1:],pavel_master[1:,i_zon,i_lat],'-o')
            # plt.xlabel('Lapse Rate (K/km)')
            # plt.ylabel('Pressure (hPa)')
            # plt.grid(True,which='both')
            # plt.ylim(1000,10)
            # plt.xlim(-10,10)


def readrrtmoutput(fn):
    f = open(fn)


ndirs=len(directories)
a = sorted(listdir(directories[0]))
if('.DS_Store' in a):
    a.remove('.DS_Store')
nfiles=len(a)

nmol=7
nclouds=nlayers

# nzoncols=1

# # latgridbounds=[-90,-66.5,-23.5,23.5,66.5,90] # 5 box poles, subtropics, tropics
latgridbounds=np.linspace(-60,60.,nlatcols+1)
xgridbounds=np.sin(np.deg2rad(latgridbounds))
# # xgridbounds=np.linspace(-1.,1.,nlatcols+1)
# # latgridbounds=np.rad2deg(np.arcsin(xgridbounds))

latgrid=np.zeros(nlatcols)
for i in range(nlatcols):
    latgrid[i]=(latgridbounds[i]+latgridbounds[i+1])/2.


# latweights=np.zeros(nlatcols)
# # latweights=np.cos(np.deg2rad(latgrid))

# for i_xg in range(len(xgridbounds)-1):
#     latweights[i_xg]=xgridbounds[i_xg+1]-xgridbounds[i_xg]

# latweights/=np.mean(latweights)

# if(nlatcols==1):
#     latgrid=np.array([65.])
#     latgridbounds=[latgrid[0]-5.,latgrid[0]+5.]

# latgrid = np.linspace(-90,90,nlatcols)

# totuflux_lw_master=np.zeros((nlayers+1,nfiles,ndirs))
# totuflux_sw_master=np.zeros((nlayers+1,nfiles,ndirs))
# pz_master=np.zeros((nlayers+1,nfiles,ndirs))
# cld_lay_master=np.zeros((1,nfiles,ndirs))


gravity=9.81
avogadro=6.022e23
iatm=0 #0 for layer values, 1 for level values
ixsect=0 #could be 1, but why?
iscat=0 #just absorption and emission
numangs=0 #can be 0-4 for higher precision
#iout=99 #for info in all spectral bands
iout=0 #for broadband only
#iout=-1 #for broadband, no printings
icld=0 #for clear sky
#icld=1  #for grey clouds
tbound=288 #surface temperature (K)
# tbound=np.zeros(1)
iemiss=1 #surface emissivity. Keep this fixed for now.
iemis=2
ireflect=0 #for Lambert reflection
iaer=0 #0=aerosols off, 1=on
istrm=1             # ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0). 
                        #0=4 streams
                        #1=8 streams
idelm=1             # flag for outputting downwelling fluxes computed using the delta-M scaling approximation. 0=output "true" direct and diffuse downwelling fluxes, 1=output direct and diffuse downwelling fluxes computed with delta-M approximation
icos=0              #0:there is no need to account for instrumental cosine response, 1:to account for instrumental cosine response in the computation of the direct and diffuse fluxes, 2:2 to account for instrumental cosine response in the computation of the diffuse fluxes only
semis=np.ones(16)   #all spectral bands the same as iemissm
semiss=np.ones(29)  #all spectral bands the same as iemissm (surface, I think)
iform=1
psurf=1000.
pmin=0.
secntk=0
cinp=1.356316e-19
ipthak=3
ipthrk=3
juldat=0        #Julian day associated with calculation (1-365/366 starting January 1). Used to calculate Earth distance from sun. A value of 0 (default) indicates no scaling of solar source function using earth-sun distance.
sza=65.             #Solar zenith angle in degrees (0 deg is overhead).
isolvar=0       #= 0 each band uses standard solar source function, corresponding to present day conditions. 
                #= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
                #= 2 0.1,0.1scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))          
lapse=5.7
tmin=150.
tmax=350.
rsp=287.05
gravity=9.81
pin2 = 1.0 * 1e5 #convert the input in bar to Pa
pico2 = 400e-6 * 1e5 #convert the input in bar to Pa
pio2 = 0.0 * 1e5
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
surf_rh=0.8
vol_mixh2o_min = 1e-6
vol_mixh2o_max = 1e6
ur_min=0.5
ur_max=1.0
eqb_maxhtr = 0.001
timesteps=800
cti=0
maxhtr=0.
cld_lay=0.

master_input=0
conv_on=0
surf_lowlev_coupled=0
lay_intp=0
lw_on=0
sw_on=0
eqb_maxdfnet=0
toa_fnet_eqb=0


# pz=np.linspace(psurf,pmin,nlayers+1)
# totuflux=np.zeros(nlayers+1)
# totdflux=np.zeros(nlayers+1)
# fnet=np.zeros(nlayers+1)
# htr=np.zeros(nlayers+1)
# pavel=np.zeros(nlayers)
# tz=np.ones(nlayers+1) * tbound
# altz=np.zeros(nlayers+1)
# tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
# tavel=np.zeros(nlayers)
# esat_liq=np.zeros(nlayers)
# rel_hum=np.zeros(nlayers)
# mperlayr = np.zeros(nlayers)
# mperlayr_air = np.zeros(nlayers)
# wbrodl = np.zeros(nlayers)
# wkl = np.zeros((nmol+1,nlayers))
# totuflux_lw=np.zeros(nlayers+1)
# totdflux_lw=np.zeros(nlayers+1)
# fnet_lw=np.zeros(nlayers+1)
# htr_lw=np.zeros(nlayers+1)
# totuflux_sw=np.zeros(nlayers+1)
# totdflux_sw=np.zeros(nlayers+1)
# fnet_sw=np.zeros(nlayers+1)
# htr_sw=np.zeros(nlayers+1)
# conv=np.zeros(nlayers+1)
# altavel = np.zeros(nlayers)
# vol_mixh2o = np.ones(nlayers) * molec_h2o / totmolec
# vol_mixo3 = np.ones(nlayers) * molec_o3 / totmolec
# solvar=np.zeros(29)

vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,nzoncols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb,nlatcols]

# nlayers, nzoncols, nfiles, ndirs

nlayers_dirfil=nlayers
nzoncols_dirfil=nzoncols
# nfiles=25
# ndirs=2

tbound_all_dirfil = np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
tz_all_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
totuflux_all_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
totuflux_all_dirfil_lw=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
fnet_all_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
zonal_transps_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
merid_transps_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
fnet_sw_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
fnet_lw_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
lapse_master_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
altz_all_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
pz_all_dirfil=np.zeros((nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))
cti_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
cti_td_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
cti_cp_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
cti_wmo_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
lapse_td_all_dirfil=np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
wkl_all_dirfil=np.zeros((8,nlayers_dirfil+1,nzoncols_dirfil,nfiles,ndirs,nlatcols))

ttrops_dirfil = np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
ptrops_dirfil = np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))
ztrops_dirfil = np.zeros((nzoncols_dirfil,nfiles,ndirs,nlatcols))

pertzons=[0]
pertlats=[0,1,2,3,4]
pertmols=[7] #don't do zero!
pertlays=[0,6,12,18,24,30,36,42,48,54]
perts=[1]

totuflux_all_prp=np.zeros((len(perts), len(pertmols), len(pertlats),len(pertzons), len(pertlays) ))
tbound_all_prp=np.zeros((len(perts), len(pertmols), len(pertlats),len(pertzons), len(pertlays) ))

dir_labels = []
dir_labels = ['absolute, $10^{-5}$ mol/mol', 'relative, 0.26 %']

i_dir=0
for directory in directories:

    filenames = []
    dir_label = directory.split('/')[-2]
    # dir_labels.append(dir_label)
    # print
    # print(dir_label)
    # print
    a = sorted(listdir(directory))
    filenames.append(a)
    if('.DS_Store' in a):
        a.remove('.DS_Store')

    i_file=0
    for fn in a:
        # if(fn=='2020_09_17 19_05_52'):
        #     continue
            
        # print(fn)
        output_file = directory + fn
        f=open(output_file,'r')


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
        ipthak  =   float   (   f.readline().rstrip('\n')   )
        ipthrk  =   float   (   f.readline().rstrip('\n')   )
        juldat  =   float   (   f.readline().rstrip('\n')   )
        sza =   float   (   f.readline().rstrip('\n')   )
        isolvar =   int (   f.readline().rstrip('\n')   )
        lapse   =   float   (   f.readline().rstrip('\n')   )
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
        timesteps   =   int (   f.readline().rstrip('\n')   )
        cti =   int (   f.readline().rstrip('\n')   )
        maxhtr  =   float   (   f.readline().rstrip('\n')   )
        cld_lay =   float   (   f.readline().rstrip('\n')   )
        nzoncols  =   int (   f.readline().rstrip('\n')   )
        master_input =  float   (   f.readline().rstrip('\n')   )
        conv_on=float   (   f.readline().rstrip('\n')   )
        surf_lowlev_coupled=float   (   f.readline().rstrip('\n')   )
        lay_intp=float  (   f.readline().rstrip('\n')   )
        lw_on=int   (   f.readline().rstrip('\n')   )
        sw_on=int   (   f.readline().rstrip('\n')   )
        eqb_maxdfnet=float  (   f.readline().rstrip('\n')   )
        toa_fnet_eqb=float  (   f.readline().rstrip('\n')   )
        nlatcols=int    (   f.readline().rstrip('\n')   )
        # nlatcols=9
        # f.readline()


        pz=np.linspace(psurf,pmin,nlayers+1)
        totuflux=np.zeros(nlayers+1)
        totdflux=np.zeros(nlayers+1)
        fnet=np.zeros(nlayers+1)
        htr=np.zeros(nlayers+1)
        pavel=np.zeros(nlayers)
        tz=np.ones(nlayers+1) * tbound
        altz=np.zeros(nlayers+1)
        altavel=np.zeros(nlayers)
        tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
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
        tauclds_master=np.ones((nzoncols,nlatcols,nclouds))*0.01
        ssaclds_master=np.ones((nzoncols,nlatcols,nclouds))*1.0
        # ssaclds_master[1,:,:]=np.ones((nlatcols,nclouds))*0.01

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

        dfnet_master=np.zeros((nlayers,nzoncols,nlatcols))
        dfnet_master_rad=np.zeros((nlayers,nzoncols,nlatcols))
        dfnet_master_adv=np.zeros((nlayers,nzoncols,nlatcols))

        dpz_master=np.zeros((nlayers,nzoncols,nlatcols))

        wkl_master=np.zeros((nlayers,nzoncols,nmol+1,nlatcols))
        
        cti_master=np.zeros((nzoncols,nlatcols))
        lapse_master=np.zeros((nzoncols,nlatcols))

        vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,nzoncols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb,nlatcols]
        vars_master_lay_zon_lat=[tavel_master,pavel_master,altavel_master,wbrodl_master,dfnet_master,dfnet_master_rad,dfnet_master_adv,dpz_master]
        vars_master_lev_zon_lat=[tz_master,pz_master,altz_master,totuflux_master,totuflux_lw_master,totuflux_sw_master,totdflux_master,totdflux_lw_master,totdflux_sw_master,fnet_master,fnet_lw_master,fnet_sw_master,htr_master,htr_lw_master,htr_sw_master,conv_master]
        vars_misc_1d=[semis,semiss,solvar]
        vars_misc_1d_lens=[16,29,29]
        vars_master_lay_zon_nmol_lat=[wkl_master]
        vars_master_zon_lat=[inflags_master,iceflags_master,liqflags_master,tbound_master,toa_fnet_master,zonal_transps_master,merid_transps_master,cti_master,lapse_master]
        vars_master_zon_lat_cld=[cld_lays_master,cld_fracs_master,tauclds_master,ssaclds_master]
        vars_master_lat=[latgrid]


        
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


        for x in vars_master_zon_lat:
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


        cti_td=np.zeros((nzoncols,nlatcols))
        for i_zon in range(nzoncols):
            for i_lat in range(nlatcols):
                for i in range(nlayers-1,-1,-1):
                    if( abs( np.mean(dfnet_master[i,:,i_lat]) ) < eqb_maxdfnet*2.0):
                        cti_td[i_zon,i_lat]=i
                    else:
                        break
        
        cti_cp=np.zeros((nzoncols,nlatcols))
        for i_zon in range(nzoncols):
            for i_lat in range(nlatcols):
                cti_cp[i_zon,i_lat]=np.argmin(tz_master[:,i_zon,i_lat])

        cti_wmo=np.zeros((nzoncols,nlatcols))
        lapse_loc=np.zeros(nlayers)
        for i_zon in range(nzoncols):
            for i_lat in range(nlatcols):
                for i in range(nlayers):
                    dT_loc=tz_master[i+1,i_zon,i_lat]-tz_master[i,i_zon,i_lat]
                    dz_loc=(altz_master[i+1,i_zon,i_lat]-altz_master[i,i_zon,i_lat])/1000.
                    lapse_loc[i]=-dT_loc/dz_loc
                    if(altz_master[i,i_zon,i_lat]/1000.>2. and lapse_loc[i]<2.):
                        cti_wmo[i_zon,i_lat]=i
                        break
                # cti_wmo[i_zon,i_lat]=np.argmin(tz_master[:,i_zon,i_lat])
                

        if(plot_switch==4):
            
            # perts = perts=[0, 1e-6, 1e-5, 1e-4, 1e-3, 4e-4, 3e-4, 2.5e-4, 2.74e-4]
        
            tc = tavel_master - 273.15
            
            print(tavel_master[0])
    
            ps_wv = (0.61078 * np.exp( 17.27 * tc / ( tc + 237.3 ) )) * 10. # tetens in hPa
            ps_ice = (0.61078 * np.exp( 21.875 * tc / ( tc + 265.5 ) )) * 10.
            
            q = wkl_master[:,:,0,:] * 0.6
            
            rh_wv = q * pavel_master / (ps_wv * (q + 0.622)) * 100.
            rh_ice = q * pavel_master / (ps_ice * (q + 0.622)) * 100.
            
            argefold = np.argmin(abs(wkl_master[0,0,0,0]/wkl_master[:,0,0,0]-2.718))
            Hh2o = altz_master[argefold,0,0]/1000.
            
            print(Hh2o)
    
            # print('Total H2O molecs: {: 8.6e}'.format( np.sum( wkl_master[:,0,0,0] )/ 2.307722e-1 ) )
            # plt.plot(perts[i_file], np.sum( wkl_master[:,0,0,0] ) / 2.307722e-1, 'o' )
            # plt.axhline(1.07)
            # plt.xlim(0.0002, 0.00041)
            # plt.ylim(1.05, 1.10)
    
            # perts = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            plt.figure(i_dir)
            # plt.title('RH profiles with relative increases in \n specific humidity in 50 hPa blocks', y=1.02)
            # plt.title('RH profiles with a 7% relative increase \n in specific humidity in all layers', y=1.02)            
            plt.title(dir_label, y=1.02)            
            # plt.semilogy(rh_wv[:,0,0],pavel_master[:,0,0],'-',label='RH water vapour')
            # plt.semilogy(rh_ice[:,0,0],pavel_master[:,0,0],'-',label='RH ice')
            # rh = np.where(tc[:,0,0] < -20., 0, 50)
            # plt.semilogy(rh,pavel_master[:,0,0],'--',label='RH max')
            rh = np.where(tc[:,0,0] > -15., rh_wv[:,0,0], rh_ice[:,0,0])
            plt.semilogy(rh,pavel_master[:,0,0],'-')
            plt.xlabel('RH (%)')
            plt.ylabel('Pressure (hPa)')
            plt.axvline(100.,linestyle='--')
            plt.ylim(1000,10)
            # plt.xlim(0,200)
            # plt.xlim(0,100)
            # plt.legend()

        # print('Total H2O molecs: {: 6.4f}'.format( np.sum( wkl_master[:,0,0,0] / 2.32879222408797e+00 ) ) )
        # print('Total H2O molecs: {: 6.4f}'.format( np.sum( wkl_master[:,0,0,0] * mperlayr ) ) )
        # print('Total H2O molecs: {: 16.14e}'.format( np.sum( wkl_master[:,0,0,0] ) ) )


        # latgridbounds=np.linspace(30,60.,nlatcols+1)
        # xgridbounds=np.sin(np.deg2rad(latgridbounds))    
        # latgrid=np.zeros(nlatcols)
        # for i in range(nlatcols):
        #     latgrid[i]=(latgridbounds[i]+latgridbounds[i+1])/2.

        # dfnet_master=np.zeros((nlayers,nzoncols,nlatcols))

        # for i_zon in range(nzoncols):
        #     for j in range(nlatcols):
        #         for i in range(nlayers):
        #             dfnet_master[i,i_zon,j]=fnet_master[i+1,i_zon,j]-fnet_master[i,i_zon,j]



        #print output for easy spreadsheet transfer
        # for i in range(nlayers):
        #   print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(pz[i],pavel[i],altz[i]/1000.,tz[i],tavel[i],totuflux[i],totuflux_lw[i],totuflux_sw[i],totdflux[i],totdflux_lw[i],totdflux_sw[i],fnet[i],fnet_lw[i],fnet_sw[i]))
        # print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(pz[nlayers],'na',altz[nlayers]/1000.,tz[nlayers],'na',totuflux[nlayers],totuflux_lw[nlayers],totuflux_sw[nlayers],totdflux[nlayers],totdflux_lw[nlayers],totdflux_sw[nlayers],fnet[nlayers],fnet_lw[nlayers],fnet_sw[nlayers],tbound))

        # totuflux_lw_master[:,i_file,i_dir]=totuflux_lw
        # totuflux_sw_master[:,i_file,i_dir]=totuflux_sw
        # pz_master[:,i_file,i_dir]=pz
        # cld_lay_master[:,i_file,i_dir]=cld_lay

        # Pierrehumbert radiator fin stuff
        # Q=np.zeros(nzoncols)
        # Qv=np.zeros(nzoncols)
        # esat_lowlay=np.zeros(nzoncols)
        # esat_bound=np.zeros(nzoncols)
        # qsat_lowlay=np.zeros(nzoncols)
        # qsat_bound=np.zeros(nzoncols)
        # qstar=np.zeros(nzoncols)
        # Evap=np.zeros(nzoncols)
        # Fah=np.zeros(nzoncols)
        # col_budg=np.zeros(nzoncols)
        # deltheta=85.
        # cp=1.003
        # L=2429.8
        # r=0.8
        # A1=1.
        # A2=1.
        # for k in range(2):
        #   for i_zon in range(nzoncols):
        #       Q[i_zon]=(fnet_master[nlayers,i_zon]-fnet_master[0,i_zon])
        #       esat_lowlay[i_zon] = 6.1094*exp(17.625*(tz_master[0,i_zon]-273.15)/(tz_master[0,i_zon]-273.15+243.04))
        #       esat_bound[i_zon] = 6.1094*exp(17.625*(tbound_master[i_zon]-273.15)/(tbound_master[i_zon]-273.15+243.04))
        #       qsat_lowlay[i_zon]=0.622*esat_lowlay[i_zon]/(pz_master[0,i_zon]-esat_lowlay[i_zon])
        #       qsat_bound[i_zon]=0.622*esat_bound[i_zon]/(pz_master[0,i_zon]-esat_bound[i_zon])
        #       qstar[i_zon]=qsat_bound[i_zon]-r*qsat_lowlay[i_zon]
        #       b=L*qstar[1]/(cp*deltheta)
        #       Qv[i_zon]=(1-b)*Q[i_zon]
        #       Evap[i_zon]=-b*Q[i_zon]
        # Fah[0]=A2/A1*Qv[1]
        # Fah[1]=-1.*Fah[0]
        # col_budg[0]=fnet_master[nlayers,0]+Fah[0]+Evap[1]
        # col_budg[1]=fnet_master[nlayers,1]+Fah[1]-Evap[1]
        # for i_zon in range(nzoncols):
        #   # print 'i_zon: {:d} Q: {:4.0f} b: {:4.2f} Qv: {:4.0f} E: {:4.0f} tbound: {:4.0f} Fah1: {:4.0f} fnet at toa: {:4.0f} col budg 1 {:4.0f} col budg 2 {:4.0f} sum col budg {:4.0f} fdown {:4.0f} olr {:4.0f}'.format(i_zon,Q[i_zon],b,Qv[i_zon],Evap[i_zon],tbound_master[i_zon],Fah[0],fnet_master[nlayers,i_zon],col_budg[0],col_budg[1],np.sum(col_budg),totdflux_sw_master[nlayers,i_zon],totuflux_lw_master[nlayers,i_zon])
        #   print 'fdown {:4.0f} olr {:4.0f} fah {:4.0f} evap {:4.0f} tmean {:6.1f}'.format(totdflux_master[nlayers,i_zon],totuflux_master[nlayers,i_zon],Fah[i_zon],Evap[i_zon],np.mean(tbound_master))
        # plt.figure(1)
        # plt.subplot(121+i_dir)
        # plt.plot(wkl_master[0,1,0],tbound_master[0],'o',c='r',label='Twarm')          
        # plt.plot(wkl_master[0,1,0],tbound_master[1],'o',c='b',label='Tcold')
        # plt.plot(wkl_master[0,1,0],np.mean(tbound_master),'o',c='g',label='Tavg')
        # plt.xlabel('wklfac')
        # plt.ylabel('Temperature (K)')
        # plt.ylim(295,307)

        if(fn=='_baseline'):
            baseline_olr=np.mean(totuflux_master[nlayers,0,:])
            # continue

        if(fn[0]=='0'):
            continue

        # if(i_file==len(a)):
        #     plt.legend()
        if(plot_switch==0):
            plotrrtmoutput_masters()
        # plt.plot(latgrid,lapse_master[0,:],'-o')
        # # plt.xlim(-60,60)
        # data=np.genfromtxt('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/Latitudinal Distributions/Doug Mason Lapse Rate vs Latitude.txt',delimiter=',')
        # plt.plot(data[:,0],data[:,1]*-1.)        
        # plt.xlabel('Latitude')
        # plt.ylabel('Lapse rate (K/km)')
        
        
        if(plot_switch==1):
            plt.figure(1)
            if(cti_type==0):
                cti=cti_master[0,:].astype('int')
            elif(cti_type==1):
                cti=cti_td[0,:].astype('int')
            elif(cti_type==2):
                cti=cti_cp[0,:].astype('int')
            elif(cti_type==3):
                cti=cti_wmo[0,:].astype('int')
            # cti_td=np.ones((nzoncols,nlatcols))*(np.int(nlayers*0.8))
            # cti_td=cti_td.astype('int')
            ztrop=np.diagonal(altz_master[cti,0,:]/1000.)
            ptrop=np.diagonal(pz_master[cti,0,:])
            ttrop=np.diagonal(tz_master[cti,0,:])


            # ztrop=np.diagonal(altz_master[cti_td,0,:]/1000.)
            # ptrop=np.diagonal(pz_master[cti_td,0,:])
            # ttrop=np.diagonal(tz_master[cti_td,0,:])
            tsurf=tz_master[0,0,:]
            
            plt.subplot(222)
            if(fn=='SC79'):
                plt.plot(np.delete(latgrid[:],1),np.delete(ztrop,1),'-o',label=fn)
            else:
                plt.plot(latgrid[:],ztrop,'-o',label=fn)
            plt.xlabel('Latitude')
            plt.ylabel('Tropopause altitude (km)')
            plt.legend()
                
            plt.subplot(223)
            if(fn=='SC79'):
                plt.plot(np.delete(latgrid[:],1),np.delete(ptrop,1),'-o',label=fn)
            else:
                plt.plot(latgrid[:],ptrop,'-o',label=fn)
            plt.xlabel('Latitude')
            plt.ylabel('Tropopause pressure (hPa)')
            plt.legend()
            
            plt.subplot(224)
            if(fn=='SC79'):
                plt.plot(np.delete(latgrid[:],1),np.delete(ttrop,1),'-o',label=fn)
            else:
                plt.plot(latgrid[:],ttrop,'-o',label=fn)
            plt.xlabel('Latitude')
            plt.ylabel('Tropopause temperature (K)')
            plt.legend()
           
            
        if(nlatcols>1):
            ztrop=np.diagonal(altz_master[cti,0,:]/1000.)
            ptrop=np.diagonal(pz_master[cti,0,:])
            ttrop=np.diagonal(tz_master[cti,0,:])
        else:
            ztrop=altz_master[cti,0,0]/1000.
            ptrop=pz_master[cti,0,0]
            ttrop=tz_master[cti,0,0]
        tsurf=tz_master[0,0,0]
        
        
        
        # print('{} ztrop: {: 4.2f} ptrop: {: 4.2f} ttrop: {: 4.2f} tsurf: {: 4.2f} '.format(fn, ztrop, ptrop, ttrop, tsurf))
        print('{: 6.4f}, {: 6.4f}, {: 6.4f}, {: 6.4f} '.format(ztrop, ptrop, ttrop, tsurf))
         
        # b_rdwvs=np.arange(1,8)
        
        # print('{: 4.2f}'.format(Hh2o))
        # plt.xlabel('Water vapour scale height (km)')
        # plt.ylabel('Surface temperature (K)')
        # plt.figure(1)
        # plt.subplot(221)
        # plt.plot(Hh2o,tsurf,'o',c='b')
        # plt.xlabel('H$_2$O scale height (km)')
        # plt.ylabel('$T_{surf}$')
        # plt.subplot(222)
        # plt.plot(Hh2o,ttrop,'o',c='b')
        # plt.xlabel('H$_2$O scale height (km)')
        # plt.ylabel('$T_{trop}$')
        # plt.subplot(223)
        # plt.plot(Hh2o,ztrop,'o',c='b')
        # plt.xlabel('H$_2$O scale height (km)')
        # plt.ylabel('$z_{trop}$')
        # plt.subplot(224)
        # plt.plot(Hh2o,ptrop,'o',c='b')
        # plt.xlabel('H$_2$O scale height (km)')
        # plt.ylabel('$p_{trop}$')
        

            
        
        tbound_all_dirfil[:,i_file,i_dir,:]=tbound_master
        tz_all_dirfil[:,:,i_file,i_dir]=tz_master
        zonal_transps_all_dirfil[:,i_file,i_dir,:]=zonal_transps_master
        merid_transps_all_dirfil[:,i_file,i_dir,:]=merid_transps_master
        totuflux_all_dirfil[:,:,i_file,i_dir,:]=totuflux_master
        totuflux_all_dirfil_lw[:,:,i_file,i_dir,:]=totuflux_lw_master
        fnet_all_dirfil[:,:,i_file,i_dir,:]=fnet_master
        fnet_sw_dirfil[:,:,i_file,i_dir,:]=fnet_sw_master
        fnet_lw_dirfil[:,:,i_file,i_dir,:]=fnet_lw_master
        lapse_master_dirfil[:,i_file,i_dir,:]=lapse_master
        altz_all_dirfil[:,:,i_file,i_dir,:]=altz_master
        pz_all_dirfil[:,:,i_file,i_dir,:]=pz_master
        cti_all_dirfil[:,i_file,i_dir,:]=cti_master
        cti_td_all_dirfil[:,i_file,i_dir,:]=cti_td
        cti_cp_all_dirfil[:,i_file,i_dir,:]=cti_cp
        cti_wmo_all_dirfil[:,i_file,i_dir,:]=cti_wmo

        ttrops_dirfil[0,i_file,i_dir,:] = ttrop
        ptrops_dirfil[0,i_file,i_dir,:] = ptrop
        ztrops_dirfil[0,i_file,i_dir,:] = ztrop


        # lapse_td_all_dirfil[:,i_file,i_dir,:]=cti_td
        
        
        i_file+=1


        
        # dTs=tbound_all_dirfil[:,1,i_dir,:]-tbound_all_dirfil[:,0,i_dir,:]

        # if(i_file%2==0):
        #     # plt.plot(latgrid,dTs[0,:],'-o',label=dir_label+' cloudy')
        #     # plt.plot(latgrid,dTs[1,:],'--o',label=dir_label+' clear')
        #     # plt.plot(latgrid,dTs[1,:]-dTs[0,:],'--o',label=dir_label+' clear')
        #     plt.plot(latgrid,np.mean(dTs[:,:],axis=0),'-o',label=dir_label)            
        #     plt.xlabel('Latitude')
        #     plt.ylabel('dT for 2xCO2')
        #     # plt.ylim(0)
            
        #     print(dir_label,np.mean(dTs))
        
        # plt.legend()
        
    if(plot_switch==3):
        pert_pwidth = 50.
        pert_pbottoms = np.arange(1000+pert_pwidth,0,-pert_pwidth*2.)
        
        plt.figure(1)

        plt.subplot(121)
        # plt.title('Total increase in column vapour of 0.26 %')
        # plt.title('Relative perturbation of 7%')
        plt.plot((tz_all_dirfil[0,0,1:,i_dir,0]-tz_all_dirfil[0,0,0,i_dir,0])*1e3,pert_pbottoms[1:]-pert_pwidth/2,'-o',label=dir_labels[i_dir])
        plt.ylim(1000,-50)
        plt.xlim(0)
        plt.xlabel('temperature change (mK)')
        plt.ylabel('pressure at centre of {: 2.0f} hPa perturbed region (hPa)'.format(pert_pwidth))
        if(i_dir==1):
            plt.legend(loc='upper center')

        plt.subplot(122)
        # plt.title('Zoomed in')
        # plt.title('Relative perturbation of 7%')
        plt.plot((tz_all_dirfil[0,0,1:,i_dir,0]-tz_all_dirfil[0,0,0,i_dir,0])*1e3,pert_pbottoms[1:]-pert_pwidth/2,'-o',label=dir_labels[i_dir])
        plt.ylim(1000,-50)
        plt.xlim(0,1.5)
        plt.xlabel('temperature change (mK)')
        plt.ylabel('pressure at centre of {: 2.0f} hPa perturbed region (hPa)'.format(pert_pwidth))
        if(i_dir==1):
            plt.legend(loc='upper center')
        
        print('Total T change: {: 6.4f} K'.format(np.sum(tz_all_dirfil[0,0,1:,i_dir,0]-tz_all_dirfil[0,0,0,i_dir,0])))

        print(wkl_all_dirfil[1,1,0,:,i_dir,0])

        # print('Total H2O molecs change: {: 6.4e}'.format(np.sum(wkl_all_dirfil[1,:,0,1:,i_dir,0]-wkl_all_dirfil[1,:,0,0,i_dir,0])))
        
        plt.figure(1)
        
        
    i_dir+=1    


########################################################################## end read files #################################################################################################################

print( 'dtsurf: {: 6.2f} | dttrop: {: 6.2f} | dptrop: {: 6.2f} | dztrop: {: 6.2f} '.format( tbound_all_dirfil[ 0, 1, 0, 0 ] - tbound_all_dirfil[ 0, 0, 0, 0 ] , ttrops_dirfil[ 0, 1, 0, 0 ] - ttrops_dirfil[ 0, 0, 0, 0 ], ptrops_dirfil[ 0, 1, 0, 0 ] - ptrops_dirfil[ 0, 0, 0, 0 ], ztrops_dirfil[ 0, 1, 0, 0 ] - ztrops_dirfil[ 0, 0, 0, 0 ] ) )
print( 'dtsurf: {: 6.2f} | dttrop: {: 6.2f} | dptrop: {: 6.2f} | dztrop: {: 6.2f} '.format( tbound_all_dirfil[ 0, 2, 0, 0 ] - tbound_all_dirfil[ 0, 0, 0, 0 ] ,ttrops_dirfil[ 0, 2, 0, 0 ] - ttrops_dirfil[ 0, 0, 0, 0 ], ptrops_dirfil[ 0, 2, 0, 0 ] - ptrops_dirfil[ 0, 0, 0, 0 ], ztrops_dirfil[ 0, 2, 0, 0 ] - ztrops_dirfil[ 0, 0, 0, 0 ] ) )

# plt.subplot(131)
# plt.imshow(crklw[0,:,::-1].T,vmin=-2.5,vmax=2.5,cmap='bwr')
# plt.colorbar()

# cf_tots = [0.5, 0.6]
# cldlats = np.arange(nlatcols)
# pclddums = np.linspace(500,900,10)

# dTs = np.zeros( ( len(cldlats), len(pclddums) ) )

# i=0
# for icl in range(len(cldlats)):
#     for ipc in range(len(pclddums)):
#         dTs[icl, ipc] = tbound_all_dirfil[0,i+1,0,icl] - tbound_all_dirfil[0,i,0,icl]
#         i+=2

# # plt.imshow(dTs)
# # plt.colorbar()

# plt.semilogy(dTs[0,:],pclddums,'-o')
# plt.ylim(900,500)

# xx,yy=np.meshgrid(latgrid,pclddums)
# zz=dTs.T

# vmax=np.abs(np.amax(dTs))
# vmin=-1.*vmax

# plt.figure(1)
# plt.contourf(xx,yy,zz,vmax=vmax,vmin=vmin,cmap='bwr',levels=100)
# plt.xlabel('Latitude')
# plt.ylabel('Pressure')
# plt.ylim(900,500)
# plt.colorbar()


    


# for i_dir in range(len(directories)):
#     for i_file in range(len(a)):
#         for i_zon in range(nzoncols):
#             for i_lat in range(nlatcols):
#                 if(cti_type==1):
#                     cti=np.int(cti_td_all_dirfil[i_zon,i_file,i_dir,i_lat])
#                 elif(cti_type==2):
#                     cti=np.int(cti_cp_all_dirfil[i_zon,i_file,i_dir,i_lat])
#                 elif(cti_type==3):
#                     cti=np.int(cti_wmo_all_dirfil[i_zon,i_file,i_dir,i_lat])
#                 dT=tz_all_dirfil[cti,i_zon,i_file,i_dir,i_lat] - tz_all_dirfil[0,i_zon,i_file,i_dir,i_lat]
#                 dz=(altz_all_dirfil[cti,i_zon,i_file,i_dir,i_lat] - altz_all_dirfil[0,i_zon,i_file,i_dir,i_lat])/1000.
#                 # print(altz_all_dirfil[100,i_zon,i_file,0,i_lat]/1000.,  altz_all_dirfil[0,i_zon,i_file,0,i_lat]/1000.)
#                 lapse_td_all_dirfil[i_zon,i_file,i_dir,i_lat]=-dT/dz
#                 ttrop=tz_all_dirfil[cti,i_zon,i_file,i_dir,i_lat]
#                 ztrop=altz_all_dirfil[cti,i_zon,i_file,i_dir,i_lat]/1000.
#                 ts_eff=ttrop+(-dT/dz)*ztrop
#                 # plt.figure(2)
#                 # plt.plot(latgrid[i_lat],ts_eff,'o')
#                 # print(latgrid[i_Lat],lapse,)
    
# print(fnet_lw_dirfil[-1,0,:,0,0])            
    
# plt.plot(fnet_lw_dirfil[-1,0,:,0,0],'-o')

    

if(plot_switch==1):
    plt.figure(1)
    
    plt.subplot(221)
    data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[lapse]_HV19.txt',delimiter=',')
    lat_obs=data[:,0]
    lapse_obs=data[:,1]
    interplat=np.linspace(-90,90,60)
    intp_lapse=np.interp(interplat,lat_obs,lapse_obs)
    plt.plot(lat_obs,lapse_obs,'--',c='b',label='HV19 from NCEP2')
    plt.xlim(70,90)
    # plt.ylim(4,7)
    plt.legend()
    
    plt.subplot(222)
    data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[ztrop]_HV19.txt',delimiter=',')
    lat_obs=data[:,0]
    ztrop_obs=data[:,1]
    intp_ztrop=np.interp(interplat,lat_obs,ztrop_obs)
    plt.plot(lat_obs,ztrop_obs,'--',c='b',label='HV19 from NCEP2')
    plt.xlim(70,90)
    # plt.ylim(7,10.5)
    plt.legend()
    
    plt.subplot(223)
    data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[ptrop]_Hoi98.txt',delimiter=',')
    lat_obs=data[:,0]
    ptrop_obs=data[:,1]
    plt.plot(lat_obs,ptrop_obs,'--',c='b',label='Hoi98 from ECMWF')
    plt.xlim(70,90)
    # plt.ylim(250,350)
    plt.legend()
    
    plt.subplot(224)
    data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[ttrop]_HV19.txt',delimiter=',')
    lat_obs=data[:,0]
    ttrop_obs=data[:,1]
    intp_ttrop=np.interp(interplat,lat_obs,ttrop_obs)
    plt.plot(lat_obs,ttrop_obs,'--',c='b',label='HV19 from NCEP2')
    plt.xlim(70,90)
    # plt.ylim(190,260)
    plt.legend()
    
    for i_file in range(len(a)):
        plt.subplot(221)
        # plt.plot(latgrid,lapse_master_dirfil[0,i_file,0,:],'-o',label=a[i_file])
        if(a[i_file]=='SC79'):
            plt.plot(np.delete(latgrid,1),np.delete(lapse_td_all_dirfil[0,i_file,0,:],1),'-o',label=a[i_file])
        else:
            plt.plot(latgrid,lapse_td_all_dirfil[0,i_file,0,:],'-o',label=a[i_file])
        # plt.xlim(60,90)
        plt.xlabel('Latitude')
        plt.ylabel('Lapse rate (K/km)')
    plt.legend()
    
    # for i_file in range(len(a)):    
    #     plt.subplot(222)
    #     plt.plot(latgrid,altz_master[cti_master[0,:].astype('int'),0,:]/1000.,'-o',label=a[i_file])
    #     plt.xlabel('Latitude')
    #     plt.ylabel('Tropopause height (km)')
    # plt.legend()
   
    

# plt.figure(1)
# data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[insol].txt',delimiter=',')
# lat_obs=data[:,0]
# insol_obs=data[:,1]
# plt.subplot(221)
# plt.plot(lat_obs,insol_obs,'--',c='b')

# data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[olr].txt',delimiter=',')
# lat_obs=data[:,0]
# olr_obs=data[:,1]
# plt.subplot(222)
# plt.plot(lat_obs,olr_obs,'--',c='r')


# plt.figure(1)
# data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[insol]_v2.txt',delimiter=',')
# lat_obs=data[:,0]
# insol_obs=data[:,1]
# plt.subplot(221)
# plt.plot(lat_obs,insol_obs,'-.',c='b')

# data=np.genfromtxt('/Users/nickedkins/Dropbox/Figure Data/x[lat]_y[olr]_v2.txt',delimiter=',')
# lat_obs=data[:,0]
# olr_obs=data[:,1]
# plt.subplot(222)
# plt.plot(lat_obs,olr_obs,'-.',c='r')

# data=np.genfromtxt('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/Latitudinal Distributions/Doug Mason Temperature vs Latitude NH.txt',delimiter=',')
# lat_obs=data[:,0]
# tg_obs=data[:,1]
# plt.subplot(223)
# plt.plot(lat_obs,tg_obs,'--',c='g')

# for i_file in range(len(a)):
#     # print(a[i_file])
#     plt.subplot(221)
#     # plt.plot(latgrid,np.mean(fnet_sw_dirfil[nlayers,:,i_file,0,:],axis=0),'-o',c='b')
#     plt.plot(latgrid,fnet_sw_dirfil[nlayers,1,i_file,0,:],'-o',label=a[i_file])
#     plt.xlabel('Latitude')
#     plt.ylabel('TOA SW')
#     plt.legend()
#     plt.xlim(0,90)
#     plt.subplot(222)
#     # plt.plot(latgrid,np.mean(fnet_lw_dirfil[nlayers,:,i_file,0,:],axis=0),'-o',c='r')
#     plt.plot(latgrid,fnet_lw_dirfil[nlayers,0,i_file,0,:],'-o',label=a[i_file])
#     plt.xlabel('Latitude')
#     plt.ylabel('TOA LW')
#     plt.legend()
#     plt.xlim(0,90)
#     # plt.plot(latgrid,fnet_lw_dirfil[nlayers,1,i_file,0,:],'-o',c='r')
#     plt.subplot(223)
#     # plt.plot(latgrid,np.mean(tbound_all_dirfil[:,i_file,0,:],axis=0),'-o',c='g')
#     plt.plot(latgrid,tbound_all_dirfil[1,i_file,0,:],'-o',label=a[i_file])
#     plt.xlabel('Latitude')
#     plt.ylabel('Tg')
#     plt.legend()
#     plt.xlim(0,90)
# plt.xlim(0,90)




# plt.figure(1)
# for i_file in range(len(a)):
#     print a[i_file]
#     plt.plot(latgrid,np.mean(tbound_all_dirfil[:,i_file,0,:],axis=0)-np.mean(tbound_all_dirfil[:,-1,0,:],axis=0),'-o',label=a[i_file])
# plt.grid(True,which='both')
# plt.xlabel('Latitude')
# plt.ylabel(r'$\Delta T$ compared to no advective heating')
# plt.legend()

# xx,yy=np.meshgrid(latgrid,np.linspace(1000,0,10))
# zz=tbound_all_dirfil[0,:,0,:]

# tbound_eqbs=tbound_all_dirfil[0,:,0,:]

# print(np.mean(tbound_all_dirfil))

baseline_tbound = 267.29358913282624-0.3

# plt.figure(1)
# plt.contourf(xx,yy,zz)
# plt.colorbar()

# for i_fn in range(len(a)):
#   plt.plot(latgrid,tbound_all_dirfil[0,i_fn,0,:]-tbound_all_dirfil[0,0,0,:])
# plt.grid(True,which='both')

# pertlats_actual = latgrid
# xx,yy=np.meshgrid(pertlats_actual,np.linspace(1000,0,nlayers+1))
# zz=tz_all_dirfil[:,0,1,0,:]-tz_all_dirfil[:,0,0,0,:]
# vabsmax=np.max(abs(zz))
# vmin=-vabsmax
# vmax=vabsmax
# plt.contourf(xx,yy,zz,levels=20,cmap='bwr',vmin=vmin,vmax=vmax)
# plt.colorbar()


# i_loop=0
# for i_pert in range(len(perts)):
#     for i_pertmol in range(len(pertmols)):
#         for i_pertlat in range(len(pertlats)):
#             for i_pertzon in range(len(pertzons)):
#                 for i_pertlay in range(len(pertlays)):
#                     # totuflux_all_prp[i_pert,i_pertmol,i_pertlat,i_pertzon,i_pertlay]=np.mean(totuflux_all_dirfil_lw[nlayers,0,i_loop,0,:],axis=0)
#                     print(i_pert, i_pertmol, i_pertlat, i_pertzon, i_pertlay)
#                     # tboound_zon_mean=np.mean(tbound_all_dirfil[:,i_loop,0,:],axis=0)
#                     tbound_all_prp[i_pert,i_pertmol,i_pertlat,i_pertzon,i_pertlay]=np.mean(tbound_all_dirfil[:,i_loop,0,:],axis=(0,1))
#                     i_loop+=1

# pertlats_actual = latgrid

# fig=plt.figure(1)
# fig.suptitle(r'$\Delta T$ from perturbation to $\tau_{cloud}$')
# # ax1 = fig.add_subplot(121)
# # ax1.title.set_text('Cloudy')
# # vmin=0
# # vmax=np.max(tbound_all_prp[0,0,:,0,:])-baseline_tbound)
# xx,yy=np.meshgrid(pertlats_actual,np.linspace(1000,0,10))
# # zz=tbound_all_prp[0,0,:,0,:].T-np.min(tbound_all_prp[0,0,:,0,:])
# zz=tbound_all_prp[0,0,:,0,:].T-baseline_tbound
# vabsmax=np.max(abs(zz))
# vmin=-vabsmax
# vmax=vabsmax
# plt.contourf(xx,yy,zz,13,vmin=vmin,vmax=vmax,cmap='coolwarm')
# plt.ylim(1000,0)
# plt.xlabel('Latitude')
# plt.ylabel('Pressure (hPa)')
# plt.colorbar()
# # ax2 = fig.add_subplot(122)
# # ax2.title.set_text('Clear')
# # xx,yy=np.meshgrid(pertlats_actual,np.linspace(1000,0,10))
# # zz=tbound_all_prp[0,0,:,1,:].T-np.min(tbound_all_prp[0,0,:,1,:])
# # plt.contourf(xx,yy,zz,13,vmin=vmin,vmax=vmax)
# # plt.ylim(1000,0)
# # plt.xlabel('Latitude')
# # plt.ylabel('Pressure (hPa)')
# # plt.colorbar()


# # tbound_all_dirfil[:,:,:,9:17]=np.NaN

# # alpha_furns=(tbound_all_dirfil[0,0,0,:]-tbound_all_dirfil[0,1,0,:])/10
# # alpha_fins=(tbound_all_dirfil[1,0,0,:]-tbound_all_dirfil[1,1,0,:])/10.

# # c_zon=4.

# # rad_eff=( 1.+ ( 4.*c_zon*alpha_furns*alpha_fins )/( alpha_furns+alpha_fins ) ) / ( 1. + c_zon * ( alpha_furns + alpha_fins ) )



# # plt.figure()
# # plt.grid(which='both')
# # plt.plot(latgrid,tbound_all_dirfil[0,0,0,:],'-o',label='cloudy')
# # plt.plot(latgrid,tbound_all_dirfil[1,0,0,:],'--o',label='clear')
# # plt.xlabel('Latitude')
# # plt.ylabel('Surface temperature (K)')
# # plt.legend()
# # plt.figure()
# # plt.grid(which='both')
# # plt.plot(latgrid,tbound_all_dirfil[0,0,0,:]-tbound_all_dirfil[0,1,0,:],'-o',label='cloudy')
# # plt.plot(latgrid,tbound_all_dirfil[1,0,0,:]-tbound_all_dirfil[1,1,0,:],'--o',label='clear')
# # plt.xlabel('Latitude')
# # plt.ylabel(r'$\Delta$ surface temperature (K)')
# # plt.legend()
# # plt.figure()
# # plt.grid(which='both')
# # plt.plot(latgrid,(tbound_all_dirfil[0,0,0,:]-tbound_all_dirfil[0,1,0,:])/10.,'-o',label='cloudy')
# # plt.plot(latgrid,(tbound_all_dirfil[1,0,0,:]-tbound_all_dirfil[1,1,0,:])/10.,'--o',label='clear')
# # plt.xlabel('Latitude')
# # plt.ylabel(r'$\alpha = \frac{dT}{dOLR}$')
# # plt.legend()
# # plt.figure()
# # plt.grid(which='both')
# # plt.plot(latgrid,(tbound_all_dirfil[0,0,0,:]-tbound_all_dirfil[0,1,0,:])/10.-(tbound_all_dirfil[1,0,0,:]-tbound_all_dirfil[1,1,0,:])/10.,'-o')
# # plt.axhline(0,linestyle='--')
# # plt.xlabel('Latitude')
# # plt.ylabel(r'$\alpha_{cloud}-\alpha_{clear}$')
# # plt.legend()
# # plt.figure()
# # plt.grid(which='both')
# # plt.plot(latgrid,(1.0-rad_eff)*100.,'-o')
# # plt.xlabel('Latitude')
# # plt.ylabel('Radiator fin effectiveness')
# # plt.legend()

# c_merids=[1,2,4]
# c_zonals=[1,2,4]

# # # energy flows master subplots fig (leave here!)
# # i_loops=0
# # for i_z in range(len(c_zonals)):
# #     for i_m in range(len(c_merids)):

# #         print i_loops*2, i_loops*2+1

# #         fig=plt.figure(1)
# #         plt.subplot(331+i_loops)
# #         plt.gca().set_title('c_zonal={}, c_merid={}'.format(c_zonals[i_z],c_merids[i_m]))
# #         plt.plot(latgrid,totuflux_all_dirfil[nlayers,0,i_loops*2,0,:]-totuflux_all_dirfil[nlayers,0,i_loops*2+1,0,:],'-o',label=r'$\Delta$ TOA Fnet (cloudy)')
# #         plt.plot(latgrid,zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:],'-o',label=r'$\Delta$ zonal_transp (cloudy)')
# #         plt.plot(latgrid,merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:],'-o',label=r'$\Delta$ merid_transp (cloudy)')
# #         plt.plot(latgrid,(merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:])+(totuflux_all_dirfil[nlayers,0,i_loops*2,0,:]-totuflux_all_dirfil[nlayers,0,i_loops*2+1,0,:]),'-o',label='Total (cloudy)')
# #         plt.axhline(0,linestyle='--')
# #         plt.axhline(10,linestyle='--')

# #         plt.gca().set_title('c_zonal={}, c_merid={}'.format(c_zonals[i_z],c_merids[i_m]))
# #         plt.plot(latgrid,totuflux_all_dirfil[nlayers,1,i_loops*2,0,:]-totuflux_all_dirfil[nlayers,1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ TOA Fnet (clear)')
# #         plt.plot(latgrid,zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ zonal_transp (clear)')
# #         plt.plot(latgrid,merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ merid_transp (clear)')
# #         plt.plot(latgrid,(merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])+(totuflux_all_dirfil[nlayers,1,i_loops*2,0,:]-totuflux_all_dirfil[nlayers,1,i_loops*2+1,0,:]),'--o',label='Total (clear)')
# #         plt.xlabel('Latitude')
# #         plt.ylabel(r'$\Delta$ budg (Wm$^{-2}$)')
# #         plt.axhline(0,linestyle='--')
# #         plt.axhline(10,linestyle='--')

# #         i_loops+=1

# #         if(i_loops==8):
# #             handles, labels = plt.gca().get_legend_handles_labels()
# #             fig.legend(handles, labels, loc='center right')



# # energy flows individual subplots fig
# i_loops=0
# for i_z in range(len(c_zonals)):
#     for i_m in range(len(c_merids)):

#         print i_loops*2, i_loops*2+1

#         fig=plt.figure(1)
#         # plt.suptitle(r'$\Delta$ TOA Fnet (cloudy)')
#         # plt.suptitle(r'$\Delta$ zonal_transp (cloudy)')
#         # plt.suptitle(r'$\Delta$ meridional transp (cloudy)')
#         # plt.suptitle(r'$\Delta$ total (cloudy)')
#         plt.suptitle(r'$\Delta$ T (cloudy)')
#         # plt.suptitle(r'$\alpha$ = dT/d(box budget) (cloudy)')

#         # plt.suptitle(r'$\Delta$ TOA Fnet (clear)')
#         # plt.suptitle(r'$\Delta$ zonal_transp (clear)')
#         # plt.suptitle(r'$\Delta$ meridional transp (clear)')
#         # plt.suptitle(r'$\Delta$ total (clear)')
#         # plt.suptitle(r'$\Delta$ T (clear)')
#         # plt.suptitle(r'$\alpha$ = dT/d(box budget) (clear)')

#         # plt.suptitle(r'$\Delta$ TOA Fnet (avg)')
#         # plt.suptitle(r'$\Delta$ zonal_transp (avg)') #pointless!
#         # plt.suptitle(r'$\Delta$ meridional transp (avg)'))
#         # plt.suptitle(r'$\Delta$ total (avg)')
#         # plt.suptitle(r'$\Delta$ T (avg)')
#         # plt.suptitle(r'$\alpha$ = dT/d(box budget) (avg)'))

#         plt.subplot(131+i_m)
#         plt.gca().set_title('c_merid={}'.format(c_merids[i_m]))
        
#         # plt.plot(latgrid,(fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:])*-1.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(7,11)
#         # plt.plot(latgrid,zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:],'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(-4,0)
#         # plt.plot(latgrid,merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:],'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(-1.1,1)
#         # plt.plot(latgrid,(merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:]),'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         plt.plot(latgrid,(tbound_all_dirfil[0,i_loops*2,0,:]-tbound_all_dirfil[0,i_loops*2+1,0,:]),'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(5,8)

#         #clear
#         # plt.plot(latgrid,(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:])*-1.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(9,14)
#         # plt.plot(latgrid,zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:],'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(0,4)
#         # plt.plot(latgrid,merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:],'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(-2,2)
#         # plt.plot(latgrid,(merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:]),'--o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(10,17)
#         # plt.plot(latgrid,(tbound_all_dirfil[1,i_loops*2,0,:]-tbound_all_dirfil[1,i_loops*2+1,0,:]),'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(5,8)

#         # average (clear + cloudy)/2
#         # plt.plot(latgrid,((fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:])*-1.+(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:])*-1.)/2.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(8,12)
#         # plt.plot(latgrid,(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:]+zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])/2.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(-1,1)
#         # plt.plot(latgrid,(merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:]+merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])/2.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(-1.5,1)
#         # plt.plot(latgrid,((merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:])+(merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:]))/2.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(9,11)
#         # plt.plot(latgrid,((tbound_all_dirfil[0,i_loops*2,0,:]-tbound_all_dirfil[0,i_loops*2+1,0,:])+(tbound_all_dirfil[1,i_loops*2,0,:]-tbound_all_dirfil[1,i_loops*2+1,0,:]))/2.,'-o',label='c_zonal={}'.format(c_zonals[i_z]))
#         # plt.ylim(5,7.5)

#         # sensitivity alpha
#         # plt.plot( latgrid, (tbound_all_dirfil[0,i_loops*2,0,:]-tbound_all_dirfil[0,i_loops*2+1,0,:]) / ( (merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:]) ),'-o',label='c_zonal={}'.format(c_zonals[i_z]) )
#         # plt.ylim(0.6,1.3)
#         # plt.plot( latgrid, (tbound_all_dirfil[1,i_loops*2,0,:]-tbound_all_dirfil[1,i_loops*2+1,0,:]) / ( (merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:]) ),'-o',label='c_zonal={}'.format(c_zonals[i_z]) )
#         # plt.ylim(0.35,0.65)
#         # plt.plot( latgrid, ((tbound_all_dirfil[0,i_loops*2,0,:]-tbound_all_dirfil[0,i_loops*2+1,0,:]) / ( (merid_transps_all_dirfil[0,i_loops*2,0,:]-merid_transps_all_dirfil[0,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[0,i_loops*2,0,:]-zonal_transps_all_dirfil[0,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,0,i_loops*2,0,:]-fnet_all_dirfil[nlayers,0,i_loops*2+1,0,:]) ) + (tbound_all_dirfil[1,i_loops*2,0,:]-tbound_all_dirfil[1,i_loops*2+1,0,:]) / ( (merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])-(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:]) )) /2. ,'-o',label='c_zonal={}'.format(c_zonals[i_z]) )
#         # plt.ylim(0.55,0.8)


#         # plt.axhline(0,linestyle='--')
#         # plt.axhline(10,linestyle='--')
#         plt.xlabel('Latitude')
#         # plt.ylabel(r'$\Delta$ budg (Wm$^{-2}$)')



#         # plt.gca().set_title('c_zonal={}, c_merid={}'.format(c_zonals[i_z],c_merids[i_m]))
#         # plt.plot(latgrid,fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ TOA Fnet (clear)')
#         # plt.plot(latgrid,zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ zonal_transp (clear)')
#         # plt.plot(latgrid,merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:],'--o',label=r'$\Delta$ merid_transp (clear)')
#         # plt.plot(latgrid,(merid_transps_all_dirfil[1,i_loops*2,0,:]-merid_transps_all_dirfil[1,i_loops*2+1,0,:])+(zonal_transps_all_dirfil[1,i_loops*2,0,:]-zonal_transps_all_dirfil[1,i_loops*2+1,0,:])+(fnet_all_dirfil[nlayers,1,i_loops*2,0,:]-fnet_all_dirfil[nlayers,1,i_loops*2+1,0,:]),'--o',label='Total (clear)')
#         # plt.xlabel('Latitude')
#         # plt.ylabel(r'$\Delta$ budg (Wm$^{-2}$)')
#         # plt.axhline(0,linestyle='--')
#         # plt.axhline(10,linestyle='--')

#         i_loops+=1

#         if(i_loops==8):
#             handles, labels = plt.gca().get_legend_handles_labels()
#             fig.legend(handles, labels, loc='center right')

# # plt.figure(1)
# # plt.plot(zonal_transps_all_dirfil[0,::3,:,0])
# # plt.plot(zonal_transps_all_dirfil[0,1::3,:,0])
# # plt.plot(zonal_transps_all_dirfil[0,2::3,:,0])

# # c_merids=[0.,2.,4.]
# # c_zonals=[0.,2.,4.]

# # t_merids=np.mean(tbound_all_dirfil[1,:3,0,:],axis=1)
# # t_zonals=np.mean(tbound_all_dirfil[1,::3,0,:],axis=1)

# # zz=np.mean(tbound_all_dirfil[1,:,0,:],axis=1).reshape((len(c_merids),len(c_zonals)))
# # zz=zz-np.mean(tbound_all_dirfil[1,0,0,:])

# # xx,yy = np.meshgrid(c_merids,c_zonals)
# # plt.contourf(xx,yy,zz)
# # plt.xlabel('c_merid')
# # plt.ylabel('c_zonal')
# # plt.colorbar()

# # plt.figure(1)
# # plt.plot(latgrid,wkl_master[0,0,0,:],'-o',label='cloudy')
# # plt.plot(latgrid,wkl_master[0,0,0,:],'-o',label='clear')
# # plt.xlabel('Latitude')
# # plt.ylabel('Surface H2O mixing ratio')
# # plt.legend()

# # plt.figure(1)
# # plt.plot(latgrid,(tbound_all_dirfil[0,1,0,:]-tbound_all_dirfil[0,0,0,:])/10.,'-o',label='cloudy')
# # plt.plot(latgrid,(tbound_all_dirfil[1,1,0,:]-tbound_all_dirfil[1,0,0,:])/10.,'-o',label='clear')
# # plt.xlabel('Latitude')
# # plt.ylabel(r'$\alpha=\frac{dT}{dOLR}$')
# # # plt.plot(latgrid,tbound_all_dirfil[0,1,0,:]-tbound_all_dirfil[0,0,0,:]-(tbound_all_dirfil[1,1,0,:]-tbound_all_dirfil[1,0,0,:]),'-o')
# # plt.legend()

# # for i in range(nlatcols):
# #     print (tbound_all_dirfil[0,1,0,i]-tbound_all_dirfil[0,0,0,i])/10.,','    
# # print
# # for i in range(nlatcols):
# #     print (tbound_all_dirfil[1,1,0,i]-tbound_all_dirfil[1,0,0,i])/10.,','    

# # print (tbound_all_dirfil[0,1,0,:]-tbound_all_dirfil[0,0,0,:])/10.
# # print (tbound_all_dirfil[1,1,0,:]-tbound_all_dirfil[1,0,0,:])/10.

# # wklfacs=np.logspace(-3,0,num=10,base=10.)
# # wklfacs=np.logspace(-3,0,num=5,base=10.)
# # print wklfacs

# # c_zonal_transps = np.linspace(1,30,num=10)
# # c_zonal_transps = np.linspace(0.1,10,num=10)

# # plt.figure(1)
# # plt.plot(c_zonal_transps,tbound_all_dirfil[1,:,0]-tbound_all_dirfil[0,:,0],'-o')
# # plt.xlabel(r'$c$: tropical zonal transport coefficient')
# # plt.ylabel(r'$T_{furn}-T_{fin}$ (K)')

# # for i in range(1,ndirs):
# #   plt.figure(1)
# #   plt.plot(wklfacs,(totuflux_all_dirfil[nlayers,1,:,i]-totuflux_all_dirfil[nlayers,1,:,i-1])/10.,'-o',label='Fin')
# #   plt.plot(wklfacs,(totuflux_all_dirfil[nlayers,0,:,i]-totuflux_all_dirfil[nlayers,0,:,i-1])/10.,'-o',label='Furnace')
# #   plt.xlabel(r'Fin H$_2$O Factor')
# #   plt.ylabel(r'$\beta = \frac{\Delta OLR}{ \Delta T}$ (Wm$^{-2}$K$^{-1}$)')
# # plt.legend()

# nfiles=2

# # plt.figure(1)
# # plt.plot(latgrid,totuflux_all_dirfil[nlayers,0,0,0,:],'-o')
# # plt.plot(latgrid,totuflux_all_dirfil[nlayers,0,1,0,:],'-o')
# # plt.plot(latgrid,10./(totuflux_all_dirfil[nlayers,0,2,0,:]-totuflux_all_dirfil[nlayers,0,0,0,:]),'-o')
# # plt.xlabel('Latitude')
# # plt.ylabel('dT/dOLR')
# # plt.plot((tbound_all_dirfil[0,1,0,:]-tbound_all_dirfil[0,0,0,:])/10.,'-o')

# # betas = np.zeros((nzoncols,nfiles,ndirs))

# # for i_zon in range(nzoncols):
# #   for i_file in range(nfiles):
# #       for i_dir in range(1,ndirs):
# #           plt.figure(1)
# #           # plt.plot(tbound_all_dirfil[1,i_file,i_dir],(totuflux_all_dirfil[nlayers,1,i_file,i_dir]-totuflux_all_dirfil[nlayers,1,i_file,i_dir-1])/(tbound_all_dirfil[1,i_file,i_dir]-tbound_all_dirfil[1,i_file,i_dir-1]),'o',c=colors[i_file])
# #           # plt.plot(tbound_all_dirfil[0,i_file,i_dir],(totuflux_all_dirfil[nlayers,0,i_file,i_dir]-totuflux_all_dirfil[nlayers,0,i_file,i_dir-1])/(tbound_all_dirfil[0,i_file,i_dir]-tbound_all_dirfil[0,i_file,i_dir-1]),'o',c='r')
# #           betas[i_zon,i_file,i_dir] = (totuflux_all_dirfil[nlayers,i_zon,i_file,i_dir]-totuflux_all_dirfil[nlayers,i_zon,i_file,i_dir-1])/(tbound_all_dirfil[i_zon,i_file,i_dir]-tbound_all_dirfil[i_zon,i_file,i_dir-1])

# # plt.figure(1)
# # for i_zon in range(nzoncols):
# #   plt.semilogy(pz_master[])

# # # for i_dir in range(1,ndirs):
# # #     plt.semilogx(wklfacs,1./betas[1,:,i_dir],'-o',label='tbound_fin='+str(tbound_all_dirfil[0,0,i_dir]))
# # #     plt.semilogx(wklfacs,1./betas[0,:,i_dir],'-o',label='tbound_furn='+str(tbound_all_dirfil[0,0,i_dir]))
# # #     plt.xlabel('H2O Factor')
# # #     plt.ylabel(r'$\alpha_{fin}$')
# # # plt.legend()

# # Z = betas[1,:,:].T

# # # print wklfacs.shape, tbound_all_dirfil[1,0,:].shape

# # X,Y = np.meshgrid(wklfacs,tbound_all_dirfil[1,0,:])

# # print X.shape,Y.shape,Z.shape

# # f = interpolate.interp2d(X,Y,Z)

# # wklfacs_grid = np.logspace(-3,0,num=100,base=10.)
# # tbound_grid=np.linspace(280,320,100)

# # Z_int = np.zeros((len(wklfacs_grid),len(tbound_grid)))

# # for i_w in range(len(wklfacs_grid)):
# #   for i_t in range(len(tbound_grid)):
# #       Z_int[i_w,i_t]=f(wklfacs_grid[i_w],tbound_grid[i_t])

# # X,Y=np.meshgrid(wklfacs_grid,tbound_grid)

# # plt.figure(1)
# # plt.contourf(X,Y,Z_int)
# # plt.gca().set_xscale('log')
# # plt.xlabel('Fin H$_2$O factor')
# # plt.ylabel('Fin surface temperature (K)')
# # cb=plt.colorbar()
# # cb.ax.set_xlabel(r'$\beta_{fin}$',)


# # tboundsnje = [324,324,324,325,326,327,328,329,331,334]

# # for i in range(nfiles):
# #   print f(wklfacs[i],tboundsnje[i])[0], ','

# # plt.figure(1)
# # # plt.contour(X,Y,Z,20)
# # plt.imshow(Z)
# # plt.gca().set_xscale('log')
# # plt.colorbar()

# # print tbound_all_dirfil[0,:,0]

# # matplotlib.pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.84, top=0.9, wspace=0.35, hspace=0.35)
# plt.annotate('test',(0.5,0.5))
# ax=plt.gca()
# ax.text(0.5, 0.5, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
fig=plt.gcf()
# fig.suptitle(str(datetime.datetime.now()))
plt.tight_layout()
show()