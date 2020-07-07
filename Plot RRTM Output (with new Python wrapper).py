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

print 'Started'

directories = [
'/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Current Output/'
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/betas v3/tbound=280/',
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/betas v3/tbound=290/',
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/betas v3/tbound=300/',
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/betas v3/tbound=310/',
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/betas v3/tbound=320/',
]


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
	plt.rcParams['figure.figsize'] = (10,10)
	plt.rcParams['font.size'] = 20
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
print colors

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
	for i_cld in range(ncloudcols):
		# plt.subplot(331)
		# plt.semilogy(tz_master[:,i_cld],pz_master[:,i_cld],'-o',label=str(i_cld))
		# # plt.semilogy(tavel_master[:,i_cld],pavel_master[:,i_cld],'-o',label=str(i_cld))
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('T (K)')
		# plt.legend()
		# plt.ylabel('Pressure (hPa)')
		# plt.legend()
		# plt.grid(which='both')
		# plt.subplot(332)
		# plt.semilogy(totuflux_master[:,i_cld],pz_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('totuflux')
		# plt.subplot(333)
		# plt.semilogy(totdflux_master[:,i_cld],pz_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('totdflux')
		# plt.subplot(334)
		# plt.semilogy(fnet_master[:,i_cld],pz_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('fnet')
		# plt.subplot(335)
		# plt.semilogy(wkl_master[:,i_cld,0],pavel_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('wkl1')
		# plt.subplot(336)
		# plt.semilogy(wkl_master[:,i_cld,1],pavel_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('wkl2')
		# plt.subplot(337)
		# plt.semilogy(wkl_master[:,i_cld,2],pavel_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('wkl3')
		# plt.subplot(338)
		# plt.semilogy(wbrodl_master[:,i_cld],pz_master[:,i_cld])
		# plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		# plt.xlabel('wbrodl')
		# plt.subplot(339)
		# plt.plot(dfnet_master[:,i_cld],pavel_master[:,i_cld],'-o',label=str(fn)+str(i_cld))
		plt.semilogy(np.mean(dfnet_master[:,:],axis=1),pavel_master[:,i_cld],'-o',label=str(fn)+str(i_cld))
		plt.plot(conv_master[:,i_cld],pz_master[:,i_cld])
		plt.axvline(-eqb_maxdfnet,linestyle='--')
		plt.axvline(eqb_maxdfnet,linestyle='--')
		plt.ylim(np.max(pz_master[:,i_cld]),np.min(pz_master[:,i_cld]))
		plt.xlabel(r'$\Delta F_{net}$ in layer (Wm$^{-2}$)')
		plt.ylabel('Pressure (hPa)')
		plt.grid(which='both')
		plt.legend()


def readrrtmoutput(fn):
	f = open(fn)


ndirs=len(directories)
a = sorted(listdir(directories[0]))
if('.DS_Store' in a):
	a.remove('.DS_Store')
nfiles=len(a)

nlayers=60
nmol=7
ncloudcols=2

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
istrm=1 			# ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0). 
						#0=4 streams
						#1=8 streams
idelm=1 			# flag for outputting downwelling fluxes computed using the delta-M scaling approximation. 0=output "true" direct and diffuse downwelling fluxes, 1=output direct and diffuse downwelling fluxes computed with delta-M approximation
icos=0 				#0:there is no need to account for instrumental cosine response, 1:to account for instrumental cosine response in the computation of the direct and diffuse fluxes, 2:2 to account for instrumental cosine response in the computation of the diffuse fluxes only
semis=np.ones(16)	#all spectral bands the same as iemissm
semiss=np.ones(29) 	#all spectral bands the same as iemissm (surface, I think)
iform=1
psurf=1000.
pmin=0.
secntk=0
cinp=1.356316e-19
ipthak=3
ipthrk=3
juldat=0 		#Julian day associated with calculation (1-365/366 starting January 1). Used to calculate Earth distance from sun. A value of 0 (default) indicates no scaling of solar source function using earth-sun distance.
sza=65. 			#Solar zenith angle in degrees (0 deg is overhead).
isolvar=0 		#= 0 each band uses standard solar source function, corresponding to present day conditions. 
				#= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
				#= 2 scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))			
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
timesteps=10000
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

vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,ncloudcols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb]

# nlayers, ncloudcols, nfiles, ndirs

nlayers_dirfil=nlayers
ncloudcols_dirfil=ncloudcols
# nfiles=7
# ndirs=2

tbound_all_dirfil = np.zeros((ncloudcols_dirfil,nfiles,ndirs))
totuflux_all_dirfil=np.zeros((nlayers_dirfil+1,ncloudcols_dirfil,nfiles,ndirs))

i_dir=0
for directory in directories:

	filenames = []
	dir_label = directory.split('/')[-2]
	print
	print(dir_label)
	print
	a = sorted(listdir(directory))
	filenames.append(a)
	if('.DS_Store' in a):
		a.remove('.DS_Store')

	i_file=0
	for fn in a:
		print(fn)
		output_file = directory + fn
		f=open(output_file,'r')


		gravity	=	float	(	f.readline().rstrip('\n')	)
		avogadro	=	float	(	f.readline().rstrip('\n')	)
		iatm	=	int	(	f.readline().rstrip('\n')	)
		ixsect	=	int	(	f.readline().rstrip('\n')	)
		iscat	=	int	(	f.readline().rstrip('\n')	)
		numangs	=	int	(	f.readline().rstrip('\n')	)
		iout	=	int	(	f.readline().rstrip('\n')	)
		icld	=	int	(	f.readline().rstrip('\n')	)
		tbound	=	float	(	f.readline().rstrip('\n')	)
		iemiss	=	int	(	f.readline().rstrip('\n')	)
		iemis	=	int	(	f.readline().rstrip('\n')	)
		ireflect	=	int	(	f.readline().rstrip('\n')	)
		iaer	=	int	(	f.readline().rstrip('\n')	)
		istrm	=	int	(	f.readline().rstrip('\n')	)
		idelm	=	int	(	f.readline().rstrip('\n')	)
		icos	=	int	(	f.readline().rstrip('\n')	)
		iform	=	int	(	f.readline().rstrip('\n')	)
		nlayers	=	int	(	f.readline().rstrip('\n')	)
		nmol	=	int	(	f.readline().rstrip('\n')	)
		psurf	=	float	(	f.readline().rstrip('\n')	)
		pmin	=	float	(	f.readline().rstrip('\n')	)
		secntk	=	str	(	f.readline().rstrip('\n')	)
		cinp	=	str	(	f.readline().rstrip('\n')	)
		ipthak	=	float	(	f.readline().rstrip('\n')	)
		ipthrk	=	float	(	f.readline().rstrip('\n')	)
		juldat	=	float	(	f.readline().rstrip('\n')	)
		sza	=	float	(	f.readline().rstrip('\n')	)
		isolvar	=	int	(	f.readline().rstrip('\n')	)
		lapse	=	float	(	f.readline().rstrip('\n')	)
		tmin	=	float	(	f.readline().rstrip('\n')	)
		tmax	=	float	(	f.readline().rstrip('\n')	)
		rsp	=	float	(	f.readline().rstrip('\n')	)
		gravity	=	float	(	f.readline().rstrip('\n')	)
		pin2	=	float	(	f.readline().rstrip('\n')	)
		pico2	=	float	(	f.readline().rstrip('\n')	)
		pio2	=	float	(	f.readline().rstrip('\n')	)
		piar	=	float	(	f.readline().rstrip('\n')	)
		pich4	=	float	(	f.readline().rstrip('\n')	)
		pih2o	=	float	(	f.readline().rstrip('\n')	)
		pio3	=	float	(	f.readline().rstrip('\n')	)
		mmwn2	=	float	(	f.readline().rstrip('\n')	)
		mmwco2	=	float	(	f.readline().rstrip('\n')	)
		mmwo2	=	float	(	f.readline().rstrip('\n')	)
		mmwar	=	float	(	f.readline().rstrip('\n')	)
		mmwch4	=	float	(	f.readline().rstrip('\n')	)
		mmwh2o	=	float	(	f.readline().rstrip('\n')	)
		mmwo3	=	float	(	f.readline().rstrip('\n')	)
		piair	=	float	(	f.readline().rstrip('\n')	)
		totmolec	=	float	(	f.readline().rstrip('\n')	)
		surf_rh	=	float	(	f.readline().rstrip('\n')	)
		vol_mixh2o_min	=	float	(	f.readline().rstrip('\n')	)
		vol_mixh2o_max	=	float	(	f.readline().rstrip('\n')	)
		ur_min	=	float	(	f.readline().rstrip('\n')	)
		ur_max	=	float	(	f.readline().rstrip('\n')	)
		eqb_maxhtr	=	float	(	f.readline().rstrip('\n')	)
		timesteps	=	int	(	f.readline().rstrip('\n')	)
		cti	=	int	(	f.readline().rstrip('\n')	)
		maxhtr	=	float	(	f.readline().rstrip('\n')	)
		cld_lay	=	float	(	f.readline().rstrip('\n')	)
		ncloudcols	=	int	(	f.readline().rstrip('\n')	)
		master_input =	float	(	f.readline().rstrip('\n')	)
		conv_on=float	(	f.readline().rstrip('\n')	)
		surf_lowlev_coupled=float	(	f.readline().rstrip('\n')	)
		lay_intp=float	(	f.readline().rstrip('\n')	)
		lw_on=float	(	f.readline().rstrip('\n')	)
		sw_on=float	(	f.readline().rstrip('\n')	)
		eqb_maxdfnet=float	(	f.readline().rstrip('\n')	)
		toa_fnet_eqb=float	(	f.readline().rstrip('\n')	)

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

		inflags=np.ones(ncloudcols)
		iceflags=np.ones(ncloudcols)
		liqflags=np.ones(ncloudcols)
		cld_lays=np.ones(ncloudcols)
		cld_fracs=np.ones(ncloudcols)
		# tauclds=np.array((3.0,0.0))
		# ssaclds=np.array((0.5,0.0))
		tauclds=np.ones(ncloudcols)
		ssaclds=np.ones(ncloudcols)
		tbound_master=np.ones(ncloudcols)
		toa_fnet_master=np.ones(ncloudcols)
		zonal_transps=np.zeros(ncloudcols)

		tz_master=np.zeros((nlayers+1,ncloudcols))
		tavel_master=np.zeros((nlayers,ncloudcols))
		pz_master=np.zeros((nlayers+1,ncloudcols))
		pavel_master=np.zeros((nlayers,ncloudcols))
		altz_master=np.zeros((nlayers+1,ncloudcols))
		altavel_master=np.zeros((nlayers,ncloudcols))

		totuflux_master=np.zeros((nlayers+1,ncloudcols))
		totuflux_lw_master=np.zeros((nlayers+1,ncloudcols))
		totuflux_sw_master=np.zeros((nlayers+1,ncloudcols))
		totdflux_master=np.zeros((nlayers+1,ncloudcols))
		totdflux_lw_master=np.zeros((nlayers+1,ncloudcols))
		totdflux_sw_master=np.zeros((nlayers+1,ncloudcols))
		fnet_master=np.zeros((nlayers+1,ncloudcols))
		fnet_lw_master=np.zeros((nlayers+1,ncloudcols))
		fnet_sw_master=np.zeros((nlayers+1,ncloudcols))
		htr_master=np.zeros((nlayers+1,ncloudcols))
		htr_lw_master=np.zeros((nlayers+1,ncloudcols))
		htr_sw_master=np.zeros((nlayers+1,ncloudcols))
		wbrodl_master=np.zeros((nlayers+1,ncloudcols))
		conv_master=np.zeros((nlayers+1,ncloudcols))

		wkl_master=np.zeros((nlayers,ncloudcols,nmol+1))

		# vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay]
		# vars_lay=[pavel,tavel,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel]
		# vars_lev=[totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,tz,altz]
		# vars_misc_1d=[semis,semiss,solvar]
		# vars_misc_1d_lens=[16,29,29]
		# vars_lay_nmol=[wkl]

		vars_0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr,cld_lay,ncloudcols,master_input,conv_on,surf_lowlev_coupled,lay_intp,lw_on,sw_on,eqb_maxdfnet,toa_fnet_eqb]
		vars_master_lay_cld=[tavel_master,pavel_master,altavel_master,wbrodl_master]
		vars_master_lev_cld=[tz_master,pz_master,altz_master,totuflux_master,totuflux_lw_master,totuflux_sw_master,totdflux_master,totdflux_lw_master,totdflux_sw_master,fnet_master,fnet_lw_master,fnet_sw_master,htr_master,htr_lw_master,htr_sw_master,conv_master]
		vars_misc_1d=[semis,semiss,solvar]
		vars_misc_1d_lens=[16,29,29]
		vars_master_lay_cld_nmol=[wkl_master]
		vars_master_cld=[inflags,iceflags,liqflags,cld_lays,cld_fracs,tauclds,ssaclds,tbound_master,toa_fnet_master,zonal_transps]


		# for x in vars_lay:
		# 	for i in range(nlayers):
		# 		x[i] = f.readline()

		# for x in vars_lev:
		# 	for i in range(nlayers+1):
		# 		x[i] = f.readline()

		# i_lens=0
		# for x in vars_misc_1d:
		# 	for i in range(vars_misc_1d_lens[i_lens]):
		# 		x[i] = f.readline()
		# 	i_lens+=1


		# for x in vars_lay_nmol:
		# 	for i in range(shape(x)[0]):
		# 		for j in range(shape(x)[1]):
		# 			x[i,j] = f.readline()


		for x in vars_master_lay_cld:
			for j in range(ncloudcols):
				for i in range(nlayers):
					x[i,j] = f.readline()

		for x in vars_master_lev_cld:
			for j in range(ncloudcols):
				for i in range(nlayers+1):
					x[i,j] = f.readline()

		i_lens=0
		for x in vars_misc_1d:
			for i in range(vars_misc_1d_lens[i_lens]):
				x[i] = f.readline()
			i_lens+=1


		for x in vars_master_lay_cld_nmol:
			for k in range(nmol+1):
				for j in range(ncloudcols):
					for i in range(nlayers):
						x[i,j,k] = f.readline()

		for x in vars_master_cld:
			for i in range(ncloudcols):
				x[i]=f.readline()


		dfnet_master=np.zeros((nlayers,ncloudcols))

		for i_cld in range(ncloudcols):
			for i in range(nlayers):
				dfnet_master[i,i_cld]=fnet_master[i+1,i_cld]-fnet_master[i,i_cld]


		#print output for easy spreadsheet transfer
		# for i in range(nlayers):
		# 	print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(pz[i],pavel[i],altz[i]/1000.,tz[i],tavel[i],totuflux[i],totuflux_lw[i],totuflux_sw[i],totdflux[i],totdflux_lw[i],totdflux_sw[i],fnet[i],fnet_lw[i],fnet_sw[i]))
		# print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(pz[nlayers],'na',altz[nlayers]/1000.,tz[nlayers],'na',totuflux[nlayers],totuflux_lw[nlayers],totuflux_sw[nlayers],totdflux[nlayers],totdflux_lw[nlayers],totdflux_sw[nlayers],fnet[nlayers],fnet_lw[nlayers],fnet_sw[nlayers],tbound))

		# totuflux_lw_master[:,i_file,i_dir]=totuflux_lw
		# totuflux_sw_master[:,i_file,i_dir]=totuflux_sw
		# pz_master[:,i_file,i_dir]=pz
		# cld_lay_master[:,i_file,i_dir]=cld_lay

		Q=np.zeros(ncloudcols)
		Qv=np.zeros(ncloudcols)
		esat_lowlay=np.zeros(ncloudcols)
		esat_bound=np.zeros(ncloudcols)
		qsat_lowlay=np.zeros(ncloudcols)
		qsat_bound=np.zeros(ncloudcols)
		qstar=np.zeros(ncloudcols)
		Evap=np.zeros(ncloudcols)
		Fah=np.zeros(ncloudcols)
		col_budg=np.zeros(ncloudcols)
		deltheta=85.
		cp=1.003
		L=2429.8
		r=0.8
		A1=1.
		A2=1.
		for k in range(2):
			for i_cld in range(ncloudcols):
				Q[i_cld]=(fnet_master[nlayers,i_cld]-fnet_master[0,i_cld])
				esat_lowlay[i_cld] = 6.1094*exp(17.625*(tz_master[0,i_cld]-273.15)/(tz_master[0,i_cld]-273.15+243.04))
				esat_bound[i_cld] = 6.1094*exp(17.625*(tbound_master[i_cld]-273.15)/(tbound_master[i_cld]-273.15+243.04))
				qsat_lowlay[i_cld]=0.622*esat_lowlay[i_cld]/(pz_master[0,i_cld]-esat_lowlay[i_cld])
				qsat_bound[i_cld]=0.622*esat_bound[i_cld]/(pz_master[0,i_cld]-esat_bound[i_cld])
				qstar[i_cld]=qsat_bound[i_cld]-r*qsat_lowlay[i_cld]
				b=L*qstar[1]/(cp*deltheta)
				Qv[i_cld]=(1-b)*Q[i_cld]
				Evap[i_cld]=-b*Q[i_cld]
		Fah[0]=A2/A1*Qv[1]
		Fah[1]=-1.*Fah[0]
		col_budg[0]=fnet_master[nlayers,0]+Fah[0]+Evap[1]
		col_budg[1]=fnet_master[nlayers,1]+Fah[1]-Evap[1]
		# for i_cld in range(ncloudcols):
		# 	# print 'i_cld: {:d} Q: {:4.0f} b: {:4.2f} Qv: {:4.0f} E: {:4.0f} tbound: {:4.0f} Fah1: {:4.0f} fnet at toa: {:4.0f} col budg 1 {:4.0f} col budg 2 {:4.0f} sum col budg {:4.0f} fdown {:4.0f} olr {:4.0f}'.format(i_cld,Q[i_cld],b,Qv[i_cld],Evap[i_cld],tbound_master[i_cld],Fah[0],fnet_master[nlayers,i_cld],col_budg[0],col_budg[1],np.sum(col_budg),totdflux_sw_master[nlayers,i_cld],totuflux_lw_master[nlayers,i_cld])
		# 	print 'fdown {:4.0f} olr {:4.0f} fah {:4.0f} evap {:4.0f} tmean {:6.1f}'.format(totdflux_master[nlayers,i_cld],totuflux_master[nlayers,i_cld],Fah[i_cld],Evap[i_cld],np.mean(tbound_master))
		# plt.figure(1)
		# plt.subplot(121+i_dir)
		# plt.plot(wkl_master[0,1,0],tbound_master[0],'o',c='r',label='Twarm')			
		# plt.plot(wkl_master[0,1,0],tbound_master[1],'o',c='b',label='Tcold')
		# plt.plot(wkl_master[0,1,0],np.mean(tbound_master),'o',c='g',label='Tavg')
		# plt.xlabel('wklfac')
		# plt.ylabel('Temperature (K)')
		# plt.ylim(295,307)
		if(i_file==0):
			plt.legend()
		plotrrtmoutput_masters()
		tbound_all_dirfil[:,i_file,i_dir]=tbound_master
		totuflux_all_dirfil[:,:,i_file,i_dir]=totuflux_master
		i_file+=1
		
	i_dir+=1	

# wklfacs=np.logspace(-3,0,num=10,base=10.)
wklfacs=np.logspace(-3,0,num=5,base=10.)
# print wklfacs

# c_zonal_transps = np.linspace(1,30,num=10)
# c_zonal_transps = np.linspace(0.1,10,num=10)

# plt.figure(1)
# plt.plot(c_zonal_transps,tbound_all_dirfil[1,:,0]-tbound_all_dirfil[0,:,0],'-o')
# plt.xlabel(r'$c$: tropical zonal transport coefficient')
# plt.ylabel(r'$T_{furn}-T_{fin}$ (K)')

# for i in range(1,ndirs):
# 	plt.figure(1)
# 	plt.plot(wklfacs,(totuflux_all_dirfil[nlayers,1,:,i]-totuflux_all_dirfil[nlayers,1,:,i-1])/10.,'-o',label='Fin')
# 	plt.plot(wklfacs,(totuflux_all_dirfil[nlayers,0,:,i]-totuflux_all_dirfil[nlayers,0,:,i-1])/10.,'-o',label='Furnace')
# 	plt.xlabel(r'Fin H$_2$O Factor')
# 	plt.ylabel(r'$\beta = \frac{\Delta OLR}{ \Delta T}$ (Wm$^{-2}$K$^{-1}$)')
# plt.legend()

nfiles=5

betas = np.zeros((ncloudcols,nfiles,ndirs))

for i_cld in range(ncloudcols):
	for i_file in range(nfiles):
		for i_dir in range(1,ndirs):
			plt.figure(1)
			# plt.plot(tbound_all_dirfil[1,i_file,i_dir],(totuflux_all_dirfil[nlayers,1,i_file,i_dir]-totuflux_all_dirfil[nlayers,1,i_file,i_dir-1])/(tbound_all_dirfil[1,i_file,i_dir]-tbound_all_dirfil[1,i_file,i_dir-1]),'o',c=colors[i_file])
			# plt.plot(tbound_all_dirfil[0,i_file,i_dir],(totuflux_all_dirfil[nlayers,0,i_file,i_dir]-totuflux_all_dirfil[nlayers,0,i_file,i_dir-1])/(tbound_all_dirfil[0,i_file,i_dir]-tbound_all_dirfil[0,i_file,i_dir-1]),'o',c='r')
			betas[i_cld,i_file,i_dir] = (totuflux_all_dirfil[nlayers,i_cld,i_file,i_dir]-totuflux_all_dirfil[nlayers,i_cld,i_file,i_dir-1])/(tbound_all_dirfil[i_cld,i_file,i_dir]-tbound_all_dirfil[i_cld,i_file,i_dir-1])

# plt.figure(1)
# for i_cld in range(ncloudcols):
# 	plt.semilogy(pz_master[])

# # for i_dir in range(1,ndirs):
# # 	plt.semilogx(wklfacs,1./betas[1,:,i_dir],'-o',label='tbound_fin='+str(tbound_all_dirfil[0,0,i_dir]))
# # 	plt.semilogx(wklfacs,1./betas[0,:,i_dir],'-o',label='tbound_furn='+str(tbound_all_dirfil[0,0,i_dir]))
# # 	plt.xlabel('H2O Factor')
# # 	plt.ylabel(r'$\alpha_{fin}$')
# # plt.legend()

# Z = betas[1,:,:].T

# # print wklfacs.shape, tbound_all_dirfil[1,0,:].shape

# X,Y = np.meshgrid(wklfacs,tbound_all_dirfil[1,0,:])

# print X.shape,Y.shape,Z.shape

# f = interpolate.interp2d(X,Y,Z)

# wklfacs_grid = np.logspace(-3,0,num=100,base=10.)
# tbound_grid=np.linspace(280,320,100)

# Z_int = np.zeros((len(wklfacs_grid),len(tbound_grid)))

# for i_w in range(len(wklfacs_grid)):
# 	for i_t in range(len(tbound_grid)):
# 		Z_int[i_w,i_t]=f(wklfacs_grid[i_w],tbound_grid[i_t])

# X,Y=np.meshgrid(wklfacs_grid,tbound_grid)

# plt.figure(1)
# plt.contourf(X,Y,Z_int)
# plt.gca().set_xscale('log')
# plt.xlabel('Fin H$_2$O factor')
# plt.ylabel('Fin surface temperature (K)')
# cb=plt.colorbar()
# cb.ax.set_xlabel(r'$\beta_{fin}$',)


# tboundsnje = [324,324,324,325,326,327,328,329,331,334]

# for i in range(nfiles):
# 	print f(wklfacs[i],tboundsnje[i])[0], ','

# plt.figure(1)
# # plt.contour(X,Y,Z,20)
# plt.imshow(Z)
# plt.gca().set_xscale('log')
# plt.colorbar()

# print tbound_all_dirfil[0,:,0]

# print (totuflux_all_dirfil[nlayers,1,:,1]-totuflux_all_dirfil[nlayers,1,:,0])/10.

# print np.mean(tbound_all_dirfil[:,:,1] - tbound_all_dirfil[:,:,0],axis=0)


show()