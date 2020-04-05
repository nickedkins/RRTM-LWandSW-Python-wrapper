# Plot RRTM Output (with new Python wrapper)

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from os import listdir

directories = [
# '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Current Output/'
'/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/_Useful Data/mls replication/'
]

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

def logpplot(x,p,xlab,ylab):
	plt.semilogy(x,p,'-')
	plt.ylim(max(p),min(p))
	plt.xlabel(xlab)
	plt.ylabel(ylab)

def plotrrtmoutput():
	plt.figure(1)
	plt.subplot(331)
	logpplot(totuflux,pz,'totuflux','pz')
	logpplot(totuflux_lw,pz,'totuflux','pz')
	logpplot(totuflux_sw,pz,'totuflux','pz')
	plt.subplot(332)
	logpplot(totdflux,pz,'totdflux','pz')
	logpplot(totdflux_lw,pz,'totdflux','pz')
	logpplot(totdflux_sw,pz,'totdflux','pz')
	plt.subplot(333)
	logpplot(fnet,pz,'fnet','pz')
	logpplot(fnet_lw,pz,'fnet','pz')
	logpplot(fnet_sw,pz,'fnet','pz')
	plt.subplot(334)
	logpplot(htr_lw[:-1]+htr_sw[:-1],pz[:-1],'htr','pz')
	logpplot(htr_lw[:-1],pz[:-1],'htr','pz')
	logpplot(htr_sw[:-1],pz[:-1],'htr','pz')
	plt.axvline(-eqb_maxhtr,ls='--')
	plt.axvline(eqb_maxhtr,ls='--')
	plt.subplot(335)
	logpplot(tz,pz,'tz','pz')
	plt.subplot(336)
	logpplot(wbrodl,pavel,'wbrodl','pavel')
	plt.subplot(337)
	logpplot(wkl[1,:],pavel,'wkl1 (h2o)','pavel')
	plt.subplot(338)
	logpplot(wkl[2,:],pavel,'wkl2 (co2)','pavel')
	plt.subplot(339)
	logpplot(wkl[3,:],pavel,'wkl3 (o3)','pavel')

def readrrtmoutput(fn):
	f = open(fn)

nlayers=51
nmol=7

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
tbound = 288 #surface temperature (K)
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
semiss=np.ones(29)*0.7 	#all spectral bands the same as iemissm (surface, I think)
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
eqb_maxhtr = 0.01
timesteps=1
cti=0
maxhtr=0.

pz=np.linspace(psurf,pmin,nlayers+1)
totuflux=np.zeros(nlayers+1)
totdflux=np.zeros(nlayers+1)
fnet=np.zeros(nlayers+1)
htr=np.zeros(nlayers+1)
pavel=np.zeros(nlayers)
tz=np.ones(nlayers+1) * tbound
altz=np.zeros(nlayers+1)
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
conv=np.zeros(nlayers)
altavel = np.zeros(nlayers)
vol_mixh2o = np.ones(nlayers) * molec_h2o / totmolec
vol_mixo3 = np.ones(nlayers) * molec_o3 / totmolec

params0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr]
params1d=[semis,semiss,totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,pavel,tz,tavel,altz,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel,]
params2d=[wkl]



for directory in directories:

	filenames = []
	dir_label = directory.split('/')[-2]
	print(dir_label)
	a = sorted(listdir(directory))
	filenames.append(a)
	if('.DS_Store' in a):
		a.remove('.DS_Store')

	for fn in a:
		print(fn)
		output_file = directory + fn
		f=open(output_file,'r')
		for x in params0d:
			x = f.readline()

		for x in params1d:
			for i in range(shape(x)[0]):
				x[i] = f.readline()
		for x in params2d:
			for i in range(shape(x)[0]):
				for j in range(shape(x)[1]):
					x[i,j] = f.readline()

		plotrrtmoutput()
for i in range(nlayers):
	print '{},{},{},{},{},{},{},{},{},{},{},{},{},{},'.format(pz[i],pavel[i],altz[i]/1000.,tz[i],tavel[i],totuflux[i],totuflux_lw[i],totuflux_sw[i],totdflux[i],totdflux_lw[i],totdflux_sw[i],fnet[i],fnet_lw[i],fnet_sw[i])

print '{},{},{},{},{},{},{},{},{},{},{},{},{},{},'.format(pz[nlayers],'na',altz[nlayers]/1000.,tz[nlayers],'na',totuflux[nlayers],totuflux_lw[nlayers],totuflux_sw[nlayers],totdflux[nlayers],totdflux_lw[nlayers],totdflux_sw[nlayers],fnet[nlayers],fnet_lw[nlayers],fnet_sw[nlayers])

show()