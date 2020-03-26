# RRTM LW and SW Python wrapper

import numpy as np
import os
import subprocess
import time
from subprocess import Popen, PIPE, STDOUT
import matplotlib.pyplot as plt
from pylab import *
import datetime

tstart = datetime.datetime.now()

project_dir = '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/'

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

def writeparams(params,f):	
	for i in params:
		f.write(str(i))
		f.write('\n')

def writeparamsarr(params,f):
	for param in params:
		for i in range(len(param)):
			f.write(str(param[i]))
			f.write('\n')

def writeinputfile_lw():
	f = open(project_dir+'LW/RRTM LW Input','w+')

	params = [iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,ireflect]
	writeparams(params,f)

	for i in range(len(semis)):
		f.write(str(semis[i]))
		f.write('\n')

	params = [iform,nlayers,nmol,secntk,cinp,ipthak]
	writeparams(params,f)

	params = [pavel,tavel,altz,pz,tz,wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
	writeparamsarr(params,f)

	f.close()

def writeinputfile_sw():
	f = open(project_dir+'SW/RRTM SW Input','w+')

	params = [iaer, iatm, iscat, istrm, iout, icld, idelm, icos,juldat,sza,isolvar,iemis,ireflect]
	writeparams(params,f)

	for i in range(len(semis)):
		f.write(str(semis[i]))
		f.write('\n')

	params = [iform,nlayers,nmol,secntk,cinp,ipthak]
	writeparams(params,f)

	params = [pavel,tavel,altz,pz,tz,wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
	writeparamsarr(params,f)

	f.close()

def callrrtmlw():
	loc = '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/LW/rrtmlw'
	os.chdir(project_dir+'/LW')
	# print(os.getcwd())  # Prints the current working directory
	p = subprocess.Popen([loc])
	stdoutdata, stderrdata = p.communicate()
	# print('return code = {}'.format(p.returncode))
	# print('------------------------------------------------------------------------------------------')
	# print

def callrrtmsw():
	loc = '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/SW/rrtmsw'
	os.chdir(project_dir+'/SW')
	p = subprocess.Popen([loc])
	stdoutdata, stderrdata = p.communicate()

def readrrtmoutput_lw():
	f=open('/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/LW/My Live Output RRTM')
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
	f=open('/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/SW/My Live Output RRTM')
	for i in range(0,nlayers+1):
		totuflux_sw[i] =  f.readline()
	for i in range(0,nlayers+1):
		totdflux_sw[i] =  f.readline()
	for i in range(0,nlayers+1):
		fnet_sw[i] =  f.readline()
	for i in range(0,nlayers+1):
		htr_sw[i] =  f.readline()

	return totuflux_sw,totdflux_sw,fnet_sw,htr_sw

def plotrrtmoutput():
	plt.figure(1)
	plt.subplot(331)
	logpplot(totuflux,pz,'totuflux','pz')
	# logpplot(totuflux_lw,pz,'totuflux','pz')
	# logpplot(totuflux_sw,pz,'totuflux','pz')
	plt.subplot(332)
	logpplot(totdflux,pz,'totdflux','pz')
	# logpplot(totdflux_lw,pz,'totdflux','pz')
	# logpplot(totdflux_sw,pz,'totdflux','pz')
	plt.subplot(333)
	logpplot(fnet,pz,'fnet','pz')
	# logpplot(fnet_lw,pz,'fnet','pz')
	# logpplot(fnet_sw,pz,'fnet','pz')
	plt.subplot(334)
	logpplot(htr[:-1],pz[:-1],'htr','pz')
	# logpplot(htr_lw[:-1],pz[:-1],'htr','pz')
	# logpplot(htr_sw[:-1],pz[:-1],'htr','pz')
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

def convection(T,z):
	for i in range(1,len(T)):
		dT = (T[i]-T[i-1])
		dz = (z[i]-z[i-1])/1000.
		if( -1.0 * dT/dz > lapse ):
			conv[i]=1.
			T[i] = T[i-1] - lapse * dz

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
semiss=np.ones(29)*0.9 	#all spectral bands the same as iemissm (surface, I think)
iform=1
nlayers=51
nmol=7
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

totuflux=np.zeros(nlayers+1)
totdflux=np.zeros(nlayers+1)
fnet=np.zeros(nlayers+1)
htr=np.zeros(nlayers+1)

pz=np.linspace(psurf,pmin,nlayers+1)
pavel=np.zeros(nlayers)
for i in range(len(pavel)):
	pavel[i]=(pz[i]+pz[i+1])/2.
tz=np.ones(nlayers+1) * tbound
tavel=np.zeros(nlayers)
for i in range(len(pavel)):
	tavel[i]=(tz[i]+tz[i+1])/2.
rsp=287.05
gravity=9.81
altz=np.zeros(nlayers+1)
altz[0] = 0.0
for i in range(1,nlayers):
	altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity
altz[nlayers] = altz[nlayers-1]+(pz[nlayers-1]-pz[nlayers])*rsp*tavel[nlayers-1]/pavel[nlayers-1]/gravity
tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
tz=np.clip(tz,tmin,tmax)
tavel=np.zeros(nlayers)
for i in range(nlayers):
	tavel[i]=(tz[i]+tz[i+1])/2.

tavel[nlayers-1] = tavel[nlayers-2]

# # Gas inventories
pin2 = 1.0 * 1e5 #convert the input in bar to Pa
pico2 = 400e-6* 1e5 #convert the input in bar to Pa
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

# # Gas volume mixing ratios
vol_mixco2 = molec_co2 / totmolec
vol_mixn2 = molec_n2 / totmolec
vol_mixo2 = molec_o2 / totmolec
vol_mixar = molec_ar / totmolec
vol_mixch4 = molec_ch4 / totmolec
vol_mixh2o = np.ones(nlayers) * molec_h2o / totmolec
vol_mixo3 = np.ones(nlayers) * molec_o3 / totmolec

surf_rh=0.8
esat_liq=np.zeros(nlayers)
rel_hum=np.zeros(nlayers)
vol_mixh2o_min = 1e-6
vol_mixh2o_max = 1e6
for i in range(nlayers):
	# h2o
	esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
	rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
	vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
	vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)

# Mean molecular weight of the atmosphere
mmwtot = mmwco2 * vol_mixco2 + mmwn2 * vol_mixn2 + mmwo2 * vol_mixo2 + mmwar*vol_mixar + mmwch4*vol_mixch4 + mmwh2o*vol_mixh2o[0]+mmwo3*vol_mixo3[0]

mperlayr = np.zeros(nlayers)
mperlayr_air = np.zeros(nlayers)

for i in range(nlayers):
	mperlayr[i] = totmolec/nlayers #Divide the molecules equally between layers
	mperlayr_air[i] = (molec_n2 + molec_o2)/(nlayers)


wbrodl = np.zeros(nlayers)
wkl = np.zeros((nmol+1,nlayers))

for i in range(nlayers):
	vol_mixo3[i] = (3.6478*(pz[i]**0.83209))*np.exp(-pz[i]/11.3515)*1e-6

#Set up mixing ratio of broadening molecules (N2 and O2 mostly)
for i in range(nlayers):
	wbrodl[i] = mperlayr_air[i] * 1.0e-4
	wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]
	wkl[2,i] = mperlayr[i] * 1.0e-4 * vol_mixco2
	wkl[3,i] = mperlayr[i] * 1.0e-4 * vol_mixo3[i]
	wkl[6,i] = mperlayr[i] * 1.0e-4 * vol_mixch4
	wkl[7,i] = mperlayr[i] * 1.0e-4 * vol_mixo2

# wbrodl=np.ones(nlayers) * 1e20
# wkl=np.zeros((nmol,nlayers))
# wkl[1,:] = 2e18*0.
# nxmol0=nmol #don't know what this is

# f = open('LW/RRTM LW Input','w+')

# params = [iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,ireflect]
# writeparams(params,f)

# for i in range(len(semis)):
# 	f.write(str(semis[i]))
# 	f.write('\n')

# params = [iform,nlayers,nmol,secntk,cinp,ipthak]
# writeparams(params,f)

# params = [pavel,tavel,altz,pz,tz,wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
# writeparamsarr(params,f)

# f.close()

ur_min=0.5
ur_max=1.0
eqb_maxhtr = 0.01
timesteps=100

cti=0

for ts in range(timesteps):
	if(cti+1 < nlayers-1):
		maxhtr=max(abs(htr[cti+1:nlayers-1]))
	else:
		maxhtr = 1.1*eqb_maxhtr
	# maxhtr=max(abs(htr[cti+1:nlayers-1]))
	if(ts>1):
		ur=maxhtr**2.0 * (nlayers/60.) + ur_min
		if(ur>ur_max):
			ur=ur_max
		# plt.figure(2)
		# plt.subplot(121)
		# plt.plot(ur,maxhtr,'o')
		# plt.xlabel('ur')
		# plt.ylabel('maxhtr')

		
		# plt.figure(2)
		# plt.subplot(122)
		# plt.plot(ts,maxhtr,'o')
		# plt.xlabel('ts')
		# plt.ylabel('maxhtr')

		tavel[1:] += htr[1:-1]/ur
		tavel=np.clip(tavel,tmin,tmax)
		for i in range(1,nlayers):
			tz[i] = (tavel[i-1] + tavel[i])/2.
		tz[nlayers] = 2*tavel[nlayers-1]-tz[nlayers-1]
		tz=np.clip(tz,tmin,tmax)

		for i in range(1,nlayers):
			altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity

		altavel = np.zeros(nlayers)
		for i in range(1,nlayers):
			altavel[i] = (altz[i-1]+altz[i])/2.0
		
		conv=np.zeros(nlayers)
		conv[0]=1
		convection(tavel,altavel)
		for i in range(1,nlayers):
			tz[i] = (tavel[i-1] + tavel[i])/2.
		tz[nlayers] = 2*tavel[nlayers-1]-tz[nlayers-1]
		tz=np.clip(tz,tmin,tmax)
		tavel=np.clip(tavel,tmin,tmax)
		convection(tz,altz)
		for i in range(1,nlayers):
			if(conv[i]==0):
				cti=i-1
				break

		#set up gas amounts

		surf_rh=0.8
		esat_liq=np.zeros(nlayers)
		rel_hum=np.zeros(nlayers)
		vol_mixh2o_min = 1e-6
		vol_mixh2o_max = 1e6
		for i in range(nlayers):
			# h2o
			esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
			rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
			vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
			vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
			wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]



	writeinputfile_lw()
	callrrtmlw()

	writeinputfile_sw()
	callrrtmsw()

	totuflux_lw=np.zeros(nlayers+1)
	totdflux_lw=np.zeros(nlayers+1)
	fnet_lw=np.zeros(nlayers+1)
	htr_lw=np.zeros(nlayers+1)

	totuflux_sw=np.zeros(nlayers+1)
	totdflux_sw=np.zeros(nlayers+1)
	fnet_sw=np.zeros(nlayers+1)
	htr_sw=np.zeros(nlayers+1)

	totuflux_lw,totdflux_lw,fnet_lw,htr_lw = readrrtmoutput_lw()
	totuflux_sw,totdflux_sw,fnet_sw,htr_sw = readrrtmoutput_sw()
	totuflux=totuflux_lw+totuflux_sw
	totdflux=totdflux_lw+totdflux_sw
	fnet=fnet_lw+fnet_sw
	htr=htr_lw+htr_sw

	if(cti+1 < nlayers-1):
		maxhtr=max(abs(htr[cti+1:nlayers-1]))
	else:
		maxhtr = 1.1*eqb_maxhtr
	print ts, maxhtr, cti
	if(maxhtr < eqb_maxhtr):
		plotrrtmoutput()
		print('Equilibrium reached!')


		break

	if(ts%50==2):
		plotrrtmoutput()

plotrrtmoutput()

# f = open('SW/RRTM SW Input','w+')

# params = [iaer, iatm, iscat, istrm, iout, icld, idelm, icos]
# writeparams(params,f)

# params = [juldat,sza,isolvar]
# writeparams(params,f)

# params = [iemiss,ireflect]
# writeparams(params,f)

# for i in range(len(semiss)):
# 	f.write(str(semiss[i]))
# 	f.write('\n')

# params = [iform,nlayers,nmol]
# writeparams(params,f)

# params = [secntk,cinp,ipthak]
# writeparams(params,f)

# params = [pavel,tavel,altz,pz,tz]
# writeparamsarr(params,f)

# params = [wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
# writeparamsarr(params,f)

# f.close()

tend = datetime.datetime.now()
ttotal = tend-tstart
print(ttotal)

print('Done')
plt.tight_layout()
show()