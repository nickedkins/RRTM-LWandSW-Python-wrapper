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

def logpplot(x,p,xlab,ylab,color='blue'):
	plt.semilogy(x,p,'-',c=color)
	plt.ylim(max(p),min(p))
	plt.xlabel(xlab)
	plt.ylabel(ylab)

def loglogplot(x,p,xlab,ylab):
	plt.loglog(x,p,'-')
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

def writeformattedinputfile_sw():
	f=open('/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/SW/Input RRTM SW NJE Formatted','w+')
	f.write('INPUT_RRTM_SW NJE created\n')

	f.write('0        1         2         3         4         5         6         7         8         9\n')
	f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
	f.write('$ Dynamic Formatted RRTM SW Input\n')
	f.write('                  {:2d}                             {:1d}                                {:1d} {:1d}   {:02d}    {:1d}   {:1d}{:1d}\n'.format(iaer,iatm,iscat,istrm,iout,icld,idelm,icos))
	f.write('            {:3d}   {:7.3f}    {:1d}\n'.format(juldat,sza,isolvar))
	f.write('           {:1d}  {:1d}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}\n'.format(iemis,ireflect,semiss[15],semiss[16],semiss[17],semiss[18],semiss[19],semiss[20],semiss[21],semiss[22],semiss[23],semiss[24],semiss[25],semiss[26],semiss[27],semiss[28] ))
	f.write(' {:1d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
	f.write('{:11.4f}{:14.2f}{:10s}{:3s}{:2d}{:8.3f}{:8.2f}{:7.2f}{:7.3f}{:8.2f}{:7.2f}\n'.format(pavel[0],tavel[0],secntk,cinp,ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
	f.write('{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
	for i in range(2,nlayers+1):
		f.write('{:11.4f}{:14.2f}{:15.0f}{:30.3f}{:8.3f}{:7.2f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		f.write('{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
	f.write('%%%%%\n')
	f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
	f.close()

def writeformattedinputfile_lw():
	f=open('/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/LW/Input RRTM LW NJE Formatted','w+')
	f.write('TAPE5 FOR MLS\n')
	f.write('0        1         2         3         4         5         6         7         8         9\n')
	f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')
	f.write('$ STANDARD MID-LATITUDE SUMMER ATMOSPHERE\n')
	f.write('{:50d}{:20d}{:13d}{:2d}{:5d}{:5d}\n'.format(iatm,ixsect,iscat,numangs,iout,icld))
	# print type(semiss[0:16])
	f.write('{:6.1f} {:1d}  {:1d}\n'.format(tbound,iemiss,ireflect))
	# for i in range(16):
	# 	f.write('{:5.3f}'.format(semiss[i]))
	f.write('{:2d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
	f.write('{:11.4f}{:14.2f}{:15d}{:8.3f}{:8.2f}{:7.2f}{:7.3f}{:8.2f}{:7.2f}\n'.format(pavel[0],tavel[0],ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
	f.write('{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
	for i in range(2,nlayers+1):
		f.write('{:11.4f}{:14.2f}{:15.0f}{:30.3f}{:8.3f}{:7.2f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		f.write('{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
	f.write('%%%%%\n')
	f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')

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
	logpplot(totuflux_lw,pz,'totuflux','pz','red')
	logpplot(totuflux_sw,pz,'totuflux','pz','green')
	plt.subplot(332)
	logpplot(totdflux,pz,'totdflux','pz')
	logpplot(totdflux_lw,pz,'totdflux','pz','red')
	logpplot(totdflux_sw,pz,'totdflux','pz','green')
	plt.subplot(333)
	logpplot(fnet,pz,'fnet','pz')
	logpplot(fnet_lw,pz,'fnet','pz','red')
	logpplot(fnet_sw,pz,'fnet','pz','green')
	plt.subplot(334)
	logpplot(htr[:-1],pz[:-1],'htr','pz')
	logpplot(htr_lw[:-1],pz[:-1],'htr','pz','red')
	logpplot(htr_sw[:-1],pz[:-1],'htr','pz','green')
	plt.axvline(-eqb_maxhtr,ls='--')
	plt.axvline(eqb_maxhtr,ls='--')
	plt.subplot(335)
	logpplot(tz,pz,'tz','pz')
	# plt.plot(tz,altz/1000.)
	plt.subplot(336)
	logpplot(wbrodl,pavel,'wbrodl','pavel')
	plt.subplot(337)
	logpplot(wkl[1,:],pavel,'wkl1 (h2o)','pavel')
	plt.subplot(338)
	loglogplot(wkl[2,:],pavel,'wkl2 (co2)','pavel')
	plt.subplot(339)
	logpplot(wkl[3,:],pavel,'wkl3 (o3)','pavel')

def convection(T,z):
	for i in range(1,len(T)):
		dT = (T[i]-T[i-1])
		dz = (z[i]-z[i-1])/1000.
		if( -1.0 * dT/dz > lapse or z[i]/1000. < -1 ):
			conv[i]=1.
			T[i] = T[i-1] - lapse * dz

def writeoutputfile():
	tlabel = datetime.datetime.now()
	tlabelstr = str(tlabel.strftime('%Y_%m_%d %H_%M_%S'))
	f = open(project_dir+'_Raw Output Data/'+tlabelstr,'w+')
	for x in params0d:
		f.write(str(x))
		f.write('\n')
	for x in params1d:
		for i in range(shape(x)[0]):
			f.write(str(x[i]))
			f.write('\n')
	for x in params2d:
		for i in range(shape(x)[0]):
			for j in range(shape(x)[1]):
				f.write(str(x[i,j]))
				f.write('\n')

#Declare variables

nlayers=51
nmol=7

gravity=9.81
avogadro=6.022e23
iatm=0 #0 for layer values, 1 for level values
ixsect=0 #could be 1, but why?
iscat=0 #just absorption and emission
# numangs=0 #can be 0-4 for higher precision
numangs=3 #can be 0-4 for higher precision 3 for SW, 0 for LW
# iout=99 #for info in all spectral bands
iout=0 #for broadband only
# iout=-1 #for broadband, no printings
# iout=29
icld=0 #for clear sky
# icld=1  #for grey clouds
tbound = 288. #surface temperature (K)
iemiss=1 #surface emissivity. Keep this fixed for now.
iemis=2
ireflect=0 #for Lambert reflection
iaer=0 #0=aerosols off, 1=on
istrm=1 			# ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0). 
						#0=4 streams
						#1=8 streams
idelm=1 			# flag for outputting downwelling fluxes computed using the delta-M scaling approximation. 0=output "true" direct and diffuse downwelling fluxes, 1=output direct and diffuse downwelling fluxes computed with delta-M approximation
icos=0 				#0:there is no need to account for instrumental cosine response, 1:to account for instrumental cosine response in the computation of the direct and diffuse fluxes, 2:2 to account for instrumental cosine response in the computation of the diffuse fluxes only
semis=np.ones(16)*1.	#all spectral bands the same as iemissm (maybe this is the surface??)
semiss=np.ones(29)*1. 	#surface emissivity
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
iform=1
psurf=1000.
pmin=0.
# secntk=0
# cinp=1.356316e-19
secntk='' #based on not appearing in input mls sw
cinp='' #based on not appearing in input mls sw
ipthak=3
ipthrk=3
juldat=0 		#Julian day associated with calculation (1-365/366 starting January 1). Used to calculate Earth distance from sun. A value of 0 (default) indicates no scaling of solar source function using earth-sun distance.
sza=45. 			#Solar zenith angle in degrees (0 deg is overhead).
isolvar=0 		#= 0 each band uses standard solar source function, corresponding to present day conditions. 
				#= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
				#= 2 scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))			
lapse=100
tmin=10.
tmax=350.
rsp=287.05
gravity=9.81
filewritten=0

totuflux=np.zeros(nlayers+1)
totdflux=np.zeros(nlayers+1)
fnet=np.zeros(nlayers+1)
htr=np.zeros(nlayers+1)
pavel=np.zeros(nlayers)
tz=np.ones(nlayers+1) * tbound
altz=np.zeros(nlayers+1)
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

ur=np.ones(nlayers-1)

# pz=np.linspace(psurf,pmin,nlayers+1)
# pz=np.logspace(3.,-1.,base=10.,num=nlayers+1)
pz=np.array([
1013	,
891.46	,
792.287	,
718.704	,
651.552	,
589.841	,
532.986	,
480.526	,
437.556	,
398.085	,
361.862	,
328.507	,
297.469	,
269.015	,
243	,
218.668	,
196.44	,
162.913	,
136.511	,
114.564	,
96.4903	,
81.2	,
68.4286	,
57.6936	,
48.6904	,
40.5354	,
33.733	,
28.1201	,
23.1557	,
18.7914	,
15.0693	,
11.8006	,
8.78628	,
6.61328	,
5.03469	,
3.85333	,
2.96408	,
2.2918	,
1.78227	,
1.339	,
0.589399	,
0.430705	,
0.333645	,
0.261262	,
0.216491	,
0.179393	,
0.148652	,
0.1255	,
0.106885	,
0.091031	,
0.077529	,
0.067	,
	])

wkl[1,:] = np.array([
1.59E-02	,
1.12E-02	,
7.68E-03	,
5.27E-03	,
3.63E-03	,
2.39E-03	,
1.71E-03	,
1.27E-03	,
9.57E-04	,
6.97E-04	,
5.08E-04	,
3.66E-04	,
2.50E-04	,
1.36E-04	,
6.55E-05	,
2.84E-05	,
9.70E-06	,
4.82E-06	,
3.43E-06	,
3.27E-06	,
3.18E-06	,
3.18E-06	,
3.26E-06	,
3.41E-06	,
3.59E-06	,
3.85E-06	,
4.06E-06	,
4.25E-06	,
4.39E-06	,
4.53E-06	,
4.68E-06	,
4.81E-06	,
4.90E-06	,
4.97E-06	,
5.02E-06	,
5.10E-06	,
5.25E-06	,
5.38E-06	,
5.47E-06	,
5.47E-06	,
5.33E-06	,
5.18E-06	,
5.05E-06	,
4.88E-06	,
4.71E-06	,
4.54E-06	,
4.38E-06	,
4.22E-06	,
4.06E-06	,
3.91E-06	,
3.77E-06	,
	])

wkl[2,:] = np.array([
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
3.55E-04	,
	])

wkl[3,:] = np.array([
3.19E-08	,
3.55E-08	,
3.95E-08	,
4.39E-08	,
4.89E-08	,
5.44E-08	,
6.13E-08	,
6.99E-08	,
7.95E-08	,
8.92E-08	,
1.02E-07	,
1.16E-07	,
1.31E-07	,
1.60E-07	,
1.94E-07	,
2.28E-07	,
3.04E-07	,
4.40E-07	,
5.24E-07	,
6.32E-07	,
8.23E-07	,
1.25E-06	,
1.80E-06	,
2.29E-06	,
2.83E-06	,
3.45E-06	,
4.22E-06	,
5.03E-06	,
5.68E-06	,
6.31E-06	,
6.96E-06	,
7.77E-06	,
8.52E-06	,
8.83E-06	,
8.49E-06	,
7.56E-06	,
6.30E-06	,
5.10E-06	,
4.08E-06	,
2.82E-06	,
1.80E-06	,
1.55E-06	,
1.36E-06	,
1.18E-06	,
1.03E-06	,
9.02E-07	,
7.88E-07	,
6.75E-07	,
5.80E-07	,
4.98E-07	,
4.30E-07	,
	])

wkl[4,:] = np.array([
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.20E-07	,
3.18E-07	,
3.15E-07	,
3.10E-07	,
3.03E-07	,
2.97E-07	,
2.93E-07	,
2.87E-07	,
2.79E-07	,
2.70E-07	,
2.55E-07	,
2.31E-07	,
2.00E-07	,
1.69E-07	,
1.40E-07	,
1.17E-07	,
1.03E-07	,
9.44E-08	,
8.76E-08	,
8.24E-08	,
7.56E-08	,
6.70E-08	,
5.42E-08	,
4.24E-08	,
3.26E-08	,
2.40E-08	,
1.78E-08	,
1.29E-08	,
9.31E-09	,
6.67E-09	,
3.59E-09	,
2.03E-09	,
1.70E-09	,
1.47E-09	,
1.32E-09	,
1.20E-09	,
1.10E-09	,
1.02E-09	,
9.55E-10	,
9.00E-10	,
8.48E-10	,
8.00E-10	,
	])

wkl[5,:] = np.array([
1.47E-07	,
1.42E-07	,
1.37E-07	,
1.34E-07	,
1.31E-07	,
1.30E-07	,
1.29E-07	,
1.27E-07	,
1.24E-07	,
1.19E-07	,
1.14E-07	,
1.07E-07	,
1.00E-07	,
9.32E-08	,
8.56E-08	,
7.72E-08	,
6.39E-08	,
4.88E-08	,
3.73E-08	,
2.87E-08	,
2.25E-08	,
1.74E-08	,
1.41E-08	,
1.26E-08	,
1.24E-08	,
1.32E-08	,
1.44E-08	,
1.56E-08	,
1.67E-08	,
1.81E-08	,
1.98E-08	,
2.19E-08	,
2.38E-08	,
2.56E-08	,
2.75E-08	,
2.94E-08	,
3.09E-08	,
3.23E-08	,
3.38E-08	,
3.65E-08	,
3.96E-08	,
4.27E-08	,
4.57E-08	,
4.98E-08	,
5.44E-08	,
5.94E-08	,
6.52E-08	,
7.46E-08	,
8.53E-08	,
9.76E-08	,
1.11E-07	,
	])

wkl[6,:] = np.array([
1.70E-06	,
1.70E-06	,
1.70E-06	,
1.70E-06	,
1.70E-06	,
1.69E-06	,
1.68E-06	,
1.66E-06	,
1.65E-06	,
1.63E-06	,
1.62E-06	,
1.61E-06	,
1.58E-06	,
1.56E-06	,
1.53E-06	,
1.51E-06	,
1.48E-06	,
1.45E-06	,
1.42E-06	,
1.38E-06	,
1.34E-06	,
1.30E-06	,
1.25E-06	,
1.17E-06	,
1.08E-06	,
9.65E-07	,
8.54E-07	,
7.71E-07	,
7.25E-07	,
6.80E-07	,
6.34E-07	,
5.79E-07	,
5.27E-07	,
4.82E-07	,
4.38E-07	,
3.95E-07	,
3.52E-07	,
3.10E-07	,
2.67E-07	,
2.01E-07	,
1.59E-07	,
1.54E-07	,
1.51E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
1.50E-07	,
	])

wkl[7,:] = np.array([
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
2.09E-01	,
	])

wbrodl = np.array([
2.02E+24	,
1.66E+24	,
1.22E+24	,
1.13E+24	,
1.04E+24	,
9.57E+23	,
8.81E+23	,
7.11E+23	,
6.59E+23	,
6.11E+23	,
5.65E+23	,
5.22E+23	,
4.81E+23	,
4.44E+23	,
4.08E+23	,
3.74E+23	,
5.68E+23	,
4.36E+23	,
3.65E+23	,
3.07E+23	,
2.58E+23	,
2.16E+23	,
1.81E+23	,
1.52E+23	,
1.38E+23	,
1.14E+23	,
9.44E+22	,
8.43E+22	,
7.36E+22	,
6.29E+22	,
5.56E+22	,
5.18E+22	,
3.61E+22	,
2.68E+22	,
2.00E+22	,
1.51E+22	,
1.14E+22	,
8.69E+21	,
7.51E+21	,
1.28E+22	,
2.70E+21	,
1.65E+21	,
1.25E+21	,
7.53E+20	,
6.35E+20	,
5.37E+20	,
3.93E+20	,
3.11E+20	,
2.71E+20	,
2.35E+20	,
1.87E+20	,
	])

pavel=np.array([
952.1147	,
841.897	,
755.3917	,
685.0609	,
620.7571	,
561.5159	,
506.7787	,
458.9778	,
417.8179	,
379.9846	,
345.1331	,
313	,
283.2681	,
255.9648	,
230.793	,
207.5901	,
179.6777	,
149.8259	,
125.467	,
105.5072	,
88.85838	,
74.81903	,
63.06029	,
53.19867	,
44.59128	,
37.16316	,
30.91292	,
25.6397	,
20.97451	,
16.9346	,
13.41941	,
10.30125	,
7.703475	,
5.824757	,
4.442682	,
3.407392	,
2.627624	,
2.037819	,
1.56118	,
0.9634139	,
0.5106084	,
0.3820259	,
0.2975729	,
0.2388066	,
0.1978831	,
0.1639725	,
0.1372726	,
0.1161604	,
9.89E-02	,
8.43E-02	,
7.22E-02	,
	])

tavel=np.array([
2.92E+02	,
287.03	,
282.23	,
277.43	,
272.63	,
267.83	,
263.03	,
258.3	,
253.75	,
249.2	,
244.65	,
240.13	,
235.64	,
231.1	,
226.55	,
222.01	,
216.81	,
215.71	,
215.7	,
215.7	,
216.18	,
217.39	,
218.72	,
220.08	,
221.46	,
222.88	,
224.24	,
225.81	,
227.61	,
230.17	,
233.52	,
237.51	,
242.34	,
247.27	,
252.17	,
257.13	,
262.09	,
267.05	,
272	,
274.41	,
268.77	,
263.53	,
258.75	,
253.76	,
249	,
244.24	,
239.61	,
234.65	,
229.81	,
224.97	,
220.34	,
	])

altz=np.array([
0.00E+00	,
1.1,
2.1	,
2.9	,
3.7	,
4.5	,
5.3	,
6.1	,
6.8	,
7.5	,
8.2	,
8.9	,
9.6	,
10.3	,
11	,
11.7	,
12.4	,
13.6	,
14.7	,
15.8	,
16.9	,
18	,
19.1	,
20.2	,
21.3	,
22.5	,
23.7	,
24.9	,
26.2	,
27.6	,
29.1	,
30.8	,
32.9	,
34.9	,
36.9	,
38.9	,
40.9	,
42.9	,
44.9	,
47.2	,
53.9	,
56.4	,
58.4	,
60.3	,
61.7	,
63.1	,
64.5	,
65.7	,
66.8	,
67.9	,
69	,
70	,
	])*1000.

tz=np.array([
294.2	,
289.25,
284.6	,
279.8	,
275	,
270.2	,
265.4	,
260.55	,
256	,
251.45	,
246.9	,
242.35	,
237.86	,
233.35	,
228.8	,
224.25	,
219.7	,
215.74	,
215.7	,
215.7	,
215.7	,
216.8	,
218.03	,
219.44	,
220.76	,
222.2	,
223.57	,
224.98	,
226.71	,
228.66	,
231.81	,
235.4	,
239.99	,
244.95	,
249.84	,
254.77	,
259.73	,
264.69	,
269.65	,
274.56	,
270.71	,
265.88	,
261	,
256.08	,
251.32	,
246.56	,
241.8	,
237.02	,
232.18	,
227.34	,
222.5	,
218.1	,
	])


# for i in range(len(pavel)):
# 	pavel[i]=(pz[i]+pz[i+1])/2.
# tavel=np.zeros(nlayers)
# for i in range(len(pavel)):
# 	tavel[i]=(tz[i]+tz[i+1])/2.
# altz[0] = 0.0
# for i in range(1,nlayers):
# 	altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity
# altz[nlayers] = altz[nlayers-1]+(pz[nlayers-1]-pz[nlayers])*rsp*tavel[nlayers-1]/pavel[nlayers-1]/gravity
# tz=np.ones(nlayers+1) * tbound-lapse*altz/1000.
# tz=np.clip(tz,tmin,tmax)
# for i in range(nlayers):
# 	tavel[i]=(tz[i]+tz[i+1])/2.

# tavel[nlayers-1] = tavel[nlayers-2]

# # Gas inventories
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
for i in range(nlayers):
	# h2o
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
# for i in range(nlayers):
# 	wbrodl[i] = mperlayr_air[i] * 1.0e-4
# 	wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]*0.
# 	# wkl[2,i] = mperlayr[i] * 1.0e-4 * vol_mixco2
# 	wkl[2,i] = mperlayr[i] * 1.0e-4 * 400e-6
# 	wkl[3,i] = mperlayr[i] * 1.0e-4 * vol_mixo3[i]*0.
# 	wkl[6,i] = mperlayr[i] * 1.0e-4 * vol_mixch4*0.
# 	wkl[7,i] = mperlayr[i] * 1.0e-4 * vol_mixo2*0.

# wkl = np.clip(wkl,1.,1e63)

ur_min=0.5
ur_max=1.5
eqb_maxhtr = 0.01
timesteps=100

cti=0

surf_rh=0.8
vol_mixh2o_min = 1e-6
vol_mixh2o_max = 1e6

esat_liq=np.zeros(nlayers)
rel_hum=np.zeros(nlayers)

maxhtr=0.



params0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr]
params1d=[semis,semiss,totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,pavel,tz,tavel,altz,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel]
params2d=[wkl]


for ts in range(timesteps):
	if(cti+1 < nlayers-1):
		maxhtr=max(abs(htr[cti+1:nlayers-1]))
	else:
		maxhtr = 1.1*eqb_maxhtr
	# maxhtr=max(abs(htr[cti+1:nlayers-1]))
	if(ts>0):
		for i in range(1,nlayers-1):
			ur[i]=maxhtr**2.0 * pz[0]/(pz[i-1]-pz[i]) + ur_min
			if(ur[i]>ur_max):
				ur[i]=ur_max
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
		# tz=np.clip(tz,tmin,tmax)

		# for i in range(1,nlayers):
		# 	altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity

		
		for i in range(1,nlayers):
			altavel[i] = (altz[i-1]+altz[i])/2.0
		
		conv=np.zeros(nlayers+1) #reset to zero
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

		
		for i in range(nlayers):
			# h2o
			esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
			rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
			vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
			if(i>1 and vol_mixh2o[i] > vol_mixh2o[i-1]):
				vol_mixh2o[i]=vol_mixh2o[i-1]
			vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
			# wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]*0.

	writeformattedinputfile_lw()
	# writeinputfile_lw()
	callrrtmlw()

	writeformattedinputfile_sw()
	# # writeinputfile_sw()
	callrrtmsw()


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

	if(ts%20==2):
		plotrrtmoutput()

	print '{:16.8f} {:16.8f} {:16.8f} {:16.8f} '.format(htr[nlayers-1],tavel[nlayers-1],tavel[nlayers-2],fnet[nlayers-1]-fnet[nlayers-2])
	print ts, maxhtr, cti

	params0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr]
	params1d=[semis,semiss,totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,pavel,tz,tavel,altz,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel]
	params2d=[wkl]

	if(maxhtr < eqb_maxhtr):
		plotrrtmoutput()
		print('Equilibrium reached!')
		writeoutputfile()
		filewritten=1
		break
	elif(ts==timesteps-1):
		print('Max timesteps')
		writeoutputfile()
		filewritten=1
	
plotrrtmoutput()
if(filewritten!=1):
	params0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr]
	params1d=[semis,semiss,totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,pavel,tz,tavel,altz,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel]
	params2d=[wkl]
	writeoutputfile()
	

tend = datetime.datetime.now()
ttotal = tend-tstart
print(ttotal)

print('Done')
# plt.tight_layout()
show()