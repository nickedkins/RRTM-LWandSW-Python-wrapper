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

tstart = datetime.datetime.now()

project_dir = '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/'

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
	# plt.semilogy(x,p,'-',c=color)
	plt.semilogy(x,p,'-',c=color,alpha=(float(ts)/timesteps)**1.0)
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
	# f.write('            {:3d}   {:7.3f}    {:1d}\n'.format(juldat,sza,isolvar))
	f.write('            {:3d}   {:7.3f} \n'.format(juldat,sza))
	f.write('           {:1d}  {:1d}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}{:5.3f}\n'.format(iemis,ireflect,semiss[15],semiss[16],semiss[17],semiss[18],semiss[19],semiss[20],semiss[21],semiss[22],semiss[23],semiss[24],semiss[25],semiss[26],semiss[27],semiss[28] ))
	f.write(' {:1d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
	f.write('{:11.4f}{:17.5f}{:10s}{:3s}{:2d}{:8.3f}{:8.2f}{:7.2f}{:7.3f}{:8.2f}{:10.5f}\n'.format(pavel[0],tavel[0],secntk,cinp,ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
	f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
	for i in range(2,nlayers+1):
		if(pavel[i-1]<0.1):
			f.write('  {:13.7E}{:13.5f}{:15.0f}{:30.3f}{:0<08.8g}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		elif(pavel[i-1]<1.):
			f.write('  {:0<08.7g}{:17.5f}{:15.0f}{:30.3f}{:0<08.8g}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		else:
			f.write('   {:0<08.8g}{:17.5f}{:15.0f}{:30.3f} {:0<07.7g}{:10.5f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
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
	f.write('{:16.11f} {:1d}  {:1d}\n'.format(tbound,iemiss,ireflect)) #add semis read here?
	f.write('{:2d}{:3d}{:5d}  1.000000MIDLATITUDE SUMM H1=    0.00 H2=   70.00 ANG=   0.000 LEN= 0\n'.format(iform,nlayers,nmol))
	f.write('{:21.14f}{:25.13f}{:15d}{:18.13f}{:18.12f}{:17.12f}{:17.13f}{:18.12f}{:17.12f}\n'.format(pavel[0],tavel[0],ipthak,altz[0]/1000.,pz[0],tz[0],altz[1]/1000.,pz[1],tz[1]))
	f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,0],wkl[2,0],wkl[3,0],wkl[4,0],wkl[5,0],wkl[6,0],wkl[7,0],wbrodl[0] ))
	for i in range(2,nlayers+1):
		f.write('{:21.14f}{:25.13f}{:15.0f}{:41.14f}{:18.13f}{:17.12f}\n'.format(pavel[i-1],tavel[i-1],ipthrk,altz[i]/1000.,pz[i],tz[i]))
		f.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(wkl[1,i-1],wkl[2,i-1],wkl[3,i-1],wkl[4,i-1],wkl[5,i-1],wkl[6,i-1],wkl[7,i-1],wbrodl[i-1] ))
	f.write('%%%%%\n')
	f.write('123456789-123456789-123456789-123456789-123456789-123456789-123456789-123456789-\n')

def callrrtmlw():
	loc = '/Users/nickedkins/Dropbox/GitHub Repositories/RRTM-LWandSW-Python-wrapper/LW/rrtmlw'
	os.chdir(project_dir+'/LW')
	p = subprocess.Popen([loc])
	stdoutdata, stderrdata = p.communicate()

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
	# logpplot(totuflux_lw,pz,'totuflux','pz','red')
	# logpplot(totuflux_sw,pz,'totuflux','pz','green')
	plt.subplot(332)
	logpplot(totdflux,pz,'totdflux','pz')
	logpplot(totdflux_lw,pz,'totdflux','pz','red')
	logpplot(totdflux_sw,pz,'totdflux','pz','green')
	plt.subplot(333)
	logpplot(dfnet,pz[:-1],'fnet','pz')
	# logpplot(fnet_lw,pz,'fnet','pz','red')
	# logpplot(fnet_sw,pz,'fnet','pz','green')
	plt.subplot(334)
	logpplot(htr[:-1],pz[:-1],'htr','pz')
	plt.semilogy(-4.*dfnet/dpz,pz[:-1])
	# logpplot(htr_lw[:-1],pz[:-1],'htr','pz','red')
	# logpplot(htr_sw[:-1],pz[:-1],'htr','pz','green')
	# plt.xlim(-2,2)
	plt.axvline(-eqb_maxhtr,ls='--')
	plt.axvline(eqb_maxhtr,ls='--')
	plt.subplot(335)
	logpplot(tavel,pavel,'tz','pz')
	logpplot(tz,pz,'tz','pz')
	plt.plot(tbound,pz[0],'o')
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
	# print T[0]
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
lw_on=1
sw_on=1
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
tbound=288.123456789123456789 #surface temperature (K)
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
# semiss[15:29] = np.array([
# 0.881,
# 0.794,
# 0.738,
# 0.727,
# 0.657,
# 0.660,
# 0.626,
# 0.655,
# 0.791,
# 0.883,
# 0.957,
# 0.958,
# 0.958,
# 0.970
# ])
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
# sza=65. 			#Solar zenith angle in degrees (0 deg is overhead).
sza=45. #(RD repl) 			#Solar zenith angle in degrees (0 deg is overhead).
isolvar=0 		#= 0 each band uses standard solar source function, corresponding to present day conditions. 
				#= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
				#= 2 scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))			
lapse=1000
tmin=10.
tmax=1000.
rsp=287.05
gravity=9.81
filewritten=0
sw_freq=100

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

ur=np.ones(nlayers)

# pz=np.linspace(psurf,pmin,nlayers+1)
# pz=np.logspace(3.,-1.,base=10.,num=nlayers+1)
pz=np.array([
1013.000000	,
891.460000	,
792.287000	,
718.704000	,
651.552000	,
589.841000	,
532.986000	,
480.526000	,
437.556000	,
398.085000	,
361.862000	,
328.507000	,
297.469000	,
269.015000	,
243.001000	,
218.668000	,
196.440000	,
162.913000	,
136.511000	,
114.564000	,
96.490300	,
81.200000	,
68.428600	,
57.693600	,
48.690400	,
40.535400	,
33.733000	,
28.120100	,
23.155700	,
18.791400	,
15.069300	,
11.800600	,
8.786280	,
6.613280	,
5.034690	,
3.853330	,
2.964080	,
2.291800	,
1.782270	,
1.339000	,
0.589399	,
0.430705	,
0.333645	,
0.261262	,
0.216491	,
0.179393	,
0.148652	,
0.125500	,
0.106885	,
0.091031	,
0.077529	,
0.067000	,
	])

wkl[1,:] = np.array([
1.5946558E-02	,
1.1230157E-02	,
7.6751928E-03	,
5.2688639E-03	,
3.6297729E-03	,
2.3900282E-03	,
1.7066754E-03	,
1.2718296E-03	,
9.5655693E-04	,
6.9666741E-04	,
5.0829613E-04	,
3.6584702E-04	,
2.4977655E-04	,
1.3636267E-04	,
6.5472166E-05	,
2.8419665E-05	,
9.6973117E-06	,
4.8207025E-06	,
3.4318521E-06	,
3.2663258E-06	,
3.1784930E-06	,
3.1768304E-06	,
3.2639416E-06	,
3.4095149E-06	,
3.5909502E-06	,
3.8500998E-06	,
4.0575464E-06	,
4.2513630E-06	,
4.3863338E-06	,
4.5309193E-06	,
4.6839027E-06	,
4.8067850E-06	,
4.9039072E-06	,
4.9670398E-06	,
5.0161370E-06	,
5.1013058E-06	,
5.2471341E-06	,
5.3810127E-06	,
5.4697343E-06	,
5.4735615E-06	,
5.3326530E-06	,
5.1831207E-06	,
5.0460312E-06	,
4.8780507E-06	,
4.7075605E-06	,
4.5413699E-06	,
4.3837813E-06	,
4.2189254E-06	,
4.0623413E-06	,
3.9098322E-06	,
3.7676771E-06	,
	])

wkl[2,:] = np.array([
3.5495765E-04	,
3.5495876E-04	,
3.5494528E-04	,
3.5497753E-04	,
3.5498614E-04	,
3.5499220E-04	,
3.5500020E-04	,
3.5500637E-04	,
3.5500794E-04	,
3.5500876E-04	,
3.5501100E-04	,
3.5501071E-04	,
3.5501161E-04	,
3.5500948E-04	,
3.5501219E-04	,
3.5501563E-04	,
3.5500890E-04	,
3.5499944E-04	,
3.5499918E-04	,
3.5499939E-04	,
3.5499930E-04	,
3.5499941E-04	,
3.5499973E-04	,
3.5499971E-04	,
3.5500005E-04	,
3.5499997E-04	,
3.5499965E-04	,
3.5500378E-04	,
3.5500416E-04	,
3.5501141E-04	,
3.5500969E-04	,
3.5500916E-04	,
3.5500980E-04	,
3.5500823E-04	,
3.5501088E-04	,
3.5501190E-04	,
3.5501042E-04	,
3.5500791E-04	,
3.5500259E-04	,
3.5500337E-04	,
3.5502302E-04	,
3.5507922E-04	,
3.5503952E-04	,
3.5511289E-04	,
3.5519004E-04	,
3.5513769E-04	,
3.5506955E-04	,
3.5528085E-04	,
3.5538705E-04	,
3.5533530E-04	,
3.5513588E-04	,
	])

wkl[3,:] = np.array([
3.1872162E-08	,
3.5456235E-08	,
3.9477314E-08	,
4.3921091E-08	,
4.8850310E-08	,
5.4422610E-08	,
6.1250461E-08	,
6.9855773E-08	,
7.9463597E-08	,
8.9151150E-08	,
1.0168034E-07	,
1.1558580E-07	,
1.3068458E-07	,
1.6048106E-07	,
1.9350828E-07	,
2.2751291E-07	,
3.0428600E-07	,
4.3981947E-07	,
5.2382995E-07	,
6.3216254E-07	,
8.2302279E-07	,
1.2512422E-06	,
1.8039109E-06	,
2.2908109E-06	,
2.8324889E-06	,
3.4517834E-06	,
4.2219772E-06	,
5.0326839E-06	,
5.6775239E-06	,
6.3139009E-06	,
6.9619100E-06	,
7.7728864E-06	,
8.5246547E-06	,
8.8305105E-06	,
8.4904723E-06	,
7.5621829E-06	,
6.2966351E-06	,
5.1043844E-06	,
4.0821087E-06	,
2.8155102E-06	,
1.8036270E-06	,
1.5450810E-06	,
1.3594723E-06	,
1.1832446E-06	,
1.0330702E-06	,
9.0162695E-07	,
7.8788491E-07	,
6.7509507E-07	,
5.7978644E-07	,
4.9771251E-07	,
4.2984522E-07	,
	])

wkl[4,:] = np.array([
3.2014773E-07	,
3.2014808E-07	,
3.2012952E-07	,
3.2017348E-07	,
3.2020259E-07	,
3.2024920E-07	,
3.2018053E-07	,
3.2015103E-07	,
3.2006952E-07	,
3.1964703E-07	,
3.1794278E-07	,
3.1485408E-07	,
3.0993951E-07	,
3.0289050E-07	,
2.9728139E-07	,
2.9330249E-07	,
2.8654489E-07	,
2.7902988E-07	,
2.6973828E-07	,
2.5467133E-07	,
2.3132466E-07	,
1.9950789E-07	,
1.6908091E-07	,
1.3991885E-07	,
1.1722680E-07	,
1.0331899E-07	,
9.4382699E-08	,
8.7561951E-08	,
8.2404142E-08	,
7.5596006E-08	,
6.6951600E-08	,
5.4150636E-08	,
4.2426844E-08	,
3.2571123E-08	,
2.4015852E-08	,
1.7783966E-08	,
1.2921510E-08	,
9.3075085E-09	,
6.6677854E-09	,
3.5912391E-09	,
2.0309472E-09	,
1.7047587E-09	,
1.4732259E-09	,
1.3152129E-09	,
1.2046001E-09	,
1.1028871E-09	,
1.0173566E-09	,
9.5524733E-10	,
9.0009833E-10	,
8.4775770E-10	,
8.0018175E-10	,
	])

wkl[5,:] = np.array([
1.4735235E-07	,
1.4203219E-07	,
1.3746356E-07	,
1.3388170E-07	,
1.3135738E-07	,
1.3046302E-07	,
1.2931390E-07	,
1.2701938E-07	,
1.2377659E-07	,
1.1940332E-07	,
1.1352941E-07	,
1.0700342E-07	,
1.0015444E-07	,
9.3152551E-08	,
8.5588468E-08	,
7.7191764E-08	,
6.3881643E-08	,
4.8797485E-08	,
3.7298612E-08	,
2.8723687E-08	,
2.2545748E-08	,
1.7379815E-08	,
1.4111547E-08	,
1.2622904E-08	,
1.2397807E-08	,
1.3167179E-08	,
1.4350868E-08	,
1.5625453E-08	,
1.6708778E-08	,
1.8091109E-08	,
1.9843396E-08	,
2.1874927E-08	,
2.3846910E-08	,
2.5646894E-08	,
2.7513584E-08	,
2.9431952E-08	,
3.0938047E-08	,
3.2309320E-08	,
3.3800561E-08	,
3.6464382E-08	,
3.9601694E-08	,
4.2654523E-08	,
4.5695458E-08	,
4.9774858E-08	,
5.4377978E-08	,
5.9385144E-08	,
6.5223382E-08	,
7.4618846E-08	,
8.5339593E-08	,
9.7556516E-08	,
1.1081534E-07	,
	])

wkl[6,:] = np.array([
1.7007853E-06	,
1.7007861E-06	,
1.7006882E-06	,
1.7000174E-06	,
1.6967191E-06	,
1.6890905E-06	,
1.6774702E-06	,
1.6625032E-06	,
1.6469684E-06	,
1.6329801E-06	,
1.6223285E-06	,
1.6071415E-06	,
1.5820669E-06	,
1.5562247E-06	,
1.5313253E-06	,
1.5080506E-06	,
1.4806419E-06	,
1.4479623E-06	,
1.4152675E-06	,
1.3795030E-06	,
1.3426010E-06	,
1.3014652E-06	,
1.2451943E-06	,
1.1722138E-06	,
1.0758683E-06	,
9.6515760E-07	,
8.5401462E-07	,
7.7107171E-07	,
7.2538978E-07	,
6.8032085E-07	,
6.3401592E-07	,
5.7941355E-07	,
5.2736578E-07	,
4.8160666E-07	,
4.3754815E-07	,
3.9457359E-07	,
3.5215132E-07	,
3.1031249E-07	,
2.6731394E-07	,
2.0088720E-07	,
1.5878383E-07	,
1.5400190E-07	,
1.5114806E-07	,
1.5015239E-07	,
1.5018485E-07	,
1.5016241E-07	,
1.5013467E-07	,
1.5023033E-07	,
1.5028188E-07	,
1.5026681E-07	,
1.5018884E-07	,
	])

wkl[7,:] = np.array([
2.0897518E-01	,
2.0897572E-01	,
2.0896780E-01	,
2.0898660E-01	,
2.0899189E-01	,
2.0899543E-01	,
2.0899996E-01	,
2.0900373E-01	,
2.0900458E-01	,
2.0900519E-01	,
2.0900649E-01	,
2.0900634E-01	,
2.0900698E-01	,
2.0900562E-01	,
2.0900711E-01	,
2.0900925E-01	,
2.0900522E-01	,
2.0899965E-01	,
2.0899954E-01	,
2.0899963E-01	,
2.0899959E-01	,
2.0899966E-01	,
2.0899986E-01	,
2.0899987E-01	,
2.0900002E-01	,
2.0899989E-01	,
2.0899986E-01	,
2.0900220E-01	,
2.0900251E-01	,
2.0900670E-01	,
2.0900570E-01	,
2.0900536E-01	,
2.0900574E-01	,
2.0900482E-01	,
2.0900646E-01	,
2.0900702E-01	,
2.0900613E-01	,
2.0900463E-01	,
2.0900150E-01	,
2.0900197E-01	,
2.0901358E-01	,
2.0904660E-01	,
2.0902328E-01	,
2.0906644E-01	,
2.0911193E-01	,
2.0908101E-01	,
2.0904104E-01	,
2.0916539E-01	,
2.0922786E-01	,
2.0919746E-01	,
2.0908001E-01	,
	])

wbrodl = np.array([
2.0212141E+24	,
1.6594377E+24	,
1.2158148E+24	,
1.1250208E+24	,
1.0383909E+24	,
9.5728074E+23	,
8.8085689E+23	,
7.1067458E+23	,
6.5919029E+23	,
6.1060294E+23	,
5.6462299E+23	,
5.2211246E+23	,
4.8122331E+23	,
4.4374356E+23	,
4.0814444E+23	,
3.7436774E+23	,
5.6808373E+23	,
4.3642727E+23	,
3.6534753E+23	,
3.0739262E+23	,
2.5828079E+23	,
2.1627581E+23	,
1.8117264E+23	,
1.5183546E+23	,
1.3798849E+23	,
1.1426864E+23	,
9.4440466E+22	,
8.4273140E+22	,
7.3593470E+22	,
6.2926780E+22	,
5.5637695E+22	,
5.1760866E+22	,
3.6146276E+22	,
2.6801360E+22	,
2.0049382E+22	,
1.5083691E+22	,
1.1414246E+22	,
8.6903496E+21	,
7.5064828E+21	,
1.2774552E+22	,
2.6989144E+21	,
1.6518584E+21	,
1.2455197E+21	,
7.5262510E+20	,
6.3531851E+20	,
5.3673077E+20	,
3.9289481E+20	,
3.1107724E+20	,
2.7061537E+20	,
2.3544404E+20	,
1.8746121E+20	,
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
313.0001	,
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
0.0989301   ,
0.0842557   ,
0.0722468   ,
	])

tavel=np.array([
291.77	,
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
0.00	,
1.10	,
2.10	,
2.90	,
3.70	,
4.50	,
5.30	,
6.10	,
6.80	,
7.50	,
8.20	,
8.90	,
9.60	,
10.30	,
11.00	,
11.70	,
12.40	,
13.60	,
14.70	,
15.80	,
16.90	,
18.00	,
19.10	,
20.20	,
21.30	,
22.50	,
23.70	,
24.90	,
26.20	,
27.60	,
29.10	,
30.80	,
32.90	,
34.90	,
36.90	,
38.90	,
40.90	,
42.90	,
44.90	,
47.20	,
53.90	,
56.40	,
58.40	,
60.30	,
61.70	,
63.10	,
64.50	,
65.70	,
66.80	,
67.90	,
69.00	,
70.00	,
	])*1000.

tz=np.array([
294.2	,
289.25	,
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

tbound=tz[0]

wkl[2,:]*=2.0

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
# wkl = np.clip(wkl,1.,1e63) #only if wkl is molec/cm, not mixing ratio!

ur_min=0.6
ur_max=3.0

eqb_maxhtr=1e-4
eqb_maxdfnet=1e-3
toa_fnet_eqb=1.0e12
timesteps=8000


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

toa_fnet=0

damps=np.ones(nlayers)

color=[]
for i in range(nlayers+1):
    color.append('#%06X' % randint(0, 0xFFFFFF))

dmax=10.0

for ts in range(timesteps):

	# if((maxhtr<eqb_maxhtr*10. and abs(toa_fnet)>toa_fnet_eqb)):
	# 	tz+=toa_fnet*0.2
	# 	tavel+=toa_fnet*0.2
	# 	tbound+=toa_fnet*0.2

	if(ts>0):
		for i in range(1,nlayers):
			# ur[i]=maxhtr**1.0 * pz[0]/(pz[i-1]-pz[i]) + ur_min
			# ur[i]=maxhtr**2.0 + ur_min
			ur[i] = ur_min
			# if(ur[i]>ur_max):
			# 	ur[i]=ur_max

		# for i in range(nlayers+1):
		# 	if(abs(htr[i])>abs(prev_htr[i])):
		# 		damps[i]*=1.02

		# ur=ur*damps

		# tavel[1:] += htr[1:-1]/ur

		# tz=np.clip(tz,tmin,tmax)

		# for i in range(1,nlayers):
		# 	altz[i]=altz[i-1]+(pz[i-1]-pz[i])*rsp*tavel[i]/pavel[i]/gravity

		# for i in range(1,nlayers):
		# 	altavel[i] = (altz[i-1]+altz[i])/2.0
		
		# conv=np.zeros(nlayers+1) #reset to zero
		# conv[0]=1
		# # tavel[0]=tz[0]
		# # conv[1]=1 #couple first layer to surface
		# convection(tavel,altavel)
		# for i in range(1,nlayers):
		# 	# tz[i] = (tavel[i-1] + tavel[i])/2.
		# 	tz[i] = tavel[i]*1.0
		# tz[0]=tbound
		# tz[nlayers] = 2.*tavel[nlayers-1]-tz[nlayers-1]
		# tz=np.clip(tz,tmin,tmax)
		# tavel=np.clip(tavel,tmin,tmax)
		# # convection(tz,altz)
		# for i in range(1,nlayers):
		# 	if(conv[i]==0):
		# 		cti=i-1
		# 		break

		#set up gas amounts

		
		# for i in range(nlayers):
		# 	# h2o
		# 	esat_liq[i] = 6.1094*exp(17.625*(tz[i]-273.15)/(tz[i]-273.15+243.04))
		# 	rel_hum[i] = surf_rh*(pz[i]/1000.0 - 0.02)/(1.0-0.02)
		# 	vol_mixh2o[i] = 0.622*rel_hum[i]*esat_liq[i]/(pavel[i]-rel_hum[i]*esat_liq[i])
		# 	if(i>1 and vol_mixh2o[i] > vol_mixh2o[i-1]):
		# 		vol_mixh2o[i]=vol_mixh2o[i-1]
		# 	vol_mixh2o=np.clip(vol_mixh2o,vol_mixh2o_min,vol_mixh2o_max)
		# 	# wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o[i]*0.

	dtbound=(fnet_sw[0]-fnet_lw[0])*0.1
	dtbound=np.clip(dtbound,-dmax,dmax)
	tbound+=dtbound

	if(lw_on==1):
		writeformattedinputfile_lw()
		callrrtmlw()
		totuflux_lw,totdflux_lw,fnet_lw,htr_lw = readrrtmoutput_lw()

	if(ts==1 and sw_on==1):
	# 	if(maxhtr<eqb_maxhtr):
		writeformattedinputfile_sw()
		callrrtmsw()
		totuflux_sw,totdflux_sw,fnet_sw,htr_sw = readrrtmoutput_sw()

	# fnet_sw=np.ones(nlayers+1)*100.0	
	# fnet_sw=np.linspace(100,0,nlayers+1)
	# htr_sw=np.linspace(0.03,0.,nlayers+1)



	prev_htr=htr

	if(ts>1):
		totuflux_sw*=(238./fnet_sw[nlayers])
		totdflux_sw*=(238./fnet_sw[nlayers])
		htr_sw*=(238./fnet_sw[nlayers])
		fnet_sw*=(238./fnet_sw[nlayers])
	totuflux=totuflux_lw+totuflux_sw
	totdflux=totdflux_lw+totdflux_sw
	fnet=fnet_sw-fnet_lw
	htr=htr_lw+htr_sw

	toa_fnet=totdflux[nlayers]-totuflux[nlayers] #net total downward flux at TOA
	
	# if(cti+1 < nlayers-1):
	# 	maxhtr=max(abs(htr[cti+1:nlayers-1]))
	# else:
	# 	maxhtr = 1.1*eqb_maxhtr

	prev_maxhtr=maxhtr*1.0
	re_htrs = np.where(conv==0,htr,0.)
	maxhtr=max(abs(re_htrs))
	maxhtr_ind=np.argmax(abs(re_htrs))
	dfnet=np.zeros(nlayers)
	dpz=np.zeros(nlayers)
	for i in range(nlayers):
		dfnet[i]=fnet[i+1]-fnet[i]
		dpz[i]=pz[i+1]-pz[i]
	maxdfnet=max(abs(dfnet))

	# toa_fnet=240.-totuflux[nlayers]

	prev_tz=tz*1.0
	for i in range(nlayers):
		dT=dfnet[i]/dpz[i]*-1.*3.0
		# print dT
		# dT = htr[i]/3. #undrelax
		dT=np.clip(dT,-dmax,dmax)
		tavel[i]+=dT

	for i in range(1,nlayers):
		# tz[i] = (tavel[i-1] + tavel[i])/2.
		tz[i] = tavel[i-1]*1.0
	# tz[0]=tbound
	tz[0]=tavel[0]
	# tz[nlayers] = 2*tavel[nlayers-1]-tz[nlayers-1]
	tz[nlayers]=tavel[nlayers-1]

	
	# if(maxhtr<eqb_maxhtr):
	# 	dT_toaeqb = np.clip(toa_fnet*0.3,-50,50)
	# 	tbound+=dT_toaeqb
	# 	tz[0]=tbound

	# if (maxhtr>0.002):
	# 	if(maxhtr<prev_maxhtr and maxhtr/prev_maxhtr>0.):
	# 		dmax*=1.1
	# 		dmax=np.clip(dmax,-10.,10.)
	# 		prev_maxhtr=maxhtr
	# 	else:
	# 		dmax*=0.95
	# elif(maxhtr/prev_maxhtr>0.):
	# 	dmax*=0.95
	# if(0.<dmax<0.0000000001):
	# 	dmax=0.0000000001
	# if(-0.0000000001<dmax<0.):
	# 	dmax=-0.0000000001

	# print prev_tz
	# print tz
	dtz = tz-prev_tz
	maxdtz=dtz[np.argmax(abs(dtz))]

	# if(ts%int((50.*(timesteps/2000.)))==0):
	# 	plt.figure(1)
	# 	# plt.subplot(221)
	# 	# plt.title('maxhtr')
	# 	# plt.plot(ts,maxhtr,'o')
	# 	# plt.subplot(222)
	# 	# plt.title('many htrs')
	# 	# i_c=0
	# 	# for i in range(nlayers):		
	# 	# 	plt.plot(ts,htr[i],'o',color=color[i_c])
	# 	# 	i_c+=1
	# 	# plt.subplot(223)
	# 	plt.subplot(121)
	# 	plt.plot(maxhtr,maxdtz,'o',c=color[np.argmax(abs(dtz))])
	# 	plt.xlabel('max htr')
	# 	plt.ylabel('max dtz')
	# 	# plt.subplot(224)
	# 	# plt.plot(ts,-dmax,'*',c='black',alpha=0.5)
	# 	# plt.plot(ts,dmax,'*',c='black',alpha=0.5)	
	# 	# plt.plot(ts,maxdtz,'o',c=color[np.argmax(abs(dtz))],alpha=0.5)
	# 	# plt.axhline(0.)

	# if(ts%50==0):
	# # if(ts>950):
	# 	plotrrtmoutput()

	if(ts%1==0):
		print('{:4d} | {:32.28f} | {:3d} | {:3d} | {:12.8f} | {:8.4f} | {:8.4f} | {:12.8f} | {:12.8f} | {:12.8f} | {:12.8f}'.format(ts,maxdfnet,maxhtr_ind,cti,maxdfnet,tbound,tz[0],-dmax,maxdtz,dmax,toa_fnet))


	params0d=[gravity,avogadro,iatm,ixsect,iscat,numangs,iout,icld,tbound,iemiss,iemis,ireflect,iaer,istrm,idelm,icos,iform,nlayers,nmol,psurf,pmin,secntk,cinp,ipthak,ipthrk,juldat,sza,isolvar,lapse,tmin,tmax,rsp,gravity,pin2,pico2,pio2,piar,pich4,pih2o,pio3,mmwn2,mmwco2,mmwo2,mmwar,mmwch4,mmwh2o,mmwo3,piair,totmolec,surf_rh,vol_mixh2o_min,vol_mixh2o_max,ur_min,ur_max,eqb_maxhtr,timesteps,cti,maxhtr]
	params1d=[semis,semiss,totuflux,totuflux_lw,totuflux_sw,totdflux,totdflux_lw,totdflux_sw,fnet,fnet_lw,fnet_sw,htr,htr_lw,htr_sw,pz,pavel,tz,tavel,altz,esat_liq,rel_hum,vol_mixh2o,wbrodl,mperlayr,mperlayr_air,conv,altavel]
	params2d=[wkl]

	# if(maxhtr < eqb_maxhtr and abs(toa_fnet) < toa_fnet_eqb):
	if(maxdfnet < eqb_maxdfnet and abs(toa_fnet) < toa_fnet_eqb and ts>1):
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
	
# plt.figure(1)
# plt.subplot(122)
# for i in range(nlayers):
# 	plt.semilogy(tz[i],pz[i],'o',c=color[i])
# 	plt.ylim(max(pz),min(pz))

tend = datetime.datetime.now()
ttotal = tend-tstart
print(ttotal)

print('Done')
# plt.tight_layout()
# show()