# RRTM LW and SW Python wrapper

import numpy as np

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
ireflect=0 #for Lambert reflection
iaer=0 #0=aerosols off, 1=on
istrm=1 			# ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0). 
						#0=4 streams
						#1=8 streams
idelm=0 			# flag for outputting downwelling fluxes computed using the delta-M scaling approximation. 0=output "true" direct and diffuse downwelling fluxes, 1=output direct and diffuse downwelling fluxes computed with delta-M approximation
icos=0 				#0:there is no need to account for instrumental cosine response, 1:to account for instrumental cosine response in the computation of the direct and diffuse fluxes, 2:2 to account for instrumental cosine response in the computation of the diffuse fluxes only
semis=np.ones(16) 	#all spectral bands the same as iemissm
semiss=np.ones(16) 	#all spectral bands the same as iemissm (surface, I think)
iform=1
nlayers=10
nmol=7
psurf=1000.
pmin=10.
secntk=0
cinp=0
ipthak=0
ipthrk=0
juldat=0 		#Julian day associated with calculation (1-365/366 starting January 1). Used to calculate Earth distance from sun. A value of 0 (default) indicates no scaling of solar source function using earth-sun distance.
sza=0 			#Solar zenith angle in degrees (0 deg is overhead).
isolvar=0 		#= 0 each band uses standard solar source function, corresponding to present day conditions. 
				#= 1 scale solar source function, each band will have the same scale factor applied, (equal to SOLVAR(16)). 
				#= 2 scale solar source function, each band has different scale factors (for band IB, equal to SOLVAR(IB))			
lapse=5.7
tmin=150.
tmax=350.

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
altz=np.log(psurf/pz)*rsp*tz/gravity/1000. #[km]
tz=np.ones(nlayers+1) * tbound-lapse*altz
tz=np.clip(tz,tmin,tmax)
tavel=np.zeros(nlayers)
for i in range(len(pavel)):
	tavel[i]=(tz[i]+tz[i+1])/2.

# # Gas inventories
pin2 = 0.8 * 1e5 #convert the input in bar to Pa
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
vol_mixh2o = molec_h2o / totmolec
vol_mixo3 = molec_o3 / totmolec

# Mean molecular weight of the atmosphere
mmwtot = mmwco2 * vol_mixco2 + mmwn2 * vol_mixn2 + mmwo2 * vol_mixo2 + mmwar*vol_mixar + mmwch4*vol_mixch4 + mmwh2o*vol_mixh2o

mperlayr = np.zeros(nlayers)
mperlayr_air = np.zeros(nlayers)

for i in range(nlayers):
    mperlayr[i] = totmolec/nlayers #Divide the molecules equally between layers
    mperlayr_air[i] = (molec_n2 + molec_o2)/(nlayers)


wbrodl = np.zeros(nlayers)
wkl = np.zeros((nmol+1,nlayers))

#Set up mixing ratio of broadening molecules (N2 and O2 mostly)
for i in range(nlayers):
    wbrodl[i] = mperlayr_air[i] * 1.0e-4
    wkl[1,i] = mperlayr[i] * 1.0e-4 * vol_mixh2o
    wkl[2,i] = mperlayr[i] * 1.0e-4 * vol_mixco2
    wkl[3,i] = mperlayr[i] * 1.0e-4 * vol_mixo3
    wkl[6,i] = mperlayr[i] * 1.0e-4 * vol_mixch4
    wkl[7,i] = mperlayr[i] * 1.0e-4 * vol_mixo2

# wbrodl=np.ones(nlayers) * 1e20
# wkl=np.zeros((nmol,nlayers))
# wkl[1,:] = 2e18*0.
nxmol0=nmol #don't know what this is


def writeparams(params,f):	
	for i in params:
		f.write(str(i))
		f.write('\n')

def writeparamsarr(params,f):
	for param in params:
		for i in range(len(param)):
			f.write(str(param[i]))
			f.write('\n')

f = open('LW/RRTM LW Input','w+')

params = [iatm,ixsect,iscat,numangs,iout,icld]
writeparams(params,f)

params = [tbound,iemiss,ireflect]
writeparams(params,f)

for i in range(len(semis)):
	f.write(str(semis[i]))
	f.write('\n')

params = [iform,nlayers,nmol]
writeparams(params,f)

# params=[pavel[1],tavel[1],secntk,cinp,ipthak,altz[0],pz[0],tz[0],altz[1],pz[1],tz[1]]
params = [secntk,cinp,ipthak]
writeparams(params,f)

params = [pavel,tavel,altz,pz,tz]
writeparamsarr(params,f)

params = [wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
writeparamsarr(params,f)

f.close()

f = open('SW/RRTM SW Input','w+')

params = [iaer, iatm, iscat, istrm, iout, icld, idelm, icos]
writeparams(params,f)

params = [juldat,sza,isolvar]
writeparams(params,f)

params = [iemiss,ireflect]
writeparams(params,f)

for i in range(len(semiss)):
	f.write(str(semiss[i]))
	f.write('\n')

params = [iform,nlayers,nmol]
writeparams(params,f)

params = [secntk,cinp,ipthak]
writeparams(params,f)

params = [pavel,tavel,altz,pz,tz]
writeparamsarr(params,f)

params = [wbrodl,wkl[0,:],wkl[1,:],wkl[2,:],wkl[3,:],wkl[4,:],wkl[5,:],wkl[6,:]]
writeparamsarr(params,f)

f.close()

print 'Done'