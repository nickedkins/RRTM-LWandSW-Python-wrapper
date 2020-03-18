# RRTM LW and SW Python wrapper

import numpy as np

f = open('LW/RRTM LW Input','w+')

iatm=0 #0 for layer values, 1 for level values
ixsect=0 #could be 1, but why?
iscat=0 #just absorption and emission
numangs=0 #can be 0-4 for higher precision
#ioutm=99 #for info in all spectral bands
iout=0 #for broadband only
#ioutm=-1 #for broadband, no printings
icld=0 #for clear sky
#icldm=1  #for grey clouds
tbound = 288 #surface temperature (K)
iemiss=1 #surface emissivity. Keep this fixed for now.
ireflect=0 #for Lambert reflection
semis=np.ones(16) #all spectral bands the same as iemissm
iform=1
nlayers=10
nmol=7
psurf=1000.
pmin=10.
secntk=0
cinp=999
ipthak=0
ipthrk=0
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
wbrodl=np.ones(nlayers) * 1e20
wkl=np.zeros((nmol,nlayers))
wkl[1,:] = 1e18
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