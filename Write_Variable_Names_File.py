project_dir = '/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/'

nlayers=590
ncloudcols=1
nmol=7

vars_0d_names=["	gravity	"	,
"	avogadro	"	,
"	iatm	"	,
"	ixsect	"	,
"	iscat	"	,
"	numangs	"	,
"	iout	"	,
"	icld	"	,
"	tbound	"	,
"	iemiss	"	,
"	iemis	"	,
"	ireflect	"	,
"	iaer	"	,
"	istrm	"	,
"	idelm	"	,
"	icos	"	,
"	iform	"	,
"	nlayers	"	,
"	nmol	"	,
"	psurf	"	,
"	pmin	"	,
"	secntk	"	,
"	cinp	"	,
"	ipthak	"	,
"	ipthrk	"	,
"	juldat	"	,
"	sza	"	,
"	isolvar	"	,
"	lapse	"	,
"	tmin	"	,
"	tmax	"	,
"	rsp	"	,
"	gravity	"	,
"	pin2	"	,
"	pico2	"	,
"	pio2	"	,
"	piar	"	,
"	pich4	"	,
"	pih2o	"	,
"	pio3	"	,
"	mmwn2	"	,
"	mmwco2	"	,
"	mmwo2	"	,
"	mmwar	"	,
"	mmwch4	"	,
"	mmwh2o	"	,
"	mmwo3	"	,
"	piair	"	,
"	totmolec	"	,
"	surf_rh	"	,
"	vol_mixh2o_min	"	,
"	vol_mixh2o_max	"	,
"	ur_min	"	,
"	ur_max	"	,
"	eqb_maxhtr	"	,
"	timesteps	"	,
"	cti	"	,
"	maxhtr	"	,
"	cld_lay	"	,
"	ncloudcols	"	,
"	master_input	"	,
"	conv_on	"	,
"	surf_lowlev_coupled	"	,
"	lay_intp	"	,
"	lw_on	"	,
"	sw_on	"	,
"	eqb_maxdfnet	"	,
"	toa_fnet_eqb	"	,]

vars_master_lay_cld_names=["	tavel_master	"	,
"	pavel_master	"	,
"	altavel_master	"	,
"	wbrodl_master	"	,]

vars_master_lev_cld_names=["	tz_master	"	,
"	pz_master	"	,
"	altz_master	"	,
"	totuflux_master	"	,
"	totuflux_lw_master	"	,
"	totuflux_sw_master	"	,
"	totdflux_master	"	,
"	totdflux_lw_master	"	,
"	totdflux_sw_master	"	,
"	fnet_master	"	,
"	fnet_lw_master	"	,
"	fnet_sw_master	"	,
"	htr_master	"	,
"	htr_lw_master	"	,
"	htr_sw_master	"	,
"	conv_master	"	,]

vars_misc_1d_names=["	semis	"	,
"	semiss	"	,
"	solvar	"	,]

vars_misc_1d_lens=[16,29,29]

vars_master_lay_cld_nmol_names=["wkl_master"]

vars_master_cld_names=["	inflags	"	,
"	iceflags	"	,
"	liqflags	"	,
"	cld_lays	"	,
"	cld_fracs	"	,
"	tauclds	"	,
"	ssaclds	"	,]

f=open(project_dir+'Output Variable Names','w+')
for i in vars_0d_names:
	f.write(i)
	f.write('\n')
for i in vars_master_lay_cld_names:
	for j in range(nlayers):
		f.write(i)
		f.write('\n')
for i in vars_master_lev_cld_names:
	for j in range(nlayers+1):
		f.write(i)
		f.write('\n')

i_lens=0
for x in vars_misc_1d_names:
	for i in range(vars_misc_1d_lens[i_lens]):
		f.write(x)
		f.write('\n')
	i_lens+=1

for i in vars_master_lay_cld_nmol_names:
	for k in range(nmol+1):
		for j in range(nlayers):
			f.write(i+str(k+1))
			f.write('\n')

for x in vars_master_cld_names:
	for i in range(ncloudcols):
		f.write(str(x))
		f.write('\n')