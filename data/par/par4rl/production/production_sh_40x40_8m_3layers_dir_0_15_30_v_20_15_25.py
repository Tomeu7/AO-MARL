import shesha.config as conf
import numpy as np

simul_name = "production_sh_40x40_8m_3layers_dir_0_15_30_v_20_15_25"

diameter_telescope = 8.0
nxsub = 40
nact = 41

# atmospheric parameters
wind_factor = 1.0
r0 = 0.16

#
d = 1.
n = -1
g = 0.7

################################################################
# loop
p_loop = conf.Param_loop()

p_loop.set_niter(2000)
p_loop.set_ittime(0.002)  # =1/500

# geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()

p_tel.set_diam(diameter_telescope)
p_tel.set_cobs(0.12)

# atmos
p_atmos = conf.Param_atmos()

p_atmos.set_r0(r0)
p_atmos.set_nscreens(3)
p_atmos.set_frac([0.6, 0.25, 0.15])
p_atmos.set_alt([0.0, 4500.0, 14000.0])
p_atmos.set_windspeed([20,
                       15,
                       25])
# one at the ground one at 5 km, both have the same wind speed, case were the layer at 5km is faster
p_atmos.set_winddir([0, 15, 30])
p_atmos.set_L0([1.e5, 1.e5, 1.e5])

# target
p_target0 = conf.Param_target()
p_target1 = conf.Param_target()
p_targets = [p_target0, p_target1]

# p_target0.set_ntargets(1)
p_target0.set_dms_seen([0, 2])
p_target0.set_xpos(0.)
p_target0.set_ypos(0.)
p_target0.set_Lambda(1.65)
p_target0.set_mag(10.)

p_target1.set_dms_seen([1, 3])
p_target1.set_xpos(0.)
p_target1.set_ypos(0.)
p_target1.set_Lambda(1.65)
p_target1.set_mag(10.)

# wfs
p_wfs0 = conf.Param_wfs()
p_wfs1 = conf.Param_wfs()
p_wfss = [p_wfs0, p_wfs1]

p_wfs0.set_type("sh")
p_wfs0.set_nxsub(nxsub)
p_wfs0.set_npix(16)
p_wfs0.set_dms_seen(np.array([0, 2]))
p_wfs0.set_pixsize(0.25)
p_wfs0.set_fracsub(0.8)
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.5)
p_wfs0.set_gsmag(4.)
p_wfs0.set_optthroughput(0.12)
p_wfs0.set_zerop(1.e11)
p_wfs0.set_noise(n)
p_wfs0.set_atmos_seen(1)

p_wfs1.set_type("sh")
p_wfs1.set_nxsub(nxsub)
p_wfs1.set_npix(16)
p_wfs1.set_dms_seen(np.array([1, 3]))
p_wfs1.set_pixsize(0.25)
p_wfs1.set_fracsub(0.8)
p_wfs1.set_xpos(0.)
p_wfs1.set_ypos(0.)
p_wfs1.set_Lambda(0.5)
p_wfs1.set_gsmag(4.)
p_wfs1.set_optthroughput(0.12)
p_wfs1.set_zerop(1.e11)
p_wfs1.set_noise(-1.)
p_wfs1.set_atmos_seen(1)

# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
# geo dm
p_dm2 = conf.Param_dm()
p_dm3 = conf.Param_dm()

# dm

p_dms = [p_dm0, p_dm2, p_dm1, p_dm3]
p_dm0.set_type("pzt")
p_dm0.set_nact(nact)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.2)
p_dm0.set_unitpervolt(0.01)
p_dm0.set_push4imat(100.)

# geo dm
p_dm2.set_type("pzt")
p_dm2.set_nact(nact)
p_dm2.set_alt(0.)
p_dm2.set_thresh(0.3)
p_dm2.set_coupling(0.2)
p_dm2.set_unitpervolt(0.01)
p_dm2.set_push4imat(100.)

# tt dm
p_dm1.set_type("tt")
p_dm1.set_alt(0.)
p_dm1.set_unitpervolt(0.0005)
p_dm1.set_push4imat(10.)

# tt geo dm
p_dm3.set_type("tt")
p_dm3.set_alt(0.)
p_dm3.set_unitpervolt(0.0005)
p_dm3.set_push4imat(10.)

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroider1 = conf.Param_centroider()
p_centroiders = [p_centroider0, p_centroider1]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("cog")
p_centroider1.set_nwfs(1)
p_centroider1.set_type("cog")
# p_centroider0.set_type("corr")
# p_centroider0.set_type_fct("model")

# controllers
p_controller0 = conf.Param_controller()
p_controller1 = conf.Param_controller()
p_controllers = [p_controller0, p_controller1]

p_controller0.set_type("ls")
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0, 2])
p_controller0.set_maxcond(1500.)
p_controller0.set_delay(d)
p_controller0.set_gain(g)


p_controller1.set_type("geo")
p_controller1.set_nwfs([1])
p_controller1.set_ndm([1, 3])
p_controller1.set_maxcond(1500.)
p_controller1.set_delay(0.)
p_controller1.set_gain(g)
