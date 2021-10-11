"""
ROKET (erROr breaKdown Estimation Tool)

Computes the error breakdown during a COMPASS simulation
and saves it in a HDF5 file
Error contributors are bandwidth, tomography, noise, aliasing,
WFS non linearity, filtered modes and fitting
Saved file contained temporal buffers of those contributors
"""

import cProfile
import pstats as ps

import os
import numpy as np
from shesha.supervisor.rlSupervisor import RlSupervisor
from shesha.supervisor.rlSupervisor import CompassSupervisor
from shesha.util.rtc_util import centroid_gain
from shesha.ao.tomo import create_nact_geom
from shesha.ao.basis import compute_btt, compute_cmat_with_Btt
import shesha.constants as scons
import matplotlib.pyplot as pl
pl.ion()
import shesha.util.hdf5_util as h5u


class Roket(RlSupervisor):
    """
    ROKET class
    Inherits from RlSupervisor
    """

    def __init__(self, config, *, config_rl=None,
                 initial_seed=1234,
                 autoencoder=None,
                 cacao: bool = False
                 ):
        """
        Initializes an instance of Roket class
        """

        self.agent = None
        self.include_tip_tilt = None
        self.nfiltered = None
        self.N_preloop = None
        self.gamma = None
        self.n = None
        self.iter_number = None
        self.n = None
        self.nactus = None
        self.nslopes = None
        self.com = None
        self.noise_com = None
        self.noise_buf = None
        self.alias_wfs_com = None
        self.ageom = None
        self.alias_meas = None
        self.wf_com = None
        self.tomo_com = None
        self.tomo_buf = None
        self.trunc_com = None
        self.trunc_buf = None
        self.trunc_meas = None
        self.H_com = None
        self.mod_com = None
        self.bp_com = None
        self.fit = None
        self.psf_ortho = None
        self.centroid_gain = None
        self.centroid_gain2 = None
        self.slopes = None
        self.IFpzt = None
        self.TT = None
        self.Btt, self.P = None, None
        self.cmat = None
        self.D = None
        self.RD = None
        self.gRD = None
        self.Nact = None
        self.delay = None
        self.rl_commands = None
        self.zeta_contributor = None
        self.n_zernike_start = None
        self.n_zernike_end = None
        self.cov = None
        self.cor = None
        self.SR2 = None
        self.SR = None

        super().__init__(config=config,
                         config_rl=config_rl,
                         build_cmat_with_modes=False,  # cmat with modes is already built with Btt in init_config_roket
                         initial_seed=initial_seed,
                         autoencoder=autoencoder,
                         cacao=cacao)

    def init_config_roket(self,
                          N_total=3000,
                          N_preloop=1000,
                          agent=None,
                          nfiltered=10,
                          gamma=1.,
                          include_tip_tilt=True,
                          n_zernike_start=None,
                          n_zernike_end=None):
        """
        Initializes the COMPASS simulation and the ROKET buffers
        Args:
            N_preloop: (int): (optional) number of iterations before starting error breakdown estimation
            N_preloop: (int): (optional) number of iterations before starting error breakdown estimation
            agent (list): RL agents
            nfiltered (int): number of filtered low sensitivity modes equivalent to n_reversed_filtered_from_cmat
            gamma: (float): (optional) centroid gain
            include_tip_tilt (bool): if tip tilt is included
            n_zernike_start (int): first mode number controlled (not counting tip tilt)
            n_zernike_end (int): last mode number controlled (not counting tip tilt)
        """
        assert N_total >= N_preloop
        self.agent = agent
        self.include_tip_tilt = include_tip_tilt
        self.nfiltered = nfiltered

        #########################################################################################
        self.N_preloop = N_preloop
        self.gamma = gamma
        self.n = N_total
        self.iter_number = 0
        self.n = N_total
        self.nactus = self.rtc.get_command(0).size
        self.nslopes = self.rtc.get_slopes(0).size
        self.com = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.noise_com = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.noise_buf = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.alias_wfs_com = np.copy(self.noise_com)
        self.ageom = np.copy(self.noise_com)
        self.alias_meas = np.zeros((self.n, self.nslopes), dtype=np.float32)
        self.wf_com = np.copy(self.noise_com)
        self.tomo_com = np.copy(self.noise_com)
        self.tomo_buf = np.copy(self.noise_com)
        self.trunc_com = np.copy(self.noise_com)
        self.trunc_buf = np.copy(self.noise_com)
        self.trunc_meas = np.copy(self.alias_meas)
        self.H_com = np.copy(self.noise_com)
        self.mod_com = np.copy(self.noise_com)
        self.bp_com = np.copy(self.noise_com)
        self.fit = np.zeros(self.n)
        self.psf_ortho = self.target.get_tar_image(0) * 0
        self.centroid_gain = 0
        self.centroid_gain2 = 0
        self.slopes = np.zeros((self.n, self.nslopes), dtype=np.float32)
        self.config.p_loop.set_niter(self.n)
        self.IFpzt = self.rtc._rtc.d_control[1].d_IFsparse.get_csr().astype(np.float32)
        self.TT = np.array(self.rtc._rtc.d_control[1].d_TT)
        self.Btt, self.P = compute_btt(self.IFpzt.T, self.TT)
        tmp = self.Btt.dot(self.Btt.T)
        self.rtc._rtc.d_control[1].load_Btt(tmp[:-2, :-2], tmp[-2:, -2:])
        compute_cmat_with_Btt(self.rtc._rtc, self.Btt, self.nfiltered)
        self.cmat = self.rtc.get_command_matrix(0)
        self.D = self.rtc.get_interaction_matrix(0)
        self.RD = np.dot(self.cmat, self.D)
        self.gRD = self.config.p_controllers[0].gain * self.gamma * self.RD
        self.Nact = create_nact_geom(self.config.p_dms[0])
        self.delay = int(self.config.p_controllers[0].delay) + 1
        #########################################################################

        self.rl_commands = np.zeros((self.n, self.Btt.shape[0]), dtype=np.float32)  # shape 2000 x actuators
        self.zeta_contributor = np.copy(self.noise_com)
        self.n_zernike_start = n_zernike_start
        self.n_zernike_end = n_zernike_end

    def do_error_breakdown(self, a):

        if self.agent is not None:
            rl = np.zeros(self.Btt.shape[1])
            if self.include_tip_tilt:
                rl[self.n_zernike_start:self.n_zernike_end] = a[:-2]
                rl[-2:] = a[-2:]
            else:
                rl[self.n_zernike_start:self.n_zernike_end] = a
            rl[self.action_range_for_roket] *= self.freedom_vector[self.action_range_for_roket]
            self.rl_commands[self.iter_number, :] = self.Btt.dot(rl)

        self.error_breakdown()
        self.rtc.apply_control(0)
        self.target.comp_tar_image(0)
        self.target.comp_strehl(0)
        self.iter_number += 1

    def calculate_zeta_contributor(self):
        rl_diff = self.rl_commands[self.iter_number - self.delay, :]  # - self.rl[self.iter_number - self.delay - 1, :]
        self.zeta_contributor[self.iter_number, :] = self.zeta_contributor[self.iter_number - 1, :] - self.gRD.dot(
            self.zeta_contributor[self.iter_number - self.delay, :]) + rl_diff

    def calculate_noise_contributor(self, Derr, g):
        if self.config.p_wfss[0].type == scons.WFSType.SH:
            ideal_img = np.array(self.wfs._wfs.d_wfs[0].d_binimg_notnoisy)
            binimg = np.array(self.wfs._wfs.d_wfs[0].d_binimg)
            if self.config.p_centroiders[0].type == scons.CentroiderType.TCOG:
                # Select the same pixels with or without noise
                invalidpix = np.where(binimg <= self.config.p_centroiders[0].thresh)
                ideal_img[invalidpix] = 0
                self.rtc.set_centroider_threshold(0, -1e16)
            self.wfs._wfs.d_wfs[0].set_binimg(ideal_img, ideal_img.size)
        elif self.config.p_wfss[0].type == scons.WFSType.PYRHR: # TODO centroider type or WFS type?
            ideal_pyrimg = np.array(self.wfs._wfs.d_wfs[0].d_binimg_notnoisy)
            self.wfs._wfs.d_wfs[0].set_binimg(ideal_pyrimg, ideal_pyrimg.size)

        self.rtc.do_centroids(0)
        if self.config.p_centroiders[0].type == scons.CentroiderType.TCOG:
            self.rtc.set_centroider_threshold(0, self.config.p_centroiders[0].thresh)

        self.rtc.do_control(0)
        E = self.rtc.get_err(0)
        E_meas = self.rtc.get_slopes(0)
        self.noise_buf[self.iter_number, :] = (Derr - E)
        # Apply loop filter to get contribution of noise on commands
        self.noise_com[self.iter_number, :] = self.noise_com[self.iter_number - 1, :] - self.gRD.dot(
            self.noise_com[self.iter_number - self.delay, :]) + g * self.noise_buf[self.iter_number - self.delay, :]

        return E, E_meas

    def calculate_sampling_truncature_contributor(self, E, E_meas, Derr, g):
        self.rtc.do_centroids_geom(0)
        self.rtc.do_control(0)
        F = self.rtc.get_err(0)
        F_meas = self.rtc.get_slopes(0)
        self.trunc_meas[self.iter_number, :] = E_meas - F_meas
        self.trunc_buf[self.iter_number, :] = (E - self.gamma * F)
        # Apply loop filter to get contribution of sampling/truncature on commands
        self.trunc_com[self.iter_number, :] = self.trunc_com[self.iter_number - 1, :] - self.gRD.dot(
            self.trunc_com[self.iter_number - self.delay, :]) + g * self.trunc_buf[self.iter_number - self.delay, :]
        self.centroid_gain += centroid_gain(E, F)
        self.centroid_gain2 += centroid_gain(Derr, F)

    def calculate_aliasing_on_wfs_direction_contributor(self, g):

        self.rtc.do_control(1, sources=self.wfs.sources, is_wfs_phase=True)
        self.rtc.apply_control(1)
        for w in range(len(self.config.p_wfss)):
            self.wfs.raytrace(w, dms=self.dms, reset=False)
        self.rtc.do_centroids_geom(0)
        self.rtc.do_control(0)
        self.ageom[self.iter_number, :] = self.rtc.get_err(0)
        self.alias_meas[self.iter_number, :] = self.rtc.get_slopes(0)
        self.alias_wfs_com[self.iter_number, :] = self.alias_wfs_com[self.iter_number - 1, :] - self.gRD.dot(
            self.alias_wfs_com[self.iter_number - self.delay, :]) + self.gamma * g * self.ageom[
                                                                                     self.iter_number - self.delay, :]

    def calculate_fitting_contributor(self):
        self.rtc.apply_control(1)
        self.target.raytrace(0, dms=self.dms, ncpa=False, reset=False)  # , comp_avg_var=False)
        self.target.comp_tar_image(0, compLE=False)
        self.target.comp_strehl(0)
        self.fit[self.iter_number] = self.target.get_strehl(0)[2]
        if self.iter_number >= self.N_preloop:
            self.psf_ortho += self.target.get_tar_image(0, expo_type='se')

    def calculate_filtered_modes_contributor(self, B):
        modes = self.P.dot(B)
        modes_filt = modes.copy() * 0.
        modes_filt[-self.nfiltered - 2:-2] = modes[-self.nfiltered - 2:-2]
        self.H_com[self.iter_number, :] = self.Btt.dot(modes_filt)
        modes[-self.nfiltered - 2:-2] = 0
        self.mod_com[self.iter_number, :] = self.Btt.dot(modes)

    def calculate_bandwidth_contributor(self):
        C = self.mod_com[self.iter_number, :] - self.mod_com[self.iter_number - 1, :]
        self.bp_com[self.iter_number, :] = self.bp_com[self.iter_number - 1, :] - self.gRD.dot(
                self.bp_com[self.iter_number - self.delay, :]) - C

    def calculate_tomographic_contributor(self, g):
        for w in range(len(self.config.p_wfss)):
            self.wfs.raytrace(w, atm=self.atmos)

        self.rtc.do_control(1, sources=self.wfs.sources, is_wfs_phase=True)
        G = self.rtc.get_command(1)
        modes = self.P.dot(G)
        modes[-self.nfiltered - 2:-2] = 0
        self.wf_com[self.iter_number, :] = self.Btt.dot(modes)

        self.tomo_buf[self.iter_number, :] = self.mod_com[self.iter_number, :] - self.wf_com[self.iter_number, :]
        self.tomo_com[self.iter_number, :] = self.tomo_com[self.iter_number - 1, :] - self.gRD.dot(
            self.tomo_com[self.iter_number - self.delay, :]) - g * self.gamma * self.RD.dot(
            self.tomo_buf[self.iter_number - self.delay, :])

    def error_breakdown(self):
        """
        Compute the error breakdown of the AO simulation. Returns the error commands of
        each contributors. Suppose no delay (for now) and only 2 controllers : the main one, controller #0, (specified on the parameter file)
        and the geometric one, controller #1 (automatically added if roket is asked in the parameter file)
        Commands are computed by applying the loop filter on various kind of commands : (see schema_simulation_budget_erreur_v2)

            - Ageom : Aliasing contribution on WFS direction
                Obtained by computing commands from DM orthogonal phase (projection + slopes_geom)

            - B : Projection on the target direction
                Obtained as the commmands output of the geometric controller

            - C : Wavefront
                Obtained by computing commands from DM parallel phase (RD*B)

            - E : Wavefront + aliasing + ech/trunc + tomo
                Obtained by performing the AO loop iteration without noise on the WFS

            - F : Wavefront + aliasing + tomo
                Obtained by performing the AO loop iteration without noise on the WFS and using phase deriving slopes

            - G : tomo

        Note : rtc.get_err returns to -CMAT.slopes
        """
        g = self.config.p_controllers[0].gain
        Dcom = self.rtc.get_command(0)
        Derr = self.rtc.get_err(0)
        self.com[self.iter_number, :] = Dcom
        tarphase = self.target.get_tar_phase(0)
        self.slopes[self.iter_number, :] = self.rtc.get_slopes(0)
        
        ###########################################################################
        # Noise contribution
        ###########################################################################

        E, E_meas = self.calculate_noise_contributor(Derr, g)

        ###########################################################################
        # Sampling/truncature contribution
        ###########################################################################

        self.calculate_sampling_truncature_contributor(E, E_meas, Derr, g)

        ###########################################################################
        # Aliasing contribution on WFS direction
        ###########################################################################

        self.calculate_aliasing_on_wfs_direction_contributor(g)

        ###########################################################################
        # Wavefront + filtered modes reconstruction
        ###########################################################################

        self.target.raytrace(0, atm=self.atmos, ncpa=False)
        self.rtc.do_control(1, sources=self.target.sources, is_wfs_phase=False)
        B = self.rtc.get_command(1)

        ###########################################################################
        # Fitting
        ###########################################################################
        self.calculate_fitting_contributor()

        ###########################################################################
        # Filtered modes error & Commanded modes
        ###########################################################################

        self.calculate_filtered_modes_contributor(B)

        ###########################################################################
        # Bandwidth error
        ###########################################################################

        self.calculate_bandwidth_contributor()

        ###########################################################################
        # RL error, zeta
        ###########################################################################

        self.calculate_zeta_contributor()

        ###########################################################################
        # Tomographic error
        ###########################################################################

        self.calculate_tomographic_contributor(g)

        # Reset target phase and rtc commands
        self.target.set_tar_phase(0, tarphase)
        self.rtc.set_command(0, Dcom)

    def save_in_hdf5(self, savename):
        """
        Saves all the ROKET buffers + simuation parameters in a HDF5 file

        Args:
            savename: (str): name of the output file
        """
        tmp = (self.config.p_geom._ipupil.shape[0] -
               (self.config.p_dms[0]._n2 - self.config.p_dms[0]._n1 + 1)) // 2
        tmp_e0 = self.config.p_geom._ipupil.shape[0] - tmp
        tmp_e1 = self.config.p_geom._ipupil.shape[1] - tmp
        pup = self.config.p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
        indx_pup = np.where(pup.flatten() > 0)[0].astype(np.int32)
        dm_dim = self.config.p_dms[0]._n2 - self.config.p_dms[0]._n1 + 1
        self.cov_cor()
        psf = self.target.get_tar_image(0, expo_type='le')
        if os.getenv("DATA_GUARDIAN") is not None:
            fname = os.getenv("DATA_GUARDIAN") + "/" + savename
        else:
            fname = savename
        print("Shapes command bp and zeta",
              self.bp_com[self.N_preloop:, :].T.shape,
              self.zeta_contributor[self.N_preloop:, :].T.shape,
              "N preloop", self.N_preloop)
        pdict = {
                "noise": self.noise_com[self.N_preloop:, :].T,
                "aliasing": self.alias_wfs_com[self.N_preloop:, :].T,
                "tomography": self.tomo_com[self.N_preloop:, :].T,
                "filtered modes": self.H_com[self.N_preloop:, :].T,
                "non linearity": self.trunc_com[self.N_preloop:, :].T,
                "bandwidth": self.bp_com[self.N_preloop:, :].T,
                "wf_com": self.wf_com[self.N_preloop:, :].T,
                "zeta_com": self.zeta_contributor[self.N_preloop:, :].T,
                "P": self.P,
                "Btt": self.Btt,
                "IF.data": self.IFpzt.data,
                "IF.indices": self.IFpzt.indices,
                "IF.indptr": self.IFpzt.indptr,
                "TT": self.TT,
                "dm_dim": dm_dim,
                "indx_pup": indx_pup,
                "fitting": np.mean(self.fit[self.N_preloop:]),
                "SR": self.SR,
                "SR2": self.SR2,
                "cov": self.cov,
                "cor": self.cor,
                "psfortho": np.fft.fftshift(self.psf_ortho) / (self.config.p_loop.niter - self.N_preloop),
                "centroid_gain": self.centroid_gain / (self.config.p_loop.niter - self.N_preloop),
                "centroid_gain2": self.centroid_gain2 / (self.config.p_loop.niter - self.N_preloop),
                "dm.xpos": self.config.p_dms[0]._xpos,
                "dm.ypos": self.config.p_dms[0]._ypos,
                "R": self.rtc.get_command_matrix(0),
                "D": self.rtc.get_interaction_matrix(0),
                "Nact": self.Nact,
                "com": self.com[self.N_preloop:, :].T,
                "slopes": self.slopes[self.N_preloop:, :].T,
                "alias_meas": self.alias_meas[self.N_preloop:, :].T,
                "trunc_meas": self.trunc_meas[self.N_preloop:, :].T
        }
        h5u.save_h5(fname, "psf", self.config, psf)
        for k in list(pdict.keys()):
            h5u.save_hdf5(fname, k, pdict[k])

    def cov_cor(self):
        if self.agent is not None:
            self.cov = np.zeros((7, 7))
            self.cor = np.zeros((7, 7))
            bufdict = {
                "0": self.noise_com.T,
                "1": self.trunc_com.T,
                "2": self.alias_wfs_com.T,
                "3": self.H_com.T,
                "4": self.bp_com.T,
                "5": self.tomo_com.T,
                "6": self.zeta_contributor.T
            }
        else:
            self.cov = np.zeros((6, 6))
            self.cor = np.zeros((6, 6))
            bufdict = {
                "0": self.noise_com.T,
                "1": self.trunc_com.T,
                "2": self.alias_wfs_com.T,
                "3": self.H_com.T,
                "4": self.bp_com.T,
                "5": self.tomo_com.T
            }

        for i in range(self.cov.shape[0]):
            for j in range(self.cov.shape[1]):
                if j >= i:
                    tmpi = self.P.dot(bufdict[str(i)])
                    tmpj = self.P.dot(bufdict[str(j)])
                    self.cov[i, j] = np.sum(
                            np.mean(tmpi * tmpj, axis=1) -
                            np.mean(tmpi, axis=1) * np.mean(tmpj, axis=1))
                else:
                    self.cov[i, j] = self.cov[j, i]

        s = np.reshape(np.diag(self.cov), (self.cov.shape[0], 1))
        sst = np.dot(s, s.T)
        ok = np.where(sst)
        self.cor[ok] = self.cov[ok] / np.sqrt(sst[ok])
