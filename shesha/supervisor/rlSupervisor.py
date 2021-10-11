print("Loading RL supervisor")
## @package   shesha.supervisor.compassSupervisor
## @brief     Initialization and execution of a COMPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
## @date      2020/05/18
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.


verbose_multiple_independent_agents = False

from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, \
    WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration
import numpy as np

import shesha.constants as scons

from typing import Iterable

from src.reinforcement_learning.helper_functions.utils.utils import *
import shesha.ao.basis as bas
import torch

debug_rl_only_some = False
debug_only_rl = False

class RlSupervisor(CompassSupervisor):
    """ This class implements generic supervisor to handle compass simulation

    Attributes:
        context : (CarmaContext) : a CarmaContext instance

        config : (config) : Parameters structure

        telescope : (TelescopeComponent) : a TelescopeComponent instance

        atmos : (AtmosComponent) : An AtmosComponent instance

        target : (targetComponent) : A TargetComponent instance

        wfs : (WfsComponent) : A WfsComponent instance

        dms : (DmComponent) : A DmComponent instance

        rtc : (RtcComponent) : A Rtc component instance

        is_init : (bool) : Flag equals to True if the supervisor has already been initialized

        iter : (int) : Frame counter

        cacao : (bool) : CACAO features enabled in the RTC

        basis : (ModalBasis) : a ModalBasis instance (optimizer)

        calibration : (Calibration) : a Calibration instance (optimizer)

        ---- RL Attributes ----

        config_rl:

        Attributes related to which Btt modes we use:
            n_modes_backward: number of Btt modes used starting from the end of the Btt vector
            n_modes_discarded: list of Btt modes discarded
            n_modes_forward: number of Btt modes used starting from the end of the Btt vector
            n_modes_start_end: number of Btt modes used starting from n_modes_start_end[0] to n_modes_start_end[1]
            include_tip_tilt: If we include TT with n_modes_forward or n_modes_start_end

        pure_delay_0: if we use pure delay 0
        autoencoder: autoencoder NN for denoising
        freedom_vector_actuator_space: action bounds vector in actuator space
        freedom_vector: action bounds vector in modal space
        modes2volts: matrix to change space modes2volts
        volts2modes: matrix to change spaces volts2modes
        initial_seed: initial seed of the simulation
        current_seed: current seed of the simulation

    """

    def __init__(self, config, config_rl,  *,
                 build_cmat_with_modes=True,
                 initial_seed=1234,
                 autoencoder=None,
                 cacao: bool = False):
        """ Instantiates a RlSupervisor object

        Args:
            config: (config module) : Configuration module
            config_rl: rl configuration

        Kwargs:
            initial_seed: seed to start experiments
            normalization_bool: if we use normalization
            autoencoder: autoencoder neural network for denoising
            cacao : (bool) : If True, enables CACAO features in RTC (Default is False)
                                      /!\ Requires OCTOPUS to be installed
        """
        self.cacao = cacao

        GenericSupervisor.__init__(self, config)
        self.basis = ModalBasis(self.config, self.dms, self.target)
        self.calibration = Calibration(self.config, self.tel, self.atmos, self.dms,
                                       self.target, self.rtc, self.wfs)

        self.config_rl = config_rl
        # TODO change
        if "which_basis" not in self.config_rl.env_rl.keys():
            self.config_rl.env_rl['which_basis'] = "Btt"

        self.n_modes_start_end = self.config_rl.env_rl['n_zernike_start_end']
        self.n_reverse_filtered_from_cmat = self.config_rl.env_rl['n_reverse_filtered_from_cmat']
        self.include_tip_tilt = self.config_rl.env_rl['include_tip_tilt']
        self.pure_delay_0 = self.config_rl.env_rl['modification_online']

        self.autoencoder = autoencoder
        self.freedom_vector_actuator_space = None
        self.freedom_vector = None

        dms = []
        p_dms = []
        if type(self.config.p_controllers) == list:
            for config_p_controller in self.config.p_controllers:
                if config_p_controller.get_type() != "geo":
                    for dm_idx in config_p_controller.get_ndm():
                        dms.append(self.dms._dms.d_dms[dm_idx])
                        p_dms.append(self.config.p_dms[dm_idx])

        else:
            dms = self.dms._dms.d_dms
            p_dms = self.config.p_dms


        # TODO KL2V, Btt Petal
        # if deformable mirrors are not only GEO do btt
        self.initial_seed = initial_seed
        self.current_seed = initial_seed
        if dms:
            self.modes2volts, self.volts2modes =\
                self.basis.compute_modes_to_volts_basis(dms=dms, p_dms=p_dms,
                                                        modal_basis_type=self.config_rl.env_rl['which_basis'])
            """
                    Btt : (np.ndarray(ndim=2,dtype=np.float32)) : Btt to Volts matrix (volts x modes shape)

                    P (projector) : (np.ndarray(ndim=2,dtype=np.float32)) : Volts to Btt matrix (modes x volts shape)
                """

            if build_cmat_with_modes and self.config_rl.env_rl['which_basis'] == "Btt":
                # if self.n_modes_forward > 0:
                #    discarded_modes = (self.modes2volts.shape[1] - 2) - self.n_modes_forward
                # else:
                #    discarded_modes = 0
                self.obtain_and_set_cmat_filtered(self.n_reverse_filtered_from_cmat)

            if self.config_rl.env_rl['which_basis'] == "Btt":
                self.projector_phase2modes = get_projector_wfsphase2modes(supervisor=self,
                                                                          v2m=self.volts2modes,
                                                                          m2v=self.modes2volts)

                self.projector_wfs2modes = get_projector_measurements2modes(imat=self.rtc.get_interaction_matrix(0),
                                                                            v2m=self.volts2modes,
                                                                            m2v=self.modes2volts,
                                                                            nfilt=self.n_reverse_filtered_from_cmat)

            self.action_range_for_roket = self.obtain_action_range_modal(self.rtc.get_command(0))
    #
    #   __             __  __     _   _            _
    #  |   \          |  \/  |___| |_| |_  ___  __| |___
    #  |   |          | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |__/ efault    |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def load_freedom_vector_from_env(self, normalization_bool):
        self.freedom_vector = self.load_freedom_parameter_modal_space(normalization_bool)

    def set_sim_seed(self, seed):
        """
        Changes the current seed
        :param seed: seed to start generating random numbers
        :return: None
        """
        self.current_seed = seed

    def obtain_and_set_cmat_filtered(self, modes_filtered):
        """
        Obtaining cmat filtering modes for controller index 0
        TODO: comment
        """
        print("Obtaining cmat filtered...")
        print("Shape self.modes2volts", self.modes2volts.shape)

        if modes_filtered > -1:
            cmat = bas.compute_cmat_with_Btt(rtc=self.rtc._rtc,
                                             Btt=self.modes2volts,
                                             nfilt=modes_filtered)
            print("Number of filtered modes", modes_filtered)
        else:
            cmat = bas.compute_cmat_with_Btt(rtc=self.rtc._rtc,
                                             Btt=self.modes2volts,
                                             nfilt=0)
            print("Number of filtered modes", 0)

        return cmat

    def reset(self):
        """ Reset the simulation to return to its original state
        """
        self.past_command_rl = None
        self.atmos.reset_turbu(self.current_seed)
        self.wfs.reset_noise(self.current_seed)
        for tar_index in range(len(self.config.p_targets)):
            self.target.reset_strehl(tar_index)
        self.dms.reset_dm()
        self.rtc.open_loop()
        self.rtc.close_loop()

    #
    #   _              __  __     _   _            _
    #  | |            |  \/  |___| |_| |_  ___  __| |___
    #  | |_           | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |___| oad      |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def load_freedom_parameter_modal_space(self, normalization_bool):
        """
        Loads freedom vector (i.e. action bounds per mode)
        :param normalization_bool: if we use normalization or freedom parameter
        :return: freedom_vector: the vector of action bounds per mode
        """
        # TODO refactor names
        if normalization_bool:
            normalization_dir = "src/reinforcement_learning/helper_functions/preprocessing/normalization/"

            if self.config_rl.env_rl['custom_freedom_path'] is not None:
                freedom_vector_path = normalization_dir + self.config_rl.env_rl['custom_freedom_path'] + ".npy"
            else:

                freedom_vector_path = self.config_rl.env_rl['parameters_telescope'][:-3]
                freedom_vector_path = normalization_dir + "normalization_action_zernike/zn_norm_" \
                           + freedom_vector_path + ".npy"
        else:
            freedom_vector_path = None

        if freedom_vector_path is not None:
            print("Loading zernike freedom parameter from", freedom_vector_path)
            freedom_vector = np.load(freedom_vector_path)
            freedom_vector /= self.config_rl.env_rl['norm_scale_zernike_actions']
        else:
            freedom_vector = None

        return freedom_vector

    #     ___                  _      __  __     _   _            _
    #    / __|___ _ _  ___ _ _(_)__  |  \/  |___| |_| |_  ___  __| |___
    #   | (_ / -_) ' \/ -_) '_| / _| | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #    \___\___|_||_\___|_| |_\__| |_|  |_\___|\__|_||_\___/\__,_/__/

    def _init_tel(self):
        """Initialize the telescope component of the supervisor as a TelescopeCompass
        """
        self.tel = TelescopeCompass(self.context, self.config)

    def _init_atmos(self):
        """Initialize the atmosphere component of the supervisor as a AtmosCompass
        """
        self.atmos = AtmosCompass(self.context, self.config)

    def _init_dms(self):
        """Initialize the DM component of the supervisor as a DmCompass
        """
        self.dms = DmCompass(self.context, self.config)

    def _init_target(self):
        """Initialize the target component of the supervisor as a TargetCompass
        """
        if self.tel is not None:
            self.target = TargetCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_wfs(self):
        """Initialize the wfs component of the supervisor as a WfsCompass
        """
        if self.tel is not None:
            self.wfs = WfsCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_rtc(self):
        """Initialize the rtc component of the supervisor as a RtcCompass
        """
        if self.wfs is not None:
            self.rtc = RtcCompass(self.context, self.config, self.tel, self.wfs,
                                  self.dms, self.atmos, cacao=self.cacao)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def generic_delay_0_next(self, *, move_atmos: bool = True, ncontrol: int = 0,
                             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
                             do_control: bool = True, apply_control: bool = True,
                             compute_tar_psf: bool = True) -> None:
        """Iterates the AO loop, with optional parameters

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration
        """
        if tar_trace is None and self.target is not None:
            tar_trace = range(len(self.config.p_targets))
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.config.p_wfss))

        if move_atmos and self.atmos is not None:
            self.atmos.move_atmos()

        if tar_trace is not None:
            for t in tar_trace:
                if self.atmos.is_enable:
                    self.target.raytrace(t, tel=self.tel, atm=self.atmos, dms=self.dms)
                else:
                    self.target.raytrace(t, tel=self.tel, dms=self.dms)

        if wfs_trace is not None:
            for w in wfs_trace:
                if self.atmos.is_enable:
                    self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
                else:
                    self.wfs.raytrace(w, tel=self.tel)

                if not self.config.p_wfss[w].open_loop and self.dms is not None:
                    self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
                self.wfs.compute_wfs_image(w)

        if do_control and self.rtc is not None:
            self.rtc.do_centroids(ncontrol)
            self.rtc.do_control(ncontrol)

        if apply_control:
            self.rtc.apply_control(ncontrol)

            # Pure delay 0 for this generic test
            self.raytrace_target(ncontrol)

        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

                if self.pure_delay_0:
                    self.raytrace_target(ncontrol)

        self.iter += 1

    def next(self, *, move_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True) -> None:
        """Iterates the AO loop, with optional parameters.

        Overload the GenericSupervisor next() method to handle the GEO controller
        specific raytrace order operations

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration
        """
        if (
                self.config.p_controllers is not None and
                self.config.p_controllers[nControl].type == scons.ControllerType.GEO):
            if tar_trace is None and self.target is not None:
                tar_trace = range(len(self.config.p_targets))
            if wfs_trace is None and self.wfs is not None:
                wfs_trace = range(len(self.config.p_wfss))

            if move_atmos and self.atmos is not None:
                self.atmos.move_atmos()

            if tar_trace is not None:
                for t in tar_trace:
                    if self.atmos.is_enable:
                        self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
                    else:
                        self.target.raytrace(t, tel=self.tel, ncpa=False)

                    if do_control and self.rtc is not None:
                        self.rtc.do_control(nControl, sources=self.target.sources)
                        self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
                        if apply_control:
                            self.rtc.apply_control(nControl)
                        if self.cacao:
                            self.rtc.publish()
            if compute_tar_psf:
                for tar_index in tar_trace:
                    self.target.comp_tar_image(tar_index)
                    self.target.comp_strehl(tar_index)

            self.iter += 1

        else:
            GenericSupervisor.next(self, move_atmos=move_atmos, nControl=nControl,
                                   tar_trace=tar_trace, wfs_trace=wfs_trace,
                                   do_control=do_control, apply_control=apply_control,
                                   compute_tar_psf=compute_tar_psf)

    #   ___  ___                  __  __     _   _            _
    #  |   \|  |                 |  \/  |___| |_| |_  ___  __| |___
    #  | |\    |                 | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |_|  \__|  ormalization   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def next_geometric_normalization(self,
                                     ncontrol: int,
                                     geometric_do_control: bool = True,
                                     geometric_apply_control: bool = True,
                                     compute_geometric_from_projection: bool = False):
        """
        Advances an iteration for normalization with the geometric controller
        :param ncontrol: the controller that is currently used
        :param geometric_do_control: if calculate geometric commands
        :param geometric_apply_control: if apply to dm geometric commands
        :param compute_geometric_from_projection: if we use the projection matrix from wfs phase to modes
        :return: None
        """
        t = ncontrol
        if self.atmos.is_enable:
            self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
        else:
            self.target.raytrace(t, tel=self.tel, ncpa=False)

        if geometric_do_control and self.rtc is not None:
            self.rtc.do_control(ncontrol, sources=self.target.sources, source_index=t)
            # self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
            if False and hasattr(self, "projector"):
                wfs_phase = self.wfs.get_wfs_phase(0)
                wfs_phase = wfs_phase - wfs_phase.mean()
                pupil = self.get_m_pupil()
                wfs_phase_reshaped = wfs_phase[np.where(pupil)]
                modes = -self.projector_phase2modes.dot(wfs_phase_reshaped)
                volts = self.modes2volts.dot(modes)
                self.rtc.set_command(ncontrol, volts)
                if geometric_apply_control:
                    self.rtc.apply_control(ncontrol, comp_voltage=False)
            else:
                if geometric_apply_control:
                    self.rtc.apply_control(ncontrol)

            # Changed raytrace from before apply_control to after to make geo pure delay 0
            self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)

            # if self.cacao:
            #    self.rtc.publish()

    def next_integrator_normalization(self,
                                      linear_control_through_modal,
                                      ncontrol: int,
                                      do_control: bool = True,
                                      apply_control: bool = True,
                                      compute_geometric_from_projection: bool = False,
                                      compute_geometric_from_projection_initial: bool = False
                                      ):
        """
        Advances an iteration for normalization with the integrator controller
        :param linear_control_through_modal TODO
        :param ncontrol: the controller that is currently used
        :param do_control: if calculate integrator commands
        :param apply_control: if apply to dm integrator commands
        :param compute_geometric_from_projection: if we use the projection matrix from wfs phase to modes
        Note: if compute_geometric_from_projection is activated linear integrator will not work
        :return: None
        """
        t = ncontrol
        if compute_geometric_from_projection:
            if self.atmos.is_enable:
                self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
            else:
                self.target.raytrace(t, tel=self.tel, ncpa=False)

        else:
            if not self.pure_delay_0:
                self.raytrace_target(t)

        w = ncontrol
        if self.atmos.is_enable:
            self.wfs.raytrace(w, atm=self.atmos, tel=self.tel, ncpa=True)
        else:
            self.wfs.raytrace(w, tel=self.tel, ncpa=True)

        wfs_phase = self.wfs.get_wfs_phase(0)
        wfs_phase = wfs_phase - wfs_phase.mean()
        pupil = self.get_m_pupil()
        wfs_phase_reshaped = wfs_phase[np.where(pupil)]
        projection_modes = -self.projector_phase2modes.dot(wfs_phase_reshaped)
        projection_volts = self.modes2volts.dot(projection_modes)

        if not self.config.p_wfss[w].open_loop and self.dms is not None:
            self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)

        self.wfs.compute_wfs_image(w)
        if do_control and self.rtc is not None:
            self.rtc.do_centroids(ncontrol)
            self.rtc.do_control(ncontrol)

        if compute_geometric_from_projection_initial:
            wfs_phase = self.wfs.get_wfs_phase(0)
            wfs_phase = wfs_phase - wfs_phase.mean()
            pupil = self.get_m_pupil()
            wfs_phase_reshaped = wfs_phase[np.where(pupil)]
            volts = -self.projector_phase2modes.dot(wfs_phase_reshaped)

            try:
                volts = self.modes2volts.dot(volts)
                self.rtc.set_command(ncontrol, volts)
            except:
                self.rtc.set_command(ncontrol, volts)

            if apply_control:
                self.rtc.apply_control(ncontrol, comp_voltage=False)
                if self.pure_delay_0:
                    self.raytrace_target(ncontrol)
        else:
            if apply_control:
                self.rtc.apply_control(ncontrol)

                if self.pure_delay_0:
                    self.raytrace_target(ncontrol)

            if compute_geometric_from_projection:
                self.dms.set_command(projection_volts[:-2], dm_index=0, shape_dm=True)
                self.dms.set_command(projection_volts[-2:], dm_index=2, shape_dm=True)

        if compute_geometric_from_projection:
            self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)

        integrator_modes = self.volts2modes.dot(self.rtc.get_voltages(0))

        return integrator_modes, projection_modes

    def next_normalization(self, linear_control_through_modal,
                           *,
                           move_atmos: bool = True,
                           tar_trace: Iterable[int] = None,
                           do_control: bool = True, apply_control: bool = True,
                           compute_tar_psf: bool = True,
                           compute_geometric_from_projection: bool = False) -> tuple:
        """Iterates the AO loop, with optional parameters

        Kwargs:
            linear_control_through_modal: TODO

            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration

            compute_geometric_from_projection: (bool) :
        """
        integrator_modes, projection_modes = [], []

        if tar_trace is None and self.target is not None:
            tar_trace = range(len(self.config.p_targets))

        if move_atmos and self.atmos is not None:
            self.atmos.move_atmos()

        for ncontrol in range(len(self.config.p_controllers)):
            if (
                    self.config.p_controllers is not None and
                    self.config.p_controllers[ncontrol].type == scons.ControllerType.GEO):
                # checking that the only controller is not geo and linear_control_through_modal is activated
                assert (not (linear_control_through_modal and ncontrol == 0))
                self.next_geometric_normalization(ncontrol=ncontrol,
                                                  compute_geometric_from_projection=compute_geometric_from_projection)
            else:
                integrator_modes, projection_modes =\
                    self.next_integrator_normalization(linear_control_through_modal=linear_control_through_modal,
                                                       ncontrol=ncontrol,
                                                       do_control=do_control,
                                                       apply_control=apply_control,
                                                       compute_geometric_from_projection=compute_geometric_from_projection)

        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

        if compute_geometric_from_projection and self.iter % 100 == 0:
            _, se_modified, _, _ = self.target.get_strehl(0)
            try:
                _, se_geo, _, _ = self.target.get_strehl(1)
                print(self.iter, se_modified, se_geo)
            except:
                print(self.iter, se_modified)

        self.iter += 1
        return integrator_modes, projection_modes
    #   _____   __        __  __     _   _            _
    #  |  _  | |  |      |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__    | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _|   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #                            I
    #   _____   __               __  __     _   _            _
    #  |  _  | |  |             |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__           | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _| control  |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #
    #       I a) Methods used by only RL and correction
    #
    #
    #

    def obtain_action_range_modal(self, final_command_modal):
        """

        :param final_command_modal: #TODO
        :return:
        """
        if self.n_modes_start_end[0] >= 0:
            if self.include_tip_tilt:
                action_range = list(range(self.n_modes_start_end[0], self.n_modes_start_end[1])) + [-2, -1]
            else:
                action_range = list(range(self.n_modes_start_end[0], self.n_modes_start_end[1]))
        else:
            action_range = list(range(len(final_command_modal)))

        return action_range

    def rl_control_single_step(self,
                               action: np.ndarray,
                               ncontrol: int):
        """
        :param ncontrol TODO
        :return:
        """
        assert self.config.p_controllers[ncontrol].delay == 1
        assert not self.pure_delay_0

        if self.past_command_rl is not None:
            volts = self.rtc.get_voltages(0)
            modes = self.volts2modes.dot(volts)
            action_range = self.obtain_action_range_modal(volts)
            modes[action_range] = modes[action_range] + (self.past_command_rl * self.freedom_vector[action_range])
            volts_modified = self.modes2volts.dot(modes)
            self.dms.set_command(commands=volts_modified[:-2], dm_index=0)
            self.dms.set_command(commands=volts_modified[-2:], dm_index=2)
        self.past_command_rl = action

    def rl_control(self,
                   action: np.ndarray,
                   ncontrol: int,
                   evaluation_rl_full_action: bool):
        """
        Calculates the final action to be applied to the DM considering the output of the RL agent
        :param action: action to apply
        :param ncontrol:  controller to be updated
        :return: None
        """

        # Scale action
        action = (action * self.config_rl.env_rl['normalization_std_inside_environment']) + \
                 self.config_rl.env_rl['normalization_mean_inside_environment']

        if self.config_rl.env_rl['level'] == "correction":
            final_command = self.correction_control(action=action, ncontrol=ncontrol, evaluation_rl_full_action=evaluation_rl_full_action)
        else:
            raise NotImplementedError

        self.rtc.set_command(ncontrol, final_command)

    #
    #       I b) Methods only used by correction
    #
    #
    #

    def correction_actuator_basis(self,
                                  action: np.ndarray,
                                  ncontrol: int):
        """
        Calculates final action after modification using actuator space
        :param action: action to apply
        :param ncontrol: controller to be updated
        :return: final_command
        """
        if self.freedom_vector_actuator_space is not None:
            action = np.array(action) * self.freedom_vector_actuator_space

        final_command_actuator = self.rtc.get_command(ncontrol) + action

        return final_command_actuator

    def correction_modal_deactivate_unused_modes(self,
                                                 final_command: np.ndarray):
        """

        :param final_command: array of commands to deactivate unused modes
        :return: final_command with deactivated unused modes
        """

        if self.n_modes_backward > 0:
            final_command[:(len(final_command) - self.n_modes_backward)] *= 0
        elif self.n_modes_start_end[0] >= 0:
            if self.include_tip_tilt:
                final_command[:self.n_modes_start_end[0]] *= 0
                final_command[self.n_modes_start_end[1]:-2] *= 0
            else:
                final_command[:self.n_modes_start_end[0]] *= 0
                final_command[self.n_modes_start_end[1]:] *= 0
        elif self.n_modes_discarded is not None:
            final_command[self.n_modes_discarded] *= 0
        else:
            if self.include_tip_tilt:
                final_command[self.n_modes_forward:-2] *= 0
            else:
                final_command[self.n_modes_forward:] *= 0

        return final_command

    def correction_modal_basis(self,
                               action: np.ndarray,
                               ncontrol: int,
                               evaluation_rl_full_action: bool):
        """
        Calculates final action after modification using modal space
        :param action: action to apply
        :param ncontrol: controller to be updated
        :return: final_command

        Notes:
        volts2modes shape is modes x volts
        modes2volts shape is volts x modes
        """

        # 1. Btt basis starts with commands in modal space
        final_command_modal = self.volts2modes.dot(self.rtc.get_command(ncontrol))

        if self.n_modes_start_end[0] < 0:
            # 2.a If we use all modes. Sum the action multiplied by the freedom vector.
            final_command_modal += (action * self.freedom_vector)
        else:
            # 2.b.1 If we do not use all the modes check how we discard them and build action_range
            # (a list with the modes we are using)
            action_range = self.obtain_action_range_modal(final_command_modal)
            if self.config_rl.env_rl['tt_treated_as_mode']:
                final_command_modal += (action * self.freedom_vector)
            else:
                # 2.b.2 Sum the action multiplied by the freedom vector.
                final_command_modal[action_range] += (action * self.freedom_vector[action_range])

        # 3.b.2 We send the modes to actuator space
        final_command = self.modes2volts.dot(final_command_modal)

        return final_command

    def correction_control(self,
                           action: np.ndarray,
                           ncontrol: int,
                           evaluation_rl_full_action: bool):
        """
        Adds the correction to the integrator controller with a RL agent action
        :param action: action to apply
        :param ncontrol: controller to be updated
        :return: final_command
        """

        if self.config_rl.env_rl['basis'] == "zernike_space":
            final_command = self.correction_modal_basis(action=action, ncontrol=ncontrol, evaluation_rl_full_action=evaluation_rl_full_action)
        else:
            raise NotImplementedError

        return final_command

    #
    #   ___  ___         __  __     _   _            _
    #  |   \|  |        |  \/  |___| |_| |_  ___  __| |___
    #  | |\    |        | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |_|  \__|  ext   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def raytrace_target(self, ncontrol: int):
        """
        Does the raytacing operation
        :param ncontrol: ncontrol that will have an associated target
        :return: None
        """
        t = ncontrol
        if self.atmos.is_enable:
            self.target.raytrace(t, tel=self.tel, atm=self.atmos, dms=self.dms)
        else:
            self.target.raytrace(t, tel=self.tel, dms=self.dms)

    def build_wfs_image_from_bincube(self, denoised_bincube):
        """
        Given a bincube build a denoised binimage
        :param denoised_bincube: the denoised WFS image in 3D
        :return: the denoised WFS image in 2d
        """
        # TODO check
        first_xpixels = np.array(self.wfs._wfs.d_wfs[0].d_validsubsx)
        first_ypixels = np.array(self.wfs._wfs.d_wfs[0].d_validsubsy)

        npixels = self.config.p_wfss[0].npix
        denoised_binimage = self.wfs.get_wfs_image(0)

        for i in range(len(first_xpixels)):
            denoised_binimage[first_xpixels[i]:first_xpixels[i]+npixels, first_ypixels[i]:first_ypixels[i]+npixels] = \
                denoised_bincube[i, :, :]

        return denoised_binimage

    def autoencoder_denoising(self):
        """
        Denoises WFS image using denoising autoencoder.
        Dimensions input autoencoder (Bsz, Channels, Height of image in pixels, W is width of image in pixels)
        Hence we will have images of size (Bsz, 1, 16, 16)
        :return: None
        """
        if self.autoencoder.type == "cnn_single_subaperture":
            image_to_denoise = np.array(self.wfs._wfs.d_wfs[0].d_bincube)  # 16, 16, 1200
            image_to_denoise = np.moveaxis(image_to_denoise, -1, 0)  # 1200, 16, 16
            noisy_tensor = torch.tensor(image_to_denoise).to(device=self.autoencoder.device)
            denoised_bincube = self.autoencoder.predict(noisy_tensor)
            denoised_binimage = self.build_wfs_image_from_bincube(denoised_bincube)
            self.wfs._wfs.d_wfs[0].set_binimg(denoised_binimage, denoised_binimage.size)
        else:
            raise NotImplementedError


    #
    #           II a) PART TWO METHODS
    #
    #
    #

    def next_part_two(self,
                      action: np.ndarray,
                      linear_control: bool = False,
                      tar_trace: Iterable[int] = None,
                      apply_control: bool = True,
                      compute_tar_psf: bool = True,
                      evaluation_rl_full_action: bool = False
                      ):
        """
        Calculate RL command (if applicable) + Apply Control + Compute Tar Image and Strehl
        :param action: action of RL
        :param linear_control: if we only do linear control (plus geo if applicable)
        :param tar_trace: number of targets
        :param apply_control: if the DM applies the voltage to the DM
        :param compute_tar_psf: if the PSF is calculated
        :return: None
        """

        if tar_trace is None and self.target is not None:
            tar_trace = range(len(self.config.p_targets))

        for ncontrol in range(len(self.config.p_controllers)):

            if self.config.p_controllers[ncontrol].type != scons.ControllerType.GEO:
                if not linear_control: # and not self.config_rl.env_rl['rl_control_single_step']:
                    self.rl_control(action, ncontrol, evaluation_rl_full_action)
                elif linear_control\
                        and self.config_rl.env_rl['level'] == "only_rl"\
                        and self.config_rl.env_rl['only_rl_integrated']\
                        and self.config.p_controllers[ncontrol].type != scons.ControllerType.GEO:
                    # In the case of only_rl + only_rl integrated linear_control needs to do
                    # do_control before apply control
                    self.rtc.do_control(ncontrol)
                    if debug_only_rl:
                        print("Linear control only rl, only rl integrated")
                if apply_control:
                    self.rtc.apply_control(ncontrol)

                    if self.pure_delay_0:
                        self.raytrace_target(ncontrol)

        # if self.cacao: TODO check cacao false always
        #    self.rtc.publish()

        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

    #
    #           II b) PART ONE METHODS
    #
    #

    def next_part_one_integrator(self, *,
                                 ncontrol: int = 0,
                                 do_control: bool = True):
        """F
        raytrace target (if not pure delay 0) + raytrace WFS + comp image WFS + autoencoder denoising (if applicable)
        + do Centroids + do Control integrator
        :param ncontrol: controller to use
        :param do_control: if the integrator calculates the commands
        :return: None
        """
        if not self.pure_delay_0:
            self.raytrace_target(ncontrol)

        w = ncontrol
        if self.atmos.is_enable:
            self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
        else:
            self.wfs.raytrace(w, tel=self.tel)

        if not self.config.p_wfss[w].open_loop and self.dms is not None:
            self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
        self.wfs.compute_wfs_image(w)

        if self.autoencoder is not None:
            self.autoencoder_denoising()

        if do_control and self.rtc is not None:
            self.rtc.do_centroids(ncontrol)
            if not (self.config_rl.env_rl['level'] == "only_rl"
                    and self.config_rl.env_rl['only_rl_integrated']):
                self.rtc.do_control(ncontrol)
            elif debug_only_rl:
                print("Not doing control step 1")
            # self.rtc.do_clipping(ncontrol) TODO do_clipping check ok

    def next_part_one_geo(self, *,
                          ncontrol: int = 0,
                          do_control: bool = True,
                          geometric_apply_control: bool = True):
        """
        does the GEOMETRIC controller raytrace + docontrol + applycontrol
        :param ncontrol: controller to use
        :param do_control: if control is calculated
        :param geometric_apply_control: if control is applied
        :return: None
        """
        t = ncontrol
        if self.atmos.is_enable:
            self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
        else:
            self.target.raytrace(t, tel=self.tel, ncpa=False)

        if do_control and self.rtc is not None:
            self.rtc.do_control(ncontrol, sources=self.target.sources, source_index=t)
            # self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
            if geometric_apply_control:
                self.rtc.apply_control(ncontrol)

            # Changed raytrace from before apply_control to after to make geo pure delay 0
            self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)

    def next_part_one(self, *,
                      move_atmos: bool = True,
                      tar_trace: Iterable[int] = None,
                      wfs_trace: Iterable[int] = None,
                      do_control: bool = True,
                      geometric_apply_control: bool = True) -> None:
        """Iterates the AO loop part one, with optional parameters.

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            geometric_apply_control: (bool) : If geometric controller applies control
        """

        if move_atmos and self.atmos is not None:
            self.atmos.move_atmos()

        for ncontrol in range(len(self.config.p_controllers)):
            if (
                    self.config.p_controllers is not None and
                    self.config.p_controllers[ncontrol].type == scons.ControllerType.GEO):
                self.next_part_one_geo(ncontrol=ncontrol,
                                       do_control=do_control,
                                       geometric_apply_control=geometric_apply_control
                                       )

            else:
                self.next_part_one_integrator(ncontrol=ncontrol,
                                              do_control=do_control)

        self.iter += 1
