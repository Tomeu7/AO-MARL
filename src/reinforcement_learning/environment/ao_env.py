import matplotlib.pyplot as plt
import gym
from gym import spaces
import numpy as np
import math
import pickle
from collections import deque, OrderedDict
import shesha.constants as scons
from src.autoencoder.autoencoder_models import Autoencoder
from scipy.ndimage.measurements import center_of_mass
from shesha.util.utilities import load_config_from_file
from src.reinforcement_learning.helper_functions.preprocessing.normalization.obtain_normalization \
   import run_obtain_normalization_and_freedom
import os
from matplotlib.gridspec import GridSpec


class AoEnv(gym.Env):
    """
    Adaptive Optics Environment using COMPASS simulator

    Attributes:

        normalization_bool: if normalization is used

        supervisor: class that manages the whole simulation

        s_wfs_history: deque containing past wfs

        s_dm_history: deque containing past commands

        wfs_shape: the shape of the measurements vector

        dm_shape: the shape of the commands vector

        reward_type: the type of reward we are using e.g. (wavefront_phase_error or avg_square_m)

        config_rl: Rl config parameters
    """

    #            Initialization
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def __init__(self, config_rl,
                 normalization_bool=True,
                 build_cmat_with_modes=True,
                 initial_seed=-1,
                 geo_policy_testing=False,
                 roket=False):
        """
        config_rl: Rl config parameters
        normalization_bool: if we normalize, bool
        initial_seed: the initial seed we use, int
        geo_policy_testing: if we are using a geo controller (only used in the final testing of a policy)
        """
        super(AoEnv, self).__init__()
        self.fig = None
        self.normalization_bool = normalization_bool

        # a) Compass Initialization

        self.supervisor = self.compass_init(config_rl=config_rl,
                                            geo_policy_testing=geo_policy_testing,
                                            initial_seed=initial_seed,
                                            build_cmat_with_modes=build_cmat_with_modes,
                                            roket=roket)

        if self.normalization_bool:
            self.norm_parameters = self.load_norm_parameters(config_rl)

        # c) Initializing env properties for future use

        # We initializy the history
        self.s_wfs_history = deque(maxlen=config_rl.env_rl['number_of_previous_wfs'])
        # [np.zeros(self.wfs_shape)] * config.env_rl['number_of_previous_dm']

        self.s_dm_history = deque(maxlen=config_rl.env_rl['number_of_previous_dm'])
        # [np.zeros(self.dm_shape)] * config.env_rl['number_of_previous_wfs']

        self.s_dm_residual_history = deque(maxlen=config_rl.env_rl['number_of_previous_dm_residuals'])

        # b) Defining state and action spaces

        self.wfs_shape, self.dm_shape, self.observation_space, self.action_space \
            = self.define_state_action_space(config_rl)

        self.reward_type = config_rl.env_rl['reward_type']
        self.config_rl = config_rl
        self.verbose = config_rl.env_rl['verbose']

    def compass_init(self, config_rl, geo_policy_testing, initial_seed, build_cmat_with_modes, roket):
        """
        Creates Supervisor that will manage the simulation
        :param config_rl: configuration for RL experiment
        :param geo_policy_testing: if we are doing geo policy testing
        :param initial_seed: initial seed
        :return:
        """
        print("------------------------Initialization of COMPASS------------------------")

        # 1. Loading Compass config
        arguments = self.process_arguments(config_rl, geo_policy_testing)
        param_file = arguments["<parameters_filename>"]
        config_compass = load_config_from_file(param_file)

        # 2. Loading Denoising Autoencoder if applicable
        if config_rl.autoencoder['path'] is not None:
            autoencoder = Autoencoder(config_rl)
        else:
            autoencoder = None

        # 3. Creating the Supervisor
        if roket:
            from guardians.roket_generalized_rl import Roket as Supervisor
            supervisor = Supervisor(config=config_compass,
                                    config_rl=config_rl,
                                    initial_seed=initial_seed,
                                    autoencoder=autoencoder,
                                    cacao=False
                                    )
        else:
            from shesha.supervisor.rlSupervisor import RlSupervisor as Supervisor
            supervisor = Supervisor(config=config_compass,
                                    config_rl=config_rl,
                                    build_cmat_with_modes=build_cmat_with_modes,
                                    initial_seed=initial_seed,
                                    autoencoder=autoencoder,
                                    cacao=False
                                    )

        # TODO check if I can remove
        # if arguments["--devices"]:
        #    self.supervisor.config.p_loop.set_devices([
        #        int(device) for device in arguments["--devices"].split(",")
        #    ])
        # if arguments["--generic"]:
        #    self.supervisor.config.p_controllers[0].set_type("generic")
        #    print("Using GENERIC controller...")

        # self.supervisor.initConfig(self.normalization_bool, autoencoder=autoencoder)
        print("-------------------------------------------------------------------------")
        return supervisor

    def define_state_action_space(self, config_rl):
        """
        Defines state action space depending on the configuration
        :param config_rl: configuration of RL properties
        """
        wfs_shape = self.supervisor.rtc.get_slopes(0).shape

        if config_rl.env_rl['basis'] == "actuator_space":
            dm_shape = self.supervisor.rtc.get_command(0).shape
            command_shape = self.supervisor.rtc.get_command(0).shape
            # residual commands and full command have the same shape
            command_residual_shape = command_shape
            command_shape_for_action = command_shape
        elif config_rl.env_rl['basis'] == "zernike_space":
            if config_rl.env_rl['n_zernike_start_end'][0] >= 0:
                if config_rl.env_rl['include_tip_tilt']:
                    command_shape = [
                        config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0] + 2]
                else:
                    command_shape = [
                        config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0]]
            else:
                command_shape = [self.supervisor.modes2volts.shape[1]]

            if config_rl.env_rl['window_n_zernike'] > -1 \
                    or config_rl.env_rl['tt_treated_as_mode']:
                command_shape_for_state = self.supervisor.volts2modes.dot(self.supervisor.rtc.get_command(0)).shape
            else:
                command_shape_for_state = command_shape

            command_shape_for_action = command_shape

            dm_shape = command_shape_for_state
            command_residual_shape = command_shape_for_state
            print("\n Total modes", self.supervisor.modes2volts.shape[1],
                  "\n Number that we are using for state", command_shape_for_state[0],
                  "\n Number that we are using for actions", command_shape_for_action[0])
        else:
            raise NotImplementedError("This space is not implemented")

        state_size = 0
        state_size += dm_shape[0] * config_rl.env_rl['number_of_previous_dm']
        state_size += wfs_shape[0] * config_rl.env_rl['number_of_previous_wfs']
        state_size += dm_shape[0] * config_rl.env_rl['number_of_previous_dm_residuals']

        if config_rl.env_rl['state_dm_before_linear']:
            state_size += dm_shape[0]
        if config_rl.env_rl['state_dm_after_linear']:
            state_size += dm_shape[0]
        if config_rl.env_rl['state_wfs']:
            state_size += wfs_shape[0]
        if config_rl.env_rl['state_dm_residual']:
            state_size += command_residual_shape[0]

        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=[state_size], dtype=np.float64)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=command_shape_for_action, dtype=np.float64)

        return wfs_shape, dm_shape, observation_space, action_space

    def process_arguments(self, config_rl, geo_policy_testing):
        """
        Loads config file for the simulation which should have been set up before.
        :param config_rl: configuration for RL experiment
        :param geo_policy_testing: if we are doing geo policy testing
        """
        arguments = dict()

        different_atmos_pars = "r8v" in config_rl.env_rl['parameters_telescope'] or "r16v" \
                               in config_rl.env_rl['parameters_telescope'] or "r24v" in config_rl.env_rl[
                                   'parameters_telescope']

        if 'production' in config_rl.env_rl['parameters_telescope']:
            fd_parameters = "data/par/par4rl/production/"
            if geo_policy_testing:
                fd_parameters += "geo/"
                if different_atmos_pars:
                    fd_parameters += "different_atmospheric_conditions/"
                    file = config_rl.env_rl['parameters_telescope'][:-3] + "_geo.py"
                elif "v40" in config_rl.env_rl['parameters_telescope']:
                    file = config_rl.env_rl['parameters_telescope'][:-7] + "_geo.py"
                else:
                    file = config_rl.env_rl['parameters_telescope'][:-3] + "_geo.py"
            elif different_atmos_pars:
                fd_parameters += "different_atmospheric_conditions/"
                file = config_rl.env_rl['parameters_telescope']
            elif "v40" in config_rl.env_rl['parameters_telescope']:
                fd_parameters += "/wind_speed/"
                file = config_rl.env_rl['parameters_telescope']
            else:
                file = config_rl.env_rl['parameters_telescope']

            arguments["<parameters_filename>"] = fd_parameters + file
        else:
            raise NotImplementedError

        # arguments["--DB"] = False
        # arguments["--devices"] = None
        # arguments["--generic"] = False
        # TODO check if I can remove
        print("1. Parameter filename", arguments["<parameters_filename>"])
        return arguments

    def load_norm_parameters(self, config_rl):
        """
        Loads normalization parameters which should have been precomputed before.
        :param config_rl: configuration for RL experiment
        """

        # Create norm if needed
        if config_rl.env_rl['create_norm_param'] and self.normalization_bool:
            run_obtain_normalization_and_freedom(parameter_file=config_rl.env_rl['parameters_telescope'],
                                                 basis=config_rl.env_rl['basis'],
                                                 include_plots=False,
                                                 pure_delay_0=config_rl.env_rl['modification_online'],
                                                 autoencoder_path=config_rl.autoencoder['path'],
                                                 modes_filtered=config_rl.env_rl['n_reverse_filtered_from_cmat'])

        self.supervisor.load_freedom_vector_from_env(self.normalization_bool)

        norm_dir = "src/reinforcement_learning/helper_functions/preprocessing/normalization/state_normalization/"
        assert config_rl.env_rl['parameters_telescope'][-3:] == '.py'
        norm_path = config_rl.env_rl['parameters_telescope'][:-3]

        if config_rl.env_rl['basis'] == "zernike_space":
            norm_path = norm_path + "_zernike_space"

        print("Loading norm parameters from: " + norm_dir + "normalization_" + norm_path + ".pickle")
        with open(norm_dir + "normalization_" + norm_path + ".pickle", 'rb') as handle:
            norm_parameters = pickle.load(handle)

        # Changing parameter vector if we only choose a subset of Btt modes
        if config_rl.env_rl['window_n_zernike'] > -1\
                or config_rl.env_rl['tt_treated_as_mode']:
            pass
        elif config_rl.env_rl['n_zernike_start_end'][0] >= 0:
            if config_rl.env_rl['include_tip_tilt']:
                action_range = list(range(config_rl.env_rl['n_zernike_start_end'][0],
                                          config_rl.env_rl['n_zernike_start_end'][1])) + [-2, -1]
            else:
                action_range = list(range(config_rl.env_rl['n_zernike_start_end'][0],
                                          config_rl.env_rl['n_zernike_start_end'][1]))

            for key, item in norm_parameters['dm'].items():
                if config_rl.env_rl['include_tip_tilt']:
                    norm_parameters['dm'][key] = item[action_range]
                else:
                    norm_parameters['dm'][key] = item[action_range]
            if config_rl.env_rl['state_dm_residual']:
                for key, item in norm_parameters['dm_residual'].items():
                    if config_rl.env_rl['include_tip_tilt']:
                        norm_parameters['dm_residual'][key] = item[action_range]
                    else:
                        norm_parameters['dm_residual'][key] = item[action_range]

        for key, item in norm_parameters.items():
            print(key, item['mean'].shape)

        return norm_parameters

    #           Basic environment
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def reset(self,
              normalization_loop: bool = False,
              return_dict: bool = False,
              geometric_do_control: bool = False,
              geometric_apply_control: bool = True):
        """
        Resets the simulator with the given seed of the environment
        :param normalization_loop:
        :param return_dict: if the state is returned as dict
        :param geometric_do_control: if geo controller is used at all
        :param geometric_apply_control: if geo controller is applied
        :return s: initial state for RL agent
        """

        # a) Resets simulator, resets noise, resets iterations
        self.supervisor.reset()
        self.supervisor.iter = 0

        if not normalization_loop:
            # b) Resets wfs and dm history
            self.s_wfs_history.clear()
            self.s_dm_history.clear()
            self.s_dm_residual_history.clear()

            for _ in range(self.config_rl.env_rl['number_of_previous_dm']):
                self.s_dm_history.append(np.zeros(self.dm_shape[0], dtype="float64"))

            for _ in range(self.config_rl.env_rl['number_of_previous_wfs']):
                self.s_wfs_history.append(np.zeros(self.wfs_shape[0], dtype="float64"))

            for _ in range(self.config_rl.env_rl['number_of_previous_dm_residuals']):
                self.s_dm_residual_history.append(np.zeros(self.dm_shape[0], dtype="float64"))

            # TODO check Compass new version
            # if self.config_rl.env_rl['level'] == "only_rl":
            #    s = self.only_rl_step_part_1(return_dict)
            # elif self.config_rl.env_rl['level'] == "correction":
            # e) for correction level the initial state is given after one step of linear approach
            s = self.linear_step(return_dict, geometric_do_control, geometric_apply_control)

            if self.config_rl.env_rl['verbose']:
                print("Initial s", s)

            return s

    def render_full_state_information(self, state, action_mu, action_std, num_episode, timestep, title):
        """
        Renders state-action for a given timestep
        :param state: current state
        :param action_mu: the action normal distribution mean
        :param action_std: the action normal distribution std
        :param num_episode: number of the episode
        :param timestep: number of timestep
        :param title: title we want to give to the histogram
        :return: None
        """

        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 20))

        gs = GridSpec(4, 8)
        self.fig.suptitle("Ep=" + str(num_episode) + " t=" + str(timestep) + " " + title)
        state_ax = self.fig.add_subplot(gs[0:4, 0:3])
        state_ax.plot(state, ".")
        state_ax.set_xlabel("State feature")
        state_ax.set_ylim(bottom=-5, top=5)

        state_ax.axvline(ymin=state.min(),
                         ymax=state.max(),
                         x=self.wfs_shape[0],
                         linestyle="dashed",
                         color="black")

        for n_dm in range(1, len(self.s_dm_history) + 1):
            state_ax.axvline(ymin=state.min(),
                             ymax=state.max(),
                             x=self.wfs_shape[0] + n_dm * self.dm_shape[0],
                             linestyle="dashed",
                             color="black")

        ax_actions = self.fig.add_subplot(gs[0:4, 4:7])
        ax_count = self.fig.add_subplot(gs[0:4, 7])
        ax_actions.errorbar(np.arange(len(action_mu)), action_mu, yerr=action_std, fmt='o')
        weights = np.ones_like(action_mu) / len(action_mu)
        ax_count.hist(action_mu, orientation="horizontal", weights=weights)
        plt.setp(ax_count.get_yticklabels(), visible=False)
        ax_actions.set_xlabel('Action Feature')
        ax_actions.set_ylabel('Action Value')
        plt.xlabel("Action Feature")
        ax_count.set_xlabel('Count action, bins 10')

    def render(self,
               state,
               action_mu,
               timestep,
               num_episode,
               title,
               action_std=None,
               save_state_action_image=False,
               show_ao_images=False):
        """
        Renders the AO env
        :param state: current state
        :param action_mu: the action normal distribution mean
        :param timestep: number of timestep
        :param num_episode: number of the episode
        :param title: title we want to give to the histogram
        :param action_std: the action normal distribution std
        :param save_state_action_image:
        :param show_ao_images: if we show the image of the WFS and DM instead of state and actions points
        :return: None
        """
        if show_ao_images:
            plt.subplot(2, 1, 1)
            plt.imshow(self.supervisor.dms.get_dm_shape(0))
            plt.axis('off')
            plt.title("DM shape")

            plt.subplot(2, 1, 2)
            plt.imshow(self.supervisor.wfs.get_wfs_image(0))
            plt.axis('off')
            plt.title("WFS image")

            plt.pause(0.0001)
            plt.clf()
        else:
            self.render_full_state_information(state, action_mu, action_std, num_episode, timestep, title)
            plt.pause(0.01)
            if save_state_action_image:
                fd = "insights/render/" + self.config_rl.env_rl['parameters_telescope'][:-3]
                if not os.path.exists(fd):
                    os.mkdir(fd)
                plt.savefig(fd + "/Ep_" + str(num_episode) + "t" + str(timestep) + ".png")
                self.fig.clf()
            else:
                plt.pause(0.001)
                self.fig.clf()

    def set_sim_seed(self, seed):
        """
        Sets the seed of the simulator
        :param seed: new seed
        """
        self.supervisor.set_sim_seed(seed)

    def is_geometric_controller_present(self):
        """
        Checks if any controller is of geo type
        """
        for ncontrol in range(len(self.supervisor.rtc.d_control)):
            if self.supervisor.rtc.d_control[ncontrol].type == scons.ControllerType.GEO:
                return True
        return False

    def standardise(self, inpt, key):
        """
        standardizes
        :param inpt: state to be normalized
        :param key: "wfs" or "dm"
        :return: input normalized
        """

        mean = self.norm_parameters[key]['mean']
        std = self.norm_parameters[key]['std']
        return (inpt - mean) / std

    def transform_state_to_zernike(self, s_dm, return_reward=False):
        """
        Changes the DM values from actuator space to zernike space
        Also chooses a subset of zernike modes
        s_dm: state from volts
        return_reward: boolean used to ignore the flag of n_zernike_personalized_observation_start_end from config
        """
        s_dm = self.supervisor.volts2modes.dot(s_dm)

        if (self.config_rl.env_rl['window_n_zernike'] > -1
            or self.config_rl.env_rl['tt_treated_as_mode'])\
                and not return_reward:
            # We return the full state if we use "window_n_zernike"
            pass
        elif self.config_rl.env_rl['n_zernike_start_end'][0] >= 0:
            if self.config_rl.env_rl['include_tip_tilt']:
                action_range = list(range(self.config_rl.env_rl['n_zernike_start_end'][0],
                                          self.config_rl.env_rl['n_zernike_start_end'][1])) + [-2, -1]
            else:
                action_range = list(range(self.config_rl.env_rl['n_zernike_start_end'][0],
                                          self.config_rl.env_rl['n_zernike_start_end'][1]))
            s_dm = s_dm[action_range]

        return s_dm

    def add_s_dm_residual_to_state(self, s_next, s_dm_residual):

        # TODO .copy()??
        # 1. Add history of residuals to the state
        for idx in range(len(self.s_dm_residual_history)):
            past_s_dm_residual = self.s_dm_residual_history[idx].copy()
            if self.normalization_bool:
                past_s_dm_residual_norm = self.standardise(past_s_dm_residual, key="dm_residual")
                s_next["dm_residual_history_" + str(len(self.s_dm_residual_history) - idx)] =\
                    past_s_dm_residual_norm.copy()
            else:
                s_next["dm_residual_history_" + str(len(self.s_dm_residual_history) - idx)] =\
                    past_s_dm_residual.copy()

        # 2. Add current residual to the history
        if self.config_rl.env_rl['number_of_previous_dm'] > 0:
            self.s_dm_residual_history.append(s_dm_residual.copy())

        # 3 Add current residual to the state
        if self.config_rl.env_rl['state_dm_residual']:
            if self.normalization_bool:
                s_dm_residual_norm = self.standardise(s_dm_residual.copy(), key="dm_residual")
                s_next["dm_residual"] = s_dm_residual_norm.copy()
            else:
                s_next["dm_residual"] = s_dm_residual.copy()

        return s_next

    def add_dm_to_state(self, s_next, s_dm_before_linear, s_dm_after_linear=None):
        """
        Creates the DM state depending on config parameters
        """

        for idx in range(len(self.s_dm_history)):
            past_s_dm = self.s_dm_history[idx]
            if self.normalization_bool:
                past_s_dm_norm = self.standardise(past_s_dm, key="dm")
                s_next["dm_history_" + str(len(self.s_dm_history) - idx)] = past_s_dm_norm
            else:
                s_next["dm_history_" + str(len(self.s_dm_history) - idx)] = past_s_dm

        if self.config_rl.env_rl['number_of_previous_dm'] > 0:
            self.s_dm_history.append(s_dm_before_linear)

        if self.config_rl.env_rl['state_dm_after_linear'] and s_dm_after_linear is not None:
            if self.normalization_bool:
                s_dm_after_linear = self.standardise(s_dm_after_linear, key="dm")
            s_next["dm_after_linear"] = s_dm_after_linear

        if self.config_rl.env_rl['state_dm_before_linear']:
            if self.normalization_bool:
                s_dm_before_linear = self.standardise(s_dm_before_linear, key="dm")
            s_next["dm_before_linear"] = s_dm_before_linear

        return s_next

    def add_wfs_to_state(self, s_next, s_wfs):
        """
        Creates the wfs state depending on config parameters
        """

        for idx in range(len(self.s_wfs_history)):
            past_s_wfs = self.s_wfs_history[idx]
            if self.normalization_bool:
                past_s_wfs_norm = self.standardise(past_s_wfs.copy(), key="wfs")
                s_next["wfs_history-" + str(len(self.s_wfs_history) - idx)] = past_s_wfs_norm
            else:
                s_next["wfs_history_" + str(len(self.s_wfs_history) - idx)] = past_s_wfs

        if self.config_rl.env_rl['number_of_previous_wfs'] > 0:
            self.s_wfs_history.append(s_wfs)

        if self.config_rl.env_rl['state_wfs']:
            if self.normalization_bool:
                s_wfs = self.standardise(s_wfs, key="wfs")
            s_next["wfs"] = s_wfs
        return s_next

    def calculate_reward(self, target=0, reward_type=None):
        """
        Choose the reward depending on config parameters
        :param target: target to get the Strehl from
        :param reward_type, if it is None use the default, otherwise use it
        """

        r_se, r_le, r_va, _ = self.supervisor.target.get_strehl(target)

        if reward_type is None:
            reward_type = self.reward_type

        if reward_type == "wavefront_phase_error":
            r = -r_va
        elif reward_type == "strehl_ratio_le":
            r = r_le
        elif reward_type == "strehl_ratio_se":
            r = r_se
        elif reward_type == "residual_dm":
            err = np.array(self.supervisor.rtc.get_err(0))
            r = -np.linalg.norm(err)
        elif reward_type == "residual_wfs":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.linalg.norm(s_wfs)
        elif reward_type == "var_wfs":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(s_wfs)
        elif reward_type == "averages_wfs":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -(np.average(s_wfs[:int(len(s_wfs) / 2)]) + np.average(s_wfs[int(len(s_wfs) / 2):]))
        elif reward_type == "average_var_wfs":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(s_wfs) - (np.average(s_wfs[:int(len(s_wfs) / 2)]) + np.average(s_wfs[int(len(s_wfs) / 2):]))
        elif reward_type == "average_residual_wfs":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(s_wfs) - (np.average(s_wfs[:int(len(s_wfs) / 2)]) + np.average(s_wfs[int(len(s_wfs) / 2):]))
        elif reward_type == "image_sharpness":
            intesity_array = self.supervisor.target.get_tar_image(0)
            r = np.sum(np.square(intesity_array)) / np.square(np.sum(intesity_array))
        elif reward_type == "r_modes_1" or self.reward_type == "r_modes_1_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.var(s_wfs))
        elif reward_type == "r_modes_2" or self.reward_type == "r_modes_2_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(s_wfs)
        elif reward_type == "r_modes_3" or self.reward_type == "r_modes_3_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.var(s_wfs)) - 1
        elif reward_type == "r_modes_4" or self.reward_type == "r_modes_4_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.var(np.square(s_wfs)))
        elif reward_type == "r_modes_5" or self.reward_type == "r_modes_5_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(np.square(s_wfs))
        elif reward_type == "r_modes_6" or self.reward_type == "r_modes_6_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.var(np.square(s_wfs))) - 1
        elif reward_type == "r_tt_1" or self.reward_type == "r_tt_1_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.square(np.average(s_wfs))
        elif reward_type == "r_tt_2" or self.reward_type == "r_tt_2_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.abs(np.average(s_wfs))
        elif reward_type == "r_tt_3" or self.reward_type == "r_tt_3_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            # x_cent = s_wfs[:int(len(s_wfs)/2.0)]
            # y_cent = s_wfs[int(len(s_wfs)/2.0):]
            # r = -np.average(np.square(x_cent) + np.square(y_cent))
            r = -np.average(np.square(s_wfs))
        elif reward_type == "r_tt_4" or self.reward_type == "r_tt_4_norm":
            s_wfs = self.supervisor.target.get_tar_image(0)
            r = -np.sum(np.square(np.array(center_of_mass(s_wfs)) - s_wfs.shape[0] / 2.0))
        elif reward_type == "r_tt_5" or self.reward_type == "r_tt_5_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.average(np.square(s_wfs)))
        elif reward_type == "r_tt_6" or self.reward_type == "r_tt_6_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.average(np.square(s_wfs))) - 1
        elif reward_type == "r_tt_7":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            x_cent = s_wfs[:int(len(s_wfs) / 2.0)]
            y_cent = s_wfs[int(len(s_wfs) / 2.0):]
            r = -(np.square(np.average(x_cent)) + np.square(np.average(y_cent)))
        elif reward_type == "r_modes_7":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            x_cent = s_wfs[:int(len(s_wfs) / 2.0)]
            y_cent = s_wfs[int(len(s_wfs) / 2.0):]
            r = -(np.square(np.var(x_cent)) + np.square(np.var(y_cent)))
        elif reward_type == "r_1_and_2" or self.reward_type == "r_1_and_2_norm":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.var(s_wfs) - np.abs(np.average(s_wfs))
        elif reward_type == "r_le":
            r = 0
        elif reward_type == "single_agent_1" or reward_type == "avg_square_m":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.average(np.square(s_wfs))
        elif reward_type == "sum_measurements_squared":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = -np.sum(np.square(s_wfs))
        elif reward_type == "single_agent_2":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.var(s_wfs)) - 1
        elif reward_type == "single_agent_3":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r_modes = np.exp(-np.var(s_wfs)) - 1
            r_tt = -np.average(np.square(s_wfs))
            r = r_tt / (r_tt + r_modes) + r_modes / (r_tt + r_modes)
            # TODO correct stupid
            # var + square of average
            # r = -(np.var(m) + np.square(np.average(m))) =equivalent= - np.average(np.square(s_wfs))
            # r = np.exp(-np.var(m)) + np.square(np.average(m)) # what we used in two agents
            # r = np.exp(-np.square(np.average(m))) * np.exp(-np.average(np.square(s_wfs)))
            # First term: depends on TT error if we no TT its one
            # Maximum of this reward function is one
            # SR ~ exp(-var(residual_wavefront_phase))
        elif reward_type == "single_agent_4":
            scaling_factor_tt = 0.4479
            scaling_factor_modes = 0.5485
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r_tt = np.exp(-np.var(s_wfs)) - 1
            r_modes = -np.average(np.square(s_wfs))
            r = scaling_factor_tt * r_tt / (r_tt + r_modes) + scaling_factor_modes * r_modes / (r_tt + r_modes)
        elif reward_type == "new_single_agent":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r = np.exp(-np.average(np.square(s_wfs)))
        elif reward_type == "log_var":
            r = -np.log(1 + r_va)
        elif reward_type == "log_var_scaled_1":
            r = -(np.log(1 + r_va) - 2.60)
            # print(r)
        elif reward_type == "log_var_scaled_2":
            r = -(np.log(1 + r_va) - 2.60) * 5
            # print(r)
        elif reward_type == "log_var_scaled_3":
            r = -(np.log(1 + r_va) - 2.60) * 10
            # print(r)
        elif reward_type == "log_var_scaled_4":
            r = -(np.log(1 + r_va) - 2.60) * 0.5
            # print(r)
        elif reward_type == "log_var_scaled_5":
            r = -(np.log(1 + r_va) - 3.50)
            # print(r)
        elif reward_type == "log_var_scaled_6":
            r = -(np.log(1 + r_va) - 3.50) * 5
            # print(r)
        elif reward_type == "log_var_scaled_7":
            r = -(np.log(1 + r_va) - 3.50) * 10
            # print(r)
        elif reward_type == "log_var_scaled_8":
            r = -(np.log(1 + r_va) - 3.50) * 0.5
            # print(r)
        elif reward_type == "projection_comparison":
            # projection modes
            wfs_phase = self.supervisor.wfs.get_wfs_phase(0)
            wfs_phase = wfs_phase - wfs_phase.mean()
            pupil = self.supervisor.get_m_pupil()
            wfs_phase_reshaped = wfs_phase[np.where(pupil)]
            projection_modes = -self.supervisor.projector_phase2modes.dot(wfs_phase_reshaped)
            # current modes
            current_modes = self.supervisor.volts2modes.dot(self.supervisor.rtc.get_voltages(0))
            action_range = self.supervisor.obtain_action_range_modal(current_modes)
            difference = projection_modes[action_range] - current_modes[action_range]
            r = -np.linalg.norm(difference)
        elif reward_type == "weighted_projection_comparison":
            # projection modes
            wfs_phase = self.supervisor.wfs.get_wfs_phase(0)
            wfs_phase = wfs_phase - wfs_phase.mean()
            pupil = self.supervisor.get_m_pupil()
            wfs_phase_reshaped = wfs_phase[np.where(pupil)]
            projection_modes = -self.supervisor.projector_phase2modes.dot(wfs_phase_reshaped)
            # current modes
            current_modes = self.supervisor.volts2modes.dot(self.supervisor.rtc.get_voltages(0))
            action_range = self.supervisor.obtain_action_range_modal(current_modes)
            difference = projection_modes[action_range] - current_modes[action_range]
            weighted_difference = difference * self.supervisor.freedom_vector[action_range]
            r = -np.linalg.norm(weighted_difference)
        elif reward_type == "log_avg_m":
            s_wfs = self.supervisor.rtc.get_slopes(0)
            r_avg = np.average(np.square(s_wfs))
            r = -np.log(1 + r_avg)
        elif reward_type == "avg_squared_modes":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -np.sum(np.square(s_dm_residual_modes))
            else:
                raise NotImplementedError
        elif reward_type == "avg_squared_modes_from_measurements":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                s_dm_residual_modes = self.supervisor.projector_wfs2modes.dot(self.supervisor.rtc.get_slopes(0))
                s_dm_residual_modes = s_dm_residual_modes[action_range]
                r = -np.sum(np.square(s_dm_residual_modes))
                """"
                    s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0))
                    r2 = -np.sum(np.square(s_dm_residual_modes))
    
    
                    # m2v_filtered = self.supervisor.modes2volts[:, action_range]
                    Btt = self.supervisor.modes2volts
                    nfilt = 30
                    Btt_filt = np.zeros((Btt.shape[0], Btt.shape[1] - nfilt))
                    Btt_filt[:, :Btt_filt.shape[1] - 2] = Btt[:, :Btt.shape[1] - (nfilt + 2)]
                    Btt_filt[:, Btt_filt.shape[1] - 2:] = Btt[:, Btt.shape[1] - 2:]
                    cmat = Btt_filt.dot(self.supervisor.projector_wfs2modes)
    
    
                    s = self.supervisor.volts2modes.dot(self.config_rl.original_gain*cmat.dot(self.supervisor.rtc.get_slopes(0)))
                    print("3.1", s.shape)
                    s = s[action_range]
                    print("3", s.shape)
                    r3 = -np.sum(np.square(s))
                    print("4", r, r2, r3)
                """
            else:
                raise NotImplementedError
        elif reward_type == "true_avg_squared_modes":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -np.average(np.square(s_dm_residual_modes))
            else:
                raise NotImplementedError
        elif reward_type == "avg_squared_modes_scaled_1":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -np.sum(np.square(s_dm_residual_modes)) * 10
            else:
                NotImplementedError
        elif reward_type == "avg_squared_modes_scaled_2":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -np.sum(np.square(s_dm_residual_modes)) * 100
            else:
                NotImplementedError
        elif reward_type == "avg_squared_modes_scaled_3":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -np.sum(np.square(s_dm_residual_modes)) * 1000
            else:
                NotImplementedError
        elif reward_type == "variance_actuators_filtered_from_modes":
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.supervisor.volts2modes.dot(self.supervisor.rtc.get_err(0))
                action_range = np.array(self.supervisor.obtain_action_range_modal(s_dm_residual_modes))
                mask = np.ones(s_dm_residual_modes.size, dtype=bool)
                mask[action_range] = False
                s_dm_residual_modes[mask] = 0
                s_dm_residual_commands_filtered = self.supervisor.modes2volts.dot(s_dm_residual_modes)
                r = -np.var(s_dm_residual_commands_filtered)
            else:
                NotImplementedError
        elif "avg_squared_modes_" in reward_type:
            factor = float(self.config_rl.env_rl['reward_type'].split("_")[-1])
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual_modes = self.transform_state_to_zernike(self.supervisor.rtc.get_err(0),
                                                                      return_reward=True)
                action_range = self.supervisor.obtain_action_range_modal(self.supervisor.rtc.get_command(0))
                r = -factor * np.average(np.square(s_dm_residual_modes))
            else:
                raise NotImplementedError
            # r = None
        elif "counterfactual_rpc" in reward_type:
            r = None
        else:
            raise NotImplementedError("This reward type not implemented")

        return r

    #
    #         Step methods
    # This works for both level = Correction and level = Gain
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def linear_step(self,
                    return_dict=False,
                    geometric_do_control=False,
                    geometric_apply_control=True):
        """
        Does a linear step
        i.e. raytraceTar (if not pure delay 0) + raytraceWFS + doCentroids + doControl
        :param return_dict: if return states a dict
        :param geometric_do_control: if geometric does control
        :param geometric_apply_control: if geometric applies control
        :return: s_next: next state
        """
        s_dm_before_linear = self.supervisor.rtc.get_command(0)

        self.supervisor.next_part_one(geometric_apply_control=geometric_apply_control)

        s_dm_after_linear = self.supervisor.rtc.get_command(0)
        s_wfs = self.supervisor.rtc.get_slopes(0)
        if self.supervisor.config.p_controllers[0].get_type() != "geo":
            s_dm_residual = self.supervisor.rtc.get_err(0)
        else:
            s_dm_residual = None

        if self.config_rl.env_rl['basis'] == "zernike_space":
            s_dm_before_linear = self.transform_state_to_zernike(s_dm_before_linear, return_reward=False)
            s_dm_after_linear = self.transform_state_to_zernike(s_dm_after_linear, return_reward=False)
            if self.supervisor.config.p_controllers[0].get_type() != "geo":
                s_dm_residual = self.transform_state_to_zernike(s_dm_residual, return_reward=False)

        s_next = OrderedDict()
        s_next = self.add_wfs_to_state(s_next, s_wfs)
        s_next = self.add_dm_to_state(s_next, s_dm_before_linear, s_dm_after_linear)
        if self.supervisor.config.p_controllers[0].get_type() != "geo":
            s_next = self.add_s_dm_residual_to_state(s_next, s_dm_residual)

        if return_dict:
            return s_next
        else:
            return np.concatenate(list(s_next.values()))

    def rl_step(self, action,
                linear_control=False,
                geometric_do_control=False,
                evaluation_rl_full_action=False,
                apply_control=True,
                compute_tar_psf=True):
        """
        Calculate RL command (if applicable) + Apply Control + Compute Tar Image and Strehl
        """
        self.supervisor.next_part_two(action=action,
                                      linear_control=linear_control,
                                      evaluation_rl_full_action=evaluation_rl_full_action,
                                      apply_control=apply_control,
                                      compute_tar_psf=compute_tar_psf
                                      )

        r = self.calculate_reward()
        if len(self.supervisor.config.p_controllers) > 1 and geometric_do_control:
            r_geo_se = self.calculate_reward(target=1, reward_type="strehl_ratio_se")
        else:
            r_geo_se = None

        d = False
        info = ""

        if r_geo_se is not None:
            return r, d, info, r_geo_se
        else:
            return r, d, info

    #
    #         gain methods
    # This works for level = Gain
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def set_gain(self, g):
        """
        Sets a certain gain in the system for the linear approach with integrator
        :param g: gain to be set
        """
        # g = self.supervisor.rtc.d_control[0].gain
        if np.isscalar(g[0]):
            self.supervisor.rtc._rtc.d_control[0].set_gain(g)
        else:
            raise ValueError("Cannot set array gain w/ generic + integrator law")

    #
    #      Used to extract norm parameters
    #      Performs only linear approach
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    def normalization_step(self, linear_control_through_modal=False):
        """
        Does a step in the environment and extracts the results WITHOUT normalizing
        This will be used to calculate the normalization parameters
        :param linear_control_through_modal: TODO
        """

        integrator_modes, projection_modes = self.supervisor.next_normalization(linear_control_through_modal)

        return integrator_modes, projection_modes
