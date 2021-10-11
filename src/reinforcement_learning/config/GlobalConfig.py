import json


class Config:
    """
    Config object that reads all the files from reinforcement_learning/config/*.cfg
    """
    def __init__(self, in_p="src/reinforcement_learning"):
        config_file_path = in_p + '/config/parameters.cfg'
        config_file_path_sac = in_p + '/config/parameters_sac.cfg'
        config_file_path_autoencoder = in_p + '/config/parameters_autoencoder.cfg'
        with open(config_file_path, 'r') as datafile:
            config = json.load(datafile)

        with open(config_file_path_sac, 'r') as datafile:
            config_sac = json.load(datafile)

        with open(config_file_path_autoencoder, 'r') as datafile:
            config_autoencoder = json.load(datafile)

        self.sac = dict()
        self.autoencoder = dict()
        self.env_rl = dict()

        self.cuda = True if config['cuda'] == "True" else False
        self.algorithm = str(config['algorithm'])

        # 1) Soft Actor Critic Config

        # 1.1 Traditional parameters

        self.sac['alpha'] = float(config_sac['alpha'])
        self.sac['automatic_entropy_tuning'] = str(config_sac['automatic_entropy_tuning'])
        self.sac['batch_size'] = int(config_sac['batch_size'])
        self.sac['gamma'] = float(config_sac['gamma'])
        self.sac['hidden_size_critic'] = [int(config_sac['hidden_size_critic'])]
        self.sac['hidden_size_actor'] = int(config_sac['hidden_size_actor'])
        self.sac['num_layers_critic'] = int(config_sac['num_layers_critic'])
        self.sac['num_layers_actor'] = int(config_sac['num_layers_actor'])
        self.sac['lr'] = float(config_sac['lr'])
        self.sac['policy'] = str(config_sac['policy'])
        self.sac['target_update_interval'] = int(config_sac['target_update_interval'])
        self.sac['tau'] = float(config_sac['tau'])
        self.sac['updates_per_step'] = int(config_sac['updates_per_step'])
        self.sac['memory_size'] = int(config_sac['memory_size'])

        # 1.2 Other parameters

        self.sac['gaussian_mu'] = float(config_sac['gaussian_mu'])
        self.sac['gaussian_std'] = float(config_sac['gaussian_std'])

        self.sac['sac_reward_scaling'] = float(1.0)
        self.sac['activation'] = str(config_sac['activation'])
        self.sac['initialize_last_layer_0'] = str(config_sac['initialize_last_layer_0'])
        self.sac['initialize_last_layer_near_0'] = str(config_sac['initialize_last_layer_near_0'])

        self.sac['save_replay_buffer'] = str(config_sac['save_replay_buffer'])
        self.sac['save_rewards_buffer'] = str(config_sac['save_rewards_buffer'])


        self.sac['l2_norm_policy'] = -1
        self.sac['LOG_SIG_MAX'] = 2.0
        self.sac['updates_per_episode_rpc'] = 1000

        # 2) Environment Reinforcement Learning Config

        self.env_rl['write_every'] = int(config['env_rl_parameters']['write_every'])
        self.env_rl['verbose'] = False
        self.env_rl['check_every'] = int(config['env_rl_parameters']['check_every'])
        self.env_rl['move_atmos'] = str(config['env_rl_parameters']['move_atmos'])
        self.env_rl['max_steps_per_episode'] = int(config['env_rl_parameters']['max_steps_per_episode'])

        # Parameters of telescope

        self.env_rl['parameters_telescope'] = str(config['env_rl_parameters']['parameters_telescope'])
        self.env_rl['integration_mode'] = str(config['env_rl_parameters']['integration_mode'])

        # Related to how do we do the process and we optimize

        self.env_rl['level'] = str(config['env_rl_parameters']['level'])
        self.env_rl['basis'] = str(config['env_rl_parameters']['basis'])
        self.env_rl['influence'] = str(config['env_rl_parameters']['influence'])

        # Related to reward

        self.env_rl['reward_type'] = "avg_squared_modes_1000"
        self.env_rl['delayed_assignment'] = int(config['env_rl_parameters']['delayed_assignment'])

        # Related to state

        self.env_rl['state_dm_before_linear'] = str(config['env_rl_parameters']['state_dm_before_linear'])
        self.env_rl['state_dm_after_linear'] = str(config['env_rl_parameters']['state_dm_after_linear'])
        self.env_rl['state_wfs'] = str(config['env_rl_parameters']['state_wfs'])
        self.env_rl['state_gain'] = str(config['env_rl_parameters']['state_gain'])
        self.env_rl['state_dm_residual'] = str(config['env_rl_parameters']['state_dm_residual'])
        self.env_rl['number_of_previous_dm'] = int(config['env_rl_parameters']['number_of_previous_dm'])
        self.env_rl['number_of_previous_wfs'] = int(config['env_rl_parameters']['number_of_previous_wfs'])
        self.env_rl['number_of_previous_dm_residuals'] = 0

        self.env_rl['reward_mode'] = str(config['env_rl_parameters']['reward_mode'])

        self.env_rl['n_zernike_start_end'] = [-1, -1]
        self.env_rl['n_reverse_filtered_from_cmat'] = 0
        self.env_rl['include_tip_tilt'] = "True"

        # Related to reward

        self.env_rl['reward_mode'] = str(config['env_rl_parameters']['reward_mode'])
        self.env_rl['TT_reward'] = 'absolute'

        # Other

        self.env_rl['normalization_std_inside_environment'] =\
            float(config['env_rl_parameters']['normalization_std_inside_environment'])
        self.env_rl['normalization_mean_inside_environment'] =\
            float(config['env_rl_parameters']['normalization_mean_inside_environment'])

        self.env_rl['norm_scale_zernike_actions'] = float(config['env_rl_parameters']['norm_scale_zernike_actions'])
        self.env_rl['modification_online'] = str(config['env_rl_parameters']['modification_online'])
        self.env_rl["custom_freedom_path"] = None

        self.env_rl['create_norm_param'] = str(config['env_rl_parameters']['create_norm_param'])
        self.env_rl['window_n_zernike'] = -1

        self.env_rl['do_more_evaluations'] = "False"
        self.env_rl['change_atmospheric_3_layers_1'] = "False"
        self.env_rl['change_atmospheric_3_layers_2'] = "False"
        self.env_rl['change_atmospheric_3_layers_3'] = "False"
        self.env_rl['change_atmospheric_3_layers_4'] = "False"
        self.env_rl['change_atmospheric_3_layers_5'] = "False"

        self.env_rl['change_atmospheric_conditions_wind_direction_1'] = "False"
        self.env_rl['change_atmospheric_conditions_wind_direction_2'] = "False"

        self.env_rl['include_tip_tilt_windowed'] = "False"
        self.env_rl['record_autoencoder_time'] = "False"

        self.env_rl['gain_change'] = -1.0
        self.env_rl['tt_treated_as_mode'] = "False"

        # 3) Variable to save original gain from integrator controller
        self.original_gain = None

        # 4) Autoencoder
        self.autoencoder['path'] = None
        self.autoencoder['type'] = str(config_autoencoder['type'])

        # Loading previous weights/replay

        self.env_rl['load_previous_weights'] = str(config['env_rl_parameters']['load_previous_weights'])
        self.sac['pretrained_replay_path'] = None
        self.sac['replay_path'] = None
        self.sac['pretrained_model_path'] = str(config_sac['pretrained_model_path'])

        # TODO ??

        self.dictionary_agent_values = None

    def update_conf_with_args(self, args):
        """
        Updates config object with the arguments
        """

        if args.algorithm == "SAC":
            self.update_sac(args)
        else:
            raise NotImplementedError

        self.update_env(args)

        self.update_autoencoder(args)

        self.strings_to_bools()

    def update_sac(self, args):
        self.algorithm = args.algorithm
        self.sac['policy'] = args.policy

        # Traditional parameters

        self.sac['gamma'] = args.gamma
        self.sac['batch_size'] = args.batch_size
        self.sac['alpha'] = args.alpha
        self.sac['automatic_entropy_tuning'] = True if args.automatic_entropy_tuning == "True" else False
        self.sac['memory_size'] = args.memory_size
        self.sac['lr'] = args.lr
        if len(args.hidden_size_critic) == 1:
            self.sac['hidden_size_critic'] = args.hidden_size_critic[0]
        else:
            self.sac['hidden_size_critic'] = args.hidden_size_critic
        self.sac['num_layers_critic'] = args.num_layers_critic
        self.sac['hidden_size_actor'] = args.hidden_size_actor
        self.sac['num_layers_actor'] = args.num_layers_actor
        self.sac['tau'] = float(args.tau)

        # Other parameters

        self.sac['gaussian_mu'] = args.gaussian_mu
        self.sac['gaussian_std'] = args.gaussian_std
        self.sac['updates_per_step'] = args.updates_per_step
        self.sac['sac_reward_scaling'] = float(args.sac_reward_scaling)
        self.sac['activation'] = str(args.activation)
        self.sac['initialize_last_layer_0'] = True if str(args.initialize_last_layer_0) == "True" else False
        self.sac['initialize_last_layer_near_0'] = True if str(args.initialize_last_layer_near_0) == "True" else False
        self.sac['l2_norm_policy'] = float(args.l2_norm_policy)
        self.sac['updates_per_episode_rpc'] = int(args.updates_per_episode_rpc)
        self.sac['LOG_SIG_MAX'] = float(args.LOG_SIG_MAX)

        # Loading
        self.sac['pretrained_replay_path'] = args.pretrained_replay_path
        self.sac['pretrained_model_path'] = None if args.pretrained_model_path == "None" else args.pretrained_model_path
        self.sac['replay_path'] = args.replay_path if args.replay_path is not None else None

    def update_env(self, args):

        self.env_rl['move_atmos'] = True if args.move_atmos == "True" else False
        self.env_rl['max_steps_per_episode'] = args.max_steps_per_episode
        self.env_rl['level'] = args.level
        self.env_rl['parameters_telescope'] = args.parameters_telescope
        self.env_rl['integration_mode'] = args.integration_mode

        if len(args.reward_type) > 1:
            self.env_rl['reward_type'] = args.reward_type
        else:
            self.env_rl['reward_type'] = args.reward_type[0]
        self.env_rl['delayed_assignment'] = args.delayed_assignment

        # Related to the state
        self.env_rl['state_dm_before_linear'] = True if args.state_dm_before_linear == "True" else False
        self.env_rl['state_dm_after_linear'] = True if args.state_dm_after_linear == "True" else False
        self.env_rl['state_wfs'] = True if args.state_wfs == "True" else False
        self.env_rl['state_dm_residual'] = True if args.state_dm_residual == "True" else False

        self.env_rl['n_zernike_start_end'] = args.n_zernike_start_end
        self.env_rl['include_tip_tilt'] = True if args.include_tip_tilt == "True" else False
        self.env_rl['number_of_previous_dm'] = int(args.number_of_previous_dm)
        self.env_rl['number_of_previous_wfs'] = int(args.number_of_previous_wfs)
        self.env_rl['number_of_previous_dm_residuals'] = int(args.number_of_previous_dm_residuals)

        # Related to the reward
        self.env_rl['reward_mode'] = args.reward_mode


        self.env_rl['max_steps_episode'] = args.max_steps_per_episode
        self.env_rl['basis'] = args.basis
        self.env_rl['normalization_std_inside_environment'] = float(args.normalization_std_inside_environment)
        self.env_rl['normalization_mean_inside_environment'] = float(args.normalization_mean_inside_environment)
        self.env_rl['norm_scale_zernike_actions'] = float(args.norm_scale_zernike_actions)

        self.env_rl['modification_online'] = True if args.modification_online == "True" else False

        self.env_rl["custom_freedom_path"] = args.custom_freedom_path
        self.env_rl["load_previous_weights"] = True if args.load_previous_weights == "True" else False

        self.env_rl['create_norm_param'] = True if args.create_norm_param == "True" else False

        self.env_rl['n_reverse_filtered_from_cmat'] = int(args.n_reverse_filtered_from_cmat)

        self.env_rl['window_n_zernike'] = int(args.window_n_zernike)

        self.env_rl['TT_reward'] = str(args.TT_reward)
        self.env_rl['tt_treated_as_mode'] = True if args.tt_treated_as_mode == "True" else False

        self.env_rl['do_more_evaluations'] = True if args.do_more_evaluations == "True" else False

        self.env_rl['include_tip_tilt_windowed'] = True if args.include_tip_tilt_windowed == "True" else False
        if int(args.window_n_zernike) >= 0:
            self.env_rl['include_tip_tilt_windowed'] = True

        self.env_rl['gain_change'] = float(args.gain_change)

        self.env_rl['change_atmospheric_3_layers_1'] = True if args.change_atmospheric_3_layers_1 == "True" else False
        self.env_rl['change_atmospheric_3_layers_2'] = True if args.change_atmospheric_3_layers_2 == "True" else False
        self.env_rl['change_atmospheric_3_layers_3'] = True if args.change_atmospheric_3_layers_3 == "True" else False
        self.env_rl['change_atmospheric_3_layers_4'] = True if args.change_atmospheric_3_layers_4 == "True" else False
        self.env_rl['change_atmospheric_3_layers_5'] = True if args.change_atmospheric_3_layers_5 == "True" else False

    def update_autoencoder(self, args):
        self.autoencoder['path'] = args.autoencoder_path
        self.autoencoder['type'] = str(args.autoencoder_type)

    def strings_to_bools(self):
        for key, item in self.env_rl.items():
            print(type(item), key, item)
            if item == "True":
                self.env_rl[key] = True
                if self.env_rl['verbose']:
                    print("correcting,", key, "to True")
            elif item == "False":
                self.env_rl[key] = False
                if self.env_rl['verbose']:
                    print("correcting,", key, "to False")

        for key, item in self.sac.items():
            print(type(item), key, item)
            if item == "True":
                self.sac[key] = True
                if self.env_rl['verbose']:
                    print("correcting,", key, "to True")
            elif item == "False":
                self.sac[key] = False
                if self.env_rl['verbose']:
                    print("correcting,", key, "to False")


    def set_original_gain(self, gain):
        self.original_gain = gain
