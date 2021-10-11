import argparse
import torch

def parser_args(config):
    """
    Parses the arguments from the input command for the experiment
    """
    parser = argparse.ArgumentParser()

    # RPC

    parser.add_argument('--updates_per_episode_rpc', type=int, default=config.env_rl['max_steps_per_episode'])

    # Other

    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--gamma",
                        default=config.sac['gamma'], type=float)
    parser.add_argument("--alpha", help="entropy strength",
                        default=config.sac['alpha'], type=float)
    parser.add_argument("--batch_size", help="batch size",
                        default=config.sac['batch_size'], type=int)
    parser.add_argument("--memory_size",
                        default=config.sac['memory_size'], type=int)
    parser.add_argument("--automatic_entropy_tuning",
                        default="True", type=str)
    parser.add_argument("--lr",
                        default=config.sac['lr'], type=float)
    parser.add_argument("--tau",
                        default=config.sac['tau'], type=float)
    parser.add_argument("--hidden_size_critic",
                        default=config.sac['hidden_size_critic'], nargs='+', type=int)
    parser.add_argument("--hidden_size_actor",
                        default=config.sac['hidden_size_actor'], type=int)
    parser.add_argument("--num_layers_critic",
                        default=config.sac['num_layers_critic'], type=int)
    parser.add_argument("--num_layers_actor",
                        default=config.sac['num_layers_critic'], type=int)
    parser.add_argument("--gaussian_mu",
                        default=config.sac['gaussian_mu'], type=float)
    parser.add_argument("--gaussian_std",
                        default=config.sac['gaussian_std'], type=float)
    parser.add_argument("--policy",
                        default=config.sac['policy'], type=str)
    parser.add_argument("--max_steps_per_episode", help="num steps for each episode",
                        default=config.env_rl['max_steps_per_episode'], type=int)
    parser.add_argument("--level",
                        default=config.env_rl['level'], type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--parameters_telescope",
                        default=config.env_rl['parameters_telescope'], type=str)
    parser.add_argument("--integration_mode",
                        default=config.env_rl['integration_mode'], type=str)

    parser.add_argument("--reward_type",
                        default=config.env_rl['reward_type'], nargs='+', type=str)
    parser.add_argument("--delayed_assignment",
                        default=config.env_rl['delayed_assignment'], type=int)

    parser.add_argument("--state_dm_before_linear",
                        default=config.env_rl['state_dm_before_linear'], type=str)
    parser.add_argument("--state_dm_after_linear",
                        default=config.env_rl['state_dm_after_linear'], type=str)
    parser.add_argument("--state_wfs",
                        default=config.env_rl['state_wfs'], type=str)
    parser.add_argument("--state_gain",
                        default=config.env_rl['state_gain'], type=str)
    parser.add_argument("--number_of_previous_dm",
                        default=config.env_rl['number_of_previous_dm'], type=int)
    parser.add_argument("--number_of_previous_wfs",
                        default=config.env_rl['number_of_previous_wfs'], type=int)
    parser.add_argument("--number_of_previous_dm_residuals",
                        default=config.env_rl['number_of_previous_dm_residuals'],
                        type=int)

    parser.add_argument("--seed",
                        help="seed for this experiment, must be greater than 10 as normalization parameters"
                                       + " are used from the first 10 seeds", type=int)
    parser.add_argument("--move_atmos",
                        default=config.env_rl['move_atmos'],
                        type=str)
    parser.add_argument("--algorithm",
                        default=config.algorithm,
                        type=str)
    parser.add_argument("--reward_mode",
                        default=config.env_rl['reward_mode'], type=str)

    parser.add_argument("--basis",
                        default=config.env_rl['basis'], type=str)

    parser.add_argument("--updates_per_step",
                        default=config.sac['updates_per_step'], type=int)

    parser.add_argument("--n_reverse_filtered_from_cmat",
                        default=config.env_rl['n_reverse_filtered_from_cmat'], type=int)
    parser.add_argument("--n_zernike_start_end",
                        default=config.env_rl['n_zernike_start_end'], nargs='+', type=int)

    parser.add_argument("--include_tip_tilt", default=config.env_rl['include_tip_tilt'], type=str)

    parser.add_argument("--normalization_mean_inside_environment",
                        default=config.env_rl['normalization_mean_inside_environment'], type=float)
    parser.add_argument("--normalization_std_inside_environment",
                        default=config.env_rl['normalization_std_inside_environment'], type=float)

    parser.add_argument("--norm_scale_zernike_actions",
                        default=config.env_rl['norm_scale_zernike_actions'], type=float) # TODO changed


    parser.add_argument("--sac_reward_scaling",
                        default=config.sac['sac_reward_scaling'], type=str)
    parser.add_argument("--pretrained_model_path",
                        default=config.sac['pretrained_model_path'], type=str)


    parser.add_argument("--activation",
                        default=config.sac['activation'], type=str)

    parser.add_argument("--modification_online",
                        default=config.env_rl['modification_online'], type=str)

    parser.add_argument("--custom_freedom_path",
                        default=config.env_rl['custom_freedom_path'], type=str)
    parser.add_argument("--load_previous_weights",
                        default=config.env_rl['load_previous_weights'], type=str)

    # autoencoder
    parser.add_argument("--autoencoder_path",
                        default=config.autoencoder['path'], type=str)
    parser.add_argument("--autoencoder_type",
                        default=config.autoencoder['type'], type=str)

    # two agents

    parser.add_argument("--create_norm_param",
                        default=config.env_rl['create_norm_param'], type=str)

    parser.add_argument("--initialize_last_layer_near_0", default="False", type=str)
    parser.add_argument("--initialize_last_layer_0", default=config.sac['initialize_last_layer_0'], type=str)

    parser.add_argument("--save_replay_buffer", default=config.sac['save_replay_buffer'],
                        type=str)
    parser.add_argument("--save_rewards_buffer", default=config.sac['save_rewards_buffer'],
                        type=str)
    parser.add_argument("--state_dm_residual", default=config.env_rl['state_dm_residual'], type=str)
    parser.add_argument("--replay_path", default=config.sac['replay_path'], nargs='+', type=str)
    parser.add_argument("--l2_norm_policy", default=config.sac['l2_norm_policy'], type=float)

    parser.add_argument("--window_n_zernike", default=config.env_rl['window_n_zernike'], type=int)

    parser.add_argument("--pretrained_replay_path",
                        default=config.sac['pretrained_replay_path'],
                        type=str)
    parser.add_argument("--LOG_SIG_MAX", default=config.sac['LOG_SIG_MAX'], type=float)

    # TT_reward
    parser.add_argument("--TT_reward", default=config.env_rl['TT_reward'], type=str)

    parser.add_argument("--port", default='29500', type=str)

    parser.add_argument("--gain_change",
                        default=config.env_rl['gain_change'], type=float)

    parser.add_argument("--change_atmospheric_3_layers_1",
                        default=config.env_rl['change_atmospheric_3_layers_1'], type=str)
    parser.add_argument("--change_atmospheric_3_layers_2",
                        default=config.env_rl['change_atmospheric_3_layers_2'], type=str)
    parser.add_argument("--change_atmospheric_3_layers_3",
                        default=config.env_rl['change_atmospheric_3_layers_3'], type=str)
    parser.add_argument("--change_atmospheric_3_layers_4",
                        default=config.env_rl['change_atmospheric_3_layers_4'], type=str)
    parser.add_argument("--change_atmospheric_3_layers_5",
                        default=config.env_rl['change_atmospheric_3_layers_5'], type=str)

    parser.add_argument("--include_tip_tilt_windowed", default=config.env_rl['include_tip_tilt_windowed'], type=str)

    parser.add_argument("--do_more_evaluations", default=config.env_rl['do_more_evaluations'], type=str)
    parser.add_argument("--tt_treated_as_mode", default=config.env_rl['tt_treated_as_mode'], type=str)
    
    return parser



def obtain_args(config):
    """
    Parses arguments from command line and obtains arguments
    """
    parser = parser_args(config)
    args = parser.parse_args()
    return args


def print_and_assertions(config, seed):
    """
    Some requirements before starting the simulation
    """
    # 1) We require seed > 100 because we obtain normalization parameters with seed < 100
    assert seed > 100
    print("__________________________")
    print("\n Initial seed:", seed)
