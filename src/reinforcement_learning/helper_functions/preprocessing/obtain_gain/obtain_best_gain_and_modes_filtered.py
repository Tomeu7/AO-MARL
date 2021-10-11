from src.reinforcement_learning.environment import ao_env
from src.reinforcement_learning.config.GlobalConfig import Config
import numpy as np
import os
import pandas as pd


def obtain_config(parameter_file,
                  pure_delay_0,
                  use_autoencoder,
                  num_steps,
                  autoencoder_path):
    """
    Creates the RL configuration
    :param parameter_file: parameter file
    :param pure_delay_0: if we use no delay
    :param use_autoencoder: if we use autoencoder
    :param autoencoder_path: the autoencoder path
    :param num_steps: the number of steps per episode
    :return: configuration of RL environment + agent
    """
    config = Config()
    config.strings_to_bools()
    config.env_rl['parameters_telescope'] = parameter_file
    # Changing the parameters telescope we want and setting up the level of correction

    print("Parameters", config.env_rl['parameters_telescope'])

    # Setting up 1000 steps per episode and level correction
    config.env_rl['max_steps_per_episode'] = num_steps
    config.env_rl['level'] = "correction"
    config.env_rl['modification_online'] = pure_delay_0

    if use_autoencoder:
        config.autoencoder['type'] = "cnn_single_subaperture"
        config.autoencoder['path'] = autoencoder_path

    return config


def performance_loop(config,
                     env,
                     gain,
                     num_episodes,
                     num_modes_filtered,
                     use_autoencoder,
                     verbose=False):
    """
    Does a performance loop considering a certain gain and number of modes filtered
    :param config: RL configuration
    :param env: RL environment
    :param gain: current gain
    :param num_episodes: number of episodes
    :param num_modes_filtered: number of modes filtered
    :param use_autoencoder: if we use autoencoder
    :param verbose: if we do prints
    :return: None
    """

    seed = 0
    for episode in range(num_episodes):
        import time
        start_time = time.time()
        step = 0
        done = False
        seed += 1
        env.set_sim_seed(seed)
        env.supervisor.obtain_and_set_cmat_filtered(num_modes_filtered)
        env.set_gain(np.array([gain]))
        env.reset()

        if verbose:
            print("Episode:", episode,
                  ", Gain:", gain,
                  ", Seed:", seed,
                  ", Noise", env.supervisor.wfs._wfs.d_wfs[0].noise,
                  ", Modification online", env.config_rl.env_rl['modification_online'],
                  ", Delay", env.supervisor.config.p_controllers[0].delay,
                  ", Use autoencoder", use_autoencoder)

        while not done:
            if config.env_rl['level'] == "correction":
                env.normalization_step()
                # print("Step {} SE SR {}".format(step, env.supervisor.target.get_strehl(0)[0]))
            else:
                raise NotImplementedError

            step += 1
            if step >= config.env_rl["max_steps_per_episode"]:
                done = True

        print("Episode:", episode,
              ", SR LE:", env.supervisor.target.get_strehl(0)[1],
              "time", time.time()-start_time)


def obtain_modes_filtered_and_gain(parameter_file,
                                   pure_delay_0,
                                   use_autoencoder,
                                   autoencoder_path=None,
                                   gains=None,
                                   output_name=None,
                                   num_episodes=2,
                                   num_steps=1000):
    """
    Does performance an analysis in a sequential way to obtain best number of modes filtered and gain:
    1) Given g=0.5 discards modes 10 to 50 and gets the best result
    2) Given the best discarded modes tries gains from 0.1 to 0.95 for 1 episode
    :param parameter_file: parameter file
    :param pure_delay_0: if we use no delay
    :param use_autoencoder: if we use autoencoder
    :param autoencoder_path: the autoencoder path
    :param num_steps: the number of steps per episode
    :return: None
    """
    config_ = obtain_config(parameter_file, pure_delay_0, use_autoencoder, num_steps, autoencoder_path)

    env_ = ao_env.AoEnv(config_, normalization_bool=False, build_cmat_with_modes=False)

    modes_filtered_list = np.arange(0, 3)*5

    sr_list_modes = []
    for modes_filtered in modes_filtered_list:
        performance_loop(config=config_,
                         env=env_,
                         gain=0.5,
                         num_episodes=num_episodes,
                         verbose=True,
                         num_modes_filtered=modes_filtered,
                         use_autoencoder=use_autoencoder)
        sr_list_modes.append(env_.supervisor.target.get_strehl(0)[1])

    modes_filtered_final = modes_filtered_list[np.argmax(sr_list_modes)]
    print("Modes filtered best", modes_filtered_final)

    sr_list_small = []
    for gain in gains:
        performance_loop(config=config_,
                         env=env_,
                         gain=gain,
                         num_episodes=num_episodes,
                         verbose=False,
                         num_modes_filtered=modes_filtered_final,
                         use_autoencoder=use_autoencoder)
        sr_list_small.append(env_.supervisor.target.get_strehl(0)[1])

    fd = "insights/gain/"
    if not os.path.exists(fd + parameter_file[:-3]):
        os.makedirs(fd + parameter_file[:-3])

    save_dict = {
        "parameter_file": [parameter_file[:-3]],
        "modification_online": [pure_delay_0],
        "modes_discared": [modes_filtered_list],
        "sr_le_modes": [sr_list_modes],
        "best_modes_discarded": [modes_filtered_final],
        "gains": [gains],
        "sr_le_gains": [sr_list_small],
        "best_gain": [gains[np.argmax(sr_list_small)]],
        "sr_le_best_gain": [np.max(sr_list_small)]
    }
    pd.DataFrame.from_dict(save_dict, orient='index').to_csv(fd + parameter_file[:-3] + "/information_best_gain.csv",
                                                             header=False)

    print("----------------parameters-----------------", parameter_file)
    print("Gains", gains)
    print("All strehls", sr_list_small),
    print("Max gain", gains[np.argmax(sr_list_small)],
          "Max LE SR", np.max(sr_list_small))
