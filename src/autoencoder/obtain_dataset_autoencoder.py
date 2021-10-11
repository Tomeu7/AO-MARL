from src.reinforcement_learning.environment import ao_env
from src.reinforcement_learning.config.GlobalConfig import Config
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os

def obtain_config(parameter_file,
                  basis,
                  pure_delay_0,
                  autoencoder_path):
    """
    Obtains configuration for normalization    
    :param parameter_file: current parameter file
    :param basis: "zernike_space" or "actuator_space"
    :param pure_delay_0: if pure delay 0
    :param autoencoder_path: where the autoencoder is located, default: None
    :return: config object
    """
    config = Config()
    config.strings_to_bools()

    config.env_rl['basis'] = basis
    config.env_rl['parameters_telescope'] = parameter_file

    config.env_rl['state_dm_after_linear'] = False
    config.env_rl['state_wfs'] = True

    config.env_rl['state_dm_before_linear'] = True
    config.env_rl['state_gain'] = False
    config.env_rl['number_of_previous_dm'] = 2
    config.env_rl['number_of_previous_wfs'] = 0
    config.env_rl['reward_type'] = "avg_square_m"
    config.env_rl['basis'] = 'zernike_space'
    config.env_rl['n_zernike'] = 1260

    # Setting up 1000 steps per episode
    config.env_rl['max_steps_per_episode'] = 1000
    config.env_rl['level'] = "correction"
    config.env_rl['modification_online'] = pure_delay_0
    config.env_rl['other_modes_from_integrator_deactivated'] = "True"
    config.autoencoder['type'] = "cnn_single_subaperture"
    if autoencoder_path is not None:
        config.autoencoder['path'] = autoencoder_path
        print("Loading autoencoder")
    return config


class OfflineDatasetObtainer:
    def __init__(self,
                 parameter_file,
                 modification_online,
                 modes_filtered,
                 autoencoder_path=None):
        self.parameter_file_name = parameter_file[:-3]

        self.config_normal = obtain_config(parameter_file,
                                           "zernike_space",
                                           modification_online,
                                           autoencoder_path)
        self.env = ao_env.AoEnv(self.config_normal, normalization_bool=False, geo_policy_testing=False)
        print("Filtering now: ", modes_filtered, " Modes")
        self.env.supervisor.obtain_and_set_cmat_filtered(modes_filtered=modes_filtered)

    def record_data(self,
                    num_episodes,
                    seed):

        wfs_image_noise3_list = []
        wfs_image_noiseminus1_list = []
        seed = seed
        assert seed < 1234  # We usually train with seed 1234
        assert seed > 200  # Seed 200 for error budget, seed 0-20 for preprocessing
        print("Warning: When testing autoencoder do not use the same seed. Current seed ", seed)
        for episode in range(num_episodes):
            step = 0
            done = False
            seed += 1
            self.env.set_sim_seed(seed)
            self.env.reset(normalization_loop=True)

            while not done:

                self.env.supervisor.generic_delay_0_next(ncontrol=0)

                wfs_image_noise3_list.append(np.array(self.env.supervisor.wfs._wfs.d_wfs[0].d_bincube))
                wfs_image_noiseminus1_list.append(np.array(self.env.supervisor.wfs._wfs.d_wfs[1].d_bincube))

                step += 1
                if step >= 1000:
                    done = True

            print("Normalization episode:", episode+1,
                  "Steps:", step,
                  "Seed:", seed,
                  "Gain:", round(self.env.supervisor.rtc._rtc.d_control[0].gain, 3),
                  "L.E. SR:", round(self.env.supervisor.target.get_strehl(0)[1], 5))

        save_folder = "output/autoencoder/output_dataset_autoencoder/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path_noise3 = "noise3_image_" + self.parameter_file_name + "_small"
        save_path_noiseminus1 = "noiseminus1_image_" + self.parameter_file_name + "_small"

        print("Saving data noise 3 on:", save_path_noise3)
        print("Saving data noise minus 1 on:", save_path_noiseminus1)
        np.save(save_folder + save_path_noise3 + ".npy", np.array(wfs_image_noise3_list))
        np.save(save_folder + save_path_noiseminus1 + ".npy", np.array(wfs_image_noiseminus1_list))


class ExperimentManager:
    def __init__(self, args):
        self.freedom_path = None
        self.number_episodes = 1
        self.autoencoder_p = None
        self.pure_delay_0 = True
        self.seed = args.seed
        self.parameter_file = args.parameter_file
        self.num_filtered = args.num_filtered

    def run(self):

        par_file_list = [self.parameter_file]
        for par_file_idx in range(len(par_file_list)):
            par_file = par_file_list[par_file_idx]
            data_obtainer = OfflineDatasetObtainer(parameter_file=par_file,
                                                   modification_online=self.pure_delay_0,
                                                   modes_filtered=self.num_filtered)

            data_obtainer.record_data(self.number_episodes, self.seed)

            del data_obtainer


parser = argparse.ArgumentParser()
parser.add_argument('--parameter_file', type=str)
parser.add_argument('--seed', type=int, default=300)
parser.add_argument('--num_filtered', type=int, default=5)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)

exp = ExperimentManager(args)
exp.run()
