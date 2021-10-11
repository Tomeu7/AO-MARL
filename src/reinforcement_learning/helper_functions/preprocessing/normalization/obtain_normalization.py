from src.reinforcement_learning.environment import ao_env
from src.reinforcement_learning.config.GlobalConfig import Config
import numpy as np
import pickle
import matplotlib.pyplot as plt
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

    config.env_rl['state_dm_after_linear'] = True
    config.env_rl['state_wfs'] = True

    config.env_rl['state_dm_before_linear'] = False
    config.env_rl['state_gain'] = False
    config.env_rl['state_wfs_minus_1'] = False
    config.env_rl['state_wfs_minus_2'] = False
    config.env_rl['state_dm_minus_1'] = False

    # Setting up 1000 steps per episode
    config.env_rl['max_steps_per_episode'] = 1000
    config.env_rl['level'] = "correction"
    config.env_rl['modification_online'] = pure_delay_0
    config.autoencoder['type'] = "cnn_single_subaperture"
    if autoencoder_path is not None:
        config.autoencoder['path'] = autoencoder_path
        print("Loading autoencoder")
    return config


class Preprocessor:
    def __init__(self,
                 parameter_file,
                 basis,
                 modification_online,
                 modes_filtered,
                 autoencoder_path=None):
        self.parameter_file_name = parameter_file[:-3]
        self.config_normal = obtain_config(parameter_file, basis, modification_online, autoencoder_path)
        self.env_normal = ao_env.AoEnv(self.config_normal, normalization_bool=False, geo_policy_testing=False)
        self.env_normal.supervisor.obtain_and_set_cmat_filtered(modes_filtered=modes_filtered)
        self.v2m = self.env_normal.supervisor.volts2modes
        self.m2v = self.env_normal.supervisor.modes2volts
        self.config_geo = obtain_config(parameter_file, basis, modification_online, autoencoder_path)
        self.config_geo.env_rl['basis'] = 'actuator_space'  # Hardcoded
        self.env_geo = ao_env.AoEnv(self.config_geo, normalization_bool=False, geo_policy_testing=True)

    def obtain_original_freedom_vector_integrator(self, btt_modes_list):
        """
        Saves freedom parameter as originally
        :param btt_modes_list: list of btt modes
        :return: None
        """
        zn_norm_ = "zn_norm_"

        peak = np.max(btt_modes_list, axis=0)
        valley = np.min(btt_modes_list, axis=0)
        save_zernike = (np.abs(peak) + np.abs(valley)) / 2.0

        fd_original_freedom =\
            "src/reinforcement_learning/helper_functions/preprocessing/normalization/normalization_action_zernike/"
        print(
            "Saving zernike matrix for modes in "
            + fd_original_freedom + zn_norm_ + self.parameter_file_name)
        np.save(fd_original_freedom + zn_norm_ + self.parameter_file_name, save_zernike)

    def obtain_p2v_freedom_vectors(self,
                                   modes_array_from_volts_integrator,
                                   modes_array_projection,
                                   btt_integrator,
                                   btt_geometric):
        """
        Saves freedom parameter as the new way
        :param modes_array_from_volts_integrator: TODO
        :param modes_array_projection:
        :param btt_integrator: btt integrator modes
        :param btt_geometric: btt geometric modes
        :return: None
        """
        subtraction_projection = np.abs(modes_array_from_volts_integrator[1:, :] - modes_array_projection[1:, :])

        mean_projection = np.mean(subtraction_projection, axis=0)
        std_projection = np.std(subtraction_projection, axis=0)
        max_projection = np.max(subtraction_projection, axis=0)

        peak_2_valley_normal = np.abs(np.max(btt_integrator, axis=0)) + np.abs(np.min(btt_integrator, axis=0))
        peak_2_valley_geo = np.abs(np.max(btt_geometric, axis=0)) + np.abs(np.min(btt_geometric, axis=0))

        modes_array_subtraction = np.abs(btt_integrator - btt_geometric)
        max_difference = np.max(modes_array_subtraction, axis=0)

        fd = "src/reinforcement_learning/helper_functions/preprocessing/normalization/freedom_parameter/peak2valley/"
        print("Saving new parameters in " + fd)
        np.save(fd + self.parameter_file_name + "_integrator.npy", peak_2_valley_normal)
        np.save(fd + self.parameter_file_name + "_geo.npy", peak_2_valley_geo)
        np.save(fd + self.parameter_file_name + "_subtraction.npy", max_difference)

        # TODO old mean + std
        # mean_delay_not_into_account = np.mean(modes_array_subtraction, axis=0)
        # std_delay_not_into_account = np.std(modes_array_subtraction, axis=0)

        np.save(fd + self.parameter_file_name + "_mean2std.npy", mean_projection +
                2*std_projection)
        np.save(fd + self.parameter_file_name + "_mean3std.npy", mean_projection +
                3*std_projection)
        np.save(fd + self.parameter_file_name + "_mean4std.npy", mean_projection +
                4*std_projection)

        # Plot insights
        fd = "insights/freedom_parameter/" + self.parameter_file_name
        if not os.path.exists(fd):
            os.makedirs(fd)
        plt.plot(mean_projection + std_projection, label="mean+std")
        plt.plot(mean_projection + 2 * std_projection, label="mean+2std")
        plt.plot(max_projection, label="max")
        plt.plot(peak_2_valley_normal / 20.0, label="original/20")
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Mode")
        plt.ylabel("Freedom")
        plt.savefig("insights/freedom_parameter/" + self.parameter_file_name + "/projection_comparison")
        plt.close("all")

    def normalization_loop(self, geometric_controller):
        """
        run 10 normalization episodes
        :param geometric_controller: if we use the geometric controller parameter file
        :return: state_norm dictionary and btt_modes list
        """

        if geometric_controller:
            env = self.env_geo
        else:
            env = self.env_normal

        state_dm_list = []
        state_wfs_list = []
        state_dm_residual_list = []
        btt_modes_list = []

        modes_projection_list = []
        modes_from_volts_integrator_list = []

        seed = 0
        for episode in range(20):
            step = 0
            done = False
            seed += 1
            env.set_sim_seed(seed)
            env.reset(geometric_do_control=True, normalization_loop=True)

            while not done:
                # TODO: A warning saying: "mean of empty slice" can appear but it does not matter.
                integrator_modes_next, projection_modes_next = env.normalization_step()

                s_wfs_next = env.supervisor.rtc.get_slopes(0)
                s_dm_next = env.supervisor.rtc.get_command(0)
                if env.config_rl.env_rl['basis'] == "zernike_space":
                    s_dm_next = env.supervisor.volts2modes.dot(s_dm_next)

                if not geometric_controller:
                    s_dm_residual_next = env.supervisor.rtc.get_err(0)
                    if env.config_rl.env_rl['basis'] == "zernike_space":
                        s_dm_residual_next = env.supervisor.volts2modes.dot(s_dm_residual_next)

                btt_modes_list.append(self.v2m.dot(env.supervisor.rtc.get_command(0)))
                if not geometric_controller:
                    state_dm_list.append(s_dm_next)
                    state_wfs_list.append(s_wfs_next)
                    state_dm_residual_list.append(s_dm_residual_next)
                    modes_projection_list.append(projection_modes_next)
                    modes_from_volts_integrator_list.append(integrator_modes_next)

                step += 1

                if step >= self.config_normal.env_rl["max_steps_per_episode"]:
                    done = True

            # TODO: Geometric controller will show DM shape as actuator shape not modal shape and 0 in wfs shape
            print("Normalization episode:", episode+1,
                  "DM shape:", s_dm_next.shape,
                  "WFS shape:", s_wfs_next.shape,
                  "Steps:", step,
                  "Len state vector", len(state_dm_list),
                  "Seed", seed,
                  "Gain", round(env.supervisor.rtc._rtc.d_control[0].gain, 3),
                  "L.E. SR", round(env.supervisor.target.get_strehl(0)[1], 4))

        if geometric_controller:
            norm_dic = {}
        else:
            state_dm_list = np.array(state_dm_list)
            state_wfs_list = np.array(state_wfs_list)

            _mean_dm = np.mean(state_dm_list, axis=0)
            _std_dm = np.std(state_dm_list, axis=0)
            _max_dm = np.max(state_dm_list, axis=0)
            _min_dm = np.min(state_dm_list, axis=0)

            _mean_wfs = np.mean(state_wfs_list, axis=0)
            _std_wfs = np.std(state_wfs_list, axis=0)
            _max_wfs = np.max(state_wfs_list, axis=0)
            _min_wfs = np.min(state_wfs_list, axis=0)

            _mean_dm_residual = np.mean(state_dm_residual_list, axis=0)
            _std_dm_residual = np.std(state_dm_residual_list, axis=0)
            _max_dm_residual = np.max(state_dm_residual_list, axis=0)
            _min_dm_residual = np.min(state_dm_residual_list, axis=0)

            norm_dic =\
                {
                 "dm": {
                    "mean": _mean_dm,
                    "std": _std_dm,
                    "max": _max_dm,
                    "min": _min_dm
                     },
                 "wfs": {
                    "mean": _mean_wfs,
                    "std": _std_wfs,
                    "max": _max_wfs,
                    "min": _min_wfs
                      },
                 "dm_residual": {
                     "mean": _mean_dm_residual,
                     "std": _std_dm_residual,
                     "max": _max_dm_residual,
                     "min": _min_dm_residual
                 }
                }

        return norm_dic,\
            np.array(btt_modes_list),\
            np.array(modes_from_volts_integrator_list),\
            np.array(modes_projection_list)


def run_obtain_normalization_and_freedom(parameter_file,
                                         basis,
                                         pure_delay_0,
                                         modes_filtered,
                                         include_plots=False,
                                         autoencoder_path=None):
    """
    run 10 episodes to obtain normalization and freedom_parameter
    :param parameter_file: current parameter file
    :param basis: zernike of actuator space
    :param include_plots: if we include plots of some metrics
    :param pure_delay_0: if the delay is actually 0
    :param modes_filtered: how many modes we filter
    :param autoencoder_path: the autoencoder path
    :return: None
    """
    if modes_filtered < 0:
        modes_filtered = 0
    print("Starting normalization, parameter file:", parameter_file,
          "Basis:", basis,
          "Pure delay 0:", pure_delay_0,
          "Number of modes filtered of cmat:", modes_filtered)

    preprocessor = Preprocessor(parameter_file, basis, pure_delay_0, modes_filtered, autoencoder_path=autoencoder_path)

    normalization_dict_integrator, btt_modes_integrator,\
        modes_array_from_volts_integrator, modes_array_projection =\
        preprocessor.normalization_loop(geometric_controller=False)

    _, btt_modes_geometric, _, _ = preprocessor.normalization_loop(geometric_controller=True)

    if basis == "zernike_space":
        parameter_file = parameter_file[:-3] + "_zernike_space"
    else:
        parameter_file = parameter_file[:-3]

    fd_normalization =\
        "src/reinforcement_learning/helper_functions/preprocessing/normalization/state_normalization/normalization_"
    print("Saving parameters in " + fd_normalization + parameter_file + ".pickle")

    with open(fd_normalization + parameter_file + ".pickle", 'wb') as handle:
        pickle.dump(normalization_dict_integrator, handle)

    preprocessor.obtain_original_freedom_vector_integrator(btt_modes_integrator)
    preprocessor.obtain_p2v_freedom_vectors(modes_array_from_volts_integrator,
                                            modes_array_projection,
                                            btt_modes_integrator,
                                            btt_modes_geometric)
