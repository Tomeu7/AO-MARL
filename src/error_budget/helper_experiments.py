def choose_experiment_error_budget_multiple_agents(experiment_name):
    if experiment_name == "14_7_3l_dir":
        par_file_list = ["production_sh_40x40_8m_3layers_same_dir_roket.py"]
        modes_filtered_list = [5]
        policy_names = []
        for worker_id in range(1, 44):
            policy_names.append("14_7_3l_direxperiment_name_worker_" + str(worker_id) + "_sac_actor_14_7_3l_dir")

        n_zernike_start = 0
        n_zernike_end = 1260
        policy_folder = "output/output_models/models_rpc/"
        policy_epoch = 2000

        policy_paths = {}
        worker_id = 1
        for item in policy_names:
            policy_paths[worker_id] = policy_folder + "14_7_3l_dir/" + item + "_episode_" + str(policy_epoch)
            worker_id += 1
    elif experiment_name == "14_8_3l_dir_w20":
        par_file_list = ["production_sh_40x40_8m_3layers_same_dir_roket.py"]
        modes_filtered_list = [5]
        policy_names = []
        for worker_id in range(1, 44):
            policy_names.append("14_8_3l_dir_w20experiment_name_worker_" + str(worker_id) +
                                "_sac_actor_14_8_3l_dir_w20")

        n_zernike_start = 0
        n_zernike_end = 1260
        policy_folder = "output/output_models/models_rpc/"
        policy_epoch = 2000

        policy_paths = {}
        worker_id = 1
        for item in policy_names:
            policy_paths[worker_id] = policy_folder + "14_8_3l_dir_w20/" + item + "_episode_" + str(policy_epoch)
            worker_id += 1
    elif experiment_name == "15_21_w20":
        par_file_list = ["production_sh_40x40_8m_3layers_same_dir_v_10_5_15.py"]
        modes_filtered_list = [5]
        policy_names = []
        for worker_id in range(1, 44):
            policy_names.append("15_21_w20experiment_name_worker_" + str(worker_id) +
                                "_sac_actor_15_21_w20")

        n_zernike_start = 0
        n_zernike_end = 1260
        policy_folder = "output/output_models/models_rpc/"
        policy_epoch = 2000

        policy_paths = {}
        worker_id = 1
        for item in policy_names:
            policy_paths[worker_id] = policy_folder + "15_21_w20/" + item + "_episode_" + str(policy_epoch)
            worker_id += 1
    elif experiment_name == "15_31_w20":
        par_file_list = ["production_sh_40x40_8m_3layers_same_dir_v_20_15_25.py"]
        modes_filtered_list = [5]
        policy_names = []
        for worker_id in range(1, 44):
            policy_names.append("15_31_w20experiment_name_worker_" + str(worker_id) +
                                "_sac_actor_15_31_w20")

        n_zernike_start = 0
        n_zernike_end = 1260
        policy_folder = "output/output_models/models_rpc/"
        policy_epoch = 2000

        policy_paths = {}
        worker_id = 1
        for item in policy_names:
            policy_paths[worker_id] = policy_folder + "15_31_w20/" + item + "_episode_" + str(policy_epoch)
            worker_id += 1
    else:
        raise NotImplementedError

    return policy_names, par_file_list, policy_paths, modes_filtered_list, n_zernike_start, n_zernike_end