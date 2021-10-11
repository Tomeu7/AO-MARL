from argparse import Namespace

def choose_experiment_two_agents(experiment_id):
    if experiment_id == 1:
        print("ID 1: Two agents test experiment delay 2")
        experiment_1 = {"experiment_name": "paper_modal_3_3_s1"}
        experiment_2 = {"experiment_name": "paper_modal_3_3_s2"}
        experiment_3 = {"experiment_name": "paper_modal_3_3_s3"}
        experiment_4 = {"experiment_name": "paper_modal_3_3_s4"}
        experiment_5 = {"experiment_name": "paper_modal_3_3_s5"}
        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["use_two_agents"] = "True"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2

        labels = {"paper_modal_3_3_s1": 'RL two agents seed 1',
                  "paper_modal_3_3_s2": 'RL two agents seed 2',
                  "paper_modal_3_3_s3": 'RL two agents seed 3',
                  "paper_modal_3_3_s4": 'RL two agents seed 4',
                  "paper_modal_3_3_s5": 'RL two agents seed 5'}
        prove_model_predictive_control = False
    elif experiment_id == 2:
        print("ID 2: Original agent test experiment delay 2")
        experiment_1 = {"experiment_name": "paper_original_s1"}
        experiment_2 = {"experiment_name": "paper_original_s2"}
        experiment_3 = {"experiment_name": "paper_original_s3"}
        experiment_4 = {"experiment_name": "paper_original_s4"}
        experiment_5 = {"experiment_name": "paper_original_s5"}
        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["use_two_agents"] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2

        labels = {"paper_original_s1": 'RL original seed 1',
                  "paper_original_s2": 'RL original seed 2',
                  "paper_original_s3": 'RL original seed 3',
                  "paper_original_s4": 'RL original seed 4',
                  "paper_original_s5": 'RL original seed 5'}
        prove_model_predictive_control = False
    elif experiment_id == 3:
        print("ID 3: proving MPC with two agents")
        experiment_1 = {"experiment_name": "s1"}
        experiment_2 = {"experiment_name": "s2"}
        experiment_3 = {"experiment_name": "s3"}
        experiment_4 = {"experiment_name": "s4"}
        experiment_5 = {"experiment_name": "s5"}
        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]
        # experiment_1 = {"experiment_name": "paper_methods_comparison"}
        # experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["use_two_agents"] = "True"
            experiment['comparison_original_vs_two_agents'] = "True"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2

        labels = {"19_2m_seed1": 'RL original seed 1',
                  "19_2m_seed2": 'RL original seed 2',
                  "19_2m_seed3": 'RL original seed 3',
                  "19_2m_seed4": 'RL original seed 4',
                  "19_2m_seed5": 'RL original seed 5'}
        prove_model_predictive_control = True
    elif experiment_id == 4:
        prove_model_predictive_control = False
        print("ID 4: scaling experiment")

        experiment_1 = {"experiment_name": "paper_modal_sc"}
        experiments = [experiment_1]
        labels = {"paper_modal_sc": 'RL sequential agent'}
        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["episode_model"] = "3000"
            experiment["sequential_agents"] = "True"
            # noinspection PyTypeChecker
            experiment["agents_start_end_list"] = [[0, 25], [25, 52], [52, 84], [84, 86]] # TODO NOTE corrected
            # noinspection PyTypeChecker
            experiment["agents_weight_path"] = ["paper_modal_sc1_s1",
                                                "paper_modal_sc2_s1",
                                                "paper_modal_sc3_s1",
                                                "paper_modal_sc4_s1"]
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
    elif experiment_id == 5:
        prove_model_predictive_control = False
        print("ID 4: scaling experiment")

        experiment_1 = {"experiment_name": "scaling_average"}
        experiments = [experiment_1]
        labels = {"paper_modal_sc": 'RL sequential agent 2'}
        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "True"
            # noinspection PyTypeChecker
            experiment["agents_start_end_list"] = [[0, 25], [25, 52], [52, 84], [84, 86]]  # TODO NOTE corrected
            # noinspection PyTypeChecker
            experiment["agents_weight_path"] = ["scaling_average_1",
                                                "scaling_average_2",
                                                "scaling_average_3",
                                                "paper_modal_sc4_s1"]
            experiment["episode_model"] = "2000"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
    elif experiment_id == 6:
        prove_model_predictive_control = False
        print("ID 6: scaling parallel")

        experiment_1 = {"experiment_name": "paper_modal_sc"}
        experiments = [experiment_1]
        labels = {"paper_modal_sc": 'RL sequential agent'}
        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["episode_model"] = "3000"
            experiment["sequential_agents"] = "True"
            # noinspection PyTypeChecker
            experiment["agents_start_end_list"] = [[0, 25], [25, 52], [52, 84], [84, 86]]  # TODO NOTE corrected
            # noinspection PyTypeChecker
            experiment["agents_weight_path"] = ["parallel_1_episode_3500",
                                                "parallel_2_episode_3500",
                                                "parallel_3_episode_3500",
                                                "parallel_4_episode_3500"]
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
    elif experiment_id == 7:
        print("ID 7: proving MPC with two agents")
        experiment_1 = {"experiment_name": "7_13_2m_20x20"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2

        labels = {"7_13_2m_20x20": 'RL 20x20 1'}
        prove_model_predictive_control = False
    elif experiment_id == 8:
        print("ID 8: proving MPC with two agents")
        experiment_1 = {"experiment_name": "8_20x20_1_v2"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 313
            experiment['episode_model'] = 5500
            experiment['hidden_size_actor'] = 256

        labels = {"8_20x20_2_v2": 'RL 20x20 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 9:
        print("ID 9: proving MPC with two agents")
        experiment_1 = {"experiment_name": "8_20x20_1_v2"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 313
            experiment['episode_model'] = 2000

        labels = {"8_20x20_2_v2": 'RL 20x20 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 10:
        print("ID 10: proving MPC with two agents")
        experiment_1 = {"experiment_name": "8_44_w4"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 75
            experiment['episode_model'] = 4000
            experiment['hidden_size_actor'] = 128

        labels = {"test_1_v500_v2": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 11:
        print("ID 11")
        experiment_1 = {"experiment_name": "9_2m_10x10_1"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 75
            experiment['episode_model'] = 5000
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['hidden_size_actor'] = 256

        labels = {"9_2m_10x10_1": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 12:
        print("ID 12")
        experiment_1 = {"experiment_name": "9_2m_18x18_1"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_18x18_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 267
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['episode_model'] = 4000
            experiment['hidden_size_actor'] = 256

        labels = {"9_2m_18x18_1": 'RL 18x18 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 13:
        print("ID 13")
        experiment_1 = {"experiment_name": "9_4m_10x10_1"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 75
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['episode_model'] = 5000
            experiment['hidden_size_actor'] = 256
            # experiment['custom_freedom_path'] = "normalization_action_zernike/zn_norm_production_sh_10x10_2m"

        labels = {"test_1_v500_v2": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 14:
        print("ID 14")
        experiment_1 = {"experiment_name": "9_4m_18x18_1"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_18x18_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 267
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['episode_model'] = 4000
            experiment['hidden_size_actor'] = 256

        labels = {"test_1_v500_v2": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 15:
        print("ID 15")
        experiment_1 = {"experiment_name": "9_4m_20x20_rest_deactivated"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 70
            experiment["n_reverse_filtered_from_cmat"] = 263
            experiment['episode_model'] = 4500
            experiment['hidden_size_actor'] = 256

        labels = {"9_4m_20x20_rest_deactivated": 'RL 20x20 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 16:
        print("ID 16")
        experiment_1 = {"experiment_name": "9_4m_18x18_rest_deactivated_3"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_18x18_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 70
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['episode_model'] = 2500
            experiment['hidden_size_actor'] = 256
            experiment['other_modes_from_integrator_deactivated'] = "True"

        labels = {"9_4m_20x20_rest_deactivated": 'RL 20x20 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 17:
        print("ID 11")
        experiment_1 = {"experiment_name": "9_2m_10x10_log_var"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 75
            experiment['episode_model'] = 4000
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['hidden_size_actor'] = 256

        labels = {"9_2m_10x10_log_var": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 18:
        print("ID 11")
        experiment_1 = {"experiment_name": "9_4m_20x20_log_var"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 303
            experiment['episode_model'] = 4500
            experiment["n_reverse_filtered_from_cmat"] = 30
            experiment['hidden_size_actor'] = 256
            experiment['norm_scale'] = 10.0

        labels = {"9_4m_20x20_log_var": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 19:
        print("ID 11")
        experiment_1 = {"experiment_name": "9_4m_20x20_new_freed_log_var"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_20x20_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 303
            experiment['episode_model'] = 4500
            experiment["n_reverse_filtered_from_cmat"] = 30
            experiment['hidden_size_actor'] = 256
            experiment['custom_freedom_path'] = "freedom_parameter/new_freedom/freedom4m20"
            experiment['norm_scale'] = 1.0

        labels = {"9_4m_20x20_log_var": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    elif experiment_id == 20:
        print("ID 11")
        experiment_1 = {"experiment_name": "9_4m_18x18_log_var"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_18x18_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 267
            experiment['episode_model'] = 4500
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['hidden_size_actor'] = 256
            experiment['norm_scale'] = 10.0

        labels = {"9_4m_18x18_log_var": 'RL 18x18 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 21:
        print("ID 21")
        experiment_1 = {"experiment_name": "9_4m_18x18_new_freed_log_var"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_18x18_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 267
            experiment['episode_model'] = 4000
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['hidden_size_actor'] = 256
            experiment['norm_scale'] = 1.0
            experiment['custom_freedom_path'] = "freedom_parameter/new_freedom/freedom4m18"

        labels = {"9_4m_18x18_new_freed_log_var": 'RL 18x18 4m'}
        prove_model_predictive_control = False
    elif experiment_id == 22:
        print("ID 22:")
        experiment_1 = {"experiment_name": "9_4m_10x10_rest_deactivated_5"}

        experiments = [experiment_1]

        for experiment in experiments:
            experiment["RL"] = "True"
            experiment["sequential_agents"] = "False"
            experiment["use_two_agents"] = "False"
            experiment['comparison_original_vs_two_agents'] = "False"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "production_sh_10x10_4m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2
            experiment["n_zernike"] = 75
            experiment['episode_model'] = 5000
            experiment["n_reverse_filtered_from_cmat"] = 10
            experiment['hidden_size_actor'] = 256
            experiment['norm_scale'] = 10.0

        labels = {"9_4m_10x10_rest_deactivated_5": 'RL 10x10 2m'}
        prove_model_predictive_control = False
    else:
        raise NotImplementedError

    return experiments, labels, prove_model_predictive_control

def choose_args_testing(args_config, seed):
    """
    use_geo_parameter_file = True <-
    use_pure_delay_0 = False <- use_pure_delay_0_args
    test_mpc = True <-
    do_special_test = False <-
    """
    print("Entering manual args")

    default_value_key = \
        {'state_dm_before_linear': "True",
         'state_dm_after_linear': "False",
         'state_wfs': "True",
         'policy_args': "Gaussian",
         'n_zernike': 84,
         'number_of_previous_wfs': 0,
         'number_of_previous_dm': 2,
         'include_tip_tilt': "True",
         'norm_scale': 10.0,
         'normalization_std_inside_environment': 1.0,
         'modification_online': "False",
         'use_two_agents': "False",
         'hidden_size_actor': 256,
         'n_reverse_filtered_from_cmat':10,
         'custom_freedom_path':None,
         'other_modes_from_integrator_deactivated':"False"}

    used_value_key = {}

    for key, value in default_value_key.items():
        if key in args_config.keys():
            used_value_key[key] = args_config[key]
        else:
            used_value_key[key] = default_value_key[key]

    if default_value_key['use_two_agents'] == "True":  # In case of two agents we must provide two rewards
        reward_type = ['wavefront_phase_error', 'wavefront_phase_error']
    else:
        reward_type = 'log_var'

    args = Namespace(automatic_entropy_tuning="True",
                     alpha=0.2,
                     batch_size=64,
                     experiment_name=args_config['experiment_name'],
                     max_steps_per_episode=1000,
                     min_alpha="False",
                     reward_type=reward_type,
                     seed=seed,
                     actuator_vers_2="False",
                     actuator_vers_2_division_param=1.0,
                     seed_same="False",
                     level=args_config['level'],
                     memory_size=1000000,
                     lr=0.0003,
                     hidden_size_critic=[256],
                     hidden_size_actor=args_config['hidden_size_actor'],
                     num_layers_critic=2,
                     num_layers_actor=2,
                     gaussian_mu=0,
                     gaussian_std=1,
                     policy=used_value_key['policy_args'],
                     parameters_telescope=args_config['parameter_file'],
                     integration_mode="after_integration",
                     updates_per_step=1,
                     delayed_assignment=1,
                     gamma=0.1,
                     state_dm_before_linear=used_value_key['state_dm_before_linear'],
                     state_dm_after_linear=used_value_key['state_dm_after_linear'],
                     state_wfs=used_value_key['state_wfs'],
                     state_gain="False",
                     number_of_seeds=-1,
                     move_atmos="True",
                     algorithm="SAC",
                     reward_mode="shaped",
                     basis=args_config['basis'],
                     remove_integrator="False",
                     n_zernike_reverse=-1,
                     RTAC="False",
                     n_zernike=used_value_key['n_zernike'],
                     number_of_previous_wfs=used_value_key['number_of_previous_wfs'],
                     number_of_previous_dm=used_value_key['number_of_previous_dm'],
                     sum_rewards="False",
                     include_tip_tilt=used_value_key['include_tip_tilt'],
                     norm_scale_zernike_actions=used_value_key['norm_scale'],
                     normalization_std_inside_environment=used_value_key['normalization_std_inside_environment'],
                     normalization_mean_inside_environment=0,
                     only_last_dm="False",
                     activation="relu",
                     ratio_update_critic_vs_actor=1,
                     sac_reward_scaling=1.0,
                     grad_control_matrix="False",
                     use_as_integrator="False",
                     pretrained_model_path="None",
                     output_normalization="False",
                     deterministic_std_noise=0.1,
                     no_squashing="False",
                     use_geo_norm_scale="False",
                     modification_online=used_value_key['modification_online'],
                     zernike_physics="False",
                     zernike_physics_factor=1.0,
                     norm_physics_std="False",
                     norm_original="False",
                     load_previous_weights="False",
                     use_cnn_normalized="False",
                     use_cnn_unnormalized="False",
                     tau=0.005,
                     use_baseline_policy="No",
                     autoencoder_path=None,
                     autoencoder_type="cnn_single_subaperture",
                     use_two_agents=used_value_key['use_two_agents'],
                     create_norm_param="False",
                     custom_freedom_path=used_value_key['custom_freedom_path'],
                     n_zernike_start_end=[-1, -1],
                     sequential_agents_experiment = None,
                     sequential_agents_experiment_v2=None,
                     other_modes_from_integrator_deactivated=used_value_key['other_modes_from_integrator_deactivated'],
                     rl_through_modal_only="False",
                     rl_control_single_step="False",
                     n_zernike_discarded="None",
                     only_rl_integrated="False",
                     n_reverse_filtered_from_cmat=used_value_key['n_reverse_filtered_from_cmat'],
                     residual_actions_0="False",
                     residual_reset_memory="False",
                     critic_beta=-1,
                     residual_add_noise=0,
                     residual_prob_random_actions=-1,
                     residual_one_action_at_a_time="False",
                     initialize_last_layer_0="False",
                     initialize_last_layer_near_0="False",
                     exponential_entropy_tuning="False",
                     linear_entropy_tuning="False",
                     save_replay_buffer="False",
                     save_rewards_buffer="False",
                     state_dm_residual="False",
                     huber_loss=-1,
                     divide_alpha=-1,
                     replay_path=None
                     )
    args.mbpo = False
    args.seed = seed

    return args

def choose_experiment(experiment_id):
    """
    Depending on the experiment number you choose an experiment
    """

    if experiment_id == 1:
        experiment_1 = {"fd": "part_19_2m",
                        "experiment_name": "19_2m_1",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_8pix_2m.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"19_2m_1": 'Correction RL', "18_2m_onlyrl_4": "Full RL"}
        experiments = [experiment_1]
    elif experiment_id == 2:
        experiment_1 = {"fd": "part_19_d0",
                        "experiment_name": "19_2m_d0_True",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_8pix_2m_delay0.py",
                        "number_of_previous_dm": 2,
                        "modification_online": "True"
                        }
        labels = {"19_2m_d0_True": 'Correction RL delay 0'}
        experiments = [experiment_1]
    elif experiment_id == 3:
        experiment_1 = {"fd": "18and19",
                        "experiment_name": "19_4m_deter5",
                        "zernike_space": False,
                        "only_rl": False,
                        "policy": "Deterministic",
                        "number_of_previous_dm": 2
                        }
        labels = {"19_4m_deter5": 'Correction RL 4m'}
        experiments = [experiment_1]
    elif experiment_id == 4:
        experiment_1 = {"experiment_name": "19_2m_1"}
        experiment_2 = {"experiment_name": "19_2m_1seed2"}
        experiment_3 = {"experiment_name": "19_2m_1seed3"}
        experiment_4 = {"experiment_name": "19_2m_1seed4"}
        experiment_5 = {"experiment_name": "19_2m_1seed5"}
        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["fd"] = "models_I_need/normal_final"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "scao_sh_10x10_8pix_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 2

        labels = {"19_2m_seed1": 'RL controller seed 1',
                  "19_2m_seed2": 'RL controller seed 2',
                  "19_2m_seed3": 'RL controller seed 3',
                  "19_2m_seed4": 'RL controller seed 4',
                  "19_2m_seed5": 'RL controller seed 5'}
    elif experiment_id == 5:
        experiment_1 = {"experiment_name": "19_2m_d0_True"}
        experiment_2 = {"experiment_name": "19_2m_d0_seed2"}
        experiment_3 = {"experiment_name": "19_2m_d0_seed3"}
        experiment_4 = {"experiment_name": "19_2m_d0_seed4"}
        experiment_5 = {"experiment_name": "19_2m_d_seed5"}

        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["fd"] = "models_I_need/d0_final"
            # noinspection PyTypeChecker
            experiment["modification_online"] = "True"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "scao_sh_10x10_8pix_2m_delay0.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 0

        labels = {"19_2m_d0_True": 'Correction RL delay 0 seed 1',
                  "19_2m_d0_seed2": 'Correction RL delay 0 seed 2',
                  "19_2m_d0_seed3": 'Correction RL delay 0 seed 3',
                  "19_2m_d0_seed4": 'Correction RL delay 0 seed 4',
                  "19_2m_d_seed5": 'Correction RL delay 0 seed 5'}
    elif experiment_id == 6:
        experiment_6 = {"fd": "only_rl",
                        "experiment_name": "2",
                        "basis": "actuator_space",
                        "level": "only_rl",
                        "parameter_file": "scao_sh_10x10_8pix_2m.py",
                        "normalization_std_inside_environment": 100.0,
                        "number_of_previous_dm": 2,
                        "state_dm_after_linear": "True"
                        }

        labels = {"2": 'Only RL'}
        experiments = [experiment_6]
    elif experiment_id == 7:
        experiment_7 = {"fd": "only_rl_zernike",
                        "experiment_name": "o_6",
                        "basis": "zernike_space",
                        "level": "only_rl",
                        "parameter_file": "scao_sh_10x10_8pix_2m.py",
                        "number_of_previous_dm": 2,
                        "norm_scale": 1.0
                        }
        labels = {"o_6": 'Only RL'}
        experiments = [experiment_7]
    elif experiment_id == 8:
        experiment_7 = {"fd": "correction_c_9",
                        "experiment_name": "c9",
                        "basis": "actuator_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_8pix_2m.py",
                        "number_of_previous_dm": 2,
                        "normalization_std_inside_environment": 2.0
                        }
        labels = {"c9": 'Correction RL Actuators'}
        experiments = [experiment_7]
    elif experiment_id == 9:
        experiment_1 = {"experiment_name": "19_2m_dm_hist_1"}
        experiment_2 = {"experiment_name": "19_2m_dm_hist_1_seed2"}
        experiment_3 = {"experiment_name": "19_2m_dm_hist_1_seed3"}
        experiment_4 = {"experiment_name": "19_2m_dm_hist_1_seed4"}
        experiment_5 = {"experiment_name": "19_2m_dm_hist_1_seed5"}

        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["fd"] = "history"
            # noinspection PyTypeChecker
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "scao_sh_10x10_8pix_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 0

        labels = {"19_2m_dm_hist_1": 'RL n = 0 seed 1',
                  "19_2m_dm_hist_1_seed2": 'RL n = 0  seed 2',
                  "19_2m_dm_hist_1_seed3": 'RL n = 0  seed 3',
                  "19_2m_dm_hist_1_seed4": 'RL n = 0  seed 4',
                  "19_2m_dm_hist_1_seed5": 'RL n = 0  seed 5'}
    elif experiment_id == 10:
        # experiment_1 = {"experiment_name": "19_2m_dm_hist_2"}
        # experiment_2 = {"experiment_name": "19_2m_dm_hist_2seed2"}
        experiment_3 = {"experiment_name": "19_2m_dm_hist_2seed3"}
        experiment_4 = {"experiment_name": "19_2m_dm_hist_2seed4"}
        experiment_5 = {"experiment_name": "19_2m_dm_hist_2seed5"}

        experiments = [experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["fd"] = "history"
            # noinspection PyTypeChecker
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "scao_sh_10x10_8pix_2m.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 4

        labels = {"19_2m_dm_hist_2seed2": 'RL n = 4  seed 2',
                  "19_2m_dm_hist_2seed3": 'RL n = 4  seed 3',
                  "19_2m_dm_hist_2seed4": 'RL n = 4  seed 4',
                  "19_2m_dm_hist_2seed5": 'RL n = 4  seed 5'}
    elif experiment_id == 11:
        experiment_1 = {"fd": "part_19_2m",
                        "experiment_name": "19_2m_1",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_8pix_2m_modal_gain_optimization.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"19_2m_1": 'Correction RL'}
        experiments = [experiment_1]
    elif experiment_id == "4m":
        experiment_1 = {"fd": "None",
                        "experiment_name": "None",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_20x20_8pix_4m.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"None": "None"}
        experiments = [experiment_1]
    elif experiment_id == "8m":
        experiment_1 = {"fd": "None",
                        "experiment_name": "None",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_40x40_8pix_8m.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"None": "None"}
        experiments = [experiment_1]
    elif experiment_id == "noise3gs9":
        experiment_1 = {"fd": "None",
                        "experiment_name": "None",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_16pix_2m_gs9_noise3_delay0.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"19_2m_1": 'Correction RL'}
        experiments = [experiment_1]
    elif experiment_id == "noise3gs10":
        experiment_1 = {"fd": "None",
                        "experiment_name": "None",
                        "basis": "zernike_space",
                        "level": "correction",
                        "parameter_file": "scao_sh_10x10_16pix_2m_gs10_noise3_delay0.py",
                        "number_of_previous_dm": 2
                        }
        labels = {"19_2m_1": 'Correction RL'}
        experiments = [experiment_1]
    elif experiment_id == 12:
        experiment_1 = {"experiment_name": "noise-1_delay0_gs4dm0seed1_again"}
        experiment_2 = {"experiment_name": "noise-1_delay0_gs4dm0seed2"}
        experiment_3 = {"experiment_name": "noise-1_delay0_gs4dm0seed3"}
        experiment_4 = {"experiment_name": "noise-1_delay0_gs4dm0seed4"}
        experiment_5 = {"experiment_name": "noise-1_delay0_gs4dm0seed5"}

        experiments = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5]

        for experiment in experiments:
            experiment["fd"] = "delay0_optimized_models"
            # noinspection PyTypeChecker
            experiment["modification_online"] = "True"
            experiment["basis"] = "zernike_space"
            # noinspection PyTypeChecker
            experiment["level"] = "correction"
            experiment["parameter_file"] = "scao_sh_10x10_16pix_2m_gs4_noise-1_delay0.py"
            # noinspection PyTypeChecker
            experiment["number_of_previous_dm"] = 0

        labels = {"noise-1_delay0_gs4dm0seed1_again": 'Correction RL delay 0 seed 1',
                  "noise-1_delay0_gs4dm0seed2": 'Correction RL delay 0 seed 2',
                  "noise-1_delay0_gs4dm0seed3": 'Correction RL delay 0 seed 3',
                  "noise-1_delay0_gs4dm0seed4": 'Correction RL delay 0 seed 4',
                  "noise-1_delay0_gs4dm0seed5": 'Correction RL delay 0 seed 5'}
    else:
        raise NotImplementedError

    return experiments, labels
