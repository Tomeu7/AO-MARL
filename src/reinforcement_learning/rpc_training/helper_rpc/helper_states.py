import numpy as np
import pandas as pd

debug_modes_chosen = True
debug_augmented_qnetwork = False

def select_correct_modes_for_array(agent_value_0, agent_value_1,
                                   total_existing_modes, total_controlled_modes, starting_mode,
                                   include_tip_tilt):
    # 1. As self.current_action = np.zeros(len_actions)
    # 2. And self.dictionary we have the current modes controlled e.g. 300 to 600
    # 3. From 300 and 600 we will remove 300 (self.starting_mode) to fit into the array
    # 4. Ending with bottom_mode 0 and top_mode 300
    # The same goes for the state when doing divide_states

    # First if is due to tip tilt
    if include_tip_tilt \
            and agent_value_0 == (total_existing_modes - 2) and agent_value_1 == total_existing_modes:
        bottom_mode = total_controlled_modes - 2 - starting_mode
        top_mode = total_controlled_modes - starting_mode
    else:
        bottom_mode = agent_value_0 - starting_mode
        top_mode = agent_value_1 - starting_mode
    if debug_augmented_qnetwork:
        print("agent value 0 {}, agent value 1 {}, total_existing_modes {}, total_controlled_modes {}, include_tip_tilt {}".format(
              agent_value_0, agent_value_1, total_existing_modes, total_controlled_modes, include_tip_tilt))
    return bottom_mode, top_mode


def modes_chosen_original(dictionary_agents, indices_of_state,
                          total_existing_modes, total_controlled_modes, starting_mode, include_tip_tilt,
                          experiment_name):
    """
    + Input state: for original state has shape ((n_zernike_start_end[1]-n_zernike_start_end[0]) x 4
     ( 1 from d_err, 1 from C_t-1 state before linear and 2 from C_t-2 and C_t-3)
     ----------------------------------------------------------------------------------------------------------
    """
    modes_chosen_dict = {}
    for worker_id, agent_values in dictionary_agents.items():
        bottom_mode_for_array, top_mode_for_array = select_correct_modes_for_array(agent_values[0],
                                                                                   agent_values[1],
                                                                                   total_existing_modes,
                                                                                   total_controlled_modes,
                                                                                   starting_mode,
                                                                                   include_tip_tilt)



        modes_chose_list_current_worker = []
        for key in indices_of_state:
            if key in "wfs":
                # FOR THE WFS WE INCLUDE ALL
                modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                       indices_of_state[key][1])
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            else:
                modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0] + bottom_mode_for_array,
                                                                       indices_of_state[key][0] + top_mode_for_array)
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
        modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)

    if debug_modes_chosen:
        import os
        if not os.path.exists("output/debug/"):
            os.makedirs("output/debug/")
        df = pd.DataFrame.from_dict(modes_chosen_dict, orient='index')
        df.transpose().to_csv("output/debug/" + experiment_name + "_modes_chosen_original.csv", index=False)

    return modes_chosen_dict

def modes_chosen_tt_treated_as_mode(dictionary_agents, indices_of_state, experiment_name):
    """
    + Input state: for original state has shape ((n_zernike_start_end[1]-n_zernike_start_end[0]) x 4
     ( 1 from d_err, 1 from C_t-1 state before linear and 2 from C_t-2 and C_t-3)
     ----------------------------------------------------------------------------------------------------------
    """
    modes_chosen_dict = {}
    for worker_id, agent_values in dictionary_agents.items():

        modes_chose_list_current_worker = []
        for key in indices_of_state:
            modes_chosen_current_worker_index_of_state = agent_values + indices_of_state[key][0]
            modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
        modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)

    if debug_modes_chosen:
        df = pd.DataFrame.from_dict(modes_chosen_dict, orient='index')
        df.transpose().to_csv("output/debug/" + experiment_name + "_modes_chosen_original.csv", index=False)

    return modes_chosen_dict


def modes_chosen_additional(dictionary_agents, indices_of_state, additional_state_start_end, include_tip_tilt,
                            experiment_name):
    modes_chosen_dict = {}
    for worker_id, agent_values in dictionary_agents.items():
        modes_chose_list_current_worker = []
        # I use len self.dictionary as worker id starts at 1,
        # e.g. 1 worker for modes and 1 for TT
        # TT id is 2 and len(self.dictionary_agents) = 2
        if include_tip_tilt and worker_id == len(dictionary_agents):
            for key in indices_of_state:
                if key in "wfs":
                    # FOR THE WFS WE INCLUDE ALL
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                           indices_of_state[key][1])
                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
                else:
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][1] - 2,
                                                                           indices_of_state[key][1])
                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)
            break

        # self.select_correct_modes_for_array not needed for separate_states_n_zernike
        bottom_mode_for_array, top_mode_for_array = agent_values[0], agent_values[1]
        for key in indices_of_state:
            if key in "wfs":
                # FOR THE WFS WE INCLUDE ALL
                modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                       indices_of_state[key][1])
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            else:
                starting_point = indices_of_state[key][0]

                # original modes
                modes_chosen_current_worker_index_of_state = set(np.arange(starting_point + bottom_mode_for_array,
                                                                           starting_point + top_mode_for_array))
                # additional modes
                additional_modes = set(np.arange(starting_point + additional_state_start_end[0],
                                                 starting_point + additional_state_start_end[1]))

                # union of the two sets so we discard repeated modes
                modes_chosen_current_worker_index_of_state = \
                    list(modes_chosen_current_worker_index_of_state.union(additional_modes))

                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
        modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)

    if debug_modes_chosen:
        df = pd.DataFrame.from_dict(modes_chosen_dict, orient='index')
        df.transpose().to_csv("output/debug/" + experiment_name + "_modes_chosen_additional.csv", index=False)

    return modes_chosen_dict


def modes_chosen_seq2seq(dictionary_agents, indices_of_state, seq2seq_partial, include_tip_tilt,
                         experiment_name):
    modes_chosen_dict = {}
    for worker_id, agent_values in dictionary_agents.items():
        modes_chose_list_current_worker = []
        # I use len self.dictionary as worker id starts at 1,
        # e.g. 1 worker for modes and 1 for TT
        # TT id is 2 and len(self.dictionary_agents) = 2
        if include_tip_tilt and worker_id == len(dictionary_agents):
            for key in indices_of_state:
                if key in "wfs":
                    # FOR THE WFS WE INCLUDE ALL
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                           indices_of_state[key][1])
                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
                else:
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][1] - 2,
                                                                           indices_of_state[key][1])
                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)
            break

        # self.select_correct_modes_for_array not needed for separate_states_n_zernike
        bottom_mode_for_array, top_mode_for_array = agent_values[0], agent_values[1]
        for key in indices_of_state:
            if key in "wfs":
                # FOR THE WFS WE INCLUDE ALL
                modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                       indices_of_state[key][1])
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            else:
                starting_point = indices_of_state[key][0]
                if seq2seq_partial > -1:
                    if bottom_mode_for_array - seq2seq_partial >= 0:
                        ini = bottom_mode_for_array - seq2seq_partial
                    else:
                        ini = bottom_mode_for_array
                    # max to not have negative values
                    # ini = max(0, ini)
                else:
                    ini = 0

                modes_chosen_current_worker_index_of_state = np.arange(starting_point + ini,
                                                                       starting_point + top_mode_for_array)

                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
        modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)

    if debug_modes_chosen:
        df = pd.DataFrame.from_dict(modes_chosen_dict, orient='index')
        df.transpose().to_csv("output/debug/" + experiment_name + "_modes_chosen_seq2seq.csv", index=False)

    return modes_chosen_dict


def modes_chosen_window_n_zernike(dictionary_agents,
                                  indices_of_state,
                                  window_n_zernike,
                                  include_tip_tilt,
                                  include_tip_tilt_windowed,
                                  n_filtered,
                                  experiment_name):
    """
    + Input state: for window_n_zernike state has shape (Commands x 4 ( 1 from d_err, 1 from C_t-1 state before
     linear and 2 from C_t-2 and C_t-3)
     ----------------------------------------------------------------------------------------------------------
    In this method state has the full shape of command
    To explain this method let's start with a self.dictionary_agents and a window.
    1. Let's say worker_id 1 controls modes from 40 to 80 and window_n_zernike is 20.
    + Then what window_n_zernike is telling the program is to add 20 more modes to its state
    + i.e. state = modes 20 to 100
    2. However, let's say worker_id 0 controls modes 0 to 40. There are no more modes below 0, then what we had done
    is to make it control modes 0 to 80 (adding 20 more modes ABOVE)
    The code above will only work if state_wfs is deactivated of course
    """
    modes_chosen_dict = {}
    for worker_id, agent_values in dictionary_agents.items():
        modes_chose_list_current_worker = []
        # I use len self.dictionary as worker id starts at 1,
        # e.g. 1 worker for modes and 1 for TT
        # TT id is 2 and len(self.dictionary_agents) = 2
        if include_tip_tilt and worker_id == len(dictionary_agents):
            for key in indices_of_state:
                if key in "wfs":
                    # FOR THE WFS WE INCLUDE ALL
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                           indices_of_state[key][1])
                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
                else:
                    modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][1] - 2,
                                                                           indices_of_state[key][1])

                    if include_tip_tilt_windowed:
                        window_for_tt = np.arange(0, int(2*window_n_zernike))
                        modes_chosen_current_worker_index_of_state =\
                            np.concatenate([modes_chosen_current_worker_index_of_state,
                                            window_for_tt])

                    modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)
            break

        # self.select_correct_modes_for_array not needed for separate_states_n_zernike
        bottom_mode_for_array, top_mode_for_array = agent_values[0], agent_values[1]
        for key in indices_of_state:
            if key in "wfs":
                # FOR THE WFS WE INCLUDE ALL
                modes_chosen_current_worker_index_of_state = np.arange(indices_of_state[key][0],
                                                                       indices_of_state[key][1])
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
            else:
                starting_point = indices_of_state[key][0]
                # We remove discarded modes and TT from end point.
                # Note self.n_filtered and not n_zernike_start_end[1] is used.
                end_point = indices_of_state[key][1] - (n_filtered - 2)
                len_indices = end_point - starting_point
                if bottom_mode_for_array - window_n_zernike < 0:
                    difference_bottom = bottom_mode_for_array - window_n_zernike
                    ini = 0
                    end = top_mode_for_array + window_n_zernike - difference_bottom
                elif (top_mode_for_array + window_n_zernike) > len_indices:
                    difference_top = top_mode_for_array + window_n_zernike - len_indices
                    ini = bottom_mode_for_array - int(window_n_zernike) - difference_top
                    end = len_indices
                else:
                    ini = bottom_mode_for_array - window_n_zernike
                    end = top_mode_for_array + window_n_zernike
                modes_chosen_current_worker_index_of_state = np.arange(starting_point + ini,
                                                                       starting_point + end)
                modes_chose_list_current_worker.append(modes_chosen_current_worker_index_of_state)
        modes_chosen_dict[worker_id] = np.concatenate(modes_chose_list_current_worker)

    if debug_modes_chosen:
        df = pd.DataFrame.from_dict(modes_chosen_dict, orient='index')
        df.transpose().to_csv("output/debug/" + experiment_name + "_modes_chosen_window_n_zernike.csv", index=False)

    return modes_chosen_dict


def get_modes_chosen(dictionary_agents,
                     indices_of_state,
                     config_rl,
                     n_filtered,
                     experiment_name,
                     total_existing_modes,
                     total_controlled_modes,
                     starting_mode):

    if config_rl.env_rl['window_n_zernike'] > -1:
        # When we use window_n_zernike the full state is in "state".
        divided_states = modes_chosen_window_n_zernike(dictionary_agents,
                                                       indices_of_state,
                                                       config_rl.env_rl['window_n_zernike'],
                                                       config_rl.env_rl['include_tip_tilt'],
                                                       config_rl.env_rl['include_tip_tilt_windowed'],
                                                       n_filtered,
                                                       experiment_name)
    else:

        if config_rl.env_rl['tt_treated_as_mode']:
            divided_states = modes_chosen_tt_treated_as_mode(dictionary_agents,
                                                             indices_of_state,
                                                             experiment_name)
            # TODO debug ??
        else:
            divided_states = modes_chosen_original(dictionary_agents, indices_of_state,
                                                   total_existing_modes, total_controlled_modes,
                                                   starting_mode, config_rl.env_rl['include_tip_tilt'],
                                                   experiment_name)

    return divided_states
