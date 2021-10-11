from src.reinforcement_learning.environment import ao_env
from src.reinforcement_learning.config.GlobalConfig import Config
import numpy as np
from src.error_budget.sac.sac import SAC
from src.reinforcement_learning.rpc_training.helper_rpc.helper_states import get_modes_chosen
from src.error_budget.helper_experiments import choose_experiment_error_budget_multiple_agents


def obtain_config(parameter_file,
                  n_zernike_start,
                  n_zernike_end,
                  experiment_name,
                  modes_filtered):
    """
    """
    config = Config()
    config.strings_to_bools()

    config.cuda = "False"
    config.env_rl['basis'] = "zernike_space"
    config.env_rl['parameters_telescope'] = parameter_file
    config.env_rl['n_reverse_filtered_from_cmat'] = modes_filtered

    config.env_rl['state_dm_after_linear'] = False
    config.env_rl['state_dm_residual'] = True
    config.env_rl['state_wfs'] = False
    config.env_rl['state_dm_before_linear'] = True
    config.env_rl['state_gain'] = False

    config.env_rl['number_of_previous_dm'] = 2
    config.env_rl['number_of_previous_wfs'] = 0
    config.env_rl['reward_type'] = "avg_square_m"

    config.env_rl['n_zernike_start_end'] = [n_zernike_start, n_zernike_end]
    config.env_rl['include_tip_tilt'] = True
    config.env_rl['max_steps_per_episode'] = 1000
    config.env_rl['level'] = "correction"
    if "window" in experiment_name or "w20" in experiment_name:
        config.env_rl['window_n_zernike'] = 20
        config.env_rl['include_tip_tilt_windowed'] = True

    return config


class RlErrorBudgetTester:
    def __init__(self,
                 world_size,
                 n_zernike_start,
                 n_zernike_end,
                 parameter_file,
                 agents_path,
                 modes_filtered,
                 experiment_name
                 ):
        ######################################### RPC ###################################################
        self.save_cmat = False
        self.world_size = world_size
        self.n_filtered = modes_filtered
        ##########################################################################################
        self.total_num_steps = 20000
        self.N_preloop = 1000
        self.linear = True if agents_path[1] is None else False
        self.parameter_file_name = parameter_file[:-3]

        self.config_rl = obtain_config(parameter_file,
                                       n_zernike_start,
                                       n_zernike_end,
                                       experiment_name=experiment_name,
                                       modes_filtered=modes_filtered)
        
        self.experiment_name = experiment_name
        self.config_rl.env_rl['error_budget_experiment'] = experiment_name

        if self.linear:
            normalization_bool = False
        else:
            normalization_bool = True
        self.env = ao_env.AoEnv(self.config_rl,
                                normalization_bool=normalization_bool,
                                geo_policy_testing=False,
                                roket=True)

        if self.linear:
            self.dictionary_agents, self.total_controlled_modes, self.local_controlled_modes, self.starting_mode, \
                self.total_existing_modes = None, None, None, None, None
        else:
            self.dictionary_agents, self.total_controlled_modes, self.local_controlled_modes, self.starting_mode, \
                self.total_existing_modes = self.create_agents_dictionary(self.config_rl)

        self.env.supervisor.obtain_and_set_cmat_filtered(modes_filtered=modes_filtered)
        self.v2m = self.env.supervisor.volts2modes
        self.m2v = self.env.supervisor.modes2volts
        self.len_actions = len(self.env.action_space.sample())

        self.indices_of_state = self.prepare_indices_of_state()

        if self.linear:
            self.modes_chosen = None
        else:
            self.modes_chosen = get_modes_chosen(self.dictionary_agents, self.indices_of_state, self.config_rl,
                                                 self.n_filtered,
                                                 experiment_name,
                                                 self.total_existing_modes, self.total_controlled_modes,
                                                 self.starting_mode)

        if self.linear:
            self.agent_dict = None
        else:
            self.agent_dict = self.load_soft_actor_critic(config_rl=self.config_rl, agent_paths=agents_path)

        self.env.supervisor.init_config_roket(N_total=self.total_num_steps,
                                              N_preloop=self.N_preloop,
                                              agent=self.agent_dict,
                                              nfiltered=self.n_filtered,
                                              include_tip_tilt=self.config_rl.env_rl['include_tip_tilt'],
                                              n_zernike_start=n_zernike_start,
                                              n_zernike_end=n_zernike_end
                                              )

        print("-----------------------------ROKET TESTING-----------------------------")
        print("\n Shape of state:", self.env.observation_space,
              "\n Original gain:", self.env.supervisor.rtc._rtc.d_control[0].gain,
              "\n Action shape:", self.env.action_space.sample().shape[0])

        print("Dictionary agents {} Total modes {} Local modes {} Total existing modes {}"
              .format(self.dictionary_agents, self.total_controlled_modes,
                      self.local_controlled_modes, self.total_existing_modes))

    def prepare_indices_of_state(self):
        """
        Prepares the dictionary self.indices_of_state.
        For each key in the state usually e.g. (s_wfs, s_dm_before_linear, s_dm_after_linear, ...)
        We have a list of two elements:
        · The first element is the starting point of that part of the state
        · The second element is the ending por of that part of the state
        """

        s = self.env.reset(return_dict=True)
        indices_of_state = dict()
        initial_index = 0
        for key in s.keys():
            indices_of_state[key] = [initial_index, initial_index + s[key].shape[0]]
            initial_index += s[key].shape[0]

        return indices_of_state

    def create_agents_dictionary(self, config_rl):

        total_existing_modes = self.env.supervisor.modes2volts.shape[1]
        total_controlled_modes = config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0]
        if config_rl.env_rl['include_tip_tilt']:
            # 1 worker will be TT, the other local controlled modes
            local_controlled_modes = int(total_controlled_modes / (self.world_size - 2))
            assert total_controlled_modes % (self.world_size - 2) == 0
        else:
            # all workers will be local controlled modes
            local_controlled_modes = int(total_controlled_modes / (self.world_size - 1))
            assert total_controlled_modes % (self.world_size - 1) == 0
        # assert we have zernike start end
        assert config_rl.env_rl['n_zernike_start_end'][0] > -1 and config_rl.env_rl['n_zernike_start_end'][1] > -1
        assert total_controlled_modes > 0
        print("local controlled modes, world size, total_controlled_modes",
              local_controlled_modes, self.world_size, total_controlled_modes)
        dictionary_agents = {}
        # id 0: Compass
        # worker_id 1, 2, 3, 4...: Agents
        worker_id = 1
        for modes in range(config_rl.env_rl['n_zernike_start_end'][0],
                           config_rl.env_rl['n_zernike_start_end'][1],
                           local_controlled_modes):
            dictionary_agents[worker_id] = [modes, modes+local_controlled_modes]
            worker_id += 1

        if config_rl.env_rl['include_tip_tilt']:
            dictionary_agents[worker_id] = [total_existing_modes-2, total_existing_modes]
            total_controlled_modes += 2
        print("dictionary agents", dictionary_agents)
        return dictionary_agents, total_controlled_modes, local_controlled_modes,\
               config_rl.env_rl['n_zernike_start_end'][0], total_existing_modes

    def get_state_shape_worker(self, agent_value, config_rl, worker_id):
        """
        Gets state shape for the current worker_id
        """
        state_multiplier = (int(config_rl.env_rl['state_dm_residual']) +
                            int(config_rl.env_rl['state_dm_after_linear']) +
                            int(config_rl.env_rl['state_dm_before_linear']) +
                            config_rl.env_rl['number_of_previous_dm'] +
                            config_rl.env_rl['number_of_previous_dm_residuals'])

        original_state_shape = (agent_value[1] - agent_value[0]) * state_multiplier

        if config_rl.env_rl['window_n_zernike'] > -1:
            # worker_id == len(self.dictionary_agents) implies TT
            if config_rl.env_rl['include_tip_tilt'] and worker_id == len(self.dictionary_agents):
                state_shape = original_state_shape
                if config_rl.env_rl['include_tip_tilt_windowed']:
                    state_shape = original_state_shape + int(
                        2 * config_rl.env_rl['window_n_zernike']) * state_multiplier
            else:
                state_shape = original_state_shape + int(
                    config_rl.env_rl['window_n_zernike'] * 2) * state_multiplier  # TODO generalize the 4
        else:
            state_shape = original_state_shape

        return state_shape

    def load_soft_actor_critic(self, config_rl, agent_paths):
        """
        Loads SAC given config_rl parameters
        """

        agents_list = {}

        for worker_id in range(1, self.world_size):
            agent_value = self.dictionary_agents[worker_id]

            state_shape = self.get_state_shape_worker(agent_value, config_rl, worker_id)

            action_shape = np.zeros([agent_value[1] - agent_value[0]])

            agent = SAC(num_inputs=state_shape,
                        action_space=action_shape,
                        config=config_rl,
                        indices_of_state=None)

            fd = agent_paths[worker_id]
            if agent_paths[worker_id] is not None:
                print("Worker id", worker_id, "agent_value", agent_value)
                agent.load_model(fd, None, map_location='cuda:0')
                agent.policy.eval()

            agents_list[worker_id] = agent

        return agents_list

    def divide_states_for_agents(self, state):
        """
        Creates a list that contains in an orderly manner the states for each agent
        :return: divided_state
        """
        divided_states = {}
        for worker_id in range(1, self.world_size):
            divided_states[worker_id] = state[self.modes_chosen[worker_id]]

        return divided_states

    def select_correct_modes_for_array(self, agent_value_0, agent_value_1):
        # 1. As self.current_action = np.zeros(len_actions)
        # 2. And self.dictionary we have the current modes controlled e.g. 300 to 600
        # 3. From 300 and 600 we will remove 300 (self.starting_mode) to fit into the array
        # 4. Ending with bottom_mode 0 and top_mode 300
        # The same goes for the state when doing divide_states

        # First if is due to tip tilt
        if self.config_rl.env_rl['include_tip_tilt'] \
                and agent_value_0 == (self.total_existing_modes - 2) and agent_value_1 == self.total_existing_modes:
            bottom_mode = self.total_controlled_modes - 2 - self.starting_mode
            top_mode = self.total_controlled_modes - self.starting_mode
        else:
            bottom_mode = agent_value_0 - self.starting_mode
            top_mode = agent_value_1 - self.starting_mode
        return bottom_mode, top_mode

    def choose_action_testing(self, s):
        """
        Chooses SAC action
        Based on some config parameters the behaviour changes
        """
        current_action = np.zeros(self.env.action_space.shape[0])
        std_per_agent = {}
        for worker_id, agent in self.agent_dict.items():
            a, std = agent.select_action(s[worker_id],
                                         eval_mode=True,
                                         return_std=True)
            bottom_mode_for_array, top_mode_for_array = \
                self.select_correct_modes_for_array(self.dictionary_agents[worker_id][0],
                                                    self.dictionary_agents[worker_id][1])
            current_action[bottom_mode_for_array:top_mode_for_array] = a
            std_per_agent[worker_id] = std

        return current_action, std_per_agent

    def test_rl_agent_performance(self):
        """

        :param num_episodes:
        :return:
        """

        seed = 200  # Seed 0-20 for preprocessing, 200 for error budget
        self.env.set_sim_seed(seed)

        s = self.env.reset(geometric_do_control=True,
                           normalization_loop=False,
                           return_dict=False)
        std_dic = {}
        for worker_id in range(self.world_size):
            std_dic[worker_id] = []

        print("V2M shape", self.env.supervisor.volts2modes.shape,
              "M2V shape", self.env.supervisor.modes2volts.shape)
        print("-----------------------------------------------------------------")
        print("iter# | SE SR | LE SR | Temp | Noise | Tomo | Filtered | Non-linear")
        print("-----------------------------------------------------------------")
        for step in range(self.total_num_steps):

            if self.linear:
                a = np.zeros(self.len_actions)
            else:
                s_divided = self.divide_states_for_agents(s)
                a, std_per_agent = self.choose_action_testing(s_divided)
                for worker_id in range(1, self.world_size):
                    std_dic[worker_id].append(std_per_agent[worker_id])

            self.env.rl_step(action=a,
                             linear_control=self.linear,
                             geometric_do_control=True,
                             apply_control=False,
                             compute_tar_psf=False
                             )

            self.env.supervisor.do_error_breakdown(a)

            s_next = self.env.linear_step(return_dict=False)

            s = s_next.copy()
            if (step+1) % 100 == 0:
                sr_se, sr_le, _, _ = self.env.supervisor.target.get_strehl(0)
                print("%d \t %.4f \t  %.4f\t" %
                      (step + 1,
                       sr_se,
                       round(sr_le, 5)
                       ))
            if (step+1) == self.N_preloop:
                self.env.supervisor.target.reset_strehl(0)

        srs = self.env.supervisor.target.get_strehl(0)
        self.env.supervisor.SR2 = np.exp(srs[3])
        self.env.supervisor.SR = srs[1]
        self.env.supervisor.save_in_hdf5("output/error_budget/h5_files/" + self.experiment_name +
                                         "_steps_" + str(self.total_num_steps) + ".h5")
        return self.env.supervisor.target.get_strehl(0)[1]


class ExperimentManager:
    def __init__(self):
        self.number_episodes = 1
        self.experiment_name = "14_8_3l_dir_w20"
        assert self.experiment_name in ["14_8_3l_dir_w20"]

    def run(self):
        policy_names, par_file_list, policy_paths, modes_filtered_list, n_zernike_start, n_zernike_end = \
            choose_experiment_error_budget_multiple_agents(self.experiment_name)

        modes_filtered = modes_filtered_list[0]
        par_file = par_file_list[0]

        rl_perf_obtainer = RlErrorBudgetTester(world_size=len(policy_names)+1,
                                               n_zernike_start=n_zernike_start,
                                               n_zernike_end=n_zernike_end,
                                               parameter_file=par_file,
                                               agents_path=policy_paths,
                                               modes_filtered=modes_filtered,
                                               experiment_name=self.experiment_name)

        rl_perf_obtainer.test_rl_agent_performance()


if __name__ == "__main__":
    exp = ExperimentManager()
    exp.run()
