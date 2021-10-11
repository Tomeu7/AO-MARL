from src.reinforcement_learning.environment import ao_env
import os
import numpy as np
from src.reinforcement_learning.environment.delayed_mdp import DelayedMDP
import time
from src.reinforcement_learning.rpc_training.algorithms_rpc.replay_memory_rpc import \
            ReplayMemory
from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _call_method
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
import torch.distributed.rpc as rpc
from src.reinforcement_learning.rpc_training.helper_rpc.helper_rewards import get_separated_rewards
from src.reinforcement_learning.rpc_training.helper_rpc.helper_states import get_modes_chosen

"""
1. worker_id: int
+ id of worker starting from 1 to args.world_size

2. dictionary_agents: dict
+ dictionary of modes controlled per agent
+ key: worker_id
+ value: modes controlled
e.g. 8m 40x40 1281 modes, let say zernike_start_end 0 1200 and world_size 11, 10 workers, 120 modes per worker
+ {1: [0:120]}
+ {2: [120:240]}
+ NOTE: changes in the state via things like window_n_zernike will be taken into account in separate_state part

3. memorys_master: ReplayMemory
+ The memory of 1 episode saved by master
+ It is reset after each episode and only serves the purpose of sending it to the worker at the end of each episode

4. indices_of_state: dict
+ The environment returns an array for (c_t, C_t-1, C_t-2, ...)
+ Indices of state has the indices for each element in this array
e.g. 8m 40x40 n_zernike_start_end 0 1200 without window_n_zernike
+ {"state_dm_before_linear":0:1200}
+ {"state_dm_history_1":1200:2400}
+ {"state_dm_history_2":2400:3600}
+ {"state_d_err": 3600:4800}

5. self.modes_chosen: dict
+ dictionary that has keys: worker_id and values: elements of the state that are to be assigned to the RL agent

6. state_separated, reward_separated: dict, dict
+ each contains in key: worker_id and value states seen by that agent and reward seen by that agent

"""


class TrainerRPC:
    """
    Class that manages everything related to training of the AO RL agent
    """
    def __init__(self,
                 config_rl,
                 writer_performance,
                 writer_metrics_1,
                 experiment_name,
                 seed,
                 world_size,
                 num_gpus):

        # 0) a. RPC

        self.n_filtered = config_rl.env_rl['n_reverse_filtered_from_cmat']
        self.ag_rrefs = []
        self.master_rref = RRef(self)
        self.world_size = world_size
        self.num_gpus = num_gpus

        folder = "output/output_models/models_rpc/" + experiment_name + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1) Initializing AO env

        self.env = ao_env.AoEnv(config_rl=config_rl,
                                normalization_bool=True,
                                initial_seed=seed)

        if config_rl.env_rl['gain_change'] > -1:
            print("-CONFIG: Changing gain to:", config_rl.env_rl['gain_change'])
            self.env.supervisor.rtc.set_gain(0, config_rl.env_rl['gain_change'])

        config_rl.original_gain = self.env.supervisor.rtc._rtc.d_control[0].gain

        # Set environment seed given for the experiment

        self.env.set_sim_seed(seed)

        # 2) b. Choose RPC experiment

        self.dictionary_agents, self.total_controlled_modes, self.local_controlled_modes, self.starting_mode,\
            self.total_existing_modes = self.create_agents_dictionary(config_rl)

        self.indices_of_state = self.prepare_indices_of_state()

        self.modes_chosen = get_modes_chosen(self.dictionary_agents, self.indices_of_state, config_rl, self.n_filtered,
                                             experiment_name,
                                             self.total_existing_modes, self.total_controlled_modes, self.starting_mode)

        # 3) Loading SAC

        self.load_soft_actor_critic(config_rl)

        # 4) Loading Replay Memory

        self.memorys_master = {}
        for worker_id in range(1, world_size):
            self.memorys_master[worker_id] = ReplayMemory(config_rl.env_rl['max_steps_per_episode'])

        # 5) Load pretrained weights for SAC

        if config_rl.sac['pretrained_model_path'] is not None:
            raise NotImplementedError

        # Other initializations that are needed
        self.writer_performance, self.writer_metrics_1 = writer_performance, writer_metrics_1
        self.config_rl, self.experiment_name = config_rl, experiment_name
        self.initial_seed, self.seed = seed, seed
        self.len_actions = len(self.env.action_space.sample())
        self.total_update, self.total_step = 0, 0
        self.num_test_episode, self.num_episode = 0, 0
        self.delayed_mdp_object = None
        self.max_num_steps = 5e8
        self.save_networks_every_episodes = 500

        print("-----------------------------RPC TRAINING-----------------------------")

        print("\n Shape of state:", self.env.observation_space,
              "\n Original gain:", self.env.supervisor.rtc._rtc.d_control[0].gain,
              "\n Action shape:", self.env.action_space.sample().shape[0])

        print("Dictionary agents {} Total modes {} Local modes {} Total existing modes {}"
              .format(self.dictionary_agents, self.total_controlled_modes,
                      self.local_controlled_modes, self.total_existing_modes))

        self.current_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.current_action_divided = {}
        self.current_mu = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        self.current_r0 = self.env.supervisor.config.p_atmos.r0
        self.current_windspeed_layer_0 = self.env.supervisor.config.p_atmos.windspeed[0]

    def write_test_performances(self, rl_performance_dict, linear_performance_dict, geo_performance_dict):
        """
        agent_performance = {"r_per_agent_test": r_per_agent_test,
                             "r_total_test": r_total_test,
                             "sr_le_test": sr_le_test,
                             "sr_se_test": sr_se_test}

        geometric_performance = {"r_geo_per_agent_test": r_geo_per_agent_test,
                                 "r_geo_total_test": r_geo_total_test,
                                 "sr_le_geo_test": sr_le_geo_test}
        """

        if self.num_episode % 100 == 0:
            for i in range(len(rl_performance_dict['r_per_agent_test'])):
                worker_id = i + 1
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/RL_worker_" + str(worker_id),
                                                   rl_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/Linear_worker_" + str(worker_id),
                                                   linear_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/Error_worker_" + str(worker_id),
                                                   rl_performance_dict['r_per_agent_test'][i] -
                                                   linear_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)

                # Geometric
                if len(self.env.supervisor.config.p_controllers) > 1:
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/RL_worker_"
                                                       + str(worker_id),
                                                       rl_performance_dict['r_per_agent_test'][i],
                                                       self.total_step)
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/Geo_worker_"
                                                       + str(worker_id),
                                                       geo_performance_dict['r_geo_per_agent_test'][i],
                                                       self.total_step)
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/Error_geo_worker_"
                                                       + str(worker_id),
                                                       rl_performance_dict['r_per_agent_test'][i] -
                                                       geo_performance_dict['r_geo_per_agent_test'][i],
                                                       self.total_step)

            self.write_update_losses_for_each_agent()

        self.writer_performance.add_scalar("Evaluation_CR/RL_CR", rl_performance_dict['r_total_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_CR/Linear_CR', linear_performance_dict['r_total_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_CR/Error_CR', rl_performance_dict['r_total_test'] -
                                           linear_performance_dict['r_total_test'], self.total_step)

        self.writer_performance.add_scalar("Evaluation_Strehl_LE/RL_SR_LE", rl_performance_dict['sr_le_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_LE/Linear_SR_LE',
                                           linear_performance_dict['sr_le_test'], self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_LE/Error_SR_LE', rl_performance_dict['sr_le_test']
                                           - linear_performance_dict['sr_le_test'], self.total_step)

        # Geometric LE
        if len(self.env.supervisor.config.p_controllers) > 1:
            self.writer_performance.add_scalar("Evaluation_Strehl_LE/GEO_SR_LE", geo_performance_dict['sr_le_geo_test'],
                                               self.total_step)

        self.writer_performance.add_scalar("Evaluation_Strehl_SE/RL_SR_SE", rl_performance_dict['sr_se_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_SE/Linear_SR_SE',
                                           linear_performance_dict['sr_se_test'], self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_SE/Error_SR_SE', rl_performance_dict['sr_se_test'] -
                                           linear_performance_dict['sr_se_test'],
                                           self.total_step)

    def manage_saving_networks(self):

        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(SAC.save_model, ag_rreff,
                          self.experiment_name, self.num_episode, self.dictionary_agents[worker_id], worker_id)
                )
            )
            worker_id += 1

        for fut in futs:
            fut.wait()

    def create_agents_dictionary_tt_treated_as_mode(self, config_rl):

        total_existing_modes = self.env.supervisor.modes2volts.shape[1]
        total_controlled_modes =\
            config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0] + 2
        local_controlled_modes = int(total_controlled_modes / (self.world_size - 1))
        assert total_controlled_modes % (self.world_size - 1) == 0
        assert config_rl.env_rl['n_zernike_start_end'][0] > -1 and config_rl.env_rl['n_zernike_start_end'][1] > -1
        assert total_controlled_modes > 0
        assert config_rl.env_rl['window_n_zernike'] == -1

        dictionary_agents = {}
        worker_id = 1

        # 0-28 if 30 modes per agent
        modes = np.arange(config_rl.env_rl['n_zernike_start_end'][0],
                          config_rl.env_rl['n_zernike_start_end'][0] + local_controlled_modes - 2)
        # 2 last
        tt = np.arange(total_existing_modes - 2, total_existing_modes)

        dictionary_agents[worker_id] = np.concatenate([tt, modes])
        # agent 1 that has TT and initial modes
        # other agents will controll other
        worker_id = 2
        for modes in range(config_rl.env_rl['n_zernike_start_end'][0] + local_controlled_modes - 2,
                           config_rl.env_rl['n_zernike_start_end'][1],
                           local_controlled_modes):
            dictionary_agents[worker_id] = np.arange(modes, modes + local_controlled_modes)
            worker_id += 1

        return dictionary_agents, total_controlled_modes, local_controlled_modes,  total_existing_modes

    def create_agents_dictionary_original(self, config_rl):

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

        dictionary_agents = {}
        # id 0: Compass
        # worker_id 1, 2, 3, 4...: Agents
        worker_id = 1
        for modes in range(config_rl.env_rl['n_zernike_start_end'][0],
                           config_rl.env_rl['n_zernike_start_end'][1],
                           local_controlled_modes):
            dictionary_agents[worker_id] = [modes, modes + local_controlled_modes]

            worker_id += 1

        if config_rl.env_rl['include_tip_tilt']:
            dictionary_agents[worker_id] = [total_existing_modes - 2, total_existing_modes]
            total_controlled_modes += 2

        return dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes

    def create_agents_dictionary(self, config_rl):

        if config_rl.env_rl['tt_treated_as_mode']:
            dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes =\
                self.create_agents_dictionary_tt_treated_as_mode(config_rl)
        else:
            dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes =\
                self.create_agents_dictionary_original(config_rl)

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
                    state_shape = original_state_shape + int(2*config_rl.env_rl['window_n_zernike']) * state_multiplier
            else:
                state_shape = original_state_shape + int(
                    config_rl.env_rl['window_n_zernike'] * 2) * state_multiplier  # TODO generalize the 4
        else:
            state_shape = original_state_shape

        return state_shape

    def load_soft_actor_critic(self, config_rl):
        """
        Loads SAC given config_rl parameters
        """

        # We get state from the environment to calculate input to SAC

        for worker_id in range(1, self.world_size):
            agent_value = self.dictionary_agents[worker_id]

            state_shape = self.get_state_shape_worker(agent_value, config_rl, worker_id)

            print("Worker id {} state shape {}".format(worker_id, state_shape))

            action_shape = np.zeros([agent_value[1] - agent_value[0]])

            ag_info = rpc.get_worker_info("Agent{}".format(worker_id))

            self.ag_rrefs.append(remote(ag_info, SAC,
                                        kwargs={"num_inputs": state_shape,
                                                "action_space": action_shape,
                                                "config": config_rl,
                                                "rank": worker_id,
                                                "num_gpus": self.num_gpus
                                                }))

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

    def divide_rewards_for_agents_geometric(self, geometric_modes):
        """
        :geometric_modes: array of geometric_modes (N steps x modes)
        Creates a list that contains in an orderly manner the rewards for each agent for the geometric controller
        :return: reward_list
        """

        residuals = geometric_modes[1:, :]-geometric_modes[:-1, :]

        reward = np.sum(np.square(residuals), axis=0)

        if self.config_rl.env_rl['reward_type'] != "avg_squared_modes":
            factor = float(self.config_rl.env_rl['reward_type'].split("_")[-1])
            separated_reward = {worker_id: -factor*np.average(reward[agent_value[0]:agent_value[1]])
                                for (worker_id, agent_value) in self.dictionary_agents.items()}
        else:
            separated_reward = {worker_id: -np.sum(reward[agent_value[0]:agent_value[1]])
                                for (worker_id, agent_value) in self.dictionary_agents.items()}

        return list(separated_reward.values())

    def divide_rewards_for_agents(self):
        """
        Creates a list that contains in an orderly manner the rewards for each agent
        :return: reward_list
        """

        # Reward modes will come from residual commands, only RL can not use d_err, needs to use D m_t
        s_dm_residual_modes = self.env.supervisor.volts2modes.dot(self.env.supervisor.rtc.get_err(0))
        reward = np.square(s_dm_residual_modes)

        separated_reward = get_separated_rewards(reward,
                                                 self.config_rl.env_rl['reward_type'],
                                                 self.dictionary_agents)

        return separated_reward

    def divide_states_for_agents(self, state):
        """
        Creates a list that contains in an orderly manner the states for each agent
        :return: divided_state
        """
        divided_states = {}
        for worker_id in range(1, self.world_size):
            divided_states[worker_id] = state[self.modes_chosen[worker_id]]

        return divided_states

    def manage_changing_conditions(self):
        if self.config_rl.env_rl['change_atmospheric_3_layers_1'] and self.num_episode == 1000:
            # From wind direction 0 0 0 to 0 15 30
            self.env.supervisor.atmos.set_wind(screen_index=1, winddir=15)
            self.env.supervisor.atmos.set_wind(screen_index=2, winddir=30)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_2'] and self.num_episode == 1000:
            # From wind speed 2 to wind speed 1
            self.env.supervisor.atmos.set_wind(screen_index=0, windspeed=15)
            self.env.supervisor.atmos.set_wind(screen_index=1, windspeed=10)
            self.env.supervisor.atmos.set_wind(screen_index=2, windspeed=20)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_3'] and self.num_episode == 1000:
            # From r0 0.16 to 0.08
            self.env.supervisor.atmos.set_r0(0.08)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_4'] and self.num_episode == 1000:
            # From wind speed 2 to wind speed 1
            self.env.supervisor.atmos.set_wind(screen_index=0, windspeed=10)
            self.env.supervisor.atmos.set_wind(screen_index=1, windspeed=5)
            self.env.supervisor.atmos.set_wind(screen_index=2, windspeed=15)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_5'] and self.num_episode == 1000:
            # From wind direction 0 0 0 to 0 45 90
            self.env.supervisor.atmos.set_wind(screen_index=1, winddir=45)
            self.env.supervisor.atmos.set_wind(screen_index=2, winddir=90)

    def train_agent(self):
        """
        Method that manages the full training loop of the agent
        """

        while True:
            # An episode of the environment
            r_total = self.episode()

            if self.num_episode % 10 == 0:
                self.writer_performance.add_scalar("Training_Reward/Evolution of SR LE",
                                                    self.env.supervisor.target.get_strehl(0)[1], self.num_episode)
                self.writer_performance.add_scalar("Training_Reward/Average Reward of last 10 episodes", r_total,
                                                    self.num_episode)

            if (self.config_rl.env_rl['change_atmospheric_3_layers_1'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_2'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_3'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_4'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_5'])\
                    and self.num_episode >= 1000:
                if self.num_episode == 1000 \
                        or self.num_episode == 1001\
                        or self.num_episode == 1002:
                    num_test = 1
                elif 1002 < self.num_episode <= 1100:
                    num_test = 10
                else:
                    num_test = 50
            elif self.config_rl.env_rl['do_more_evaluations']:
                num_test = 5
            else:
                num_test = 50
            if self.num_episode % num_test == 0:
                self.seed += 1
                self.env.set_sim_seed(self.seed)
                rl_performance_dict, geo_performance_dict = self.test_episode(controller="RL")
                linear_performance_dict, _ = self.test_episode(controller="Integrator")
                self.write_test_performances(rl_performance_dict, linear_performance_dict, geo_performance_dict)

            if self.num_episode % self.save_networks_every_episodes == 0 and self.num_episode > 750:
                self.manage_saving_networks()

            self.seed += 1
            self.env.set_sim_seed(self.seed)

            self.manage_changing_conditions()

            if self.total_step > self.max_num_steps:
                break

    def episode(self):
        """
        Does an episode of the environment
        The episode definition depends on the configuration
        """

        step, r_total, done, s, start_time = 0, 0, False, self.env.reset(), time.time()

        self.delayed_mdp_object = DelayedMDP(self.config_rl.env_rl['delayed_assignment'],
                                             self.config_rl.env_rl['modification_online'])

        for _ in range(self.config_rl.env_rl['max_steps_per_episode']):

            # 0. Divided states for agents
            s_divided = self.divide_states_for_agents(s)

            # 1. Choose action based on state
            a, a_divided, mu = self.choose_action(s_divided)

            # 2. Step on the environment
            s_next, reward_divided, done = self.env_step(a)

            # 3. Divided states for agents
            s_next_divided = self.divide_states_for_agents(s_next)

            # 4. if the delayed_mdp is ready
            # Save on replay (s, a, r, s_next) which comes from delayed_mdp and the s_next and reward this timestep
            if self.delayed_mdp_object.check_update_possibility():
                self.manage_memory(reward_divided, done)

            # 5. Save s, a, s_next, r, a_next to do the correct credit assignment in replay memory later
            # We use this object because we have delay
            self.manage_delayed_mdp(s_divided, a_divided, s_next_divided)

            step += 1
            r_total += np.sum(list(reward_divided.values()))
            self.total_step += 1

            # 6. s = s_next
            s = s_next.copy()

        self.update_all_agents()

        print('Episode: {} \tTotal steps: {} \tEpisode steps: {} \tNum updates: {}'
              ' \tSeed: {} \tCurrent Reward: {:.4f} \tSR LE: {:.4f} \tTime {:.4f}'
              .format(self.num_episode, self.total_step, step, self.total_update, self.seed,
                      r_total, self.env.supervisor.target.get_strehl(0)[1], time.time()-start_time))

        self.num_episode += 1

        return r_total

    def test_episode(self, controller):
        """
        Does an episode of the environment
        The episode definition depends on the configuration
        """
        geometric_modes = np.zeros((self.config_rl.env_rl['max_steps_per_episode'],
                                    self.env.supervisor.volts2modes.shape[0]))
        r_per_agent_test, r_total_test, done, s = np.zeros(len(self.dictionary_agents)), 0, False, self.env.reset()

        sr_se_test_list = []
        for test_step in range(self.config_rl.env_rl['max_steps_per_episode']):

            # 0. Divided states for agents
            s_divided = self.divide_states_for_agents(s)

            # 1. Choose action based on state
            if controller == "RL":
                a, _, _ = self.choose_action(s_divided, eval_mode=True)
            else:
                a = None

            # 2. Step on the environment
            s_next, reward_divided, done =\
                self.env_step(a, linear_control=True if controller == "Integrator" else False)

            # Agent metrics
            r_total_test += np.sum(list(reward_divided.values()))
            r_per_agent_test += np.array(list(reward_divided.values()))
            sr_se_test = self.env.supervisor.target.get_strehl(0)[0]
            sr_se_test_list.append(sr_se_test)

            # Geometric save commands
            if len(self.env.supervisor.config.p_controllers) > 1:
                geometric_modes[test_step, :] =\
                    self.env.supervisor.volts2modes.dot(self.env.supervisor.rtc.get_command(1))

            # 3. s = s_next
            s = s_next.copy()

        sr_le_test = self.env.supervisor.target.get_strehl(0)[1]
        sr_se_test = np.average(sr_se_test_list)
        print('Test episode: {} \tSeed: {} \tCurrent Reward: {:.4f} \tSR LE: {:.4f} \tAvg SR SE: {:.4f}'
              .format(self.num_test_episode, self.seed, r_total_test, sr_le_test, sr_se_test))

        self.num_test_episode += 1

        agent_performance = {"r_per_agent_test": r_per_agent_test,
                             "r_total_test": r_total_test,
                             "sr_le_test": sr_le_test,
                             "sr_se_test": sr_se_test}

        if len(self.env.supervisor.config.p_controllers) > 1:
            # Geometric metrics, geometric index is 1
            r_geo_per_agent_test = self.divide_rewards_for_agents_geometric(geometric_modes=geometric_modes)
            r_geo_total_test = np.sum(r_geo_per_agent_test)
            sr_le_geo_test = self.env.supervisor.target.get_strehl(1)[1]

            geometric_performance = {"r_geo_per_agent_test": r_geo_per_agent_test,
                                     "r_geo_total_test": r_geo_total_test,
                                     "sr_le_geo_test": sr_le_geo_test}
        else:
            geometric_performance = None

        return agent_performance, geometric_performance

    def manage_delayed_mdp(self, s, a, s_next):
        """
        delayed_mdp object serves the purpose to remember s, a and s_next for when the correct r appears in the train
        loop and we can assign it correctly. This effect happens because we have delay in the system.
        e.g. delay 1
        s, a -> s', r'
        s', a' -> s'', r''
        s'', a'' -> s''', r'''
        The correct assignment would be (s, a, s'', r''') due to how the simulator works
        e.g. delay 0 -> (s, a, s', r'')
        """
        self.delayed_mdp_object.save(s, a, s_next)

    def env_step(self, a, return_dict=False, linear_control=False):
        """
        Does an step inside the environment
        a: action that will change the environment
        return_dict: if we want the next_state as an array or as a ordered_dict
        """
        if self.config_rl.env_rl['level'] == "only_rl" or self.config_rl.env_rl['level'] == "correction":

            _, done, info = self.env.rl_step(a, linear_control)
            r = self.divide_rewards_for_agents()

            s_next = self.env.linear_step(return_dict)
        else:
            raise NotImplementedError

        return s_next, r, done

    def select_correct_modes_for_array(self, agent_value_0, agent_value_1):
        # 1. As self.current_action = np.zeros(len_actions)
        # 2. And self.dictionary we have the current modes controlled e.g. 300 to 600
        # 3. From 300 and 600 we will remove 300 (self.starting_mode) to fit into the array
        # 4. Ending with bottom_mode 0 and top_mode 300
        # The same goes for the state when doing divide_states

        # First if is due to tip tilt
        if self.config_rl.env_rl['include_tip_tilt']\
                and agent_value_0 == (self.total_existing_modes-2) and agent_value_1 == self.total_existing_modes:
            bottom_mode = self.total_controlled_modes-2-self.starting_mode
            top_mode = self.total_controlled_modes-self.starting_mode
        else:
            bottom_mode = agent_value_0 - self.starting_mode
            top_mode = agent_value_1 - self.starting_mode
        return bottom_mode, top_mode

    def report_action(self, a, mu, worker_id):

        # TODO disable record mu?
        bottom_mode_for_array, top_mode_for_array =\
            self.select_correct_modes_for_array(
                self.dictionary_agents[worker_id][0], self.dictionary_agents[worker_id][1])
        self.current_action[bottom_mode_for_array:top_mode_for_array] = a
        self.current_mu[bottom_mode_for_array:top_mode_for_array] = mu
        self.current_action_divided[worker_id] = a

    def report_metrics(self, worker_id,
                       qf1_loss_list, qf2_loss_list, alpha_loss_list, alpha_tlogs_list, policy_loss_list):

        total_step_qf1, qf1_loss_value = qf1_loss_list
        total_step_qf2, qf2_loss_value = qf2_loss_list
        total_step_alpha, alpha_loss_value = alpha_loss_list
        total_step_alpha_tlogs, alpha_tlogs_value = alpha_tlogs_list
        total_step_policy, policy_loss_value = policy_loss_list

        self.writer_metrics_1.add_scalar("qf1_loss/" + str(worker_id), qf1_loss_value, total_step_qf1)
        self.writer_metrics_1.add_scalar("qf2_loss/" + str(worker_id), qf2_loss_value, total_step_qf2)
        self.writer_metrics_1.add_scalar("alpha_loss/" + str(worker_id), alpha_loss_value, total_step_alpha)
        self.writer_metrics_1.add_scalar("alpha_tlogs/" + str(worker_id), alpha_tlogs_value, total_step_alpha_tlogs)
        self.writer_metrics_1.add_scalar("policy_loss/" + str(worker_id), policy_loss_value, total_step_policy)

    def write_update_losses_for_each_agent(self):
        """
        Writing losses had to be syncronized otherwise it seemed not to work
        """
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_sync(
                    ag_rreff.owner(),
                    _call_method,
                    args=(SAC.master_ask_metrics, ag_rreff, self.master_rref)
                )
            )

    def choose_action(self, s, eval_mode=False):
        """
        Chooses SAC action
        Based on some config parameters the behaviour changes
        """

        self.current_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.current_action_divided = {}
        self.current_mu = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(SAC.master_ask_action, ag_rreff, self.master_rref, s[worker_id], worker_id, eval_mode)
                )
            )
            worker_id += 1

        for fut in futs:
            fut.wait()

        return self.current_action, self.current_action_divided, self.current_mu

    def manage_memory(self, reward_divided, done=False):
        """
        Pushes to memory and does updates. Some explanations below.
        ------------------------------------------------------------------------------------------------------------
        Following what is explained in manage_delayed_mdp
        manage_delayed_mdp has deques of len(delay+1)
        So once we have the reward, r, the input of this function we can extract the following:
        s = delayed_mdp.state_list[0], a = delayed_mdp.action_list[0], s' = delayed_mdp.next_state_list[-1]
        r = input
        This should work for all delays and the credit assignment should be ok
        -------------------------------------------------------------------------------------------------------------
        """

        # a) Credit assignment for different configs
        state_divided, action_divided, state_next_divided = self.delayed_mdp_object.credit_assignment()

        for worker_id in range(1, self.world_size):
            state = state_divided[worker_id]
            action = action_divided[worker_id]
            state_next = state_next_divided[worker_id]
            reward = reward_divided[worker_id]
            mask = float(not done)

            self.memorys_master[worker_id].push(state, action, reward, state_next, mask)

    def update_all_agents(self):

        worker_id = 1
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(SAC.update_parameters_sac,
                          ag_rreff,
                          self.memorys_master[worker_id],
                          self.config_rl.sac['batch_size'],
                          self.total_update,
                          self.total_step)
                )
            )
            worker_id += 1
        for fut in futs:
            fut.wait()
        self.total_update += self.config_rl.sac['updates_per_episode_rpc']
        for worker_id in range(1, self.world_size):
            self.memorys_master[worker_id].reset()


import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.reinforcement_learning.rpc_training.algorithms_rpc.utils import soft_update, hard_update
from src.reinforcement_learning.rpc_training.algorithms_rpc.model_rpc import GaussianPolicy, QNetwork
from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _remote_method


class SAC(object):
    def __init__(self,
                 num_inputs,
                 action_space,
                 config,
                 rank,
                 num_gpus):
        self.action_space = action_space.shape[0]
        self.state_space = num_inputs

        self.rpc_id = rpc.get_worker_info().id
        if num_gpus <= 0:
            device = "cpu"
            self.device = torch.device(device)
        else:
            device = (self.rpc_id - 1) % num_gpus
            self.device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

        self.worker_id = rank
        print("0. Worker id {} Rpc id {} Device {}".format(self.worker_id, self.rpc_id, device))
        self.write_update_statistics_every_updates = 50000  # 50000 updates approx 50 episodes
        self.config = config

        self.sac_writing_critic = 0

        self.gamma = config.sac['gamma']
        self.tau = config.sac['tau']
        self.alpha = config.sac['alpha']

        self.policy_type = config.sac['policy']
        self.target_update_interval = config.sac['target_update_interval']
        self.automatic_entropy_tuning = config.sac['automatic_entropy_tuning']

        self.initialize_last_layer_zero = config.sac['initialize_last_layer_0']
        self.initialize_last_layer_near_zero = config.sac['initialize_last_layer_near_0']

        self.lr = config.sac['lr']
        hidden_size_critic = config.sac['hidden_size_critic']
        num_layers_critic = config.sac['num_layers_critic']
        hidden_size_actor = config.sac['hidden_size_actor']
        num_layers_actor = config.sac['num_layers_actor']

        self.total_update = 0

        self.critic, self.critic_target, self.critic_optim = \
            self.initialise_critic(num_inputs=num_inputs,
                                   action_space=action_space,
                                   hidden_size_critic=hidden_size_critic,
                                   num_layers_critic=num_layers_critic)

        self.policy, self.policy_optim = self.initialise_policy(num_inputs=num_inputs,
                                                                action_space=action_space,
                                                                hidden_size_actor=hidden_size_actor,
                                                                num_layers_actor=num_layers_actor)

        self.log_alpha, self.alpha_optim, self.target_entropy = self.initialise_alpha(action_space)

        self.memory = ReplayMemory(config.sac['memory_size'])

        self.qf1_loss_list = None
        self.qf2_loss_list = None
        self.alpha_loss_value_list = None
        self.alpha_tlogs_value_list = None
        self.policy_loss_value_list = None

    def initialise_alpha(self, action_space):
        print("3. Initialasing SAC Alpha")
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        target_entropy = None
        log_alpha = None
        alpha_optim = None

        if self.automatic_entropy_tuning is True:
            # TODO Changed from .Tensor to .tensor
            target_entropy = -torch.prod(torch.tensor(action_space.shape).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha_optim = Adam([log_alpha], lr=self.lr)

        elif self.policy_type == "Deterministic":
            self.alpha = 0
            self.automatic_entropy_tuning = False

        return log_alpha, alpha_optim, target_entropy

    def initialise_critic(self, num_inputs, action_space, hidden_size_critic, num_layers_critic):

        print("1. Initialasing SAC Critic")

        critic = QNetwork(num_inputs, action_space.shape[0], hidden_size_critic, num_layers_critic).to(self.device)
        critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size_critic, num_layers_critic).to(
            self.device)
        critic_optim = Adam(critic.parameters(), lr=self.lr)
        hard_update(critic_target, critic)

        return critic, critic_target, critic_optim

    def initialise_policy(self,
                          num_inputs,
                          action_space,
                          hidden_size_actor,
                          num_layers_actor
                          ):

        print("2. Initialising Policy; Type:", self.policy_type)

        if self.policy_type == "Gaussian":
            policy = GaussianPolicy(num_inputs=num_inputs,
                                    num_actions=action_space.shape[0],
                                    hidden_dim=hidden_size_actor,
                                    action_scale=self.config.sac['gaussian_std'],
                                    action_bias=self.config.sac['gaussian_mu'],
                                    num_layers=num_layers_actor,
                                    initialize_last_layer_zero=self.initialize_last_layer_zero,
                                    initialize_last_layer_near_zero=self.initialize_last_layer_near_zero,
                                    activation=self.config.sac['activation'],
                                    LOG_SIG_MAX=self.config.sac['LOG_SIG_MAX']).to(self.device)
            policy_optim_decay = self.config.sac['l2_norm_policy'] if self.config.sac['l2_norm_policy'] > 0 else 0
            policy_optim = Adam(policy.parameters(), lr=self.lr, weight_decay=policy_optim_decay)

        else:
            raise NotImplementedError

        return policy, policy_optim

    def reset_optimizers(self):
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.memory.reset()

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    # noinspection PyArgumentList
    def select_action(self,
                      state,
                      eval_mode=False,
                      only_choosing_action=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action, _, mean = self.policy.sample(state,
                                             only_choosing_action=only_choosing_action)

        mean_to_return = mean.detach().cpu().numpy()[0]
        if eval_mode is False:
            action_to_return = action.detach().cpu().numpy()[0]
        else:
            action_to_return = mean_to_return

        return action_to_return, mean_to_return

    def master_ask_action(self, master_rref, state, worker_id, eval_mode):
        """
        When master asks for an action
        """
        assert self.worker_id == worker_id
        with torch.no_grad():
            a, mu = self.select_action(state=state,
                                       eval_mode=eval_mode,
                                       only_choosing_action=True)

        _remote_method(TrainerRPC.report_action, master_rref, a, mu, self.worker_id)

    def master_ask_metrics(self, master_rref):
        _remote_method(TrainerRPC.report_metrics, master_rref, self.worker_id,
                       self.qf1_loss_list,
                       self.qf2_loss_list,
                       self.alpha_loss_value_list,
                       self.alpha_tlogs_value_list,
                       self.policy_loss_value_list)
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    # --SAC UPDATE--

    # This is done to remove warning of torch.FloatTensor()
    # noinspection PyArgumentList
    def get_tensors_from_memory(self, memory, batch_size):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def get_bellman_backup(self,
                           reward_batch,
                           next_state_batch,
                           mask_batch
                           ):
        with torch.no_grad():

            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target,
                                           qf2_next_target) - self.alpha * next_state_log_pi

            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        return next_q_value

    @staticmethod
    def calculate_q_loss(qf1, qf2, next_q_value):

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)

        qf_loss = qf1_loss + qf2_loss

        return qf_loss, qf1_loss, qf2_loss

    def update_critic(self, state_batch,
                      action_batch,
                      reward_batch,
                      next_state_batch,
                      mask_batch):

        next_q_value = self.get_bellman_backup(reward_batch,
                                               next_state_batch,
                                               mask_batch
                                               )
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)

        qf_loss, qf1_loss, qf2_loss = self.calculate_q_loss(qf1,
                                                            qf2,
                                                            next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        return qf1_loss.detach().item(), qf2_loss.detach().item()

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def calculate_policy_loss(self, log_pi, min_qf_pi):

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        return policy_loss

    def update_actor(self, state_batch):
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = self.calculate_policy_loss(log_pi, min_qf_pi)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return log_pi, policy_loss.detach().item()

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def update_alpha(self, log_pi):
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha).item()  # For TensorboardX logs

        return alpha_loss.detach().item(), alpha_tlogs

    def update_parameters_sac(self,
                              master_memory,
                              batch_size,
                              updates,
                              total_step):
        """
        memory: memory buffer to sample transitions (s,a,s',r) or (s,a,s',r,linear)
        batch_size: int, size of batch to update SAC
        updates: int, number of updates that have been done until now
        linear_warmup_only_rl: if we are doing warm up for only RL
        time_to_update_actor: when to update the actor if we do more updates for critic than actor
        writer: only valid when we are using beta parameter as it will write when the agent activates
        """
        self.total_update = updates

        master_memory_idx = 0
        for _ in range(self.config.sac['updates_per_episode_rpc']):

            if master_memory_idx < len(master_memory):
                state_master, action_master, reward_master, next_state_master, mask_master =\
                    master_memory.buffer[master_memory_idx]

                self.memory.push(state_master, action_master, reward_master, next_state_master, mask_master)

                master_memory_idx += 1

            if len(self.memory) > batch_size:
                # Sample a batch from memory
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch\
                    = self.get_tensors_from_memory(self.memory, batch_size)

                qf1_loss_value, qf2_loss_value = self.update_critic(state_batch,
                                                                    action_batch,
                                                                    reward_batch,
                                                                    next_state_batch,
                                                                    mask_batch)

                log_pi, policy_loss_value = self.update_actor(state_batch=state_batch)

                alpha_loss_value, alpha_tlogs_value = self.update_alpha(log_pi)

                if updates % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)

        if len(self.memory) > batch_size and total_step % (10*self.config.sac['updates_per_episode_rpc']) == 0:
            self.qf1_loss_list = [total_step, qf1_loss_value]
            self.qf2_loss_list = [total_step, qf2_loss_value]
            self.alpha_loss_value_list = [total_step, alpha_loss_value]
            self.alpha_tlogs_value_list = [total_step, alpha_tlogs_value.item()]
            self.policy_loss_value_list = [total_step, policy_loss_value]

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    # Save model parameters

    def save_model(self, experiment_name, episode, modes_controlled, worker_id):
        assert worker_id == self.worker_id

        folder = "output/output_models/models_rpc/" + experiment_name + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        actor_path = folder + experiment_name + "experiment_name_worker_{}_sac_actor_{}_episode_{}"\
            .format(str(worker_id), experiment_name, str(episode))
        critic_path = folder + experiment_name + "_worker_{}_sac_critic_{}_episode_{}"\
            .format(str(worker_id), experiment_name, str(episode))

        print('Saving actor to {}'.format(actor_path))
        print('Saving critic to {}'.format(critic_path))

        torch.save({
            'worker_id': worker_id,
            'models_controlled': modes_controlled,
            'model_state_dict': self.policy.state_dict()
        }, actor_path)

        torch.save(self.critic.state_dict(), critic_path)
