import numpy as np
import os
import matplotlib.pyplot as plt
"""
Python file related to rewards in RPC loop
linear_rewards: list
+ Reward of -c_t^2 for integrator for modes [0:last]
"""

#####################################################################################################################



def get_separated_rewards(reward,
                          reward_type,
                          dictionary_agents):
    # TODO else
    factor = float(reward_type.split("_")[-1])
    separated_reward = {worker_id: -factor * np.average(reward[agent_value[0]:agent_value[1]])
                        for (worker_id, agent_value) in dictionary_agents.items()}

    return separated_reward

#####################################################################################################################