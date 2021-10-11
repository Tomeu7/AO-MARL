from collections import deque
import numpy as np


class DelayedMDP:
    def __init__(self, delay, modification):
        self.delay = delay
        self.not_modification = int(not modification)

        # for reward assignment
        self.action_list = deque(maxlen=delay+int(not modification))
        self.state_list = deque(maxlen=delay+int(not modification))
        self.next_state_list = deque(maxlen=delay+int(not modification))
        self.next_action_list = deque(maxlen=delay+int(not modification))
        # Reward list only used for model or for sum_rewards
        self.reward_list = deque(maxlen=delay+int(not modification))

    def check_update_possibility(self):
        """
        Checks that action list (and all the lists by the same rule)
        have enough information to take into account the delay
        """
        return len(self.action_list) >= self.delay + self.not_modification

    def save_agent_deactivated(self, s, a, s_next, a_next):
        self.action_list.append(a)
        self.state_list.append(s)
        self.next_action_list.append(a_next)
        self.next_state_list.append(s_next)

    def credit_assignment_agent_deactivated(self):

        return self.state_list[0], self.action_list[0], self.next_state_list[0], self.next_action_list[0]

    def save(self, s, a, s_next, r=None):

        self.action_list.append(a)
        self.state_list.append(s)
        self.next_state_list.append(s_next)
        if r is not None:
            self.reward_list.append(r)

    def obtain_rewards(self):
        return sum(self.reward_list)

    def credit_assignment(self):
        return self.state_list[0], self.action_list[0], self.next_state_list[-1]

    def model_information(self):
        """
        a) Suppose delay 1
        b) Suppose C*_t-1 (dm_before_linear), C_t (dm_after_linear) in state (for now I discard W_t (wfs) and previous dm and wfs)
        c) To predict C_t+1 I need C*_t-1 and C_t
        Hence I need to return self.state_list[0] and self.next_state_list[0]
        WAIT! BUT WHAT ABOUT THE REWARD
        """

        return self.state_list[0], self.next_state_list[0], self.reward_list[-1]
