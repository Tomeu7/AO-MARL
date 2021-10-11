import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, level, use_cnn):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.level = level
        self.use_cnn = use_cnn

    def push(self, state, action, reward, next_state, done, linear_command=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        if self.level == "only_rl":
            if linear_command is None:
                linear_command = np.zeros(action.shape)
            # self.buffer[self.position] = (state, action, reward, next_state, done, linear_command)
            if self.use_cnn:
                self.buffer[self.position] =\
                    (state['state_wfs'], state['state_dm'], action, reward,
                     next_state['state_wfs'], next_state['state_dm'], done, linear_command)
            else:
                self.buffer[self.position] = (state, action, reward, next_state, done, linear_command)
        else:
            if self.use_cnn:
                self.buffer[self.position] =\
                    (state['state_wfs'], state['state_dm'], action, reward,
                     next_state['state_wfs'], next_state['state_dm'], done)
            else:
                self.buffer[self.position] = (state, action, reward, next_state, done)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        if self.level == "only_rl":
            if self.use_cnn:
                state_wfs, state_dm, action, reward, next_state_wfs, next_state_dm, done, linear_action =\
                    map(np.stack, zip(*batch))
                return state_wfs, state_dm, action, reward, next_state_wfs, next_state_dm, done, linear_action
            else:
                state, action, reward, next_state, done, linear_action = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done, linear_action
        else:
            if self.use_cnn:
                state_wfs, state_dm, action, reward, next_state_wfs, next_state_dm, done = map(np.stack, zip(*batch))
                return state_wfs, state_dm, action, reward, next_state_wfs, next_state_dm, done
            else:
                state, action, reward, next_state, done = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.position = 0

    def load_replay(self, path):
        self.buffer = []
        self.position = 0
        print("Loading replay from: /gpfs/scratch/bsc28/bsc28921/outputs/" + path + ".npz")
        replay = np.load("/gpfs/scratch/bsc28/bsc28921/outputs/" + path + ".npz")
        states, actions, next_states, rewards = replay['state'],\
                                                replay['action'],\
                                                replay['next_state'],\
                                                replay['reward']
        done = False
        for idx in range(states.shape[0]):
            # the Replay Memory for SAC uses not done so the last item should be true
            self.push(states[idx], actions[idx], rewards[idx], next_states[idx], float(not done))
