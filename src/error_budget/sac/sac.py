import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.error_budget.sac.utils import soft_update, hard_update
from src.error_budget.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np


class SAC(object):
    def __init__(self,
                 num_inputs,
                 action_space,
                 config,
                 indices_of_state
                 ):

        self.config = config

        self.indices_of_state = indices_of_state

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

        self.device = torch.device("cuda" if config.cuda else "cpu")

        self.critic, self.critic_target, self.critic_optim = \
            self.initialise_critic(num_inputs=num_inputs,
                                   action_space=action_space,
                                   hidden_size_critic=hidden_size_critic,
                                   num_layers_critic=num_layers_critic)

        self.policy, self.policy_optim = \
            self.initialise_policy(num_inputs=num_inputs, action_space=action_space,
                                   hidden_size_actor=hidden_size_actor,
                                   num_layers_actor=num_layers_actor)

        self.log_alpha, self.alpha_optim, self.target_entropy = self.initialise_alpha(action_space)

    def initialise_alpha(self, action_space):
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
        print("1. Critic")
        critic = QNetwork(num_inputs, action_space.shape[0], hidden_size_critic, num_layers_critic).to(
            device=self.device)
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

        if self.policy_type == "Gaussian":
            print("Policy type:", "Normal Gauss")
            policy = GaussianPolicy(num_inputs=num_inputs,
                                    num_actions=action_space.shape[0],
                                    hidden_dim=hidden_size_actor,
                                    action_scale=self.config.sac['gaussian_std'],
                                    action_bias=self.config.sac['gaussian_mu'],
                                    num_layers=num_layers_actor,
                                    initialize_last_layer_zero=self.initialize_last_layer_zero,
                                    initialize_last_layer_near_zero=self.initialize_last_layer_near_zero,
                                    no_squashing=False,
                                    output_normalization=False,
                                    activation=self.config.sac['activation']).to(self.device)
            policy_optim_decay = self.config.sac['l2_norm_policy'] if self.config.sac['l2_norm_policy'] > 0 else 0
            policy_optim = Adam(policy.parameters(), lr=self.lr, weight_decay=policy_optim_decay)

        elif self.policy_type == "Deterministic":
            print("Policy type:", "Deterministic")

            policy = DeterministicPolicy(num_inputs=num_inputs,
                                         num_actions=action_space.shape[0],
                                         hidden_dim=hidden_size_actor,
                                         action_scale=self.config.sac['gaussian_std'],
                                         action_bias=self.config.sac['gaussian_mu'],
                                         num_layers=num_layers_actor,
                                         no_squashing=False,
                                         initialize_last_layer_zero=self.initialize_last_layer_zero,
                                         initialize_last_layer_near_zero=self.initialize_last_layer_near_zero,
                                         output_normalization=False,
                                         deterministic_std_noise=self.config.sac['deterministic_std_noise']
                                         ).to(self.device)
            policy_optim_decay = self.config.sac['l2_norm_policy'] if self.config.sac['l2_norm_policy'] > 0 else 0
            policy_optim = Adam(self.policy.parameters(), lr=self.lr, weight_decay=policy_optim_decay)
        else:
            raise NotImplementedError
        print("Final Policy type:", self.policy_type)

        return policy, policy_optim

    def reset_optimizers(self):
        policy_optim_decay = self.config.sac['l2_norm_policy'] if self.config.sac['l2_norm_policy'] > 0 else 0
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr / 5.0, weight_decay=policy_optim_decay)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr / 5.0)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    def select_action_single_agent(self,
                                   state,
                                   eval_mode,
                                   return_std,
                                   return_entropy):
        std, entropy, logpi = None, None, None
        if eval_mode is False:
            action, _, _ = self.policy.sample(state, clip_action_value=-1) # TODO
        else:
            eval_out = self.policy.sample(state, return_std=return_std, return_entropy=return_entropy)
            if return_std:
                _, logpi, action, std = eval_out
            elif return_entropy:
                _, logpi, action, entropy = eval_out
            else:
                _, logpi, action = eval_out

        action_to_return = action.detach().cpu().numpy()[0]

        return action_to_return, std, entropy, logpi

    # noinspection PyArgumentList
    def select_action(self,
                      state,
                      eval_mode=False,
                      return_log_pi=False,
                      return_std=False,
                      return_entropy=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action_to_return, std, entropy, logpi = self.select_action_single_agent(state=state,
                                                                                eval_mode=eval_mode,
                                                                                return_std=return_std,
                                                                                return_entropy=return_entropy)
        if return_std:
            return action_to_return, std.detach().cpu().numpy()[0]
        elif return_entropy:
            return action_to_return, entropy.detach().cpu().numpy()[0]
        elif return_log_pi:
            return action_to_return, logpi.detach().cpu().numpy().item()
        else:
            return action_to_return

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    # Load model parameters
    def load_model(self, actor_path,
                   critic_path,
                   optimizer_actor_path=None,
                   optimizer_critic_path=None,
                   optimizer_alpha_path=None,
                   map_location=None):
        print('Loading models from {} and {} opt {} {} {}'.format(actor_path,
                                                                  critic_path,
                                                                  optimizer_actor_path,
                                                                  optimizer_critic_path,
                                                                  optimizer_alpha_path))
        if map_location is not None:
            policy = torch.load(actor_path, map_location=map_location)
        else:
            policy = torch.load(actor_path)

        print("Before: alpha {} log_alpha {} target_entropy {}".format(self.alpha, self.log_alpha, self.target_entropy))

        if 'alpha' in policy.keys():
            print("Loading policy with alpha, target entropy and log alpha")
            print(policy.keys())
            # print("The three musketeers alpha {} target entropy {} log alpha {}".format(policy['alpha'],
            #                                                                            policy['target_entropy'],
            #                                                                            policy['log_alpha']))

            self.policy.load_state_dict(policy['model_state_dict'])
            self.policy.train()

            self.alpha = policy['alpha'].detach().item()
            self.target_entropy = \
                -torch.prod(torch.tensor(-policy['target_entropy']).to(self.device)).item()
            self.log_alpha = \
                torch.tensor([policy['log_alpha'].detach().item()], requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            # self.automatic_entropy_tuning = False
            # self.target_entropy = -torch.prod(torch.tensor(action_space.shape).to(self.device)).item()
            # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            # self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            print("After: alpha {} log_alpha {} target_entropy {}".format(self.alpha,
                                                                          self.log_alpha,
                                                                          self.target_entropy))
        elif "worker_id" in policy.keys():
            self.policy.load_state_dict(policy['model_state_dict'])
            self.policy.train()
        else:
            print("Loading just the policy")
            self.policy.load_state_dict(policy)

        if critic_path is not None:
            print("Loading critic")
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.train()

        if optimizer_actor_path is not None:
            print("Loading optimizer policy")
            print("Before opt alpha", self.policy_optim.state_dict())
            self.policy_optim.load_state_dict(torch.load(optimizer_actor_path))
            print("After opt alpha", self.policy_optim.state_dict())

        if optimizer_critic_path is not None:
            print("Loading optimizer critic")
            print("Before opt alpha", self.critic_optim.state_dict())
            self.critic_optim.load_state_dict(torch.load(optimizer_critic_path))
            print("After opt alpha", self.critic_optim.state_dict())

        if optimizer_alpha_path is not None:
            print("Loading optimizer alpha")
            print("Before opt alpha", self.alpha_optim.state_dict())
            self.alpha_optim.load_state_dict(torch.load(optimizer_alpha_path))
            print("After opt alpha", self.alpha_optim.state_dict())
