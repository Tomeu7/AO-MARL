import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# TODO CHECK THIS

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-5


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """
    Value network
    Deprecated for now
    """
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class RtacVNetwork(nn.Module):
    """
    RtacVNetwork i.e. V(s) parametrized with weights as a neural network
    From an input state pairs it outputs the expected return following policy pi after starting in state s
    RTAC: real time actor critic
    The cool thing about this is that it is thought to handle delay
    """
    def __init__(self, num_inputs, hidden_dim, num_layers):
        super(RtacVNetwork, self).__init__()

        # Q1 inputs and outputs
        self.Q1_input = nn.Linear(num_inputs, hidden_dim)
        self.Q1_output = nn.Linear(hidden_dim, 1)

        # Q2 inputs and outputs
        self.Q2_input = nn.Linear(num_inputs, hidden_dim)
        self.Q2_output = nn.Linear(hidden_dim, 1)

        self.hidden_Q1 = nn.ModuleList()
        self.hidden_Q2 = nn.ModuleList()
        for i in range(num_layers-1):
            self.hidden_Q1.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_Q2.append(nn.Linear(hidden_dim, hidden_dim))

        self.apply(weights_init_)

    def forward(self, x):

        x1 = F.relu(self.Q1_input(x))
        x2 = F.relu(self.Q2_input(x))

        for i in range(len(self.hidden_Q1)):
            x1 = F.relu(self.hidden_Q1[i](x1))
            x2 = F.relu(self.hidden_Q2[i](x2))

        x1 = self.Q1_output(x1)
        x2 = self.Q2_output(x2)

        return x1, x2


class QNetwork(nn.Module):
    """
    QNetwork i.e. Q(s,a) parametrized with weights as a neural network
    From an input state-action pairs it outputs the expected return following policy pi after taking action a in state s
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers):
        super(QNetwork, self).__init__()

        if isinstance(hidden_dim, list):
            # Q1 inputs and outputs
            self.Q1_input = nn.Linear(num_inputs + num_actions, hidden_dim[0])
            self.Q1_output = nn.Linear(hidden_dim[-1], 1)

            # Q2 inputs and outputs
            self.Q2_input = nn.Linear(num_inputs + num_actions, hidden_dim[0])
            self.Q2_output = nn.Linear(hidden_dim[-1], 1)

            self.hidden_Q1 = nn.ModuleList()
            self.hidden_Q2 = nn.ModuleList()
            for i in range(len(hidden_dim) - 1):
                self.hidden_Q1.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                self.hidden_Q2.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        else:
            # Q1 inputs and outputs
            self.Q1_input = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.Q1_output = nn.Linear(hidden_dim, 1)

            # Q2 inputs and outputs
            self.Q2_input = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.Q2_output = nn.Linear(hidden_dim, 1)

            self.hidden_Q1 = nn.ModuleList()
            self.hidden_Q2 = nn.ModuleList()
            for i in range(num_layers-1):
                self.hidden_Q1.append(nn.Linear(hidden_dim, hidden_dim))
                self.hidden_Q2.append(nn.Linear(hidden_dim, hidden_dim))

        self.apply(weights_init_)

    def forward(self, state, action):

        x = torch.cat([state, action], 1)

        x1 = F.relu(self.Q1_input(x))
        x2 = F.relu(self.Q2_input(x))
        for i in range(len(self.hidden_Q1)):
            x1 = F.relu(self.hidden_Q1[i](x1))
            x2 = F.relu(self.hidden_Q2[i](x2))

        x1 = self.Q1_output(x1)
        x2 = self.Q2_output(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    """
    A Gaussian policy from an input outputs the values of a gaussian distribution mean and std
    Forward: mean and std
    Sample: from mean and std we sample action, reescale if necessary and we return logprob needed for update
    """
    def __init__(self, **kwargs):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(kwargs['num_inputs'], kwargs['hidden_dim'])
        self.hidden = nn.ModuleList()
        for i in range(kwargs['num_layers'] - 1):
            self.hidden.append(nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim']))

        self.mean_linear = nn.Linear(kwargs['hidden_dim'], kwargs['num_actions'])
        self.log_std_linear = nn.Linear(kwargs['hidden_dim'], kwargs['num_actions'])
        self.squashing = not kwargs['no_squashing']

        if kwargs['activation'] == "relu":
            self.activation = F.relu
        elif kwargs['activation'] == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        """
        Initialization of NN:
        weights: xavier_uniform
        bias: 0
        Last layer of RL: 0 if initialize_last_layer_zero
        """
        self.apply(weights_init_)
        if kwargs['initialize_last_layer_zero']:
            with torch.no_grad():
                self.mean_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                self.log_std_linear.weight = torch.nn.Parameter(torch.zeros_like(self.log_std_linear.weight))
        elif kwargs['initialize_last_layer_near_zero']:
            with torch.no_grad():
                self.mean_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                self.log_std_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                torch.nn.init.xavier_uniform_(self.mean_linear.weight,
                                              gain=1e-4)
                torch.nn.init.xavier_uniform_(self.log_std_linear.weight,
                                              gain=1e-4)

        self.action_scale = torch.tensor(float(kwargs['action_scale']))
        self.action_bias = torch.tensor(float(kwargs['action_bias']))
        self.output_normalization = kwargs['output_normalization']

    def forward(self, state):

        x = self.activation(self.linear1(state))
        for i in range(len(self.hidden)):
            x = self.activation(self.hidden[i](x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self,
               state,
               return_std=False,
               clip_action_value=-1,
               return_entropy=False,
               return_full_logprob=False):
        mean, log_std = self.forward(state)

        if self.output_normalization:
            # https://github.com/AutumnWu/Streamlined-Off-Policy-Learning/
            # spinup/algos/sop_pytorch/SOP_core_auto.py
            abs_mean = torch.abs(mean)
            G = torch.mean(abs_mean, dim=1).view(-1,1)
            ones = torch.ones(G.size()).to(G.device.type)
            actual_g = torch.where(G >= 1, G, ones)
            mean = mean/actual_g

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        if clip_action_value > 0:
            x_t = x_t.clamp(min=-clip_action_value, max=clip_action_value)

        if self.squashing:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2).clamp(min=0, max=1)) + epsilon)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = torch.clamp(x_t, -self.action_scale.item(), self.action_scale.item())
            log_prob = normal.log_prob(x_t)

        if not return_full_logprob:
            # (bsz x a)
            # (bsz x 1)
            log_prob = log_prob.sum(1, keepdim=True)
            # si sumam log de probabilities es com producte de probabilitats = probabilitat conjunta

        if return_std:
            return action, log_prob, mean, std
        elif return_entropy:
            return action, log_prob, mean, normal.entropy()
        else:
            return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy from and input outputs an action fully deterministically
    We add some noise for exploration
    """
    def __init__(self, **kwargs):
        super(DeterministicPolicy, self).__init__()

        self.linear1 = nn.Linear(kwargs['num_inputs'], kwargs['hidden_dim'])
        self.hidden = nn.ModuleList()
        for i in range(kwargs['num_layers'] - 1):
            self.hidden.append(nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim']))

        self.mean = nn.Linear(kwargs['hidden_dim'], kwargs['num_actions'])
        self.noise = torch.Tensor(kwargs['num_actions'])
        self.std_noise = float(kwargs['deterministic_std_noise'])

        if kwargs['initialize_last_layer_zero']:
            with torch.no_grad():
                self.mean_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                self.log_std_linear.weight = torch.nn.Parameter(torch.zeros_like(self.log_std_linear.weight))
        elif kwargs['initialize_last_layer_near_zero']:
            with torch.no_grad():
                self.mean_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                self.log_std_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
                torch.nn.init.xavier_uniform_(self.mean_linear.weight,
                                              gain=1e-4)
                torch.nn.init.xavier_uniform_(self.log_std_linear.weight,
                                              gain=1e-4)
        """
        Initialization of NN:
        weights: xavier_uniform
        bias: 0
        """
        self.apply(weights_init_)

        # action rescaling
        self.action_scale = torch.tensor(float(kwargs['action_scale']))
        self.action_bias = torch.tensor(float(kwargs['action_bias']))

        self.output_normalization = kwargs['output_normalization']

    def forward(self, state):
        x = F.relu(self.linear1(state))
        for i in range(len(self.hidden)):
            x = F.relu(self.hidden[i](x))

        mean = self.mean(x)

        if self.output_normalization:
            # https://github.com/AutumnWu/Streamlined-Off-Policy-Learning/
            # spinup/algos/sop_pytorch/SOP_core_auto.py
            abs_mean = torch.abs(mean)
            G = torch.mean(abs_mean, dim=1).view(-1, 1)
            ones = torch.ones(G.size()).to(G.device.type)
            actual_g = torch.where(G >= 1, G, ones)
            mean = mean / actual_g

        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=self.std_noise)
        noise = noise.clamp(-0.25, 0.25)
        if self.output_normalization:
            # action = torch.tanh(mean + noise) * self.action_scale + self.action_bias
            # squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
            squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
            action = squashed_mean + noise
        else:
            squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
            action = squashed_mean + noise

        return action, torch.tensor(0.), squashed_mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class GaussianPolicyCorrector(nn.Module):
    """
    Gaussian policy that has a control matrix connected to the output i.e.
     output
    ||   ||
    CM  hidden_layers
    ||  ||       ||
     WFS     Rest of the state
    """
    def __init__(self, **kwargs):

        super(GaussianPolicyCorrector, self).__init__()

        self.use_as_integrator = True if kwargs['num_dm_before_linear'] is not None else False
        # self.grad_control_matrix = kwargs['grad_control_matrix']

        self.linear1 = nn.Linear(kwargs['num_inputs'], kwargs['hidden_dim'])
        self.hidden = nn.ModuleList()
        for i in range(kwargs['num_layers'] - 1):
            self.hidden.append(nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim']))

        self.control_matrix = nn.Linear(kwargs['num_wfs'].shape[0], kwargs['num_actions'], bias=False)

        if not kwargs['grad_control_matrix']:
            self.control_matrix.weight.requires_grad = False
            self.control_matrix.requires_grad = False

        self.mean_linear = nn.Linear(kwargs['hidden_dim'], kwargs['num_actions'])
        self.log_std_linear = nn.Linear(kwargs['hidden_dim'], kwargs['num_actions'])

        """
        Initialization of NN:
        weights: xavier_uniform
        bias: 0
        Last layer of RL: 0
        Control matrix part: control matrix weights without bias
        """
        self.apply(weights_init_)

        with torch.no_grad():
            self.num_wfs = torch.tensor(kwargs['num_wfs'], dtype=torch.int64)
            if kwargs['num_dm_before_linear'] is not None:
                self.num_dm_before_linear = torch.tensor(kwargs['num_dm_before_linear'], dtype=torch.int64)

            self.mean_linear.weight = torch.nn.Parameter(torch.zeros_like(self.mean_linear.weight))
            self.log_std_linear.weight = torch.nn.Parameter(torch.zeros_like(self.log_std_linear.weight))
            self.control_matrix.weight = torch.nn.Parameter(torch.tensor(kwargs['gain'] *
                                                                         kwargs['control_matrix'],
                                                                         dtype=torch.float32),
                                                            requires_grad=kwargs['grad_control_matrix'])

        # action rescaling
        self.action_scale = torch.tensor(float(kwargs['action_scale']))
        self.action_bias = torch.tensor(float(kwargs['action_bias']))

        self.mu_wfs = nn.Parameter(torch.tensor(kwargs['norm_pars']['wfs']['mean'], dtype=torch.float32),
                                   requires_grad=False)
        self.std_wfs = nn.Parameter(torch.tensor(kwargs['norm_pars']['wfs']['std'], dtype=torch.float32),
                                    requires_grad=False)

        self.mu_dm_before_linear = nn.Parameter(torch.tensor(kwargs['norm_pars']['dm']['mean'],
                                                             dtype=torch.float32), requires_grad=False)
        self.std_dm_before_linear = nn.Parameter(torch.tensor(kwargs['norm_pars']['dm']['std'],
                                                              dtype=torch.float32), requires_grad=False)

        self.output_normalization = kwargs['output_normalization']

    def forward(self, state):

        x = F.relu(self.linear1(state))
        for i in range(len(self.hidden)):
            x = F.relu(self.hidden[i](x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        result_control_matrix = self.control_matrix(state[:, self.num_wfs]*self.std_wfs+self.mu_wfs)

        return mean, log_std, result_control_matrix

    def sample(self, state):

        mean, log_std, result_control_matrix = self.forward(state)

        if self.output_normalization:
            # https://github.com/AutumnWu/Streamlined-Off-Policy-Learning/
            # spinup/algos/sop_pytorch/SOP_core_auto.py
            abs_mean = torch.abs(mean)
            G = torch.mean(abs_mean, dim=1).view(-1, 1)
            ones = torch.ones(G.size()).to(G.device.type)
            actual_g = torch.where(G >= 1, G, ones)
            mean = mean / actual_g

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        if self.use_as_integrator:
            with torch.no_grad():
                result_previous_dm = state[:, self.num_dm_before_linear] * self.std_dm_before_linear + self.mu_dm_before_linear
            action = action + result_previous_dm - result_control_matrix
        else:
            action = action + result_control_matrix

        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if self.use_as_integrator:
            with torch.no_grad():
                result_previous_dm = state[:, self.num_dm_before_linear] * self.std_dm_before_linear + self.mu_dm_before_linear
            mean = mean + result_previous_dm - result_control_matrix
        else:
            mean = mean + result_control_matrix

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.num_wfs = self.num_wfs.to(device)
        return super(GaussianPolicyCorrector, self).to(device)
