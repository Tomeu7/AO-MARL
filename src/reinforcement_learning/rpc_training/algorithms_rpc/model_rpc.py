import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-5

def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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

        self.LOG_SIG_MAX = kwargs['LOG_SIG_MAX']

    def forward(self, state):

        x = self.activation(self.linear1(state))
        for i in range(len(self.hidden)):
            x = self.activation(self.hidden[i](x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self,
               state,
               only_choosing_action=False):

        mean, log_std = self.forward(state)

        if only_choosing_action:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = None
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:

            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()

            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2).clamp(min=0, max=1)) + epsilon)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)