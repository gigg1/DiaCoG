import math

import torch
from torch import nn
from torch.nn.parameter import Parameter

class GraphConvolutionModule(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, id=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.id = id
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if len(inputs.shape) == 2:
            support = torch.mm(inputs, self.weight)
            outputs = torch.mm(adj, support)
        elif len(inputs.shape) > 2:
            support = torch.matmul(inputs, self.weight)
            # adj dim = (n_paths, max_path_length, n_agents, n_agents)
            # support dim = (n_paths, max_path_length, n_agents, emb_feat_dim)
            outputs = torch.matmul(adj, support)
            # outputs dim = (n_paths, max_path_length, n_agents, emb_feat_dim)

        if self.bias is not None:
            return torch.tanh(outputs + self.bias)
        else:
            return torch.tanh(outputs)

class GraphConvolutionModule2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, num_features, id=None, activate=True):
        super().__init__()
        self.num_features = num_features
        self.id = id
        self.activate = activate
        self.activate_type = "relu"

    def forward(self, inputs, adj):
        if len(inputs.shape) == 2:
            outputs = torch.mm(adj, inputs)
        elif len(inputs.shape) > 2:
            # adj dim = (n_paths, max_path_length, n_agents, n_agents)
            outputs = torch.matmul(adj, inputs)
            # outputs dim = (n_paths, max_path_length, n_agents, emb_feat_dim)
        if self.activate == True:
            if self.activate_type == "relu":
                return torch.relu(outputs)
            if self.activate_type == "tanh":
                return torch.tanh(outputs)
        else:
            return outputs

class GraphAPPNP(nn.Module):
    def __init__(self, num_features, id=None):
        super().__init__()
        self.num_features = num_features
        self.gamma = 0.05
        self.id = id

    def forward(self, inputs, adj, initial_inputs):
        if len(inputs.shape) == 2:
            outputs = torch.mm(adj, inputs)
            outputs = (1-self.gamma) * outputs + self.gamma * initial_inputs

        elif len(inputs.shape) > 2:
            # adj dim = (n_paths, max_path_length, n_agents, n_agents)
            outputs = torch.matmul(adj, inputs)
            outputs = (1 - self.gamma) * outputs + self.gamma * initial_inputs
            # outputs dim = (n_paths, max_path_length, n_agents, emb_feat_dim)

        return outputs
