import torch
import torch.nn as nn
import numpy as np


class AttentionModule(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, dimensions, attention_type='general', position_dimensions=None):
        super().__init__()

        self.attention_type = attention_type

        supported = ('general', 'dot', 'diff', 'identity', 'uniform', 'polarized3')
        if attention_type not in supported:
            raise ValueError(
                "Unsupported attention_type '{}'. Supported values are: {}. ".format(
                    attention_type, ', '.join(supported)))

        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        elif self.attention_type == 'diff':
            self.linear_in = nn.Linear(dimensions, 1, bias=False)

        # DiaCoG
        self.add_theta = 0
        if self.attention_type == 'polarized3':
            # weights of the heterogeneous differences (features and positions)
            self.auto_balance_alpha = True
            if self.auto_balance_alpha == True:
                self.a_alpha = nn.Parameter(torch.zeros(size=(1, 2)))
            else:
                self.alpha = 0.75

            self.a = nn.Parameter(torch.FloatTensor(size=(1, dimensions * 2)))
            self.b = nn.Parameter(torch.FloatTensor(size=(1, position_dimensions)))
            # automatically learn \beta
            self.beta = nn.Parameter(torch.zeros(size=(1, 1)))
            if self.add_theta == 1:
                self.theta = nn.Parameter(torch.zeros(size=(1, 1)))
            # initiate parameters
            self.reset_parameters()

        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self):
        if self.attention_type == 'polarized3':
            nn.init.xavier_normal_(self.a, gain=1.414)
            nn.init.xavier_normal_(self.b, gain=1.414)
            nn.init.xavier_normal_(self.beta.data, gain=1.414)
            if self.add_theta == 1:
                nn.init.xavier_normal_(self.theta.data, gain=1.414)

            if self.auto_balance_alpha == True:
                nn.init.xavier_normal_(self.a_alpha.data, gain=1.414)

    def forward(self, query, position_matrix = None):
        """
            Self attention

            n_paths, max_path_length, n_agents, emb_feat_dim = query.shape
            OR
            bs, n_agents, emb_feat_dim = query.shape
            OR
            n_agents, emb_feat_dim = query.shape

        """
        # DiaCoG
        if self.attention_type == 'polarized3':
            n_agents = query.shape[-2]
            device = query.device
            A = torch.ones(size=(n_agents, n_agents)).to(device)
            P = position_matrix

            repeats = (1,) * (len(query.shape) - 2) + (n_agents, 1)
            # Change query shape to (..., n_agents, n_agents, emb_feat_dim)
            augmented_shape = query.shape[:-1] + (n_agents,) + query.shape[-1:]
            query = query.repeat(*repeats).reshape(*augmented_shape)
            context = query.transpose(-3, -2).contiguous()

            concate_e = torch.cat((context, query), axis=-1)
            diff_e = torch.abs(context - query)

            C = (concate_e @ self.a.transpose(-2, -1).contiguous()).squeeze(-1)
            C = C + P @ torch.diag(self.b.squeeze(0)) @ P.transpose(-2, -1).contiguous()
            C = A * C

            # calculate the dissimilarity matrix S
            repeats = (1,) * (len(position_matrix.shape) - 2) + (n_agents, 1)
            augmented_shape = position_matrix.shape[:-1] + (n_agents,) + position_matrix.shape[-1:]
            P_j = P.repeat(*repeats).reshape(*augmented_shape)
            P_i = P_j.transpose(-3, -2).contiguous()
            diff_P = torch.abs(P_i - P_j)

            if self.auto_balance_alpha == False:
                S = self.alpha * torch.sum(torch.pow(diff_e, 2), axis=-1)
                S = S + (1-self.alpha) * torch.sum(torch.pow(diff_P, 2), axis=-1)
            else:
                weights = nn.functional.softmax(self.a_alpha, dim = 1)
                S = weights[0,0] * torch.sum(torch.pow(diff_e, 2), axis=-1)
                S = S + weights[0,1] * torch.sum(torch.pow(diff_P, 2), axis=-1)

            S = A * S

            E = C - self.beta * S
            E = torch.clamp(E, min=-50, max=50)
            E = A * torch.exp(E)

            # numeric stabilization beta to 0-2

            # ---------------------- graph attention ---------------------------
            # normalize the attention weights by softmax

            # ---------------------- graph convolution -------------------------
            degree = torch.sum(E, axis=-1)

            if self.add_theta == 1:
                perm = 1e-9 / (torch.exp(-self.theta) + 1)
                degree_inv = perm / (degree + 1e-9)

                D_inv = torch.zeros(size=E.shape)
                if len(D_inv.shape) == 2:
                    D_inv = torch.diag(degree_inv)
                if len(D_inv.shape) == 3:
                    for i in range(D_inv.shape[0]):
                        D_inv[i] = torch.diag(degree_inv[i])
                if len(D_inv.shape) == 4:
                    for i in range(D_inv.shape[0]):
                        for j in range(D_inv.shape[1]):
                            D_inv[i][j] = torch.diag(degree_inv[i][j])

                # add the weights of self connection
                E = E + D_inv

            # calculate the attention weights by vector row and column multiples
            #   (the results are same with above approach)
            degree_inv = torch.pow(degree, -0.5).unsqueeze(-1)

            attention_weights = degree_inv * E * degree_inv.transpose(-2, -1).contiguous()


        if self.attention_type in ['general', 'dot']:
            context = query.transpose(-2, -1).contiguous()
            if self.attention_type == 'general':
                query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)
            attention_weights = self.softmax(attention_scores)

        elif self.attention_type == 'diff':
            """
                Symmetric
                Kind of unstable
            """
            n_agents = query.shape[-2]
            repeats = (1, ) * (len(query.shape) - 2) + (n_agents, 1)
            augmented_shape = query.shape[:-1] + (n_agents, ) + query.shape[-1:]
            # Change query shape to (..., n_agents, n_agents, emb_feat_dim)
            query = query.repeat(*repeats).reshape(*augmented_shape)
            context = query.transpose(-3, -2).contiguous()

            attention_scores = torch.abs(query - context)
            attention_scores = self.linear_in(attention_scores).squeeze(-1)
            attention_scores = torch.tanh(attention_scores)
            attention_weights = self.softmax(attention_scores)

        elif self.attention_type == 'identity':
            n_agents = query.shape[-2]
            attention_weights = torch.zeros(query.shape[:-2] + (n_agents, n_agents))
            attention_weights.reshape(-1, n_agents, n_agents)
            for i in range(n_agents):
                if len(query.shape) > 2:
                    attention_weights[:, i, i] = 1
                else:
                    attention_weights[i, i] = 1
            attention_weights = \
                attention_weights.reshape(query.shape[:-2]+ (n_agents, n_agents))

        elif self.attention_type == 'uniform':
            n_agents = query.shape[-2]
            attention_weights = torch.ones(query.shape[:-2] + (n_agents, n_agents))
            attention_weights = attention_weights / n_agents

        return attention_weights

