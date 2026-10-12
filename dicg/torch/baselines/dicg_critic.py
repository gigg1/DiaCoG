import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Normal
from dicg.torch.modules.gaussian_mlp_module import GaussianMLPModule
from dicg.torch.modules.dicg_base import DICGBase

class DICGCritic(DICGBase):

    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 decoder_hidden_sizes=(64, ),
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 share_std=False,
                 state_include_actions=False,
                 aggregator_type='sum', # 'sum', 'nonl', 'direct'
                 name='dicg_critic'):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            state_include_actions=state_include_actions,
            name=name,
        )

        self.aggregator_type = aggregator_type
        self.n_agents = n_agents

        if aggregator_type == 'sum':
            aggregator_input_dim = embedding_dim
        elif aggregator_type == 'direct':
            aggregator_input_dim = embedding_dim * self._n_agents
        elif aggregator_type == 'nonl':
            # Added: nonlinear mixer MLP
            aggregator_input_dim = embedding_dim
            mixer_hidden_sizes = (64, 32)
            self.value_mixer = self._build_mixer_mlp(mixer_hidden_sizes)

        self.baseline_aggregator = GaussianMLPModule(
            input_dim=aggregator_input_dim,
            output_dim=1,
            hidden_sizes=decoder_hidden_sizes,
            hidden_nonlinearity=torch.tanh,
            share_std=True,
        )

        self.residual = residual

    def _build_mixer_mlp(self, hidden_sizes):
        """Build the nonlinear mixer MLP."""
        layers = []
        input_dim = self.n_agents  # input is the n_agents individual values

        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())  # use ReLU activation
            # Optional: add dropout to prevent overfitting
            input_dim = hidden_size

        # Output layer - outputs a single global value
        layers.append(nn.Linear(input_dim, 1))

        mixer_mlp = nn.Sequential(*layers)

        # Initialize weights
        for layer in mixer_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)  # small positive bias

        return mixer_mlp

    def compute_loss(self, obs_n, returns):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        embeddings_collection, attention_weights = super().forward(obs_n)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]
        # shared std, any std = std.mean()
        if self.aggregator_type == 'sum':
            mean, std = self.baseline_aggregator(emb)     
            baseline_dist = Normal(mean.squeeze(-1).sum(-1), std.mean())
        elif self.aggregator_type == 'direct':
            emb = emb.reshape(emb.shape[:-2] + (-1, )) # concatenate embeddings
            mean, std = self.baseline_aggregator(emb)
            baseline_dist = Normal(mean.squeeze(-1), std.mean())
        elif self.aggregator_type == 'nonl':
            mean, std = self.baseline_aggregator(emb)
            individual_values = mean.squeeze(-1)  # [batch_size, n_agents]
            global_mean = self.value_mixer(individual_values).squeeze(-1)
            baseline_dist = Normal(global_mean, std.mean())

        ll = baseline_dist.log_prob(returns)
        return -ll.mean() # baseline loss

    def get_embeddings(self, obs_n):
        device = next(self.parameters()).device
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        with torch.no_grad():
            embeddings_collection, attention_weights = super().forward(obs_n)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]
        return emb

    def get_embeddings_nodes(self, obs_n):
        device = next(self.parameters()).device
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        with torch.no_grad():
            embeddings_collection, attention_weights = super().forward(obs_n)
        return embeddings_collection[0]

    def get_obses(self, obs_n):
        device = next(self.parameters()).device
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        return obs_n

    def forward(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        embeddings_collection, attention_weights = super().forward(obs_n)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]

        if self.aggregator_type == 'sum':
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1).sum(-1)
        elif self.aggregator_type == 'direct':
            emb = emb.reshape(emb.shape[:-2] + (-1, ))
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1)
        elif self.aggregator_type == 'nonl':
            mean, _ = self.baseline_aggregator(emb)
            individual_values = mean.squeeze(-1)  # [batch_size, n_agents]
            return self.value_mixer(individual_values).squeeze(-1)

    def get_attention_weights(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        _, attention_weights = super().forward(obs_n)
        return attention_weights