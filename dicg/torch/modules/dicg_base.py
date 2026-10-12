import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Categorical
from garage.torch.policies import Policy

from .mlp_encoder_module import MLPEncoderModule
from .attention_module import AttentionModule
from .graph_conv_module import *

class DICGBase(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 gcn_bias=True,
                 state_include_actions=False,
                 name='dicg_base'):

        super().__init__()

        self.name = name

        self.dicg = True
        self.centralized = True
        self.vectorized = True
        
        self._n_agents = n_agents
        self._cent_obs_dim = env_spec.observation_space.flat_dim
        self._dec_obs_dim = int(self._cent_obs_dim / n_agents)
        if isinstance(env_spec.action_space, akro.Discrete): 
            self._action_dim = env_spec.action_space.n
        else:
            self._action_dim = env_spec.action_space.shape[0]
        self._embedding_dim = embedding_dim # dev

        self.n_gcn_layers = n_gcn_layers # dev

        self.layers = []

        self.attention_type = attention_type

        if attention_type not in ('general', 'polarized3'):
            raise ValueError(
                "Unsupported attention_type '{}'. DICGBase supports 'general' and "
                "'polarized3'.".format(attention_type))

        if state_include_actions:
            self._dec_obs_dim += self._action_dim

        self.activation = torch.tanh
        self.encoder = MLPEncoderModule(input_dim=self._dec_obs_dim,
                                        output_dim=self._embedding_dim,
                                        hidden_sizes=encoder_hidden_sizes,
                                        output_nonlinearity=self.activation)
        self.layers.append(self.encoder)

        # original approach, not the polarized approaches
        if self.attention_type != 'polarized3':
            self.attention_layer = AttentionModule(dimensions=self._embedding_dim,
                                                   attention_type=attention_type)
            self.layers.append(self.attention_layer)

            self.gcn_layers = [
                GraphConvolutionModule(in_features=self._embedding_dim,
                                       out_features=self._embedding_dim,
                                       bias=gcn_bias,
                                       id=i) for i in range(self.n_gcn_layers)
            ]
            self.layers.extend(self.gcn_layers)

        self.gnn_type = "GCN"
        self.if_gcn_activate = True


        if self.attention_type == 'polarized3':
            # set the dimension of position matrix
            self.if_SVD = 0
            if self.if_SVD == 0:
                self.position_dimensions = self._n_agents
            else:
                self.position_dimensions = 6

        if self.attention_type == 'polarized3':
            # calculate the matrix for position matrix
            self.attention_layer_for_position = AttentionModule(dimensions=self._embedding_dim,
                                                                attention_type='general')
            self.layers.append(self.attention_layer_for_position)

            self.feature_encode_2 = nn.Linear(self._embedding_dim, self._embedding_dim, bias=False)
            self.layers.append(self.feature_encode_2)
            self.position_encode_2 = nn.Linear(self.position_dimensions, self.position_dimensions, bias=False)
            self.layers.append(self.position_encode_2)

            # layer for calculating the polarized matrix
            self.attention_layer = AttentionModule(dimensions=self._embedding_dim,
                                                   attention_type=attention_type,
                                                   position_dimensions=self.position_dimensions)
            self.layers.append(self.attention_layer)
            if self.gnn_type == "GCN":
                self.gcn_layers = [
                    GraphConvolutionModule2(num_features=self._embedding_dim,
                                            id=i,
                                            activate=self.if_gcn_activate) for i in range(self.n_gcn_layers)
                ]
                self.layers.extend(self.gcn_layers)

            if self.gnn_type == "APPNP":
                self.gcn_layers = [
                    GraphAPPNP(num_features=self._embedding_dim,
                               id=i) for i in range(self.n_gcn_layers)
                ]
                self.layers.extend(self.gcn_layers)

    def grad_norm(self):

        return np.sqrt(np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs_n):
        # Partially decentralize, treating agents as being independent

        # (n_paths, max_path_length, n_agents, emb_feat_dim)
        embeddings_collection = []
        embeddings_0 = self.encoder.forward(obs_n)
        embeddings_collection.append(embeddings_0)

        # (n_paths, max_path_length, n_agents, n_agents)
        if self.attention_type == 'general':
            # position_matrix is None
            attention_weights = self.attention_layer.forward(embeddings_0)

            for i_layer, gcn_layer in enumerate(self.gcn_layers):
                # (n_paths, max_path_length, n_agents, emb_feat_dim)
                embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer],
                                                   attention_weights)
                embeddings_collection.append(embeddings_gcn)

        # calculate position matrix
        if self.attention_type == 'polarized3':
            # position_matrix is None
            similarity_matrix = self.attention_layer_for_position.forward(embeddings_0)

            # KNN make the similarity matrix as sparse matrix
            import math
            KNN_k = math.ceil(self._n_agents * 0.6)
            values, indices = similarity_matrix.topk(self._n_agents-KNN_k, dim=-1, largest=False, sorted=False)
            device = obs_n.device
            src = torch.zeros(size=similarity_matrix.shape).to(device)
            similarity_matrix = similarity_matrix.scatter(-1, indices, src)

            if self.if_SVD == 1:
                Laplace = 1
                # calculate the position matrix based on the similarity matrix by SVD
                if Laplace == 0:
                    U, S, VT = torch.linalg.svd(similarity_matrix)

                if Laplace == 1:
                    # Laplace matrix =  degree matrix - adjacent matrix
                    # degree matrix is diag(1)


                    degree = torch.sum(similarity_matrix, axis=-1)
                    sim_D = torch.zeros(size=similarity_matrix.shape)
                    if len(sim_D.shape) == 2:
                        sim_D = torch.diag(degree)
                    if len(sim_D.shape) == 3:
                        for i in range(sim_D.shape[0]):
                            sim_D[i] = torch.diag(degree[i])
                    if len(sim_D.shape) == 4:
                        for i in range(sim_D.shape[0]):
                            for j in range(sim_D.shape[1]):
                                sim_D[i][j] = torch.diag(degree[i][j])

                    similarity_matrix = sim_D - similarity_matrix
                    U, S, VT = torch.linalg.svd(similarity_matrix)

                # select the top K vectors and conduct encoding
                position_matrix = U[..., :self.position_dimensions]


                # map the position matrix to higher dimensions
            else:
                # not do SVD, the position matrix is n*n sparse matrix
                position_matrix = similarity_matrix[..., :self.position_dimensions]

                
            embeddings = self.feature_encode_2.forward(embeddings_0)
            embeddings_collection[0] = embeddings
            position_matrix = self.position_encode_2.forward(position_matrix)
            attention_weights = self.attention_layer.forward(embeddings, position_matrix)

            for i_layer, gcn_layer in enumerate(self.gcn_layers):
                if self.gnn_type == "GCN":
                    embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], attention_weights)
                if self.gnn_type == "APPNP":
                    embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], attention_weights,
                                                       embeddings_collection[0])
                embeddings_collection.append(embeddings_gcn)

        return embeddings_collection, attention_weights


    def reset(self, dones):
        pass

    