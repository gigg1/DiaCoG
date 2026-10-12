"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from torch.distributions import Normal
from dicg.torch.modules.gaussian_mlp_module import GaussianMLPModule

# embeddings as the input
class GaussianMLPMixBaseline2(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 embedding_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 aggregator_type='sum',  # 'sum', 'nonl', 'direct'
                 name='GaussianMLPMixBaseline'):
        super().__init__()

        self.aggregator_type = aggregator_type
        self.n_agents = n_agents
        self.embedding_dim = embedding_dim

        if aggregator_type == 'sum':
            self.input_dim = self.embedding_dim
        elif aggregator_type == 'direct':
            self.input_dim = self.embedding_dim*self.n_agents
        elif aggregator_type == 'nonl':
            # 新增：非线性混合器MLP
            self.input_dim = self.embedding_dim

            mixer_hidden_sizes = (64, 32)
            self.value_mixer = self._build_mixer_mlp(mixer_hidden_sizes)

        output_dim = 1  # V(s), value function

        self.baseline_aggregator= GaussianMLPModule(
            input_dim=self.input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)

    def _build_mixer_mlp(self, hidden_sizes):
        """构建MLP非线性混合器"""
        layers = []
        input_dim = self.n_agents  # 输入是n_agents个个体value

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())  # 使用ReLU激活函数
            # 可选：添加dropout防止过拟合
            input_dim = hidden_size

        # 输出层 - 输出单个全局价值
        layers.append(nn.Linear(input_dim, 1))

        mixer_mlp = nn.Sequential(*layers)

        # 初始化权重
        for layer in mixer_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)  # 小的正偏置

        return mixer_mlp

    def compute_loss(self, obs, returns):
        emb = torch.Tensor(obs)
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

    # pylint: disable=arguments-differ
    def forward(self, obs):
        emb = torch.Tensor(obs)

        if self.aggregator_type == 'sum':
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1).sum(-1)
        elif self.aggregator_type == 'direct':
            emb = emb.reshape(emb.shape[:-2] + (-1,))
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1)
        elif self.aggregator_type == 'nonl':
            mean, _ = self.baseline_aggregator(emb)
            individual_values = mean.squeeze(-1)  # [batch_size, n_agents]
            return self.value_mixer(individual_values).squeeze(-1)

