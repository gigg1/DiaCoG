"""CategoricalMLPPolicy."""
import akro
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from dicg.torch.modules import CategoricalMLPModule
from garage.torch.policies import Policy


class DecCategoricalMLPPolicy(Policy, CategoricalMLPModule):
    """CategoricalMLPPolicy

     A policy that contains a MLP to make prediction based on a categorical
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 n_agents,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='DecCategoricalMLPPolicy'):

        assert isinstance(env_spec.action_space, akro.Discrete), (
            'CategoricalMLPPolicy only works with akro.Discrete action '
            'space.')

        self.centralized = True # centralized training

        self._n_agents = n_agents
        self._obs_dim = int(env_spec.observation_space.flat_dim / n_agents) # dec obs_dim
        self._action_dim = env_spec.action_space.n

        Policy.__init__(self, env_spec, name)
        CategoricalMLPModule.__init__(self,
                                      input_dim=self._obs_dim,
                                      output_dim=self._action_dim,
                                      hidden_sizes=hidden_sizes,
                                      hidden_nonlinearity=hidden_nonlinearity,
                                      hidden_w_init=hidden_w_init,
                                      hidden_b_init=hidden_b_init,
                                      output_nonlinearity=output_nonlinearity,
                                      output_w_init=output_w_init,
                                      output_b_init=output_b_init,
                                      layer_normalization=layer_normalization)

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs, avail_actions):
        obs = obs.reshape(obs.shape[:-1] + (self._n_agents, -1))
        dist = super().forward(obs)
        # Apply available actions mask
        avail_actions = avail_actions.reshape(
            avail_actions.shape[:-1] + (self._n_agents, -1))
        masked_probs = dist.probs * avail_actions # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dist = Categorical(probs=masked_probs) # redefine distribution
        return masked_dist

    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted action.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution

        """
        # Not maintained
        with torch.no_grad():
            observation = torch.Tensor(observation).unsqueeze(0)
            dist = self.forward(observation)
            return (dist.sample().squeeze(0).numpy(), 
                        dict(probs=dist._param.numpy()))

    def get_actions(self, observations, avail_actions, greedy=False):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution

        """

        # Get the device of the model
        device = next(self.parameters()).device
        # If observations is a list, convert to a numpy array first (performance)
        if isinstance(observations, list):
            observations = np.array(observations)
        if isinstance(avail_actions, list):
            avail_actions = np.array(avail_actions)

        # Convert inputs to tensors on the same device as the model
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        avail_actions_tensor = torch.as_tensor(avail_actions, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist = self.forward(observations_tensor, avail_actions_tensor)


            if not greedy:
                actions_tensor = dist.sample()  # on GPU
                actions = actions_tensor.cpu().numpy()
            else:
                probs_tensor = dist.probs  # on GPU
                probs_cpu = probs_tensor.cpu().numpy()
                actions = np.argmax(probs_cpu, axis=-1)


            agent_infos = {}
            probs_tensor = dist.probs
            probs_cpu = probs_tensor.cpu().numpy()
            agent_infos['action_probs'] = [probs_cpu[i] for i in range(len(actions))]

            return actions, agent_infos

    def log_likelihood(self, observations, avail_actions, actions):
        """Compute log likelihood given observations and action.

        Args:
            observation (torch.Tensor): Observation from the environment.
            action (torch.Tensor): Predicted action.

        Returns:
            torch.Tensor: Calculated log likelihood value of the action given
                observation

        """
        dist = self.forward(observations, avail_actions)
        # For n agents, action probabilities are treated as independent
        # => log(Pa) = sum_i^n log(Pa_i)
        return dist.log_prob(actions).sum(axis=-1)

    def entropy(self, observations, avail_actions):
        """Get entropy given observations.

        Args:
            observation (torch.Tensor): Observation from the environment.

        Returns:
             torch.Tensor: Calculated entropy values given observation

        """
        dist = self.forward(observations, avail_actions)
        return dist.entropy().mean(axis=-1)

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """
        pass

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: flag for vectorized

        """
        return True

    @property
    def recurrent(self):
        return False