"""Asymmetric actor-critic for SB3 PPO: the critic sees privileged extras, the actor does not.

The env (MazeEnv(critic_extras=True)) appends training-only inputs after the actor's own observation.
`PrivilegedCriticPolicy` slices: the actor's first layer takes only the first `n_actor_obs` features;
the critic takes all of them (actor inputs + extras, the "history + state" form that avoids the bias of
a state-only critic). The exported actor (mlp_extractor.policy_net + action_net) is therefore exactly
the 10-input network that ships, and never depends on information the robot can't sense.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class SlicedMlpExtractor(nn.Module):
    def __init__(self, feature_dim: int, n_actor_obs: int, pi: list[int], vf: list[int], activation_fn):
        super().__init__()
        self.n_actor_obs = n_actor_obs

        def mlp(n_in, sizes):
            layers = []
            for size in sizes:
                layers += [nn.Linear(n_in, size), activation_fn()]
                n_in = size
            return nn.Sequential(*layers), n_in

        self.policy_net, self.latent_dim_pi = mlp(n_actor_obs, pi)
        self.value_net, self.latent_dim_vf = mlp(feature_dim, vf)

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[..., : self.n_actor_obs])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class PrivilegedCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, n_actor_obs: int, **kwargs):
        self.n_actor_obs = n_actor_obs
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        arch = (
            self.net_arch if isinstance(self.net_arch, dict) else {"pi": self.net_arch, "vf": self.net_arch}
        )
        self.mlp_extractor = SlicedMlpExtractor(
            self.features_dim, self.n_actor_obs, list(arch["pi"]), list(arch["vf"]), self.activation_fn
        )
