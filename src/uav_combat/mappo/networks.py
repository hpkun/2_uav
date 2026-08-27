"""Feed-forward squashed-Gaussian actor and centralized critic."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
from ..mavuav import GLOBAL_STATE_DIM, OBS_DIM


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int = OBS_DIM, action_dim: int = 3, hidden_dim: int = 128, log_std_init: float = -0.5) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.epsilon = 1e-6

    def _distribution(self, observations: torch.Tensor) -> Normal:
        return Normal(self.network(observations), self.log_std.clamp(-5.0, 2.0).exp())

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - actions.square() + self.epsilon)
        return actions, log_probs.sum(dim=-1)

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        raw = torch.atanh(clipped)
        distribution = self._distribution(observations)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - clipped.square() + self.epsilon)
        return log_probs.sum(dim=-1), distribution.entropy().sum(dim=-1)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int = GLOBAL_STATE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
