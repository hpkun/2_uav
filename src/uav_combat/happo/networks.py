"""Independent feed-forward actors used by vanilla HAPPO."""
from __future__ import annotations

from torch import nn
from ..mappo.networks import CentralizedCritic, GaussianActor


class IndependentActors(nn.Module):
    def __init__(self, num_agents: int = 3, observation_dim: int = 40, action_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actors = nn.ModuleList([GaussianActor(observation_dim, action_dim, hidden_dim) for _ in range(num_agents)])


__all__ = ["CentralizedCritic", "GaussianActor", "IndependentActors"]
