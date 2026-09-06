"""Independent feed-forward actors used by vanilla HAPPO."""
from __future__ import annotations

from torch import nn
from algorithm.common.networks import CentralizedCritic, GaussianActor
from env.mavuav import OBS_DIM, RED_IDS


class IndependentActors(nn.Module):
    def __init__(self, num_agents: int = len(RED_IDS), observation_dim: int = OBS_DIM, action_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actors = nn.ModuleList([GaussianActor(observation_dim, action_dim, hidden_dim) for _ in range(num_agents)])


__all__ = ["CentralizedCritic", "GaussianActor", "IndependentActors"]
