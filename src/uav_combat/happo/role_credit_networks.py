"""Simple role-shared centralized value networks for v14B."""
from __future__ import annotations

import torch
from torch import nn


def _value_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


class RoleSharedCentralizedCritics4v3(nn.Module):
    """One Support critic and one parameter-shared Combat critic."""

    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = int(state_dim) + int(obs_dim)
        self.support_critic = _value_mlp(input_dim, int(hidden_dim))
        self.combat_critic = _value_mlp(input_dim, int(hidden_dim))

    def forward(self, global_states: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-2] != 4:
            raise ValueError("role critic observations must have four red slots")
        support_input = torch.cat((global_states, observations[..., 0, :]), dim=-1)
        support = self.support_critic(support_input).squeeze(-1)
        expanded_state = global_states.unsqueeze(-2).expand(*observations.shape[:-2], 3, global_states.shape[-1])
        combat_input = torch.cat((expanded_state, observations[..., 1:4, :]), dim=-1)
        combat = self.combat_critic(combat_input).squeeze(-1)
        return torch.cat((support.unsqueeze(-1), combat), dim=-1)


__all__ = ["RoleSharedCentralizedCritics4v3"]
