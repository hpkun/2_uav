"""Role-guided auxiliary-advantage components for RGAA-HAPPO."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from algorithm.common.buffer import RolloutBuffer
from env.mavuav import OBS_DIM, RED_IDS


RGAA_METHOD = "rgaa"
ROLE_REWARD_KEYS = (
    "mav_process_reward",
    "uav1_process_reward",
    "uav2_process_reward",
    "uav3_process_reward",
)


def extract_role_rewards(infos: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Read the four realized v3.9 process rewards without reconstructing reward."""
    return np.asarray(
        [[float(info[key]) for key in ROLE_REWARD_KEYS] for info in infos],
        dtype=np.float32,
    )


def normalize_active_advantage(
    advantage: torch.Tensor, active: torch.Tensor, epsilon: float = 1e-8,
) -> tuple[torch.Tensor, bool]:
    """Normalize active samples and return zero correction for degenerate inputs."""
    normalized = torch.zeros_like(advantage)
    active = active > 0.5
    if not active.any():
        return normalized, True
    values = advantage[active]
    std = values.std(unbiased=False)
    if std <= epsilon:
        return normalized, True
    normalized[active] = (values - values.mean()) / std
    return normalized, False


class RoleValueNetwork(nn.Module):
    """Observation-conditioned scalar role critic."""

    def __init__(self, observation_dim: int = OBS_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)

    def architecture(self) -> dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": [self.hidden_dim, self.hidden_dim],
            "activation": "tanh",
            "output_dim": 1,
        }


class RoleAdvantageRolloutBuffer(RolloutBuffer):
    """Vanilla buffer plus four role reward/value/GAE streams."""

    def __init__(self, horizon: int, num_envs: int) -> None:
        super().__init__(horizon, num_envs)
        shape = (self.horizon, self.num_envs, len(RED_IDS))
        self.role_rewards = np.zeros(shape, dtype=np.float32)
        self.role_values = np.zeros(shape, dtype=np.float32)
        self.role_advantages = np.zeros(shape, dtype=np.float32)
        self.role_returns = np.zeros(shape, dtype=np.float32)

    def insert(
        self, observations, states, actions, log_probs, rewards, values,
        terminated, truncated, active_masks, *, role_rewards, role_values,
    ) -> None:
        index = self.position
        super().insert(
            observations, states, actions, log_probs, rewards, values,
            terminated, truncated, active_masks,
        )
        self.role_rewards[index] = np.asarray(role_rewards, dtype=np.float32)
        self.role_values[index] = np.asarray(role_values, dtype=np.float32)

    def compute_role_returns_and_advantages(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("role GAE requires a complete fixed-horizon rollout")
        next_values = np.asarray(last_values, dtype=np.float32)
        gae = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for step in reversed(range(self.horizon)):
            boundary = np.logical_or(self.terminated[step], self.truncated[step]).astype(np.float32)
            continuation = (1.0 - boundary)[:, None]
            delta = (
                self.role_rewards[step]
                + gamma * next_values * continuation
                - self.role_values[step]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            self.role_advantages[step] = gae
            next_values = self.role_values[step]
        self.role_returns[:] = self.role_advantages + self.role_values
