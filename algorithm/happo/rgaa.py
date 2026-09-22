"""Role-guided auxiliary-advantage components for RGAA-HAPPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from algorithm.common.buffer import RolloutBuffer
from env.mavuav import OBS_DIM, RED_IDS


RGAA_METHOD = "rgaa"
ROLE_AUX_REWARD_MODE = "process_plus_own_loss"
ROLE_REWARD_KEYS = (
    "mav_process_reward",
    "uav1_process_reward",
    "uav2_process_reward",
    "uav3_process_reward",
)


@dataclass(frozen=True)
class AuxiliaryRewardBatch:
    process_rewards: np.ndarray
    auxiliary_rewards: np.ndarray
    own_loss_events: np.ndarray
    boundary_loss_events: np.ndarray
    blue_attack_loss_events: np.ndarray


def extract_rgaa_auxiliary_rewards(
    infos: Sequence[Mapping[str, Any]], reward_config: Mapping[str, Any],
) -> AuxiliaryRewardBatch:
    """Build process-plus-own-loss rewards without importing shared event reward."""
    process_rewards = np.asarray(
        [[float(info[key]) for key in ROLE_REWARD_KEYS] for info in infos],
        dtype=np.float32,
    )
    auxiliary_rewards = process_rewards.copy()
    event_shape = process_rewards.shape
    own_loss_events = np.zeros(event_shape, dtype=np.float32)
    boundary_loss_events = np.zeros(event_shape, dtype=np.float32)
    blue_attack_loss_events = np.zeros(event_shape, dtype=np.float32)
    for env_index, info in enumerate(infos):
        death_causes = info.get("death_causes", {})
        for agent_index, agent_id in enumerate(RED_IDS):
            cause = death_causes.get(agent_id)
            if cause is None:
                continue
            penalty_key = "mav_loss" if agent_id == "MAV" else "uav_loss"
            auxiliary_rewards[env_index, agent_index] += float(reward_config[penalty_key])
            own_loss_events[env_index, agent_index] = 1.0
            boundary_loss_events[env_index, agent_index] = float(cause == "boundary")
            blue_attack_loss_events[env_index, agent_index] = float(cause == "blue_attack")
    return AuxiliaryRewardBatch(
        process_rewards=process_rewards,
        auxiliary_rewards=auxiliary_rewards,
        own_loss_events=own_loss_events,
        boundary_loss_events=boundary_loss_events,
        blue_attack_loss_events=blue_attack_loss_events,
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
    """Vanilla buffer plus RGAA auxiliary reward/value/GAE streams.

    ``role_rewards`` means process reward plus the same agent's configured loss
    event. It intentionally excludes shared kill, terminal, and safety rewards.
    """

    def __init__(self, horizon: int, num_envs: int) -> None:
        super().__init__(horizon, num_envs)
        shape = (self.horizon, self.num_envs, len(RED_IDS))
        self.role_rewards = np.zeros(shape, dtype=np.float32)
        self.role_process_rewards = np.zeros(shape, dtype=np.float32)
        self.own_loss_events = np.zeros(shape, dtype=np.float32)
        self.boundary_loss_events = np.zeros(shape, dtype=np.float32)
        self.blue_attack_loss_events = np.zeros(shape, dtype=np.float32)
        self.role_values = np.zeros(shape, dtype=np.float32)
        self.role_advantages = np.zeros(shape, dtype=np.float32)
        self.role_returns = np.zeros(shape, dtype=np.float32)

    def insert(
        self, observations, states, actions, log_probs, rewards, values,
        terminated, truncated, active_masks, *, role_rewards, role_process_rewards,
        own_loss_events, boundary_loss_events, blue_attack_loss_events, role_values,
    ) -> None:
        index = self.position
        super().insert(
            observations, states, actions, log_probs, rewards, values,
            terminated, truncated, active_masks,
        )
        self.role_rewards[index] = np.asarray(role_rewards, dtype=np.float32)
        self.role_process_rewards[index] = np.asarray(role_process_rewards, dtype=np.float32)
        self.own_loss_events[index] = np.asarray(own_loss_events, dtype=np.float32)
        self.boundary_loss_events[index] = np.asarray(boundary_loss_events, dtype=np.float32)
        self.blue_attack_loss_events[index] = np.asarray(blue_attack_loss_events, dtype=np.float32)
        self.role_values[index] = np.asarray(role_values, dtype=np.float32)

    def compute_role_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_active_masks: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("role GAE requires a complete fixed-horizon rollout")
        next_values = np.asarray(last_values, dtype=np.float32)
        next_active_masks = np.asarray(last_active_masks, dtype=np.float32)
        expected_shape = (self.num_envs, self.num_agents)
        if next_values.shape != expected_shape or next_active_masks.shape != expected_shape:
            raise ValueError(f"last role values and active masks must have shape {expected_shape}")
        gae = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for step in reversed(range(self.horizon)):
            boundary = np.logical_or(self.terminated[step], self.truncated[step]).astype(np.float32)
            continuation = (1.0 - boundary)[:, None] * next_active_masks
            delta = (
                self.role_rewards[step]
                + gamma * next_values * continuation
                - self.role_values[step]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            self.role_advantages[step] = gae
            next_values = self.role_values[step]
            next_active_masks = self.active_masks[step]
        self.role_returns[:] = self.role_advantages + self.role_values
