"""Loss-separated auxiliary credit for LS-RGAA-HAPPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from algorithm.common.buffer import RolloutBuffer
from env.mavuav import OBS_DIM, RED_IDS
from .rgaa import ROLE_REWARD_KEYS


LS_RGAA_METHOD = "ls_rgaa"
LS_RGAA_VERSION = 1
LS_AUXILIARY_SEMANTICS = "loss_separated_process_and_binary_own_loss_v1"


@dataclass(frozen=True)
class LossSeparatedRewardBatch:
    process_rewards: np.ndarray
    loss_rewards: np.ndarray
    own_loss_events: np.ndarray
    boundary_loss_events: np.ndarray
    blue_attack_loss_events: np.ndarray


def extract_ls_rewards(infos: Sequence[Mapping[str, Any]]) -> LossSeparatedRewardBatch:
    """Extract process-only rewards and binary own-loss rewards."""
    process_rewards = np.asarray(
        [[float(info[key]) for key in ROLE_REWARD_KEYS] for info in infos], dtype=np.float32,
    )
    shape = process_rewards.shape
    own_loss_events = np.zeros(shape, dtype=np.float32)
    boundary_loss_events = np.zeros(shape, dtype=np.float32)
    blue_attack_loss_events = np.zeros(shape, dtype=np.float32)
    for env_index, info in enumerate(infos):
        death_causes = info.get("death_causes", {})
        for agent_index, agent_id in enumerate(RED_IDS):
            cause = death_causes.get(agent_id)
            if cause not in ("boundary", "blue_attack"):
                continue
            own_loss_events[env_index, agent_index] = 1.0
            boundary_loss_events[env_index, agent_index] = float(cause == "boundary")
            blue_attack_loss_events[env_index, agent_index] = float(cause == "blue_attack")
    return LossSeparatedRewardBatch(
        process_rewards=process_rewards,
        loss_rewards=-own_loss_events,
        own_loss_events=own_loss_events,
        boundary_loss_events=boundary_loss_events,
        blue_attack_loss_events=blue_attack_loss_events,
    )


class LossValueNetwork(nn.Module):
    """Observation-conditioned value constrained strictly to (-1, 0)."""

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
        return -torch.sigmoid(self.network(observation).squeeze(-1))

    def architecture(self) -> dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": [self.hidden_dim, self.hidden_dim],
            "activation": "tanh",
            "output_dim": 1,
            "output_transform": "negative_sigmoid",
            "output_range": "(-1,0)",
        }


class LossSeparatedRoleRolloutBuffer(RolloutBuffer):
    """HAPPO rollout with separate process GAE and binary-loss TD(lambda)."""

    def __init__(self, horizon: int, num_envs: int) -> None:
        super().__init__(horizon, num_envs)
        shape = (self.horizon, self.num_envs, len(RED_IDS))
        self.process_rewards = np.zeros(shape, dtype=np.float32)
        self.process_values = np.zeros(shape, dtype=np.float32)
        self.process_advantages = np.zeros(shape, dtype=np.float32)
        self.process_returns = np.zeros(shape, dtype=np.float32)
        self.loss_rewards = np.zeros(shape, dtype=np.float32)
        self.loss_values = np.zeros(shape, dtype=np.float32)
        self.loss_returns = np.zeros(shape, dtype=np.float32)
        self.own_loss_events = np.zeros(shape, dtype=np.float32)
        self.boundary_loss_events = np.zeros(shape, dtype=np.float32)
        self.blue_attack_loss_events = np.zeros(shape, dtype=np.float32)

    def insert(
        self, observations, states, actions, log_probs, rewards, values,
        terminated, truncated, active_masks, *, process_rewards, process_values,
        loss_rewards, loss_values, own_loss_events, boundary_loss_events,
        blue_attack_loss_events,
    ) -> None:
        index = self.position
        super().insert(
            observations, states, actions, log_probs, rewards, values,
            terminated, truncated, active_masks,
        )
        self.process_rewards[index] = np.asarray(process_rewards, dtype=np.float32)
        self.process_values[index] = np.asarray(process_values, dtype=np.float32)
        self.loss_rewards[index] = np.asarray(loss_rewards, dtype=np.float32)
        self.loss_values[index] = np.asarray(loss_values, dtype=np.float32)
        self.own_loss_events[index] = np.asarray(own_loss_events, dtype=np.float32)
        self.boundary_loss_events[index] = np.asarray(boundary_loss_events, dtype=np.float32)
        self.blue_attack_loss_events[index] = np.asarray(blue_attack_loss_events, dtype=np.float32)

    def compute_auxiliary_returns(
        self,
        last_process_values: np.ndarray,
        last_loss_values: np.ndarray,
        last_active_masks: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("LS-RGAA returns require a complete fixed-horizon rollout")
        expected = (self.num_envs, self.num_agents)
        next_process_values = np.asarray(last_process_values, dtype=np.float32)
        next_loss_values = np.asarray(last_loss_values, dtype=np.float32)
        next_active_masks = np.asarray(last_active_masks, dtype=np.float32)
        if any(value.shape != expected for value in (
            next_process_values, next_loss_values, next_active_masks,
        )):
            raise ValueError(f"last auxiliary values and active masks must have shape {expected}")
        process_gae = np.zeros(expected, dtype=np.float32)
        next_loss_return = next_loss_values.copy()
        for step in reversed(range(self.horizon)):
            episode_boundary = np.logical_or(
                self.terminated[step], self.truncated[step],
            ).astype(np.float32)
            continuation = (1.0 - episode_boundary)[:, None] * next_active_masks
            process_delta = (
                self.process_rewards[step]
                + gamma * next_process_values * continuation
                - self.process_values[step]
            )
            process_gae = (
                process_delta + gamma * gae_lambda * continuation * process_gae
            )
            self.process_advantages[step] = process_gae
            self.loss_returns[step] = self.loss_rewards[step] + gamma * continuation * (
                (1.0 - gae_lambda) * next_loss_values
                + gae_lambda * next_loss_return
            )
            next_process_values = self.process_values[step]
            next_loss_values = self.loss_values[step]
            next_loss_return = self.loss_returns[step]
            next_active_masks = self.active_masks[step]
        self.process_returns[:] = self.process_advantages + self.process_values
