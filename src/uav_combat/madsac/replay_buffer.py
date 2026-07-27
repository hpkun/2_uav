"""Uniform replay buffer for homogeneous 3v3 MADSAC."""
from __future__ import annotations

import numpy as np
import torch


class MADSACReplayBuffer:
    """Fixed-capacity circular replay buffer with preallocated NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int = 68,
        team_size: int = 3,
        action_dim: int = 3,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.team_size = team_size
        self.action_dim = action_dim
        self.observations = np.zeros((capacity, team_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, team_size, action_dim), dtype=np.float32)
        self.team_rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, team_size, obs_dim), dtype=np.float32)
        self.alive_masks = np.zeros((capacity, team_size), dtype=np.float32)
        self.next_alive_masks = np.zeros((capacity, team_size), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        team_rewards: np.ndarray,
        next_observations: np.ndarray,
        alive_masks: np.ndarray,
        next_alive_masks: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        n = int(len(team_rewards))
        for i in range(n):
            j = self.position
            self.observations[j] = observations[i]
            self.actions[j] = actions[i]
            self.team_rewards[j] = team_rewards[i]
            self.next_observations[j] = next_observations[i]
            self.alive_masks[j] = alive_masks[i]
            self.next_alive_masks[j] = next_alive_masks[i]
            self.terminated[j] = bool(terminated[i])
            self.truncated[j] = bool(truncated[i])
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device | str | None = None) -> dict[str, np.ndarray | torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"not enough replay samples: {self.size} < {batch_size}")
        idx = rng.integers(0, self.size, size=batch_size)
        batch = {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "team_rewards": self.team_rewards[idx],
            "next_observations": self.next_observations[idx],
            "alive_masks": self.alive_masks[idx],
            "next_alive_masks": self.next_alive_masks[idx],
            "terminated": self.terminated[idx],
            "truncated": self.truncated[idx],
            "done_for_bootstrap": np.logical_or(self.terminated[idx], self.truncated[idx]),
        }
        if device is None:
            return batch
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if value.dtype == bool:
                out[key] = torch.as_tensor(value, device=device)
            else:
                out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        return out

    def metadata(self, include_full_replay: bool = False) -> dict[str, int | bool]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "full_replay_persisted": bool(include_full_replay),
        }
