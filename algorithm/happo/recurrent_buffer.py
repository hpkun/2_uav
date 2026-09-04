"""Recurrent rollout storage and deterministic truncated-sequence partitioning."""
from __future__ import annotations

import numpy as np

from algorithm.common.buffer import RolloutBuffer


def sequence_chunks(horizon: int, num_envs: int, sequence_length: int) -> list[tuple[int, int, int]]:
    if min(int(horizon), int(num_envs), int(sequence_length)) <= 0:
        raise ValueError("horizon, num_envs and sequence_length must be positive")
    return [
        (env_index, start, min(start + int(sequence_length), int(horizon)))
        for env_index in range(int(num_envs))
        for start in range(0, int(horizon), int(sequence_length))
    ]


class RecurrentRolloutBuffer(RolloutBuffer):
    def __init__(self, horizon: int, num_envs: int, recurrent_hidden_dim: int, **kwargs) -> None:
        super().__init__(horizon, num_envs, **kwargs)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.actor_hidden_states = np.zeros(
            (self.horizon + 1, self.num_envs, self.num_agents, self.recurrent_hidden_dim), np.float32,
        )
        self.recurrent_masks = np.zeros((self.horizon, self.num_envs, self.num_agents), np.float32)

    def insert(
        self, observations, states, actions, log_probs, rewards, values, terminated, truncated,
        active_masks, actor_hidden_states, recurrent_masks, next_actor_hidden_states,
    ) -> None:
        index = self.position
        self.actor_hidden_states[index] = actor_hidden_states
        self.recurrent_masks[index] = recurrent_masks
        super().insert(
            observations, states, actions, log_probs, rewards, values, terminated, truncated, active_masks,
        )
        self.actor_hidden_states[index + 1] = next_actor_hidden_states

    def chunks(self, sequence_length: int) -> list[tuple[int, int, int]]:
        return sequence_chunks(self.horizon, self.num_envs, sequence_length)
