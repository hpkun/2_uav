"""TAM rollout storage with independent actor and critic recurrent state."""
from __future__ import annotations

import numpy as np

from .recurrent_buffer import RecurrentRolloutBuffer


class TAMRolloutBuffer(RecurrentRolloutBuffer):
    def __init__(self, horizon: int, num_envs: int, actor_hidden_dim: int, critic_hidden_dim: int) -> None:
        super().__init__(horizon, num_envs, actor_hidden_dim)
        self.critic_hidden_states = np.zeros(
            (self.horizon + 1, self.num_envs, int(critic_hidden_dim)), np.float32,
        )
        self.critic_recurrent_masks = np.zeros((self.horizon, self.num_envs), np.float32)

    def insert(
        self, observations, states, actions, log_probs, rewards, values, terminated, truncated,
        active_masks, actor_hidden_states, recurrent_masks, next_actor_hidden_states,
        critic_hidden_states, critic_recurrent_masks, next_critic_hidden_states,
    ) -> None:
        index = self.position
        self.critic_hidden_states[index] = critic_hidden_states
        self.critic_recurrent_masks[index] = critic_recurrent_masks
        super().insert(
            observations, states, actions, log_probs, rewards, values, terminated, truncated,
            active_masks, actor_hidden_states, recurrent_masks, next_actor_hidden_states,
        )
        self.critic_hidden_states[index + 1] = next_critic_hidden_states


__all__ = ["TAMRolloutBuffer"]
