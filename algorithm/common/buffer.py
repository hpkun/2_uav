"""Fixed-horizon on-policy rollout buffer with episode-boundary-safe GAE."""
from __future__ import annotations

import numpy as np
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS


class RolloutBuffer:
    def __init__(self, horizon: int, num_envs: int, num_agents: int = len(RED_IDS), obs_dim: int = OBS_DIM, state_dim: int = GLOBAL_STATE_DIM, action_dim: int = 3) -> None:
        self.horizon, self.num_envs, self.num_agents = int(horizon), int(num_envs), int(num_agents)
        shape = (self.horizon, self.num_envs)
        self.observations = np.zeros(shape + (num_agents, obs_dim), np.float32)
        self.global_states = np.zeros(shape + (state_dim,), np.float32)
        self.actions = np.zeros(shape + (num_agents, action_dim), np.float32)
        self.log_probs = np.zeros(shape + (num_agents,), np.float32)
        self.rewards = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)
        self.terminated = np.zeros(shape, bool)
        self.truncated = np.zeros(shape, bool)
        self.active_masks = np.zeros(shape + (num_agents,), np.float32)
        self.advantages = np.zeros(shape, np.float32)
        self.returns = np.zeros(shape, np.float32)
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def insert(self, observations, states, actions, log_probs, rewards, values, terminated, truncated, active_masks) -> None:
        if self.position >= self.horizon:
            raise RuntimeError("rollout buffer is full")
        i = self.position
        self.observations[i], self.global_states[i], self.actions[i] = observations, states, actions
        self.log_probs[i], self.rewards[i], self.values[i] = log_probs, np.asarray(rewards).mean(axis=-1), values
        self.terminated[i], self.truncated[i], self.active_masks[i] = terminated, truncated, active_masks
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        if self.position != self.horizon:
            raise RuntimeError("GAE requires a complete fixed-horizon rollout")
        next_values = np.asarray(last_values, dtype=np.float32)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.horizon)):
            boundary = np.logical_or(self.terminated[step], self.truncated[step]).astype(np.float32)
            continuation = 1.0 - boundary
            delta = self.rewards[step] + gamma * next_values * continuation - self.values[step]
            gae = delta + gamma * gae_lambda * continuation * gae
            self.advantages[step] = gae
            next_values = self.values[step]
        self.returns[:] = self.advantages + self.values
