"""Rollout buffer and GAE for homogeneous 3v3 HAPPO."""
from __future__ import annotations

import numpy as np


class HAPPORolloutBuffer3v3:
    """Fixed-length on-policy rollout buffer for three red HAPPO actors."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        team_size: int = 3,
        obs_dim: int = 68,
        act_dim: int = 3,
        state_dim: int = 48,
    ) -> None:
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.team_size = int(team_size)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.state_dim = int(state_dim)
        self.observations = np.zeros((rollout_steps, num_envs, team_size, obs_dim), np.float32)
        self.global_states = np.zeros((rollout_steps, num_envs, state_dim), np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, team_size, act_dim), np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs, team_size), np.float32)
        self.agent_alive_masks = np.zeros((rollout_steps, num_envs, team_size), np.float32)
        self.team_rewards = np.zeros((rollout_steps, num_envs), np.float32)
        self.team_values = np.zeros((rollout_steps, num_envs), np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), bool)
        self.advantages = np.zeros((rollout_steps, num_envs), np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        agent_alive_masks: np.ndarray,
        team_rewards: np.ndarray,
        team_values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self.position >= self.rollout_steps:
            raise RuntimeError("buffer is full")
        i = self.position
        self.observations[i] = observations
        self.global_states[i] = global_states
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.agent_alive_masks[i] = agent_alive_masks
        self.team_rewards[i] = team_rewards
        self.team_values[i] = team_values
        self.dones[i] = dones
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        if self.position != self.rollout_steps:
            raise ValueError("buffer must be full before GAE")
        if np.asarray(last_values).shape != (self.num_envs,):
            raise ValueError(f"last_values must have shape ({self.num_envs},)")
        next_advantage = np.zeros(self.num_envs, np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_value = last_values if step == self.rollout_steps - 1 else self.team_values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)
            delta = self.team_rewards[step] + gamma * next_value * nonterminal - self.team_values[step]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.team_values
        if not np.isfinite(self.advantages).all() or not np.isfinite(self.returns).all():
            raise FloatingPointError("non-finite HAPPO GAE result")

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0


__all__ = ["HAPPORolloutBuffer3v3"]
