"""Policy-centric fixed-length rollout buffer and GAE."""
import numpy as np


class MAPPOBuffer:
    """Store only the active policy's [T,N] on-policy transitions."""

    def __init__(self, rollout_steps: int, num_envs: int, observation_dim: int = 14, action_dim: int = 3) -> None:
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        self.rollout_steps, self.num_envs = rollout_steps, num_envs
        self.observations = np.zeros((rollout_steps, num_envs, observation_dim), np.float32)
        self.global_states = np.zeros((rollout_steps, num_envs, 14), np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, action_dim), np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), np.float32)
        self.values = np.zeros((rollout_steps, num_envs), np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), bool)
        self.active_teams = np.zeros((rollout_steps, num_envs), np.int8)
        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)
        self.position = 0

    def add(self, observations, global_states, actions, log_probs, rewards, values, dones, active_teams) -> None:
        if self.position >= self.rollout_steps:
            raise RuntimeError("buffer is full")
        expected = ((observations, self.observations.shape[1:]), (global_states, self.global_states.shape[1:]),
                    (actions, self.actions.shape[1:]), (log_probs, self.log_probs.shape[1:]),
                    (rewards, self.rewards.shape[1:]), (values, self.values.shape[1:]),
                    (dones, self.dones.shape[1:]), (active_teams, self.active_teams.shape[1:]))
        for value, shape in expected:
            if np.asarray(value).shape != shape:
                raise ValueError(f"expected shape {shape}, got {np.asarray(value).shape}")
        i = self.position
        for target, value in ((self.observations, observations), (self.global_states, global_states),
                              (self.actions, actions), (self.log_probs, log_probs), (self.rewards, rewards),
                              (self.values, values), (self.dones, dones), (self.active_teams, active_teams)):
            target[i] = value
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        if self.position != self.rollout_steps or np.asarray(last_values).shape != (self.num_envs,):
            raise ValueError("buffer must be full and last_values must have shape (N,)")
        next_advantage = np.zeros(self.num_envs, np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_value = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)
            delta = self.rewards[step] + gamma * next_value * nonterminal - self.values[step]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.values
        if not np.isfinite(self.advantages).all() or not np.isfinite(self.returns).all():
            raise FloatingPointError("non-finite GAE result")

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0
