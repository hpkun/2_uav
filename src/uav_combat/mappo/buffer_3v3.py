"""MAPPO buffer for 3v3 fixed-blue training – shared red-actor, team critic."""
import numpy as np


class MAPPOBuffer3v3:
    """Fixed-length rollout buffer: [T, N, 3, ...] for three red agents."""

    def __init__(
        self, rollout_steps: int, num_envs: int,
        obs_dim: int = 68, act_dim: int = 3, state_dim: int = 48,
    ) -> None:
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.observations = np.zeros((rollout_steps, num_envs, 3, obs_dim), np.float32)
        self.global_states = np.zeros((rollout_steps, num_envs, state_dim), np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, 3, act_dim), np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs, 3), np.float32)
        self.agent_alive_masks = np.zeros((rollout_steps, num_envs, 3), np.float32)
        self.team_rewards = np.zeros((rollout_steps, num_envs), np.float32)
        self.team_values = np.zeros((rollout_steps, num_envs), np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), bool)
        self.advantages = np.zeros((rollout_steps, num_envs), np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,       # [N, 3, 68]
        global_states: np.ndarray,       # [N, 48]
        actions: np.ndarray,             # [N, 3, 3]
        log_probs: np.ndarray,           # [N, 3]
        agent_alive_masks: np.ndarray,   # [N, 3]
        team_rewards: np.ndarray,        # [N]
        team_values: np.ndarray,         # [N]
        dones: np.ndarray,               # [N] bool
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

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float,
    ) -> None:
        """Team-level GAE per environment.

        last_values : [N]  – team value for next (unseen) state.
        """
        if self.position != self.rollout_steps:
            raise ValueError("buffer must be full")
        if np.asarray(last_values).shape != (self.num_envs,):
            raise ValueError(f"last_values shape must be ({self.num_envs},)")

        next_advantage = np.zeros(self.num_envs, np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_value = last_values
            else:
                next_value = self.team_values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)
            delta = self.team_rewards[step] + gamma * next_value * nonterminal - self.team_values[step]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.team_values
        if not np.isfinite(self.advantages).all() or not np.isfinite(self.returns).all():
            raise FloatingPointError("non-finite GAE result")

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0
