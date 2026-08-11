"""Per-agent reward/value/GAE storage for the v14B credit-aware variant."""
from __future__ import annotations

import numpy as np


class AgentCreditRolloutBuffer4v3:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        state_dim: int,
    ) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        if self.rollout_steps <= 0 or self.num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        shape = (self.rollout_steps, self.num_envs)
        self.observations = np.zeros((*shape, 4, self.obs_dim), np.float32)
        self.global_states = np.zeros((*shape, self.state_dim), np.float32)
        self.actions = np.zeros((*shape, 4, 3), np.float32)
        self.log_probs = np.zeros((*shape, 4), np.float32)
        self.agent_alive_masks = np.zeros((*shape, 4), np.float32)
        self.team_rewards = np.zeros(shape, np.float32)
        self.agent_rewards = np.zeros((*shape, 4), np.float32)
        self.agent_values = np.zeros((*shape, 4), np.float32)
        self.dones = np.zeros(shape, bool)
        self.advantages = np.zeros((*shape, 4), np.float32)
        self.returns = np.zeros((*shape, 4), np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        alive_masks: np.ndarray,
        team_rewards: np.ndarray,
        agent_rewards: np.ndarray,
        agent_values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self.position >= self.rollout_steps:
            raise RuntimeError("buffer is full")
        i = self.position
        self.observations[i] = observations
        self.global_states[i] = global_states
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.agent_alive_masks[i] = alive_masks
        self.team_rewards[i] = team_rewards
        self.agent_rewards[i] = agent_rewards
        self.agent_values[i] = agent_values
        self.dones[i] = dones
        self.position += 1

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> None:
        """Use episode done only; individual death never truncates reward-to-go."""
        if self.position != self.rollout_steps:
            raise ValueError("buffer must be full before GAE")
        last = np.asarray(last_values, dtype=np.float32)
        if last.shape != (self.num_envs, 4):
            raise ValueError(f"last_values must have shape ({self.num_envs}, 4)")
        next_advantage = np.zeros((self.num_envs, 4), np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_value = last if step == self.rollout_steps - 1 else self.agent_values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)[:, None]
            delta = (
                self.agent_rewards[step]
                + float(gamma) * next_value * nonterminal
                - self.agent_values[step]
            )
            next_advantage = (
                delta
                + float(gamma)
                * float(gae_lambda)
                * nonterminal
                * next_advantage
            )
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.agent_values
        if not np.isfinite(self.advantages).all() or not np.isfinite(self.returns).all():
            raise FloatingPointError("non-finite v14B per-agent GAE")

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0


def normalize_role_advantages(
    advantages: np.ndarray, alive_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize Support alone and all alive Combat slot samples together."""
    values = np.asarray(advantages, dtype=np.float32)
    masks = np.asarray(alive_masks, dtype=np.float32)
    if values.shape != masks.shape or values.shape[-1] != 4:
        raise ValueError("advantages and masks must have shape [...,4]")
    support = np.zeros_like(values[..., 0])
    support_active = masks[..., 0] > 0.5
    if support_active.any():
        selected = values[..., 0][support_active]
        support = (values[..., 0] - selected.mean()) / (selected.std() + 1e-8)
    combat = np.zeros_like(values[..., 1:4])
    combat_active = masks[..., 1:4] > 0.5
    if combat_active.any():
        selected = values[..., 1:4][combat_active]
        combat = (values[..., 1:4] - selected.mean()) / (selected.std() + 1e-8)
    support = np.where(support_active, support, 0.0).astype(np.float32)
    combat = np.where(combat_active, combat, 0.0).astype(np.float32)
    if not np.isfinite(support).all() or not np.isfinite(combat).all():
        raise FloatingPointError("non-finite normalized role advantages")
    return support, combat


__all__ = ["AgentCreditRolloutBuffer4v3", "normalize_role_advantages"]
