"""Rollout storage and recurrent chunking for role-shared 4v3 HAPPO."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceChunk:
    env_index: int
    start: int
    stop: int


class RoleSharedRolloutBuffer4v3:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        state_dim: int,
        *,
        recurrent: bool = False,
        recurrent_hidden_dim: int = 128,
    ) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.team_size = 4
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.recurrent = bool(recurrent)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        if self.rollout_steps <= 0 or self.num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        shape = (self.rollout_steps, self.num_envs)
        self.observations = np.zeros((*shape, 4, self.obs_dim), np.float32)
        self.global_states = np.zeros((*shape, self.state_dim), np.float32)
        self.actions = np.zeros((*shape, 4, 3), np.float32)
        self.log_probs = np.zeros((*shape, 4), np.float32)
        self.agent_alive_masks = np.zeros((*shape, 4), np.float32)
        self.hidden_reset_masks = np.zeros((*shape, 4), np.float32)
        self.team_rewards = np.zeros(shape, np.float32)
        self.team_values = np.zeros(shape, np.float32)
        self.dones = np.zeros(shape, bool)
        self.advantages = np.zeros(shape, np.float32)
        self.returns = np.zeros(shape, np.float32)
        self.support_hidden_before = (
            np.zeros((*shape, self.recurrent_hidden_dim), np.float32) if self.recurrent else None
        )
        self.combat_hidden_before = (
            np.zeros((*shape, 3, self.recurrent_hidden_dim), np.float32) if self.recurrent else None
        )
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        alive_masks: np.ndarray,
        hidden_reset_masks: np.ndarray,
        team_rewards: np.ndarray,
        team_values: np.ndarray,
        dones: np.ndarray,
        *,
        support_hidden_before: np.ndarray | None = None,
        combat_hidden_before: np.ndarray | None = None,
    ) -> None:
        if self.position >= self.rollout_steps:
            raise RuntimeError("buffer is full")
        i = self.position
        self.observations[i] = observations
        self.global_states[i] = global_states
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.agent_alive_masks[i] = alive_masks
        self.hidden_reset_masks[i] = hidden_reset_masks
        self.team_rewards[i] = team_rewards
        self.team_values[i] = team_values
        self.dones[i] = dones
        if self.recurrent:
            if support_hidden_before is None or combat_hidden_before is None:
                raise ValueError("recurrent buffer requires hidden state before action")
            self.support_hidden_before[i] = support_hidden_before
            self.combat_hidden_before[i] = combat_hidden_before
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        if self.position != self.rollout_steps:
            raise ValueError("buffer must be full before GAE")
        next_advantage = np.zeros(self.num_envs, np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_value = last_values if step == self.rollout_steps - 1 else self.team_values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)
            delta = self.team_rewards[step] + float(gamma) * next_value * nonterminal - self.team_values[step]
            next_advantage = delta + float(gamma) * float(gae_lambda) * nonterminal * next_advantage
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.team_values
        if not np.isfinite(self.advantages).all() or not np.isfinite(self.returns).all():
            raise FloatingPointError("non-finite role-shared GAE")

    def sequence_chunks(self, chunk_length: int) -> list[SequenceChunk]:
        """Split each environment timeline at episode ends, then into chunks."""
        length = int(chunk_length)
        if length <= 0:
            raise ValueError("sequence_chunk_length must be positive")
        chunks: list[SequenceChunk] = []
        for env_index in range(self.num_envs):
            segment_start = 0
            for step in range(self.rollout_steps):
                if self.dones[step, env_index]:
                    segment_stop = step + 1
                    for start in range(segment_start, segment_stop, length):
                        chunks.append(SequenceChunk(env_index, start, min(start + length, segment_stop)))
                    segment_start = segment_stop
            if segment_start < self.rollout_steps:
                for start in range(segment_start, self.rollout_steps, length):
                    chunks.append(SequenceChunk(env_index, start, min(start + length, self.rollout_steps)))
        return chunks

    def padded_chunk_batch(self, chunks: list[SequenceChunk], chunk_length: int) -> dict[str, np.ndarray]:
        if not chunks:
            raise ValueError("chunk batch must not be empty")
        batch = len(chunks)
        length = int(chunk_length)
        out = {
            "observations": np.zeros((batch, length, 4, self.obs_dim), np.float32),
            "actions": np.zeros((batch, length, 4, 3), np.float32),
            "old_log_probs": np.zeros((batch, length, 4), np.float32),
            "alive_masks": np.zeros((batch, length, 4), np.float32),
            "reset_masks": np.zeros((batch, length, 4), np.float32),
            "advantages": np.zeros((batch, length), np.float32),
            "factor_indices": np.full((batch, length), -1, np.int64),
            "valid_mask": np.zeros((batch, length), np.float32),
            "support_initial_hidden": np.zeros((batch, self.recurrent_hidden_dim), np.float32),
            "combat_initial_hidden": np.zeros((batch, 3, self.recurrent_hidden_dim), np.float32),
        }
        for row, chunk in enumerate(chunks):
            n = chunk.stop - chunk.start
            sl = slice(chunk.start, chunk.stop)
            out["observations"][row, :n] = self.observations[sl, chunk.env_index]
            out["actions"][row, :n] = self.actions[sl, chunk.env_index]
            out["old_log_probs"][row, :n] = self.log_probs[sl, chunk.env_index]
            out["alive_masks"][row, :n] = self.agent_alive_masks[sl, chunk.env_index]
            out["reset_masks"][row, :n] = self.hidden_reset_masks[sl, chunk.env_index]
            out["advantages"][row, :n] = self.advantages[sl, chunk.env_index]
            out["valid_mask"][row, :n] = 1.0
            out["factor_indices"][row, :n] = np.arange(chunk.start, chunk.stop) * self.num_envs + chunk.env_index
            out["support_initial_hidden"][row] = self.support_hidden_before[chunk.start, chunk.env_index]
            out["combat_initial_hidden"][row] = self.combat_hidden_before[chunk.start, chunk.env_index]
        return out

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0


__all__ = ["RoleSharedRolloutBuffer4v3", "SequenceChunk"]
