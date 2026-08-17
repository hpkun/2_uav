"""Recurrent extension of the existing role-local 4v3 credit buffer."""
from __future__ import annotations

import numpy as np

from .role_credit_buffer import AgentCreditRolloutBuffer4v3
from .role_shared_buffer import SequenceChunk


class RecurrentAgentCreditRolloutBuffer4v3(AgentCreditRolloutBuffer4v3):
    """Combine role-local GAE with the existing recurrent sequence contract."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        state_dim: int,
        recurrent_hidden_dim: int = 128,
    ) -> None:
        super().__init__(rollout_steps, num_envs, obs_dim, state_dim)
        shape = (self.rollout_steps, self.num_envs)
        self.recurrent = True
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.hidden_reset_masks = np.zeros((*shape, 4), np.float32)
        self.support_hidden_before = np.zeros(
            (*shape, self.recurrent_hidden_dim), np.float32
        )
        self.combat_hidden_before = np.zeros(
            (*shape, 3, self.recurrent_hidden_dim), np.float32
        )

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        alive_masks: np.ndarray,
        hidden_reset_masks: np.ndarray,
        team_rewards: np.ndarray,
        agent_rewards: np.ndarray,
        agent_values: np.ndarray,
        dones: np.ndarray,
        *,
        support_hidden_before: np.ndarray,
        combat_hidden_before: np.ndarray,
    ) -> None:
        index = self.position
        super().add(
            observations,
            global_states,
            actions,
            log_probs,
            alive_masks,
            team_rewards,
            agent_rewards,
            agent_values,
            dones,
        )
        self.hidden_reset_masks[index] = hidden_reset_masks
        self.support_hidden_before[index] = support_hidden_before
        self.combat_hidden_before[index] = combat_hidden_before

    def sequence_chunks(self, chunk_length: int) -> list[SequenceChunk]:
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
                        chunks.append(
                            SequenceChunk(
                                env_index, start, min(start + length, segment_stop)
                            )
                        )
                    segment_start = segment_stop
            for start in range(segment_start, self.rollout_steps, length):
                chunks.append(
                    SequenceChunk(
                        env_index, start, min(start + length, self.rollout_steps)
                    )
                )
        return chunks

    def padded_chunk_batch(
        self, chunks: list[SequenceChunk], chunk_length: int
    ) -> dict[str, np.ndarray]:
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
            "advantages": np.zeros((batch, length, 4), np.float32),
            "factor_indices": np.full((batch, length), -1, np.int64),
            "valid_mask": np.zeros((batch, length), np.float32),
            "support_initial_hidden": np.zeros(
                (batch, self.recurrent_hidden_dim), np.float32
            ),
            "combat_initial_hidden": np.zeros(
                (batch, 3, self.recurrent_hidden_dim), np.float32
            ),
        }
        for row, chunk in enumerate(chunks):
            count = chunk.stop - chunk.start
            source = slice(chunk.start, chunk.stop)
            out["observations"][row, :count] = self.observations[
                source, chunk.env_index
            ]
            out["actions"][row, :count] = self.actions[source, chunk.env_index]
            out["old_log_probs"][row, :count] = self.log_probs[
                source, chunk.env_index
            ]
            out["alive_masks"][row, :count] = self.agent_alive_masks[
                source, chunk.env_index
            ]
            out["reset_masks"][row, :count] = self.hidden_reset_masks[
                source, chunk.env_index
            ]
            out["advantages"][row, :count] = self.advantages[
                source, chunk.env_index
            ]
            out["valid_mask"][row, :count] = 1.0
            out["factor_indices"][row, :count] = (
                np.arange(chunk.start, chunk.stop) * self.num_envs + chunk.env_index
            )
            out["support_initial_hidden"][row] = self.support_hidden_before[
                chunk.start, chunk.env_index
            ]
            out["combat_initial_hidden"][row] = self.combat_hidden_before[
                chunk.start, chunk.env_index
            ]
        return out


__all__ = ["RecurrentAgentCreditRolloutBuffer4v3"]
