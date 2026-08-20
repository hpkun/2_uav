"""Synchronous vector adapter for the heterogeneous MAV/UAV environment."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..mavuav import HeterogeneousMAVUAVAirCombatEnv


class MAVUAVVectorEnv:
    def __init__(self, num_envs: int, config_path: str | Path = "configs/heterogeneous_mavuav_3v2.yaml") -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.envs = [HeterogeneousMAVUAVAirCombatEnv(config_path) for _ in range(self.num_envs)]
        self.agent_ids = HeterogeneousMAVUAVAirCombatEnv.red_ids

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict]]:
        results = [env.reset(None if seed is None else seed+i) for i, env in enumerate(self.envs)]
        return self._stack_observations([result[0] for result in results]), [result[1] for result in results]

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        values = np.asarray(actions, dtype=float)
        expected = (self.num_envs, len(self.agent_ids), 3)
        if values.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {values.shape}")
        results = [env.step(values[i]) for i, env in enumerate(self.envs)]
        observations = self._stack_observations([result[0] for result in results])
        rewards = np.asarray([[result[1][aid] for aid in self.agent_ids] for result in results], dtype=np.float32)
        terminated = np.asarray([result[2] for result in results], dtype=bool)
        truncated = np.asarray([result[3] for result in results], dtype=bool)
        return observations, rewards, terminated, truncated, [result[4] for result in results]

    def _stack_observations(self, observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.asarray([[obs[aid] for aid in self.agent_ids] for obs in observations], dtype=np.float32)
