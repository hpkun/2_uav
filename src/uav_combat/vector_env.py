"""Simple synchronous auto-reset vector environment for MARL rollouts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np

from .mavuav import HeterogeneousMAVUAVAirCombatEnv, RED_IDS


class MAVUAVVectorEnv:
    """Run independent environments sequentially in one Python process."""

    def __init__(self, num_envs: int, config_path: str | Path | Mapping[str, Any] | None = None, *, seed: int | None = None, blue_target_mode: str | None = None, randomize: bool | None = None) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.agent_ids = RED_IDS
        self.base_seed = seed
        self.reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.envs = [HeterogeneousMAVUAVAirCombatEnv(config_path, seed=None if seed is None else seed + i, blue_target_mode=blue_target_mode, randomize=randomize) for i in range(self.num_envs)]

    def _seed(self, index: int) -> int | None:
        if self.base_seed is None:
            return None
        return int(self.base_seed + index + 1000003 * self.reset_counts[index])

    @staticmethod
    def _stack_observations(observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.asarray([[obs[aid] for aid in RED_IDS] for obs in observations], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if seed is not None:
            self.base_seed = int(seed)
        self.reset_counts.fill(0)
        results = [env.reset(seed=self._seed(index)) for index, env in enumerate(self.envs)]
        observations = self._stack_observations([item[0] for item in results])
        states = np.asarray([env.global_state() for env in self.envs], dtype=np.float32)
        masks = np.asarray([env.active_masks for env in self.envs], dtype=np.float32)
        return observations, states, masks, [item[1] for item in results]

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        values = np.asarray(actions, dtype=np.float64)
        expected = (self.num_envs, len(RED_IDS), 3)
        if values.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {values.shape}")
        observations: list[dict[str, np.ndarray]] = []
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        rewards = np.empty((self.num_envs, len(RED_IDS)), dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for index, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(values[index])
            rewards[index] = [reward[aid] for aid in RED_IDS]
            terminated[index], truncated[index] = term, trunc
            if term or trunc:
                terminal_state = env.global_state().copy()
                terminal_masks = env.active_masks.copy()
                self.reset_counts[index] += 1
                obs, reset_info = env.reset(seed=self._seed(index))
                info = dict(info)
                info["terminal_global_state"] = terminal_state
                info["terminal_active_masks"] = terminal_masks
                info["reset_info"] = reset_info
                info["auto_reset"] = True
            else:
                info = dict(info)
                info["auto_reset"] = False
            observations.append(obs)
            states.append(env.global_state())
            masks.append(env.active_masks)
            infos.append(info)
        return self._stack_observations(observations), np.asarray(states, dtype=np.float32), rewards, terminated, truncated, np.asarray(masks, dtype=np.float32), infos

    def close(self) -> None:
        return None

