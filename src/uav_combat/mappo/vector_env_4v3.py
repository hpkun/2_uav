"""Vector environment adapter for functional heterogeneous red 4v3 v9."""
from __future__ import annotations

import multiprocessing
import multiprocessing as mp
import multiprocessing.connection
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..config import load_config
from ..environment_4v3 import (
    BLUE_TEAM_SIZE_4V3,
    GS_DIM_4V3,
    OBS_DIM_4V3,
    RED_REWARD_COMPONENT_KEYS_4V3,
    RED_TEAM_SIZE_4V3,
    FunctionalHeterogeneous4v3AirCombatEnv,
)
from ..scenario_4v3 import BLUE_IDS_4V3, RED_IDS_4V3


class VectorStepResult4v3(NamedTuple):
    observations: np.ndarray
    global_states: np.ndarray
    team_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    alive_masks: np.ndarray
    red_reward_components: np.ndarray
    episode_valid: np.ndarray
    episode_summaries: list[dict[str, Any] | None]


def _stack_reset(envs: list[FunctionalHeterogeneous4v3AirCombatEnv]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, states, masks = zip(*(env._observations() for env in envs))
    return np.stack(obs), np.stack(states), np.stack(masks)


def _actions_for_env(actions: np.ndarray) -> dict[str, np.ndarray]:
    if np.asarray(actions).shape != (RED_TEAM_SIZE_4V3, 3):
        raise ValueError(f"4v3 red actions must have shape (4, 3), got {np.asarray(actions).shape}")
    return {aid: np.asarray(actions[i], dtype=np.float32) for i, aid in enumerate(RED_IDS_4V3)}


def _pack_step_results(results: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]]) -> VectorStepResult4v3:
    obs, gs, masks, rewards, terms, truncs, infos = zip(*results)
    comp = np.zeros((len(results), len(RED_REWARD_COMPONENT_KEYS_4V3)), dtype=np.float32)
    valid = np.zeros(len(results), dtype=bool)
    summaries: list[dict[str, Any] | None] = []
    for i, info in enumerate(infos):
        rc = info.get("reward_components", {})
        for j, key in enumerate(RED_REWARD_COMPONENT_KEYS_4V3):
            comp[i, j] = float(rc.get(key, 0.0))
        summary = info.get("episode_summary")
        summaries.append(summary)
        valid[i] = summary is not None
    return VectorStepResult4v3(
        observations=np.stack(obs).astype(np.float32),
        global_states=np.stack(gs).astype(np.float32),
        team_rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terms, dtype=bool),
        truncated=np.asarray(truncs, dtype=bool),
        alive_masks=np.stack(masks).astype(np.float32),
        red_reward_components=comp,
        episode_valid=valid,
        episode_summaries=summaries,
    )


class LocalCombatVectorEnv4v3:
    def __init__(self, config_path: str | Path, num_envs: int, seed: int = 0) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.config_path = str(config_path)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.envs = [FunctionalHeterogeneous4v3AirCombatEnv(self.config_path) for _ in range(self.num_envs)]
        for i, env in enumerate(self.envs):
            env.reset(self.seed + i)

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for i, env in enumerate(self.envs):
            env.reset(self.seed + i)
        return _stack_reset(self.envs)

    def reset_at(self, index: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.envs[int(index)].reset(int(seed))
        return self.envs[int(index)]._observations()

    def step(self, actions: np.ndarray) -> VectorStepResult4v3:
        if np.asarray(actions).shape != (self.num_envs, RED_TEAM_SIZE_4V3, 3):
            raise ValueError(f"batched 4v3 red actions must have shape ({self.num_envs}, 4, 3)")
        results = [env.step(_actions_for_env(actions[i])) for i, env in enumerate(self.envs)]
        return _pack_step_results(results)

    def policy_modes(self) -> dict[str, list[str]]:
        cfg = load_config(self.config_path)
        blue_mode = cfg.get("blue_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9")
        red_mode = cfg.get("red_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9")
        mapping = cfg.get("action", {}).get("mapping_mode", "legacy_delta")
        return {
            "blue": [blue_mode for _ in self.envs],
            "red": [red_mode for _ in self.envs],
            "blue_action_mapping": [mapping for _ in self.envs],
            "red_action_mapping": [mapping for _ in self.envs],
        }

    def close(self) -> None:
        return None


def _worker(conn: multiprocessing.connection.Connection, config_path: str, seed: int) -> None:
    env = FunctionalHeterogeneous4v3AirCombatEnv(config_path)
    env.reset(seed)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(env.reset(seed))
            elif cmd == "reset_at":
                conn.send(env.reset(int(payload)))
            elif cmd == "step":
                conn.send(env.step(_actions_for_env(payload)))
            elif cmd == "policy_modes":
                cfg = load_config(config_path)
                mapping = cfg.get("action", {}).get("mapping_mode", "legacy_delta")
                conn.send({
                    "blue": cfg.get("blue_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9"),
                    "red": cfg.get("red_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9"),
                    "blue_action_mapping": mapping,
                    "red_action_mapping": mapping,
                })
            elif cmd == "close":
                conn.close()
                return
            else:
                raise ValueError(f"unknown worker command {cmd!r}")
    except BaseException as exc:  # pragma: no cover - defensive worker bridge
        conn.send(("error", "".join(traceback.format_exception(exc))))


class SubprocessCombatVectorEnv4v3:
    def __init__(self, config_path: str | Path, num_envs: int, num_workers: int, seed: int = 0) -> None:
        if int(num_workers) <= 0:
            raise ValueError("num_workers must be positive")
        self.config_path = str(config_path)
        self.num_envs = int(num_envs)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self._ctx = mp.get_context("spawn")
        self._parents: list[multiprocessing.connection.Connection] = []
        self._processes: list[mp.Process] = []
        for i in range(self.num_envs):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(target=_worker, args=(child, self.config_path, self.seed + i), daemon=True)
            proc.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(proc)

    def _recv_all(self) -> list[Any]:
        out = [p.recv() for p in self._parents]
        for item in out:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
                raise RuntimeError(item[1])
        return out

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for p in self._parents:
            p.send(("reset", None))
        obs, states, masks = zip(*self._recv_all())
        return np.stack(obs), np.stack(states), np.stack(masks)

    def reset_at(self, index: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = int(index)
        self._parents[idx].send(("reset_at", int(seed)))
        return self._parents[idx].recv()

    def step(self, actions: np.ndarray) -> VectorStepResult4v3:
        if np.asarray(actions).shape != (self.num_envs, RED_TEAM_SIZE_4V3, 3):
            raise ValueError(f"batched 4v3 red actions must have shape ({self.num_envs}, 4, 3)")
        for i, p in enumerate(self._parents):
            p.send(("step", np.asarray(actions[i], dtype=np.float32)))
        return _pack_step_results(self._recv_all())

    def policy_modes(self) -> dict[str, list[str]]:
        for p in self._parents:
            p.send(("policy_modes", None))
        rows = self._recv_all()
        return {
            "blue": [r["blue"] for r in rows],
            "red": [r["red"] for r in rows],
            "blue_action_mapping": [r["blue_action_mapping"] for r in rows],
            "red_action_mapping": [r["red_action_mapping"] for r in rows],
        }

    def close(self) -> None:
        for p in self._parents:
            try:
                p.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._processes:
            proc.join(timeout=1.0)


def make_combat_vector_env_4v3(config_path: str | Path, num_envs: int, num_env_workers: int = 0, seed: int = 0):
    if int(num_env_workers) <= 0:
        return LocalCombatVectorEnv4v3(config_path, num_envs, seed)
    return SubprocessCombatVectorEnv4v3(config_path, num_envs, num_env_workers, seed)


__all__ = [
    "BLUE_TEAM_SIZE_4V3",
    "GS_DIM_4V3",
    "OBS_DIM_4V3",
    "RED_REWARD_COMPONENT_KEYS_4V3",
    "RED_TEAM_SIZE_4V3",
    "VectorStepResult4v3",
    "make_combat_vector_env_4v3",
]
