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


def _terminal_hold_result(env: FunctionalHeterogeneous4v3AirCombatEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
    """Keep a completed slot stable while an evaluation batch drains."""
    obs, states, masks = env._observations()
    return (
        obs,
        states,
        masks,
        0.0,
        False,
        False,
        {"reward_components": {key: 0.0 for key in RED_REWARD_COMPONENT_KEYS_4V3}, "episode_summary": None},
    )


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

    def reset_at(self, index: int | np.ndarray, seed: int | list[int] | np.ndarray):
        scalar = np.asarray(index).ndim == 0
        indices = np.atleast_1d(np.asarray(index, dtype=np.int32))
        seeds = np.atleast_1d(np.asarray(seed, dtype=np.int64))
        if len(indices) != len(seeds):
            raise ValueError("reset_at indices and seeds must have the same length")
        for idx, item_seed in zip(indices, seeds):
            self.envs[int(idx)].reset(int(item_seed))
        values = [self.envs[int(idx)]._observations() for idx in indices]
        result = tuple(np.stack([v[i] for v in values]) for i in range(3))
        return tuple(value[0] for value in result) if scalar else result

    def step(self, actions: np.ndarray) -> VectorStepResult4v3:
        if np.asarray(actions).shape != (self.num_envs, RED_TEAM_SIZE_4V3, 3):
            raise ValueError(f"batched 4v3 red actions must have shape ({self.num_envs}, 4, 3)")
        results = [
            env.step(_actions_for_env(actions[i])) if env._running else _terminal_hold_result(env)
            for i, env in enumerate(self.envs)
        ]
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

    def state_dict(self) -> dict[str, Any]:
        return {"envs": [env.state_dict() for env in self.envs]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for env, saved in zip(self.envs, state["envs"]):
            env.load_state_dict(saved)

    def close(self) -> None:
        return None


def _worker(conn: multiprocessing.connection.Connection, config_path: str, envs_per_worker: int, worker_index: int, seed: int) -> None:
    try:
        envs = [FunctionalHeterogeneous4v3AirCombatEnv(config_path) for _ in range(envs_per_worker)]
        for i, env in enumerate(envs):
            env.reset(seed + i)
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                values = [env.reset(seed + i) for i, env in enumerate(envs)]
                conn.send(tuple(np.stack([v[j] for v in values]) for j in range(3)))
            elif cmd == "reset_at":
                indices, seeds = payload
                values = []
                for idx, item_seed in zip(indices, seeds):
                    values.append(envs[int(idx)].reset(int(item_seed)))
                conn.send(tuple(np.stack([v[j] for v in values]) for j in range(3)))
            elif cmd == "step":
                results = [
                    envs[i].step(_actions_for_env(payload[i])) if envs[i]._running else _terminal_hold_result(envs[i])
                    for i in range(envs_per_worker)
                ]
                conn.send(_pack_step_results(results))
            elif cmd == "policy_modes":
                cfg = load_config(config_path)
                mapping = cfg.get("action", {}).get("mapping_mode", "legacy_delta")
                conn.send({
                    "blue": [cfg.get("blue_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9")] * envs_per_worker,
                    "red": [cfg.get("red_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9")] * envs_per_worker,
                    "blue_action_mapping": [mapping] * envs_per_worker,
                    "red_action_mapping": [mapping] * envs_per_worker,
                })
            elif cmd == "state_dict":
                conn.send({"envs": [env.state_dict() for env in envs]})
            elif cmd == "load_state_dict":
                for env, saved in zip(envs, payload["envs"]):
                    env.load_state_dict(saved)
                conn.send(True)
            elif cmd == "close":
                conn.close()
                return
            else:
                raise ValueError(f"unknown worker command {cmd!r}")
    except BaseException:
        conn.send(("error", worker_index, traceback.format_exc()))


class SubprocessCombatVectorEnv4v3:
    def __init__(self, config_path: str | Path, num_envs: int, num_workers: int, seed: int = 0) -> None:
        if int(num_workers) <= 0 or int(num_workers) > int(num_envs):
            raise ValueError("num_workers must be in [1, num_envs]")
        if int(num_envs) % int(num_workers) != 0:
            raise ValueError("num_envs must be divisible by num_workers")
        self.config_path = str(config_path)
        self.num_envs = int(num_envs)
        self.num_workers = int(num_workers)
        self.num_env_workers = self.num_workers
        self.envs_per_worker = self.num_envs // self.num_workers
        self.seed = int(seed)
        self._ctx = mp.get_context("spawn")
        self._parents: list[multiprocessing.connection.Connection] = []
        self._processes: list[mp.Process] = []
        for worker_index in range(self.num_workers):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_worker,
                args=(child, self.config_path, self.envs_per_worker, worker_index, self.seed + worker_index * self.envs_per_worker),
                daemon=True,
                name=f"heterogeneous-4v3-worker-{worker_index}",
            )
            proc.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(proc)
        self._closed = False

    def _recv_all(self) -> list[Any]:
        out = [p.recv() for p in self._parents]
        for worker_index, item in enumerate(out):
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str) and item[0] == "error":
                raise RuntimeError(f"Worker {item[1]} error:\n{item[2]}")
        return out

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("vector environment is closed")

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_open()
        for parent in self._parents:
            parent.send(("reset", None))
        parts = self._recv_all()
        return tuple(np.concatenate([part[i] for part in parts], axis=0) for i in range(3))

    def reset_at(self, index: int | np.ndarray, seed: int | list[int] | np.ndarray):
        self._check_open()
        scalar = np.asarray(index).ndim == 0
        indices = np.atleast_1d(np.asarray(index, dtype=np.int32))
        seeds = np.atleast_1d(np.asarray(seed, dtype=np.int64))
        if len(indices) != len(seeds):
            raise ValueError("reset_at indices and seeds must have the same length")
        grouped: dict[int, tuple[list[int], list[int]]] = {}
        for global_index, item_seed in zip(indices, seeds):
            worker = int(global_index) // self.envs_per_worker
            local = int(global_index) % self.envs_per_worker
            grouped.setdefault(worker, ([], []))[0].append(local)
            grouped[worker][1].append(int(item_seed))
        for worker, (local_indices, local_seeds) in grouped.items():
            self._parents[worker].send(("reset_at", (np.asarray(local_indices, dtype=np.int32), local_seeds)))
        received: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for worker in grouped:
            item = self._parents[worker].recv()
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str) and item[0] == "error":
                raise RuntimeError(f"Worker {item[1]} error:\n{item[2]}")
            local_indices = grouped[worker][0]
            for j, local_index in enumerate(local_indices):
                received[worker * self.envs_per_worker + local_index] = tuple(item[i][j:j + 1] for i in range(3))
        result = tuple(np.concatenate([received[int(global_index)][i] for global_index in indices], axis=0) for i in range(3))
        return tuple(value[0] for value in result) if scalar else result

    def step(self, actions: np.ndarray) -> VectorStepResult4v3:
        self._check_open()
        if np.asarray(actions).shape != (self.num_envs, RED_TEAM_SIZE_4V3, 3):
            raise ValueError(f"batched 4v3 red actions must have shape ({self.num_envs}, 4, 3)")
        for worker, parent in enumerate(self._parents):
            start = worker * self.envs_per_worker
            parent.send(("step", np.asarray(actions[start:start + self.envs_per_worker], dtype=np.float32)))
        parts = self._recv_all()
        fields = VectorStepResult4v3._fields
        return VectorStepResult4v3(**{
            field: np.concatenate([getattr(part, field) for part in parts], axis=0)
            if field != "episode_summaries"
            else [summary for part in parts for summary in part.episode_summaries]
            for field in fields
        })

    def policy_modes(self) -> dict[str, list[str]]:
        self._check_open()
        for parent in self._parents:
            parent.send(("policy_modes", None))
        parts = self._recv_all()
        return {key: [value for part in parts for value in part[key]] for key in parts[0]}

    def state_dict(self) -> dict[str, Any]:
        self._check_open()
        for parent in self._parents:
            parent.send(("state_dict", None))
        parts = self._recv_all()
        return {"envs": [env for part in parts for env in part["envs"]]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._check_open()
        for worker, parent in enumerate(self._parents):
            start = worker * self.envs_per_worker
            parent.send(("load_state_dict", {"envs": state["envs"][start:start + self.envs_per_worker]}))
        self._recv_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self._processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        for parent in self._parents:
            parent.close()


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
