"""Vector adapter for the v14 mission-aligned 4v3 environment."""
from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..config import load_config
from ..environment_4v3_v14 import (
    AGENT_REWARD_COMPONENT_KEYS_V14,
    GS_DIM_V14,
    OBS_DIM_V14,
    RED_TEAM_SIZE_V14,
    REWARD_COMPONENT_KEYS_V14,
    FunctionalHeterogeneous4v3V14MissionAlignedEnv,
)
from ..scenario_4v3_v14 import RED_IDS_V14


class VectorStepResult4v3V14(NamedTuple):
    observations: np.ndarray
    global_states: np.ndarray
    team_rewards: np.ndarray
    agent_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    alive_masks: np.ndarray
    red_reward_components: np.ndarray
    red_agent_reward_components: np.ndarray
    episode_valid: np.ndarray
    episode_summaries: list[dict[str, Any] | None]


def _actions_for_env(actions: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(actions)
    if values.shape != (RED_TEAM_SIZE_V14, 3):
        raise ValueError(f"v14 red actions must have shape (4, 3), got {values.shape}")
    return {
        agent_id: np.asarray(values[index], dtype=np.float32)
        for index, agent_id in enumerate(RED_IDS_V14)
    }


def _hold(env: FunctionalHeterogeneous4v3V14MissionAlignedEnv):
    obs, state, masks = env._observations()
    return obs, state, masks, 0.0, False, False, {
        "reward_components": {key: 0.0 for key in REWARD_COMPONENT_KEYS_V14},
        "agent_rewards": {agent_id: 0.0 for agent_id in RED_IDS_V14},
        "agent_reward_components": {
            agent_id: {key: 0.0 for key in AGENT_REWARD_COMPONENT_KEYS_V14}
            for agent_id in RED_IDS_V14
        },
        "episode_summary": None,
    }


def _pack(results: list[tuple[Any, ...]]) -> VectorStepResult4v3V14:
    obs, states, masks, rewards, terms, truncs, infos = zip(*results)
    team_components = np.zeros(
        (len(results), len(REWARD_COMPONENT_KEYS_V14)), dtype=np.float32
    )
    agent_rewards = np.zeros((len(results), 4), dtype=np.float32)
    agent_components = np.zeros(
        (len(results), 4, len(AGENT_REWARD_COMPONENT_KEYS_V14)), dtype=np.float32
    )
    summaries: list[dict[str, Any] | None] = []
    valid = np.zeros(len(results), dtype=bool)
    for row, info in enumerate(infos):
        for column, key in enumerate(REWARD_COMPONENT_KEYS_V14):
            team_components[row, column] = float(
                info.get("reward_components", {}).get(key, 0.0)
            )
        for slot, agent_id in enumerate(RED_IDS_V14):
            agent_rewards[row, slot] = float(
                info.get("agent_rewards", {}).get(agent_id, 0.0)
            )
            values = info.get("agent_reward_components", {}).get(agent_id, {})
            for column, key in enumerate(AGENT_REWARD_COMPONENT_KEYS_V14):
                agent_components[row, slot, column] = float(values.get(key, 0.0))
        summary = info.get("episode_summary")
        summaries.append(summary)
        valid[row] = summary is not None
    result = VectorStepResult4v3V14(
        np.stack(obs).astype(np.float32),
        np.stack(states).astype(np.float32),
        np.asarray(rewards, dtype=np.float32),
        agent_rewards,
        np.asarray(terms, dtype=bool),
        np.asarray(truncs, dtype=bool),
        np.stack(masks).astype(np.float32),
        team_components,
        agent_components,
        valid,
        summaries,
    )
    numeric = (
        result.observations,
        result.global_states,
        result.team_rewards,
        result.agent_rewards,
        result.red_reward_components,
        result.red_agent_reward_components,
    )
    if not all(np.isfinite(value).all() for value in numeric):
        raise FloatingPointError("v14 vector result must be finite")
    return result


class LocalCombatVectorEnv4v3V14:
    def __init__(self, config_path: str | Path, num_envs: int, seed: int = 0) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.config_path = str(config_path)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.envs = [
            FunctionalHeterogeneous4v3V14MissionAlignedEnv(config_path)
            for _ in range(self.num_envs)
        ]
        self.reset()

    def reset(self):
        values = [env.reset(self.seed + index) for index, env in enumerate(self.envs)]
        return tuple(np.stack([value[column] for value in values]) for column in range(3))

    def reset_at(self, index, seed):
        scalar = np.asarray(index).ndim == 0
        indices = np.atleast_1d(np.asarray(index, dtype=np.int32))
        seeds = np.atleast_1d(np.asarray(seed, dtype=np.int64))
        if len(indices) != len(seeds):
            raise ValueError("reset_at indices and seeds must have the same length")
        values = [
            self.envs[int(item)].reset(int(item_seed))
            for item, item_seed in zip(indices, seeds)
        ]
        output = tuple(
            np.stack([value[column] for value in values]) for column in range(3)
        )
        return tuple(value[0] for value in output) if scalar else output

    def step(self, actions: np.ndarray) -> VectorStepResult4v3V14:
        values = np.asarray(actions)
        if values.shape != (self.num_envs, RED_TEAM_SIZE_V14, 3):
            raise ValueError(
                f"v14 batched actions must have shape ({self.num_envs}, 4, 3)"
            )
        return _pack(
            [
                env.step(_actions_for_env(values[index]))
                if env._running
                else _hold(env)
                for index, env in enumerate(self.envs)
            ]
        )

    def policy_modes(self):
        config = load_config(self.config_path)
        return {
            "red": [config.get("red_rule_policy", {}).get("mode", "v12_rule")]
            * self.num_envs,
            "blue": [config.get("blue_rule_policy", {}).get("mode", "v12_rule")]
            * self.num_envs,
        }

    def state_dict(self):
        return {"envs": [env.state_dict() for env in self.envs]}

    def load_state_dict(self, state):
        for env, saved in zip(self.envs, state["envs"]):
            env.load_state_dict(saved)

    def close(self):
        return None


def _worker(conn, config_path: str, envs_per_worker: int, worker_index: int, seed: int):
    try:
        envs = [
            FunctionalHeterogeneous4v3V14MissionAlignedEnv(config_path)
            for _ in range(envs_per_worker)
        ]
        for index, env in enumerate(envs):
            env.reset(seed + index)
        while True:
            command, payload = conn.recv()
            if command == "reset":
                values = [env.reset(seed + index) for index, env in enumerate(envs)]
                conn.send(
                    tuple(
                        np.stack([value[column] for value in values])
                        for column in range(3)
                    )
                )
            elif command == "reset_at":
                indices, seeds = payload
                values = [
                    envs[int(index)].reset(int(item_seed))
                    for index, item_seed in zip(indices, seeds)
                ]
                conn.send(
                    tuple(
                        np.stack([value[column] for value in values])
                        for column in range(3)
                    )
                )
            elif command == "step":
                conn.send(
                    _pack(
                        [
                            envs[index].step(_actions_for_env(payload[index]))
                            if envs[index]._running
                            else _hold(envs[index])
                            for index in range(envs_per_worker)
                        ]
                    )
                )
            elif command == "policy_modes":
                config = load_config(config_path)
                conn.send(
                    {
                        "red": [
                            config.get("red_rule_policy", {}).get("mode", "v12_rule")
                        ]
                        * envs_per_worker,
                        "blue": [
                            config.get("blue_rule_policy", {}).get("mode", "v12_rule")
                        ]
                        * envs_per_worker,
                    }
                )
            elif command == "state_dict":
                conn.send({"envs": [env.state_dict() for env in envs]})
            elif command == "load_state_dict":
                for env, saved in zip(envs, payload["envs"]):
                    env.load_state_dict(saved)
                conn.send(True)
            elif command == "close":
                conn.close()
                return
            else:
                raise ValueError(f"unknown worker command {command!r}")
    except BaseException:
        conn.send(("error", worker_index, traceback.format_exc()))


class SubprocessCombatVectorEnv4v3V14:
    def __init__(
        self,
        config_path: str | Path,
        num_envs: int,
        num_workers: int,
        seed: int = 0,
    ) -> None:
        if (
            int(num_workers) <= 0
            or int(num_workers) > int(num_envs)
            or int(num_envs) % int(num_workers) != 0
        ):
            raise ValueError("v14 num_workers must divide num_envs and be positive")
        self.config_path = str(config_path)
        self.num_envs = int(num_envs)
        self.num_workers = int(num_workers)
        self.num_env_workers = self.num_workers
        self.envs_per_worker = self.num_envs // self.num_workers
        self.seed = int(seed)
        context = mp.get_context("spawn")
        self._parents = []
        self._processes = []
        for worker_index in range(self.num_workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    self.config_path,
                    self.envs_per_worker,
                    worker_index,
                    self.seed + worker_index * self.envs_per_worker,
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(process)
        self._closed = False

    def _recv(self):
        values = [parent.recv() for parent in self._parents]
        for value in values:
            if (
                isinstance(value, tuple)
                and len(value) == 3
                and isinstance(value[0], str)
                and value[0] == "error"
            ):
                raise RuntimeError(f"worker {value[1]} error:\n{value[2]}")
        return values

    def reset(self):
        for parent in self._parents:
            parent.send(("reset", None))
        parts = self._recv()
        return tuple(
            np.concatenate([part[column] for part in parts], axis=0)
            for column in range(3)
        )

    def reset_at(self, index, seed):
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
        for worker, payload in grouped.items():
            self._parents[worker].send(
                ("reset_at", (np.asarray(payload[0], dtype=np.int32), payload[1]))
            )
        received = {}
        for worker, payload in grouped.items():
            result = self._parents[worker].recv()
            for position, local in enumerate(payload[0]):
                received[worker * self.envs_per_worker + local] = tuple(
                    result[column][position : position + 1] for column in range(3)
                )
        output = tuple(
            np.concatenate(
                [received[int(item)][column] for item in indices], axis=0
            )
            for column in range(3)
        )
        return tuple(value[0] for value in output) if scalar else output

    def step(self, actions: np.ndarray) -> VectorStepResult4v3V14:
        values = np.asarray(actions)
        if values.shape != (self.num_envs, RED_TEAM_SIZE_V14, 3):
            raise ValueError(
                f"v14 batched actions must have shape ({self.num_envs}, 4, 3)"
            )
        for worker, parent in enumerate(self._parents):
            start = worker * self.envs_per_worker
            parent.send(
                (
                    "step",
                    values[start : start + self.envs_per_worker].astype(np.float32),
                )
            )
        parts = self._recv()
        return VectorStepResult4v3V14(
            *(
                np.concatenate([getattr(part, field) for part in parts], axis=0)
                if field != "episode_summaries"
                else [
                    summary
                    for part in parts
                    for summary in part.episode_summaries
                ]
                for field in VectorStepResult4v3V14._fields
            )
        )

    def policy_modes(self):
        for parent in self._parents:
            parent.send(("policy_modes", None))
        parts = self._recv()
        return {
            key: [value for part in parts for value in part[key]]
            for key in parts[0]
        }

    def state_dict(self):
        for parent in self._parents:
            parent.send(("state_dict", None))
        parts = self._recv()
        return {"envs": [env for part in parts for env in part["envs"]]}

    def load_state_dict(self, state):
        for worker, parent in enumerate(self._parents):
            start = worker * self.envs_per_worker
            parent.send(
                (
                    "load_state_dict",
                    {
                        "envs": state["envs"][
                            start : start + self.envs_per_worker
                        ]
                    },
                )
            )
        self._recv()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
        for parent in self._parents:
            parent.close()


def make_combat_vector_env_4v3_v14(
    config_path: str | Path,
    num_envs: int,
    num_env_workers: int = 0,
    seed: int = 0,
):
    if int(num_env_workers) <= 0:
        return LocalCombatVectorEnv4v3V14(config_path, num_envs, seed)
    return SubprocessCombatVectorEnv4v3V14(
        config_path, num_envs, num_env_workers, seed
    )


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V14",
    "GS_DIM_V14",
    "OBS_DIM_V14",
    "REWARD_COMPONENT_KEYS_V14",
    "VectorStepResult4v3V14",
    "make_combat_vector_env_4v3_v14",
]
