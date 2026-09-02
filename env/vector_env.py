"""Synchronous multi-process vector environment for MARL rollouts."""
from __future__ import annotations

from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import numpy as np

from .mavuav import HeterogeneousMAVUAVAirCombatEnv, RED_IDS


def _environment_state(env: HeterogeneousMAVUAVAirCombatEnv) -> dict[str, Any]:
    return {
        "entities": deepcopy(env.entities), "step_count": env.step_count,
        "episode_return": env.episode_return, "running": env._running,
        "attack_streak": deepcopy(env._attack_streak),
        "red_attack_kills": set(env._red_attack_kills),
        "blue_attack_kills": set(env._blue_attack_kills),
        "rng": deepcopy(env.rng.bit_generator.state),
        "blue_episode_mode": env.blue_policy.episode_mode,
        "profile": env.profile,
    }


def _restore_environment_state(env: HeterogeneousMAVUAVAirCombatEnv, state: Mapping[str, Any]) -> None:
    env.entities = deepcopy(state["entities"])
    env.step_count = int(state["step_count"])
    env.episode_return = float(state["episode_return"])
    env._running = bool(state["running"])
    env._attack_streak = deepcopy(state["attack_streak"])
    env._red_attack_kills = set(state["red_attack_kills"])
    env._blue_attack_kills = set(state["blue_attack_kills"])
    env.rng.bit_generator.state = deepcopy(state["rng"])
    env.blue_policy.episode_mode = state["blue_episode_mode"]
    env.profile = state["profile"]


def _worker(
    connection: Connection,
    config_path: str | Path | Mapping[str, Any] | None,
    index: int,
    base_seed: int | None,
    blue_target_mode: str | None,
    randomize: bool | None,
    profile: str | None,
) -> None:
    """Own and advance one environment inside a dedicated process."""
    reset_count = 0

    def episode_seed() -> int | None:
        if base_seed is None:
            return None
        return int(base_seed + index + 1_000_003 * reset_count)

    try:
        env = HeterogeneousMAVUAVAirCombatEnv(
            config_path, seed=episode_seed(), blue_target_mode=blue_target_mode, randomize=randomize, profile=profile,
        )
        while True:
            command, payload = connection.recv()
            try:
                if command == "reset":
                    base_seed, nearest_probability = payload
                    reset_count = 0
                    reset_options = None if nearest_probability is None else {
                        "nearest_probability": nearest_probability,
                    }
                    observation, info = env.reset(seed=episode_seed(), options=reset_options)
                    result = (observation, env.global_state(), env.active_masks, info, reset_count)
                elif command == "step":
                    actions, reset_nearest_probability = payload
                    observation, reward, terminated, truncated, info = env.step(actions)
                    info = dict(info)
                    if terminated or truncated:
                        terminal_state = env.global_state().copy()
                        terminal_masks = env.active_masks.copy()
                        reset_count += 1
                        reset_options = None if reset_nearest_probability is None else {
                            "nearest_probability": reset_nearest_probability,
                        }
                        observation, reset_info = env.reset(seed=episode_seed(), options=reset_options)
                        info.update({
                            "terminal_global_state": terminal_state,
                            "terminal_active_masks": terminal_masks,
                            "reset_info": reset_info,
                            "auto_reset": True,
                        })
                    else:
                        info["auto_reset"] = False
                    result = (
                        observation, env.global_state(),
                        np.asarray([reward[aid] for aid in RED_IDS], dtype=np.float32),
                        terminated, truncated, env.active_masks, info, reset_count,
                    )
                elif command == "get_state":
                    result = (_environment_state(env), reset_count, base_seed)
                elif command == "set_state":
                    state, reset_count, restored_base_seed = payload
                    if restored_base_seed is not None:
                        base_seed = int(restored_base_seed)
                    _restore_environment_state(env, state)
                    result = None
                elif command == "get_pid":
                    result = os.getpid()
                elif command == "close":
                    connection.send((True, None))
                    break
                else:
                    raise ValueError(f"unknown vector-environment command: {command}")
                connection.send((True, result))
            except BaseException:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class MAVUAVVectorEnv:
    """Run independent environments concurrently in worker processes.

    Calls remain synchronous from the trainer's perspective, but every worker is
    sent its action before any result is collected. Thus the CPU-heavy physics
    and Blue-policy calculations execute concurrently across environments.
    """

    def __init__(
        self,
        num_envs: int,
        config_path: str | Path | Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        blue_target_mode: str | None = None,
        randomize: bool | None = None,
        profile: str | None = None,
        parallel: bool = True,
        start_method: str | None = None,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.agent_ids = RED_IDS
        self.base_seed = seed
        self.reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.parallel = bool(parallel)
        self.start_method: str | None = None
        self._closed = False
        self.envs: list[HeterogeneousMAVUAVAirCombatEnv] = []
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []

        if not self.parallel:
            self.envs = [
                HeterogeneousMAVUAVAirCombatEnv(
                    config_path, seed=None if seed is None else seed + index,
                    blue_target_mode=blue_target_mode, randomize=randomize, profile=profile,
                )
                for index in range(self.num_envs)
            ]
            return

        methods = mp.get_all_start_methods()
        selected_method = start_method or ("forkserver" if "forkserver" in methods else "spawn")
        if selected_method not in methods:
            raise ValueError(f"unsupported multiprocessing start method: {selected_method}")
        context = mp.get_context(selected_method)
        self.start_method = selected_method
        for index in range(self.num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, config_path, index, seed, blue_target_mode, randomize, profile),
                name=f"mavuav-env-{index}", daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def _seed(self, index: int) -> int | None:
        if self.base_seed is None:
            return None
        return int(self.base_seed + index + 1_000_003 * self.reset_counts[index])

    def _send_all(self, command: str, payloads: list[Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("vector environment is closed")
        for connection, payload in zip(self._connections, payloads):
            connection.send((command, payload))
        results = []
        failures: list[tuple[int, str]] = []
        for index, connection in enumerate(self._connections):
            try:
                success, result = connection.recv()
            except (EOFError, BrokenPipeError) as error:
                failures.append((index, f"worker exited unexpectedly: {error}"))
                results.append(None)
                continue
            if not success:
                failures.append((index, str(result)))
            results.append(result)
        if failures:
            index, detail = failures[0]
            raise RuntimeError(f"environment worker {index} failed:\n{detail}")
        return results

    @staticmethod
    def _stack_observations(observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.asarray([[obs[aid] for aid in RED_IDS] for obs in observations], dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        nearest_probability: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if seed is not None:
            self.base_seed = int(seed)
        self.reset_counts.fill(0)
        if self.parallel:
            results = self._send_all(
                "reset", [(self.base_seed, nearest_probability)] * self.num_envs,
            )
            observations, states, masks, infos, counts = zip(*results)
            self.reset_counts[:] = counts
        else:
            reset_options = None if nearest_probability is None else {
                "nearest_probability": nearest_probability,
            }
            local_results = [
                env.reset(seed=self._seed(index), options=reset_options)
                for index, env in enumerate(self.envs)
            ]
            observations = [item[0] for item in local_results]
            states = [env.global_state() for env in self.envs]
            masks = [env.active_masks for env in self.envs]
            infos = [item[1] for item in local_results]
        return (
            self._stack_observations(list(observations)), np.asarray(states, dtype=np.float32),
            np.asarray(masks, dtype=np.float32), list(infos),
        )

    def step(
        self,
        actions: np.ndarray,
        reset_nearest_probability: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        values = np.asarray(actions, dtype=np.float64)
        expected = (self.num_envs, len(RED_IDS), 3)
        if values.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {values.shape}")
        if self.parallel:
            results = self._send_all(
                "step",
                [(values[index], reset_nearest_probability) for index in range(self.num_envs)],
            )
            observations, states, rewards, terminated, truncated, masks, infos, counts = zip(*results)
            self.reset_counts[:] = counts
            return (
                self._stack_observations(list(observations)), np.asarray(states, dtype=np.float32),
                np.asarray(rewards, dtype=np.float32), np.asarray(terminated, dtype=bool),
                np.asarray(truncated, dtype=bool), np.asarray(masks, dtype=np.float32), list(infos),
            )

        observations: list[dict[str, np.ndarray]] = []
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        rewards = np.empty((self.num_envs, len(RED_IDS)), dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for index, env in enumerate(self.envs):
            observation, reward, term, trunc, info = env.step(values[index])
            rewards[index] = [reward[aid] for aid in RED_IDS]
            terminated[index], truncated[index] = term, trunc
            if term or trunc:
                terminal_state = env.global_state().copy()
                terminal_masks = env.active_masks.copy()
                self.reset_counts[index] += 1
                reset_options = None if reset_nearest_probability is None else {
                    "nearest_probability": reset_nearest_probability,
                }
                observation, reset_info = env.reset(seed=self._seed(index), options=reset_options)
                info = dict(info)
                info.update({
                    "terminal_global_state": terminal_state,
                    "terminal_active_masks": terminal_masks,
                    "reset_info": reset_info,
                    "auto_reset": True,
                })
            else:
                info = dict(info)
                info["auto_reset"] = False
            observations.append(observation)
            states.append(env.global_state())
            masks.append(env.active_masks)
            infos.append(info)
        return (
            self._stack_observations(observations), np.asarray(states, dtype=np.float32), rewards,
            terminated, truncated, np.asarray(masks, dtype=np.float32), infos,
        )

    @property
    def worker_pids(self) -> tuple[int, ...]:
        if not self.parallel:
            return (os.getpid(),) * self.num_envs
        return tuple(int(value) for value in self._send_all("get_pid", [None] * self.num_envs))

    def get_env_states(self) -> list[dict[str, Any]]:
        if self.parallel:
            results = self._send_all("get_state", [None] * self.num_envs)
            states = []
            for index, (state, reset_count, worker_base_seed) in enumerate(results):
                self.reset_counts[index] = reset_count
                if worker_base_seed != self.base_seed:
                    raise RuntimeError("worker and vector environment base seeds diverged")
                states.append(state)
            return states
        return [_environment_state(env) for env in self.envs]

    def set_env_states(
        self,
        states: list[Mapping[str, Any]],
        reset_counts: np.ndarray | list[int] | None = None,
        base_seed: int | None = None,
    ) -> None:
        if len(states) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} environment states, got {len(states)}")
        if base_seed is not None:
            self.base_seed = int(base_seed)
        if reset_counts is not None:
            counts = np.asarray(reset_counts, dtype=np.int64)
            if counts.shape != (self.num_envs,):
                raise ValueError(f"reset_counts must have shape {(self.num_envs,)}, got {counts.shape}")
            self.reset_counts = counts.copy()
        if self.parallel:
            payloads = [(states[i], int(self.reset_counts[i]), self.base_seed) for i in range(self.num_envs)]
            self._send_all("set_state", payloads)
        else:
            for env, state in zip(self.envs, states):
                _restore_environment_state(env, state)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.parallel:
            return
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for connection in self._connections:
            try:
                if connection.poll(1.0):
                    connection.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def __enter__(self) -> "MAVUAVVectorEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
