"""Persistent multi-process parallel vector environment backend.

Provides two backends with identical minimal APIs:

- ``LocalCombatVectorEnv`` — sequential, single-process (num_env_workers=1).
- ``SubprocessCombatVectorEnv`` — persistent worker processes using ``spawn``.

Neither backend depends on Ray, Dask, MPI, or Gymnasium.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..environment import HomogeneousAirCombatEnv

# ---------------------------------------------------------------------------
# Compact termination / outcome encoding
# ---------------------------------------------------------------------------

REASON_CODE_MAP: dict[str | None, int] = {
    None: 0,
    "red_kill": 1,
    "blue_kill": 2,
    "mutual_kill": 3,
    "collision": 4,
    "altitude_boundary": 5,
    "xy_boundary": 6,
    "boundary": 7,
    "max_steps": 8,
}

REASON_CODE_INV: dict[int, str | None] = {v: k for k, v in REASON_CODE_MAP.items()}

OUTCOME_CODE_MAP: dict[str | None, int] = {
    None: 0,
    "red": 1,
    "blue": 2,
    "draw": 3,
}

OUTCOME_CODE_INV: dict[int, str | None] = {v: k for k, v in OUTCOME_CODE_MAP.items()}


def encode_termination_reason(reason: str | None) -> int:
    """Map a string termination reason to a compact integer code."""
    return REASON_CODE_MAP.get(reason, 0)


def decode_termination_reason(code: int) -> str | None:
    """Recover the string termination reason from an integer code."""
    return REASON_CODE_INV.get(code)


def encode_outcome(outcome: str | None) -> int:
    """Map a string outcome to a compact integer code."""
    return OUTCOME_CODE_MAP.get(outcome, 0)


def decode_outcome(code: int) -> str | None:
    """Recover the string outcome from an integer code."""
    return OUTCOME_CODE_INV.get(code)


# ---------------------------------------------------------------------------
# Fixed control diagnostic keys — must include every key produced by
# controller.diagnostics() plus the extra keys added in env.step().
# Bool values are stored as 0.0 / 1.0 in float32 arrays.
# ---------------------------------------------------------------------------

CONTROL_DIAGNOSTIC_KEYS: list[str] = [
    # from env.step() — action and error summaries
    "action_yaw",
    "action_pitch",
    "action_speed",
    "delta_yaw",
    "delta_pitch",
    "delta_speed",
    # from controller.diagnostics()
    "yaw_error",
    "pitch_error",
    "speed_error",
    "unclipped_yaw_rate",
    "unclipped_pitch_rate",
    "unclipped_acceleration",
    "clipped_yaw_rate",
    "clipped_pitch_rate",
    "clipped_acceleration",
    "nx",
    "nz",
    "phi",
    "yaw_rate_saturated",
    "pitch_rate_saturated",
    "acceleration_saturated",
    "nx_saturated",
    "nz_saturated",
    "phi_saturated",
    # from env.step() — derivative-based tracking
    "actual_acceleration",
    "actual_pitch_rate",
    "actual_yaw_rate",
    "acceleration_tracking_error",
    "pitch_rate_tracking_error",
    "yaw_rate_tracking_error",
    "acceleration_tracking_absolute_error",
    "pitch_rate_tracking_absolute_error",
    "yaw_rate_tracking_absolute_error",
]

K = len(CONTROL_DIAGNOSTIC_KEYS)

# ---------------------------------------------------------------------------
# Helper — extract control diagnostics for one agent into a float32 vector
# ---------------------------------------------------------------------------


def _extract_diagnostics(diag: dict[str, Any]) -> np.ndarray:
    """Pull CONTROL_DIAGNOSTIC_KEYS from a diagnostics dict, coercing bool→float."""
    vec = np.empty(K, dtype=np.float32)
    for i, key in enumerate(CONTROL_DIAGNOSTIC_KEYS):
        val = diag.get(key, 0.0)
        vec[i] = float(val) if not isinstance(val, (bool, np.bool_)) else float(val)
    return vec


# ===================================================================
# Local (sequential) backend
# ===================================================================


class LocalCombatVectorEnv:
    """Sequential vector environment — the correctness baseline."""

    def __init__(self, env_config: str | Path, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.envs = [HomogeneousAirCombatEnv(env_config) for _ in range(num_envs)]
        self._closed = False

    # ------------------------------------------------------------------
    def reset(self, reset_specs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Reset every environment and return (observations, global_states).

        Returns
        -------
        observations : ndarray [num_envs, 2, 14]   (dim 1 = red, blue)
        global_states : ndarray [num_envs, 2, 14]   (dim 1 = red-persp, blue-persp)
        """
        N = self.num_envs
        observations = np.empty((N, 2, 14), dtype=np.float32)
        global_states = np.empty((N, 2, 14), dtype=np.float32)
        for i, (env, spec) in enumerate(zip(self.envs, reset_specs)):
            obs_dict, _info = env.reset(
                int(spec["seed"]), spec["scenario"], spec.get("rear_team")
            )
            observations[i, 0] = obs_dict["red_0"]
            observations[i, 1] = obs_dict["blue_0"]
            global_states[i, 0] = env.global_state("red")
            global_states[i, 1] = env.global_state("blue")
        return observations, global_states

    # ------------------------------------------------------------------
    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray,  # observations     [N, 2, 14]
        np.ndarray,  # global_states    [N, 2, 14]
        np.ndarray,  # rewards          [N, 2]
        np.ndarray,  # terminated       [N] bool
        np.ndarray,  # truncated        [N] bool
        np.ndarray,  # step_counts      [N] int32
        np.ndarray,  # attacks          [N, 2] bool
        np.ndarray,  # geometry         [N, 2, 3]  (distance, ATA, AA)
        np.ndarray,  # control_diag     [N, 2, K]
        np.ndarray,  # reason_codes     [N] int8
        np.ndarray,  # outcome_codes    [N] int8
    ]:
        """Step every environment with the given actions.

        Parameters
        ----------
        actions : ndarray [num_envs, 2, 3]
            dim 1 = red, blue;  dim 2 = [yaw, pitch, speed]

        Returns
        -------
        Eleven compact arrays (see source for shapes).
        """
        N = self.num_envs
        observations = np.empty((N, 2, 14), dtype=np.float32)
        global_states = np.empty((N, 2, 14), dtype=np.float32)
        rewards = np.empty((N, 2), dtype=np.float32)
        terminated = np.empty(N, dtype=bool)
        truncated = np.empty(N, dtype=bool)
        step_counts = np.empty(N, dtype=np.int32)
        attacks = np.empty((N, 2), dtype=bool)
        geometry = np.empty((N, 2, 3), dtype=np.float32)
        control_diag = np.empty((N, 2, K), dtype=np.float32)
        reason_codes = np.empty(N, dtype=np.int8)
        outcome_codes = np.empty(N, dtype=np.int8)

        for i, (env, act) in enumerate(zip(self.envs, actions)):
            action_dict = {"red_0": act[0], "blue_0": act[1]}
            obs_dict, reward, term, trunc, info = env.step(action_dict)

            observations[i, 0] = obs_dict["red_0"]
            observations[i, 1] = obs_dict["blue_0"]
            global_states[i, 0] = env.global_state("red")
            global_states[i, 1] = env.global_state("blue")
            rewards[i, 0] = reward["red_0"]
            rewards[i, 1] = reward["blue_0"]
            terminated[i] = term
            truncated[i] = trunc
            step_counts[i] = info["step_count"]
            attacks[i, 0] = info["attacks"]["red_0"]
            attacks[i, 1] = info["attacks"]["blue_0"]

            geo_red = info["geometries"]["red_0"]
            geo_blue = info["geometries"]["blue_0"]
            geometry[i, 0] = [geo_red.distance, geo_red.ata, geo_red.aa]
            geometry[i, 1] = [geo_blue.distance, geo_blue.ata, geo_blue.aa]

            control_diag[i, 0] = _extract_diagnostics(
                info["control_diagnostics"]["red_0"]
            )
            control_diag[i, 1] = _extract_diagnostics(
                info["control_diagnostics"]["blue_0"]
            )

            reason_codes[i] = encode_termination_reason(info["termination_reason"])
            outcome_codes[i] = encode_outcome(info["outcome"])

        return (
            observations,
            global_states,
            rewards,
            terminated,
            truncated,
            step_counts,
            attacks,
            geometry,
            control_diag,
            reason_codes,
            outcome_codes,
        )

    # ------------------------------------------------------------------
    def reset_at(
        self, indices: np.ndarray, reset_specs: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset only the environments whose *local* indices are in ``indices``.

        Returns arrays sized to ``[len(indices), …]``.
        """
        L = len(indices)
        observations = np.empty((L, 2, 14), dtype=np.float32)
        global_states = np.empty((L, 2, 14), dtype=np.float32)
        for j, idx in enumerate(indices):
            i = int(idx)
            env = self.envs[i]
            spec = reset_specs[j]
            obs_dict, _info = env.reset(
                int(spec["seed"]), spec["scenario"], spec.get("rear_team")
            )
            observations[j, 0] = obs_dict["red_0"]
            observations[j, 1] = obs_dict["blue_0"]
            global_states[j, 0] = env.global_state("red")
            global_states[j, 1] = env.global_state("blue")
        return observations, global_states

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Idempotent no-op for the local backend."""
        self._closed = True

    def __enter__(self) -> "LocalCombatVectorEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ===================================================================
# Subprocess (parallel) backend
# ===================================================================


def _worker_main(
    conn: multiprocessing.connection.Connection,
    env_config: str,
    num_local_envs: int,
    worker_id: int,
) -> None:
    """Top-level entry point for a persistent worker process (spawn-safe).

    The worker owns *num_local_envs* ``HomogeneousAirCombatEnv`` instances
    and loops receiving commands over *conn* until ``"close"`` is received.
    """
    envs = [HomogeneousAirCombatEnv(env_config) for _ in range(num_local_envs)]
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset_all":
                result = _worker_reset_all(envs, payload)
                conn.send(result)
            elif cmd == "step":
                result = _worker_step_all(envs, payload)
                conn.send(result)
            elif cmd == "reset_at":
                result = _worker_reset_at(envs, payload)
                conn.send(result)
            elif cmd == "close":
                break
            else:
                conn.send(
                    (
                        "error",
                        f"unknown command: {cmd}",
                        "",
                    )
                )
    except Exception:
        conn.send(("error", traceback.format_exc(), ""))
    finally:
        conn.close()


def _worker_reset_all(
    envs: list[HomogeneousAirCombatEnv], specs: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    L = len(envs)
    observations = np.empty((L, 2, 14), dtype=np.float32)
    global_states = np.empty((L, 2, 14), dtype=np.float32)
    for i, (env, spec) in enumerate(zip(envs, specs)):
        obs_dict, _info = env.reset(
            int(spec["seed"]), spec["scenario"], spec.get("rear_team")
        )
        observations[i, 0] = obs_dict["red_0"]
        observations[i, 1] = obs_dict["blue_0"]
        global_states[i, 0] = env.global_state("red")
        global_states[i, 1] = env.global_state("blue")
    return observations, global_states


def _worker_step_all(
    envs: list[HomogeneousAirCombatEnv], actions: np.ndarray
) -> tuple:
    L = len(envs)
    observations = np.empty((L, 2, 14), dtype=np.float32)
    global_states = np.empty((L, 2, 14), dtype=np.float32)
    rewards = np.empty((L, 2), dtype=np.float32)
    terminated = np.empty(L, dtype=bool)
    truncated = np.empty(L, dtype=bool)
    step_counts = np.empty(L, dtype=np.int32)
    attacks = np.empty((L, 2), dtype=bool)
    geometry = np.empty((L, 2, 3), dtype=np.float32)
    control_diag = np.empty((L, 2, K), dtype=np.float32)
    reason_codes = np.empty(L, dtype=np.int8)
    outcome_codes = np.empty(L, dtype=np.int8)

    for i, (env, act) in enumerate(zip(envs, actions)):
        action_dict = {"red_0": act[0], "blue_0": act[1]}
        obs_dict, reward, term, trunc, info = env.step(action_dict)

        observations[i, 0] = obs_dict["red_0"]
        observations[i, 1] = obs_dict["blue_0"]
        global_states[i, 0] = env.global_state("red")
        global_states[i, 1] = env.global_state("blue")
        rewards[i, 0] = reward["red_0"]
        rewards[i, 1] = reward["blue_0"]
        terminated[i] = term
        truncated[i] = trunc
        step_counts[i] = info["step_count"]
        attacks[i, 0] = info["attacks"]["red_0"]
        attacks[i, 1] = info["attacks"]["blue_0"]

        geo_red = info["geometries"]["red_0"]
        geo_blue = info["geometries"]["blue_0"]
        geometry[i, 0] = [geo_red.distance, geo_red.ata, geo_red.aa]
        geometry[i, 1] = [geo_blue.distance, geo_blue.ata, geo_blue.aa]

        control_diag[i, 0] = _extract_diagnostics(
            info["control_diagnostics"]["red_0"]
        )
        control_diag[i, 1] = _extract_diagnostics(
            info["control_diagnostics"]["blue_0"]
        )

        reason_codes[i] = encode_termination_reason(info["termination_reason"])
        outcome_codes[i] = encode_outcome(info["outcome"])

    return (
        observations,
        global_states,
        rewards,
        terminated,
        truncated,
        step_counts,
        attacks,
        geometry,
        control_diag,
        reason_codes,
        outcome_codes,
    )


def _worker_reset_at(
    envs: list[HomogeneousAirCombatEnv],
    payload: tuple[np.ndarray, list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray]:
    indices, specs = payload
    L = len(indices)
    observations = np.empty((L, 2, 14), dtype=np.float32)
    global_states = np.empty((L, 2, 14), dtype=np.float32)
    for j, idx in enumerate(indices):
        i = int(idx)
        spec = specs[j]
        env = envs[i]
        obs_dict, _info = env.reset(
            int(spec["seed"]), spec["scenario"], spec.get("rear_team")
        )
        observations[j, 0] = obs_dict["red_0"]
        observations[j, 1] = obs_dict["blue_0"]
        global_states[j, 0] = env.global_state("red")
        global_states[j, 1] = env.global_state("blue")
    return observations, global_states


class SubprocessCombatVectorEnv:
    """Persistent multi-process vector environment using ``spawn``.

    Each worker manages a fixed contiguous block of global environment indices
    and communicates via ``multiprocessing.Pipe`` using compact NumPy arrays.
    No full environment objects are ever pickled between processes.
    """

    def __init__(
        self, env_config: str | Path, num_envs: int, num_env_workers: int
    ) -> None:
        if num_env_workers < 2:
            raise ValueError(
                "SubprocessCombatVectorEnv requires num_env_workers >= 2; "
                "use LocalCombatVectorEnv for single-worker mode"
            )
        if num_envs % num_env_workers != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by "
                f"num_env_workers ({num_env_workers})"
            )
        self.num_envs = num_envs
        self.num_env_workers = num_env_workers
        self.envs_per_worker = num_envs // num_env_workers
        self._closed = False

        env_config_str = str(env_config)
        ctx = multiprocessing.get_context("spawn")
        self._parent_conns: list[multiprocessing.connection.Connection] = []
        self._workers: list[multiprocessing.Process] = []

        for w in range(num_env_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_main,
                args=(child_conn, env_config_str, self.envs_per_worker, w),
                name=f"combat-worker-{w}",
            )
            p.start()
            child_conn.close()  # close child end in parent
            self._parent_conns.append(parent_conn)
            self._workers.append(p)

    # ------------------------------------------------------------------
    def _worker_slice(self, worker_id: int) -> slice:
        """Return the global env-index slice managed by *worker_id*."""
        start = worker_id * self.envs_per_worker
        return slice(start, start + self.envs_per_worker)

    # ------------------------------------------------------------------
    def reset(self, reset_specs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Reset every environment and return (observations, global_states)."""
        self._check_not_closed()
        for w in range(self.num_env_workers):
            sl = self._worker_slice(w)
            self._parent_conns[w].send(("reset_all", reset_specs[sl]))

        obs_parts = []
        gs_parts = []
        for w in range(self.num_env_workers):
            result = self._parent_conns[w].recv()
            self._check_worker_error(result, w)
            obs, gs = result
            obs_parts.append(obs)
            gs_parts.append(gs)

        return np.concatenate(obs_parts, axis=0), np.concatenate(gs_parts, axis=0)

    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple:
        """Step every environment.

        Parameters
        ----------
        actions : ndarray [num_envs, 2, 3]

        Returns
        -------
        Eleven compact arrays; see ``LocalCombatVectorEnv.step``.
        """
        self._check_not_closed()
        for w in range(self.num_env_workers):
            sl = self._worker_slice(w)
            self._parent_conns[w].send(("step", actions[sl]))

        results = []
        for w in range(self.num_env_workers):
            result = self._parent_conns[w].recv()
            self._check_worker_error(result, w)
            results.append(result)

        # results: list of 11-tuples; concatenate each position along axis 0
        return tuple(
            np.concatenate([r[i] for r in results], axis=0) for i in range(11)
        )

    # ------------------------------------------------------------------
    def reset_at(
        self, global_indices: np.ndarray, reset_specs: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset environments at *global_indices* and return
        (observations [L, 2, 14], global_states [L, 2, 14]).
        """
        self._check_not_closed()
        # Group by worker
        grouped: dict[int, tuple[list[int], list[dict]]] = {}
        for j, gi in enumerate(global_indices):
            w = int(gi) // self.envs_per_worker
            local = int(gi) - w * self.envs_per_worker
            grouped.setdefault(w, ([], []))
            grouped[w][0].append(local)
            grouped[w][1].append(reset_specs[j])

        # We need to assemble results in global-index order, so store per-global
        results_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for w, (local_indices, specs) in grouped.items():
            self._parent_conns[w].send(
                ("reset_at", (np.asarray(local_indices, dtype=np.int32), specs))
            )

        for w in grouped:
            result = self._parent_conns[w].recv()
            self._check_worker_error(result, w)
            obs_part, gs_part = result
            for j, gi in enumerate(
                [  # rebuild global indices for this worker
                    int(gi)
                    for gi in global_indices
                    if int(gi) // self.envs_per_worker == w
                ]
            ):
                results_map[gi] = (obs_part[j : j + 1], gs_part[j : j + 1])

        # Assemble in global index order
        obs_list = [results_map[int(gi)][0] for gi in global_indices]
        gs_list = [results_map[int(gi)][1] for gi in global_indices]
        return np.concatenate(obs_list, axis=0), np.concatenate(gs_list, axis=0)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down all workers gracefully.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (OSError, BrokenPipeError):
                pass
        for w_id, worker in enumerate(self._workers):
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        for conn in self._parent_conns:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("vector environment has been closed")

    @staticmethod
    def _check_worker_error(result: Any, worker_id: int) -> None:
        if isinstance(result, tuple) and len(result) == 3 and result[0] == "error":
            tb = result[1]
            raise RuntimeError(
                f"Worker {worker_id} raised an exception:\n{tb}"
            )

    # ------------------------------------------------------------------
    def __enter__(self) -> "SubprocessCombatVectorEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ===================================================================
# Factory
# ===================================================================


def make_combat_vector_env(
    env_config: str | Path,
    num_envs: int,
    num_env_workers: int = 4,
) -> LocalCombatVectorEnv | SubprocessCombatVectorEnv:
    """Create the appropriate vector-environment backend.

    Parameters
    ----------
    env_config : path
        Config file for ``HomogeneousAirCombatEnv``.
    num_envs : int
        Total number of environment slots (must be >= num_env_workers and
        divisible by num_env_workers when workers > 1).
    num_env_workers : int
        Number of persistent CPU worker processes.
        * 1  → ``LocalCombatVectorEnv`` (main-process, sequential).
        * >1 → ``SubprocessCombatVectorEnv`` (spawn-based, parallel).

    Returns
    -------
    LocalCombatVectorEnv or SubprocessCombatVectorEnv
    """
    if num_env_workers < 1:
        raise ValueError("num_env_workers must be >= 1")
    if num_env_workers > num_envs:
        raise ValueError(
            f"num_env_workers ({num_env_workers}) cannot exceed num_envs ({num_envs})"
        )
    if num_env_workers == 1:
        return LocalCombatVectorEnv(env_config, num_envs)
    if num_envs % num_env_workers != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by "
            f"num_env_workers ({num_env_workers})"
        )
    return SubprocessCombatVectorEnv(env_config, num_envs, num_env_workers)
