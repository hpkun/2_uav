"""Deterministic fixed-seed evaluation for v13 role-shared HAPPO."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from ..environment_4v3_v12 import FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv
from ..scenario_4v3_v12 import BLUE_IDS_V12, RED_COMBAT_IDS_V12, RED_IDS_V12
from ..mappo.trainer_3v3 import resolve_device
from .evaluation_4v3 import sha256_json_4v3, validate_evaluation_seed_manifest
from .role_shared_networks import RoleSharedHAPPOActors
from .trainer_4v3 import summarize_4v3_episodes


class _RoleMetricEnv(FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv):
    def _update_locks(self, *args: Any, **kwargs: Any):
        half_events, killers = super()._update_locks(*args, **kwargs)
        self._evaluation_last_half_events = set(half_events)
        self._evaluation_last_killers = dict(killers)
        return half_events, killers


def _resolve_seeds(episodes: int | None, seeds: Sequence[int] | None, seed: int) -> list[int]:
    if seeds is not None:
        values = [int(v) for v in seeds]
        if episodes is not None and int(episodes) != len(values):
            raise ValueError("episodes must match explicit seeds")
    else:
        if episodes is None or int(episodes) <= 0:
            raise ValueError("episodes must be positive")
        values = [int(seed) + i for i in range(int(episodes))]
    if not values or len(values) != len(set(values)):
        raise ValueError("evaluation seeds must be nonempty and unique")
    return values


def _add_slot_aggregates(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    slot_metrics: dict[str, dict[str, float]] = {}
    for slot, agent_id in enumerate(RED_COMBAT_IDS_V12, start=1):
        values = {
            "kills": float(np.mean([float(r[f"{agent_id}_kills"]) for r in records])),
            "max_lock": float(np.mean([float(r[f"{agent_id}_max_lock"]) for r in records])),
            "half_lock": float(np.mean([float(r[f"{agent_id}_half_lock"]) for r in records])),
            "survival": float(np.mean([float(r[f"{agent_id}_survival"]) for r in records])),
        }
        slot_metrics[f"red_{slot}"] = values
        for key, value in values.items():
            summary[f"red_{slot}_{key}"] = value
    for metric in ("kills", "max_lock", "half_lock", "survival"):
        values = np.asarray([slot_metrics[f"red_{slot}"][metric] for slot in (1, 2, 3)], dtype=np.float64)
        summary[f"combat_slot_{metric}_mean"] = float(values.mean())
        summary[f"combat_slot_{metric}_std"] = float(values.std())
        summary[f"combat_slot_{metric}_range"] = float(values.max() - values.min())


@torch.no_grad()
def evaluate_role_shared_happo_fixed_blue_4v3(
    actors: RoleSharedHAPPOActors,
    env_config: str | Path,
    *,
    episodes: int | None = None,
    num_envs: int = 4,
    num_env_workers: int = 0,
    device: str | torch.device = "cpu",
    seed: int = 10000,
    seeds: Sequence[int] | None = None,
    split: str = "selection",
    seed_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic mean actions with per-slot hidden reset semantics."""
    del num_env_workers  # Direct environments expose the per-slot lock attribution required here.
    dev = resolve_device(str(device))
    seed_list = _resolve_seeds(episodes, seeds, seed)
    if seed_manifest is not None:
        validate_evaluation_seed_manifest(seed_manifest)
    was_training = actors.training
    actors.eval()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        capacity = max(1, int(num_envs))
        for start in range(0, len(seed_list), capacity):
            batch_seeds = seed_list[start:start + capacity]
            envs = [_RoleMetricEnv(env_config) for _ in batch_seeds]
            reset_values = [env.reset(item_seed) for env, item_seed in zip(envs, batch_seeds)]
            obs = np.stack([value[0] for value in reset_values]).astype(np.float32)
            alive = np.stack([value[2] for value in reset_values]).astype(np.float32)[:, :4]
            hidden = actors.initial_hidden(len(envs), dev)
            reset_masks = np.zeros((len(envs), 4), np.float32)
            finished = np.zeros(len(envs), bool)
            slot_kills = np.zeros((len(envs), 3), np.float32)
            slot_max_lock = np.zeros((len(envs), 3), np.float32)
            slot_half = np.zeros((len(envs), 3), np.float32)
            while not bool(finished.all()):
                obs_t = torch.as_tensor(obs[:, :4], dtype=torch.float32, device=dev)
                alive_t = torch.as_tensor(alive, dtype=torch.float32, device=dev)
                reset_t = torch.as_tensor(reset_masks, dtype=torch.float32, device=dev)
                actions_t, next_hidden = actors.deterministic_actions(obs_t, alive_t, hidden, reset_t)
                actions = actions_t.cpu().numpy().astype(np.float32)
                next_obs = obs.copy(); next_alive = alive.copy(); done_flags = np.zeros(len(envs), bool)
                for index, env in enumerate(envs):
                    if finished[index]:
                        continue
                    red_actions = {aid: actions[index, slot] for slot, aid in enumerate(RED_IDS_V12)}
                    item_obs, _, item_alive, _, done, _, info = env.step(red_actions)
                    next_obs[index] = item_obs; next_alive[index] = item_alive[:4]; done_flags[index] = done
                    for slot, agent_id in enumerate(RED_COMBAT_IDS_V12):
                        slot_max_lock[index, slot] = max(slot_max_lock[index, slot], float(env.lock_progress.get(agent_id, 0.0)))
                    for attacker_id, target_id in getattr(env, "_evaluation_last_half_events", set()):
                        if attacker_id in RED_COMBAT_IDS_V12 and target_id in BLUE_IDS_V12:
                            slot_half[index, RED_COMBAT_IDS_V12.index(attacker_id)] = 1.0
                    for target_id, killer_id in getattr(env, "_evaluation_last_killers", {}).items():
                        if killer_id in RED_COMBAT_IDS_V12 and target_id in BLUE_IDS_V12:
                            slot_kills[index, RED_COMBAT_IDS_V12.index(killer_id)] += 1.0
                    if done:
                        record = deepcopy(info["episode_summary"])
                        record["episode_seed"] = int(batch_seeds[index])
                        for slot, agent_id in enumerate(RED_COMBAT_IDS_V12):
                            record[f"{agent_id}_kills"] = float(slot_kills[index, slot])
                            record[f"{agent_id}_max_lock"] = float(slot_max_lock[index, slot])
                            record[f"{agent_id}_half_lock"] = float(slot_half[index, slot])
                            record[f"{agent_id}_survival"] = float(env._alive(agent_id))
                        records.append(record)
                        finished[index] = True
                continuation = alive * next_alive * (~done_flags).astype(np.float32)[:, None]
                hidden = next_hidden
                if hidden is not None:
                    mask = torch.as_tensor(continuation, dtype=torch.float32, device=dev)
                    hidden.support.mul_(mask[:, 0:1]); hidden.combat.mul_(mask[:, 1:4].unsqueeze(-1))
                reset_masks = continuation
                obs, alive = next_obs, next_alive
            del envs
        records.sort(key=lambda row: int(row["episode_seed"]))
        summary = summarize_4v3_episodes(records)
        _add_slot_aggregates(summary, records)
        summary.update({
            "split": split, "seed_list": seed_list, "seed_hash": sha256_json_4v3(seed_list),
            "manifest_hash": seed_manifest.get("manifest_hash") if seed_manifest else None,
            "evaluation_seconds": float(time.perf_counter() - started), "episode_records": records,
            "deterministic": True, "recurrent_actor": actors.recurrent,
        })
        return summary
    finally:
        if was_training:
            actors.train()


__all__ = ["evaluate_role_shared_happo_fixed_blue_4v3"]
