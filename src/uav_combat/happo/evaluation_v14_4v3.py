"""Deterministic evaluation for v14 mission-aligned role-shared HAPPO."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from ..environment_4v3_v14 import FunctionalHeterogeneous4v3V14MissionAlignedEnv
from ..mappo.trainer_3v3 import resolve_device
from ..scenario_4v3_v14 import BLUE_IDS_V14, RED_COMBAT_IDS_V14, RED_IDS_V14
from .evaluation_4v3 import sha256_json_4v3, validate_evaluation_seed_manifest
from .role_shared_networks import RoleSharedHAPPOActors
from .trainer_4v3 import summarize_4v3_episodes


class _RoleMetricEnvV14(FunctionalHeterogeneous4v3V14MissionAlignedEnv):
    def _update_locks(self, *args: Any, **kwargs: Any):
        half_events, killers = super()._update_locks(*args, **kwargs)
        self._evaluation_last_half_events = set(half_events)
        self._evaluation_last_killers = dict(killers)
        return half_events, killers


def _resolve_seeds(
    episodes: int | None, seeds: Sequence[int] | None, seed: int
) -> list[int]:
    if seeds is not None:
        values = [int(value) for value in seeds]
        if episodes is not None and int(episodes) != len(values):
            raise ValueError("episodes must match explicit seeds")
    else:
        if episodes is None or int(episodes) <= 0:
            raise ValueError("episodes must be positive")
        values = [int(seed) + index for index in range(int(episodes))]
    if not values or len(values) != len(set(values)):
        raise ValueError("evaluation seeds must be nonempty and unique")
    return values


def _add_slot_aggregates(
    summary: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    slot_metrics: dict[str, dict[str, float]] = {}
    for slot, agent_id in enumerate(RED_COMBAT_IDS_V14, start=1):
        values = {
            "kills": float(np.mean([row[f"{agent_id}_kills"] for row in records])),
            "max_lock": float(
                np.mean([row[f"{agent_id}_max_lock"] for row in records])
            ),
            "half_lock": float(
                np.mean([row[f"{agent_id}_half_lock"] for row in records])
            ),
            "survival": float(
                np.mean([row[f"{agent_id}_survival"] for row in records])
            ),
        }
        slot_metrics[f"red_{slot}"] = values
        for key, value in values.items():
            summary[f"red_{slot}_{key}"] = value
    for metric in ("kills", "max_lock", "half_lock", "survival"):
        values = np.asarray(
            [slot_metrics[f"red_{slot}"][metric] for slot in (1, 2, 3)],
            dtype=np.float64,
        )
        summary[f"combat_slot_{metric}_mean"] = float(values.mean())
        summary[f"combat_slot_{metric}_std"] = float(values.std())
        summary[f"combat_slot_{metric}_range"] = float(values.max() - values.min())


@torch.no_grad()
def evaluate_v14_happo_fixed_blue_4v3(
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
    training_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del num_env_workers
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
            batch_seeds = seed_list[start : start + capacity]
            envs = [_RoleMetricEnvV14(env_config) for _ in batch_seeds]
            reset_values = [
                env.reset(item_seed) for env, item_seed in zip(envs, batch_seeds)
            ]
            obs = np.stack([value[0] for value in reset_values]).astype(np.float32)
            alive = np.stack([value[2] for value in reset_values]).astype(np.float32)[
                :, :4
            ]
            finished = np.zeros(len(envs), bool)
            slot_kills = np.zeros((len(envs), 3), np.float32)
            slot_max_lock = np.zeros((len(envs), 3), np.float32)
            slot_half = np.zeros((len(envs), 3), np.float32)
            while not bool(finished.all()):
                obs_t = torch.as_tensor(
                    obs[:, :4], dtype=torch.float32, device=dev
                )
                alive_t = torch.as_tensor(alive, dtype=torch.float32, device=dev)
                actions_t, _ = actors.deterministic_actions(
                    obs_t, alive_t, None, None
                )
                actions = actions_t.cpu().numpy().astype(np.float32)
                next_obs = obs.copy()
                next_alive = alive.copy()
                for index, env in enumerate(envs):
                    if finished[index]:
                        continue
                    red_actions = {
                        agent_id: actions[index, slot]
                        for slot, agent_id in enumerate(RED_IDS_V14)
                    }
                    item_obs, _, item_alive, _, done, _, info = env.step(red_actions)
                    next_obs[index] = item_obs
                    next_alive[index] = item_alive[:4]
                    for slot, agent_id in enumerate(RED_COMBAT_IDS_V14):
                        slot_max_lock[index, slot] = max(
                            slot_max_lock[index, slot],
                            float(env.lock_progress.get(agent_id, 0.0)),
                        )
                    for attacker_id, target_id in getattr(
                        env, "_evaluation_last_half_events", set()
                    ):
                        if (
                            attacker_id in RED_COMBAT_IDS_V14
                            and target_id in BLUE_IDS_V14
                        ):
                            slot_half[
                                index, RED_COMBAT_IDS_V14.index(attacker_id)
                            ] = 1.0
                    for target_id, killer_id in getattr(
                        env, "_evaluation_last_killers", {}
                    ).items():
                        if (
                            killer_id in RED_COMBAT_IDS_V14
                            and target_id in BLUE_IDS_V14
                        ):
                            slot_kills[
                                index, RED_COMBAT_IDS_V14.index(killer_id)
                            ] += 1.0
                    if done:
                        record = deepcopy(info["episode_summary"])
                        record["episode_seed"] = int(batch_seeds[index])
                        for slot, agent_id in enumerate(RED_COMBAT_IDS_V14):
                            record[f"{agent_id}_kills"] = float(
                                slot_kills[index, slot]
                            )
                            record[f"{agent_id}_max_lock"] = float(
                                slot_max_lock[index, slot]
                            )
                            record[f"{agent_id}_half_lock"] = float(
                                slot_half[index, slot]
                            )
                            record[f"{agent_id}_survival"] = float(
                                env._alive(agent_id)
                            )
                        records.append(record)
                        finished[index] = True
                obs, alive = next_obs, next_alive
            del envs
        records.sort(key=lambda row: int(row["episode_seed"]))
        summary = summarize_4v3_episodes(records)
        _add_slot_aggregates(summary, records)
        for agent_id in RED_IDS_V14:
            summary[f"mean_{agent_id}_agent_return"] = float(
                np.mean(
                    [float(row["agent_returns"][agent_id]) for row in records]
                )
            )
        summary.update(
            {
                "mean_support_agent_return": summary["mean_red_0_agent_return"],
                "mean_combat1_agent_return": summary["mean_red_1_agent_return"],
                "mean_combat2_agent_return": summary["mean_red_2_agent_return"],
                "mean_combat3_agent_return": summary["mean_red_3_agent_return"],
                "split": split,
                "seed_list": seed_list,
                "seed_hash": sha256_json_4v3(seed_list),
                "manifest_hash": (
                    seed_manifest.get("manifest_hash") if seed_manifest else None
                ),
                "evaluation_seconds": float(time.perf_counter() - started),
                "episode_records": records,
                "deterministic": True,
                "recurrent_actor": False,
            }
        )
        diagnostics = training_diagnostics or {}
        for key in (
            "support_advantage_mean",
            "support_advantage_std",
            "pooled_combat_advantage_mean",
            "pooled_combat_advantage_std",
            "combat_slot_1_raw_advantage_mean",
            "combat_slot_1_raw_advantage_std",
            "combat_slot_2_raw_advantage_mean",
            "combat_slot_2_raw_advantage_std",
            "combat_slot_3_raw_advantage_mean",
            "combat_slot_3_raw_advantage_std",
            "support_critic_explained_variance",
            "pooled_combat_critic_explained_variance",
            "combat_slot_1_critic_explained_variance",
            "combat_slot_2_critic_explained_variance",
            "combat_slot_3_critic_explained_variance",
        ):
            summary[key] = diagnostics.get(key)
        return summary
    finally:
        if was_training:
            actors.train()


__all__ = ["evaluate_v14_happo_fixed_blue_4v3"]
