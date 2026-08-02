"""Deterministic evaluation helpers for HAPPO 3v3 fixed-blue experiments."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..environment_3v3 import OBS_DIM
from ..mappo.vector_env_3v3 import (
    decode_3v3_outcome,
    decode_3v3_termination_reason,
    make_combat_vector_env_3v3,
)
from .networks import IndependentHAPPOActors


def _summarize(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"episodes": 0, "evaluation_seconds": elapsed}
    rk = sum(r["red_attack_kills"] for r in records)
    bk = sum(r["blue_attack_kills"] for r in records)
    total_steps = sum(r["episode_length"] for r in records)
    base = {
        "episodes": n,
        "red_complete_elimination_success_rate": sum(r["red_complete_elimination_success"] for r in records) / n,
        "blue_complete_elimination_success_rate": sum(r["blue_complete_elimination_success"] for r in records) / n,
        "environment_red_outcome_rate": sum(r["environment_outcome"] == "red" for r in records) / n,
        "environment_blue_outcome_rate": sum(r["environment_outcome"] == "blue" for r in records) / n,
        "draw_rate": sum(r["environment_outcome"] == "draw" for r in records) / n,
        "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
        "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
        "mean_red_survivors": float(np.mean([r["red_survivors"] for r in records])),
        "mean_blue_survivors": float(np.mean([r["blue_survivors"] for r in records])),
        "red_kd_numerator": int(rk),
        "red_kd_denominator": int(bk),
        "red_kd_ratio": rk / bk if bk > 0 else None,
        "mean_red_boundary_deaths": float(np.mean([r["red_boundary_deaths"] for r in records])),
        "mean_blue_boundary_deaths": float(np.mean([r["blue_boundary_deaths"] for r in records])),
        "mean_red_boundary_altitude_deaths": float(np.mean([r["red_boundary_altitude_deaths"] for r in records])),
        "mean_blue_boundary_altitude_deaths": float(np.mean([r["blue_boundary_altitude_deaths"] for r in records])),
        "mean_red_boundary_xy_deaths": float(np.mean([r["red_boundary_xy_deaths"] for r in records])),
        "mean_blue_boundary_xy_deaths": float(np.mean([r["blue_boundary_xy_deaths"] for r in records])),
        "mean_red_friendly_collision_deaths": float(np.mean([r["red_friendly_collision_deaths"] for r in records])),
        "mean_blue_friendly_collision_deaths": float(np.mean([r["blue_friendly_collision_deaths"] for r in records])),
        "mean_red_cross_collision_deaths": float(np.mean([r["red_cross_collision_deaths"] for r in records])),
        "mean_blue_cross_collision_deaths": float(np.mean([r["blue_cross_collision_deaths"] for r in records])),
        "mean_red_kills_with_shared_observation": float(np.mean([r["red_kills_with_shared_observation"] for r in records])),
        "mean_blue_kills_with_shared_observation": float(np.mean([r["blue_kills_with_shared_observation"] for r in records])),
        "mean_red_support_coverage_ratio": float(np.mean([r["red_mean_support_coverage_ratio"] for r in records])),
        "mean_blue_support_coverage_ratio": float(np.mean([r["blue_mean_support_coverage_ratio"] for r in records])),
        "red_support_survival_rate": sum(r["red_support_survived"] for r in records) / n,
        "blue_support_survival_rate": sum(r["blue_support_survived"] for r in records) / n,
        "max_steps_rate": sum(r["termination_reason"] == "max_steps" for r in records) / n,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "evaluation_seconds": elapsed,
        "environment_steps_per_second": total_steps / elapsed if elapsed > 0 else 0.0,
    }
    base.update({
        "red_any_attack_kill_rate": sum(1 for r in records if r.get("red_any_attack_kill")) / n,
        "blue_any_attack_kill_rate": sum(1 for r in records if r.get("blue_any_attack_kill")) / n,
    })
    return base


def evaluate_happo_fixed_blue_3v3(
    actors: IndependentHAPPOActors,
    env_config: str | Path,
    episodes: int,
    num_envs: int = 8,
    num_env_workers: int = 4,
    device: torch.device | None = None,
    seed_start: int = 100000,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    was_training = actors.training
    actors.eval()
    vec_env = make_combat_vector_env_3v3(str(env_config), num_envs, num_env_workers)
    try:
        obs, gs, am = vec_env.reset([{"seed": seed_start + i} for i in range(num_envs)])
        next_seed = seed_start + num_envs
        records: list[dict[str, Any]] = []
        action_sum = np.zeros(3, dtype=np.float64)
        action_sat_sum = np.zeros(3, dtype=np.float64)
        action_count = 0
        t0 = time.perf_counter()
        while len(records) < episodes:
            red_obs = torch.as_tensor(obs[:, :3, :], device=device)
            with torch.no_grad():
                actions = actors.deterministic_actions(red_obs).cpu().numpy().reshape(num_envs, 3, 3)
            alive_actions = actions[am[:, :3].astype(bool)]
            if alive_actions.size:
                action_sum += alive_actions.sum(axis=0)
                action_sat_sum += (np.abs(alive_actions) >= 0.95).sum(axis=0)
                action_count += alive_actions.shape[0]
            actions *= am[:, :3, None]
            result = vec_env.step(actions)
            obs, gs, am = result.observations, result.global_states, result.alive_masks
            done_idx = np.where(result.episode_valid)[0]
            for i in done_idx:
                records.append({
                    "red_complete_elimination_success": bool(result.red_complete_elimination_success[i]),
                    "blue_complete_elimination_success": bool(result.blue_complete_elimination_success[i]),
                    "red_attack_kills": int(result.episode_red_attack_kills[i]),
                    "blue_attack_kills": int(result.episode_blue_attack_kills[i]),
                    "red_survivors": int(result.episode_red_survivors[i]),
                    "blue_survivors": int(result.episode_blue_survivors[i]),
                    "red_boundary_deaths": int(result.episode_red_boundary_deaths[i]),
                    "blue_boundary_deaths": int(result.episode_blue_boundary_deaths[i]),
                    "red_boundary_altitude_deaths": int(result.episode_red_boundary_altitude_deaths[i]),
                    "blue_boundary_altitude_deaths": int(result.episode_blue_boundary_altitude_deaths[i]),
                    "red_boundary_xy_deaths": int(result.episode_red_boundary_xy_deaths[i]),
                    "blue_boundary_xy_deaths": int(result.episode_blue_boundary_xy_deaths[i]),
                    "red_friendly_collision_deaths": int(result.episode_red_friendly_collision_deaths[i]),
                    "blue_friendly_collision_deaths": int(result.episode_blue_friendly_collision_deaths[i]),
                    "red_cross_collision_deaths": int(result.episode_red_cross_collision_deaths[i]),
                    "blue_cross_collision_deaths": int(result.episode_blue_cross_collision_deaths[i]),
                    "red_kills_with_shared_observation": int(result.episode_red_kills_with_shared_observation[i]),
                    "blue_kills_with_shared_observation": int(result.episode_blue_kills_with_shared_observation[i]),
                    "red_mean_support_coverage_ratio": float(result.episode_red_mean_support_coverage_ratio[i]),
                    "blue_mean_support_coverage_ratio": float(result.episode_blue_mean_support_coverage_ratio[i]),
                    "red_support_survived": bool(result.episode_red_support_survived[i]),
                    "blue_support_survived": bool(result.episode_blue_support_survived[i]),
                    "red_any_attack_kill": bool(result.episode_red_any_attack_kill[i]),
                    "blue_any_attack_kill": bool(result.episode_blue_any_attack_kill[i]),
                    "episode_length": int(result.episode_length[i]),
                    "termination_reason": decode_3v3_termination_reason(int(result.termination_reason_codes[i])),
                    "environment_outcome": decode_3v3_outcome(int(result.outcome_codes[i])),
                })
            if len(done_idx) > 0:
                seeds = [{"seed": next_seed + j} for j in range(len(done_idx))]
                next_seed += len(done_idx)
                no, ng, na = vec_env.reset_at(done_idx, seeds)
                obs[done_idx], gs[done_idx], am[done_idx] = no, ng, na
        summary = _summarize(records[:episodes], time.perf_counter() - t0)
        if action_count > 0:
            action_mean = action_sum / action_count
            action_sat = action_sat_sum / action_count
            for dim, name in enumerate(("yaw", "pitch", "speed")):
                summary[f"deterministic_action_mean_{name}"] = float(action_mean[dim])
                summary[f"deterministic_action_saturation_rate_{name}"] = float(action_sat[dim])
        return summary
    finally:
        vec_env.close()
        if was_training:
            actors.train()


__all__ = ["evaluate_happo_fixed_blue_3v3"]
