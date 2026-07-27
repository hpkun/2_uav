"""Deterministic evaluation helpers for 3v3 MADSAC."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..mappo.vector_env_3v3 import (
    decode_3v3_outcome,
    decode_3v3_termination_reason,
    make_combat_vector_env_3v3,
)
from .networks import SharedSquashedGaussianActor


def _summarize(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"episodes": 0, "evaluation_seconds": elapsed}
    rk = sum(r["red_attack_kills"] for r in records)
    bk = sum(r["blue_attack_kills"] for r in records)
    total_steps = sum(r["episode_length"] for r in records)
    return {
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
        "max_steps_rate": sum(r["termination_reason"] == "max_steps" for r in records) / n,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "evaluation_seconds": elapsed,
        "environment_steps_per_second": total_steps / elapsed if elapsed > 0 else 0.0,
    }


def evaluate_madsac_fixed_blue_3v3(
    actor: SharedSquashedGaussianActor,
    env_config: str | Path,
    episodes: int,
    num_envs: int = 16,
    num_env_workers: int = 4,
    device: torch.device | None = None,
    seed_start: int = 100000,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    was_training = actor.training
    actor.eval()
    vec_env = make_combat_vector_env_3v3(str(env_config), num_envs, num_env_workers)
    try:
        obs, gs, am = vec_env.reset([{"seed": seed_start + i} for i in range(num_envs)])
        next_seed = seed_start + num_envs
        records: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        while len(records) < episodes:
            red_obs = torch.as_tensor(obs[:, :3, :], device=device)
            with torch.no_grad():
                actions = actor.deterministic(red_obs).cpu().numpy().reshape(num_envs, 3, 3)
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
                    "episode_length": int(result.episode_length[i]),
                    "termination_reason": decode_3v3_termination_reason(int(result.termination_reason_codes[i])),
                    "environment_outcome": decode_3v3_outcome(int(result.outcome_codes[i])),
                })
            if len(done_idx) > 0:
                seeds = [{"seed": next_seed + j} for j in range(len(done_idx))]
                next_seed += len(done_idx)
                no, ng, na = vec_env.reset_at(done_idx, seeds)
                obs[done_idx], gs[done_idx], am[done_idx] = no, ng, na
        return _summarize(records[:episodes], time.perf_counter() - t0)
    finally:
        vec_env.close()
        if was_training:
            actor.train()
