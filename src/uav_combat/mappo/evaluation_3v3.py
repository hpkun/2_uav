"""Parallel MAPPO and rule evaluation for 3v3."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..environment_3v3 import GS_DIM, OBS_DIM
from .networks import GaussianActor
from .vector_env_3v3 import (
    VectorStepResult3v3, make_combat_vector_env_3v3,
    decode_3v3_outcome, decode_3v3_termination_reason,
)


def _summarize(summaries, elapsed):
    n = len(summaries)
    if n == 0: return {"episodes": 0}
    rk = sum(s["red_attack_kills"] for s in summaries)
    bk = sum(s["blue_attack_kills"] for s in summaries)
    def r(key): return sum(1 for s in summaries if s.get(key)) / n
    # Validate outcome rates sum to 1
    red_rate = sum(1 for s in summaries if s.get("environment_outcome") == "red") / n
    blue_rate = sum(1 for s in summaries if s.get("environment_outcome") == "blue") / n
    draw_rate = sum(1 for s in summaries if s.get("environment_outcome") == "draw") / n
    total = red_rate + blue_rate + draw_rate
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(f"Outcome rates sum to {total} != 1: red={red_rate:.4f} blue={blue_rate:.4f} draw={draw_rate:.4f}")
    for s in summaries:
        for team in ("red", "blue"):
            if s[f"{team}_boundary_deaths"] != (
                s[f"{team}_boundary_altitude_deaths"] + s[f"{team}_boundary_xy_deaths"]
            ):
                raise RuntimeError(f"Boundary death mismatch for {team}: {s}")
            if s[f"{team}_attack_kills"] != s[f"{'blue' if team == 'red' else 'red'}_attack_deaths"]:
                raise RuntimeError(f"Attack kill/death mismatch for {team}: {s}")
    total_episode_steps = sum(s["episode_length"] for s in summaries)

    def kill_distribution(team: str) -> dict[str, int]:
        return {str(k): sum(1 for s in summaries if int(s[f"{team}_attack_kills"]) == k) for k in range(4)}

    def step_values(team: str, label: str) -> list[int]:
        key = f"{team}_{label}_attack_kill_step"
        return [int(s[key]) for s in summaries if s.get(key) is not None and int(s[key]) >= 0]

    def mean_or_none(vals: list[int]) -> float | None:
        return float(np.mean(vals)) if vals else None

    def median_or_none(vals: list[int]) -> float | None:
        return float(np.median(vals)) if vals else None

    def neutral_outcome(s: dict[str, Any]) -> str:
        if s["blue_survivors"] == 0 and s["red_survivors"] > 0:
            return "red"
        if s["red_survivors"] == 0 and s["blue_survivors"] > 0:
            return "blue"
        return "draw"

    base = {
        "episodes": n,
        "red_complete_elimination_success_rate": r("red_complete_elimination_success"),
        "blue_complete_elimination_success_rate": r("blue_complete_elimination_success"),
        "environment_red_outcome_rate": sum(1 for s in summaries if s.get("environment_outcome") == "red") / n,
        "environment_blue_outcome_rate": sum(1 for s in summaries if s.get("environment_outcome") == "blue") / n,
        "draw_rate": sum(1 for s in summaries if s.get("environment_outcome") == "draw") / n,
        "mean_red_attack_kills": float(np.mean([s["red_attack_kills"] for s in summaries])),
        "mean_blue_attack_kills": float(np.mean([s["blue_attack_kills"] for s in summaries])),
        "mean_red_attack_deaths": float(np.mean([s["red_attack_deaths"] for s in summaries])),
        "mean_blue_attack_deaths": float(np.mean([s["blue_attack_deaths"] for s in summaries])),
        "mean_red_survivors": float(np.mean([s["red_survivors"] for s in summaries])),
        "mean_blue_survivors": float(np.mean([s["blue_survivors"] for s in summaries])),
        "red_kd_numerator": rk, "red_kd_denominator": bk,
        "red_kd_ratio": rk / bk if bk > 0 else None,
        "mean_red_boundary_deaths": float(np.mean([s["red_boundary_deaths"] for s in summaries])),
        "mean_blue_boundary_deaths": float(np.mean([s["blue_boundary_deaths"] for s in summaries])),
        "mean_red_boundary_altitude_deaths": float(np.mean([s["red_boundary_altitude_deaths"] for s in summaries])),
        "mean_blue_boundary_altitude_deaths": float(np.mean([s["blue_boundary_altitude_deaths"] for s in summaries])),
        "mean_red_boundary_xy_deaths": float(np.mean([s["red_boundary_xy_deaths"] for s in summaries])),
        "mean_blue_boundary_xy_deaths": float(np.mean([s["blue_boundary_xy_deaths"] for s in summaries])),
        "red_altitude_boundary_episode_rate": sum(1 for s in summaries if s["red_boundary_altitude_deaths"] > 0) / n,
        "red_xy_boundary_episode_rate": sum(1 for s in summaries if s["red_boundary_xy_deaths"] > 0) / n,
        "mean_red_friendly_collision_deaths": float(np.mean([s["red_friendly_collision_deaths"] for s in summaries])),
        "mean_blue_friendly_collision_deaths": float(np.mean([s["blue_friendly_collision_deaths"] for s in summaries])),
        "mean_red_cross_collision_deaths": float(np.mean([s["red_cross_collision_deaths"] for s in summaries])),
        "mean_blue_cross_collision_deaths": float(np.mean([s["blue_cross_collision_deaths"] for s in summaries])),
        "mean_red_collision_deaths": float(np.mean([
            s["red_friendly_collision_deaths"] + s["red_cross_collision_deaths"]
            for s in summaries
        ])),
        "mean_blue_collision_deaths": float(np.mean([
            s["blue_friendly_collision_deaths"] + s["blue_cross_collision_deaths"]
            for s in summaries
        ])),
        "max_steps_rate": sum(1 for s in summaries if s.get("termination_reason") == "max_steps") / n,
        "mean_episode_length": float(np.mean([s["episode_length"] for s in summaries])),
        "evaluation_seconds": elapsed,
    }
    base.update({
        "red_any_attack_kill_rate": sum(1 for s in summaries if s.get("red_any_attack_kill")) / n,
        "blue_any_attack_kill_rate": sum(1 for s in summaries if s.get("blue_any_attack_kill")) / n,
        "neutral_rule_red_win_rate": sum(1 for s in summaries if neutral_outcome(s) == "red") / n,
        "neutral_rule_blue_win_rate": sum(1 for s in summaries if neutral_outcome(s) == "blue") / n,
        "neutral_rule_draw_rate": sum(1 for s in summaries if neutral_outcome(s) == "draw") / n,
    })
    for team in ("red", "blue"):
        dist = kill_distribution(team)
        base[f"{team}_attack_kill_count_distribution"] = dist
        base[f"{team}_zero_attack_kill_rate"] = dist["0"] / n
        base[f"{team}_at_least_one_attack_kill_rate"] = sum(int(s[f"{team}_attack_kills"]) >= 1 for s in summaries) / n
        base[f"{team}_at_least_two_attack_kill_rate"] = sum(int(s[f"{team}_attack_kills"]) >= 2 for s in summaries) / n
        base[f"{team}_three_attack_kill_rate"] = dist["3"] / n
        first = step_values(team, "first")
        second = step_values(team, "second")
        third = step_values(team, "third")
        base[f"mean_{team}_first_attack_kill_step"] = mean_or_none(first)
        base[f"median_{team}_first_attack_kill_step"] = median_or_none(first)
        base[f"mean_{team}_second_attack_kill_step"] = mean_or_none(second)
        base[f"mean_{team}_third_attack_kill_step"] = mean_or_none(third)
        remaining = [
            int(s["episode_length"]) - int(s[f"{team}_first_attack_kill_step"])
            for s in summaries
            if s.get(f"{team}_first_attack_kill_step") is not None and int(s[f"{team}_first_attack_kill_step"]) >= 0
        ]
        base[f"mean_{team}_remaining_steps_after_first_kill"] = mean_or_none(remaining)
        denom = total_episode_steps if total_episode_steps > 0 else 1
        for metric in ("r3", "r41", "r42", "attack_window"):
            base[f"{team}_{metric}_active_step_rate" if metric != "attack_window" else f"{team}_attack_window_step_rate"] = (
                sum(int(s.get(f"{team}_{metric}_active_steps" if metric != "attack_window" else f"{team}_attack_window_steps", 0)) for s in summaries) / denom
            )
    return base


def evaluate_mappo_fixed_blue_3v3(
    actor: GaussianActor, env_config: str | Path, episodes: int,
    num_envs: int = 8, num_env_workers: int = 4,
    device: torch.device | None = None, seed_start: int = 100000,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    was_training = actor.training
    actor.eval()
    vec_env = make_combat_vector_env_3v3(str(env_config), num_envs, num_env_workers)
    try:
        specs = [{"seed": seed_start + i} for i in range(num_envs)]
        obs, gs, am = vec_env.reset(specs)
        summaries = []
        action_sum = np.zeros(3, dtype=np.float64)
        action_sat_sum = np.zeros(3, dtype=np.float64)
        action_count = 0
        next_seed = seed_start + num_envs

        t0 = time.perf_counter()
        while len(summaries) < episodes:
            red_obs = obs[:, :3, :].reshape(-1, OBS_DIM)
            with torch.no_grad():
                ra = actor.deterministic_action(torch.as_tensor(red_obs, device=device)).cpu().numpy().reshape(num_envs, 3, 3)
            alive_actions = ra[am[:, :3].astype(bool)]
            if alive_actions.size:
                action_sum += alive_actions.sum(axis=0)
                action_sat_sum += (np.abs(alive_actions) >= 0.95).sum(axis=0)
                action_count += alive_actions.shape[0]
            r = vec_env.step(ra)
            obs, gs, am = r.observations, r.global_states, r.alive_masks
            done_idx = [i for i in range(num_envs) if r.terminated[i] or r.truncated[i]]
            for gi in done_idx:
                if r.episode_valid[gi]:
                    summaries.append({
                        "red_complete_elimination_success": bool(r.red_complete_elimination_success[gi]),
                        "blue_complete_elimination_success": bool(r.blue_complete_elimination_success[gi]),
                        "red_attack_kills": int(r.episode_red_attack_kills[gi]),
                        "blue_attack_kills": int(r.episode_blue_attack_kills[gi]),
                        "red_survivors": int(r.episode_red_survivors[gi]),
                        "blue_survivors": int(r.episode_blue_survivors[gi]),
                        "red_attack_deaths": int(r.episode_red_attack_deaths[gi]),
                        "blue_attack_deaths": int(r.episode_blue_attack_deaths[gi]),
                        "red_boundary_deaths": int(r.episode_red_boundary_deaths[gi]),
                        "blue_boundary_deaths": int(r.episode_blue_boundary_deaths[gi]),
                        "red_boundary_altitude_deaths": int(r.episode_red_boundary_altitude_deaths[gi]),
                        "blue_boundary_altitude_deaths": int(r.episode_blue_boundary_altitude_deaths[gi]),
                        "red_boundary_xy_deaths": int(r.episode_red_boundary_xy_deaths[gi]),
                        "blue_boundary_xy_deaths": int(r.episode_blue_boundary_xy_deaths[gi]),
                        "red_friendly_collision_deaths": int(r.episode_red_friendly_collision_deaths[gi]),
                        "blue_friendly_collision_deaths": int(r.episode_blue_friendly_collision_deaths[gi]),
                        "red_cross_collision_deaths": int(r.episode_red_cross_collision_deaths[gi]),
                        "blue_cross_collision_deaths": int(r.episode_blue_cross_collision_deaths[gi]),
                        "episode_length": int(r.episode_length[gi]),
                        "termination_reason": decode_3v3_termination_reason(int(r.termination_reason_codes[gi])),
                        "environment_outcome": decode_3v3_outcome(int(r.outcome_codes[gi])),
                        "red_any_attack_kill": bool(r.episode_red_any_attack_kill[gi]),
                        "blue_any_attack_kill": bool(r.episode_blue_any_attack_kill[gi]),
                        "red_first_attack_kill_step": None if int(r.episode_red_first_attack_kill_step[gi]) < 0 else int(r.episode_red_first_attack_kill_step[gi]),
                        "blue_first_attack_kill_step": None if int(r.episode_blue_first_attack_kill_step[gi]) < 0 else int(r.episode_blue_first_attack_kill_step[gi]),
                        "red_second_attack_kill_step": None if int(r.episode_red_second_attack_kill_step[gi]) < 0 else int(r.episode_red_second_attack_kill_step[gi]),
                        "blue_second_attack_kill_step": None if int(r.episode_blue_second_attack_kill_step[gi]) < 0 else int(r.episode_blue_second_attack_kill_step[gi]),
                        "red_third_attack_kill_step": None if int(r.episode_red_third_attack_kill_step[gi]) < 0 else int(r.episode_red_third_attack_kill_step[gi]),
                        "blue_third_attack_kill_step": None if int(r.episode_blue_third_attack_kill_step[gi]) < 0 else int(r.episode_blue_third_attack_kill_step[gi]),
                        "red_r3_active_steps": int(r.episode_red_r3_active_steps[gi]),
                        "blue_r3_active_steps": int(r.episode_blue_r3_active_steps[gi]),
                        "red_r41_active_steps": int(r.episode_red_r41_active_steps[gi]),
                        "blue_r41_active_steps": int(r.episode_blue_r41_active_steps[gi]),
                        "red_r42_active_steps": int(r.episode_red_r42_active_steps[gi]),
                        "blue_r42_active_steps": int(r.episode_blue_r42_active_steps[gi]),
                        "red_attack_window_steps": int(r.episode_red_attack_window_steps[gi]),
                        "blue_attack_window_steps": int(r.episode_blue_attack_window_steps[gi]),
                    })
                use_seed = next_seed; next_seed += 1
                no, ng, na = vec_env.reset_at(np.array([gi], dtype=np.int32), [{"seed": use_seed}])
                obs[gi], gs[gi], am[gi] = no[0], ng[0], na[0]

        elapsed = time.perf_counter() - t0
        result = _summarize(summaries[:episodes], elapsed)
        if action_count > 0:
            action_mean = action_sum / action_count
            action_sat = action_sat_sum / action_count
            for dim, name in enumerate(("yaw", "pitch", "speed")):
                result[f"deterministic_action_mean_{name}"] = float(action_mean[dim])
                result[f"deterministic_action_saturation_rate_{name}"] = float(action_sat[dim])
        result["environment_steps_per_second"] = sum(s["episode_length"] for s in summaries[:episodes]) / elapsed if elapsed > 0 else 0.0
    finally:
        vec_env.close()
    if was_training: actor.train()
    return result


def evaluate_rule_matchup_3v3(
    env_config: str | Path, red_mode: str, blue_mode: str,
    episodes: int, num_envs: int = 8, num_env_workers: int = 4,
    seed_start: int = 1000, include_episode_details: bool = False,
) -> dict[str, Any]:
    mode_map = {"zero": 0, "pursuit": 1}
    red_m, blue_m = mode_map[red_mode], mode_map[blue_mode]
    vec_env = make_combat_vector_env_3v3(str(env_config), num_envs, num_env_workers)
    try:
        specs = [{"seed": seed_start + i} for i in range(num_envs)]
        obs, gs, am = vec_env.reset(specs)
        summaries = []
        next_seed = seed_start + num_envs
        modes_arr = np.full((num_envs, 2), [red_m, blue_m], dtype=np.int8)

        t0 = time.perf_counter()
        while len(summaries) < episodes:
            r = vec_env.step_rules(modes_arr)
            obs, gs, am = r.observations, r.global_states, r.alive_masks
            done_idx = [i for i in range(num_envs) if r.terminated[i] or r.truncated[i]]
            for gi in done_idx:
                if r.episode_valid[gi]:
                    summaries.append({
                        "red_complete_elimination_success": bool(r.red_complete_elimination_success[gi]),
                        "blue_complete_elimination_success": bool(r.blue_complete_elimination_success[gi]),
                        "red_attack_kills": int(r.episode_red_attack_kills[gi]),
                        "blue_attack_kills": int(r.episode_blue_attack_kills[gi]),
                        "red_survivors": int(r.episode_red_survivors[gi]),
                        "blue_survivors": int(r.episode_blue_survivors[gi]),
                        "red_attack_deaths": int(r.episode_red_attack_deaths[gi]),
                        "blue_attack_deaths": int(r.episode_blue_attack_deaths[gi]),
                        "red_boundary_deaths": int(r.episode_red_boundary_deaths[gi]),
                        "blue_boundary_deaths": int(r.episode_blue_boundary_deaths[gi]),
                        "red_boundary_altitude_deaths": int(r.episode_red_boundary_altitude_deaths[gi]),
                        "blue_boundary_altitude_deaths": int(r.episode_blue_boundary_altitude_deaths[gi]),
                        "red_boundary_xy_deaths": int(r.episode_red_boundary_xy_deaths[gi]),
                        "blue_boundary_xy_deaths": int(r.episode_blue_boundary_xy_deaths[gi]),
                        "red_friendly_collision_deaths": int(r.episode_red_friendly_collision_deaths[gi]),
                        "blue_friendly_collision_deaths": int(r.episode_blue_friendly_collision_deaths[gi]),
                        "red_cross_collision_deaths": int(r.episode_red_cross_collision_deaths[gi]),
                        "blue_cross_collision_deaths": int(r.episode_blue_cross_collision_deaths[gi]),
                        "episode_length": int(r.episode_length[gi]),
                        "termination_reason": decode_3v3_termination_reason(int(r.termination_reason_codes[gi])),
                        "environment_outcome": decode_3v3_outcome(int(r.outcome_codes[gi])),
                        "red_any_attack_kill": bool(r.episode_red_any_attack_kill[gi]),
                        "blue_any_attack_kill": bool(r.episode_blue_any_attack_kill[gi]),
                        "red_first_attack_kill_step": None if int(r.episode_red_first_attack_kill_step[gi]) < 0 else int(r.episode_red_first_attack_kill_step[gi]),
                        "blue_first_attack_kill_step": None if int(r.episode_blue_first_attack_kill_step[gi]) < 0 else int(r.episode_blue_first_attack_kill_step[gi]),
                        "red_second_attack_kill_step": None if int(r.episode_red_second_attack_kill_step[gi]) < 0 else int(r.episode_red_second_attack_kill_step[gi]),
                        "blue_second_attack_kill_step": None if int(r.episode_blue_second_attack_kill_step[gi]) < 0 else int(r.episode_blue_second_attack_kill_step[gi]),
                        "red_third_attack_kill_step": None if int(r.episode_red_third_attack_kill_step[gi]) < 0 else int(r.episode_red_third_attack_kill_step[gi]),
                        "blue_third_attack_kill_step": None if int(r.episode_blue_third_attack_kill_step[gi]) < 0 else int(r.episode_blue_third_attack_kill_step[gi]),
                        "red_r3_active_steps": int(r.episode_red_r3_active_steps[gi]),
                        "blue_r3_active_steps": int(r.episode_blue_r3_active_steps[gi]),
                        "red_r41_active_steps": int(r.episode_red_r41_active_steps[gi]),
                        "blue_r41_active_steps": int(r.episode_blue_r41_active_steps[gi]),
                        "red_r42_active_steps": int(r.episode_red_r42_active_steps[gi]),
                        "blue_r42_active_steps": int(r.episode_blue_r42_active_steps[gi]),
                        "red_attack_window_steps": int(r.episode_red_attack_window_steps[gi]),
                        "blue_attack_window_steps": int(r.episode_blue_attack_window_steps[gi]),
                    })
                use_seed = next_seed; next_seed += 1
                no, ng, na = vec_env.reset_at(np.array([gi], dtype=np.int32), [{"seed": use_seed}])
                obs[gi], gs[gi], am[gi] = no[0], ng[0], na[0]
        elapsed = time.perf_counter() - t0
        result = _summarize(summaries[:episodes], elapsed)
        if include_episode_details:
            result["episode_details"] = summaries[:episodes]
    finally:
        vec_env.close()
    return result
