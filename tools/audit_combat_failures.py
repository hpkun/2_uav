"""Episode-level failure audit for canonical v3.2 Vanilla HAPPO checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithm.happo.networks import IndependentActors
from env.blue_policy import BLUE_ACTION_CANDIDATES
from env.dynamics import integrate_interval
from env.geometry import compute_pairwise_geometry
from env.mavuav import (
    BLUE_IDS, ENTITY_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.reward import situation_reward


EPISODE_FIELDS = (
    "episode", "seed", "policy_mode", "outcome", "episode_length", "episode_return",
    "red_attack_kills", "blue_attack_kills", "mav_survived", "red_uav_survivors",
    "first_kill_step", "second_kill_step", "third_kill_step", "fourth_kill_step",
    "final_alive_blue_count", "final_alive_red_count", "remaining_blue_id",
    "post_first_remaining_blue_visible_fraction", "post_second_remaining_blue_visible_fraction",
    "post_third_remaining_blue_visible_fraction", "steps_after_last_kill",
    "tail_visible_fraction", "tail_longest_all_invisible_streak", "tail_far_fraction",
    "tail_distance_window_fraction", "tail_ATA_ok_fraction", "tail_AA_ok_fraction",
    "tail_full_geometry_fraction", "tail_mean_best_ATA_deg", "tail_mean_best_AA_deg",
    "tail_mean_best_closing_rate", "tail_fraction_positive_closing", "tail_max_streak",
    "final_max_attack_streak", "post_third_steps_after_kill", "post_third_visible_fraction",
    "post_third_longest_invisible_streak", "post_third_min_distance",
    "post_third_mean_distance", "post_third_final_distance",
    "post_third_mean_best_closing_rate", "post_third_fraction_positive_closing",
    "post_third_distance_window_fraction", "post_third_ATA_ok_fraction",
    "post_third_AA_ok_fraction", "post_third_full_geometry_fraction",
    "post_third_mean_best_ATA_deg", "post_third_mean_best_AA_deg",
    "post_third_max_streak", "post_third_steps_streak_ge_1", "post_third_steps_streak_ge_2",
    "post_third_mean_speed", "post_third_min_speed", "post_third_max_speed",
    "post_third_mean_altitude", "post_third_min_altitude", "post_third_max_altitude",
    "post_third_mean_theta", "post_third_mean_heading",
    "recovery_candidate_count", "recovery_one_step_rejected_count",
    "recovery_altitude_rejected_count", "recovery_changed_choice_count",
    "recovery_changed_choice_fraction", "max_recovery_guard",
    "post_third_recovery_changed_choice_count", "post_third_recovery_changed_choice_fraction",
    "post_third_max_recovery_guard", "reward_credit_samples",
    "reward_best_equals_closest_count", "reward_best_equals_closest_fraction",
    "reward_best_closing_comparable_samples", "reward_best_equals_best_closing_count",
    "reward_best_equals_best_closing_fraction", "post_last_reward_best_equals_closest_fraction",
    "post_last_reward_best_equals_best_closing_fraction", "failure_classification",
)

FAILURE_THRESHOLDS = {
    "late_kill_step": 60,
    "target_loss_longest_invisible_streak": 10,
    "cannot_close_far_fraction": 0.5,
    "cannot_close_positive_closing_fraction": 0.5,
    "bad_geometry_distance_window_fraction": 0.25,
}


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _fraction(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def _longest_false_streak(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = 0 if flag else current + 1
        longest = max(longest, current)
    return longest


def _action_array(actors: Any, observations: Mapping[str, np.ndarray], device: str | torch.device,
                  policy_mode: str) -> np.ndarray:
    deterministic = policy_mode == "deterministic"
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for index, red_id in enumerate(RED_IDS):
            observation = torch.as_tensor(observations[red_id], device=device).unsqueeze(0)
            action, _ = actors.actors[index].sample(observation, deterministic=deterministic)
            actions.append(action.squeeze(0).cpu().numpy())
    return np.asarray(actions, dtype=np.float32)


def blue_action_diagnostics(env: HeterogeneousMAVUAVAirCombatEnv, blue_id: str) -> dict[str, Any]:
    """Replicate Blue candidate scoring without changing the policy or environment."""
    blue = env.entities[blue_id]
    red = {red_id: env.entities[red_id] for red_id in RED_IDS}
    target = env.blue_policy.select_target(blue, red)
    if not blue.state.alive or target is None:
        zero = np.zeros(3, dtype=np.float64)
        return {
            "raw_greedy_action": zero, "actual_safe_greedy_action": zero,
            "one_step_rejected_count": 0, "altitude_rejected_count": 0,
            "recovery_changed_choice": False,
            "recovery_guard": env.blue_policy._altitude_recovery_guard(blue.state, blue),
        }
    raw_action = safe_action = None
    raw_score = safe_score = -np.inf
    one_step_rejected = altitude_rejected = 0
    for candidate in BLUE_ACTION_CANDIDATES:
        predicted = integrate_interval(blue.state, candidate, blue.spec, env.physics_dt, env.physics_substeps)
        if not env.blue_policy._within_battlefield(predicted):
            one_step_rejected += 1
            continue
        score = situation_reward(predicted, target.state)
        if score > raw_score:
            raw_score, raw_action = score, candidate
        if not env.blue_policy._has_safe_altitude_recovery(predicted, blue):
            altitude_rejected += 1
            continue
        if score > safe_score:
            safe_score, safe_action = score, candidate
    if raw_action is None or safe_action is None:
        raise RuntimeError(f"Blue audit found no valid candidate for {blue_id}")
    return {
        "raw_greedy_action": raw_action.copy(),
        "actual_safe_greedy_action": safe_action.copy(),
        "one_step_rejected_count": one_step_rejected,
        "altitude_rejected_count": altitude_rejected,
        "recovery_changed_choice": not np.array_equal(raw_action, safe_action),
        "recovery_guard": env.blue_policy._altitude_recovery_guard(blue.state, blue),
    }


def blue_geometry_diagnostics(
    env: HeterogeneousMAVUAVAirCombatEnv,
    blue_id: str,
    previous_distances: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Measure real same-pair geometry, closing, reward credit and attack streak."""
    blue = env.entities[blue_id]
    combat = env.config["combat"]
    pairs: list[dict[str, Any]] = []
    for red_id in RED_IDS:
        red = env.entities[red_id]
        if not red.state.alive:
            continue
        geometry = compute_pairwise_geometry(red.state, blue.state)
        previous = previous_distances.get((red_id, blue_id))
        closing = None if previous is None else (float(previous) - geometry.distance) / env.decision_dt
        distance_ok = combat["distance"][0] <= geometry.distance <= combat["distance"][1]
        ata_ok = geometry.ata < np.deg2rad(combat["ata_deg"])
        aa_ok = geometry.aa < np.deg2rad(combat["aa_deg"])
        pairs.append({
            "red_id": red_id, "distance": geometry.distance, "closing_rate": closing,
            "ata": geometry.ata, "aa": geometry.aa, "distance_ok": distance_ok,
            "ata_ok": ata_ok, "aa_ok": aa_ok,
            "full_geometry_ok": distance_ok and ata_ok and aa_ok,
            "reward": situation_reward(red.state, blue.state),
            "streak": int(env._attack_streak.get((red_id, blue_id), 0)),
        })
    if not pairs:
        return {"blue_id": blue_id, "team_visible": False, "pairs": []}
    closest = min(pairs, key=lambda pair: pair["distance"])
    closing_pairs = [pair for pair in pairs if pair["closing_rate"] is not None]
    best_closing = max(closing_pairs, key=lambda pair: pair["closing_rate"]) if closing_pairs else None
    reward_best = max(pairs, key=lambda pair: pair["reward"])
    best_combat = max(
        pairs,
        key=lambda pair: (
            int(pair["full_geometry_ok"]),
            int(pair["distance_ok"]) + int(pair["ata_ok"]) + int(pair["aa_ok"]),
            pair["reward"],
        ),
    )
    return {
        "blue_id": blue_id, "team_visible": bool(env.team_visible(blue_id)),
        "min_distance": closest["distance"], "closest_red_id": closest["red_id"],
        "best_closing_red_id": None if best_closing is None else best_closing["red_id"],
        "best_closing_rate": None if best_closing is None else best_closing["closing_rate"],
        "reward_best_red_id": reward_best["red_id"],
        "reward_best_equals_closest": reward_best["red_id"] == closest["red_id"],
        "reward_best_equals_best_closing": None if best_closing is None else reward_best["red_id"] == best_closing["red_id"],
        "best_ata": min(pair["ata"] for pair in pairs),
        "best_aa": min(pair["aa"] for pair in pairs),
        "best_combat_pair_id": best_combat["red_id"],
        "any_pair_in_distance_window": any(pair["distance_ok"] for pair in pairs),
        "any_pair_ATA_ok": any(pair["ata_ok"] for pair in pairs),
        "any_pair_AA_ok": any(pair["aa_ok"] for pair in pairs),
        "any_pair_full_geometry_ok": any(pair["full_geometry_ok"] for pair in pairs),
        "max_red_attack_streak": max(pair["streak"] for pair in pairs),
        "speed": float(blue.state.v), "altitude": float(blue.state.h),
        "theta": float(blue.state.theta), "heading": float(blue.state.psi), "pairs": pairs,
    }


def _record_red_kill_steps(kill_steps: list[int], info: Mapping[str, Any], step: int) -> None:
    killed_blue = sorted(
        entity for entity, cause in info.get("death_causes", {}).items()
        if entity in BLUE_IDS and cause == "red_attack"
    )
    kill_steps.extend([int(step)] * len(killed_blue))


def _phase_records(records: list[dict[str, Any]], after_step: int, blue_id: str | None = None) -> list[dict[str, Any]]:
    return [
        record for record in records
        if record["decision_step"] > after_step and (blue_id is None or record["blue_id"] == blue_id)
    ]


def _step_visible_fraction(records: list[dict[str, Any]]) -> float | None:
    by_step: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        by_step[int(record["decision_step"])].append(bool(record["team_visible"]))
    return _mean([float(np.mean(flags)) for flags in by_step.values()])


def _classify_draw(row: Mapping[str, Any]) -> str:
    last_kill = max(
        [int(value) for value in (row.get("first_kill_step"), row.get("second_kill_step"),
                                  row.get("third_kill_step"), row.get("fourth_kill_step")) if value is not None],
        default=0,
    )
    if last_kill >= FAILURE_THRESHOLDS["late_kill_step"] and (
        int(row["final_max_attack_streak"]) >= 2 or float(row.get("tail_full_geometry_fraction") or 0.0) > 0.0
    ):
        return "LATE_PROGRESS / POSSIBLE_HORIZON"
    if int(row["tail_longest_all_invisible_streak"]) >= FAILURE_THRESHOLDS["target_loss_longest_invisible_streak"]:
        return "TARGET_LOSS"
    if (
        float(row.get("tail_visible_fraction") or 0.0) >= 0.5
        and float(row.get("tail_far_fraction") or 0.0) >= FAILURE_THRESHOLDS["cannot_close_far_fraction"]
        and float(row.get("tail_fraction_positive_closing") or 0.0) < FAILURE_THRESHOLDS["cannot_close_positive_closing_fraction"]
    ):
        return "CANNOT_CLOSE"
    if (
        float(row.get("tail_distance_window_fraction") or 0.0) >= FAILURE_THRESHOLDS["bad_geometry_distance_window_fraction"]
        and float(row.get("tail_full_geometry_fraction") or 0.0) == 0.0
    ):
        return "BAD_GEOMETRY"
    if int(row["tail_max_streak"]) in (1, 2) or float(row.get("tail_full_geometry_fraction") or 0.0) > 0.0:
        return "STREAK_INTERRUPTED"
    return "OTHER"


def _episode_row(
    episode: int,
    episode_seed: int,
    policy_mode: str,
    env: HeterogeneousMAVUAVAirCombatEnv,
    final_info: Mapping[str, Any],
    kill_steps: list[int],
    step_records: list[dict[str, Any]],
    recovery_records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = final_info["episode_summary"]
    final_blues = [blue_id for blue_id in BLUE_IDS if env.entities[blue_id].state.alive]
    final_reds = [red_id for red_id in RED_IDS if env.entities[red_id].state.alive]
    padded_kills: list[int | None] = (kill_steps + [None] * 4)[:4]
    last_kill = kill_steps[-1] if kill_steps else 0
    tail = _phase_records(step_records, last_kill)
    tail_recovery = [record for record in recovery_records if record["decision_step"] > last_kill]
    tail_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in tail:
        tail_by_step[int(record["decision_step"])].append(record)
    tail_step_visible = [any(record["team_visible"] for record in records) for records in tail_by_step.values()]
    closing = [record["best_closing_rate"] for record in tail if record["best_closing_rate"] is not None]
    comparable = [record for record in step_records if record["best_closing_red_id"] is not None]
    post_last_comparable = [record for record in tail if record["best_closing_red_id"] is not None]
    remaining_blue = final_blues[0] if len(final_blues) == 1 else None
    post_third = _phase_records(step_records, padded_kills[2], remaining_blue) if padded_kills[2] is not None and remaining_blue else []
    post_third_recovery = [
        record for record in recovery_records
        if padded_kills[2] is not None and remaining_blue and record["decision_step"] > padded_kills[2]
        and record["blue_id"] == remaining_blue
    ]
    post_third_closing = [record["best_closing_rate"] for record in post_third if record["best_closing_rate"] is not None]
    final_distance = None
    if remaining_blue and final_reds:
        final_distance = min(
            compute_pairwise_geometry(env.entities[red_id].state, env.entities[remaining_blue].state).distance
            for red_id in final_reds
        )
    final_max_streak = max(
        (int(env._attack_streak.get((red_id, blue_id), 0)) for red_id in final_reds for blue_id in final_blues),
        default=0,
    )
    row: dict[str, Any] = {
        "episode": episode, "seed": episode_seed, "policy_mode": policy_mode,
        "outcome": summary["outcome"], "episode_length": summary["episode_length"],
        "episode_return": summary["episode_return"], "red_attack_kills": summary["red_attack_kills"],
        "blue_attack_kills": summary["blue_attack_kills"], "mav_survived": int(summary["mav_survived"]),
        "red_uav_survivors": summary["red_uav_survivors"],
        "first_kill_step": padded_kills[0], "second_kill_step": padded_kills[1],
        "third_kill_step": padded_kills[2], "fourth_kill_step": padded_kills[3],
        "final_alive_blue_count": len(final_blues), "final_alive_red_count": len(final_reds),
        "remaining_blue_id": remaining_blue, "steps_after_last_kill": summary["episode_length"] - last_kill,
        "post_first_remaining_blue_visible_fraction": _step_visible_fraction(_phase_records(step_records, padded_kills[0])) if padded_kills[0] is not None else None,
        "post_second_remaining_blue_visible_fraction": _step_visible_fraction(_phase_records(step_records, padded_kills[1])) if padded_kills[1] is not None else None,
        "post_third_remaining_blue_visible_fraction": _step_visible_fraction(_phase_records(step_records, padded_kills[2])) if padded_kills[2] is not None else None,
        "tail_visible_fraction": _step_visible_fraction(tail),
        "tail_longest_all_invisible_streak": _longest_false_streak(tail_step_visible),
        "tail_far_fraction": _fraction([record["min_distance"] > env.config["combat"]["distance"][1] for record in tail]),
        "tail_distance_window_fraction": _fraction([record["any_pair_in_distance_window"] for record in tail]),
        "tail_ATA_ok_fraction": _fraction([record["any_pair_ATA_ok"] for record in tail]),
        "tail_AA_ok_fraction": _fraction([record["any_pair_AA_ok"] for record in tail]),
        "tail_full_geometry_fraction": _fraction([record["any_pair_full_geometry_ok"] for record in tail]),
        "tail_mean_best_ATA_deg": _mean([np.rad2deg(record["best_ata"]) for record in tail]),
        "tail_mean_best_AA_deg": _mean([np.rad2deg(record["best_aa"]) for record in tail]),
        "tail_mean_best_closing_rate": _mean(closing),
        "tail_fraction_positive_closing": _fraction([value > 0.0 for value in closing]),
        "tail_max_streak": max((record["max_red_attack_streak"] for record in tail), default=0),
        "final_max_attack_streak": final_max_streak,
        "post_third_steps_after_kill": summary["episode_length"] - padded_kills[2] if padded_kills[2] is not None else None,
        "post_third_visible_fraction": _step_visible_fraction(post_third),
        "post_third_longest_invisible_streak": _longest_false_streak([record["team_visible"] for record in post_third]) if post_third else None,
        "post_third_min_distance": min((record["min_distance"] for record in post_third), default=None),
        "post_third_mean_distance": _mean([record["min_distance"] for record in post_third]),
        "post_third_final_distance": final_distance,
        "post_third_mean_best_closing_rate": _mean(post_third_closing),
        "post_third_fraction_positive_closing": _fraction([value > 0.0 for value in post_third_closing]),
        "post_third_distance_window_fraction": _fraction([record["any_pair_in_distance_window"] for record in post_third]),
        "post_third_ATA_ok_fraction": _fraction([record["any_pair_ATA_ok"] for record in post_third]),
        "post_third_AA_ok_fraction": _fraction([record["any_pair_AA_ok"] for record in post_third]),
        "post_third_full_geometry_fraction": _fraction([record["any_pair_full_geometry_ok"] for record in post_third]),
        "post_third_mean_best_ATA_deg": _mean([np.rad2deg(record["best_ata"]) for record in post_third]),
        "post_third_mean_best_AA_deg": _mean([np.rad2deg(record["best_aa"]) for record in post_third]),
        "post_third_max_streak": max((record["max_red_attack_streak"] for record in post_third), default=None),
        "post_third_steps_streak_ge_1": sum(record["max_red_attack_streak"] >= 1 for record in post_third) if post_third else None,
        "post_third_steps_streak_ge_2": sum(record["max_red_attack_streak"] >= 2 for record in post_third) if post_third else None,
        "post_third_mean_speed": _mean([record["speed"] for record in post_third]),
        "post_third_min_speed": min((record["speed"] for record in post_third), default=None),
        "post_third_max_speed": max((record["speed"] for record in post_third), default=None),
        "post_third_mean_altitude": _mean([record["altitude"] for record in post_third]),
        "post_third_min_altitude": min((record["altitude"] for record in post_third), default=None),
        "post_third_max_altitude": max((record["altitude"] for record in post_third), default=None),
        "post_third_mean_theta": _mean([record["theta"] for record in post_third]),
        "post_third_mean_heading": _mean([record["heading"] for record in post_third]),
        "recovery_candidate_count": len(recovery_records) * len(BLUE_ACTION_CANDIDATES),
        "recovery_one_step_rejected_count": sum(record["one_step_rejected_count"] for record in recovery_records),
        "recovery_altitude_rejected_count": sum(record["altitude_rejected_count"] for record in recovery_records),
        "recovery_changed_choice_count": sum(record["recovery_changed_choice"] for record in recovery_records),
        "recovery_changed_choice_fraction": _fraction([record["recovery_changed_choice"] for record in recovery_records]),
        "max_recovery_guard": max((record["recovery_guard"] for record in recovery_records), default=None),
        "post_third_recovery_changed_choice_count": sum(record["recovery_changed_choice"] for record in post_third_recovery) if post_third_recovery else None,
        "post_third_recovery_changed_choice_fraction": _fraction([record["recovery_changed_choice"] for record in post_third_recovery]),
        "post_third_max_recovery_guard": max((record["recovery_guard"] for record in post_third_recovery), default=None),
        "reward_credit_samples": len(step_records),
        "reward_best_equals_closest_count": sum(record["reward_best_equals_closest"] for record in step_records),
        "reward_best_equals_closest_fraction": _fraction([record["reward_best_equals_closest"] for record in step_records]),
        "reward_best_closing_comparable_samples": len(comparable),
        "reward_best_equals_best_closing_count": sum(record["reward_best_equals_best_closing"] for record in comparable),
        "reward_best_equals_best_closing_fraction": _fraction([record["reward_best_equals_best_closing"] for record in comparable]),
        "post_last_reward_best_equals_closest_fraction": _fraction([record["reward_best_equals_closest"] for record in tail]),
        "post_last_reward_best_equals_best_closing_fraction": _fraction([record["reward_best_equals_best_closing"] for record in post_last_comparable]),
    }
    row["failure_classification"] = _classify_draw(row) if summary["outcome"] == "draw" else ""
    return row


def summarize_audit(rows: list[dict[str, Any]], policy_mode: str, seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one audit episode is required")
    n = len(rows)
    kills = [int(row["red_attack_kills"]) for row in rows]
    histogram = Counter(kills)
    by_outcome = {
        outcome: {str(kill): sum(row["outcome"] == outcome and int(row["red_attack_kills"]) == kill for row in rows) for kill in range(5)}
        for outcome in ("draw", "blue", "red")
    }
    if any(row["outcome"] == "red" and int(row["red_attack_kills"]) != 4 for row in rows):
        raise AssertionError("Red-win audit rows must contain exactly four Red attack kills")
    three_kill_draws = [row for row in rows if row["outcome"] == "draw" and int(row["red_attack_kills"]) == 3]
    credit_samples = sum(int(row["reward_credit_samples"]) for row in rows)
    closing_samples = sum(int(row["reward_best_closing_comparable_samples"]) for row in rows)
    recovery_samples = sum(int(row["recovery_candidate_count"]) // len(BLUE_ACTION_CANDIDATES) for row in rows)
    return {
        "policy_mode": policy_mode, "episodes": n, "seed_start": seed, "seed_end": seed + n - 1,
        "red_win_rate": sum(row["outcome"] == "red" for row in rows) / n,
        "blue_win_rate": sum(row["outcome"] == "blue" for row in rows) / n,
        "draw_rate": sum(row["outcome"] == "draw" for row in rows) / n,
        "mean_red_kills": _mean([row["red_attack_kills"] for row in rows]),
        "mean_blue_kills": _mean([row["blue_attack_kills"] for row in rows]),
        "mean_episode_length": _mean([row["episode_length"] for row in rows]),
        "mean_return": _mean([row["episode_return"] for row in rows]),
        "MAV_survival": _mean([row["mav_survived"] for row in rows]),
        "mean_UAV_survivors": _mean([row["red_uav_survivors"] for row in rows]),
        "red_kill_histogram": {str(kill): histogram.get(kill, 0) for kill in range(5)},
        "draw_kill_histogram": by_outcome["draw"],
        "blue_win_kill_histogram": by_outcome["blue"],
        "red_win_kill_histogram": by_outcome["red"],
        "mean_first_kill_step": _mean([row["first_kill_step"] for row in rows]),
        "mean_second_kill_step": _mean([row["second_kill_step"] for row in rows]),
        "mean_third_kill_step": _mean([row["third_kill_step"] for row in rows]),
        "mean_fourth_kill_step": _mean([row["fourth_kill_step"] for row in rows]),
        "three_kill_draw": {
            "count": len(three_kill_draws),
            "mean_third_kill_step": _mean([row["third_kill_step"] for row in three_kill_draws]),
            "mean_steps_after_third": _mean([row["post_third_steps_after_kill"] for row in three_kill_draws]),
            "mean_post_third_visible_fraction": _mean([row["post_third_visible_fraction"] for row in three_kill_draws]),
            "mean_longest_invisible_streak": _mean([row["post_third_longest_invisible_streak"] for row in three_kill_draws]),
            "mean_min_distance": _mean([row["post_third_min_distance"] for row in three_kill_draws]),
            "mean_final_distance": _mean([row["post_third_final_distance"] for row in three_kill_draws]),
            "mean_best_closing_rate": _mean([row["post_third_mean_best_closing_rate"] for row in three_kill_draws]),
            "mean_max_streak": _mean([row["post_third_max_streak"] for row in three_kill_draws]),
        },
        "recovery": {
            "changed_choice_fraction": sum(int(row["recovery_changed_choice_count"]) for row in rows) / max(recovery_samples, 1),
            "max_guard": max(float(row["max_recovery_guard"]) for row in rows if row["max_recovery_guard"] is not None),
        },
        "reward_credit": {
            "reward_best_equals_closest_fraction": sum(int(row["reward_best_equals_closest_count"]) for row in rows) / max(credit_samples, 1),
            "reward_best_equals_best_closing_fraction": sum(int(row["reward_best_equals_best_closing_count"]) for row in rows) / max(closing_samples, 1),
        },
        "task_completion_structure": {
            "P_K_ge_1": sum(kill >= 1 for kill in kills) / n,
            "P_K_ge_2": sum(kill >= 2 for kill in kills) / n,
            "P_K_ge_3": sum(kill >= 3 for kill in kills) / n,
            "P_K_eq_4": sum(kill == 4 for kill in kills) / n,
            "P_Blue_win": sum(row["outcome"] == "blue" for row in rows) / n,
        },
        "failure_classification_counts": dict(sorted(Counter(
            row["failure_classification"] for row in rows if row["outcome"] == "draw"
        ).items())),
        "failure_classification_by_kill_count": {
            str(kill): dict(sorted(Counter(
                row["failure_classification"] for row in rows
                if row["outcome"] == "draw" and int(row["red_attack_kills"]) == kill
            ).items()))
            for kill in range(4)
        },
        "failure_classification_thresholds": dict(FAILURE_THRESHOLDS),
        "definitions": {
            "closing_rate": "(previous pair distance - current pair distance) / decision_dt; positive means closing",
            "remaining_blue_visible_fraction": "mean per-decision fraction of currently alive Blue aircraft for which env.team_visible is true, strictly after the indexed Red kill",
            "tail": "decision steps strictly after the episode's last Red kill; the whole episode when no Red kill occurs",
            "full_geometry_ok": "one identical Red-Blue pair simultaneously satisfies distance, ATA and AA thresholds",
        },
    }


def audit_actors(
    actors: Any,
    env_config: str | Path | Mapping[str, Any] | None,
    episodes: int,
    profile: str,
    seed: int = 1000,
    device: str | torch.device = "cpu",
    policy_mode: str = "deterministic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    if policy_mode not in ("deterministic", "stochastic"):
        raise ValueError("policy_mode must be deterministic or stochastic")
    env = HeterogeneousMAVUAVAirCombatEnv(env_config, profile=profile)
    rows: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        episode_seed = int(seed) + episode
        if policy_mode == "stochastic":
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
        observations, _ = env.reset(seed=episode_seed)
        previous_distances: dict[tuple[str, str], float] = {}
        kill_steps: list[int] = []
        step_records: list[dict[str, Any]] = []
        recovery_records: list[dict[str, Any]] = []
        done = False
        final_info: dict[str, Any] = {}
        while not done:
            decision_step = env.step_count + 1
            for blue_id in BLUE_IDS:
                if not env.entities[blue_id].state.alive:
                    continue
                geometry = blue_geometry_diagnostics(env, blue_id, previous_distances)
                geometry["decision_step"] = decision_step
                step_records.append(geometry)
                recovery = blue_action_diagnostics(env, blue_id)
                recovery.update({"decision_step": decision_step, "blue_id": blue_id})
                recovery_records.append(recovery)
            previous_distances = {
                (red_id, blue_id): compute_pairwise_geometry(env.entities[red_id].state, env.entities[blue_id].state).distance
                for red_id in RED_IDS for blue_id in BLUE_IDS
                if env.entities[red_id].state.alive and env.entities[blue_id].state.alive
            }
            actions = _action_array(actors, observations, device, policy_mode)
            observations, _, terminated, truncated, final_info = env.step(actions)
            _record_red_kill_steps(kill_steps, final_info, env.step_count)
            done = bool(terminated or truncated)
        if len(kill_steps) != int(final_info["episode_summary"]["red_attack_kills"]):
            raise AssertionError("Red kill-step count does not match the environment episode summary")
        rows.append(_episode_row(
            episode, episode_seed, policy_mode, env, final_info, kill_steps, step_records, recovery_records,
        ))
    return rows, summarize_audit(rows, policy_mode, int(seed))


def load_vanilla_baseline_checkpoint(
    checkpoint: str | Path, device: str | torch.device,
) -> tuple[IndependentActors, dict[str, Any], torch.device]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = torch.device("cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    expected = (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM)
    if actual != expected:
        raise RuntimeError(f"incompatible checkpoint environment contract: expected={expected!r}, actual={actual!r}")
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    contract = (
        payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla")),
        payload.get("critic_variant", trainer_config.get("critic_variant", "mlp")),
        payload.get("method_variant", trainer_config.get("method_variant", "baseline")),
    )
    if contract != ("vanilla", "mlp", "baseline"):
        raise RuntimeError(
            "failure audit requires actor_variant='vanilla', critic_variant='mlp', method_variant='baseline'; "
            f"got {contract!r}"
        )
    actors = IndependentActors(hidden_dim=int(trainer_config["hidden_dim"])).to(resolved)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    return actors, payload, resolved


def write_audit(output: str | Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "audit_episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EPISODE_FIELDS} for row in rows)
    json_path = output_dir / "audit_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    actors, payload, device = load_vanilla_baseline_checkpoint(args.checkpoint, args.device)
    env_config = load_environment_config(payload.get("environment_config"))
    rows, summary = audit_actors(
        actors, env_config, args.episodes, args.profile, seed=args.seed,
        device=device, policy_mode=args.policy_mode,
    )
    summary.update({
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sampled_steps": int(payload.get("sampled_steps", 0)),
        "environment_version": ENVIRONMENT_VERSION,
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla", "critic_variant": "mlp", "method_variant": "baseline",
        "profile": args.profile, "device": str(device),
    })
    csv_path, json_path = write_audit(args.output, rows, summary)
    print(json.dumps({"episodes_csv": str(csv_path), "summary_json": str(json_path), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
