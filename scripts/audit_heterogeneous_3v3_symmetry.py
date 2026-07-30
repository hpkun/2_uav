"""Rule-only audit for functional heterogeneous 3v3 mirror symmetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv
from uav_combat.mappo.vector_env_3v3 import (
    decode_3v3_outcome,
    decode_3v3_termination_reason,
    make_combat_vector_env_3v3,
)
from uav_combat.scenario_3v3 import BLUE_IDS, RED_IDS


SPEC_FIELDS = (
    "v_min", "v_max", "theta_min", "theta_max", "nx_min", "nx_max",
    "nz_min", "nz_max", "phi_max", "yaw_rate_max", "pitch_rate_max",
    "acceleration_max", "k_yaw", "k_pitch", "k_speed",
)


def _heading_opposite(a: float, b: float, atol: float = 1e-9) -> bool:
    diff = abs(a - b)
    diff = min(diff, 2.0 * np.pi - diff)
    return bool(np.isclose(diff, np.pi, atol=atol))


def _check_mirror(env: Homogeneous3v3AirCombatEnv, seed: int) -> list[str]:
    env.reset(seed)
    failures: list[str] = []
    for rid, bid in zip(RED_IDS, BLUE_IDS):
        red = env._aircraft_by_id(rid)
        blue = env._aircraft_by_id(bid)
        if not np.allclose(
            [red.state.x, red.state.y, red.state.z],
            [-blue.state.x, -blue.state.y, blue.state.z],
            atol=1e-9,
        ):
            failures.append(f"{rid}/{bid} position is not mirrored")
        if not np.isclose(red.state.v, blue.state.v, atol=1e-9):
            failures.append(f"{rid}/{bid} speed differs")
        if not np.isclose(red.state.theta, blue.state.theta, atol=1e-9):
            failures.append(f"{rid}/{bid} theta differs")
        if not _heading_opposite(red.state.psi, blue.state.psi):
            failures.append(f"{rid}/{bid} heading is not opposite")
        if red.state.alive != blue.state.alive:
            failures.append(f"{rid}/{bid} alive differs")
        if red.role != blue.role:
            failures.append(f"{rid}/{bid} role differs")
        if red.sensor_range != blue.sensor_range:
            failures.append(f"{rid}/{bid} sensor_range differs")
        if red.can_attack != blue.can_attack:
            failures.append(f"{rid}/{bid} can_attack differs")
        for field in SPEC_FIELDS:
            if getattr(red.spec, field) != getattr(blue.spec, field):
                failures.append(f"{rid}/{bid} spec.{field} differs")
    return failures


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in records])) if records else 0.0


def _sum(records: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(r[key]) for r in records))


def _validate_ledger(record: dict[str, Any]) -> bool:
    for team in ("red", "blue"):
        total = (
            int(record[f"{team}_survivors"])
            + int(record[f"{team}_attack_deaths"])
            + int(record[f"{team}_boundary_deaths"])
            + int(record[f"{team}_friendly_collision_deaths"])
            + int(record[f"{team}_cross_collision_deaths"])
        )
        if total != 3:
            return False
    if int(record["red_attack_kills"]) != int(record["blue_attack_deaths"]):
        return False
    if int(record["blue_attack_kills"]) != int(record["red_attack_deaths"]):
        return False
    return True


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.env_config)
    preflight_env = Homogeneous3v3AirCombatEnv(args.env_config)
    seeds = [int(args.seed + i) for i in range(args.episodes)]
    mirror_failures: dict[int, list[str]] = {}
    for seed in seeds:
        failures = _check_mirror(preflight_env, seed)
        if failures:
            mirror_failures[seed] = failures
    if mirror_failures:
        raise RuntimeError(f"mirror preflight failed: {mirror_failures}")

    backend = "local" if args.env_workers == 1 else "worker"
    vec = make_combat_vector_env_3v3(args.env_config, args.num_envs, args.env_workers)
    records: list[dict[str, Any]] = []
    finite_failures = 0
    ledger_failures = 0
    worker_failures = 0
    try:
        obs, gs, am = vec.reset([{"seed": args.seed + i} for i in range(args.num_envs)])
        next_seed = args.seed + args.num_envs
        modes = np.ones((args.num_envs, 2), dtype=np.int8)
        while len(records) < args.episodes:
            result = vec.step_rules(modes)
            arrays = (result.observations, result.global_states, result.team_rewards, result.red_reward_components)
            if not all(np.all(np.isfinite(arr)) for arr in arrays):
                finite_failures += 1
            done_idx = np.where(result.episode_valid)[0]
            for idx in done_idx:
                rec = {
                    "termination_reason": decode_3v3_termination_reason(int(result.termination_reason_codes[idx])),
                    "outcome": decode_3v3_outcome(int(result.outcome_codes[idx])),
                    "red_complete_elimination_success": bool(result.red_complete_elimination_success[idx]),
                    "blue_complete_elimination_success": bool(result.blue_complete_elimination_success[idx]),
                    "red_attack_kills": int(result.episode_red_attack_kills[idx]),
                    "blue_attack_kills": int(result.episode_blue_attack_kills[idx]),
                    "red_survivors": int(result.episode_red_survivors[idx]),
                    "blue_survivors": int(result.episode_blue_survivors[idx]),
                    "red_attack_deaths": int(result.episode_red_attack_deaths[idx]),
                    "blue_attack_deaths": int(result.episode_blue_attack_deaths[idx]),
                    "red_boundary_deaths": int(result.episode_red_boundary_deaths[idx]),
                    "blue_boundary_deaths": int(result.episode_blue_boundary_deaths[idx]),
                    "red_friendly_collision_deaths": int(result.episode_red_friendly_collision_deaths[idx]),
                    "blue_friendly_collision_deaths": int(result.episode_blue_friendly_collision_deaths[idx]),
                    "red_cross_collision_deaths": int(result.episode_red_cross_collision_deaths[idx]),
                    "blue_cross_collision_deaths": int(result.episode_blue_cross_collision_deaths[idx]),
                    "red_support_survived": bool(result.episode_red_support_survived[idx]),
                    "blue_support_survived": bool(result.episode_blue_support_survived[idx]),
                    "red_mean_support_coverage_ratio": float(result.episode_red_mean_support_coverage_ratio[idx]),
                    "blue_mean_support_coverage_ratio": float(result.episode_blue_mean_support_coverage_ratio[idx]),
                    "red_kills_with_shared_observation": int(result.episode_red_kills_with_shared_observation[idx]),
                    "blue_kills_with_shared_observation": int(result.episode_blue_kills_with_shared_observation[idx]),
                    "episode_length": int(result.episode_length[idx]),
                }
                if not _validate_ledger(rec):
                    ledger_failures += 1
                records.append(rec)
                if len(records) >= args.episodes:
                    break
            if len(done_idx) > 0:
                reset_specs = [{"seed": next_seed + j} for j in range(len(done_idx))]
                next_seed += len(done_idx)
                vec.reset_at(done_idx, reset_specs)
    except Exception:
        worker_failures += 1
        raise
    finally:
        vec.close()

    attack_max = float(cfg["combat"]["attack_distance_max"])
    combat_sensor = float(cfg["heterogeneous"]["sensor_range"]["combat"])
    summary = {
        "audit_version": "functional_heterogeneous_3v3_symmetry_v1",
        "env_config": str(args.env_config),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed),
        "episode_seeds": seeds,
        "num_envs": int(args.num_envs),
        "env_workers": int(args.env_workers),
        "backend": backend,
        "mirror_preflight_passed": True,
        "role_mapping": cfg["heterogeneous"]["roles"],
        "sensor_ranges": cfg["heterogeneous"]["sensor_range"],
        "attack_distance_max": attack_max,
        "support_to_combat": bool(cfg["heterogeneous"]["information_sharing"]["support_to_combat"]),
        "support_rule": cfg["heterogeneous"]["support_rule"],
        "shared_observation_kills_structurally_expected_zero": bool(combat_sensor >= attack_max),
        "timeout_note": "v6-style timeout is encoded as blue environment outcome, not blue complete attack elimination.",
        "termination": {
            "max_steps_count": sum(r["termination_reason"] == "max_steps" for r in records),
            "red_elimination_count": sum(r["termination_reason"] == "red_elimination" for r in records),
            "blue_elimination_count": sum(r["termination_reason"] == "blue_elimination" for r in records),
            "mutual_elimination_count": sum(r["termination_reason"] == "mutual_elimination" for r in records),
        },
        "outcomes": {
            "red_outcome_count": sum(r["outcome"] == "red" for r in records),
            "blue_outcome_count": sum(r["outcome"] == "blue" for r in records),
            "draw_count": sum(r["outcome"] == "draw" for r in records),
        },
        "complete_attack_elimination": {
            "red_complete_elimination_success_count": sum(r["red_complete_elimination_success"] for r in records),
            "blue_complete_elimination_success_count": sum(r["blue_complete_elimination_success"] for r in records),
        },
        "combat_statistics": {
            "mean_red_attack_kills": _mean(records, "red_attack_kills"),
            "mean_blue_attack_kills": _mean(records, "blue_attack_kills"),
            "mean_red_survivors": _mean(records, "red_survivors"),
            "mean_blue_survivors": _mean(records, "blue_survivors"),
            "mean_red_boundary_deaths": _mean(records, "red_boundary_deaths"),
            "mean_blue_boundary_deaths": _mean(records, "blue_boundary_deaths"),
            "mean_red_collision_deaths": _mean(records, "red_friendly_collision_deaths") + _mean(records, "red_cross_collision_deaths"),
            "mean_blue_collision_deaths": _mean(records, "blue_friendly_collision_deaths") + _mean(records, "blue_cross_collision_deaths"),
            "mean_episode_length": _mean(records, "episode_length"),
        },
        "heterogeneous_statistics": {
            "red_support_survival_rate": _mean(records, "red_support_survived"),
            "blue_support_survival_rate": _mean(records, "blue_support_survived"),
            "mean_red_support_coverage_ratio": _mean(records, "red_mean_support_coverage_ratio"),
            "mean_blue_support_coverage_ratio": _mean(records, "blue_mean_support_coverage_ratio"),
            "red_shared_observation_kills": _sum(records, "red_kills_with_shared_observation"),
            "blue_shared_observation_kills": _sum(records, "blue_kills_with_shared_observation"),
        },
        "integrity": {
            "finite_failures": int(finite_failures),
            "ledger_failures": int(ledger_failures),
            "worker_failures": int(worker_failures),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_3v3_functional_v1.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--env-workers", type=int, default=2)
    parser.add_argument("--output-json", default="outputs/audits/heterogeneous_3v3_symmetry_seed42000.json")
    args = parser.parse_args()
    summary = run_audit(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
