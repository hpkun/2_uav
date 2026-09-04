"""Read-only v2.2/61D Vanilla HAPPO nearest mechanism audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.happo.networks import IndependentActors
from env.geometry import PairwiseGeometry, compute_pairwise_geometry
from env.mavuav import (
    BLUE_IDS,
    ENVIRONMENT_VERSION,
    GLOBAL_STATE_DIM,
    OBS_DIM,
    RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv,
    load_environment_config,
)


EXPECTED_ENVIRONMENT_VERSION = "heterogeneous_mavuav_3v2_v2_2"
EXPECTED_OBSERVATION_DIM = 61
EXPECTED_GLOBAL_STATE_DIM = 67
EPISODE_FILENAME = "v22_nearest_episode_mechanism.csv"
EARLY_FILENAME = "v22_nearest_early20_pairs.csv"
SUMMARY_FILENAME = "v22_nearest_seed_summary.csv"
METADATA_FILENAME = "v22_nearest_mechanism_metadata.json"


EPISODE_FIELDS = (
    "training_seed", "evaluation_seed", "outcome", "red_win", "blue_win", "draw",
    "episode_length", "episode_return", "mav_survived", "uav_survivors",
    "red_attack_kills", "blue_attack_kills", "distance_gate", "ata_gate",
    "full_geometry", "streak2", "first_kill", "two_kill", "first_distance_step",
    "first_ata_step", "first_full_geometry_step", "first_streak2_step", "first_kill_step",
    "second_kill_step", "failure_stage", "terminal_cause", "MAV_red_kills",
    "UAV1_red_kills", "UAV2_red_kills", "MAV_attack_event_count",
    "UAV1_attack_event_count", "UAV2_attack_event_count",
)

EARLY_FIELDS = (
    "training_seed", "evaluation_seed", "step", "red_id", "blue_id", "red_alive",
    "blue_alive", "distance", "ATA_deg", "AA_deg", "relative_vx", "relative_vy",
    "relative_vz", "closure_rate", "full_geometry", "attack_streak", "u0", "u1", "u2",
)

SUMMARY_FIELDS = (
    "training_seed", "episodes", "red_win_rate", "blue_win_rate", "draw_rate",
    "mean_red_kills", "MAV_survival", "distance_gate_rate", "ata_gate_rate",
    "full_geometry_rate", "streak2_rate", "first_kill_rate", "two_kill_rate",
    "P_ATA_given_distance", "P_AA_given_distance_ATA", "P_streak2_given_full_geometry",
    "P_first_kill_given_streak2", "mean_first_distance_step", "mean_first_ata_step",
    "mean_first_full_geometry_step", "mean_first_kill_step", "kill_share_MAV",
    "kill_share_UAV1", "kill_share_UAV2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="v2.2/61D Vanilla run directory; repeat for multiple seeds")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_checkpoint(run_dir: Path) -> Path:
    run = run_dir.expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory not found: {run}")
    for name in ("checkpoint_2000000.pt", "checkpoint_final.pt"):
        checkpoint = run / name
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError(f"no checkpoint_2000000.pt or checkpoint_final.pt in {run}")


def validate_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    actual = {
        "environment_version": payload.get("environment_version"),
        "observation_dim": payload.get("observation_dim"),
        "global_state_dim": payload.get("global_state_dim"),
        "actor_variant": payload.get("actor_variant", trainer_config.get("actor_variant")),
        "method_variant": payload.get("method_variant", trainer_config.get("method_variant")),
    }
    expected = {
        "environment_version": EXPECTED_ENVIRONMENT_VERSION,
        "observation_dim": EXPECTED_OBSERVATION_DIM,
        "global_state_dim": EXPECTED_GLOBAL_STATE_DIM,
        "actor_variant": "vanilla",
        "method_variant": "baseline",
    }
    if actual != expected:
        raise RuntimeError(f"incompatible v2.2 nearest-audit checkpoint: expected={expected}, actual={actual}")
    if ENVIRONMENT_VERSION != EXPECTED_ENVIRONMENT_VERSION or OBS_DIM != EXPECTED_OBSERVATION_DIM or GLOBAL_STATE_DIM != EXPECTED_GLOBAL_STATE_DIM:
        raise RuntimeError("current source tree is not the required v2.2/61D/67D environment")
    return actual


def gate_flags(
    distance: float,
    ata_rad: float,
    aa_rad: float,
    distance_limits: tuple[float, float] = (1000.0, 3000.0),
    ata_deg_limit: float = 30.0,
    aa_deg_limit: float = 90.0,
) -> tuple[bool, bool, bool]:
    distance_ok = float(distance_limits[0]) <= float(distance) <= float(distance_limits[1])
    ata_ok = distance_ok and float(ata_rad) < math.radians(float(ata_deg_limit))
    full = ata_ok and float(aa_rad) < math.radians(float(aa_deg_limit))
    return distance_ok, ata_ok, full


def closure_rate(relative_position: np.ndarray, relative_velocity: np.ndarray) -> float:
    position = np.asarray(relative_position, dtype=np.float64)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    distance = float(np.linalg.norm(position))
    if distance <= 1e-12:
        return 0.0
    return float(-np.dot(velocity, position / distance))


def record_first(first_steps: dict[str, float], event: str, occurred: bool, step: int) -> None:
    if occurred and math.isnan(first_steps[event]):
        first_steps[event] = float(step)


def classify_failure(row: Mapping[str, Any]) -> str:
    if row["outcome"] == "red":
        return "WIN"
    if not row["distance_gate"]:
        return "F1_distance"
    if not row["ata_gate"]:
        return "F2_ATA"
    if not row["full_geometry"]:
        return "F3_AA"
    if not row["streak2"]:
        return "F4_hold1"
    if not row["first_kill"]:
        return "F5_kill"
    return "F6_second_kill"


def _resolved_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _mean_finite(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else float("nan")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _terminal_cause(env: HeterogeneousMAVUAVAirCombatEnv, outcome: str, truncated: bool) -> str:
    if outcome == "red":
        return "red_two_kill"
    mav = env.entities["MAV"]
    if not mav.state.alive:
        return f"MAV_{mav.inactive_cause or 'inactive'}"
    if any(env.entities[blue].inactive_cause == "blue_escape" for blue in BLUE_IDS):
        return "Blue_escape"
    if any(env.entities[red].inactive_cause == "boundary" for red in RED_IDS):
        return "Red_boundary"
    if truncated:
        return "timeout"
    return "other_terminal"


def _early_pair_row(
    training_seed: int,
    evaluation_seed: int,
    step: int,
    red_id: str,
    blue_id: str,
    geometry: PairwiseGeometry,
    action: np.ndarray,
    env: HeterogeneousMAVUAVAirCombatEnv,
) -> dict[str, Any]:
    _, _, geometric_full = gate_flags(
        geometry.distance,
        geometry.ata,
        geometry.aa,
        tuple(env.config["combat"]["distance"]),
        float(env.config["combat"]["ata_deg"]),
        float(env.config["combat"]["aa_deg"]),
    )
    return {
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "step": step,
        "red_id": red_id,
        "blue_id": blue_id,
        "red_alive": int(env.entities[red_id].state.alive),
        "blue_alive": int(env.entities[blue_id].state.alive),
        "distance": geometry.distance,
        "ATA_deg": math.degrees(geometry.ata),
        "AA_deg": math.degrees(geometry.aa),
        "relative_vx": float(geometry.relative_velocity[0]),
        "relative_vy": float(geometry.relative_velocity[1]),
        "relative_vz": float(geometry.relative_velocity[2]),
        "closure_rate": closure_rate(geometry.relative_position, geometry.relative_velocity),
        "full_geometry": int(
            env.entities[red_id].state.alive
            and env.entities[blue_id].state.alive
            and geometric_full
        ),
        "attack_streak": int(env._attack_streak.get((red_id, blue_id), 0)),
        "u0": float(action[0]),
        "u1": float(action[1]),
        "u2": float(action[2]),
    }


def evaluate_run(
    run_dir: Path,
    episodes: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = select_checkpoint(run_dir)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    contract = validate_checkpoint_contract(payload)
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    training_seed = int(trainer_config["seed"])
    actors = IndependentActors(hidden_dim=int(trainer_config["hidden_dim"])).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    environment_config = load_environment_config(payload.get("environment_config"))
    combat = environment_config["combat"]
    episode_rows: list[dict[str, Any]] = []
    early_rows: list[dict[str, Any]] = []

    env = HeterogeneousMAVUAVAirCombatEnv(
        environment_config,
        blue_target_mode="nearest",
        profile="main",
    )
    for episode_index in range(int(episodes)):
        evaluation_seed = 1000 + episode_index
        observations, _ = env.reset(seed=evaluation_seed, options={"profile": "main"})
        first_steps = {
            "distance": float("nan"),
            "ata": float("nan"),
            "full_geometry": float("nan"),
            "streak2": float("nan"),
            "kill": float("nan"),
            "second_kill": float("nan"),
        }
        ever = {"distance": False, "ata": False, "full_geometry": False, "streak2": False}
        red_kill_credit = {red: 0.0 for red in RED_IDS}
        red_attack_events = {red: 0 for red in RED_IDS}
        done = False
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        while not done:
            actions = []
            with torch.no_grad():
                for agent_index, red_id in enumerate(RED_IDS):
                    actor_observation = torch.as_tensor(observations[red_id], device=device).unsqueeze(0)
                    action, _ = actors.actors[agent_index].sample(actor_observation, deterministic=True)
                    actions.append(action.squeeze(0).cpu().numpy())
            action_array = np.asarray(actions, dtype=np.float64)
            if env.step_count < 20:
                for red_index, red_id in enumerate(RED_IDS):
                    for blue_id in BLUE_IDS:
                        geometry = compute_pairwise_geometry(
                            env.entities[red_id].state,
                            env.entities[blue_id].state,
                        )
                        early_rows.append(_early_pair_row(
                            training_seed,
                            evaluation_seed,
                            env.step_count,
                            red_id,
                            blue_id,
                            geometry,
                            action_array[red_index],
                            env,
                        ))

            alive_before = {
                entity_id: bool(env.entities[entity_id].state.alive)
                for entity_id in (*RED_IDS, *BLUE_IDS)
            }
            observations, _, terminated, truncated, info = env.step(action_array)
            event_step = int(env.step_count)
            step_distance = step_ata = step_full = False
            for red_id in RED_IDS:
                for blue_id in BLUE_IDS:
                    if not (alive_before[red_id] and alive_before[blue_id]):
                        continue
                    geometry = compute_pairwise_geometry(
                        env.entities[red_id].state,
                        env.entities[blue_id].state,
                    )
                    distance_ok, ata_ok, full = gate_flags(
                        geometry.distance,
                        geometry.ata,
                        geometry.aa,
                        tuple(combat["distance"]),
                        float(combat["ata_deg"]),
                        float(combat["aa_deg"]),
                    )
                    step_distance |= distance_ok
                    step_ata |= ata_ok
                    step_full |= full
            ever["distance"] |= step_distance
            ever["ata"] |= step_ata
            ever["full_geometry"] |= step_full
            step_streak2 = any(
                env._attack_streak.get((red_id, blue_id), 0) >= 2
                for red_id in RED_IDS for blue_id in BLUE_IDS
            )
            ever["streak2"] |= step_streak2
            record_first(first_steps, "distance", step_distance, event_step)
            record_first(first_steps, "ata", step_ata, event_step)
            record_first(first_steps, "full_geometry", step_full, event_step)
            record_first(first_steps, "streak2", step_streak2, event_step)

            red_events_by_target: dict[str, list[str]] = defaultdict(list)
            for event in info.get("attack_events", []):
                attacker, target = str(event["attacker"]), str(event["target"])
                if attacker in RED_IDS and target in BLUE_IDS:
                    red_attack_events[attacker] += 1
                    red_events_by_target[target].append(attacker)
            for target, attackers in red_events_by_target.items():
                if info.get("death_causes", {}).get(target) != "red_attack":
                    continue
                credit = 1.0 / len(attackers)
                for attacker in attackers:
                    red_kill_credit[attacker] += credit
            red_kill_count = len(env._red_attack_kills)
            record_first(first_steps, "kill", red_kill_count >= 1, event_step)
            record_first(first_steps, "second_kill", red_kill_count >= 2, event_step)
            done = bool(terminated or truncated)

        summary = info["episode_summary"]
        total_kill_credit = sum(red_kill_credit.values())
        if not np.isclose(total_kill_credit, float(summary["red_attack_kills"])):
            raise RuntimeError(
                "Red attacker credit does not reconcile to environment red_attack_kills: "
                f"credit={total_kill_credit}, kills={summary['red_attack_kills']}"
            )
        row: dict[str, Any] = {
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "outcome": summary["outcome"],
            "red_win": int(summary["outcome"] == "red"),
            "blue_win": int(summary["outcome"] == "blue"),
            "draw": int(summary["outcome"] == "draw"),
            "episode_length": int(summary["episode_length"]),
            "episode_return": float(summary["episode_return"]),
            "mav_survived": int(summary["mav_survived"]),
            "uav_survivors": int(summary["red_uav_survivors"]),
            "red_attack_kills": int(summary["red_attack_kills"]),
            "blue_attack_kills": int(summary["blue_attack_kills"]),
            "distance_gate": int(ever["distance"]),
            "ata_gate": int(ever["ata"]),
            "full_geometry": int(ever["full_geometry"]),
            "streak2": int(ever["streak2"]),
            "first_kill": int(summary["red_attack_kills"] >= 1),
            "two_kill": int(summary["red_attack_kills"] >= 2),
            "first_distance_step": first_steps["distance"],
            "first_ata_step": first_steps["ata"],
            "first_full_geometry_step": first_steps["full_geometry"],
            "first_streak2_step": first_steps["streak2"],
            "first_kill_step": first_steps["kill"],
            "second_kill_step": first_steps["second_kill"],
            "terminal_cause": _terminal_cause(env, str(summary["outcome"]), bool(truncated)),
        }
        for red_id in RED_IDS:
            row[f"{red_id}_red_kills"] = red_kill_credit[red_id]
            row[f"{red_id}_attack_event_count"] = red_attack_events[red_id]
        row["failure_stage"] = classify_failure(row)
        episode_rows.append(row)

    run_metadata = {
        "run_path": str(run_dir.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "sampled_steps": int(payload.get("sampled_steps", 0)),
        "training_seed": training_seed,
        "training_profile": str(payload.get("environment_profile", trainer_config.get("environment_profile", "unknown"))),
        "combat_thresholds": {
            "distance_m": list(combat["distance"]),
            "ATA_deg_strict_less_than": float(combat["ata_deg"]),
            "AA_deg_strict_less_than": float(combat["aa_deg"]),
            "hold_steps": int(combat["hold_steps"]),
        },
        **contract,
    }
    return episode_rows, early_rows, run_metadata


def summarize_seed(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero episodes")
    episodes = len(rows)
    distance = sum(int(row["distance_gate"]) for row in rows)
    ata = sum(int(row["ata_gate"]) for row in rows)
    full = sum(int(row["full_geometry"]) for row in rows)
    streak2 = sum(int(row["streak2"]) for row in rows)
    first_kill = sum(int(row["first_kill"]) for row in rows)
    two_kill = sum(int(row["two_kill"]) for row in rows)
    total_credit = sum(float(row[f"{red}_red_kills"]) for row in rows for red in RED_IDS)
    shares = {
        red: (
            sum(float(row[f"{red}_red_kills"]) for row in rows) / total_credit
            if total_credit > 0.0 else float("nan")
        )
        for red in RED_IDS
    }
    return {
        "training_seed": int(rows[0]["training_seed"]),
        "episodes": episodes,
        "red_win_rate": sum(int(row["red_win"]) for row in rows) / episodes,
        "blue_win_rate": sum(int(row["blue_win"]) for row in rows) / episodes,
        "draw_rate": sum(int(row["draw"]) for row in rows) / episodes,
        "mean_red_kills": float(np.mean([float(row["red_attack_kills"]) for row in rows])),
        "MAV_survival": float(np.mean([float(row["mav_survived"]) for row in rows])),
        "distance_gate_rate": distance / episodes,
        "ata_gate_rate": ata / episodes,
        "full_geometry_rate": full / episodes,
        "streak2_rate": streak2 / episodes,
        "first_kill_rate": first_kill / episodes,
        "two_kill_rate": two_kill / episodes,
        "P_ATA_given_distance": _ratio(ata, distance),
        "P_AA_given_distance_ATA": _ratio(full, ata),
        "P_streak2_given_full_geometry": _ratio(streak2, full),
        "P_first_kill_given_streak2": _ratio(first_kill, streak2),
        "mean_first_distance_step": _mean_finite(rows, "first_distance_step"),
        "mean_first_ata_step": _mean_finite(rows, "first_ata_step"),
        "mean_first_full_geometry_step": _mean_finite(rows, "first_full_geometry_step"),
        "mean_first_kill_step": _mean_finite(rows, "first_kill_step"),
        "kill_share_MAV": shares["MAV"],
        "kill_share_UAV1": shares["UAV1"],
        "kill_share_UAV2": shares["UAV2"],
    }


def run_audit(
    run_dirs: list[Path],
    episodes: int,
    device_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    device = _resolved_device(device_name)
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_episode_rows: list[dict[str, Any]] = []
    all_early_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for run_dir in run_dirs:
        episode_rows, early_rows, metadata = evaluate_run(run_dir, episodes, device)
        seed = int(metadata["training_seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate training seed in --run inputs: {seed}")
        seen_seeds.add(seed)
        all_episode_rows.extend(episode_rows)
        all_early_rows.extend(early_rows)
        run_metadata.append(metadata)
    all_episode_rows.sort(key=lambda row: (int(row["training_seed"]), int(row["evaluation_seed"])))
    all_early_rows.sort(key=lambda row: (
        int(row["training_seed"]), int(row["evaluation_seed"]), int(row["step"]),
        RED_IDS.index(str(row["red_id"])), BLUE_IDS.index(str(row["blue_id"])),
    ))
    grouped = defaultdict(list)
    for row in all_episode_rows:
        grouped[int(row["training_seed"])].append(row)
    summary_rows = [summarize_seed(grouped[seed]) for seed in sorted(grouped)]

    paths = {
        "episodes": output / EPISODE_FILENAME,
        "early20": output / EARLY_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "metadata": output / METADATA_FILENAME,
    }
    _write_csv(paths["episodes"], all_episode_rows, EPISODE_FIELDS)
    _write_csv(paths["early20"], all_early_rows, EARLY_FIELDS)
    _write_csv(paths["summary"], summary_rows, SUMMARY_FIELDS)
    combat_thresholds = run_metadata[0]["combat_thresholds"]
    if any(item["combat_thresholds"] != combat_thresholds for item in run_metadata[1:]):
        raise RuntimeError("run checkpoints use different combat thresholds")
    metadata = {
        "environment_version": EXPECTED_ENVIRONMENT_VERSION,
        "observation_dim": EXPECTED_OBSERVATION_DIM,
        "global_state_dim": EXPECTED_GLOBAL_STATE_DIM,
        "runs": run_metadata,
        "episodes_per_run": int(episodes),
        "evaluation_seed_range": [1000, 1000 + int(episodes) - 1],
        "profile": "main",
        "blue_mode": "nearest",
        "deterministic": True,
        "device": str(device),
        "combat_thresholds": combat_thresholds,
        "gate_definitions": {
            "distance_gate": "episode-level any alive-before-step Red-Blue pair with 1000 <= post-action distance <= 3000 m",
            "ata_gate": "same post-action pair and step satisfies distance gate and ATA < 30 deg",
            "full_geometry": "same post-action pair and step satisfies distance, ATA < 30 deg and AA < 90 deg",
            "streak2": "any Red-Blue pair reaches environment attack streak >= 2",
            "first_kill": "at least one Blue has inactive cause red_attack",
            "two_kill": "both Blue aircraft have inactive cause red_attack",
            "first_event_steps": "1-based post-action decision step; absent events are NaN",
        },
        "early20_semantics": "steps 0..19 use pre-action geometry and the deterministic normalized action selected at that state; all 3 Red x 2 Blue pairs are retained while the episode is active",
        "closure_rate_definition": "-dot(Blue_velocity - Red_velocity, relative_position / distance); positive means closing",
        "attacker_kill_credit": "fractional credit among simultaneous Red attack events killing the same Blue; attack_event_count is unmodified",
        "legacy_comparability": {
            "exact": [
                "all_Red episode-level distance/ATA/full-geometry/streak2/first-kill rates",
                "all_Red conditional gate conversions",
                "two-kill rate from red_attack kill count",
                "fractional attacker kill share and unmodified attack-event count",
            ],
            "not_exact": [
                "legacy dominant_interceptor scope (not selected by this tool)",
                "legacy aggregated early_kinematics windows (this tool emits all pair-level pre-action rows)",
                "legacy first-step implementation cannot be source-verified because its generating script is absent; the new tool uses the apparent 1-based post-action convention",
                "terminal_cause is a new explicit field",
            ],
        },
        "compatibility_note": "v2.2 current code only; no legacy v2.1 checkpoint compatibility",
    }
    with paths["metadata"].open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    return paths


def main() -> None:
    args = parse_args()
    paths = run_audit(args.run, args.episodes, args.device, args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
