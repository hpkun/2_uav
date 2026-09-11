"""Reset-only foundation audit for the canonical v3.5 4v4 environment."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.geometry import compute_pairwise_geometry
from env.mavuav import (
    BLUE_IDS, ENTITY_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    TYPE_BY_ID, HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)


QUANTILES = (
    ("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("median", 0.5),
    ("p95", 0.95), ("p99", 0.99), ("max", 1.0),
)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("foundation quantiles require non-empty finite values")
    return {name: float(np.quantile(array, probability)) for name, probability in QUANTILES}


def _distribution(values: list[int], maximum: int = 4) -> dict[str, dict[str, float | int]]:
    counts = Counter(values)
    total = len(values)
    return {
        str(value): {"count": int(counts[value]), "fraction": float(counts[value] / total)}
        for value in range(maximum + 1)
    }


def _nominal_geometry(config: Mapping[str, Any]) -> dict[str, float]:
    initial = config["scenario"]["initial"]
    rear_offset = float(initial["UAV2"]["position"][0] - initial["MAV"]["position"][0])
    red_uav_x = float(np.mean([initial[aid]["position"][0] for aid in RED_IDS[1:]]))
    blue_x = float(np.mean([initial[aid]["position"][0] for aid in BLUE_IDS]))
    separation = blue_x - red_uav_x
    return {"MAV_rear_offset_m": rear_offset, "red_uav_blue_longitudinal_separation_m": separation}


def audit_foundations(
    profile: str = "main",
    samples: int = 10_000,
    seed: int = 1000,
    env_config: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate seeded initial-state properties without retaining raw reset states."""
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    config = load_environment_config(env_config)
    if profile not in config["randomization_profiles"]:
        raise ValueError(f"unknown randomization profile: {profile}")
    env = HeterogeneousMAVUAVAirCombatEnv(config, profile=profile, randomize=True)
    all_distances: list[float] = []
    mav_distances: list[float] = []
    uav_distances: list[float] = []
    direct_counts: dict[str, list[int]] = {aid: [] for aid in RED_IDS}
    team_visible_counts: list[int] = []
    minimum_rear_offsets: list[float] = []
    speeds: dict[str, list[float]] = {"MAV": [], "Red_UAV": [], "Blue_UAV": []}
    speed_clipping = Counter({"MAV": 0, "Red_UAV": 0, "Blue_UAV": 0})
    distance_violation_count = full_geometry_count = hold_condition_count = 0
    combat = config["combat"]
    attack_min, attack_max = (float(value) for value in combat["distance"])
    ata_limit = np.deg2rad(float(combat["ata_deg"]))
    aa_limit = np.deg2rad(float(combat["aa_deg"]))

    for sample in range(int(samples)):
        env.reset(seed=int(seed) + sample, options={"profile": profile, "randomize": True})
        rear_offsets = [env.entities[aid].state.x - env.entities["MAV"].state.x for aid in RED_IDS[1:]]
        minimum_rear_offsets.append(float(min(rear_offsets)))
        for red_id in RED_IDS:
            direct_counts[red_id].append(sum(env.direct_visible(red_id, blue_id) for blue_id in BLUE_IDS))
        team_visible_counts.append(sum(env.team_visible(blue_id) for blue_id in BLUE_IDS))

        sample_distance_violation = sample_full_geometry = False
        for red_id in RED_IDS:
            for blue_id in BLUE_IDS:
                red_to_blue = compute_pairwise_geometry(
                    env.entities[red_id].state, env.entities[blue_id].state,
                )
                distance = float(red_to_blue.distance)
                all_distances.append(distance)
                (mav_distances if red_id == "MAV" else uav_distances).append(distance)
                sample_distance_violation |= distance <= attack_max
                blue_to_red = compute_pairwise_geometry(
                    env.entities[blue_id].state, env.entities[red_id].state,
                )
                sample_full_geometry |= (
                    attack_min <= distance <= attack_max
                    and (
                        (red_to_blue.ata < ata_limit and red_to_blue.aa < aa_limit)
                        or (blue_to_red.ata < ata_limit and blue_to_red.aa < aa_limit)
                    )
                )
        distance_violation_count += int(sample_distance_violation)
        full_geometry_count += int(sample_full_geometry)
        hold_condition_count += int(any(
            streak >= int(combat["hold_steps"]) for streak in env._attack_streak.values()
        ))

        for aircraft_id in ENTITY_IDS:
            entity = env.entities[aircraft_id]
            group = "MAV" if aircraft_id == "MAV" else "Red_UAV" if aircraft_id in RED_IDS else "Blue_UAV"
            speeds[group].append(float(entity.state.v))
            speed_clipping[group] += int(
                entity.state.v <= entity.spec.v_min or entity.state.v >= entity.spec.v_max
            )

    distance_stats = {
        "all_Red_Blue": _quantiles(all_distances),
        "MAV_Blue": _quantiles(mav_distances),
        "UAV_Blue": _quantiles(uav_distances),
    }
    nominal = _nominal_geometry(config)
    sensing = config["sensing"]
    randomization = config["randomization_profiles"][profile]
    normalization = config["normalization"]
    battlefield = config["battlefield"]
    mav_all = sum(value == 4 for value in direct_counts["MAV"])
    team_all = sum(value == 4 for value in team_visible_counts)
    nominal_speed = float(config["scenario"]["initial"]["MAV"]["speed"])
    summary = {
        "environment_version": ENVIRONMENT_VERSION,
        "observation_dim": OBS_DIM,
        "global_state_dim": GLOBAL_STATE_DIM,
        "profile": profile,
        "samples": int(samples),
        "seed_start": int(seed),
        "seed_end": int(seed) + int(samples) - 1,
        "nominal_parameters": {
            "initial": config["scenario"]["initial"],
            "sensing": config["sensing"],
            "aircraft_specs": config["aircraft_specs"],
            "combat": config["combat"],
            "simulation": config["simulation"],
            "randomization": randomization,
            "normalization": config["normalization"],
            "MAV_rear_offset_m": nominal["MAV_rear_offset_m"],
        },
        "initial_pair_distance_m": distance_stats,
        "initial_sensing": {
            "direct_visible_Blue_count_distribution": {
                red_id: _distribution(direct_counts[red_id]) for red_id in RED_IDS
            },
            "team_visible_Blue_count_distribution": _distribution(team_visible_counts),
            "P_MAV_sees_all_4": float(mav_all / samples),
            "P_team_sees_all_4": float(team_all / samples),
        },
        "formation": {
            "P_MAV_behind_all_UAVs": float(np.mean(np.asarray(minimum_rear_offsets) > 0.0)),
            "minimum_MAV_rear_offset_m": float(min(minimum_rear_offsets)),
            "mean_MAV_rear_offset_m": float(np.mean(minimum_rear_offsets)),
            "MAV_rear_offset_quantiles_m": _quantiles(minimum_rear_offsets),
            "rear_offset_definition": "per-reset minimum UAV_x minus MAV_x across UAV1/UAV2/UAV3",
        },
        "initial_speed_m_per_s": {
            group: _quantiles(values) for group, values in speeds.items()
        },
        "speed_clipping": {
            "count": int(sum(speed_clipping.values())),
            "count_by_type": dict(speed_clipping),
            "definition": "initial speed at or beyond its aircraft type speed boundary",
        },
        "initial_combat_legality": {
            "samples_with_any_Red_Blue_distance_le_attack_max": int(distance_violation_count),
            "samples_with_any_directed_pair_full_attack_geometry": int(full_geometry_count),
            "samples_with_any_pair_at_hold_condition": int(hold_condition_count),
        },
        "dimensionless_ratios": {
            "MAV_sensor_over_median_MAV_Blue_distance": float(sensing["MAV_range"] / distance_stats["MAV_Blue"]["median"]),
            "UAV_sensor_over_median_UAV_Blue_distance": float(sensing["UAV_range"] / distance_stats["UAV_Blue"]["median"]),
            "MAV_sensor_over_attack_max": float(sensing["MAV_range"] / attack_max),
            "UAV_sensor_over_attack_max": float(sensing["UAV_range"] / attack_max),
            "attack_max_over_median_all_initial_distance": float(attack_max / distance_stats["all_Red_Blue"]["median"]),
            "rear_offset_over_nominal_team_separation": float(
                nominal["MAV_rear_offset_m"] / nominal["red_uav_blue_longitudinal_separation_m"]
            ),
        },
        "randomization_scale_ratios": {
            "team_xy_jitter_over_nominal_team_separation": float(
                randomization["team_xy_jitter"] / nominal["red_uav_blue_longitudinal_separation_m"]
            ),
            "slot_xy_jitter_over_MAV_rear_offset": float(
                randomization["slot_xy_jitter"] / nominal["MAV_rear_offset_m"]
            ),
            "altitude_jitter_over_nominal_altitude": float(
                randomization["altitude_jitter"] / config["scenario"]["initial"]["MAV"]["position"][2]
            ),
            "speed_jitter_over_nominal_speed": float(randomization["speed_jitter"] / nominal_speed),
            "heading_jitter_over_combat_ATA": float(randomization["heading_jitter_deg"] / combat["ata_deg"]),
        },
        "normalization_scale_ratios": {
            "self_xy_scale_over_horizontal_half_extent": float(
                normalization["self_xy_scale"] / max(abs(value) for value in battlefield["x"])
            ),
            "relative_xy_scale_over_MAV_sensor": float(normalization["relative_xy_scale"] / sensing["MAV_range"]),
            "distance_scale_over_MAV_sensor": float(normalization["distance_scale"] / sensing["MAV_range"]),
            "relative_altitude_scale_over_altitude_span": float(
                normalization["relative_altitude_scale"] / (battlefield["altitude"][1] - battlefield["altitude"][0])
            ),
            "relative_velocity_scale_over_max_cross_team_speed_sum": float(
                normalization["relative_velocity_scale"]
                / (config["aircraft_specs"]["MAV"]["v_max"] + config["aircraft_specs"]["Blue"]["v_max"])
            ),
        },
        "v32_v33_analytical_comparison": {
            "v3_2": {
                "MAV_nominal_speed_m_per_s": 325.0,
                "UAV_nominal_speed_m_per_s": 225.0,
                "MAV_rear_offset_m": 500.0,
                "MAV_relative_catch_up_time_s": 5.0,
                "nominal_time_to_attack_max_s": float((8000.0 - attack_max) / (225.0 + 225.0)),
            },
            "v3_3": {
                "all_nominal_speeds_m_per_s": 275.0,
                "MAV_rear_offset_m": 1000.0,
                "MAV_relative_speed_m_per_s": 0.0,
                "MAV_relative_catch_up_time_s": None,
                "nominal_time_to_attack_max_s": float((8000.0 - attack_max) / (275.0 + 275.0)),
            },
        },
    }
    return summary


def write_summary(output: str | Path, summary: Mapping[str, Any]) -> Path:
    requested = Path(output).expanduser().resolve()
    path = requested if requested.suffix.lower() == ".json" else requested / "foundation_summary.json"
    if path.exists():
        raise FileExistsError(f"foundation audit output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main", "learnability"), default="main")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_foundations(args.profile, args.samples, args.seed)
    path = write_summary(args.output, summary)
    print(json.dumps({"foundation_summary": str(path), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
