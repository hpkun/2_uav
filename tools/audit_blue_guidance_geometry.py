"""Audit v3.5 periodic Blue guidance geometry with deterministic Red scripts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.geometry import compute_pairwise_geometry
from env.mavuav import BLUE_IDS, ENVIRONMENT_VERSION, RED_IDS, HeterogeneousMAVUAVAirCombatEnv


MANEUVERS = {
    "straight": np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
    "constant_left_turn": np.asarray((0.0, 0.0, 0.65), dtype=np.float64),
    "constant_right_turn": np.asarray((0.0, 0.0, -0.65), dtype=np.float64),
    "climb_turn": np.asarray((0.0, 0.55, 0.55), dtype=np.float64),
    "dive_turn": np.asarray((0.0, -0.55, -0.55), dtype=np.float64),
}


def _summary(samples: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "samples": 0, "ATA_median_deg": None, "ATA_p25_deg": None, "ATA_p75_deg": None,
            "AA_median_deg": None, "AA_p25_deg": None, "AA_p75_deg": None,
            "fraction_ATA_lt_30": None, "fraction_AA_lt_90": None,
            "fraction_full_angle_gate": None,
        }
    ata = np.asarray([sample["ata_deg"] for sample in samples], dtype=np.float64)
    aa = np.asarray([sample["aa_deg"] for sample in samples], dtype=np.float64)
    return {
        "samples": len(samples),
        "ATA_median_deg": float(np.median(ata)),
        "ATA_p25_deg": float(np.percentile(ata, 25)),
        "ATA_p75_deg": float(np.percentile(ata, 75)),
        "AA_median_deg": float(np.median(aa)),
        "AA_p25_deg": float(np.percentile(aa, 25)),
        "AA_p75_deg": float(np.percentile(aa, 75)),
        "fraction_ATA_lt_30": float(np.mean(ata < 30.0)),
        "fraction_AA_lt_90": float(np.mean(aa < 90.0)),
        "fraction_full_angle_gate": float(np.mean((ata < 30.0) & (aa < 90.0))),
    }


def run_audit(profile: str, base_seed: int, seeds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for maneuver, command in MANEUVERS.items():
        for offset in range(seeds):
            env = HeterogeneousMAVUAVAirCombatEnv(profile=profile)
            env.reset(seed=base_seed + offset)
            for _ in range(int(env.config["simulation"]["max_decision_steps"])):
                actions = np.repeat(command[None, :], len(RED_IDS), axis=0)
                _, _, terminated, truncated, _ = env.step(actions)
                for red_id in RED_IDS:
                    red = env.entities[red_id]
                    if not red.state.alive:
                        continue
                    for blue_id in BLUE_IDS:
                        blue = env.entities[blue_id]
                        if not blue.state.alive:
                            continue
                        for direction, attacker, target in (
                            ("red_to_blue", red, blue), ("blue_to_red", blue, red),
                        ):
                            geometry = compute_pairwise_geometry(attacker.state, target.state)
                            sample = {
                                "ata_deg": float(np.rad2deg(geometry.ata)),
                                "aa_deg": float(np.rad2deg(geometry.aa)),
                            }
                            if geometry.distance <= 5000.0:
                                grouped[(maneuver, direction, "within_5km")].append(sample)
                                grouped[("all", direction, "within_5km")].append(sample)
                            if 1000.0 <= geometry.distance <= 3000.0:
                                grouped[(maneuver, direction, "engagement_1_3km")].append(sample)
                                grouped[("all", direction, "engagement_1_3km")].append(sample)
                if terminated or truncated:
                    break
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        maneuver, direction, distance_band = key
        rows.append({
            "maneuver": maneuver, "direction": direction, "distance_band": distance_band,
            **_summary(grouped[key]),
        })
    overall = {
        f"{direction}_{band}": _summary(grouped.get(("all", direction, band), []))
        for direction in ("red_to_blue", "blue_to_red")
        for band in ("within_5km", "engagement_1_3km")
    }
    return rows, {
        "environment_version": ENVIRONMENT_VERSION,
        "profile": profile,
        "base_seed": base_seed,
        "seeds_per_maneuver": seeds,
        "maneuvers": list(MANEUVERS),
        "overall": overall,
        "blue_has_legal_attack_geometry": bool(
            overall["blue_to_red_engagement_1_3km"]["samples"]
            and float(overall["blue_to_red_engagement_1_3km"]["fraction_full_angle_gate"] or 0.0) > 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blue_guidance_geometry_v35"))
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("seeds must be positive")
    rows, summary = run_audit(args.profile, args.base_seed, args.seeds)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "geometry_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_path = output / "summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
