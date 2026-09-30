"""Run the fixed RGAA/Wide/DBM qualitative replay protocol and aggregate it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.mavuav import load_environment_config
from tools.record_combat_episode import record_episode
from tools.replay_policy import load_replay_actors
from tools.report_combat_episode import report_directory


DETERMINISTIC_SEEDS = tuple(range(424242, 424247))
STOCHASTIC_ENV_SEEDS = tuple(range(1000, 1010))
STOCHASTIC_ACTION_SEEDS = tuple(range(2000, 2010))
REQUIRED_METHODS = ("rgaa", "rgaa_wide", "dbm_rgaa")
REQUIRED_TRAINING_SEEDS = (7, 9, 11)
REQUIRED_ENVIRONMENT_VERSION = "heterogeneous_mavuav_4v4_v3_9"
REQUIRED_SAMPLED_STEPS = 2_000_000


def _mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_mapping, required=True)
    parser.add_argument("--profile", choices=("learnability",), default="learnability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.run) != 9:
        parser.error("the fixed protocol requires exactly nine --run entries")
    return args


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for action_mode, method in sorted({(r["action_mode"], r["method_variant"]) for r in rows}):
        selected = [r for r in rows if r["action_mode"] == action_mode and r["method_variant"] == method]
        count = len(selected)
        output.append({
            "action_mode": action_mode, "method_variant": method, "episodes": count,
            "red_win_rate": sum(r["outcome"] == "red" for r in selected) / count,
            "blue_win_rate": sum(r["outcome"] == "blue" for r in selected) / count,
            "draw_rate": sum(r["outcome"] == "draw" for r in selected) / count,
            "mav_survival_rate": _mean(selected, "mav_survived"),
            "mean_uav_survivors": _mean(selected, "uav_survivors"),
            "mean_red_kills": _mean(selected, "red_attack_kills"),
            "mean_blue_kills": _mean(selected, "blue_attack_kills"),
            "mean_episode_length": _mean(selected, "episode_length"),
            "mean_mav_nearest_blue_distance_m": _mean(selected, "mav_nearest_blue_distance_m"),
            "mean_uav_nearest_blue_distance_m": _mean(selected, "uav_nearest_blue_distance_m"),
            "mean_mav_attack_events": _mean(selected, "mav_attack_events"),
            "mean_uav_attack_events": _mean(selected, "uav_attack_events"),
            "uav_boundary_death_count": int(sum(r["uav_boundary_death_count"] for r in selected)),
            "uav_altitude_lower_death_count": int(sum(r["uav_altitude_lower_death_count"] for r in selected)),
            "separation_warning_episode_rate": sum(r["separation_warning_count"] > 0 for r in selected) / count,
            "mean_mav_behind_uav_fraction": _mean(selected, "mav_behind_uav_fraction"),
            "mean_uav_attack_share": _mean(selected, "uav_attack_share"),
        })
    return output


def select_representatives(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predeclared selection: deterministic win nearest method win-length median."""
    selected: dict[str, Any] = {}
    for method in sorted({r["method_variant"] for r in rows}):
        candidates = [r for r in rows if r["method_variant"] == method
                      and r["action_mode"] == "deterministic" and r["outcome"] == "red"]
        if not candidates:
            selected[method] = None
            continue
        median = float(np.median([r["episode_length"] for r in candidates]))
        choice = min(candidates, key=lambda r: (
            abs(float(r["episode_length"]) - median),
            int(r["training_seed"]), int(r["environment_seed"]), str(r["run"]),
        ))
        selected[method] = {
            "rule": "deterministic red-win episode closest to method win-episode median length; ties by training seed, environment seed, run label",
            "win_episode_median_length": median,
            **dict(choice),
        }
    return selected


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_matrix(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate the fixed experiment matrix using checkpoint-derived metadata."""
    expected = {(method, seed) for method in REQUIRED_METHODS for seed in REQUIRED_TRAINING_SEEDS}
    observed: list[tuple[str, int]] = []
    reference_environment = None
    for record in records:
        method = str(record["method_variant"])
        seed = int(record["training_seed"])
        pair = (method, seed)
        if pair not in expected:
            raise RuntimeError(f"unexpected method/training-seed pair in trajectory audit: {pair}")
        if pair in observed:
            raise RuntimeError(f"duplicate method/training-seed pair in trajectory audit: {pair}")
        observed.append(pair)
        if int(record["sampled_steps"]) != REQUIRED_SAMPLED_STEPS:
            raise RuntimeError(f"{pair} is not an exact-2M checkpoint: {record['sampled_steps']}")
        if record.get("training_profile") != "learnability" or record.get("evaluation_profile") != "learnability":
            raise RuntimeError(f"{pair} must use learnability for training and evaluation profiles")
        if record.get("environment_version") != REQUIRED_ENVIRONMENT_VERSION:
            raise RuntimeError(f"{pair} must use environment {REQUIRED_ENVIRONMENT_VERSION}")
        environment = record.get("environment_config")
        if reference_environment is None:
            reference_environment = environment
        elif environment != reference_environment:
            raise RuntimeError("trajectory audit requires identical resolved environment configs")
    missing = expected - set(observed)
    if missing or len(observed) != len(expected):
        raise RuntimeError(f"incomplete 3 methods x 3 seeds trajectory-audit matrix; missing={sorted(missing)}")


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_metadata = {}
    prepared_runs = []
    matrix_records = []
    integrity_rows = []
    for label, raw_run in args.run:
        run_dir = raw_run.resolve(); checkpoint = run_dir / "checkpoint_final.pt"
        sha_before = checkpoint_sha256(checkpoint)
        adapter = load_replay_actors(checkpoint, args.device)
        env_config = load_environment_config(adapter.payload.get("environment_config"))
        method = adapter.method_variant
        training_seed = int(adapter.payload.get("trainer_config", adapter.payload.get("config", {}))["seed"])
        sampled_steps = int(adapter.payload.get("sampled_steps", 0))
        training_profile = adapter.payload.get("environment_profile")
        environment_version = adapter.payload.get("environment_version")
        matrix_records.append({
            "method_variant": method, "training_seed": training_seed,
            "sampled_steps": sampled_steps, "training_profile": training_profile,
            "evaluation_profile": args.profile, "environment_version": environment_version,
            "environment_config": env_config,
        })
        prepared_runs.append((label, checkpoint, adapter, env_config, method, training_seed, sampled_steps))
        integrity_rows.append({
            "run": label, "method": method, "training_seed": training_seed,
            "checkpoint_path": str(checkpoint), "sha256_before": sha_before,
        })
        run_metadata[label] = {
            "checkpoint": str(checkpoint), "method_variant": method,
            "training_seed": training_seed, "sampled_steps": sampled_steps,
            "training_profile": training_profile, "evaluation_profile": args.profile,
            "environment_version": environment_version, "sha256_before": sha_before,
        }
    validate_run_matrix(matrix_records)

    for label, checkpoint, adapter, env_config, method, training_seed, sampled_steps in prepared_runs:
        episode_specs = [
            ("deterministic", seed, None) for seed in DETERMINISTIC_SEEDS
        ] + [
            ("stochastic", env_seed, action_seed)
            for env_seed, action_seed in zip(STOCHASTIC_ENV_SEEDS, STOCHASTIC_ACTION_SEEDS)
        ]
        for action_mode, env_seed, action_seed in episode_specs:
            name = f"{action_mode}_env{env_seed}" + (f"_action{action_seed}" if action_seed is not None else "")
            episode_dir = output / "episodes" / label / name
            collect_dbm = method == "dbm_rgaa"
            metadata = record_episode(
                adapter, checkpoint, episode_dir, profile=args.profile, seed=env_seed,
                env_config=env_config, action_mode=action_mode, action_seed=action_seed,
                collect_dbm_diagnostics=collect_dbm,
            )
            report = report_directory(episode_dir)
            overview = report["overview"]; spatial = report["spatial_roles"]; coordination = report["coordination"]
            rows.append({
                "run": label, "method_variant": method, "training_seed": training_seed,
                "sampled_steps": sampled_steps, "action_mode": action_mode,
                "environment_seed": env_seed, "action_seed": action_seed,
                "episode_dir": str(episode_dir), "outcome": overview["outcome"],
                "episode_return": overview["episode_return"],
                "episode_length": overview["episode_length"],
                "red_attack_kills": overview["red_attack_kills"],
                "blue_attack_kills": overview["blue_attack_kills"],
                "mav_survived": int(bool(overview["mav_survived"])),
                "uav_survivors": overview["red_uav_survivors"],
                "mav_nearest_blue_distance_m": spatial["mav_minimum_enemy_distance_m"],
                "uav_nearest_blue_distance_m": spatial["mean_uav_minimum_enemy_distance_m"],
                "mav_attack_events": coordination["mav_attack_event_count"],
                "uav_attack_events": coordination["uav_attack_event_count"],
                "uav_boundary_death_count": coordination["uav_boundary_death_count"],
                "uav_altitude_lower_death_count": coordination["uav_altitude_lower_death_count"],
                "separation_warning_count": coordination["separation_warning_count"],
                "mav_behind_uav_fraction": spatial["mav_behind_mean_alive_uav_fraction"],
                "uav_attack_share": coordination["uav_attack_share"],
            })
            print(f"{label} {name}: {metadata['outcome']} length={metadata['episode_length']}", flush=True)
    for record in integrity_rows:
        sha_after = checkpoint_sha256(Path(record["checkpoint_path"]))
        record["sha256_after"] = sha_after
        record["unchanged"] = sha_after == record["sha256_before"]
        if not record["unchanged"]:
            raise RuntimeError(f"checkpoint changed during trajectory audit: {record['checkpoint_path']}")
    (output / "checkpoint_integrity.json").write_text(
        json.dumps(integrity_rows, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    aggregates = aggregate(rows)
    representatives = select_representatives(rows)
    _write_csv(output / "episode_index.csv", rows)
    _write_csv(output / "aggregate_by_method.csv", aggregates)
    (output / "representative_episodes.json").write_text(
        json.dumps(representatives, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    summary = {
        "protocol": {
            "profile": args.profile, "device": str(args.device),
            "deterministic_environment_seeds": list(DETERMINISTIC_SEEDS),
            "stochastic_environment_seeds": list(STOCHASTIC_ENV_SEEDS),
            "stochastic_action_seeds": list(STOCHASTIC_ACTION_SEEDS),
            "deterministic_episode_count": 45, "stochastic_episode_count": 90,
            "representative_selection_rule": "deterministic red-win nearest method win-episode median length",
        },
        "runs": run_metadata, "checkpoint_integrity": integrity_rows,
        "aggregate": aggregates, "representatives": representatives,
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_protocol(args)
    print(json.dumps({"output": str(args.output.resolve()), "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
