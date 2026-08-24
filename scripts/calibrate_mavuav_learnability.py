"""Formal small-scale learnability calibration without environment tuning."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from uav_combat.calibration import (
    TrainingDiagnostics, draw_return_exceeds_red_win, evaluate_policy, evaluation_summary, fixed_evaluation_seeds,
    geometry_summary, load_calibration_checkpoint, reward_diagnostic_rows,
    run_rule_baselines, save_calibration_checkpoint, target_concentration_summary, train_to_sampled_steps,
    write_csv, write_json,
)
from uav_combat.happo import HAPPOTrainer
from uav_combat.mappo import MAPPOTrainer

ENV_CONFIG = "configs/heterogeneous_mavuav_3v2.yaml"
TRAINER_CONFIGS = {"happo": "configs/happo_mavuav_3v2.yaml", "mappo": "configs/mappo_mavuav_3v2.yaml"}
TRAINERS = {"happo": HAPPOTrainer, "mappo": MAPPOTrainer}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("happo", "mappo"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/calibration")
    parser.add_argument("--benchmark-steps", type=int, default=2_000)
    parser.add_argument("--baseline-episodes", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, str | None]:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", f"requested {requested}, but torch.cuda.is_available() is false"
    return requested, None


def trainer_config(algorithm: str, seed: int, device: str, num_envs: int) -> dict[str, Any]:
    with Path(TRAINER_CONFIGS[algorithm]).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["training"]["seed"] = int(seed)
    config["training"]["device"] = device
    config["training"]["num_envs"] = int(num_envs)
    return config


def make_trainer(algorithm: str, seed: int, device: str, num_envs: int):
    return TRAINERS[algorithm](ENV_CONFIG, trainer_config(algorithm, seed, device, num_envs))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run_benchmark(algorithm: str, seed: int, device: str, num_envs: int, sampled_steps: int) -> dict[str, Any]:
    if sampled_steps <= 0:
        return {"sampled_steps": 0, "elapsed_seconds": 0.0, "environment_steps_per_second": 0.0, "estimated_50k_seconds": 0.0}
    if sampled_steps % num_envs:
        raise ValueError("benchmark steps must be divisible by num_envs")
    trainer = make_trainer(algorithm, seed, device, num_envs)
    diagnostics = TrainingDiagnostics(max_observation_samples=10_000)
    started = time.perf_counter()
    train_to_sampled_steps(trainer, sampled_steps, diagnostics)
    elapsed = time.perf_counter() - started
    trainer.close()
    throughput = sampled_steps / elapsed
    return {
        "algorithm": algorithm, "device": device, "num_envs": num_envs,
        "sampled_steps": sampled_steps, "elapsed_seconds": elapsed,
        "environment_steps_per_second": throughput,
        "estimated_50k_seconds": 50_000 / throughput,
    }


def _numeric(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def build_summary(
    algorithm: str, seed: int, device: str, fallback: str | None, benchmark: dict[str, Any],
    evaluation_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]], reward_rows: list[dict[str, Any]], sampled_steps: int,
) -> dict[str, Any]:
    final_eval = [row for row in evaluation_rows if int(float(row["sampled_steps"])) == sampled_steps]
    first_step = min(int(float(row["sampled_steps"])) for row in evaluation_rows)
    first_eval = [row for row in evaluation_rows if int(float(row["sampled_steps"])) == first_step]
    trends = {}
    for mode in ("nearest", "mav_priority"):
        first = next(row for row in first_eval if row["blue_mode"] == mode)
        final = next(row for row in final_eval if row["blue_mode"] == mode)
        trends[mode] = {
            "red_win_rate_change": _numeric(final, "red_win_rate") - _numeric(first, "red_win_rate"),
            "MAV_survival_rate_change": _numeric(final, "MAV_survival_rate") - _numeric(first, "MAV_survival_rate"),
            "red_attack_kills_change": _numeric(final, "mean_red_attack_kills") - _numeric(first, "mean_red_attack_kills"),
            "episode_return_change": _numeric(final, "mean_episode_return") - _numeric(first, "mean_episode_return"),
        }
    final_actions = [row for row in action_rows if int(row["sampled_steps"]) == sampled_steps]
    final_groups = [row for row in observation_rows if row["row_type"] == "group" and int(row["sampled_steps"]) == sampled_steps]
    final_geometry = [row for row in geometry_rows if int(float(row["sampled_steps"])) == sampled_steps]
    final_targets = [row for row in target_rows if int(float(row["sampled_steps"])) == sampled_steps]
    hacking = {}
    for mode in ("nearest", "mav_priority"):
        selected = [row for row in reward_rows if row["blue_mode"] == mode and int(float(row["sampled_steps"])) == sampled_steps]
        hacking[mode] = {
            "draw_mean_return_exceeds_red_win": draw_return_exceeds_red_win(selected),
            "return_red_attack_kills_correlation": _numeric(selected[0], "return_red_attack_kills_correlation") if selected else 0.0,
        }
    return {
        "algorithm": algorithm, "seed": seed, "device": device, "device_fallback_reason": fallback,
        "sampled_steps": sampled_steps, "sample_step_definition": "one transition from one environment; one vector step contributes num_envs sampled steps",
        "benchmark": benchmark, "final_evaluations": final_eval, "learning_trends_from_first_checkpoint": trends,
        "maximum_action_saturation_rate": max((_numeric(row, "saturation_rate") for row in final_actions), default=0.0),
        "observation_group_fraction_abs_le_0p05": {row["feature"]: _numeric(row, "fraction_abs_le_0p05") for row in final_groups},
        "geometry": final_geometry, "target_concentration": final_targets, "reward_hacking": hacking,
    }


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.sample_steps <= 0 or args.eval_interval <= 0:
        raise ValueError("num-envs, sample-steps and eval-interval must be positive")
    if args.sample_steps % args.num_envs or args.eval_interval % args.num_envs:
        raise ValueError("sample-steps and eval-interval must be divisible by num-envs")
    device, fallback = resolve_device(args.device)
    output_root = Path(args.output_dir)
    run_dir = output_root / f"{args.algorithm}_seed{args.seed}"
    checkpoints = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    if not args.skip_baselines:
        baseline_rows = run_rule_baselines(output_root, args.baseline_episodes, ENV_CONFIG)
        if args.baselines_only:
            print({"rule_baselines": baseline_rows}, flush=True)
            return
    benchmark = run_benchmark(args.algorithm, args.seed, device, args.num_envs, args.benchmark_steps)
    write_json(run_dir / "benchmark.json", benchmark)
    print({"benchmark": benchmark, "device_fallback_reason": fallback}, flush=True)
    if args.benchmark_only:
        return

    trainer = make_trainer(args.algorithm, args.seed, device, args.num_envs)
    diagnostics = TrainingDiagnostics()
    sampled_steps = 0
    if args.resume:
        sampled_steps, diagnostics = load_calibration_checkpoint(args.resume, trainer, args.algorithm)

    training_rows = _read_csv(run_dir / "training.csv") if sampled_steps else []
    evaluation_rows = _read_csv(run_dir / "evaluations.csv") if sampled_steps else []
    action_rows = _read_csv(run_dir / "action_stats.csv") if sampled_steps else []
    observation_rows = _read_csv(run_dir / "observation_stats.csv") if sampled_steps else []
    geometry_rows = _read_csv(run_dir / "geometry_stats.csv") if sampled_steps else []
    target_rows = _read_csv(run_dir / "target_concentration.csv") if sampled_steps else []
    reward_rows = _read_csv(run_dir / "reward_diagnostics.csv") if sampled_steps else []

    targets = list(range(args.eval_interval, args.sample_steps + 1, args.eval_interval))
    if not targets or targets[-1] != args.sample_steps:
        targets.append(args.sample_steps)
    for target in targets:
        if target <= sampled_steps:
            continue
        _, updates = train_to_sampled_steps(trainer, target, diagnostics)
        training_rows.extend({"algorithm": args.algorithm, "seed": args.seed, **row} for row in updates)
        sampled_steps = target
        checkpoint = checkpoints / f"checkpoint_{sampled_steps:06d}.pt"
        save_calibration_checkpoint(checkpoint, trainer, args.algorithm, sampled_steps, diagnostics)
        count = args.final_eval_episodes if sampled_steps == args.sample_steps else args.eval_episodes
        seeds = fixed_evaluation_seeds(count)
        policy = trainer.actor if args.algorithm == "mappo" else trainer.actors
        for blue_mode in ("nearest", "mav_priority"):
            records = evaluate_policy(policy, args.algorithm, ENV_CONFIG, blue_mode, seeds, sampled_steps, device)
            evaluation_rows.append(evaluation_summary(records, args.algorithm, args.seed, sampled_steps, blue_mode))
            geometry_rows.append(geometry_summary(records, args.algorithm, args.seed, sampled_steps, blue_mode))
            target_rows.append(target_concentration_summary(records, args.algorithm, args.seed, sampled_steps, blue_mode))
            reward_rows.extend(reward_diagnostic_rows(records, args.algorithm, args.seed, sampled_steps, blue_mode))
        action_rows.extend(diagnostics.action_rows(args.algorithm, args.seed, sampled_steps))
        observation_rows.extend(diagnostics.observation_rows(args.algorithm, args.seed, sampled_steps))
        write_csv(run_dir / "training.csv", training_rows); write_csv(run_dir / "evaluations.csv", evaluation_rows)
        write_csv(run_dir / "action_stats.csv", action_rows); write_csv(run_dir / "observation_stats.csv", observation_rows)
        write_csv(run_dir / "geometry_stats.csv", geometry_rows); write_csv(run_dir / "target_concentration.csv", target_rows)
        write_csv(run_dir / "reward_diagnostics.csv", reward_rows)
        print({"algorithm": args.algorithm, "sampled_steps": sampled_steps, "checkpoint": str(checkpoint)}, flush=True)

    summary = build_summary(args.algorithm, args.seed, device, fallback, benchmark, evaluation_rows, action_rows, observation_rows, geometry_rows, target_rows, reward_rows, sampled_steps)
    write_json(run_dir / "summary.json", summary)
    trainer.close()
    print(summary, flush=True)


if __name__ == "__main__":
    main()
