"""Primary flat-output training entry point for vanilla HAPPO."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from datetime import datetime
import json
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from algorithm.common import RolloutBuffer
from algorithm.happo import HAPPOTrainer
from algorithm.happo.evaluation import evaluate_actors, summarize_records
from env.mavuav import load_environment_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "happo.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "env.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TRAINING_FIELDS = (
    "sampled_steps", "completed_episodes", "mean_episode_return", "red_win_rate", "blue_win_rate",
    "draw_rate", "MAV_survival_rate", "mean_UAV_survivors", "mean_red_attack_kills",
    "mean_blue_attack_kills", "mean_episode_length", "actor_0_loss", "actor_1_loss",
    "actor_2_loss", "critic_loss", "entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total sampled single-environment transitions")
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--output-name")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _device(requested: str) -> tuple[str, str | None]:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", f"requested {requested}, but CUDA is unavailable"
    return requested, None


def _step_label(steps: int) -> str:
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def _new_run_dir(args: argparse.Namespace) -> Path:
    if args.resume:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.output_name:
            raise ValueError("--output-name cannot be combined with --resume")
        return checkpoint.parent
    name = args.output_name or (
        f"happo_{args.profile}_seed{args.seed}_{_step_label(args.steps)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = OUTPUT_ROOT / name
    if run_dir.exists():
        raise FileExistsError(f"run folder already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _append_csv(path: Path, row: Mapping[str, Any], fields: tuple[str, ...] | None = None) -> None:
    fieldnames = list(fields or row.keys())
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        stream.flush()


def _episode_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "mean_episode_return": 0.0, "red_win_rate": 0.0, "blue_win_rate": 0.0,
            "draw_rate": 0.0, "MAV_survival_rate": 0.0, "mean_UAV_survivors": 0.0,
            "mean_red_attack_kills": 0.0, "mean_blue_attack_kills": 0.0, "mean_episode_length": 0.0,
        }
    n = len(records)
    return {
        "mean_episode_return": float(np.mean([r["episode_return"] for r in records])),
        "red_win_rate": sum(r["outcome"] == "red" for r in records) / n,
        "blue_win_rate": sum(r["outcome"] == "blue" for r in records) / n,
        "draw_rate": sum(r["outcome"] == "draw" for r in records) / n,
        "MAV_survival_rate": float(np.mean([r["mav_survived"] for r in records])),
        "mean_UAV_survivors": float(np.mean([r["red_uav_survivors"] for r in records])),
        "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
        "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
    }


def _evaluation_row(trainer: HAPPOTrainer, episodes: int, mode: str, profile: str, seed: int, device: str) -> dict[str, Any]:
    records = evaluate_actors(
        trainer.actors, trainer.environment_config, episodes, mode, profile, seed=1000, device=device,
    )
    return {
        "sampled_steps": trainer.env_steps, "algorithm": "happo", "seed": seed,
        "blue_mode": mode, "training_profile": trainer.config["environment_profile"],
        "evaluation_profile": profile, "episodes": episodes, **summarize_records(records),
    }


def _train_to(trainer: HAPPOTrainer, target: int, training_path: Path, completed_episodes: int) -> int:
    configured_horizon = int(trainer.config["rollout_steps"])
    num_envs = int(trainer.config["num_envs"])
    while trainer.env_steps < target:
        vector_steps = min(configured_horizon, (target - trainer.env_steps) // num_envs)
        trainer.buffer = RolloutBuffer(vector_steps, num_envs)
        episodes = trainer.collect_rollout()
        metrics = trainer.update()
        completed_episodes += len(episodes)
        row = {
            "sampled_steps": trainer.env_steps,
            "completed_episodes": completed_episodes,
            **_episode_metrics(episodes),
            "actor_0_loss": metrics["actor_0_loss"], "actor_1_loss": metrics["actor_1_loss"],
            "actor_2_loss": metrics["actor_2_loss"], "critic_loss": metrics["critic_loss"],
            "entropy": metrics["entropy"],
        }
        _append_csv(training_path, row, TRAINING_FIELDS)
    trainer.buffer = RolloutBuffer(configured_horizon, num_envs)
    return completed_episodes


def _last_completed_episodes(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return int(float(rows[-1]["completed_episodes"])) if rows else 0


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.num_envs <= 0 or args.checkpoint_interval < 0 or args.eval_interval < 0:
        raise ValueError("steps and num-envs must be positive; intervals cannot be negative")
    for name, value in (("steps", args.steps), ("checkpoint-interval", args.checkpoint_interval), ("eval-interval", args.eval_interval)):
        if value and value % args.num_envs:
            raise ValueError(f"{name} must be divisible by num-envs")
    if args.eval_episodes <= 0 or args.final_eval_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")

    device, fallback = _device(args.device)
    run_dir = _new_run_dir(args)
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
            stream.flush()

    with args.config.expanduser().resolve().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    env_config = load_environment_config(args.env_config.expanduser().resolve())
    config["training"].update({
        "environment_profile": args.profile, "seed": args.seed, "device": device, "num_envs": args.num_envs,
    })
    trainer = HAPPOTrainer(env_config, config)
    try:
        resumed_from = None
        if args.resume:
            resumed_from = str(args.resume.expanduser().resolve())
            trainer.load_checkpoint(args.resume.expanduser().resolve())
        if trainer.env_steps >= args.steps:
            raise ValueError(f"target steps {args.steps} must exceed current checkpoint steps {trainer.env_steps}")
        resolved = {
            "environment": env_config, "happo": trainer.config, "profile": args.profile, "seed": args.seed,
            "requested_device": args.device, "resolved_device": device, "device_fallback_reason": fallback,
            "num_envs": args.num_envs, "total_steps": args.steps,
            "checkpoint_interval": args.checkpoint_interval, "evaluation_interval": args.eval_interval,
            "evaluation_episodes": args.eval_episodes, "final_evaluation_episodes": args.final_eval_episodes,
            "resume_from": resumed_from,
        }
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(resolved, stream, sort_keys=False, allow_unicode=True)
        log(f"Run folder: {run_dir}")
        log(f"Training HAPPO from {trainer.env_steps} to {args.steps} sampled environment steps on {device}")
        if fallback:
            log(f"Device fallback: {fallback}")

        boundaries = {args.steps}
        if args.checkpoint_interval:
            boundaries.update(range(((trainer.env_steps // args.checkpoint_interval) + 1) * args.checkpoint_interval, args.steps + 1, args.checkpoint_interval))
        if args.eval_interval:
            boundaries.update(range(((trainer.env_steps // args.eval_interval) + 1) * args.eval_interval, args.steps + 1, args.eval_interval))
        completed = _last_completed_episodes(run_dir / "training.csv")
        evaluation_fields: tuple[str, ...] | None = None
        for target in sorted(boundary for boundary in boundaries if boundary > trainer.env_steps):
            completed = _train_to(trainer, target, run_dir / "training.csv", completed)
            if args.checkpoint_interval and target % args.checkpoint_interval == 0:
                checkpoint = run_dir / f"checkpoint_{target}.pt"
                trainer.save_checkpoint(checkpoint)
                log(f"Saved checkpoint: {checkpoint.name}")
            if args.eval_interval and target % args.eval_interval == 0 and target != args.steps:
                for mode in ("nearest", "mav_priority"):
                    row = _evaluation_row(trainer, args.eval_episodes, mode, args.profile, args.seed, device)
                    evaluation_fields = evaluation_fields or tuple(row.keys())
                    _append_csv(run_dir / "evaluations.csv", row, evaluation_fields)
                log(f"Completed intermediate evaluation at {target} steps")

        trainer.save_checkpoint(run_dir / "checkpoint_final.pt")
        final_rows = []
        for mode in ("nearest", "mav_priority"):
            row = _evaluation_row(trainer, args.final_eval_episodes, mode, args.profile, args.seed, device)
            evaluation_fields = evaluation_fields or tuple(row.keys())
            _append_csv(run_dir / "evaluations.csv", row, evaluation_fields)
            final_rows.append(row)
        summary = {
            "algorithm": "happo", "status": "complete", "sampled_steps": trainer.env_steps,
            "training_profile": args.profile, "seed": args.seed, "device": device, "num_envs": args.num_envs,
            "completed_episodes": completed, "final_evaluations": final_rows,
            "checkpoint_final": "checkpoint_final.pt",
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
        log(f"Training complete at {trainer.env_steps} sampled environment steps")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
