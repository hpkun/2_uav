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
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from algorithm.happo import HAPPOTrainer
from algorithm.happo.evaluation import evaluate_actors, evaluate_recurrent_actors, summarize_records
from env.mavuav import RED_IDS, load_environment_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "happo.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "env.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TRAINING_FIELDS = (
    "sampled_steps", "completed_episodes", "mean_episode_return", "red_win_rate", "blue_win_rate",
    "draw_rate", "MAV_survival_rate", "mean_UAV_survivors", "mean_red_attack_kills",
    "mean_blue_attack_kills", "mean_episode_length", *(f"actor_{i}_loss" for i in range(len(RED_IDS))),
    "critic_loss", "entropy", "method_variant", "p_nearest",
    "agp_raw_mean", "agp_raw_mean_abs", "agp_shaping_mean", "agp_shaping_mean_abs",
    "transitions_nearest", "transitions_mav_priority", "episodes_nearest", "episodes_mav_priority",
)
LOSS_FIELDS = (*(f"actor_{i}_loss" for i in range(len(RED_IDS))), "critic_loss", "entropy")


def _algorithm_name(
    actor_variant: str, method_variant: str = "baseline", critic_variant: str = "mlp",
) -> str:
    if critic_variant == "relational":
        if actor_variant != "vanilla" or method_variant != "baseline":
            raise ValueError("relational critic only supports vanilla baseline HAPPO")
        return "rc_happo"
    if critic_variant != "mlp":
        raise ValueError(f"unsupported critic_variant: {critic_variant!r}")
    if actor_variant == "vanilla" and method_variant != "baseline":
        method_names = {
            "agp": "happo_agp",
            "curriculum": "happo_curriculum",
            "agp_curriculum": "happo_agp_curriculum",
        }
        try:
            return method_names[method_variant]
        except KeyError as error:
            raise ValueError(f"unsupported method_variant: {method_variant!r}") from error
    if method_variant != "baseline":
        raise ValueError("non-vanilla actors only support method_variant='baseline'")
    names = {
        "vanilla": "happo",
        "hrta": "happo_hrta",
        "structured_uniform": "happo_structured_uniform",
        "recurrent": "happo_recurrent",
    }
    try:
        return names[actor_variant]
    except KeyError as error:
        raise ValueError(f"unsupported actor_variant: {actor_variant!r}") from error


class MilestoneObserver:
    """Observe thresholds without influencing rollout boundaries."""

    def __init__(self, interval: int, current_step: int = 0) -> None:
        self.interval = int(interval)
        self.next_milestone = (
            ((int(current_step) // self.interval) + 1) * self.interval if self.interval > 0 else None
        )

    def consume(self, sampled_steps: int) -> list[int]:
        crossed: list[int] = []
        while self.next_milestone is not None and self.next_milestone <= int(sampled_steps):
            crossed.append(self.next_milestone)
            self.next_milestone += self.interval
        return crossed


class ProgressWindow:
    """Accumulate training-only metrics between human-readable log records."""

    def __init__(self) -> None:
        self.episodes: list[Mapping[str, Any]] = []
        self.updates: list[Mapping[str, Any]] = []
        self.sampled_steps = 0
        self.training_seconds = 0.0

    def add(
        self,
        episodes: list[Mapping[str, Any]],
        update: Mapping[str, Any],
        sampled_steps: int,
        training_seconds: float,
    ) -> None:
        self.episodes.extend(episodes)
        self.updates.append(update)
        self.sampled_steps += int(sampled_steps)
        self.training_seconds += float(training_seconds)

    def clear(self) -> None:
        self.episodes.clear()
        self.updates.clear()
        self.sampled_steps = 0
        self.training_seconds = 0.0


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
    parser.add_argument("--log-interval", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def planned_rollout_horizon(
    current_steps: int,
    target_steps: int,
    configured_horizon: int,
    num_envs: int,
) -> int:
    """Use a full rollout except for the one update that reaches total steps."""
    remaining = int(target_steps) - int(current_steps)
    if remaining <= 0:
        raise ValueError("target steps must exceed current steps")
    if remaining % int(num_envs):
        raise ValueError("remaining sampled steps must be divisible by num_envs")
    return min(int(configured_horizon), remaining // int(num_envs))


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
    algorithm = _algorithm_name(args.actor_variant, args.method_variant, args.critic_variant)
    name = args.output_name or (
        f"{algorithm}_{args.profile}_seed{args.seed}_{_step_label(args.steps)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = OUTPUT_ROOT / name
    if run_dir.exists():
        raise FileExistsError(f"run folder already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _append_csv(path: Path, row: Mapping[str, Any], fields: tuple[str, ...] | None = None) -> None:
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as stream:
            fieldnames = next(csv.reader(stream))
    else:
        fieldnames = list(fields or row.keys())
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


def _evaluation_row(
    trainer: HAPPOTrainer,
    episodes: int,
    mode: str,
    profile: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    evaluator = evaluate_recurrent_actors if trainer.is_recurrent else evaluate_actors
    records = evaluator(
        trainer.actors, trainer.environment_config, episodes, mode, profile, seed=1000, device=device,
    )
    return {
        "sampled_steps": trainer.env_steps,
        "algorithm": _algorithm_name(
            trainer.config["actor_variant"], trainer.config["method_variant"],
            trainer.config["critic_variant"],
        ),
        "method_variant": trainer.config["method_variant"],
        "seed": seed,
        "blue_mode": mode, "training_profile": trainer.config["environment_profile"],
        "evaluation_profile": profile, "episodes": episodes, **summarize_records(records),
    }


def _last_completed_episodes(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return int(float(rows[-1]["completed_episodes"])) if rows else 0


def _duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _interval_text(value: int) -> str:
    return "disabled" if int(value) == 0 else f"{int(value):,}"


def _progress_lines(
    current_steps: int,
    target_steps: int,
    completed_episodes: int,
    training_elapsed: float,
    window: ProgressWindow,
) -> str:
    speed = window.sampled_steps / window.training_seconds if window.training_seconds > 0 else 0.0
    eta = (target_steps - current_steps) / speed if speed > 0 else None
    losses = {
        field: float(np.mean([update[field] for update in window.updates]))
        for field in LOSS_FIELDS
    }
    lines = [
        f"[TRAIN] {current_steps:,} / {target_steps:,} ({100.0 * current_steps / target_steps:.1f}%)",
        f"        elapsed {_duration(training_elapsed)} | speed {speed:.1f} steps/s | ETA {_duration(eta)}",
        f"        episodes +{len(window.episodes)} / total {completed_episodes}",
    ]
    if window.episodes:
        recent = _episode_metrics(window.episodes)
        lines.extend([
            f"        return {recent['mean_episode_return']:.2f} | "
            f"W/B/D {recent['red_win_rate']:.1%} / {recent['blue_win_rate']:.1%} / {recent['draw_rate']:.1%}",
            f"        MAV survival {recent['MAV_survival_rate']:.1%} | "
            f"UAV survivors {recent['mean_UAV_survivors']:.2f}",
            f"        Red kills {recent['mean_red_attack_kills']:.2f} | "
            f"Blue kills {recent['mean_blue_attack_kills']:.2f} | "
            f"ep length {recent['mean_episode_length']:.1f}",
        ])
    else:
        lines.extend([
            "        return n/a | W/B/D n/a",
            "        MAV survival n/a | UAV survivors n/a",
            "        Red kills n/a | Blue kills n/a | ep length n/a",
        ])
    actor_loss_text = ", ".join(f"{losses[f'actor_{i}_loss']:.3f}" for i in range(len(RED_IDS)))
    lines.extend([
        f"        actor loss [{actor_loss_text}]",
        f"        critic loss {losses['critic_loss']:.3f} | entropy {losses['entropy']:.3f}",
    ])
    return "\n".join(lines)


def _evaluation_lines(prefix: str, row: Mapping[str, Any]) -> str:
    return "\n".join([
        f"[{prefix}] step {int(row['sampled_steps']):,} | {row['blue_mode']}",
        f"       win {row['red_win_rate']:.1%} | blue {row['blue_win_rate']:.1%} | "
        f"draw {row['draw_rate']:.1%}",
        f"       return {row['mean_episode_return']:.2f} | "
        f"Red kills {row['mean_red_attack_kills']:.2f} | "
        f"MAV survival {row['MAV_survival_rate']:.1%}",
    ])


def _initial_resolved(
    args: argparse.Namespace,
    env_config: Mapping[str, Any],
    trainer: HAPPOTrainer,
    device: str,
    fallback: str | None,
) -> dict[str, Any]:
    return {
        "environment": dict(env_config), "happo": dict(trainer.config), "profile": args.profile,
        "algorithm": _algorithm_name(
            trainer.config["actor_variant"], trainer.config["method_variant"],
            trainer.config["critic_variant"],
        ),
        "method_variant": trainer.config["method_variant"],
        "actor_variant": trainer.config["actor_variant"],
        "critic_variant": trainer.config["critic_variant"],
        "actor_architecture": trainer.actor_architecture,
        "actor_parameter_count_per_agent": trainer.actor_parameter_counts["per_agent"],
        "actor_parameter_count_total": trainer.actor_parameter_counts["total"],
        "critic_architecture": trainer.critic_architecture,
        "critic_parameter_count": trainer.critic_parameter_count,
        "seed": args.seed, "requested_device": args.device, "resolved_device": device,
        "device_fallback_reason": fallback, "num_envs": args.num_envs, "total_steps": args.steps,
        "checkpoint_interval": args.checkpoint_interval, "evaluation_interval": args.eval_interval,
        "log_interval": args.log_interval, "evaluation_episodes": args.eval_episodes,
        "final_evaluation_episodes": args.final_eval_episodes, "resume_history": [],
    }


def _write_resolved_config(
    path: Path,
    args: argparse.Namespace,
    env_config: Mapping[str, Any],
    trainer: HAPPOTrainer,
    device: str,
    fallback: str | None,
    resumed_steps: int | None,
) -> None:
    if resumed_steps is None:
        resolved = _initial_resolved(args, env_config, trainer, device, fallback)
    else:
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                resolved = yaml.safe_load(stream) or {}
        else:
            resolved = _initial_resolved(args, env_config, trainer, device, fallback)
        history = resolved.setdefault("resume_history", [])
        history.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": args.resume.expanduser().resolve().name,
            "resumed_from_steps": int(resumed_steps),
            "target_steps": int(args.steps),
            "requested_device": args.device,
            "resolved_device": device,
            "checkpoint_interval": int(args.checkpoint_interval),
            "evaluation_interval": int(args.eval_interval),
            "log_interval": int(args.log_interval),
        })
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False, allow_unicode=True)


def main(
    actor_variant: str = "vanilla", method_variant: str = "baseline", critic_variant: str = "mlp",
) -> None:
    algorithm = _algorithm_name(actor_variant, method_variant, critic_variant)
    args = parse_args()
    args.actor_variant = actor_variant
    args.method_variant = method_variant
    args.critic_variant = critic_variant
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    if args.steps % args.num_envs:
        raise ValueError("steps must be divisible by num-envs")
    if min(args.checkpoint_interval, args.eval_interval, args.log_interval) < 0:
        raise ValueError("checkpoint, evaluation and log intervals cannot be negative")
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
        "actor_variant": actor_variant, "critic_variant": critic_variant, "method_variant": method_variant,
        "curriculum_total_steps": args.steps if method_variant in ("curriculum", "agp_curriculum") else None,
    })
    trainer = HAPPOTrainer(env_config, config)
    try:
        resumed_steps: int | None = None
        if args.resume:
            resumed_steps = trainer.load_checkpoint(args.resume.expanduser().resolve())
        if trainer.env_steps >= args.steps:
            raise ValueError(f"target steps {args.steps} must exceed current checkpoint steps {trainer.env_steps}")
        _write_resolved_config(
            run_dir / "resolved_config.yaml", args, env_config, trainer, device, fallback, resumed_steps,
        )

        separator = "=" * 60
        start_lines = [
            separator, f"{'RC-HAPPO' if algorithm == 'rc_happo' else algorithm.upper()} TRAINING",
            f"Run: {run_dir.name}", f"Profile: {args.profile}",
            f"Seed: {args.seed}", f"Device: {device}", f"Envs: {args.num_envs}",
            f"Method: {method_variant}",
            f"Critic: {critic_variant}",
            f"Rollout: {trainer.config['rollout_steps']}", f"Target steps: {args.steps:,}",
            f"Checkpoint interval: {_interval_text(args.checkpoint_interval)}",
            f"Evaluation interval: {_interval_text(args.eval_interval)}",
            f"Log interval: {_interval_text(args.log_interval)}",
        ]
        if resumed_steps is not None:
            start_lines.extend([
                f"Resume checkpoint: {args.resume.expanduser().resolve().name}",
                f"Resume steps: {resumed_steps:,}", f"Target steps: {args.steps:,}",
            ])
        if fallback:
            start_lines.append(f"Device fallback: {fallback}")
        start_lines.append(separator)
        log("\n".join(start_lines))

        checkpoint_observer = MilestoneObserver(args.checkpoint_interval, trainer.env_steps)
        evaluation_observer = MilestoneObserver(args.eval_interval, trainer.env_steps)
        log_observer = MilestoneObserver(args.log_interval, trainer.env_steps)
        completed = _last_completed_episodes(run_dir / "training.csv")
        evaluation_fields: tuple[str, ...] | None = None
        configured_horizon = int(trainer.config["rollout_steps"])
        num_envs = int(trainer.config["num_envs"])
        window = ProgressWindow()
        training_elapsed = 0.0

        while trainer.env_steps < args.steps:
            horizon = planned_rollout_horizon(
                trainer.env_steps, args.steps, configured_horizon, num_envs,
            )
            trainer.buffer = trainer.make_buffer(horizon)
            before_steps = trainer.env_steps
            update_started = time.perf_counter()
            episodes = trainer.collect_rollout()
            metrics = trainer.update()
            update_elapsed = time.perf_counter() - update_started
            sampled_this_update = trainer.env_steps - before_steps
            training_elapsed += update_elapsed
            completed += len(episodes)
            window.add(episodes, metrics, sampled_this_update, update_elapsed)
            row = {
                "sampled_steps": trainer.env_steps, "completed_episodes": completed,
                **_episode_metrics(episodes),
                **{f"actor_{i}_loss": metrics[f"actor_{i}_loss"] for i in range(len(RED_IDS))},
                "critic_loss": metrics["critic_loss"],
                "entropy": metrics["entropy"],
                **{field: metrics[field] for field in TRAINING_FIELDS if field in metrics},
            }
            _append_csv(run_dir / "training.csv", row, TRAINING_FIELDS)

            crossed_logs = log_observer.consume(trainer.env_steps)
            if crossed_logs:
                log(_progress_lines(trainer.env_steps, args.steps, completed, training_elapsed, window))
                window.clear()

            crossed_checkpoints = checkpoint_observer.consume(trainer.env_steps)
            if crossed_checkpoints:
                checkpoint = run_dir / f"checkpoint_{trainer.env_steps}.pt"
                trainer.save_checkpoint(checkpoint)
                milestone_text = (
                    f"{crossed_checkpoints[0]:,}" if len(crossed_checkpoints) == 1
                    else f"{crossed_checkpoints[0]:,}..{crossed_checkpoints[-1]:,}"
                )
                log(
                    f"[CKPT] milestone {milestone_text} crossed at step {trainer.env_steps:,} | "
                    f"{checkpoint.name}"
                )

            crossed_evaluations = evaluation_observer.consume(trainer.env_steps)
            if crossed_evaluations and trainer.env_steps < args.steps:
                for mode in ("nearest", "mav_priority"):
                    eval_row = _evaluation_row(
                        trainer, args.eval_episodes, mode, args.profile, args.seed, device,
                    )
                    evaluation_fields = evaluation_fields or tuple(eval_row.keys())
                    _append_csv(run_dir / "evaluations.csv", eval_row, evaluation_fields)
                    log(_evaluation_lines("EVAL", eval_row))

        trainer.buffer = trainer.make_buffer(configured_horizon)
        if args.log_interval and window.updates:
            log(_progress_lines(trainer.env_steps, args.steps, completed, training_elapsed, window))
            window.clear()
        log(f"[TRAIN] optimization finished at {trainer.env_steps:,} steps")
        trainer.save_checkpoint(run_dir / "checkpoint_final.pt")

        final_evaluation_started = time.perf_counter()
        final_rows = []
        for mode in ("nearest", "mav_priority"):
            eval_row = _evaluation_row(
                trainer, args.final_eval_episodes, mode, args.profile, args.seed, device,
            )
            evaluation_fields = evaluation_fields or tuple(eval_row.keys())
            _append_csv(run_dir / "evaluations.csv", eval_row, evaluation_fields)
            final_rows.append(eval_row)
            log(_evaluation_lines("FINAL EVAL", eval_row))
        final_evaluation_elapsed = time.perf_counter() - final_evaluation_started

        summary = {
            "algorithm": algorithm,
            "actor_variant": actor_variant, "critic_variant": critic_variant,
            "method_variant": method_variant,
            "agp_lambda": float(trainer.config["agp_lambda"]),
            "curriculum_schedule": [list(item) for item in trainer.curriculum_schedule],
            "curriculum_total_steps": trainer.curriculum_total_steps,
            "mode_transition_counts": trainer.mode_transition_counts,
            "mode_episode_counts": trainer.mode_episode_counts,
            "actor_architecture": trainer.actor_architecture,
            "actor_parameter_count_per_agent": trainer.actor_parameter_counts["per_agent"],
            "actor_parameter_count_total": trainer.actor_parameter_counts["total"],
            "critic_architecture": trainer.critic_architecture,
            "critic_parameter_count": trainer.critic_parameter_count,
            "status": "complete", "sampled_steps": trainer.env_steps,
            "training_profile": args.profile, "seed": args.seed, "device": device,
            "num_envs": args.num_envs, "completed_episodes": completed,
            "training_elapsed_seconds": training_elapsed,
            "final_evaluation_elapsed_seconds": final_evaluation_elapsed,
            "final_evaluations": final_rows, "checkpoint_final": "checkpoint_final.pt",
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
        log("\n".join([
            separator, "TRAINING COMPLETE", f"Steps: {trainer.env_steps:,}",
            f"Elapsed training time: {_duration(training_elapsed)}",
            f"Final evaluation time: {_duration(final_evaluation_elapsed)}",
            f"Completed episodes: {completed}", "Final checkpoint: checkpoint_final.pt",
            f"Run folder: {run_dir}", separator,
        ]))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
