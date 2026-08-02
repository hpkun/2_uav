"""Train homogeneous 3v3 HAPPO red actors against fixed-rule blue."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_combat.happo.evaluation_3v3 import evaluate_happo_fixed_blue_3v3
from uav_combat.happo.metrics import finite_numeric_dict
from uav_combat.happo.trainer_3v3 import (
    CHECKPOINT_FAMILY_HAPPO_3V3,
    CHECKPOINT_VERSION_HAPPO_3V3,
    HAPPO3v3Trainer,
    compute_best_score,
    compute_best_score_fields,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3_learnable_v4.yaml")
    p.add_argument("--train-config", default="configs/happo_3v3_fixed_blue.yaml")
    p.add_argument("--total-env-steps", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--env-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_config(args) -> dict:
    with open(args.train_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["training"].get("training_mode") != "fixed_rule_blue_3v3_happo":
        raise ValueError("requires fixed_rule_blue_3v3_happo")
    if args.smoke:
        cfg["training"].update(
            total_env_steps=16384,
            num_envs=4,
            num_env_workers=1,
            rollout_steps=64,
            ppo_epochs=2,
            minibatch_size=256,
            quick_evaluation_episodes=8,
            evaluation_interval_env_steps=8192,
            checkpoint_interval_env_steps=8192,
        )
        cfg["evaluation"]["episodes"] = 8
        cfg["experiment"]["output_dir"] = "outputs/happo_3v3_fixed_blue_smoke"
    for value, section, key in (
        (args.total_env_steps, "training", "total_env_steps"),
        (args.num_envs, "training", "num_envs"),
        (args.env_workers, "training", "num_env_workers"),
        (args.seed, "experiment", "seed"),
        (args.device, "experiment", "device"),
        (args.output_dir, "experiment", "output_dir"),
    ):
        if value is not None:
            cfg[section][key] = value
    return cfg


def next_strict_milestone(current: int, interval: int) -> int:
    if interval <= 0:
        raise ValueError("milestone interval must be positive")
    return ((int(current) // int(interval)) + 1) * int(interval)


def episode_stats(records: list[dict]) -> dict:
    if not records:
        return {"completed_episodes": 0}
    n = len(records)
    return {
        "completed_episodes": n,
        "red_complete_elimination_success_rate": sum(r["red_complete_elimination_success"] for r in records) / n,
        "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
        "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
        "mean_red_survivors": float(np.mean([r["red_survivors"] for r in records])),
        "mean_blue_survivors": float(np.mean([r["blue_survivors"] for r in records])),
        "mean_red_boundary_deaths": float(np.mean([r["red_boundary_deaths"] for r in records])),
        "mean_red_collision_deaths": float(np.mean([
            r["red_friendly_collision_deaths"] + r["red_cross_collision_deaths"] for r in records
        ])),
        "red_any_attack_kill_rate": sum(r.get("red_any_attack_kill", False) for r in records) / n,
        "blue_any_attack_kill_rate": sum(r.get("blue_any_attack_kill", False) for r in records) / n,
        "mean_red_kills_with_shared_observation": float(np.mean([
            r.get("red_kills_with_shared_observation", 0) for r in records
        ])),
        "mean_blue_kills_with_shared_observation": float(np.mean([
            r.get("blue_kills_with_shared_observation", 0) for r in records
        ])),
        "mean_red_support_coverage_ratio": float(np.mean([
            r.get("red_mean_support_coverage_ratio", 0.0) for r in records
        ])),
        "mean_blue_support_coverage_ratio": float(np.mean([
            r.get("blue_mean_support_coverage_ratio", 0.0) for r in records
        ])),
        "red_support_survival_rate": float(np.mean([
            bool(r.get("red_support_survived", False)) for r in records
        ])),
        "blue_support_survival_rate": float(np.mean([
            bool(r.get("blue_support_survived", False)) for r in records
        ])),
        "max_steps_rate": sum(r["termination_reason"] == "max_steps" for r in records) / n,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_episode_return": float(np.mean([r["episode_return"] for r in records])),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    output = Path(cfg["experiment"]["output_dir"])
    ckpt_dir = output / "checkpoints"
    eval_dir = output / "evaluations"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output / "console.log").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    seed = int(cfg["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = HAPPO3v3Trainer(args.env_config, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed_from={args.resume}", flush=True)
    else:
        trainer.save_checkpoint(ckpt_dir / "initial.pt")

    print(
        f"device={trainer.device} num_envs={trainer.num_envs} workers={trainer.num_env_workers} "
        f"rule_modes={trainer.rule_policy_mapping_modes}",
        flush=True,
    )

    total_steps = int(cfg["training"]["total_env_steps"])
    eval_interval = int(cfg["training"].get("evaluation_interval_env_steps", 100000))
    ckpt_interval = int(cfg["training"].get("checkpoint_interval_env_steps", 100000))
    quick_eps = int(cfg["training"].get("quick_evaluation_episodes", cfg["evaluation"]["episodes"]))
    next_eval = next_strict_milestone(trainer.env_steps, eval_interval)
    next_ckpt = next_strict_milestone(trainer.env_steps, ckpt_interval)
    rows: list[dict] = []
    completed_since_log: list[dict] = []
    start = time.perf_counter()

    if not args.resume:
        initial_eval = evaluate_happo_fixed_blue_3v3(
            trainer.actors, args.env_config, quick_eps, trainer.num_envs,
            trainer.num_env_workers, trainer.device, seed + 100000,
        )
        (eval_dir / "evaluation_initial.json").write_text(json.dumps(initial_eval, indent=2, default=str))
        trainer.best_score = compute_best_score(initial_eval)
        trainer.best_evaluation = initial_eval
        trainer.best_checkpoint_name = "initial.pt"
        trainer.save_checkpoint(ckpt_dir / "best.pt")
        (eval_dir / "evaluation_best.json").write_text(json.dumps(initial_eval, indent=2, default=str))

    try:
        while trainer.env_steps < total_steps:
            completed = trainer.collect_rollout(total_steps - trainer.env_steps)
            completed_since_log.extend(completed)
            metrics = trainer.update()
            elapsed = time.perf_counter() - start
            row = {
                "env_steps": trainer.env_steps,
                "vector_steps": trainer.vector_steps,
                "updates": trainer.update_count,
                "environment_steps_per_second": trainer.env_steps / elapsed if elapsed > 0 else 0.0,
                **metrics,
                **episode_stats(completed_since_log),
            }
            if not finite_numeric_dict({k: v for k, v in row.items() if not isinstance(v, list)}):
                raise FloatingPointError(f"non-finite HAPPO training row: {row}")
            rows.append(row)
            completed_since_log.clear()
            print(json.dumps(row, default=str), flush=True)

            if trainer.env_steps >= next_eval:
                ev = evaluate_happo_fixed_blue_3v3(
                    trainer.actors, args.env_config, quick_eps, trainer.num_envs,
                    trainer.num_env_workers, trainer.device, seed + 100000,
                )
                (eval_dir / f"evaluation_step_{trainer.env_steps:06d}.json").write_text(
                    json.dumps(ev, indent=2, default=str)
                )
                score = compute_best_score(ev)
                trainer.evaluation_history.append({"env_steps": trainer.env_steps, "score": list(score), **ev})
                if trainer.best_score is None or score > trainer.best_score:
                    trainer.best_score = score
                    trainer.best_evaluation = ev
                    trainer.best_checkpoint_name = f"step_{trainer.env_steps:06d}.pt"
                    trainer.save_checkpoint(ckpt_dir / "best.pt")
                    (eval_dir / "evaluation_best.json").write_text(json.dumps(ev, indent=2, default=str))
                next_eval += eval_interval
            if trainer.env_steps >= next_ckpt:
                trainer.save_checkpoint(ckpt_dir / f"step_{trainer.env_steps:06d}.pt")
                next_ckpt += ckpt_interval
            trainer.save_checkpoint(ckpt_dir / "latest.pt")
    finally:
        trainer.close()

    final_eval = evaluate_happo_fixed_blue_3v3(
        trainer.actors, args.env_config, int(cfg["evaluation"]["episodes"]), trainer.num_envs,
        trainer.num_env_workers, trainer.device, seed + 100000,
    )
    (eval_dir / "evaluation_final.json").write_text(json.dumps(final_eval, indent=2, default=str))
    final_score = compute_best_score(final_eval)
    if trainer.best_score is None or final_score > trainer.best_score:
        trainer.best_score = final_score
        trainer.best_evaluation = final_eval
        trainer.best_checkpoint_name = "final.pt"
        trainer.save_checkpoint(ckpt_dir / "best.pt")
        (eval_dir / "evaluation_best.json").write_text(json.dumps(final_eval, indent=2, default=str))
    trainer.save_checkpoint(ckpt_dir / "final.pt")
    trainer.save_checkpoint(ckpt_dir / "latest.pt")
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_3V3,
        "checkpoint_version": CHECKPOINT_VERSION_HAPPO_3V3,
        "device": str(trainer.device),
        "actual_environment_steps": trainer.env_steps,
        "updates": trainer.update_count,
        "rule_policy_mapping_modes": trainer.rule_policy_mapping_modes,
        "environment_metadata": trainer.environment_metadata,
        "best_checkpoint": trainer.best_checkpoint_name,
        "best_score": list(trainer.best_score) if trainer.best_score else None,
        "best_score_fields": list(compute_best_score_fields(trainer.best_evaluation).keys()) if trainer.best_evaluation else None,
        "best_score_values": compute_best_score_fields(trainer.best_evaluation) if trainer.best_evaluation else None,
        "final_evaluation": final_eval,
        "final_metrics": rows[-1] if rows else {},
        "total_seconds": time.perf_counter() - start,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    main()
