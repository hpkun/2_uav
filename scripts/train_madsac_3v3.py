"""Train homogeneous 3v3 MADSAC red policy against fixed-rule blue."""
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

from uav_combat.environment_3v3 import OBS_DIM
from uav_combat.madsac.evaluation_3v3 import evaluate_madsac_fixed_blue_3v3
from uav_combat.madsac.metrics import MADSACMetricAccumulator
from uav_combat.madsac.networks import SharedSquashedGaussianActor
from uav_combat.madsac.trainer_3v3 import (
    CHECKPOINT_FAMILY_MADSAC_3V3,
    CHECKPOINT_VERSION_MADSAC_3V3,
    MADSAC3v3Trainer,
    compute_best_score,
    signature_mismatches,
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
    p.add_argument("--train-config", default="configs/madsac_3v3_paper.yaml")
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
    if args.smoke:
        cfg["training"].update(
            replay_capacity=50000,
            batch_size=256,
            learning_starts=1024,
            quick_evaluation_episodes=12,
            evaluation_interval_env_steps=10000,
            checkpoint_interval_env_steps=10000,
        )
        cfg["evaluation"]["episodes"] = 12
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


def finite_row(row: dict) -> bool:
    for value in row.values():
        if value is None:
            continue
        if isinstance(value, (float, int, np.floating, np.integer)) and not np.isfinite(float(value)):
            return False
    return True


def next_strict_milestone(current: int, interval: int) -> int:
    if interval <= 0:
        raise ValueError("milestone interval must be positive")
    return ((int(current) // int(interval)) + 1) * int(interval)


def build_actor_from_config(cfg: dict, device: torch.device) -> SharedSquashedGaussianActor:
    n = cfg["network"]
    actor_hidden = int(n.get("actor_hidden_dim", n.get("hidden_dim", 256)))
    log_std_bias = float(n.get("log_std_bias_init", n.get("log_std_init", -0.5)))
    return SharedSquashedGaussianActor(
        OBS_DIM, 3, 3, actor_hidden, log_std_bias,
        float(n.get("log_std_min", -5.0)), float(n.get("log_std_max", 2.0)),
    ).to(device)


def validate_final_checkpoint_lightweight(trainer: MADSAC3v3Trainer, final_checkpoint: Path) -> bool:
    probe = torch.zeros(2, 3, OBS_DIM, device=trainer.device)
    with torch.no_grad():
        before = trainer.actor.deterministic(probe).cpu()
    ckpt = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_MADSAC_3V3:
        raise RuntimeError(f"unexpected checkpoint family: {ckpt.get('checkpoint_family')}")
    if ckpt.get("checkpoint_version") != CHECKPOINT_VERSION_MADSAC_3V3:
        raise RuntimeError(f"unexpected checkpoint version: {ckpt.get('checkpoint_version')}")
    diffs = signature_mismatches(ckpt.get("training_signature"), trainer.training_signature())
    if diffs:
        raise RuntimeError("checkpoint signature mismatch:\n" + "\n".join(diffs))
    actor = build_actor_from_config(trainer.config, trainer.device)
    actor.load_state_dict(ckpt["online_actor"])
    actor.eval()
    with torch.no_grad():
        after = actor.deterministic(probe).cpu()
    return bool(torch.allclose(before, after, atol=1e-6, rtol=1e-6))


def episode_stats(records: list[dict]) -> dict:
    if not records:
        return {
            "completed_episodes": 0,
            "red_complete_elimination_success_rate": np.nan,
            "mean_red_attack_kills": np.nan,
            "mean_blue_attack_kills": np.nan,
            "mean_red_survivors": np.nan,
            "mean_blue_survivors": np.nan,
            "mean_red_boundary_deaths": np.nan,
            "mean_red_collision_deaths": np.nan,
            "max_steps_rate": np.nan,
            "mean_episode_length": np.nan,
            "mean_episode_return": np.nan,
        }
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
    trainer = MADSAC3v3Trainer(args.env_config, cfg)
    restored_action_match = None
    resumed = bool(args.resume)
    replay_restored = False
    initial_eval = None
    resume_start_eval = None
    if args.resume:
        trainer.load_checkpoint(args.resume)
        replay_restored = trainer.replay_restored
        print(f"resumed_from={args.resume} replay_restored={replay_restored}", flush=True)
        print("resume is not lossless because replay buffer is not restored", flush=True)
    else:
        trainer.save_checkpoint(ckpt_dir / "initial.pt")

    print(
        f"device={trainer.device} num_envs={trainer.num_envs} workers={trainer.num_env_workers} "
        f"rule_modes={trainer.rule_policy_mapping_modes}",
        flush=True,
    )

    rows: list[dict] = []
    completed_since_log: list[dict] = []
    start = time.perf_counter()
    total_steps = int(cfg["training"]["total_env_steps"])
    eval_interval = int(cfg["training"]["evaluation_interval_env_steps"])
    ckpt_interval = int(cfg["training"]["checkpoint_interval_env_steps"])
    log_interval = int(cfg["training"].get("log_interval_env_steps", 2048))
    quick_eps = int(cfg["training"]["quick_evaluation_episodes"])
    next_eval = next_strict_milestone(trainer.env_steps, eval_interval)
    next_ckpt = next_strict_milestone(trainer.env_steps, ckpt_interval)
    next_log = next_strict_milestone(trainer.env_steps, log_interval)
    metric_accumulator = MADSACMetricAccumulator()

    if args.resume:
        resume_start_eval = evaluate_madsac_fixed_blue_3v3(
            trainer.actor, args.env_config, quick_eps, trainer.num_envs,
            trainer.num_env_workers, trainer.device, seed + 100000,
        )
        (eval_dir / "evaluation_resume_start.json").write_text(
            json.dumps(resume_start_eval, indent=2, default=str)
        )
        score = compute_best_score(resume_start_eval)
        trainer.evaluation_history.append({"env_steps": trainer.env_steps, "score": list(score), **resume_start_eval})
        if trainer.best_score is None or score > trainer.best_score:
            trainer.best_score = score
            trainer.best_evaluation = resume_start_eval
            trainer.best_checkpoint_name = f"resume_start_{trainer.env_steps:06d}.pt"
            trainer.save_checkpoint(ckpt_dir / "best.pt")
            (eval_dir / "evaluation_best.json").write_text(json.dumps(resume_start_eval, indent=2, default=str))
    else:
        initial_eval = evaluate_madsac_fixed_blue_3v3(
            trainer.actor, args.env_config, quick_eps, trainer.num_envs,
            trainer.num_env_workers, trainer.device, seed + 100000,
        )
        (eval_dir / "evaluation_initial.json").write_text(json.dumps(initial_eval, indent=2, default=str))
        trainer.best_evaluation = initial_eval
        trainer.best_score = compute_best_score(initial_eval)
        trainer.best_checkpoint_name = "initial.pt"
        trainer.save_checkpoint(ckpt_dir / "best.pt")
        (eval_dir / "evaluation_best.json").write_text(json.dumps(initial_eval, indent=2, default=str))

    try:
        if trainer.env_steps >= total_steps:
            print(
                f"restored env_steps={trainer.env_steps} >= total_env_steps={total_steps}; "
                "skipping additional training and running final evaluation only",
                flush=True,
            )
        while trainer.env_steps < total_steps:
            completed_since_log.extend(trainer.step_environment())
            metrics = trainer.update()
            if metrics:
                metric_accumulator.add(metrics)
            if trainer.env_steps >= next_eval:
                ev = evaluate_madsac_fixed_blue_3v3(
                    trainer.actor, args.env_config, quick_eps, trainer.num_envs,
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
            if trainer.env_steps >= next_log or trainer.env_steps >= total_steps:
                elapsed = time.perf_counter() - start
                interval_metrics = metric_accumulator.summarize()
                row = {
                    "env_steps": trainer.env_steps,
                    "vector_steps": trainer.vector_steps,
                    "replay_size": trainer.replay.size,
                    "critic_updates": trainer.critic_update_count,
                    "actor_updates": trainer.actor_update_count,
                    "environment_steps_per_second": trainer.env_steps / elapsed if elapsed > 0 else 0.0,
                    **interval_metrics,
                    **episode_stats(completed_since_log),
                }
                if not finite_row({k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}):
                    raise FloatingPointError(f"non-finite training row: {row}")
                rows.append(row)
                print(json.dumps(row, default=str), flush=True)
                completed_since_log.clear()
                metric_accumulator.reset()
                trainer.save_checkpoint(ckpt_dir / "latest.pt")
                next_log += log_interval
    finally:
        trainer.close()

    final_eval = evaluate_madsac_fixed_blue_3v3(
        trainer.actor, args.env_config, int(cfg["evaluation"]["episodes"]), trainer.num_envs,
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

    restored_action_match = validate_final_checkpoint_lightweight(trainer, ckpt_dir / "final.pt")

    summary = {
        "checkpoint_family": CHECKPOINT_FAMILY_MADSAC_3V3,
        "checkpoint_version": CHECKPOINT_VERSION_MADSAC_3V3,
        "device": str(trainer.device),
        "actual_environment_steps": trainer.env_steps,
        "vector_steps": trainer.vector_steps,
        "critic_updates": trainer.critic_update_count,
        "actor_updates": trainer.actor_update_count,
        "target_updates": trainer.target_update_count,
        "rule_policy_mapping_modes": trainer.rule_policy_mapping_modes,
        "metrics_rows": len(rows),
        "metrics_finite": all(finite_row({k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}) for row in rows),
        "checkpoints": {name: (ckpt_dir / name).exists() for name in ("initial.pt", "latest.pt", "final.pt", "best.pt")},
        "replay_metadata": trainer.replay.metadata(include_full_replay=False),
        "replay_restored": replay_restored,
        "checkpoint_reload_deterministic_action_match": restored_action_match,
        "best_score": list(trainer.best_score) if trainer.best_score else None,
        "best_checkpoint": trainer.best_checkpoint_name,
        "resumed": resumed,
        "resumed_from": args.resume,
        "initial_evaluation": initial_eval,
        "resume_start_evaluation": resume_start_eval,
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
