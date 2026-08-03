"""Train or smoke-test HAPPO on the functional heterogeneous 4v3 v9 environment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from uav_combat.happo.evaluation_4v3 import evaluate_happo_fixed_blue_4v3
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer, compute_best_score_4v3


def load_train_config(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    t, e = cfg["training"], cfg["experiment"]
    if args.device is not None:
        e["device"] = args.device
    if args.output_dir is not None:
        e["output_dir"] = args.output_dir
    if args.total_env_steps is not None:
        t["total_env_steps"] = int(args.total_env_steps)
    if args.num_envs is not None:
        t["num_envs"] = int(args.num_envs)
    if args.env_workers is not None:
        t["num_env_workers"] = int(args.env_workers)
    if args.smoke:
        t["total_env_steps"] = min(int(t["total_env_steps"]), 8192)
        t["evaluation_interval_env_steps"] = int(t["total_env_steps"])
        t["checkpoint_interval_env_steps"] = int(t["total_env_steps"])
        cfg["evaluation"]["episodes"] = min(int(cfg["evaluation"].get("episodes", 8)), 8)
        t["quick_evaluation_episodes"] = min(int(t.get("quick_evaluation_episodes", 8)), 8)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--train-config", default="configs/happo_heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_train_config(args.train_config, args)
    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    trainer = HAPPO4v3Trainer(args.env_config, cfg)
    metrics_path = out / "training_metrics.csv"
    fields = [
        "env_steps", "vector_steps", "update_count", "actor_loss", "critic_loss",
        "entropy", "approx_kl", "advantage_mean", "advantage_std", "explained_variance",
    ]
    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            while trainer.env_steps < int(cfg["training"]["total_env_steps"]):
                trainer.collect_rollout()
                metrics = trainer.update()
                writer.writerow({k: metrics.get(k, 0.0) for k in fields})
                fh.flush()

        eval_summary = evaluate_happo_fixed_blue_4v3(
            trainer.actors,
            args.env_config,
            episodes=int(cfg["evaluation"]["episodes"]),
            num_envs=min(int(cfg["training"]["num_envs"]), 4),
            num_env_workers=0,
            device=trainer.device,
            seed=int(cfg["experiment"]["seed"]) + 50000,
        )
        score, score_fields = compute_best_score_4v3(eval_summary)
        trainer.best_score = score
        trainer.best_score_fields = score_fields
        trainer.evaluation_history.append({"env_steps": trainer.env_steps, "summary": eval_summary, "score": score})
        trainer.save_checkpoint(out / "checkpoint_final.pt")
        trainer.save_checkpoint(out / "checkpoint_best.pt", is_best=True)
        trainer.write_summary(out)
        (out / "final_evaluation.json").write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
