"""Train or smoke-test HAPPO on the functional heterogeneous 4v3 v9 environment."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from uav_combat.happo.evaluation_4v3 import evaluate_happo_fixed_blue_4v3
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer, best_score_fields_4v3, compute_best_score_4v3, summarize_4v3_episodes


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


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _evaluate_and_maybe_save_best(
    trainer: HAPPO4v3Trainer,
    env_config: str,
    cfg: dict[str, Any],
    out: Path,
    *,
    label: str,
    env_steps: int,
    checkpoint_name: str | None = None,
) -> dict[str, float]:
    summary = evaluate_happo_fixed_blue_4v3(
        trainer.actors,
        env_config,
        episodes=int(cfg["evaluation"]["episodes"]),
        num_envs=min(int(cfg["training"]["num_envs"]), 4),
        num_env_workers=0,
        device=trainer.device,
        seed=int(cfg["experiment"]["seed"]) + 50000 + int(env_steps),
    )
    eval_path = out / f"evaluation_{label}.json"
    _write_json(eval_path, summary)
    score, score_fields = compute_best_score_4v3(summary)
    row = {
        "label": label,
        "env_steps": int(env_steps),
        "summary": summary,
        "score": list(score),
        "score_fields": score_fields,
        "checkpoint": checkpoint_name,
    }
    trainer.evaluation_history.append(row)
    if trainer.best_score is None or score > trainer.best_score:
        trainer.best_score = score
        trainer.best_score_fields = score_fields
        trainer.best_evaluation = summary
        trainer.best_checkpoint_name = checkpoint_name or f"evaluation_{label}"
        trainer.save_checkpoint(out / "best.pt", is_best=True)
    return summary


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
    metric_fields = [
        "env_steps", "vector_steps", "update_count", "policy_loss", "actor_loss", "value_loss", "critic_loss",
        "entropy", "approx_kl", "clip_fraction", "actor_grad_norm", "actor_updates", "agents_updated",
        "alive_actor_samples", "advantage_mean", "advantage_std", "explained_variance",
        "current_actor_lr", "current_critic_lr", "current_entropy_coef",
    ]
    outcome_fields = [
        "recent_red_win_rate", "recent_red_complete_elimination_success_rate",
        "recent_red_at_least_two_attack_kill_rate", "recent_red_any_attack_kill_rate",
        "recent_mean_red_attack_kills", "recent_timeout_rate",
    ]
    reward_fields = [f"mean_rollout_{key}" for key in (
        "mission_reward", "kill_event_reward", "combat_loss_event_penalty",
        "support_loss_event_penalty", "boundary_event_penalty", "support_assisted_kill_reward",
        "total_dense_reward", "team_total_reward",
    )]
    fields = [
        *metric_fields, *outcome_fields, *reward_fields,
    ]
    try:
        trainer.save_checkpoint(out / "initial.pt")
        _evaluate_and_maybe_save_best(
            trainer, args.env_config, cfg, out, label="initial", env_steps=0, checkpoint_name="initial.pt"
        )
        next_eval = int(cfg["training"].get("evaluation_interval_env_steps", 100000))
        next_ckpt = int(cfg["training"].get("checkpoint_interval_env_steps", 100000))
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            while trainer.env_steps < int(cfg["training"]["total_env_steps"]):
                episodes = trainer.collect_rollout()
                metrics = trainer.update()
                recent = summarize_4v3_episodes(trainer.recent_episodes)
                row = {k: metrics.get(k, 0.0) for k in metric_fields + reward_fields}
                row.update({
                    "recent_red_win_rate": recent.get("red_win_rate", 0.0),
                    "recent_red_complete_elimination_success_rate": recent.get("red_complete_elimination_success_rate", 0.0),
                    "recent_red_at_least_two_attack_kill_rate": recent.get("red_at_least_two_attack_kill_rate", 0.0),
                    "recent_red_any_attack_kill_rate": recent.get("red_any_attack_kill_rate", 0.0),
                    "recent_mean_red_attack_kills": recent.get("mean_red_attack_kills", 0.0),
                    "recent_timeout_rate": recent.get("timeout_rate", 0.0),
                })
                writer.writerow({k: row.get(k, 0.0) for k in fields})
                fh.flush()
                while trainer.env_steps >= next_ckpt:
                    ckpt_name = f"step_{next_ckpt:07d}.pt"
                    trainer.save_checkpoint(out / ckpt_name)
                    next_ckpt += int(cfg["training"].get("checkpoint_interval_env_steps", 100000))
                while trainer.env_steps >= next_eval:
                    checkpoint_name = f"step_{next_eval:07d}.pt"
                    if not (out / checkpoint_name).exists():
                        trainer.save_checkpoint(out / checkpoint_name)
                    _evaluate_and_maybe_save_best(
                        trainer,
                        args.env_config,
                        cfg,
                        out,
                        label=f"step_{next_eval:07d}",
                        env_steps=next_eval,
                        checkpoint_name=checkpoint_name,
                    )
                    next_eval += int(cfg["training"].get("evaluation_interval_env_steps", 100000))

        eval_summary = evaluate_happo_fixed_blue_4v3(
            trainer.actors,
            args.env_config,
            episodes=int(cfg["evaluation"]["episodes"]),
            num_envs=min(int(cfg["training"]["num_envs"]), 4),
            num_env_workers=0,
            device=trainer.device,
            seed=int(cfg["experiment"]["seed"]) + 50000,
        )
        _write_json(out / "evaluation_final.json", eval_summary)
        score, score_fields = compute_best_score_4v3(eval_summary)
        trainer.evaluation_history.append({
            "label": "final",
            "env_steps": trainer.env_steps,
            "summary": eval_summary,
            "score": list(score),
            "score_fields": score_fields,
            "checkpoint": "final.pt",
        })
        if trainer.best_score is None or score > trainer.best_score:
            trainer.best_score = score
            trainer.best_score_fields = score_fields
            trainer.best_evaluation = eval_summary
            trainer.best_checkpoint_name = "final.pt"
            trainer.save_checkpoint(out / "best.pt", is_best=True)
        trainer.save_checkpoint(out / "final.pt")
        if not (out / "best.pt").exists():
            shutil.copyfile(out / "final.pt", out / "best.pt")
        trainer.write_summary(out)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
