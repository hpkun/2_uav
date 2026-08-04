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
from uav_combat.environment_4v3 import RED_REWARD_COMPONENT_KEYS_4V3


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


def _evaluation_seed_base(cfg: dict[str, Any]) -> int:
    """Return the stable seed base shared by initial, periodic, and final evals."""
    return int(cfg["experiment"]["seed"]) + int(cfg["evaluation"].get("seed_offset", 50000))


def _evaluate_and_maybe_save_best(
    trainer: HAPPO4v3Trainer,
    env_config: str,
    cfg: dict[str, Any],
    out: Path,
    *,
    label: str,
    scheduled_env_steps: int,
    actual_env_steps: int,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    evaluation_seed_base = int(getattr(trainer, "evaluation_seed_base", _evaluation_seed_base(cfg)))
    summary = evaluate_happo_fixed_blue_4v3(
        trainer.actors,
        env_config,
        episodes=int(cfg["evaluation"]["episodes"]),
        num_envs=min(int(cfg["training"]["num_envs"]), 4),
        num_env_workers=0,
        device=trainer.device,
        seed=evaluation_seed_base,
    )
    eval_path = out / f"evaluation_{label}.json"
    score, score_fields = compute_best_score_4v3(summary)
    evaluation_payload = {
        **summary,
        "label": label,
        "scheduled_env_steps": int(scheduled_env_steps),
        "actual_env_steps": int(actual_env_steps),
        "checkpoint": checkpoint_name,
        "score": list(score),
        "score_fields": score_fields,
        "evaluation_seed_base": evaluation_seed_base,
        "summary": summary,
    }
    _write_json(eval_path, evaluation_payload)
    row = {
        "label": label,
        "env_steps": int(actual_env_steps),
        "scheduled_env_steps": int(scheduled_env_steps),
        "actual_env_steps": int(actual_env_steps),
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
        trainer.best_scheduled_env_steps = int(scheduled_env_steps)
        trainer.best_actual_env_steps = int(actual_env_steps)
        trainer.save_checkpoint(
            out / "best.pt",
            is_best=True,
            scheduled_env_steps=scheduled_env_steps,
        )
    return summary


def _save_actual_step_checkpoint(
    trainer: HAPPO4v3Trainer,
    out: Path,
    *,
    scheduled_env_steps: int,
) -> str:
    actual_env_steps = int(trainer.env_steps)
    checkpoint_name = f"step_{actual_env_steps:07d}.pt"
    trainer.save_checkpoint(
        out / checkpoint_name,
        scheduled_env_steps=int(scheduled_env_steps),
    )
    return checkpoint_name


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
    reward_fields = [f"mean_rollout_{key}" for key in RED_REWARD_COMPONENT_KEYS_4V3]
    fields = [
        *metric_fields, *outcome_fields, *reward_fields,
    ]
    try:
        trainer.save_checkpoint(out / "initial.pt", scheduled_env_steps=0)
        _evaluate_and_maybe_save_best(
            trainer,
            args.env_config,
            cfg,
            out,
            label="initial",
            scheduled_env_steps=0,
            actual_env_steps=trainer.env_steps,
            checkpoint_name="initial.pt",
        )
        next_eval = int(cfg["training"].get("evaluation_interval_env_steps", 100000))
        next_ckpt = int(cfg["training"].get("checkpoint_interval_env_steps", 100000))
        eval_interval = int(cfg["training"].get("evaluation_interval_env_steps", 100000))
        ckpt_interval = int(cfg["training"].get("checkpoint_interval_env_steps", 100000))
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
                actual_env_steps = int(trainer.env_steps)
                crossed_checkpoints: list[int] = []
                while actual_env_steps >= next_ckpt:
                    crossed_checkpoints.append(next_ckpt)
                    next_ckpt += ckpt_interval
                crossed_evaluations: list[int] = []
                while actual_env_steps >= next_eval:
                    crossed_evaluations.append(next_eval)
                    next_eval += eval_interval

                # A rollout may cross several planned milestones. The current
                # model is saved once and evaluated once, with the first
                # crossed evaluation threshold retained as its scheduled label.
                crossed = [*crossed_checkpoints, *crossed_evaluations]
                if crossed:
                    checkpoint_schedule = min(crossed)
                    checkpoint_name = _save_actual_step_checkpoint(
                        trainer,
                        out,
                        scheduled_env_steps=checkpoint_schedule,
                    )
                    if crossed_evaluations:
                        scheduled_eval = crossed_evaluations[0]
                        _evaluate_and_maybe_save_best(
                            trainer,
                            args.env_config,
                            cfg,
                            out,
                            label=f"step_{actual_env_steps:07d}",
                            scheduled_env_steps=scheduled_eval,
                            actual_env_steps=actual_env_steps,
                            checkpoint_name=checkpoint_name,
                        )

        final_actual_env_steps = int(trainer.env_steps)
        trainer.save_checkpoint(out / "final.pt", scheduled_env_steps=final_actual_env_steps)
        _evaluate_and_maybe_save_best(
            trainer,
            args.env_config,
            cfg,
            out,
            label="final",
            scheduled_env_steps=final_actual_env_steps,
            actual_env_steps=final_actual_env_steps,
            checkpoint_name="final.pt",
        )
        if not (out / "best.pt").exists():
            shutil.copyfile(out / "final.pt", out / "best.pt")
        trainer.write_summary(out)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
