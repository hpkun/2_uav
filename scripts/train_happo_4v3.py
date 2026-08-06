"""Train, resume, or smoke-test the functional heterogeneous 4v3 v9 HAPPO run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from uav_combat.config import load_config
from uav_combat.environment_4v3 import RED_REWARD_COMPONENT_KEYS_4V3
from uav_combat.happo.evaluation_4v3 import (
    build_evaluation_seed_manifest,
    evaluate_happo_fixed_blue_4v3,
    evaluation_seeds_from_manifest,
    sha256_json_4v3,
    validate_evaluation_seed_manifest,
)
from uav_combat.happo.trainer_4v3 import (
    HAPPO4v3Trainer,
    best_score_fields_4v3,
    compute_best_score_4v3,
    summarize_4v3_episodes,
)


def load_train_config(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if args.device is not None:
        cfg["experiment"]["device"] = args.device
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.total_env_steps is not None:
        cfg["training"]["total_env_steps"] = int(args.total_env_steps)
    if args.num_envs is not None:
        cfg["training"]["num_envs"] = int(args.num_envs)
    if args.env_workers is not None:
        cfg["training"]["num_env_workers"] = int(args.env_workers)
    if args.resume is not None and args.output_dir is None:
        cfg["experiment"]["output_dir"] = str(Path(args.resume).resolve().parent)
    if args.smoke:
        num_envs = int(cfg["training"]["num_envs"])
        total = min(int(cfg["training"]["total_env_steps"]), 8192)
        total -= total % num_envs
        cfg["training"]["total_env_steps"] = max(num_envs, total)
        cfg["training"]["evaluation_interval_env_steps"] = cfg["training"]["total_env_steps"]
        cfg["training"]["checkpoint_interval_env_steps"] = cfg["training"]["total_env_steps"]
        cfg["evaluation"]["selection_episodes"] = min(int(cfg["evaluation"].get("selection_episodes", 1)), 1)
        cfg["evaluation"]["test_episodes"] = min(int(cfg["evaluation"].get("test_episodes", 1)), 1)
    return cfg


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_episode_csv(path: Path, records: list[dict[str, Any]], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    rows: list[dict[str, Any]] = []
    fields: set[str] = set()
    for record in records:
        row = {}
        for key, value in record.items():
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            fields.add(key)
        rows.append(row)
    ordered = ["episode_seed"] + sorted(fields - {"episode_seed"}) if rows else ["episode_seed"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_payload(summary: dict[str, Any], *, label: str, checkpoint: str | None, checkpoint_sha256: str | None) -> dict[str, Any]:
    payload = deepcopy(summary)
    records = payload.pop("episode_records", [])
    payload.update({
        "label": label,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "episode_record_count": len(records),
    })
    return payload


def _refresh_evaluation_checkpoint_sha(path: Path, checkpoint_path: Path) -> None:
    """Keep a periodic evaluation's recorded SHA aligned with its checkpoint."""
    if not path.exists() or not checkpoint_path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = _checkpoint_sha256(checkpoint_path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_or_create_manifest(cfg: dict[str, Any], out: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.seed_manifest) if args.seed_manifest else out / "evaluation_seed_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_evaluation_seed_manifest(manifest)
    else:
        evaluation = cfg["evaluation"]
        manifest = build_evaluation_seed_manifest(
            int(cfg["experiment"]["seed"]),
            selection_episodes=int(evaluation["selection_episodes"]),
            test_episodes=int(evaluation["test_episodes"]),
            selection_seed_offset=int(evaluation["selection_seed_offset"]),
            test_seed_offset=int(evaluation["test_seed_offset"]),
        )
    if manifest_path.resolve() != (out / "evaluation_seed_manifest.json").resolve():
        _write_json(out / "evaluation_seed_manifest.json", manifest, overwrite=not bool(args.resume))
    else:
        _write_json(manifest_path, manifest, overwrite=not bool(args.resume))
    return manifest


def _write_contract(
    out: Path,
    env_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    manifest: dict[str, Any],
    env_config_path: str | Path,
    train_config_path: str | Path,
) -> None:
    contract = {
        "checkpoint_family": "functional_heterogeneous_4v3_v9_happo",
        "checkpoint_version": 1,
        "variant": train_cfg.get("experiment", {}).get("variant"),
        "reward_contract_version": env_cfg.get("combat", {}).get("reward_contract_version"),
        "environment_config_sha256": _checkpoint_sha256(Path(env_config_path)),
        "training_config_sha256": _checkpoint_sha256(Path(train_config_path)),
        "resolved_environment_config_sha256": sha256_json_4v3(env_cfg),
        "resolved_training_config_sha256": sha256_json_4v3(train_cfg),
        "reward_contract": deepcopy(env_cfg["rewards"]),
        "agent_roles": deepcopy(env_cfg["heterogeneous"]["roles"]),
        "sensor_ranges": deepcopy(env_cfg["heterogeneous"]["sensor_range"]),
        "can_attack": deepcopy(env_cfg["heterogeneous"]["can_attack"]),
        "team_sizes": {
            "red": int(env_cfg["scenario"]["red_team_size"]),
            "blue": int(env_cfg["scenario"]["blue_team_size"]),
        },
        "observation_dims": deepcopy(train_cfg["training"]["observation_dims"]),
        "state_dim": 70,
        "action_dims": deepcopy(train_cfg["training"]["action_dims"]),
        "happo": {
            key: deepcopy(value)
            for key, value in train_cfg["training"].items()
            if key in {"ppo_epochs", "minibatch_size", "gamma", "gae_lambda", "clip_coef", "value_loss_coef"}
        },
        "evaluation_seed_manifest_hash": manifest["manifest_hash"],
        "selection_seed_hash": manifest["selection"]["seed_hash"],
        "test_seed_hash": manifest["test"]["seed_hash"],
    }
    _write_json(out / "experiment_contract.json", contract)


def _eval_and_write(
    trainer: HAPPO4v3Trainer,
    env_config: str,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    out: Path,
    *,
    label: str,
    split: str,
    checkpoint_path: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    seeds = evaluation_seeds_from_manifest(manifest, split)
    summary = evaluate_happo_fixed_blue_4v3(
        trainer.actors,
        env_config,
        seeds=seeds,
        num_envs=min(int(cfg["training"]["num_envs"]), 8),
        num_env_workers=min(int(cfg["training"].get("num_env_workers", 0)), 4),
        device=trainer.device,
        split=split,
        seed_manifest=manifest,
    )
    checkpoint_sha = _checkpoint_sha256(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None
    aggregate = _aggregate_payload(summary, label=label, checkpoint=str(checkpoint_path) if checkpoint_path else None, checkpoint_sha256=checkpoint_sha)
    _write_json(out / f"{label}_evaluation.json", aggregate, overwrite=overwrite)
    _write_episode_csv(out / f"{label}_per_episode.csv", summary.get("episode_records", []), overwrite=overwrite)
    trainer.evaluation_history.append({
        "label": label,
        "env_steps": int(trainer.env_steps),
        "scheduled_env_steps": int(trainer.env_steps),
        "actual_env_steps": int(trainer.env_steps),
        "summary": deepcopy(summary),
        "score": list(compute_best_score_4v3(summary)[0]),
        "score_fields": compute_best_score_4v3(summary)[1],
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    })
    return summary


def _load_manifest_from_checkpoint(path: str | Path) -> dict[str, Any] | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    manifest = ckpt.get("seed_manifest")
    return manifest if isinstance(manifest, dict) and manifest.get("selection") else None


def _validate_milestones(cfg: dict[str, Any]) -> None:
    training = cfg["training"]
    num_envs = int(training["num_envs"])
    total = int(training["total_env_steps"])
    eval_interval = int(training["evaluation_interval_env_steps"])
    ckpt_interval = int(training["checkpoint_interval_env_steps"])
    if total <= 0 or total % num_envs:
        raise ValueError("total_env_steps must be a positive multiple of num_envs")
    if total > 3_000_000:
        raise ValueError("total_env_steps must not exceed the formal 3M experiment budget")
    for name, value in (("evaluation_interval_env_steps", eval_interval), ("checkpoint_interval_env_steps", ckpt_interval)):
        if value <= 0 or value % num_envs:
            raise ValueError(f"{name} must be a positive multiple of num_envs")


def rollout_lengths_to_milestone(
    current_env_steps: int,
    milestone_target: int,
    num_envs: int,
    configured_rollout_steps: int,
) -> list[int]:
    """Return full/partial rollout lengths needed to hit one exact target."""
    current = int(current_env_steps)
    target = int(milestone_target)
    envs = int(num_envs)
    configured = int(configured_rollout_steps)
    if envs <= 0 or configured <= 0 or target < current:
        raise ValueError("invalid rollout milestone arguments")
    if (target - current) % envs:
        raise ValueError("milestone target must be reachable in whole vector steps")
    lengths: list[int] = []
    while current < target:
        steps = min(configured, (target - current) // envs)
        if steps <= 0:
            raise ValueError("milestone requires at least one vector step")
        lengths.append(steps)
        current += steps * envs
    return lengths


def training_throughput(env_steps: int, run_start_env_steps: int, elapsed_seconds: float) -> float:
    return float(max(0, int(env_steps) - int(run_start_env_steps)) / max(float(elapsed_seconds), 1e-9))


def format_train_log_v11(
    *, step: int, total: int, update: int, throughput: float, r_step: float,
    episode_return: float, task_win: float, full: float, mean_kills: float,
    timeout: float, mission: float, event: float, geom: float, lock: float, support: float,
) -> str:
    """Pure formatter used by the v11 terminal contract and its tests."""
    return (
        f"[train] step={step}/{total} upd={update} fps={throughput:.1f} "
        f"r_step={r_step:+.5f} ep_return={episode_return:+.2f} "
        f"win={task_win:.3f} full={full:.3f} kills={mean_kills:.3f} timeout={timeout:.3f} "
        f"reward{{mission={mission:+.5f} event={event:+.5f} geom={geom:+.5f} "
        f"lock={lock:+.5f} support={support:+.5f}}}"
    )


def format_eval_log_v11(step: int, summary: dict[str, Any]) -> str:
    return (
        f"[eval] step={step} episodes={int(summary.get('episodes', 0))} "
        f"task_win={summary.get('task_win_rate', 0.0):.3f} "
        f"full={summary.get('full_elimination_rate', 0.0):.3f} "
        f"any_kill={summary.get('any_kill_rate', 0.0):.3f} "
        f"two_plus={summary.get('at_least_two_kill_rate', 0.0):.3f} "
        f"mean_kills={summary.get('mean_red_kills', 0.0):.3f} "
        f"timeout_win={summary.get('timeout_win_rate', 0.0):.3f} "
        f"timeout_loss={summary.get('timeout_loss_rate', 0.0):.3f} "
        f"draw={summary.get('timeout_draw_rate', 0.0):.3f} "
        f"mean_return={summary.get('mean_return', 0.0):+.2f} "
        f"lock_episode={summary.get('lock_episode_rate', 0.0):.3f} "
        f"support_assist={summary.get('support_assisted_kill_rate', 0.0):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--train-config", default="configs/happo_heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-workers", type=int)
    parser.add_argument("--seed-manifest")
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_train_config(args.train_config, args)
    _validate_milestones(cfg)
    out = Path(cfg["experiment"]["output_dir"])
    if out.exists() and any(out.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; use --resume or --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = load_config(args.env_config)
    _write_yaml(out / "resolved_environment_config.yaml", env_cfg)
    _write_yaml(out / "resolved_training_config.yaml", cfg)
    manifest = _load_or_create_manifest(cfg, out, args)
    _write_contract(out, env_cfg, cfg, manifest, args.env_config, args.train_config)

    trainer = HAPPO4v3Trainer(args.env_config, cfg)
    trainer.seed_manifest = deepcopy(manifest)
    is_v11 = trainer.is_v11
    metrics_path = out / "training_metrics.csv"
    metric_fields = [
        "env_steps", "total_env_steps", "vector_steps", "update_count", "effective_rollout_steps",
        "wall_clock_seconds", "throughput_env_steps_per_second", "policy_loss", "actor_loss", "value_loss",
        "critic_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm", "actor_updates",
        "agents_updated", "alive_actor_samples", "unique_alive_actor_samples", "advantage_mean", "advantage_std",
        "explained_variance", "current_actor_lr", "current_critic_lr", "current_entropy_coef", "ratio_min",
        "ratio_mean", "ratio_max", "factor_min", "factor_mean", "factor_max",
    ]
    for agent_id in range(4):
        for name in ("yaw", "pitch", "speed"):
            metric_fields.extend([f"actor_{agent_id}_log_std_{name}", f"actor_{agent_id}_std_{name}"])
    outcome_fields = [
        "recent_red_win_rate", "recent_red_at_least_two_attack_kill_rate", "recent_red_any_attack_kill_rate",
        "recent_mean_red_attack_kills", "recent_timeout_rate", "recent_support_assisted_kill_rate",
        "recent_support_assisted_episode_rate", "recent_support_active_steps", "recent_support_shared_pair_step_rate",
        "recent_mean_shared_only_pair_ratio",
        "recent_dense_clip_positive_saturation_rate", "recent_dense_clip_negative_saturation_rate",
        "recent_dense_clip_saturation_rate",
    ]
    if is_v11:
        outcome_fields = [
            "last_rollout_team_reward_per_step", "recent_episode_return_mean", "recent_episode_return_std",
            "recent_task_win_rate", "recent_full_elimination_rate", "recent_timeout_win_rate",
            "recent_timeout_loss_rate", "recent_timeout_draw_rate", "recent_mean_red_kills",
            "recent_mean_blue_kills", "recent_lock_episode_rate", "recent_half_lock_episode_rate",
            "recent_mean_max_lock_progress", "recent_target_switches", "recent_support_cue_rate",
            "recent_support_cue_to_direct_rate", "recent_support_assisted_kill_rate",
        ]
    fields = [*metric_fields, *outcome_fields, *[f"mean_rollout_{key}" for key in trainer.reward_keys]]
    resumed = bool(args.resume)
    try:
        if resumed:
            checkpoint_manifest = _load_manifest_from_checkpoint(args.resume)
            if checkpoint_manifest is not None:
                validate_evaluation_seed_manifest(checkpoint_manifest)
                if checkpoint_manifest["manifest_hash"] != manifest["manifest_hash"]:
                    raise ValueError("resume checkpoint seed manifest differs from output manifest")
            trainer.load_checkpoint(args.resume)
            trainer.seed_manifest = deepcopy(manifest)
        else:
            trainer.next_evaluation_env_steps = int(cfg["training"]["evaluation_interval_env_steps"])
            trainer.next_checkpoint_env_steps = int(cfg["training"]["checkpoint_interval_env_steps"])
            trainer.save_checkpoint(out / "initial.pt", scheduled_env_steps=0)
            _eval_and_write(
                trainer, args.env_config, cfg, manifest, out,
                label="initial_selection", split="selection", checkpoint_path=out / "initial.pt", overwrite=True,
            )
            initial = trainer.evaluation_history[-1]["summary"] if trainer.evaluation_history else None
            if initial is None:
                initial = summarize_4v3_episodes([])
            score, score_fields = compute_best_score_4v3(initial)
            trainer.best_score = score
            trainer.best_score_fields = score_fields
            trainer.best_evaluation = initial
            trainer.best_checkpoint_name = "initial.pt"
            trainer.best_scheduled_env_steps = 0
            trainer.best_actual_env_steps = 0
            trainer.save_checkpoint(out / "best.pt", is_best=True, scheduled_env_steps=0)

        run_start_env_steps = int(trainer.env_steps)
        started = time.perf_counter()
        header_exists = resumed and metrics_path.exists() and metrics_path.stat().st_size > 0
        mode = "a" if resumed else "w"
        with metrics_path.open(mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if header_exists:
                with metrics_path.open(encoding="utf-8") as header_fh:
                    existing = next(csv.reader([header_fh.readline().rstrip("\n")]))
                if existing != fields:
                    raise ValueError("training_metrics.csv header does not match the resolved contract")
            else:
                writer.writeheader()
            next_eval = int(trainer.next_evaluation_env_steps or cfg["training"]["evaluation_interval_env_steps"])
            next_ckpt = int(trainer.next_checkpoint_env_steps or cfg["training"]["checkpoint_interval_env_steps"])
            eval_interval = int(cfg["training"]["evaluation_interval_env_steps"])
            ckpt_interval = int(cfg["training"]["checkpoint_interval_env_steps"])
            total_target = int(cfg["training"]["total_env_steps"])

            # A legacy checkpoint may have stored the threshold that was just
            # reached. Normalize it before entering the next milestone.
            while next_eval <= trainer.env_steps:
                next_eval += eval_interval
            while next_ckpt <= trainer.env_steps:
                next_ckpt += ckpt_interval
            trainer.next_evaluation_env_steps = next_eval
            trainer.next_checkpoint_env_steps = next_ckpt

            def record_update(metrics: dict[str, float]) -> dict[str, Any]:
                recent = summarize_4v3_episodes(trainer.recent_episodes)
                elapsed = time.perf_counter() - started
                actual = int(trainer.env_steps)
                row = {key: metrics.get(key, 0.0) for key in metric_fields}
                row.update({
                    "total_env_steps": total_target,
                    "wall_clock_seconds": elapsed,
                    "throughput_env_steps_per_second": training_throughput(actual, run_start_env_steps, elapsed),
                    "recent_red_win_rate": recent.get("red_win_rate", 0.0),
                    "recent_red_at_least_two_attack_kill_rate": recent.get("red_at_least_two_attack_kill_rate", 0.0),
                    "recent_red_any_attack_kill_rate": recent.get("red_any_attack_kill_rate", 0.0),
                    "recent_mean_red_attack_kills": recent.get("mean_red_attack_kills", 0.0),
                    "recent_timeout_rate": recent.get("timeout_rate", 0.0),
                    "recent_support_assisted_kill_rate": recent.get("support_assisted_kill_rate", 0.0),
                    "recent_support_assisted_episode_rate": recent.get("support_assisted_episode_rate", 0.0),
                    "recent_support_active_steps": recent.get("mean_support_active_steps", 0.0),
                    "recent_support_shared_pair_step_rate": recent.get("support_shared_pair_step_rate", 0.0),
                    "recent_mean_shared_only_pair_ratio": recent.get("mean_shared_only_pair_ratio", 0.0),
                    "recent_dense_clip_positive_saturation_rate": recent.get("mean_dense_clip_positive_saturation_rate", 0.0),
                    "recent_dense_clip_negative_saturation_rate": recent.get("mean_dense_clip_negative_saturation_rate", 0.0),
                    "recent_dense_clip_saturation_rate": recent.get("mean_dense_clip_saturation_rate", 0.0),
                })
                if is_v11:
                    returns = [float(record.get("episode_return", 0.0)) for record in trainer.recent_episodes]
                    reward_means = trainer.last_rollout_reward_means
                    event_keys = (
                        "blue_kill_event_reward", "red_combat_loss_event_penalty",
                        "support_loss_event_penalty", "boundary_event_penalty",
                        "support_assisted_kill_reward",
                    )
                    support_keys = (
                        "support_unique_detection_reward", "support_cue_to_direct_reward",
                        "support_cue_to_half_lock_reward", "support_assisted_kill_reward",
                        "support_formation_progress_reward",
                    )
                    row.update({
                        "last_rollout_team_reward_per_step": reward_means.get("mean_rollout_team_total_reward", 0.0),
                        "recent_episode_return_mean": float(sum(returns) / max(1, len(returns))),
                        "recent_episode_return_std": float(torch.tensor(returns).std(unbiased=False).item()) if returns else 0.0,
                        "recent_task_win_rate": recent.get("task_win_rate", 0.0),
                        "recent_full_elimination_rate": recent.get("full_elimination_rate", 0.0),
                        "recent_timeout_win_rate": recent.get("timeout_win_rate", 0.0),
                        "recent_timeout_loss_rate": recent.get("timeout_loss_rate", 0.0),
                        "recent_timeout_draw_rate": recent.get("timeout_draw_rate", 0.0),
                        "recent_mean_red_kills": recent.get("mean_red_kills", 0.0),
                        "recent_mean_blue_kills": recent.get("mean_blue_kills", 0.0),
                        "recent_lock_episode_rate": recent.get("lock_episode_rate", 0.0),
                        "recent_half_lock_episode_rate": recent.get("half_lock_episode_rate", 0.0),
                        "recent_mean_max_lock_progress": recent.get("mean_max_lock_progress", 0.0),
                        "recent_target_switches": recent.get("mean_target_switch_count", 0.0),
                        "recent_support_cue_rate": recent.get("support_cue_rate", 0.0),
                        "recent_support_cue_to_direct_rate": recent.get("support_cue_to_direct_rate", 0.0),
                        "recent_support_assisted_kill_rate": recent.get("support_assisted_kill_rate", 0.0),
                    })
                    row["_v11_log"] = format_train_log_v11(
                        step=actual, total=total_target, update=trainer.update_count,
                        throughput=row["throughput_env_steps_per_second"],
                        r_step=row["last_rollout_team_reward_per_step"],
                        episode_return=row["recent_episode_return_mean"],
                        task_win=row["recent_task_win_rate"], full=row["recent_full_elimination_rate"],
                        mean_kills=row["recent_mean_red_kills"], timeout=recent.get("timeout_rate", 0.0),
                        mission=reward_means.get("mean_rollout_mission_outcome_reward", 0.0),
                        event=sum(reward_means.get(f"mean_rollout_{key}", 0.0) for key in event_keys),
                        geom=reward_means.get("mean_rollout_combat_geometry_progress_reward", 0.0),
                        lock=reward_means.get("mean_rollout_combat_lock_progress_reward", 0.0) + reward_means.get("mean_rollout_combat_half_lock_event_reward", 0.0),
                        support=sum(reward_means.get(f"mean_rollout_{key}", 0.0) for key in support_keys),
                    )
                row.update(trainer.last_rollout_reward_means)
                writer.writerow({key: row.get(key, 0.0) for key in fields})
                fh.flush()
                trainer.save_checkpoint(out / "latest.pt", scheduled_env_steps=actual)
                if is_v11:
                    print(row["_v11_log"], flush=True)
                else:
                    print(
                        f"env_steps={actual}/{total_target} update={trainer.update_count} "
                        f"throughput={training_throughput(actual, run_start_env_steps, elapsed):.1f}/s "
                        f"recent_win={recent.get('red_win_rate', 0.0):.3f} "
                        f"kills={recent.get('mean_red_attack_kills', 0.0):.3f} "
                        f"support_assisted={recent.get('support_assisted_kill_rate', 0.0):.3f}", flush=True,
                    )
                return recent

            while trainer.env_steps < total_target:
                milestone_target = min(total_target, next_eval, next_ckpt)
                for _ in rollout_lengths_to_milestone(
                    trainer.env_steps,
                    milestone_target,
                    trainer.num_envs,
                    trainer.rollout_steps,
                ):
                    remaining = milestone_target - int(trainer.env_steps)
                    trainer.collect_rollout(max_env_steps=remaining)
                    metrics = trainer.update()
                    record_update(metrics)

                if trainer.env_steps != milestone_target:
                    raise RuntimeError(
                        f"inexact milestone after inner rollout loop: "
                        f"scheduled={milestone_target}, actual={trainer.env_steps}"
                    )

                actual = int(trainer.env_steps)
                hit_ckpt = actual == next_ckpt
                hit_eval = actual == next_eval
                if hit_ckpt:
                    next_ckpt += ckpt_interval
                if hit_eval:
                    next_eval += eval_interval
                trainer.next_evaluation_env_steps = next_eval
                trainer.next_checkpoint_env_steps = next_ckpt
                if hit_ckpt:
                    trainer.save_checkpoint(out / f"step_{actual:07d}.pt", scheduled_env_steps=milestone_target)
                if hit_eval:
                    periodic_label = f"evaluation_selection_step_{actual:07d}"
                    periodic_checkpoint = (
                        out / f"step_{actual:07d}.pt" if hit_ckpt else out / "latest.pt"
                    )
                    periodic_report_path = out / f"{periodic_label}_evaluation.json"
                    periodic_report_existed = periodic_report_path.exists()
                    summary = _eval_and_write(
                        trainer, args.env_config, cfg, manifest, out,
                        label=periodic_label, split="selection", checkpoint_path=periodic_checkpoint,
                        overwrite=not resumed,
                    )
                    if is_v11:
                        print(format_eval_log_v11(actual, summary), flush=True)
                    score, score_fields = compute_best_score_4v3(summary)
                    if trainer.best_score is None or score > trainer.best_score:
                        trainer.best_score = score
                        trainer.best_score_fields = score_fields
                        trainer.best_evaluation = summary
                        trainer.best_checkpoint_name = f"step_{actual:07d}.pt" if hit_ckpt else "latest.pt"
                        trainer.best_scheduled_env_steps = actual
                        trainer.best_actual_env_steps = actual
                        trainer.save_checkpoint(out / "best.pt", is_best=True, scheduled_env_steps=actual)
                        if is_v11:
                            print(f"[best] step={actual} previous={trainer.best_checkpoint_name} new=best.pt score_fields={score_fields}", flush=True)
                # This is intentionally after evaluation/best metadata updates.
                trainer.save_checkpoint(out / "latest.pt", scheduled_env_steps=actual)
                if hit_eval and (not resumed or not periodic_report_existed):
                    _refresh_evaluation_checkpoint_sha(
                        periodic_report_path,
                        periodic_checkpoint,
                    )
            trainer.next_evaluation_env_steps = next_eval
            trainer.next_checkpoint_env_steps = next_ckpt
            trainer.save_checkpoint(out / "final.pt", scheduled_env_steps=trainer.env_steps)

        # Final reporting evaluates both checkpoint families on both fixed splits.
        for checkpoint_name in ("best", "final"):
            checkpoint_path = out / f"{checkpoint_name}.pt"
            report_trainer = HAPPO4v3Trainer(args.env_config, cfg)
            report_trainer.seed_manifest = deepcopy(manifest)
            try:
                report_trainer.load_checkpoint(checkpoint_path)
                for split in ("selection", "test"):
                    _eval_and_write(
                        report_trainer, args.env_config, cfg, manifest, out,
                        label=f"{checkpoint_name}_{split}", split=split, checkpoint_path=checkpoint_path,
                        overwrite=True,
                    )
            finally:
                report_trainer.close()
        trainer.write_summary(out)
        if is_v11:
            final = trainer.evaluation_history[-1]["summary"] if trainer.evaluation_history else {}
            print(
                f"[done] final_env_steps={trainer.env_steps} updates={trainer.update_count} "
                f"best_step={trainer.best_actual_env_steps} best_task_win={trainer.best_evaluation.get('task_win_rate', 0.0) if trainer.best_evaluation else 0.0:.3f} "
                f"best_full={trainer.best_evaluation.get('full_elimination_rate', 0.0) if trainer.best_evaluation else 0.0:.3f} "
                f"best_mean_kills={trainer.best_evaluation.get('mean_red_kills', 0.0) if trainer.best_evaluation else 0.0:.3f} "
                f"final_task_win={final.get('task_win_rate', 0.0):.3f} final_full={final.get('full_elimination_rate', 0.0):.3f} "
                f"final_mean_kills={final.get('mean_red_kills', 0.0):.3f} output={out}", flush=True,
            )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
