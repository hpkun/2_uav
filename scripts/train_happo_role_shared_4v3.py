"""Train v13 or mission-aligned v14 role-shared HAPPO experiments."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from uav_combat.config import load_config
from uav_combat.happo.evaluation_4v3 import (
    build_evaluation_seed_manifest,
    evaluation_seeds_from_manifest,
    validate_evaluation_seed_manifest,
)
from uav_combat.happo.evaluation_role_shared_4v3 import evaluate_role_shared_happo_fixed_blue_4v3
from uav_combat.happo.evaluation_v14_4v3 import evaluate_v14_happo_fixed_blue_4v3
from uav_combat.happo.trainer_4v3 import (
    V15_BEST_SCORE_FIELDS_4V3,
    V15_REWARD_CONTRACT_VERSION,
    V16_REWARD_CONTRACT_VERSION,
    compute_experiment_best_score_4v3,
)
from uav_combat.happo.trainer_role_shared_4v3 import (
    CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3,
    ROLE_POLICY_MAPPING,
    RoleSharedHAPPO4v3Trainer,
)
from uav_combat.happo.trainer_v14_4v3 import (
    CHECKPOINT_FAMILY_V14_HAPPO_4V3,
    CREDIT_MODE_ROLE_LOCAL,
    MissionAlignedRoleSharedHAPPO4v3Trainer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else x.tolist()), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_episode_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in records for key in row})
    if "episode_seed" in fields:
        fields.remove("episode_seed"); fields.insert(0, "episode_seed")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or ["episode_seed"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})


def _load_config(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if args.device is not None: cfg["experiment"]["device"] = args.device
    if args.output_dir is not None: cfg["experiment"]["output_dir"] = args.output_dir
    if args.total_env_steps is not None: cfg["training"]["total_env_steps"] = int(args.total_env_steps)
    if args.num_envs is not None: cfg["training"]["num_envs"] = int(args.num_envs)
    if args.env_workers is not None: cfg["training"]["num_env_workers"] = int(args.env_workers)
    if args.resume is not None and args.output_dir is None:
        cfg["experiment"]["output_dir"] = str(Path(args.resume).resolve().parent)
    if args.smoke:
        envs = int(cfg["training"]["num_envs"])
        total = min(8192, int(cfg["training"]["total_env_steps"]))
        cfg["training"]["total_env_steps"] = max(envs, total - total % envs)
        cfg["training"]["evaluation_interval_env_steps"] = cfg["training"]["total_env_steps"]
        cfg["training"]["checkpoint_interval_env_steps"] = cfg["training"]["total_env_steps"]
        cfg["evaluation"]["selection_episodes"] = min(2, int(cfg["evaluation"]["selection_episodes"]))
        cfg["evaluation"]["test_episodes"] = min(2, int(cfg["evaluation"]["test_episodes"]))
    return cfg


def _manifest(cfg: dict[str, Any], out: Path, supplied: str | None) -> dict[str, Any]:
    path = Path(supplied) if supplied else out / "evaluation_seed_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8")); validate_evaluation_seed_manifest(manifest)
    else:
        ev = cfg["evaluation"]
        manifest = build_evaluation_seed_manifest(
            int(cfg["experiment"]["seed"]), selection_episodes=int(ev["selection_episodes"]),
            test_episodes=int(ev["test_episodes"]), selection_seed_offset=int(ev["selection_seed_offset"]),
            test_seed_offset=int(ev["test_seed_offset"]),
        )
    _write_json(out / "evaluation_seed_manifest.json", manifest)
    return manifest


def _is_v14(cfg: dict[str, Any]) -> bool:
    return cfg["training"].get("training_mode") in {
        "fixed_rule_blue_heterogeneous_4v3_v14_happo",
        "fixed_rule_blue_heterogeneous_4v3_v15_happo",
        "fixed_rule_blue_heterogeneous_4v3_v16_happo",
    }


def _trainer_class(cfg: dict[str, Any]):
    return MissionAlignedRoleSharedHAPPO4v3Trainer if _is_v14(cfg) else RoleSharedHAPPO4v3Trainer


def _evaluate(
    trainer: Any,
    env_config: str,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    out: Path,
    label: str,
    split: str,
    checkpoint: Path,
    reward_contract_version: str,
) -> dict[str, Any]:
    seeds = evaluation_seeds_from_manifest(manifest, split)
    if _is_v14(cfg):
        summary = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, env_config, seeds=seeds,
            num_envs=min(8, int(cfg["training"]["num_envs"])),
            device=trainer.device, split=split, seed_manifest=manifest,
            training_diagnostics=trainer.last_update_metrics,
        )
    else:
        summary = evaluate_role_shared_happo_fixed_blue_4v3(
            trainer.actors, env_config, seeds=seeds,
            num_envs=min(8, int(cfg["training"]["num_envs"])),
            device=trainer.device, split=split, seed_manifest=manifest,
        )
    records = summary.pop("episode_records")
    _write_json(out / f"{label}_evaluation.json", {**summary, "label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint)})
    _write_episode_csv(out / f"{label}_per_episode.csv", records)
    full = {**summary, "episode_records": records}
    score, fields = compute_experiment_best_score_4v3(
        full, reward_contract_version
    )
    trainer.evaluation_history.append({
        "label": label, "env_steps": trainer.env_steps, "summary": deepcopy(full),
        "score": list(score), "score_fields": fields, "checkpoint": str(checkpoint),
    })
    return full


def _format_update_line(
    trainer: Any, total: int, metrics: dict[str, Any], *, is_v15: bool
) -> str:
    prefix = f"env_steps={trainer.env_steps}/{total} update={trainer.update_count}"
    if is_v15:
        prefix += (
            f" reward={float(metrics.get('mean_rollout_team_total_reward', 0.0)):.5f}"
            f" combat_state_r={float(metrics.get('mean_rollout_combat_state_reward', 0.0)):.5f}"
            f" support_state_r={float(metrics.get('mean_rollout_support_state_reward', 0.0)):.5f}"
        )
    return (
        f"{prefix} order={metrics['group_update_order']} "
        f"support_kl={metrics['support_kl']:.5f} "
        f"combat_kl={metrics['combat_joint_kl']:.5f} "
        f"entropy={metrics['entropy']:.4f}"
    )


def _update_best_from_latest_evaluation(
    trainer: Any, checkpoint: Path, scheduled_env_steps: int
) -> bool:
    """Apply the score already produced by the single selector dispatch."""
    entry = trainer.evaluation_history[-1]
    score = tuple(float(value) for value in entry["score"])
    if trainer.best_score is not None and score <= trainer.best_score:
        return False
    trainer.best_score = score
    trainer.best_score_fields = dict(entry["score_fields"])
    trainer.best_evaluation = entry["summary"]
    trainer.best_checkpoint_name = checkpoint.name
    trainer.best_scheduled_env_steps = int(scheduled_env_steps)
    trainer.best_actual_env_steps = int(trainer.env_steps)
    return True


def _build_experiment_contract(
    *,
    checkpoint_family: str,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    env_config: str,
    train_config: str,
    manifest_hash: str,
) -> dict[str, Any]:
    reward_contract_version = str(env_cfg["combat"]["reward_contract_version"])
    contract = {
        "checkpoint_family": checkpoint_family,
        "checkpoint_version": 1,
        "algorithm_variant": cfg["experiment"]["variant"],
        "role_policy_mapping": ROLE_POLICY_MAPPING,
        "environment_config": env_config,
        "training_config": train_config,
        "reward_contract_version": reward_contract_version,
        "reward_contract": env_cfg["rewards"],
        "training_credit_mode": cfg["training"].get("credit_mode"),
        "team_reward_usage": cfg["training"].get("team_reward_usage", "training"),
        "evaluation_seed_manifest_hash": manifest_hash,
    }
    if reward_contract_version in {
        V15_REWARD_CONTRACT_VERSION, V16_REWARD_CONTRACT_VERSION
    }:
        contract["best_checkpoint_selection"] = list(V15_BEST_SCORE_FIELDS_4V3)
    if reward_contract_version == V16_REWARD_CONTRACT_VERSION:
        contract["observation_contract"] = env_cfg["combat"][
            "observation_contract"
        ]
    return contract


def _append_metrics(path: Path, metrics: dict[str, Any], elapsed: float, start_steps: int) -> None:
    row = dict(metrics)
    row["wall_clock_seconds"] = float(elapsed)
    row["throughput_env_steps_per_second"] = float((float(metrics["env_steps"]) - start_steps) / max(elapsed, 1e-9))
    fields = sorted(row)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8") as fh:
            existing = next(csv.reader(fh))
        if existing != fields:
            raise ValueError("training_metrics.csv schema changed during the run")
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow(row)


def _parameter_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def _parameters_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(old, new.detach().cpu()) for old, new in zip(before, module.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-workers", type=int)
    parser.add_argument("--seed-manifest")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = _load_config(args.train_config, args)
    out = Path(cfg["experiment"]["output_dir"])
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = load_config(args.env_config)
    _write_yaml(out / "resolved_environment_config.yaml", env_cfg)
    _write_yaml(out / "resolved_training_config.yaml", cfg)
    manifest = _manifest(cfg, out, args.seed_manifest)
    trainer_class = _trainer_class(cfg)
    checkpoint_family = CHECKPOINT_FAMILY_V14_HAPPO_4V3 if _is_v14(cfg) else CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3
    reward_contract_version = str(env_cfg["combat"]["reward_contract_version"])
    is_v15 = reward_contract_version in {
        V15_REWARD_CONTRACT_VERSION, V16_REWARD_CONTRACT_VERSION
    }
    _write_json(
        out / "experiment_contract.json",
        _build_experiment_contract(
            checkpoint_family=checkpoint_family,
            cfg=cfg,
            env_cfg=env_cfg,
            env_config=str(args.env_config),
            train_config=str(args.train_config),
            manifest_hash=manifest["manifest_hash"],
        ),
    )
    trainer = trainer_class(args.env_config, cfg)
    trainer.seed_manifest = deepcopy(manifest)
    support_before = _parameter_snapshot(trainer.actors.support_actor)
    combat_before = _parameter_snapshot(trainer.actors.combat_actor)
    critic_before = _parameter_snapshot(trainer.critic)
    support_critic_before = (
        _parameter_snapshot(trainer.role_critics.support_critic)
        if _is_v14(cfg) and trainer.credit_mode == CREDIT_MODE_ROLE_LOCAL
        else []
    )
    combat_critic_before = (
        _parameter_snapshot(trainer.role_critics.combat_critic)
        if _is_v14(cfg) and trainer.credit_mode == CREDIT_MODE_ROLE_LOCAL
        else []
    )
    try:
        if args.resume:
            trainer.load_checkpoint(args.resume)
        else:
            trainer.next_evaluation_env_steps = int(cfg["training"]["evaluation_interval_env_steps"])
            trainer.next_checkpoint_env_steps = int(cfg["training"]["checkpoint_interval_env_steps"])
            trainer.save_checkpoint(out / "initial.pt", scheduled_env_steps=0)
            initial = _evaluate(trainer, args.env_config, cfg, manifest, out, "initial_selection", "selection", out / "initial.pt", reward_contract_version)
            _update_best_from_latest_evaluation(trainer, out / "initial.pt", 0)
            trainer.save_checkpoint(out / "best.pt", is_best=True, scheduled_env_steps=0)
        total = int(cfg["training"]["total_env_steps"])
        eval_interval = int(cfg["training"]["evaluation_interval_env_steps"])
        checkpoint_interval = int(cfg["training"]["checkpoint_interval_env_steps"])
        next_eval = int(trainer.next_evaluation_env_steps or eval_interval)
        next_checkpoint = int(trainer.next_checkpoint_env_steps or checkpoint_interval)
        while next_eval <= trainer.env_steps: next_eval += eval_interval
        while next_checkpoint <= trainer.env_steps: next_checkpoint += checkpoint_interval
        started = time.perf_counter(); start_steps = trainer.env_steps
        while trainer.env_steps < total:
            milestone = min(total, next_eval, next_checkpoint)
            while trainer.env_steps < milestone:
                trainer.collect_rollout(max_env_steps=milestone - trainer.env_steps)
                metrics = trainer.update()
                elapsed = time.perf_counter() - started
                _append_metrics(out / "training_metrics.csv", metrics, elapsed, start_steps)
                trainer.save_checkpoint(out / "latest.pt", scheduled_env_steps=trainer.env_steps)
                print(_format_update_line(trainer, total, metrics, is_v15=is_v15), flush=True)
            hit_eval = trainer.env_steps == next_eval
            hit_checkpoint = trainer.env_steps == next_checkpoint
            if hit_checkpoint:
                trainer.save_checkpoint(out / f"step_{trainer.env_steps:07d}.pt", scheduled_env_steps=trainer.env_steps)
                next_checkpoint += checkpoint_interval
            if hit_eval:
                checkpoint = out / f"step_{trainer.env_steps:07d}.pt" if hit_checkpoint else out / "latest.pt"
                summary = _evaluate(trainer, args.env_config, cfg, manifest, out, f"evaluation_selection_step_{trainer.env_steps:07d}", "selection", checkpoint, reward_contract_version)
                if _update_best_from_latest_evaluation(
                    trainer, checkpoint, trainer.env_steps
                ):
                    trainer.save_checkpoint(out / "best.pt", is_best=True, scheduled_env_steps=trainer.env_steps)
                next_eval += eval_interval
            trainer.next_evaluation_env_steps = next_eval; trainer.next_checkpoint_env_steps = next_checkpoint
            trainer.save_checkpoint(out / "latest.pt", scheduled_env_steps=trainer.env_steps)
        trainer.save_checkpoint(out / "final.pt", scheduled_env_steps=trainer.env_steps)

        smoke_validation: dict[str, Any] | None = None
        if args.smoke:
            main_actions, main_lp, main_values, _ = trainer._select_actions()
            restored = trainer_class(args.env_config, cfg)
            try:
                restored.load_checkpoint(out / "final.pt")
                restored_actions, restored_lp, restored_values, _ = restored._select_actions()
                hidden_equal = True
                if trainer.hidden is not None:
                    hidden_equal = torch.equal(trainer.hidden.support, restored.hidden.support) and torch.equal(trainer.hidden.combat, restored.hidden.combat)
                numeric_metrics = [float(v) for v in trainer.last_update_metrics.values() if isinstance(v, (int, float, np.number))]
                smoke_validation = {
                    "env_steps": trainer.env_steps, "updates": trainer.update_count,
                    "reward_and_losses_finite": bool(np.isfinite(numeric_metrics).all()),
                    "support_parameters_changed": _parameters_changed(support_before, trainer.actors.support_actor),
                    "combat_shared_parameters_changed": _parameters_changed(combat_before, trainer.actors.combat_actor),
                    "critic_parameters_changed": _parameters_changed(critic_before, trainer.critic),
                    "actor_optimizer_count": len(trainer.actor_optimizers),
                    "combat_optimizer_is_single": trainer.actor_optimizers["combat"] is trainer.combat_optimizer,
                    "recurrent_hidden_activity": trainer.last_update_metrics.get("recurrent_hidden_activity", 0.0),
                    "hidden_reset_zero_count": trainer.last_update_metrics.get("hidden_reset_zero_count", 0),
                    "checkpoint_hidden_equal": hidden_equal,
                    "resume_next_action_equal": bool(np.array_equal(main_actions, restored_actions)),
                    "resume_next_log_prob_equal": bool(np.array_equal(main_lp, restored_lp)),
                    "resume_value_equal": bool(np.array_equal(main_values, restored_values)),
                    "checkpoint_exists": (out / "final.pt").exists(),
                }
                if _is_v14(cfg):
                    smoke_validation.update({
                        "credit_mode": trainer.credit_mode,
                        "team_rewards_finite": bool(np.isfinite(trainer.buffer.team_rewards).all()),
                        "agent_rewards_finite": bool(
                            trainer.credit_mode != CREDIT_MODE_ROLE_LOCAL
                            or np.isfinite(trainer.buffer.agent_rewards).all()
                        ),
                        "agent_advantages_finite": bool(
                            trainer.credit_mode != CREDIT_MODE_ROLE_LOCAL
                            or np.isfinite(trainer.buffer.advantages).all()
                        ),
                        "support_critic_parameters_changed": bool(
                            trainer.credit_mode != CREDIT_MODE_ROLE_LOCAL
                            or _parameters_changed(support_critic_before, trainer.role_critics.support_critic)
                        ),
                        "combat_critic_parameters_changed": bool(
                            trainer.credit_mode != CREDIT_MODE_ROLE_LOCAL
                            or _parameters_changed(combat_critic_before, trainer.role_critics.combat_critic)
                        ),
                        "combat_critic_is_single": bool(
                            trainer.credit_mode != CREDIT_MODE_ROLE_LOCAL
                            or hasattr(trainer.role_critics, "combat_critic")
                        ),
                    })
                    if is_v15:
                        combat_scale = float(
                            trainer.reward_contract["combat_state"]["scale"]
                        )
                        combat_state_min = float(
                            trainer.last_update_metrics[
                                "min_rollout_combat_state_reward"
                            ]
                        )
                        combat_state_max = float(
                            trainer.last_update_metrics[
                                "max_rollout_combat_state_reward"
                            ]
                        )
                        combat_state_lower_bound = (
                            0.0
                            if reward_contract_version == V16_REWARD_CONTRACT_VERSION
                            else -combat_scale
                        )
                        smoke_validation.update({
                            "combat_lock_quality_finite": bool(
                                trainer.last_update_metrics[
                                    "combat_lock_quality_finite"
                                ]
                            ),
                            "combat_state_reward_in_bounds": bool(
                                np.isfinite([combat_state_min, combat_state_max]).all()
                                and combat_state_min >= combat_state_lower_bound - 1e-7
                                and combat_state_max <= combat_scale + 1e-7
                            ),
                            "agent_returns_finite": bool(
                                np.isfinite(trainer.buffer.returns).all()
                            ),
                            "terminal_reward_printed": "reward=" in _format_update_line(
                                trainer, total, trainer.last_update_metrics, is_v15=True
                            ),
                        })
                _write_json(out / "smoke_validation.json", smoke_validation)
            finally:
                restored.close()

        final_entries: list[dict[str, Any]] = []
        for checkpoint_name in ("best", "final"):
            checkpoint = out / f"{checkpoint_name}.pt"
            report = trainer_class(args.env_config, cfg)
            report.seed_manifest = deepcopy(manifest)
            try:
                report.load_checkpoint(checkpoint)
                for split in ("selection", "test"):
                    summary = _evaluate(report, args.env_config, cfg, manifest, out, f"{checkpoint_name}_{split}", split, checkpoint, reward_contract_version)
                    final_entries.append(report.evaluation_history[-1])
            finally:
                report.close()
        trainer.evaluation_history.extend(final_entries)
        trainer.write_summary(out)
        print(json.dumps({
            "completed": True, "env_steps": trainer.env_steps, "updates": trainer.update_count,
            "variant": trainer.experiment_variant, "recurrent": trainer.recurrent,
            "output_dir": str(out), "smoke_validation": smoke_validation,
        }, ensure_ascii=False), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
