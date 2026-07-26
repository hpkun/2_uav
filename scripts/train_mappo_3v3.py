"""Train 3v3 red MAPPO vs fixed-rule blue with evaluation, best selection, plots."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch, yaml
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_combat.mappo.trainer_3v3 import (FixedBlue3v3MAPPOTrainer, compute_best_score,
                                            CHECKPOINT_FAMILY, CHECKPOINT_VERSION_3V3)
from uav_combat.mappo.evaluation_3v3 import evaluate_mappo_fixed_blue_3v3

ROLLOUT_REWARD_FIELDS = (
    "mean_rollout_approach_reward",
    "mean_rollout_attack_advantage_reward",
    "mean_rollout_threat_penalty",
    "mean_rollout_soft_boundary_penalty",
    "mean_rollout_friendly_separation_penalty",
    "mean_rollout_head_on_risk_penalty",
    "mean_rollout_dense_reward",
    "mean_rollout_event_reward",
    "mean_rollout_terminal_reward",
    "mean_rollout_total_step_reward",
    "mean_rollout_tactical_reward",
    "mean_rollout_safety_penalty",
    "mean_rollout_event_terminal_reward",
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--train-config", default="configs/mappo_3v3_fixed_blue.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--total-env-steps", type=int); p.add_argument("--num-envs", type=int)
    p.add_argument("--env-workers", type=int, default=None)
    p.add_argument("--seed", type=int); p.add_argument("--device")
    p.add_argument("--output-dir"); p.add_argument("--resume")
    return p.parse_args()

def load_config(args):
    with open(args.train_config, encoding="utf-8") as f: config = yaml.safe_load(f)
    if config["training"].get("training_mode") != "fixed_rule_blue_3v3":
        raise ValueError("requires fixed_rule_blue_3v3")
    config["experiment"]["output_dir"] = config["experiment"].get("output_dir", "outputs/mappo_3v3_fixed_blue")
    config["training"].setdefault("num_env_workers", 4)
    config["training"].setdefault("evaluation_interval_env_steps", 100000)
    config["training"].setdefault("quick_evaluation_episodes", 60)
    if args.smoke:
        config["training"].update(total_env_steps=16384, num_envs=8, rollout_steps=128, ppo_epochs=2, minibatch_size=512,
                                   evaluation_interval_env_steps=8192, quick_evaluation_episodes=12)
        config["experiment"]["output_dir"] = "outputs/mappo_3v3_fixed_blue_smoke_v2"
    for val, sec, key in ((args.total_env_steps,"training","total_env_steps"),(args.num_envs,"training","num_envs"),
                          (args.env_workers,"training","num_env_workers"),(args.seed,"experiment","seed"),
                          (args.device,"experiment","device"),(args.output_dir,"experiment","output_dir")):
        if val is not None: config[sec][key] = val
    return config

def _plot_curves(rows, output, prefix, title, ycols, ylabels):
    if not rows: return
    fig, axes = plt.subplots(len(ycols), 1, figsize=(10, 2.5*len(ycols)), sharex=True)
    if len(ycols) == 1: axes = [axes]
    for ax, col, lbl in zip(axes, ycols, ylabels):
        vals = [r.get(col, np.nan) for r in rows]
        ax.plot(vals, linewidth=0.8)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Update")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(output / f"{prefix}.png", dpi=100); plt.close(fig)

def main():
    args = parse_args(); config = load_config(args)
    tc = config["training"]; seed = config["experiment"]["seed"]
    np.random.seed(seed); torch.manual_seed(seed)
    total, num_envs = int(tc["total_env_steps"]), int(tc["num_envs"])
    num_workers = int(tc.get("num_env_workers", 4))
    eval_interval = int(tc.get("evaluation_interval_env_steps", 100000))
    quick_eps = int(tc.get("quick_evaluation_episodes", 60))
    device_str = config["experiment"]["device"]
    print(f"num_envs={num_envs} workers={num_workers} envs_per_worker={num_envs//num_workers}", flush=True)

    trainer = FixedBlue3v3MAPPOTrainer(args.env_config, config)
    rule_policy_mapping_modes = trainer.vector_env.policy_modes()
    output = Path(config["experiment"]["output_dir"])
    ckpt_dir = output / "checkpoints"; eval_dir = output / "evaluations"
    ckpt_dir.mkdir(parents=True, exist_ok=True); eval_dir.mkdir(parents=True, exist_ok=True)

    # Resume or init
    is_resume = bool(args.resume)
    if is_resume:
        trainer.load_checkpoint(args.resume)
    else:
        if not (ckpt_dir / "initial.pt").exists():
            trainer.save_checkpoint(ckpt_dir / "initial.pt")

    print(f"device={trainer.device}", flush=True)
    start = time.perf_counter(); rows = []; pure_train_start = time.perf_counter()
    eval_accum = trainer.total_evaluation_seconds

    # Initial evaluation (only if not resume or initial eval missing)
    if not is_resume or not (eval_dir / "evaluation_initial.json").exists():
        t_ev = time.perf_counter()
        init_eval = evaluate_mappo_fixed_blue_3v3(trainer.red_actor, args.env_config, quick_eps,
                                                    num_envs, num_workers, trainer.device, seed + 100000)
        eval_accum += time.perf_counter() - t_ev
        (eval_dir / "evaluation_initial.json").write_text(json.dumps(init_eval, indent=2, default=str))
        if trainer.best_score is None:
            trainer.best_score = compute_best_score(init_eval)
            trainer.best_evaluation = init_eval
            trainer.best_checkpoint_name = "initial.pt"
            trainer.save_checkpoint(ckpt_dir / "best.pt")
            (eval_dir / "evaluation_best.json").write_text(json.dumps(init_eval, indent=2, default=str))
    trainer.total_evaluation_seconds = eval_accum

    # Track last milestone evaluated
    last_eval_milestone = (trainer.env_steps // eval_interval) * eval_interval

    def _episode_stats(records):
        if not records: return {}
        n = len(records)
        rk = sum(r["red_attack_kills"] for r in records)
        return {
            "completed_episodes": n,
            "red_complete_elimination_success_rate": sum(1 for r in records if r["red_complete_elimination_success"]) / n,
            "blue_complete_elimination_success_rate": sum(1 for r in records if r["blue_complete_elimination_success"]) / n,
            "environment_red_outcome_rate": sum(1 for r in records if r.get("environment_outcome") == "red") / n,
            "environment_blue_outcome_rate": sum(1 for r in records if r.get("environment_outcome") == "blue") / n,
            "draw_rate": sum(1 for r in records if r.get("environment_outcome") == "draw") / n,
            "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
            "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
            "mean_red_survivors": float(np.mean([r["red_survivors"] for r in records])),
            "mean_blue_survivors": float(np.mean([r["blue_survivors"] for r in records])),
            "mean_red_boundary_deaths": float(np.mean([r["red_boundary_deaths"] for r in records])),
            "mean_red_boundary_altitude_deaths": float(np.mean([r["red_boundary_altitude_deaths"] for r in records])),
            "mean_red_boundary_xy_deaths": float(np.mean([r["red_boundary_xy_deaths"] for r in records])),
            "mean_blue_boundary_altitude_deaths": float(np.mean([r["blue_boundary_altitude_deaths"] for r in records])),
            "mean_blue_boundary_xy_deaths": float(np.mean([r["blue_boundary_xy_deaths"] for r in records])),
            "mean_red_collision_deaths": float(np.mean([
                r["red_friendly_collision_deaths"] + r["red_cross_collision_deaths"] for r in records])),
            "max_steps_rate": sum(1 for r in records if r.get("termination_reason") == "max_steps") / n,
            "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
            "mean_episode_return": float(np.mean([r["episode_return"] for r in records])),
        }

    # Main loop
    all_completed: list[dict[str, Any]] = []
    try:
        while trainer.env_steps < total:
            remaining = total - trainer.env_steps
            completed = trainer.collect_rollout(remaining)
            all_completed.extend(completed)
            metrics = trainer.update()
            elapsed = time.perf_counter() - start
            env_sps = trainer.env_steps / elapsed if elapsed > 0 else 0

            ep_stats = _episode_stats(completed) if completed else {
                "completed_episodes": 0,
                "red_complete_elimination_success_rate": np.nan,
                "blue_complete_elimination_success_rate": np.nan,
                "environment_red_outcome_rate": np.nan,
                "environment_blue_outcome_rate": np.nan,
                "draw_rate": np.nan,
                "mean_red_attack_kills": np.nan, "mean_blue_attack_kills": np.nan,
                "mean_red_survivors": np.nan, "mean_blue_survivors": np.nan,
                "mean_red_boundary_deaths": np.nan, "mean_red_collision_deaths": np.nan,
                "mean_red_boundary_altitude_deaths": np.nan, "mean_red_boundary_xy_deaths": np.nan,
                "mean_blue_boundary_altitude_deaths": np.nan, "mean_blue_boundary_xy_deaths": np.nan,
                "max_steps_rate": np.nan, "mean_episode_length": np.nan, "mean_episode_return": np.nan,
            }

            row = {"update": trainer.update_count, "env_steps": trainer.env_steps,
                   "policy_loss": metrics["policy_loss"], "value_loss": metrics["value_loss"],
                   "entropy": metrics["entropy"], "approx_kl": metrics["approx_kl"],
                   "clip_fraction": metrics["clip_fraction"],
                   "actor_grad_norm": metrics["actor_grad_norm"],
                   "advantage_mean": metrics["advantage_mean"],
                   "advantage_std": metrics["advantage_std"],
                   "alive_actor_sample_fraction": metrics["alive_actor_sample_fraction"],
                   "current_learning_rate": metrics["current_learning_rate"],
                   "current_entropy_coef": metrics["current_entropy_coef"],
                   "effective_log_std_mean": metrics["effective_log_std_mean"],
                   "effective_std_mean": metrics["effective_std_mean"],
                   "actor_epochs_completed": metrics["actor_epochs_completed"],
                   "actor_minibatches_completed": metrics["actor_minibatches_completed"],
                   "kl_early_stop": metrics["kl_early_stop"],
                   "max_minibatch_kl": metrics["max_minibatch_kl"],
                   "env_steps_per_second": env_sps,
                   "environment_step_seconds": trainer._timing["env_step"],
                   "policy_inference_seconds": trainer._timing["policy_inference"],
                   "ppo_update_seconds": trainer._timing["ppo_update"],
                   "evaluation_seconds": trainer.total_evaluation_seconds,
                   **ep_stats}
            for key in ROLLOUT_REWARD_FIELDS:
                row[key] = np.nan
            row.update(trainer.last_rollout_reward_means)
            rows.append(row)

            # Evaluation at milestones
            cur_milestone = (trainer.env_steps // eval_interval) * eval_interval
            if cur_milestone > last_eval_milestone and cur_milestone > 0:
                last_eval_milestone = cur_milestone
                t_ev = time.perf_counter()
                ev = evaluate_mappo_fixed_blue_3v3(trainer.red_actor, args.env_config, quick_eps,
                                                     num_envs, num_workers, trainer.device, seed + 100000)
                ev_elapsed = time.perf_counter() - t_ev
                trainer.total_evaluation_seconds += ev_elapsed
                (eval_dir / f"evaluation_step_{cur_milestone:06d}.json").write_text(json.dumps(ev, indent=2, default=str))
                sc = compute_best_score(ev)
                if trainer.best_score is None or sc > trainer.best_score:
                    trainer.best_score = sc; trainer.best_evaluation = ev
                    trainer.best_checkpoint_name = f"step_{cur_milestone:06d}.pt"
                    trainer.save_checkpoint(ckpt_dir / "best.pt")
                    (eval_dir / "evaluation_best.json").write_text(json.dumps(ev, indent=2, default=str))
                trainer.evaluation_history.append({"env_steps": trainer.env_steps, "score": list(sc), **ev})
                # Keep deterministic evaluation metrics separate from stochastic rollout metrics.
                for r in rows[-1:]:
                    r.update({
                        "eval_episodes": ev.get("episodes", 0),
                        "eval_red_complete_elimination_success_rate": ev.get("red_complete_elimination_success_rate", np.nan),
                        "eval_mean_red_attack_kills": ev.get("mean_red_attack_kills", np.nan),
                        "eval_mean_blue_attack_kills": ev.get("mean_blue_attack_kills", np.nan),
                        "eval_mean_red_survivors": ev.get("mean_red_survivors", np.nan),
                        "eval_mean_blue_survivors": ev.get("mean_blue_survivors", np.nan),
                        "eval_mean_red_boundary_deaths": ev.get("mean_red_boundary_deaths", np.nan),
                        "eval_mean_red_boundary_altitude_deaths": ev.get("mean_red_boundary_altitude_deaths", np.nan),
                        "eval_mean_red_boundary_xy_deaths": ev.get("mean_red_boundary_xy_deaths", np.nan),
                        "eval_draw_rate": ev.get("draw_rate", np.nan),
                        "eval_max_steps_rate": ev.get("max_steps_rate", np.nan),
                    })

            trainer.save_checkpoint(ckpt_dir / "latest.pt")
            tmd = trainer._timing
            print(f"update={trainer.update_count} steps={trainer.env_steps} env_sps={env_sps:.1f} "
                  f"loss={metrics['policy_loss']:.4f} val={metrics['value_loss']:.4f} "
                  f"kl={metrics['approx_kl']:.5f} lr={metrics['current_learning_rate']:.2e} "
                  f"ent_coef={metrics['current_entropy_coef']:.4f} std={metrics['effective_std_mean']:.3f} "
                  f"act_ep={metrics['actor_epochs_completed']} kl_stop={metrics['kl_early_stop']}", flush=True)

            # Write metrics CSV
            if rows:
                keys = list(dict.fromkeys(k for row in rows for k in row))
                with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    finally:
        trainer.close()

    trainer.save_checkpoint(ckpt_dir / "final.pt")
    # Final evaluation
    t_ev = time.perf_counter()
    final_eval = evaluate_mappo_fixed_blue_3v3(trainer.red_actor, args.env_config, quick_eps,
                                                 num_envs, num_workers, trainer.device, seed + 100000)
    trainer.total_evaluation_seconds += time.perf_counter() - t_ev
    (eval_dir / "evaluation_final.json").write_text(json.dumps(final_eval, indent=2, default=str))
    sc = compute_best_score(final_eval)
    if trainer.best_score is None or sc > trainer.best_score:
        trainer.best_score = sc; trainer.best_evaluation = final_eval
        trainer.best_checkpoint_name = "final.pt"
        trainer.save_checkpoint(ckpt_dir / "best.pt")
        (eval_dir / "evaluation_best.json").write_text(json.dumps(final_eval, indent=2, default=str))

    # Smoke restore
    restored_ok = None
    if args.smoke:
        restored = FixedBlue3v3MAPPOTrainer(args.env_config, config)
        restored.load_checkpoint(ckpt_dir / "final.pt")
        restored.collect_rollout()
        restored_ok = restored.env_steps == trainer.env_steps + restored.rollout_steps * restored.num_envs
        restored.close()
        if not restored_ok: raise AssertionError("smoke continuation failed")

    elapsed_total = time.perf_counter() - start
    tmd = trainer._timing
    summary = {
        "checkpoint_family": CHECKPOINT_FAMILY, "checkpoint_version": CHECKPOINT_VERSION_3V3,
        "device": str(trainer.device), "actual_environment_steps": trainer.env_steps,
        "updates": trainer.update_count, "num_envs": num_envs, "num_env_workers": num_workers,
        "environments_per_worker": num_envs // num_workers,
        "total_training_seconds": elapsed_total,
        "pure_training_seconds": elapsed_total - trainer.total_evaluation_seconds,
        "evaluation_seconds": trainer.total_evaluation_seconds,
        "environment_steps_per_second": trainer.env_steps / elapsed_total if elapsed_total > 0 else 0,
        "learning_rate_initial": trainer.initial_learning_rate,
        "learning_rate_final": trainer.final_learning_rate,
        "entropy_coef_initial": trainer.initial_entropy_coef,
        "entropy_coef_final": trainer.final_entropy_coef,
        "target_kl": trainer.target_kl,
        "log_std_min": trainer.red_actor.log_std_min,
        "log_std_max": trainer.red_actor.log_std_max,
        "kl_early_stop_count": trainer.kl_early_stop_count,
        "rule_policy_mapping_modes": rule_policy_mapping_modes,
        "initial_evaluation": json.loads((eval_dir / "evaluation_initial.json").read_text()) if (eval_dir / "evaluation_initial.json").exists() else None,
        "best_evaluation": trainer.best_evaluation,
        "best_checkpoint": trainer.best_checkpoint_name,
        "final_evaluation": final_eval,
        "best_score": list(trainer.best_score) if trainer.best_score else None,
        "smoke_restore_and_continue_ok": restored_ok,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Plots
    if rows:
        _plot_curves(rows, output, "training_curves", "Training Curves",
                     ["policy_loss", "value_loss", "entropy", "approx_kl", "actor_grad_norm"],
                     ["Policy Loss", "Value Loss", "Entropy", "Approx KL", "Grad Norm"])
        _plot_curves(rows, output, "combat_learning_curves", "Combat Learning",
                     ["red_complete_elimination_success_rate", "mean_red_attack_kills",
                      "mean_blue_attack_kills", "mean_red_survivors", "mean_blue_survivors", "max_steps_rate"],
                     ["Red Success Rate", "Red Attack Kills", "Blue Attack Kills",
                      "Red Survivors", "Blue Survivors", "Max Steps Rate"])
        _plot_curves(rows, output, "safety_curves", "Safety Metrics",
                     ["mean_red_boundary_deaths", "mean_red_collision_deaths"],
                     ["Red Boundary Deaths", "Red Collision Deaths"])
        # evaluation_progress from eval history
        if trainer.evaluation_history:
            milestones = [e["env_steps"] for e in trainer.evaluation_history]
            rcsr = [e.get("red_complete_elimination_success_rate", 0) for e in trainer.evaluation_history]
            bs = [e.get("mean_blue_survivors", 3) for e in trainer.evaluation_history]
            rs = [e.get("mean_red_survivors", 0) for e in trainer.evaluation_history]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(milestones, rcsr, "o-", label="Red Complete Elim Success")
            ax.plot(milestones, bs, "s-", label="Mean Blue Survivors")
            ax.plot(milestones, rs, "^-", label="Mean Red Survivors")
            ax.set_xlabel("Env Steps"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.suptitle("Evaluation Progress"); fig.tight_layout()
            fig.savefig(output / "evaluation_progress.png", dpi=100); plt.close(fig)

    print(json.dumps(summary, indent=2, default=str), flush=True)

if __name__ == "__main__":
    main()
