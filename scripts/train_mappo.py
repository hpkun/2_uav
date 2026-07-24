"""Train the alternating-freeze competitive MAPPO v5 baseline."""
import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import MAPPOTrainer, SCENARIOS, TEAMS, evaluate_competitive_match, summarize_competitive_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/homogeneous_1v1.yaml")
    parser.add_argument("--train-config", default="configs/mappo_1v1.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    return parser.parse_args()


def load_config(args):
    with open(args.train_config, encoding="utf-8") as file: config = yaml.safe_load(file)
    if config["training"].get("training_mode") != "alternating_self_play":
        raise ValueError("train_mappo.py requires training_mode=alternating_self_play")
    config["experiment"]["output_dir"] = config["experiment"].get("output_dir", "outputs/mappo_v5")
    config["training"].setdefault("opponent_history_latest_probability", 0.7)
    if args.smoke:
        config["training"].update(total_env_steps=8192, num_envs=2, rollout_steps=64, alternating_block_env_steps=2048, ppo_epochs=2, minibatch_size=128)
        config["evaluation"]["episodes"] = 12
        config["experiment"]["output_dir"] = "outputs/mappo_v5_smoke"
    for value, section, key in ((args.total_env_steps, "training", "total_env_steps"), (args.num_envs, "training", "num_envs"), (args.seed, "experiment", "seed"), (args.device, "experiment", "device")):
        if value is not None: config[section][key] = value
    return config


def _rate(episodes, predicate):
    return float(np.mean([predicate(episode) for episode in episodes])) if episodes else np.nan


def _episode_summary(episodes):
    if not episodes:
        keys = ("red_outcome_win_rate", "blue_outcome_win_rate", "draw_rate", "non_draw_rate", "red_kill_rate", "blue_kill_rate", "combat_decisive_rate", "red_boundary_loss_rate", "blue_boundary_loss_rate", "boundary_rate", "altitude_boundary_rate", "xy_boundary_rate", "collision_rate", "mutual_kill_rate", "max_steps_rate", "mean_episode_length", "red_mean_return", "blue_mean_return")
        return {key: np.nan for key in keys}
    return summarize_competitive_records(episodes)


def diagnostic_row(trainer, episodes, metrics, evaluation, active_side, block_index):
    row = {"update": trainer.update_count, "env_steps": trainer.env_steps, "active_side": active_side, "block_index": block_index, **metrics}
    local = _episode_summary(episodes)
    row.update({
        "red_mean_episode_return": local.get("red_mean_return", np.nan),
        "blue_mean_episode_return": local.get("blue_mean_return", np.nan),
        "mean_episode_length": local.get("mean_episode_length", np.nan),
        "red_outcome_win_rate": local.get("red_outcome_win_rate", np.nan),
        "blue_outcome_win_rate": local.get("blue_outcome_win_rate", np.nan),
        "draw_rate": local.get("draw_rate", np.nan),
        "non_draw_rate": local.get("non_draw_rate", np.nan),
        "red_kill_rate": local.get("red_kill_rate", np.nan),
        "blue_kill_rate": local.get("blue_kill_rate", np.nan),
        "combat_decisive_rate": local.get("combat_decisive_rate", np.nan),
        "red_boundary_loss_rate": local.get("red_boundary_loss_rate", np.nan),
        "blue_boundary_loss_rate": local.get("blue_boundary_loss_rate", np.nan),
        "boundary_rate": local.get("boundary_rate", np.nan),
        "altitude_boundary_rate": local.get("altitude_boundary_rate", np.nan),
        "xy_boundary_rate": local.get("xy_boundary_rate", np.nan),
        "collision_rate": local.get("collision_rate", np.nan),
        "mutual_kill_rate": local.get("mutual_kill_rate", np.nan),
        "max_steps_rate": local.get("max_steps_rate", np.nan),
    })
    for scenario in SCENARIOS: row[f"scenario_{scenario}_count"] = trainer.scenario_counts[scenario]
    for team in TEAMS: row[f"tail_rear_{team}_count"] = trainer.tail_rear_counts[team]
    for team in TEAMS:
        funnels = [e["funnels"][team] for e in episodes]
        for source, target in (("ever_within_4000m", "within_4000_rate"), ("ever_within_attack_distance", "attack_distance_entry_rate"), ("ever_satisfy_ata", "ata_gate_rate"), ("ever_satisfy_aa", "aa_gate_rate"), ("ever_distance_and_ata", "distance_and_ata_entry_rate"), ("ever_distance_and_aa", "distance_and_aa_entry_rate"), ("ever_ata_and_aa", "ata_and_aa_entry_rate"), ("ever_full_attack_envelope", "full_attack_envelope_entry_rate")):
            row[f"{team}_{target}"] = float(np.mean([f[source] for f in funnels])) if funnels else np.nan
        diagnostics = trainer.last_control_diagnostics[team]
        for key in ("yaw_rate", "pitch_rate", "acceleration", "nx", "nz", "phi"):
            row[f"{team}_{key}_saturation_rate"] = float(np.mean([d[f"{key}_saturated"] for d in diagnostics]))
        for label in ("acceleration", "pitch_rate", "yaw_rate"):
            values = np.asarray([d[f"{label}_tracking_absolute_error"] for d in diagnostics])
            row[f"{team}_{label}_tracking_mae"] = float(values.mean())
            row[f"{team}_{label}_tracking_p95"] = float(np.percentile(values, 95))
        for action in ("yaw", "pitch", "speed"):
            values = np.asarray([d[f"action_{action}"] for d in diagnostics])
            row[f"{team}_action_{action}_mean"] = float(values.mean()); row[f"{team}_action_{action}_std"] = float(values.std())
    overall = evaluation["overall"]
    for key in ("red_outcome_win_rate", "blue_outcome_win_rate", "draw_rate", "non_draw_rate", "red_kill_rate", "blue_kill_rate", "combat_decisive_rate", "red_boundary_loss_rate", "blue_boundary_loss_rate", "boundary_rate", "altitude_boundary_rate", "xy_boundary_rate", "collision_rate", "mutual_kill_rate", "max_steps_rate", "mean_episode_length"):
        row[f"eval_{key}"] = overall[key]
    row["eval_bilateral_kill_rate"] = min(overall["red_kill_rate"], overall["blue_kill_rate"])
    row["eval_kill_imbalance"] = abs(overall["red_kill_rate"] - overall["blue_kill_rate"])
    for team in TEAMS:
        for key, value in overall[f"{team}_funnel"].items(): row[f"eval_{team}_{key}"] = value
    return row


def write_metrics(rows, path):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def plot_series(rows, output, boundaries):
    x = [row["env_steps"] for row in rows]
    groups = {
        "training_curves.png": (("red_mean_episode_return", "blue_mean_episode_return"), ("red_policy_loss", "blue_policy_loss"), ("red_value_loss", "blue_value_loss")),
        "competitive_outcomes.png": (("eval_red_kill_rate", "eval_blue_kill_rate", "eval_combat_decisive_rate"), ("eval_red_boundary_loss_rate", "eval_blue_boundary_loss_rate", "eval_boundary_rate"), ("eval_red_outcome_win_rate", "eval_blue_outcome_win_rate", "eval_draw_rate")),
        "attack_funnel.png": tuple(tuple(f"{team}_{key}" for key in ("attack_distance_entry_rate", "ata_gate_rate", "aa_gate_rate", "full_attack_envelope_entry_rate")) for team in TEAMS),
        "control_saturation.png": tuple(tuple(f"{team}_{key}_saturation_rate" for key in ("yaw_rate", "pitch_rate", "acceleration", "nx", "nz", "phi")) for team in TEAMS),
        "control_tracking_error.png": tuple(tuple(f"{team}_{key}_tracking_mae" for key in ("acceleration", "pitch_rate", "yaw_rate")) for team in TEAMS),
    }
    for filename, panels in groups.items():
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.5 * len(panels)), squeeze=False)
        for axis, keys in zip(axes[:, 0], panels):
            for key in keys: axis.plot(x, [row.get(key, np.nan) for row in rows], label=key)
            for boundary in boundaries: axis.axvline(boundary, color="k", linestyle="--", alpha=.45)
            axis.grid(True); axis.legend(fontsize=7); axis.set_xlabel("environment steps")
        fig.tight_layout(); fig.savefig(output / filename, dpi=150); plt.close(fig)


def plot_opponent_history(blocks, output):
    if not blocks:
        return
    fig, axis = plt.subplots(figsize=(11, 4))
    x = [row["block_index"] for row in blocks]
    y = [row["opponent_generation"] for row in blocks]
    colors = ["#1f77b4" if row["opponent_is_latest"] else "#ff7f0e" for row in blocks]
    markers = ["o" if row["active_side"] == "red" else "s" for row in blocks]
    for xi, yi, color, marker, row in zip(x, y, colors, markers, blocks):
        axis.scatter([xi], [yi], color=color, marker=marker, s=70, label=f"{row['active_side']}_{'latest' if row['opponent_is_latest'] else 'old'}")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), fontsize=8)
    axis.set_xlabel("block index"); axis.set_ylabel("selected opponent generation"); axis.grid(True)
    fig.tight_layout(); fig.savefig(output / "opponent_history.png", dpi=150); plt.close(fig)


def score(evaluation):
    overall = evaluation["overall"]
    return (overall["combat_decisive_rate"], -(overall["boundary_rate"] + overall["collision_rate"]), -overall["mean_episode_length"])


def _load_checkpoint_actors(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != 5:
        raise RuntimeError("v5 checkpoint required for v5 evaluation")
    n = checkpoint["config"]["network"]
    red = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(device)
    blue = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(device)
    red.load_state_dict(checkpoint["red_actor"]); blue.load_state_dict(checkpoint["blue_actor"])
    return red, blue, checkpoint


def evaluate_block_checkpoints(checkpoints, env_config, episodes, device, output):
    rows = []
    for path in sorted(checkpoints.glob("block_*.pt")):
        red, blue, checkpoint = _load_checkpoint_actors(path, device)
        result = evaluate_competitive_match(red, blue, env_config, episodes, device, seed=checkpoint["config"]["experiment"]["seed"] + 300000)
        overall = result["overall"]
        block = int(path.stem.split("_")[1])
        block_meta = next((row for row in checkpoint.get("block_history", []) if row["block_index"] == block), {})
        rows.append({
            "block": block,
            "active_side": block_meta.get("active_side"),
            "opponent_generation": block_meta.get("opponent_generation"),
            "red_kill_rate": overall["red_kill_rate"],
            "blue_kill_rate": overall["blue_kill_rate"],
            "combat_decisive_rate": overall["combat_decisive_rate"],
            "boundary_rate": overall["boundary_rate"],
            "collision_rate": overall["collision_rate"],
            "mean_episode_length": overall["mean_episode_length"],
        })
    if rows:
        write_metrics(rows, output / "block_evaluation_summary.csv")
    return rows


def _state_bytes(value):
    stream = io.BytesIO(); torch.save(value, stream); return stream.getvalue()


def _finite_smoke_checks(row):
    active = row["active_side"]
    suffixes = ("policy_loss", "value_loss", "approx_kl", "grad_norm", "critic_grad_norm")
    keys = [key for key in row if key.startswith(f"{active}_") and key.endswith(suffixes)]
    bad = [key for key in keys if not np.isfinite(row[key])]
    if bad:
        raise FloatingPointError(f"non-finite smoke metrics: {bad}")


def main():
    args = parse_args(); config = load_config(args); training = config["training"]
    seed = config["experiment"]["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    total = int(training["total_env_steps"]); num_envs = int(training["num_envs"]); block_size = int(training["alternating_block_env_steps"])
    if total % num_envs or block_size % num_envs: raise ValueError("total_env_steps and alternating_block_env_steps must be divisible by num_envs")
    trainer = MAPPOTrainer(args.env_config, config)
    if args.resume: trainer.load_checkpoint(args.resume)
    else: trainer.configure_block_opponent(0, "red")
    output = Path(config["experiment"]["output_dir"]); checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    initial = checkpoints / "initial.pt"
    if not args.resume: trainer.save_checkpoint(initial)
    if trainer.device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    print(f"device={trainer.device} gpu={torch.cuda.get_device_name(0) if trainer.device.type == 'cuda' else None} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    rows = []; start = time.perf_counter(); evaluation = evaluate_competitive_match(trainer.red_actor, trainer.blue_actor, args.env_config, config["evaluation"]["episodes"], trainer.device, seed=seed + 100000)
    best_score = score(evaluation); best_score_basis = ("combat_decisive_rate", "-(boundary_rate + collision_rate)", "-mean_episode_length")
    if not args.resume: trainer.save_checkpoint(checkpoints / "competitive_best.pt")
    elif not (checkpoints / "competitive_best.pt").exists(): trainer.save_checkpoint(checkpoints / "competitive_best.pt")
    block_summaries = []
    while trainer.env_steps < total:
        block_index = trainer.block_index(); active = trainer.active_side(); block_end = min(total, (block_index + 1) * block_size)
        trainer.configure_block_opponent(block_index, active)
        if args.smoke:
            active_actor = trainer.red_actor if active == "red" else trainer.blue_actor
            active_critic = trainer.red_critic if active == "red" else trainer.blue_critic
            frozen = "blue" if active == "red" else "red"
            frozen_bundle = _state_bytes({
                "actor": getattr(trainer, f"{frozen}_actor").state_dict(),
                "critic": getattr(trainer, f"{frozen}_critic").state_dict(),
                "actor_optimizer": getattr(trainer, f"{frozen}_actor_optimizer").state_dict(),
                "critic_optimizer": getattr(trainer, f"{frozen}_critic_optimizer").state_dict(),
                "behavior": getattr(trainer, f"{frozen}_behavior_actor").state_dict(),
            })
            active_actor_before = _state_bytes(active_actor.state_dict())
            active_critic_before = _state_bytes(active_critic.state_dict())
        completed = trainer.collect_rollout(block_end - trainer.env_steps)
        if args.smoke and not np.isfinite(trainer.buffer.rewards).all():
            raise FloatingPointError("non-finite smoke rollout reward")
        metrics = trainer.update(active)
        if args.smoke:
            current_frozen_bundle = _state_bytes({
                "actor": getattr(trainer, f"{frozen}_actor").state_dict(),
                "critic": getattr(trainer, f"{frozen}_critic").state_dict(),
                "actor_optimizer": getattr(trainer, f"{frozen}_actor_optimizer").state_dict(),
                "critic_optimizer": getattr(trainer, f"{frozen}_critic_optimizer").state_dict(),
                "behavior": getattr(trainer, f"{frozen}_behavior_actor").state_dict(),
            })
            if current_frozen_bundle != frozen_bundle:
                raise AssertionError("frozen actor, critic, optimizer, or behavior actor changed during smoke update")
            if _state_bytes(active_actor.state_dict()) == active_actor_before or _state_bytes(active_critic.state_dict()) == active_critic_before:
                raise AssertionError("active actor and critic must update during smoke")
        finished_block = trainer.env_steps == block_end
        if finished_block:
            metrics.update(trainer.finish_block(active, block_index))
        if trainer.update_count % training["eval_interval_updates"] == 0 or finished_block:
            evaluation = evaluate_competitive_match(trainer.red_actor, trainer.blue_actor, args.env_config, config["evaluation"]["episodes"], trainer.device, seed=seed + 100000)
            candidate = score(evaluation)
            if candidate > best_score: best_score = candidate; trainer.save_checkpoint(checkpoints / "competitive_best.pt")
        row = diagnostic_row(trainer, completed, metrics, evaluation, active, block_index); rows.append(row)
        if args.smoke: _finite_smoke_checks(row)
        trainer.save_checkpoint(checkpoints / "latest.pt"); write_metrics(rows, output / "training_metrics.csv")
        print(f"update={trainer.update_count} steps={trainer.env_steps} block={block_index} active={active} red_kill={row['eval_red_kill_rate']:.3f} blue_kill={row['eval_blue_kill_rate']:.3f} combat={row['eval_combat_decisive_rate']:.3f} boundary={row['eval_boundary_rate']:.3f} opponent={row['opponent_source_side']}:{row['opponent_generation']}", flush=True)
        if finished_block:
            trainer.save_checkpoint(checkpoints / f"block_{block_index:03d}.pt")
            block_summaries.append({**trainer.block_history[-1]})
            if trainer.env_steps < total: trainer.reset_environments()
    final_checkpoint = checkpoints / "final.pt"
    trainer.save_checkpoint(final_checkpoint)
    smoke_restore_ok = None
    if args.smoke:
        restored = MAPPOTrainer(args.env_config, config); restored.load_checkpoint(final_checkpoint)
        smoke_restore_ok = (
            restored.env_steps == trainer.env_steps
            and restored.current_opponent_side == trainer.current_opponent_side
            and restored.current_opponent_generation == trainer.current_opponent_generation
            and len(restored.red_actor_history) == len(trainer.red_actor_history)
            and len(restored.blue_actor_history) == len(trainer.blue_actor_history)
        )
        if not smoke_restore_ok:
            raise AssertionError("v5 smoke checkpoint restoration mismatch")
    boundaries = range(block_size, total, block_size); plot_series(rows, output, boundaries); plot_opponent_history(trainer.block_history, output)
    block_eval_rows = evaluate_block_checkpoints(checkpoints, args.env_config, int(config["evaluation"]["episodes"]), trainer.device, output)
    summary = {"device": str(trainer.device), "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "actual_environment_steps": trainer.env_steps, "updates": trainer.update_count, "elapsed_seconds": time.perf_counter() - start, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else 0, "blocks": block_summaries, "scenario_counts": trainer.scenario_counts, "tail_rear_counts": trainer.tail_rear_counts, "best_score": best_score, "best_score_basis": best_score_basis, "bilateral_kill_rate": min(evaluation["overall"]["red_kill_rate"], evaluation["overall"]["blue_kill_rate"]), "kill_imbalance": abs(evaluation["overall"]["red_kill_rate"] - evaluation["overall"]["blue_kill_rate"]), "red_history_length": len(trainer.red_actor_history), "blue_history_length": len(trainer.blue_actor_history), "history_selection_counts": trainer.history_selection_counts, "opponent_history_latest_probability": trainer.opponent_history_latest_probability, "smoke_v5_restore_ok": smoke_restore_ok, "block_evaluations": block_eval_rows}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
