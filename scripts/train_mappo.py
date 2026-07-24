"""Train the alternating-freeze competitive MAPPO baseline."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from uav_combat.mappo.trainer import MAPPOTrainer, SCENARIOS, TEAMS, evaluate_competitive_match


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
    if args.smoke:
        config["training"].update(total_env_steps=8192, num_envs=2, rollout_steps=64, alternating_block_env_steps=2048, ppo_epochs=2, minibatch_size=128)
        config["evaluation"]["episodes"] = 12
        config["experiment"]["output_dir"] = "outputs/mappo_smoke"
    for value, section, key in ((args.total_env_steps, "training", "total_env_steps"), (args.num_envs, "training", "num_envs"), (args.seed, "experiment", "seed"), (args.device, "experiment", "device")):
        if value is not None: config[section][key] = value
    return config


def _rate(episodes, predicate):
    return float(np.mean([predicate(episode) for episode in episodes])) if episodes else np.nan


def diagnostic_row(trainer, episodes, metrics, evaluation, active_side, block_index):
    row = {"update": trainer.update_count, "env_steps": trainer.env_steps, "active_side": active_side, "block_index": block_index, **metrics}
    row.update({
        "red_mean_episode_return": float(np.mean([e["returns"][0] for e in episodes])) if episodes else np.nan,
        "blue_mean_episode_return": float(np.mean([e["returns"][1] for e in episodes])) if episodes else np.nan,
        "mean_episode_length": float(np.mean([e["length"] for e in episodes])) if episodes else np.nan,
        "red_win_rate": _rate(episodes, lambda e: e["outcome"] == "red"),
        "blue_win_rate": _rate(episodes, lambda e: e["outcome"] == "blue"),
        "draw_rate": _rate(episodes, lambda e: e["outcome"] == "draw"),
        "boundary_rate": _rate(episodes, lambda e: e["reason"] in {"boundary", "xy_boundary", "altitude_boundary"}),
        "collision_rate": _rate(episodes, lambda e: e["reason"] == "collision"),
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
    for key in ("red_win_rate", "blue_win_rate", "draw_rate", "decisive_rate", "boundary_rate", "collision_rate", "mean_episode_length"):
        row[f"eval_{key}"] = overall[key]
    for team in TEAMS:
        for key, value in overall[f"{team}_funnel"].items(): row[f"eval_{team}_{key}"] = value
    return row


def write_metrics(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_series(rows, output, boundaries):
    x = [row["env_steps"] for row in rows]
    groups = {
        "training_curves.png": (("red_mean_episode_return", "blue_mean_episode_return"), ("red_policy_loss", "blue_policy_loss"), ("red_value_loss", "blue_value_loss")),
        "competitive_outcomes.png": (("red_win_rate", "blue_win_rate", "draw_rate"), ("eval_red_win_rate", "eval_blue_win_rate", "eval_draw_rate")),
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


def score(evaluation):
    overall = evaluation["overall"]
    return (overall["decisive_rate"], -(overall["boundary_rate"] + overall["collision_rate"]), -overall["mean_episode_length"])


def main():
    args = parse_args(); config = load_config(args); training = config["training"]
    seed = config["experiment"]["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    total = int(training["total_env_steps"]); num_envs = int(training["num_envs"]); block_size = int(training["alternating_block_env_steps"])
    if total % num_envs or block_size % num_envs: raise ValueError("total_env_steps and alternating_block_env_steps must be divisible by num_envs")
    trainer = MAPPOTrainer(args.env_config, config)
    if args.resume: trainer.load_checkpoint(args.resume)
    output = Path(config["experiment"]["output_dir"]); checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    initial = checkpoints / "initial.pt"
    if not args.resume and not initial.exists(): trainer.save_checkpoint(initial)
    if trainer.device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    print(f"device={trainer.device} gpu={torch.cuda.get_device_name(0) if trainer.device.type == 'cuda' else None} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    rows = []; start = time.perf_counter(); evaluation = evaluate_competitive_match(trainer.red_actor, trainer.blue_actor, args.env_config, config["evaluation"]["episodes"], trainer.device, seed=seed + 100000)
    best_score = score(evaluation)
    if not (checkpoints / "competitive_best.pt").exists(): trainer.save_checkpoint(checkpoints / "competitive_best.pt")
    block_summaries = []
    while trainer.env_steps < total:
        block_index = trainer.block_index(); active = trainer.active_side(); block_end = min(total, (block_index + 1) * block_size)
        completed = trainer.collect_rollout(block_end - trainer.env_steps); metrics = trainer.update(active)
        if trainer.update_count % training["eval_interval_updates"] == 0 or trainer.env_steps == block_end:
            evaluation = evaluate_competitive_match(trainer.red_actor, trainer.blue_actor, args.env_config, config["evaluation"]["episodes"], trainer.device, seed=seed + 100000)
            candidate = score(evaluation)
            if candidate > best_score: best_score = candidate; trainer.save_checkpoint(checkpoints / "competitive_best.pt")
        row = diagnostic_row(trainer, completed, metrics, evaluation, active, block_index); rows.append(row)
        trainer.save_checkpoint(checkpoints / "latest.pt"); write_metrics(rows, output / "training_metrics.csv")
        print(f"update={trainer.update_count} steps={trainer.env_steps} block={block_index} active={active} red={row['red_win_rate']} blue={row['blue_win_rate']} draw={row['draw_rate']}", flush=True)
        if trainer.env_steps == block_end:
            trainer.save_checkpoint(checkpoints / f"block_{block_index:03d}.pt")
            block_summaries.append({"block_index": block_index, "active_side": active, "end_env_steps": trainer.env_steps})
            if trainer.env_steps < total: trainer.reset_environments()
    trainer.save_checkpoint(checkpoints / "final.pt")
    boundaries = range(block_size, total, block_size); plot_series(rows, output, boundaries)
    summary = {"device": str(trainer.device), "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "actual_environment_steps": trainer.env_steps, "updates": trainer.update_count, "elapsed_seconds": time.perf_counter() - start, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else 0, "blocks": block_summaries, "scenario_counts": trainer.scenario_counts, "tail_rear_counts": trainer.tail_rear_counts, "best_score": best_score}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
