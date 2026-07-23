"""训练参数共享 MAPPO 环境验证基线。"""
import argparse
import csv
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from uav_combat.mappo.trainer import MAPPOTrainer, evaluate_policy


def parse_args() -> argparse.Namespace:
    """解析训练配置和常用覆盖参数。"""
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


def load_training_config(args: argparse.Namespace) -> dict:
    """读取 YAML，并应用 smoke 与 CLI 覆盖。"""
    with Path(args.train_config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.smoke:
        config["training"].update(total_env_steps=4096, num_envs=2, rollout_steps=64, ppo_epochs=2, minibatch_size=128)
        config["evaluation"]["episodes"] = 6
    for value, section, key in (
        (args.total_env_steps, "training", "total_env_steps"), (args.num_envs, "training", "num_envs"),
        (args.seed, "experiment", "seed"), (args.device, "experiment", "device"),
    ):
        if value is not None:
            config[section][key] = value
    return config


def save_curves(rows: list[dict], path: Path) -> None:
    """保存四面板训练曲线。"""
    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(updates, [row["mean_episode_return"] for row in rows]); axes[0, 0].set_title("Mean episode return")
    axes[0, 1].plot(updates, [row["policy_loss"] for row in rows], label="policy")
    axes[0, 1].plot(updates, [row["value_loss"] for row in rows], label="value"); axes[0, 1].legend(); axes[0, 1].set_title("Losses")
    axes[1, 0].plot(updates, [row["entropy"] for row in rows]); axes[1, 0].set_title("Entropy")
    axes[1, 1].plot(updates, [row["eval_zero_win_rate"] for row in rows]); axes[1, 1].set_title("Evaluation win rate")
    for axis in axes.flat:
        axis.set_xlabel("Update"); axis.grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    """执行初始评估、训练、周期评估、记录和检查点保存。"""
    args = parse_args()
    config = load_training_config(args)
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trainer = MAPPOTrainer(args.env_config, config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoints = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    evaluation_episodes = int(config["evaluation"]["episodes"])
    initial_eval = evaluate_policy(trainer.actor, args.env_config, evaluation_episodes, trainer.device, opponent="zero", side="both", scenario="all", seed=seed + 100000)
    print(f"initial_eval return={initial_eval['mean_return']:.3f} win={initial_eval['win_rate']:.3f} draw={initial_eval['draw_rate']:.3f} length={initial_eval['mean_episode_length']:.1f}")
    trainer.save_checkpoint(checkpoints / "initial.pt")
    trainer.save_checkpoint(checkpoints / "best.pt")
    best_score = (float(initial_eval["win_rate"]), float(initial_eval["mean_return"]))
    latest_eval = initial_eval
    rows: list[dict] = []
    total_steps = int(config["training"]["total_env_steps"])
    while trainer.env_steps < total_steps:
        episodes = trainer.collect_rollout()
        metrics = trainer.update()
        if trainer.update_count % int(config["training"]["eval_interval_updates"]) == 0 or trainer.env_steps >= total_steps:
            latest_eval = evaluate_policy(trainer.actor, args.env_config, evaluation_episodes, trainer.device, opponent="zero", side="both", scenario="all", seed=seed + 100000)
            candidate_score = (float(latest_eval["win_rate"]), float(latest_eval["mean_return"]))
            if candidate_score > best_score:
                best_score = candidate_score
                trainer.save_checkpoint(checkpoints / "best.pt")
        outcomes = [episode["outcome"] for episode in episodes]
        all_returns = [value for episode in episodes for value in episode["returns"]]
        row = {
            "update": trainer.update_count, "env_steps": trainer.env_steps,
            **{key: metrics[key] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance")},
            "mean_episode_return": float(np.mean(all_returns)) if all_returns else 0.0,
            "mean_episode_length": float(np.mean([episode["length"] for episode in episodes])) if episodes else 0.0,
            "red_win_rate": outcomes.count("red") / len(outcomes) if outcomes else 0.0,
            "blue_win_rate": outcomes.count("blue") / len(outcomes) if outcomes else 0.0,
            "draw_rate": outcomes.count("draw") / len(outcomes) if outcomes else 0.0,
            "eval_zero_win_rate": latest_eval["win_rate"], "eval_zero_mean_return": latest_eval["mean_return"],
        }
        rows.append(row)
        trainer.save_checkpoint(checkpoints / "latest.pt")
        if trainer.update_count % int(config["training"]["checkpoint_interval_updates"]) == 0:
            trainer.save_checkpoint(checkpoints / f"update_{trainer.update_count}.pt")
        print(
            f"update={trainer.update_count} steps={trainer.env_steps} return={row['mean_episode_return']:.3f} "
            f"win={row['red_win_rate'] + row['blue_win_rate']:.2f}/draw={row['draw_rate']:.2f} "
            f"policy={row['policy_loss']:.4f} value={row['value_loss']:.4f} entropy={row['entropy']:.3f} "
            f"eval_win={row['eval_zero_win_rate']:.3f}"
        )
    trainer.save_checkpoint(checkpoints / "final.pt")
    fields = list(rows[0].keys())
    with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    save_curves(rows, output_dir / "training_curves.png")
    print(f"final_eval return={latest_eval['mean_return']:.3f} win={latest_eval['win_rate']:.3f} draw={latest_eval['draw_rate']:.3f} length={latest_eval['mean_episode_length']:.1f}")


if __name__ == "__main__":
    main()
