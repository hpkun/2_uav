"""Short, transparent MAPPO training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml
from uav_combat.mappo import MAPPOTrainer


def summarize(records):
    if not records: return {"completed_episodes": 0}
    n = len(records)
    return {"completed_episodes": n, "mean_episode_return": float(np.mean([r["episode_return"] for r in records])), "red_win_rate": sum(r["outcome"] == "red" for r in records) / n, "blue_win_rate": sum(r["outcome"] == "blue" for r in records) / n, "draw_rate": sum(r["outcome"] == "draw" for r in records) / n, "mav_survival_rate": np.mean([r["mav_survived"] for r in records]), "mean_uav_survivors": np.mean([r["red_uav_survivors"] for r in records]), "mean_blue_attack_kills_by_red": np.mean([r["red_attack_kills"] for r in records]), "mean_episode_length": np.mean([r["episode_length"] for r in records])}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--updates", type=int, default=10); p.add_argument("--config", default="configs/mappo_mavuav_3v2.yaml"); p.add_argument("--env-config", default="configs/heterogeneous_mavuav_3v2.yaml"); p.add_argument("--profile", choices=("learnability", "main"), default="main"); p.add_argument("--output", default="outputs/mappo_mavuav.pt"); args = p.parse_args()
    with open(args.config, encoding="utf-8") as f: config = yaml.safe_load(f)
    config["training"]["environment_profile"] = args.profile
    trainer = MAPPOTrainer(args.env_config, config)
    for _ in range(args.updates):
        episodes, losses = trainer.train_update(); print({"sampled_environment_steps": trainer.env_steps, **summarize(episodes), **losses})
    trainer.save(Path(args.output)); trainer.close()


if __name__ == "__main__": main()
