"""Evaluate a homogeneous 3v3 HAPPO checkpoint against fixed-rule blue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from uav_combat.happo.evaluation_3v3 import evaluate_happo_fixed_blue_3v3
from uav_combat.happo.trainer_3v3 import HAPPO3v3Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-config", default="configs/homogeneous_3v3_learnable_v4.yaml")
    p.add_argument("--train-config", default="configs/happo_3v3_fixed_blue.yaml")
    p.add_argument("--episodes", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--env-workers", type=int)
    p.add_argument("--device")
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.train_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for value, section, key in (
        (args.num_envs, "training", "num_envs"),
        (args.env_workers, "training", "num_env_workers"),
        (args.device, "experiment", "device"),
    ):
        if value is not None:
            cfg[section][key] = value
    trainer = HAPPO3v3Trainer(args.env_config, cfg)
    try:
        trainer.load_checkpoint(args.checkpoint)
        episodes = int(args.episodes or cfg["evaluation"]["episodes"])
        result = evaluate_happo_fixed_blue_3v3(
            trainer.actors,
            args.env_config,
            episodes,
            trainer.num_envs,
            trainer.num_env_workers,
            trainer.device,
            args.seed_start,
        )
    finally:
        trainer.close()
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
