"""Evaluate MAPPO separately against nearest and MAV-priority Blue."""
from __future__ import annotations

import argparse
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.evaluation import evaluate_actor, summarize_records


def main():
    p = argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--episodes", type=int, default=100); p.add_argument("--env-config", default="configs/heterogeneous_mavuav_3v2.yaml"); args = p.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False); actor = GaussianActor(hidden_dim=int(data["config"]["hidden_dim"])); actor.load_state_dict(data["actor"]); actor.eval()
    for mode in ("nearest", "mav_priority"): print(mode, summarize_records(evaluate_actor(actor, args.env_config, args.episodes, mode)))


if __name__ == "__main__": main()
