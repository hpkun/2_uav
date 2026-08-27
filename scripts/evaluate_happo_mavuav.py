"""Evaluate HAPPO separately against nearest and MAV-priority Blue."""
from __future__ import annotations

import argparse
import torch
from uav_combat.happo.networks import IndependentActors
from uav_combat.happo.evaluation import evaluate_actors, summarize_records
from uav_combat.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM


def main():
    p = argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--episodes", type=int, default=100); p.add_argument("--env-config", default="configs/heterogeneous_mavuav_3v2.yaml"); p.add_argument("--profile", choices=("learnability", "main"), default="main"); args = p.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim")) != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM): raise RuntimeError("incompatible HAPPO checkpoint environment contract")
    if data.get("environment_profile") != args.profile: raise RuntimeError(f"incompatible HAPPO checkpoint environment profile: {data.get('environment_profile')!r} (expected {args.profile!r})")
    actors = IndependentActors(hidden_dim=int(data["config"]["hidden_dim"])); actors.load_state_dict(data["actors"]); actors.eval()
    for mode in ("nearest", "mav_priority"): print(mode, summarize_records(evaluate_actors(actors, args.env_config, args.episodes, mode, args.profile)))


if __name__ == "__main__": main()
