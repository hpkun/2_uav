"""Evaluate a v5 competitive checkpoint in one requested matchup."""
import argparse
import json
from pathlib import Path
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import evaluate_matchup, resolve_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", default="configs/homogeneous_1v1.yaml")
    parser.add_argument("--matchup", required=True, choices=("self_play", "red_vs_zero", "red_vs_pursuit", "blue_vs_zero", "blue_vs_pursuit"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--scenario", choices=("all", "tail_chase", "offset_head_on", "crossing"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seedset", default="seedset0")
    args = parser.parse_args(); device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != 5:
        raise RuntimeError("evaluate_mappo.py requires a v5 checkpoint; v4 and earlier lack historical opponents and revised reward/evaluation semantics")
    config = checkpoint["config"]; n = config["network"]
    red = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(device); blue = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(device)
    red.load_state_dict(checkpoint["red_actor"]); blue.load_state_dict(checkpoint["blue_actor"])
    result = evaluate_matchup(red, blue, args.env_config, args.episodes, device, args.matchup, args.scenario, config["experiment"]["seed"] + 200000)
    output = Path(config["experiment"]["output_dir"]) / f"evaluation_{Path(args.checkpoint).stem}_{args.matchup}_{args.scenario}_{args.seedset}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"saved: {output}")

if __name__ == "__main__": main()
