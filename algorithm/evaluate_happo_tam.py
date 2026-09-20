"""Evaluate a TAM-HAPPO checkpoint with recurrent deterministic/stochastic actions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.evaluation import evaluate_recurrent_actors, summarize_records
from algorithm.happo.tam import TAMIndependentActors
from env.mavuav import load_environment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--action-seed", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    device = "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if args.env_config:
        env_config: dict[str, Any] = load_environment_config(args.env_config.expanduser().resolve())
    else:
        env_config = load_environment_config(payload.get("environment_config"))
    validate_checkpoint_contract(payload, env_config)
    config = payload.get("trainer_config", payload.get("config", {}))
    if payload.get("actor_variant") != "tam" or payload.get("critic_variant") != "tam_attention":
        raise RuntimeError("TAM evaluator requires actor_variant='tam' and critic_variant='tam_attention'")
    if payload.get("method_variant") != "baseline":
        raise RuntimeError("TAM evaluator requires baseline HAPPO credit semantics")
    actors = TAMIndependentActors(
        observation_dim=int(payload["actor_architecture"]["observation_dim"]), action_dim=3,
        recurrent_hidden_dim=int(config["tam_actor_gru_hidden_dim"]),
        hidden_layers=tuple(config["tam_actor_hidden_layers"]),
        log_std_init=float(config["actor_log_std_init"]),
        state_memory=bool(config["tam_state_memory"]),
    ).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    deterministic = args.action_mode == "deterministic"
    records = evaluate_recurrent_actors(
        actors, env_config, args.episodes, args.profile, seed=1000, device=device,
        deterministic=deterministic, action_seed=None if deterministic else args.action_seed,
        inactive_mask=bool(config.get("tam_inactive_mask", True)),
    )
    row = {
        "checkpoint": checkpoint.name, "sampled_steps": int(payload.get("sampled_steps", 0)),
        "algorithm": "tam_happo", "actor_variant": "tam", "critic_variant": "tam_attention",
        "method_variant": "baseline", "training_profile": payload["environment_profile"],
        "evaluation_profile": args.profile, "episodes": args.episodes,
        "action_mode": args.action_mode, "action_seed": args.action_seed,
        "blue_target_strategy": "nearest_red_aircraft", **summarize_records(records),
    }
    label = checkpoint.stem.removeprefix("checkpoint_")
    suffix = "" if deterministic else "_stochastic"
    csv_path = checkpoint.parent / f"evaluation_tam_{label}{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    summary_path = checkpoint.parent / f"evaluation_tam_{label}{suffix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({
            "algorithm": "tam_happo", "checkpoint": str(checkpoint), "device": device,
            "action_mode": args.action_mode, "action_seed": args.action_seed,
            "actor_architecture": payload["actor_architecture"],
            "critic_architecture": payload["critic_architecture"], "results": [row],
        }, stream, indent=2, ensure_ascii=False)
    print({"evaluation_csv": str(csv_path), "summary_json": str(summary_path), "results": [row]}, flush=True)


if __name__ == "__main__":
    main()
