"""Deterministically evaluate a PCTA-HAPPO checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithm.happo.evaluation import evaluate_actors, summarize_records
from algorithm.modules.pcta import PCTAIndependentActors
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-config", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    if actual != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM):
        raise RuntimeError("incompatible HAPPO checkpoint environment contract")
    config = payload.get("trainer_config", payload.get("config", {}))
    if payload.get("actor_variant", config.get("actor_variant")) != "pcta":
        raise RuntimeError("incompatible actor architecture: expected a PCTA checkpoint")
    if payload.get("method_variant", config.get("method_variant", "baseline")) != "baseline":
        raise RuntimeError("PCTA evaluator requires method_variant='baseline'")
    if payload.get("critic_variant", config.get("critic_variant", "mlp")) != "mlp":
        raise RuntimeError("PCTA evaluator requires critic_variant='mlp'")
    architecture = payload.get("actor_architecture")
    expected_keys = {
        "observation_dim", "context_input_dim", "context_dim", "enemy_block_dim",
        "enemy_dim", "enemy_slots", "head_hidden_dim", "action_dim",
    }
    if (
        not isinstance(architecture, dict)
        or not expected_keys <= set(architecture)
        or not set(architecture) <= expected_keys | {"attention_mode"}
    ):
        raise RuntimeError("PCTA checkpoint has incompatible actor architecture metadata")
    if architecture.get("attention_mode", "learned") != "learned":
        raise RuntimeError("Full PCTA evaluator requires attention_mode='learned'")
    if (
        int(architecture["observation_dim"]) != OBS_DIM
        or int(architecture["context_input_dim"]) != 44
        or int(architecture["enemy_block_dim"]) != 14
        or int(architecture["enemy_slots"]) != 4
        or int(architecture["action_dim"]) != 3
    ):
        raise RuntimeError("PCTA checkpoint has incompatible actor architecture dimensions")
    actors = PCTAIndependentActors(
        observation_dim=int(architecture["observation_dim"]), action_dim=int(architecture["action_dim"]),
        context_dim=int(architecture["context_dim"]), enemy_dim=int(architecture["enemy_dim"]),
        hidden_dim=int(architecture["head_hidden_dim"]),
    ).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    env_config = load_environment_config(
        args.env_config.expanduser().resolve() if args.env_config else payload.get("environment_config")
    )
    records = evaluate_actors(actors, env_config, args.episodes, args.profile, seed=1000, device=device)
    row = {
        "checkpoint": checkpoint.name, "sampled_steps": int(payload.get("sampled_steps", 0)),
        "algorithm": "pcta_happo", "actor_variant": "pcta", "method_variant": "baseline",
        "blue_target_strategy": "nearest_red_aircraft",
        "training_profile": payload.get("environment_profile"), "evaluation_profile": args.profile,
        "episodes": args.episodes, **summarize_records(records),
    }
    label = checkpoint.stem.removeprefix("checkpoint_")
    csv_path = checkpoint.parent / f"evaluation_pcta_{label}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    summary_path = checkpoint.parent / f"evaluation_pcta_{label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({
            "algorithm": "pcta_happo", "checkpoint": str(checkpoint),
            "actor_variant": "pcta", "actor_architecture": architecture,
            "pcta_consistency_coef": payload.get("pcta_consistency_coef"),
            "device": device, "results": [row],
        }, stream, indent=2, ensure_ascii=False)
    print({"evaluation_csv": str(csv_path), "summary_json": str(summary_path), "results": [row]}, flush=True)


if __name__ == "__main__":
    main()
