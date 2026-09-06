"""Deterministically evaluate a HAPPO checkpoint without resuming training."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from typing import Any

import torch

from algorithm.happo.evaluation import evaluate_actors, summarize_records
from algorithm.happo.networks import IndependentActors
from algorithm.happo.relational_critic import RelationalCentralizedCritic
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blue-mode", choices=("nearest", "mav_priority", "both"), default="both")
    parser.add_argument("--env-config", type=Path)
    return parser.parse_args()


def _resolved_device(requested: str) -> str:
    return "cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested


def main(expected_critic_variant: str = "mlp") -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = _resolved_device(args.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    if actual != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM):
        raise RuntimeError("incompatible HAPPO checkpoint environment contract")
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    actor_variant = payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla"))
    if actor_variant != "vanilla":
        raise RuntimeError("incompatible actor architecture: vanilla evaluator requires a vanilla checkpoint")
    method_variant = payload.get("method_variant", trainer_config.get("method_variant", "baseline"))
    if method_variant not in ("baseline", "agp", "curriculum", "agp_curriculum"):
        raise RuntimeError(f"unsupported HAPPO method_variant: {method_variant!r}")
    critic_variant = payload.get("critic_variant", trainer_config.get("critic_variant", "mlp"))
    if critic_variant != expected_critic_variant:
        raise RuntimeError(
            f"incompatible critic variant: evaluator requires {expected_critic_variant!r}, "
            f"checkpoint contains {critic_variant!r}"
        )
    critic_architecture = payload.get("critic_architecture")
    if critic_variant == "relational":
        if method_variant != "baseline":
            raise RuntimeError("RC-HAPPO evaluator requires method_variant='baseline'")
        if critic_architecture != RelationalCentralizedCritic.architecture():
            raise RuntimeError("incompatible relational critic architecture metadata")
    actors = IndependentActors(hidden_dim=int(trainer_config["hidden_dim"])).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    if args.env_config:
        env_config: dict[str, Any] = load_environment_config(args.env_config.expanduser().resolve())
    elif "environment_config" in payload:
        env_config = load_environment_config(payload["environment_config"])
    else:
        env_config = load_environment_config(None)
    training_profile = str(payload["environment_profile"])
    modes = ("nearest", "mav_priority") if args.blue_mode == "both" else (args.blue_mode,)
    rows = []
    algorithm = "rc_happo" if critic_variant == "relational" else "happo"
    for mode in modes:
        records = evaluate_actors(
            actors, env_config, args.episodes, mode, args.profile, seed=1000, device=device,
        )
        rows.append({
            "checkpoint": checkpoint.name, "sampled_steps": int(payload.get("sampled_steps", 0)),
            "algorithm": algorithm,
            "method_variant": method_variant,
            "critic_variant": critic_variant,
            "blue_mode": mode, "training_profile": training_profile,
            "evaluation_profile": args.profile, "episodes": args.episodes,
            **summarize_records(records),
        })
    label = checkpoint.stem.removeprefix("checkpoint_")
    csv_path = checkpoint.parent / f"evaluation_{label}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    summary_path = checkpoint.parent / f"evaluation_{label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({
            "algorithm": algorithm, "checkpoint": str(checkpoint), "training_profile": training_profile,
            "evaluation_profile": args.profile, "method_variant": method_variant,
            "critic_variant": critic_variant, "critic_architecture": critic_architecture,
            "device": device, "results": rows,
        }, stream, indent=2, ensure_ascii=False)
    print({"evaluation_csv": str(csv_path), "summary_json": str(summary_path), "results": rows}, flush=True)


if __name__ == "__main__":
    main()
