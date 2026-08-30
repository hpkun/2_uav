"""Deterministically evaluate a HAPPO+HRTA checkpoint and optionally export attention."""
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
from algorithm.modules.hrta import HRTAIndependentActors
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blue-mode", choices=("nearest", "mav_priority", "both"), default="both")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--attention-output", type=Path)
    return parser.parse_args()


def _resolved_device(requested: str) -> str:
    return "cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested


def main() -> None:
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
    variant = payload.get("actor_variant", payload.get("trainer_config", payload.get("config", {})).get("actor_variant", "vanilla"))
    architecture = payload.get("actor_architecture")
    if variant != "hrta" or not isinstance(architecture, dict):
        raise RuntimeError("incompatible actor architecture: expected an HRTA checkpoint")
    actors = HRTAIndependentActors(
        entity_dim=int(architecture["entity_dim"]), role_dim=int(architecture["role_dim"]),
        fusion_hidden_dim=int(architecture["fusion_hidden_dim"]), action_dim=int(architecture["action_dim"]),
    ).to(device)
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
    rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] | None = [] if args.attention_output else None
    for mode in modes:
        records = evaluate_actors(
            actors, env_config, args.episodes, mode, args.profile, seed=1000, device=device,
            attention_records=attention_rows,
        )
        rows.append({
            "checkpoint": checkpoint.name, "sampled_steps": int(payload.get("sampled_steps", 0)),
            "algorithm": "happo_hrta", "blue_mode": mode, "training_profile": training_profile,
            "evaluation_profile": args.profile, "episodes": args.episodes,
            **summarize_records(records),
        })
    label = checkpoint.stem.removeprefix("checkpoint_")
    csv_path = checkpoint.parent / f"evaluation_hrta_{label}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary_path = checkpoint.parent / f"evaluation_hrta_{label}_summary.json"
    summary = {
        "algorithm": "happo_hrta", "checkpoint": str(checkpoint),
        "actor_variant": variant, "actor_architecture": architecture,
        "training_profile": training_profile, "evaluation_profile": args.profile,
        "device": device, "results": rows,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    attention_path = None
    if args.attention_output:
        attention_path = args.attention_output.expanduser().resolve()
        attention_path.parent.mkdir(parents=True, exist_ok=True)
        with attention_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(attention_rows[0].keys()))
            writer.writeheader(); writer.writerows(attention_rows)
    print({
        "algorithm": "happo_hrta", "evaluation_csv": str(csv_path),
        "summary_json": str(summary_path), "attention_csv": str(attention_path) if attention_path else None,
        "results": rows,
    }, flush=True)


if __name__ == "__main__":
    main()
