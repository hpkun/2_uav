"""Plot one deterministic HAPPO episode directly into a flat run folder."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithm.happo.networks import IndependentActors
from env.mavuav import (
    ENTITY_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)


COLORS = {
    "MAV": "#d62728", "UAV1": "#ff7f0e", "UAV2": "#bcbd22", "UAV3": "#e377c2",
    "Blue1": "#1f77b4", "Blue2": "#17becf", "Blue3": "#9467bd", "Blue4": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--blue-mode", choices=("nearest", "mav_priority"), default="nearest")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path, device: torch.device) -> tuple[IndependentActors, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    if actual != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM) or "actors" not in payload:
        raise RuntimeError("incompatible HAPPO checkpoint environment contract")
    config = payload.get("trainer_config", payload.get("config", {}))
    actors = IndependentActors(hidden_dim=int(config.get("hidden_dim", 128))).to(device)
    actors.load_state_dict(payload["actors"])
    return actors.eval(), payload


def _actions(actors: IndependentActors, observations: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    result = []
    with torch.no_grad():
        for index, aircraft_id in enumerate(RED_IDS):
            value, _ = actors.actors[index].sample(
                torch.as_tensor(observations[aircraft_id], device=device).unsqueeze(0), deterministic=True,
            )
            result.append(value.squeeze(0).cpu().numpy())
    return np.asarray(result, dtype=np.float32)


def _snapshot(env: HeterogeneousMAVUAVAirCombatEnv) -> dict[str, tuple[float, float, float, bool]]:
    return {
        aircraft_id: (entity.state.x / 1000, entity.state.y / 1000, entity.state.h / 1000, entity.state.alive)
        for aircraft_id, entity in env.entities.items()
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    actors, payload = _load(checkpoint, device)
    if args.env_config:
        env_config = load_environment_config(args.env_config.expanduser().resolve())
    else:
        env_config = load_environment_config(payload.get("environment_config"))
    env = HeterogeneousMAVUAVAirCombatEnv(
        env_config, seed=args.seed, blue_target_mode=args.blue_mode, profile=args.profile,
    )
    observations, _ = env.reset(seed=args.seed)
    history = [_snapshot(env)]
    killed_at: dict[str, int] = {}
    done = False
    info: dict[str, Any] = {}
    while not done:
        observations, _, terminated, truncated, info = env.step(_actions(actors, observations, device))
        history.append(_snapshot(env))
        for aircraft_id in info.get("killed_ids", []):
            killed_at.setdefault(aircraft_id, env.step_count)
        done = bool(terminated or truncated)

    output = args.output.expanduser().resolve() if args.output else checkpoint.parent / f"trajectory_{args.blue_mode}_seed{args.seed}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    for aircraft_id in ENTITY_IDS:
        end = killed_at.get(aircraft_id, len(history) - 1)
        points = [state[aircraft_id] for state in history[:end + 1]]
        x, y, z = zip(*[(point[0], point[1], point[2]) for point in points])
        axis.plot(x, y, z, "-" if aircraft_id in RED_IDS else "--", color=COLORS[aircraft_id], linewidth=2, label=aircraft_id)
        axis.scatter(x[0], y[0], z[0], marker="o", s=42, color=COLORS[aircraft_id])
        axis.scatter(x[-1], y[-1], z[-1], marker="X" if aircraft_id in killed_at else "^", s=85, color=COLORS[aircraft_id])
        axis.text(x[-1], y[-1], z[-1], f" {aircraft_id}", color=COLORS[aircraft_id], fontsize=8)
    summary = info["episode_summary"]
    axis.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Altitude (km)")
    axis.set_title(
        f"HAPPO deterministic trajectory | Blue={args.blue_mode} | seed={args.seed}\n"
        f"outcome={summary['outcome']}, length={summary['episode_length']}, "
        f"Red kills={summary['red_attack_kills']}, Blue kills={summary['blue_attack_kills']}"
    )
    axis.legend(loc="upper left")
    axis.view_init(elev=24, azim=-55)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "checkpoint": str(checkpoint), "training_profile": payload.get("environment_profile"),
        "evaluation_profile": args.profile, "blue_mode": args.blue_mode, "evaluation_seed": args.seed,
        "output": str(output), **summary,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
