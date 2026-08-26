"""Generate a deterministic 3D trajectory from a trained HAPPO checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from uav_combat.happo.networks import IndependentActors
from uav_combat.mavuav import ENTITY_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from uav_combat.models import Aircraft, AircraftSpec, AircraftState


COLORS = {
    "MAV": "#d62728", "UAV1": "#ff7f0e", "UAV2": "#bcbd22",
    "Blue1": "#1f77b4", "Blue2": "#17becf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blue-mode", choices=("nearest", "mav_priority"), default="mav_priority")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_actors(path: Path, device: torch.device) -> IndependentActors:
    safe_types = [
        np._core.multiarray._reconstruct, np.ndarray, np.dtype,
        *(type(np.dtype(name)) for name in ("bool", "float32", "float64", "int64", "uint32")),
        Aircraft, AircraftSpec, AircraftState,
    ]
    with torch.serialization.safe_globals(safe_types):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("algorithm") != "happo" or "actors" not in payload:
        raise RuntimeError("checkpoint is not a HAPPO calibration checkpoint")
    hidden_dim = int(payload.get("trainer_config", {}).get("hidden_dim", 128))
    actors = IndependentActors(hidden_dim=hidden_dim)
    actors.load_state_dict(payload["actors"])
    return actors.to(device).eval()


def deterministic_actions(actors: IndependentActors, observations: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    actions = []
    with torch.no_grad():
        for index, aircraft_id in enumerate(RED_IDS):
            observation = torch.as_tensor(observations[aircraft_id], device=device).unsqueeze(0)
            action, _ = actors.actors[index].sample(observation, deterministic=True)
            actions.append(action.squeeze(0).cpu().numpy())
    return np.asarray(actions, dtype=np.float32)


def record_states(env: HeterogeneousMAVUAVAirCombatEnv, step: int, killed_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for aircraft_id in ENTITY_IDS:
        entity = env.entities[aircraft_id]
        state = entity.state
        rows.append({
            "step": step, "time_s": step * env.decision_dt, "aircraft": aircraft_id,
            "team": entity.team, "x_km": state.x / 1000.0, "y_km": state.y / 1000.0,
            "altitude_km": state.h / 1000.0, "speed_mps": state.v,
            "alive": bool(state.alive), "killed_this_step": aircraft_id in killed_ids,
            "inactive_cause": entity.inactive_cause or "",
        })
    return rows


def rollout(actors: IndependentActors, blue_mode: str, seed: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = HeterogeneousMAVUAVAirCombatEnv(None, seed=seed, blue_target_mode=blue_mode)
    observations, _ = env.reset(seed=seed)
    rows = record_states(env, 0, set())
    done = False
    final_info: dict[str, Any] = {}
    while not done:
        actions = deterministic_actions(actors, observations, device)
        observations, _, terminated, truncated, final_info = env.step(actions)
        rows.extend(record_states(env, env.step_count, set(final_info["killed_ids"])))
        done = bool(terminated or truncated)
    return rows, dict(final_info["episode_summary"])


def plot_trajectory(rows: list[dict[str, Any]], summary: dict[str, Any], seed: int, blue_mode: str, output: Path) -> None:
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    for aircraft_id in ENTITY_IDS:
        trajectory = [row for row in rows if row["aircraft"] == aircraft_id]
        end = next((index for index, row in enumerate(trajectory) if not row["alive"]), len(trajectory) - 1)
        shown = trajectory[:end + 1]
        x = [row["x_km"] for row in shown]
        y = [row["y_km"] for row in shown]
        z = [row["altitude_km"] for row in shown]
        linestyle = "-" if aircraft_id in RED_IDS else "--"
        axis.plot(x, y, z, linestyle=linestyle, linewidth=2.2, color=COLORS[aircraft_id], label=aircraft_id)
        axis.scatter(x[0], y[0], z[0], marker="o", s=45, color=COLORS[aircraft_id])
        terminal_marker = "X" if not shown[-1]["alive"] else "^"
        axis.scatter(x[-1], y[-1], z[-1], marker=terminal_marker, s=85, color=COLORS[aircraft_id])
        axis.text(x[-1], y[-1], z[-1], f" {aircraft_id}", color=COLORS[aircraft_id], fontsize=9)
    axis.set_xlabel("X (km)")
    axis.set_ylabel("Y (km)")
    axis.set_zlabel("Altitude (km)")
    axis.set_title(
        f"HAPPO 5M deterministic trajectory | Blue={blue_mode} | seed={seed}\n"
        f"outcome={summary['outcome']}, length={summary['episode_length']} s, return={summary['episode_return']:.2f}"
    )
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.35)
    axis.view_init(elev=25, azim=-55)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.seed_count <= 0:
        raise ValueError("seed-count must be positive")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    actors = load_actors(args.checkpoint, device)
    selected: tuple[int, list[dict[str, Any]], dict[str, Any]] | None = None
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        rows, summary = rollout(actors, args.blue_mode, seed, device)
        if summary["outcome"] == "red":
            selected = (seed, rows, summary)
            break
    if selected is None:
        raise RuntimeError("no Red-win episode was found in the requested seed range")
    seed, rows, summary = selected
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"checkpoint": str(args.checkpoint), "evaluation_seed": seed, "blue_mode": args.blue_mode, **summary}
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_trajectory(rows, summary, seed, args.blue_mode, args.output_dir / "trajectory_3d.png")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
