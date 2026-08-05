"""Diagnose deterministic HAPPO 4v3 trajectories without changing v9 artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_combat.environment_4v3 import FunctionalHeterogeneous4v3AirCombatEnv
from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.happo.networks import IndependentHAPPOActors
from uav_combat.scenario_4v3 import BLUE_IDS_4V3, RED_COMBAT_IDS_4V3, RED_IDS_4V3


ACTION_NAMES = ("yaw", "pitch", "speed")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _actor_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    # values has shape [time, actor, action_dim].
    out: dict[str, Any] = {}
    for actor_id in range(values.shape[1]):
        for dim, name in enumerate(ACTION_NAMES):
            series = values[:, actor_id, dim]
            out[f"{prefix}_actor_{actor_id}_{name}_mean"] = float(np.mean(series))
            out[f"{prefix}_actor_{actor_id}_{name}_std"] = float(np.std(series))
            out[f"{prefix}_actor_{actor_id}_{name}_min"] = float(np.min(series))
            out[f"{prefix}_actor_{actor_id}_{name}_max"] = float(np.max(series))
            if prefix == "deterministic_action":
                out[f"{prefix}_actor_{actor_id}_{name}_saturation_rate_095"] = float(np.mean(np.abs(series) >= 0.95))
    return out


def _geometry_funnel(env: FunctionalHeterogeneous4v3AirCombatEnv) -> dict[str, int]:
    counts = {
        "red_combat_visible_pair_count": 0,
        "red_combat_distance_gate_pair_count": 0,
        "red_combat_ata_gate_pair_count": 0,
        "red_combat_aa_gate_pair_count": 0,
        "red_combat_attack_window_pair_count": 0,
    }
    d_min = float(env.config["combat"]["attack_distance_min"])
    d_max = float(env.config["combat"]["attack_distance_max"])
    ata_max = float(env.config["combat"]["attack_ata_max"])
    aa_max = float(env.config["combat"]["attack_aa_max"])
    direct = env._direct_visible_ids()
    for cid in RED_COMBAT_IDS_4V3:
        combat = env._by_id(cid)
        if not combat.state.alive:
            continue
        for bid in BLUE_IDS_4V3:
            target = env._by_id(bid)
            if not target.state.alive or bid not in direct[cid]:
                continue
            counts["red_combat_visible_pair_count"] += 1
            pair = compute_pairwise_geometry(combat.state, target.state)
            distance_ok = d_min <= pair.distance <= d_max
            ata_ok = pair.ata <= ata_max
            aa_ok = pair.aa <= aa_max
            counts["red_combat_distance_gate_pair_count"] += int(distance_ok)
            counts["red_combat_ata_gate_pair_count"] += int(ata_ok)
            counts["red_combat_aa_gate_pair_count"] += int(aa_ok)
            counts["red_combat_attack_window_pair_count"] += int(distance_ok and ata_ok and aa_ok)
    return counts


def _run_episode(
    actors: IndependentHAPPOActors,
    env_config: str,
    seed: int,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = FunctionalHeterogeneous4v3AirCombatEnv(env_config)
    observations, _, _ = env.reset(seed)
    rows: list[dict[str, Any]] = []
    raw_actions: list[np.ndarray] = []
    deterministic_actions: list[np.ndarray] = []
    total_reward = 0.0
    try:
        done = False
        while not done:
            red_obs = torch.as_tensor(np.asarray(observations[0:4], dtype=np.float32), device=device)
            raw = np.stack([
                actor.network(red_obs[i, : actor.observation_dim]).detach().cpu().numpy()
                for i, actor in enumerate(actors.actors)
            ])
            action = np.tanh(raw).astype(np.float32)
            funnel = _geometry_funnel(env)
            direct = env._direct_visible_ids()
            effective = env._effective_visible_ids(direct)
            support_visible = sum(int(bid in direct["red_0"]) for bid in BLUE_IDS_4V3)
            combat_direct_visible = sum(
                int(bid in direct[cid])
                for cid in RED_COMBAT_IDS_4V3
                for bid in BLUE_IDS_4V3
                if env._by_id(cid).state.alive and env._by_id(bid).state.alive
            )
            combat_effective_visible = sum(
                int(bid in effective[cid])
                for cid in RED_COMBAT_IDS_4V3
                for bid in BLUE_IDS_4V3
                if env._by_id(cid).state.alive and env._by_id(bid).state.alive
            )
            observations, _, _, reward, done, _, info = env.step({aid: action[i] for i, aid in enumerate(RED_IDS_4V3)})
            components = info.get("reward_components", {})
            row = {
                "checkpoint": checkpoint_name,
                "episode_seed": int(seed),
                "step": int(env.step_count),
                "reward": float(reward),
                "support_visible_targets": int(support_visible),
                "red_combat_direct_visible_pairs": int(combat_direct_visible),
                "red_combat_effective_visible_pairs": int(combat_effective_visible),
                **funnel,
                "red_combat_attack_window": bool(funnel["red_combat_attack_window_pair_count"] > 0),
                "deterministic_action_mean": float(np.mean(action)),
                "deterministic_action_std": float(np.std(action)),
                "deterministic_action_any_saturated_095": bool(np.any(np.abs(action) >= 0.95)),
                "raw_network_mean": float(np.mean(raw)),
                "raw_network_std": float(np.std(raw)),
            }
            row.update({f"reward_component_{key}": float(value) for key, value in components.items()})
            rows.append(row)
            raw_actions.append(raw)
            deterministic_actions.append(action)
            total_reward += float(reward)
        summary = deepcopy(info.get("episode_summary") or {})
        summary.update({
            "checkpoint": checkpoint_name,
            "episode_seed": int(seed),
            "total_reward_from_steps": float(total_reward),
            "trajectory_steps": int(len(rows)),
        })
        raw_np = np.asarray(raw_actions, dtype=np.float64)
        det_np = np.asarray(deterministic_actions, dtype=np.float64)
        summary.update(_actor_stats(raw_np, "raw_network_mean"))
        summary.update(_actor_stats(det_np, "deterministic_action"))
        summary["trajectory_attack_window_step_rate"] = float(
            np.mean([row["red_combat_attack_window"] for row in rows]) if rows else 0.0
        )
        for key in (
            "red_combat_visible_pair_count",
            "red_combat_distance_gate_pair_count",
            "red_combat_ata_gate_pair_count",
            "red_combat_aa_gate_pair_count",
            "red_combat_attack_window_pair_count",
        ):
            summary[f"mean_{key}"] = _mean([float(row[key]) for row in rows])
        return summary, rows
    finally:
        del env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    network = config["network"]
    training = config["training"]
    actors = IndependentHAPPOActors(
        [int(v) for v in training["observation_dims"]],
        [int(v) for v in training["action_dims"]],
        hidden_dim=int(network["hidden_dim"]),
        log_std_init=float(network["log_std_init"]),
        log_std_min=float(network["log_std_min"]),
        log_std_max=float(network["log_std_max"]),
    ).to(device)
    actors.load_state_dict(checkpoint["actors"])
    actors.eval()

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        summary, episode_rows = _run_episode(actors, args.env_config, int(seed), device, Path(args.checkpoint).stem)
        summaries.append(summary)
        rows.extend(episode_rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory_summary.json").write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint),
            "checkpoint_actual_env_steps": checkpoint.get("actual_env_steps", checkpoint.get("env_steps")),
            "seeds": [int(seed) for seed in args.seeds],
            "episodes": summaries,
        }, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "trajectory_steps.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"episodes": len(summaries), "steps": len(rows), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
