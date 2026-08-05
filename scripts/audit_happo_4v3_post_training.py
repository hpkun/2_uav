"""Diagnose deterministic HAPPO 4v3 trajectories without changing run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_combat.environment_4v3 import (
    FunctionalHeterogeneous4v3AirCombatEnv,
    _angle_score,
    _attack_readiness,
    _f_distance,
)
from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.happo.networks import IndependentHAPPOActors
from uav_combat.scenario_4v3 import BLUE_IDS_4V3, RED_COMBAT_IDS_4V3, RED_IDS_4V3


ACTION_NAMES = ("yaw", "pitch", "speed")
PAIR_FIELDS = (
    "checkpoint", "episode_seed", "transition_step", "combat_id", "target_id",
    "pre_action_target_is_current_effective_target", "pre_action_visibility_source",
    "pre_action_distance", "pre_action_ATA", "pre_action_AA", "pre_action_distance_score",
    "pre_action_angle_score", "pre_action_geometry_readiness", "pre_action_distance_gate",
    "pre_action_ATA_gate", "pre_action_AA_gate", "pre_action_attack_window",
    "action_yaw", "action_pitch", "action_speed", "pre_action_combat_alive",
    "pre_action_target_alive", "transition_reward", "transition_raw_dense_reward",
    "transition_done", "transition_termination_reason",
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
            direct = env._direct_visible_ids()
            effective = env._effective_visible_ids(direct)
            d_min = float(env.config["combat"]["attack_distance_min"])
            d_max = float(env.config["combat"]["attack_distance_max"])
            fade_distance = float(env.reward_contract["combat_dense"]["readiness_fade_distance"])
            pair_rows: list[dict[str, Any]] = []
            # Geometry and gates are measured immediately before the selected action.
            current_targets = {
                cid: env._nearest_effective_target(env._by_id(cid), effective)
                for cid in RED_COMBAT_IDS_4V3
                if env._by_id(cid).state.alive
            }
            for combat_index, cid in enumerate(RED_COMBAT_IDS_4V3, start=1):
                combat = env._by_id(cid)
                if not combat.state.alive:
                    continue
                for bid in BLUE_IDS_4V3:
                    target = env._by_id(bid)
                    if not target.state.alive:
                        continue
                    pair = compute_pairwise_geometry(combat.state, target.state)
                    if bid in direct[cid]:
                        visibility_source = "direct"
                    elif bid in effective[cid]:
                        visibility_source = "shared"
                    else:
                        visibility_source = "hidden"
                    distance_score = _f_distance(pair.distance, d_min, d_max, fade_distance)
                    angle_score = _angle_score(pair.ata, pair.aa)
                    readiness = _attack_readiness(
                        combat.state, target.state, d_min, d_max, fade_distance, env._readiness_mode
                    )
                    distance_gate = bool(d_min <= pair.distance <= d_max)
                    ata_gate = bool(pair.ata <= float(env.config["combat"]["attack_ata_max"]))
                    aa_gate = bool(pair.aa <= float(env.config["combat"]["attack_aa_max"]))
                    pair_rows.append({
                        "checkpoint": checkpoint_name,
                        "episode_seed": int(seed),
                        "transition_step": int(env.step_count + 1),
                        "combat_id": cid,
                        "target_id": bid,
                        "pre_action_target_is_current_effective_target": bool(current_targets[cid] is target),
                        "pre_action_visibility_source": visibility_source,
                        "pre_action_distance": float(pair.distance),
                        "pre_action_ATA": float(pair.ata),
                        "pre_action_AA": float(pair.aa),
                        "pre_action_distance_score": float(distance_score),
                        "pre_action_angle_score": float(angle_score),
                        "pre_action_geometry_readiness": float(readiness),
                        "pre_action_distance_gate": distance_gate,
                        "pre_action_ATA_gate": ata_gate,
                        "pre_action_AA_gate": aa_gate,
                        "pre_action_attack_window": bool(visibility_source == "direct" and distance_gate and ata_gate and aa_gate),
                        "action_yaw": float(action[combat_index, 0]),
                        "action_pitch": float(action[combat_index, 1]),
                        "action_speed": float(action[combat_index, 2]),
                        "pre_action_combat_alive": bool(combat.state.alive),
                        "pre_action_target_alive": bool(target.state.alive),
                    })
            observations, _, _, reward, done, _, info = env.step({aid: action[i] for i, aid in enumerate(RED_IDS_4V3)})
            components = info.get("reward_components", {})
            episode_summary = info.get("episode_summary") or {}
            # Reward/components and termination belong to the transition produced by that action.
            for row in pair_rows:
                row.update({
                    "transition_reward": float(reward),
                    "transition_raw_dense_reward": float(info.get("raw_dense_reward", 0.0)),
                    "transition_done": bool(done),
                    "transition_termination_reason": episode_summary.get("termination_reason") if done else None,
                    "deterministic_action_mean": float(np.mean(action)),
                    "deterministic_action_std": float(np.std(action)),
                    "deterministic_action_any_saturated_095": bool(np.any(np.abs(action) >= 0.95)),
                    "raw_network_mean": float(np.mean(raw)),
                    "raw_network_std": float(np.std(raw)),
                })
                row.update({f"reward_component_{key}": float(value) for key, value in components.items()})
            rows.extend(pair_rows)
            raw_actions.append(raw)
            deterministic_actions.append(action)
            total_reward += float(reward)
        summary = deepcopy(info.get("episode_summary") or {})
        summary.update({
            "checkpoint": checkpoint_name,
            "episode_seed": int(seed),
            "total_reward_from_steps": float(total_reward),
            "trajectory_steps": int(len({int(row["transition_step"]) for row in rows})),
            "trajectory_pair_rows": int(len(rows)),
        })
        raw_np = np.asarray(raw_actions, dtype=np.float64)
        det_np = np.asarray(deterministic_actions, dtype=np.float64)
        summary.update(_actor_stats(raw_np, "raw_network_mean"))
        summary.update(_actor_stats(det_np, "deterministic_action"))
        pair_count = len(rows)
        summary.update({
            "pre_action_pair_current_effective_target_rate": float(np.mean([
                row["pre_action_target_is_current_effective_target"] for row in rows
            ]) if rows else 0.0),
            "pre_action_pair_direct_visibility_rate": float(np.mean([
                row["pre_action_visibility_source"] == "direct" for row in rows
            ]) if rows else 0.0),
            "pre_action_pair_shared_visibility_rate": float(np.mean([
                row["pre_action_visibility_source"] == "shared" for row in rows
            ]) if rows else 0.0),
            "pre_action_pair_hidden_visibility_rate": float(np.mean([
                row["pre_action_visibility_source"] == "hidden" for row in rows
            ]) if rows else 0.0),
            "pre_action_pair_distance_gate_rate": float(np.mean([row["pre_action_distance_gate"] for row in rows]) if rows else 0.0),
            "pre_action_pair_ATA_gate_rate": float(np.mean([row["pre_action_ATA_gate"] for row in rows]) if rows else 0.0),
            "pre_action_pair_AA_gate_rate": float(np.mean([row["pre_action_AA_gate"] for row in rows]) if rows else 0.0),
            "pre_action_pair_attack_window_rate": float(np.mean([row["pre_action_attack_window"] for row in rows]) if rows else 0.0),
            "pre_action_pair_min_distance": float(min((row["pre_action_distance"] for row in rows), default=0.0)),
            "pre_action_pair_min_ATA": float(min((row["pre_action_ATA"] for row in rows), default=0.0)),
            "pre_action_pair_min_AA": float(min((row["pre_action_AA"] for row in rows), default=0.0)),
            "pre_action_pair_max_geometry_readiness": float(max((row["pre_action_geometry_readiness"] for row in rows), default=0.0)),
            "pre_action_pair_count": int(pair_count),
        })
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
    fields = [*PAIR_FIELDS, *sorted({key for row in rows for key in row} - set(PAIR_FIELDS))]
    with (output / "trajectory_steps.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"episodes": len(summaries), "steps": len(rows), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
