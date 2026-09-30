"""Record one real decision-boundary combat episode without policy mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.mavuav import (BLUE_IDS, ENTITY_IDS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, TYPE_BY_ID,
                        HeterogeneousMAVUAVAirCombatEnv, load_environment_config)
from tools.combat_visualization import HETERO_COMBAT_TRACE_SCHEMA_VERSION
from tools.replay_policy import ReplayPolicyAdapter, load_replay_actors


def _snapshot(env: HeterogeneousMAVUAVAirCombatEnv) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([env.entities[aid].state.as_array() for aid in ENTITY_IDS]),
            np.asarray([env.entities[aid].state.alive for aid in ENTITY_IDS], dtype=bool))


def _event_rows(info: dict[str, Any], trace_frame: int, time_s: float) -> list[dict[str, Any]]:
    rows = [{"trace_frame": trace_frame, "time_s": time_s, "type": "attack", **event}
            for event in info.get("attack_events", [])]
    for entity in info.get("killed_ids", []):
        rows.append({"trace_frame": trace_frame, "time_s": time_s, "type": "death",
                     "entity": entity, "cause": info["death_causes"][entity]})
    if info.get("red_safe_distance_violation"):
        rows.append({"trace_frame": trace_frame, "time_s": time_s,
                     "type": "red_separation_warning",
                     "minimum_distance": float(info["minimum_friendly_red_distance"])})
    return rows


def record_episode(
    adapter: ReplayPolicyAdapter, checkpoint: Path, output_dir: Path, *, profile: str,
    seed: int, env_config: dict[str, Any], overwrite: bool = False,
    action_mode: str = "deterministic", action_seed: int | None = None,
    collect_dbm_diagnostics: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = HeterogeneousMAVUAVAirCombatEnv(env_config, seed=seed, profile=profile)
    observations, reset_info = env.reset(seed=seed)
    adapter.reset_episode(action_mode=action_mode, action_seed=action_seed)
    states, alive = _snapshot(env)
    frames, alive_frames = [states], [alive]
    actions: list[np.ndarray] = []
    metrics = {name: [] for name in ("team_reward", "team_situation", "event_reward", "terminal_reward",
                                      "minimum_friendly_red_distance", "red_safe_distance_violation")}
    events: list[dict[str, Any]] = []
    dbm_diagnostics: list[list[dict[str, Any]]] = []
    done = False; final_info: dict[str, Any] = {}
    while not done:
        action = adapter.actions(
            observations, active_masks=env.active_masks,
            collect_diagnostics=collect_dbm_diagnostics,
        )
        dbm_diagnostics.append(list(adapter.last_diagnostics))
        observations, rewards, terminated, truncated, final_info = env.step(action)
        actions.append(action); states, alive = _snapshot(env)
        frames.append(states); alive_frames.append(alive)
        metrics["team_reward"].append(float(rewards[RED_IDS[0]]))
        for name in metrics:
            if name != "team_reward": metrics[name].append(final_info[name])
        frame = env.step_count
        events.extend(_event_rows(final_info, frame, frame * env.decision_dt))
        done = bool(terminated or truncated)
        adapter.after_step(final_info["active_masks"], done)
    summary = final_info["episode_summary"]
    arrays = {
        "kinematics": np.asarray(frames, dtype=np.float64),
        "alive": np.asarray(alive_frames, dtype=bool),
        "steps": np.arange(len(frames), dtype=np.int64),
        "time_s": np.arange(len(frames), dtype=np.float64) * env.decision_dt,
        "red_actions": np.asarray(actions, dtype=np.float32),
        "team_reward": np.asarray(metrics["team_reward"], dtype=np.float64),
        "team_situation": np.asarray(metrics["team_situation"], dtype=np.float64),
        "event_reward": np.asarray(metrics["event_reward"], dtype=np.float64),
        "terminal_reward": np.asarray(metrics["terminal_reward"], dtype=np.float64),
        "minimum_friendly_red_distance": np.asarray(metrics["minimum_friendly_red_distance"], dtype=np.float64),
        "red_safe_distance_violation": np.asarray(metrics["red_safe_distance_violation"], dtype=bool),
    }
    if collect_dbm_diagnostics:
        router = np.full((len(actions), 3, 2), np.nan, dtype=np.float64)
        hard = np.full((len(actions), 3), -1, dtype=np.int8)
        divergence = np.full((len(actions), 3), np.nan, dtype=np.float64)
        residual = np.full((len(actions), 3), np.nan, dtype=np.float64)
        for step, records in enumerate(dbm_diagnostics):
            for record in records:
                index = RED_IDS.index(record["agent_id"]) - 1
                if index < 0:
                    continue
                router[step, index] = record["router_probabilities"]
                hard[step, index] = record["hard_mode_proxy"]
                divergence[step, index] = record["expert_divergence"]
                residual[step, index] = record["residual_magnitude"]
        arrays.update({
            "dbm_router_probabilities": router,
            "dbm_hard_mode_proxy": hard,
            "dbm_expert_divergence": divergence,
            "dbm_residual_magnitude": residual,
        })
    np.savez_compressed(output_dir / "episode_trace.npz", **arrays)
    specs = {kind: {key: value for key, value in spec.items()}
             for kind, spec in env.config["aircraft_specs"].items()}
    metadata = {
        "trace_schema_version": HETERO_COMBAT_TRACE_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_sampled_steps": int(adapter.payload.get("sampled_steps", 0)),
        "algorithm": adapter.method_display_name, "actor_variant": adapter.actor_variant,
        "method_variant": adapter.method_variant, "actor_architecture": adapter.actor_architecture,
        "environment_version": adapter.payload["environment_version"],
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "training_profile": adapter.payload.get("environment_profile"), "evaluation_profile": profile,
        "training_seed": int(adapter.payload.get("trainer_config", adapter.payload.get("config", {})).get("seed", -1)),
        "blue_target_strategy": reset_info["blue_target_strategy"], "episode_seed": seed,
        "action_mode": action_mode,
        "action_seed": int(action_seed) if action_mode == "stochastic" and action_seed is not None else None,
        "dbm_execution_diagnostics": bool(collect_dbm_diagnostics),
        "episode_role": "qualitative_visualization_only", "used_for_quantitative_metrics": False,
        "decision_dt": env.decision_dt, "physics_dt": env.physics_dt,
        "max_decision_steps": env.max_decision_steps, "raw_trace_dt": env.decision_dt,
        "visual_interpolation": True, "visual_interpolation_dt": 0.1,
        "visual_interpolation_method": "linear x/y/h/v; shortest-angle theta/psi; left-continuous alive",
        "raw_state_features": ["x", "y", "h", "v", "theta", "psi"],
        "raw_state_units": ["m", "m", "m", "m/s", "rad", "rad"],
        "altitude_semantics": "h is altitude positive up; visual altitude = h",
        "battlefield": env.config["battlefield"], "aircraft_specs": specs,
        "entity_ids": list(ENTITY_IDS), "entity_types": TYPE_BY_ID,
        "entity_teams": {aid: ("red" if aid in RED_IDS else "blue") for aid in ENTITY_IDS},
        "events": events, **summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--dbm-diagnostics", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter = load_replay_actors(checkpoint, args.device)
    config = load_environment_config(args.env_config.expanduser().resolve() if args.env_config else adapter.payload.get("environment_config"))
    result = record_episode(adapter, checkpoint, args.output_dir, profile=args.profile,
                            seed=args.seed, env_config=config, overwrite=args.overwrite,
                            action_mode=args.action_mode,
                            action_seed=args.action_seed if args.action_mode == "stochastic" else None,
                            collect_dbm_diagnostics=args.dbm_diagnostics)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
