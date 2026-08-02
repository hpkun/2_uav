"""Offline audit for the v7 single-dense MAPPO 10M run.

The script is intentionally read-only with respect to the training run
directory.  It writes compact JSON/CSV diagnostics to a separate audit output
directory and does not modify checkpoints, metrics, or evaluation files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_combat.config import load_config
from uav_combat.environment_3v3 import (
    ALL_IDS,
    BLUE_IDS,
    RED_IDS,
    DEATH_ATTACK,
    DEATH_BOUNDARY_ALTITUDE,
    DEATH_BOUNDARY_XY,
    DEATH_COLLISION_CROSS,
    DEATH_COLLISION_FRIENDLY,
    DEATH_NONE,
    Homogeneous3v3AirCombatEnv,
    OBS_DIM,
)
from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.mappo.networks import GaussianActor
from uav_combat.rule_policy_3v3 import make_team_rule_policy_3v3


KEY_STEPS = [100_000, 900_000, 1_000_000, 2_000_000, 5_000_000, 8_000_000, 9_000_000, 9_900_000, 10_000_000]
CAUSE_NAME = {
    DEATH_NONE: "none",
    DEATH_ATTACK: "attack",
    DEATH_BOUNDARY_ALTITUDE: "boundary_altitude",
    DEATH_BOUNDARY_XY: "boundary_xy",
    DEATH_COLLISION_FRIENDLY: "collision_friendly",
    DEATH_COLLISION_CROSS: "collision_cross",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_actor(checkpoint_path: Path) -> tuple[GaussianActor, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = cfg["network"]
    actor = GaussianActor(
        OBS_DIM,
        3,
        int(net["hidden_dim"]),
        float(net["log_std_init"]),
        log_std_min=float(net.get("log_std_min", -5.0)),
        log_std_max=float(net.get("log_std_max", 2.0)),
    )
    actor.load_state_dict(ckpt["shared_red_actor"])
    actor.eval()
    return actor, ckpt


def _action_stats(actor: GaussianActor, obs_bank: np.ndarray) -> dict[str, Any]:
    obs = torch.as_tensor(obs_bank.reshape(-1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        actions = actor.deterministic_action(obs).cpu().numpy()
    saturation = np.abs(actions) >= 0.95
    return {
        "samples": int(actions.shape[0]),
        "mean_action": actions.mean(axis=0).astype(float).tolist(),
        "std_action": actions.std(axis=0).astype(float).tolist(),
        "min_action": actions.min(axis=0).astype(float).tolist(),
        "max_action": actions.max(axis=0).astype(float).tolist(),
        "saturation_fraction_by_dim": saturation.mean(axis=0).astype(float).tolist(),
        "pitch_action_mean": float(actions[:, 1].mean()),
        "pitch_action_positive_fraction": float((actions[:, 1] > 0.0).mean()),
        "pitch_action_negative_fraction": float((actions[:, 1] < 0.0).mean()),
        "any_nonfinite": bool(not np.isfinite(actions).all()),
        "effective_log_std_mean": float(actor.effective_log_std_mean),
        "effective_std_mean": float(actor.effective_std_mean),
    }


def _build_observation_bank(env_config: Path, seeds: list[int]) -> np.ndarray:
    env = Homogeneous3v3AirCombatEnv(env_config)
    obs_rows: list[np.ndarray] = []
    try:
        for seed in seeds:
            observations, _ = env.reset(seed)
            obs_rows.append(np.stack([observations[aid] for aid in RED_IDS], axis=0))
    finally:
        pass
    return np.stack(obs_rows, axis=0).astype(np.float32)


def _team_aircraft(env: Homogeneous3v3AirCombatEnv, team: str):
    ids = RED_IDS if team == "red" else BLUE_IDS
    return [env._aircraft_by_id(aid) for aid in ids]


def _red_actions(actor: GaussianActor, observations: dict[str, np.ndarray], env: Homogeneous3v3AirCombatEnv) -> dict[str, np.ndarray]:
    obs = np.stack([observations[aid] for aid in RED_IDS], axis=0).astype(np.float32)
    with torch.no_grad():
        actions = actor.deterministic_action(torch.as_tensor(obs)).cpu().numpy().astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for i, aid in enumerate(RED_IDS):
        ac = env._aircraft_by_id(aid)
        if ac.state.alive:
            out[aid] = actions[i]
    return out


def _nearest_geometry_row(env: Homogeneous3v3AirCombatEnv, aid: str) -> dict[str, Any]:
    own = env._aircraft_by_id(aid)
    enemies = [a for a in env.aircraft if a.team != own.team and a.state.alive]
    if not own.state.alive or not enemies:
        return {"target_id": None, "distance": None, "ata": None, "aa": None}
    target = min(enemies, key=lambda e: (float(np.linalg.norm(own.state.as_array()[:3] - e.state.as_array()[:3])), e.aircraft_id))
    geo = compute_pairwise_geometry(own.state, target.state)
    return {
        "target_id": target.aircraft_id,
        "distance": float(geo.distance),
        "ata": float(geo.ata),
        "aa": float(geo.aa),
    }


def _trace_checkpoint(
    name: str,
    actor: GaussianActor,
    env_config: Path,
    seeds: list[int],
    max_history: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = load_config(env_config)
    rows: list[dict[str, Any]] = []
    predeath_rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = Homogeneous3v3AirCombatEnv(env_config)
        blue_policy = make_team_rule_policy_3v3(cfg, team="blue")
        observations, _ = env.reset(seed)
        history: dict[str, list[dict[str, Any]]] = {aid: [] for aid in RED_IDS}
        done = False
        while not done:
            actions = _red_actions(actor, observations, env)
            blue_actions, blue_targets = blue_policy.select_actions(_team_aircraft(env, "blue"), _team_aircraft(env, "red"))
            actions.update({aid: act for aid, act in blue_actions.items() if env._aircraft_by_id(aid).state.alive})
            for aid in RED_IDS:
                ac = env._aircraft_by_id(aid)
                if not ac.state.alive:
                    continue
                geom = _nearest_geometry_row(env, aid)
                action = actions.get(aid, np.zeros(3, np.float32))
                history[aid].append({
                    "checkpoint": name,
                    "seed": int(seed),
                    "step": int(env.step_count),
                    "aircraft_id": aid,
                    "altitude": float(ac.state.altitude),
                    "theta": float(ac.state.theta),
                    "speed": float(ac.state.v),
                    "action_yaw": float(action[0]),
                    "action_pitch": float(action[1]),
                    "action_speed": float(action[2]),
                    "action_saturated": bool(np.any(np.abs(action) >= 0.95)),
                    **geom,
                })
                history[aid] = history[aid][-max_history:]
            observations, _, terminated, truncated, info = env.step(actions)
            rc = info.get("reward_components", {})
            for aid, cause in info.get("death_causes", {}).items():
                if aid in RED_IDS and cause == DEATH_BOUNDARY_ALTITUDE:
                    final_state = env._aircraft_by_id(aid).state
                    for h in history[aid]:
                        h = dict(h)
                        h["death_cause"] = CAUSE_NAME.get(cause, str(cause))
                        h["final_altitude"] = float(final_state.altitude)
                        h["upper_altitude_death"] = bool(final_state.altitude > cfg["battlefield"]["altitude_max"])
                        h["lower_altitude_death"] = bool(final_state.altitude < cfg["battlefield"]["altitude_min"])
                        h["red_r3"] = float(rc.get("red_approach_reward", 0.0))
                        h["red_r41"] = float(rc.get("red_attack_advantage_reward", 0.0))
                        h["red_r42"] = float(rc.get("red_threat_penalty", 0.0))
                        h["red_team_total_reward"] = float(rc.get("red_team_total_reward", 0.0))
                        predeath_rows.append(h)
            done = bool(terminated or truncated)
        summary = info["episode_summary"]
        red_causes = summary["red_death_causes"]
        blue_causes = summary["blue_death_causes"]
        rows.append({
            "checkpoint": name,
            "seed": int(seed),
            "episode_length": int(summary["episode_length"]),
            "outcome": summary["environment_outcome"],
            "termination_reason": summary["termination_reason"],
            "red_attack_kills": int(summary["red_attack_kills"]),
            "blue_attack_kills": int(summary["blue_attack_kills"]),
            "red_survivors": int(summary["red_survivors"]),
            "blue_survivors": int(summary["blue_survivors"]),
            "red_boundary_deaths": int(red_causes["boundary_deaths"]),
            "red_boundary_altitude_deaths": int(red_causes["boundary_altitude_deaths"]),
            "red_boundary_xy_deaths": int(red_causes["boundary_xy_deaths"]),
            "red_friendly_collision_deaths": int(red_causes["friendly_collision_deaths"]),
            "red_cross_collision_deaths": int(red_causes["cross_team_collision_deaths"]),
            "blue_boundary_deaths": int(blue_causes["boundary_deaths"]),
            "blue_boundary_altitude_deaths": int(blue_causes["boundary_altitude_deaths"]),
            "blue_boundary_xy_deaths": int(blue_causes["boundary_xy_deaths"]),
            "blue_friendly_collision_deaths": int(blue_causes["friendly_collision_deaths"]),
            "blue_cross_collision_deaths": int(blue_causes["cross_team_collision_deaths"]),
        })
    return rows, predeath_rows


def _selected_evaluations(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    eval_dir = run_dir / "evaluations"
    for label, path in {
        "initial": eval_dir / "evaluation_initial.json",
        "best": eval_dir / "evaluation_best.json",
        "final": eval_dir / "evaluation_final.json",
    }.items():
        if path.exists():
            out[label] = _load_json(path)
    for step in KEY_STEPS:
        path = eval_dir / f"evaluation_step_{step}.json"
        if path.exists():
            out[f"step_{step}"] = _load_json(path)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/mappo_3v3_v7_single_dense_10m_seed42")
    parser.add_argument("--env-config", default="configs/homogeneous_3v3_learnable_v7_paper_segmented.yaml")
    parser.add_argument("--output-dir", default="outputs/audits/mappo_v7_single_dense_10m_failure_audit")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=240000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    env_config = Path(args.env_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluations = _selected_evaluations(run_dir)
    (out_dir / "selected_evaluations.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")

    obs_bank = _build_observation_bank(env_config, [args.seed_start + i for i in range(16)])
    checkpoint_paths = {
        name: run_dir / "checkpoints" / f"{name}.pt"
        for name in ("initial", "best", "latest", "final")
        if (run_dir / "checkpoints" / f"{name}.pt").exists()
    }
    action_bank: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    predeath_rows: list[dict[str, Any]] = []
    trace_seeds = [args.seed_start + 10_000 + i for i in range(args.episodes)]
    for name, path in checkpoint_paths.items():
        actor, ckpt = _load_actor(path)
        action_bank[name] = {
            "checkpoint_file": str(path),
            "env_steps": int(ckpt.get("env_steps", 0)),
            "update_count": int(ckpt.get("update_count", 0)),
            **_action_stats(actor, obs_bank),
        }
        rows, deaths = _trace_checkpoint(name, actor, env_config, trace_seeds)
        trace_rows.extend(rows)
        predeath_rows.extend(deaths)

    (out_dir / "checkpoint_action_bank.json").write_text(json.dumps(action_bank, indent=2), encoding="utf-8")
    _write_csv(out_dir / "checkpoint_trace_summary.csv", trace_rows)
    _write_csv(out_dir / "altitude_death_pre100.csv", predeath_rows)
    summary = {
        "run_dir": str(run_dir),
        "env_config": str(env_config),
        "checkpoints_audited": sorted(checkpoint_paths),
        "observation_bank_shape": list(obs_bank.shape),
        "trace_episodes_per_checkpoint": int(args.episodes),
        "altitude_death_pre100_rows": len(predeath_rows),
        "raw_outputs_modified": False,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
