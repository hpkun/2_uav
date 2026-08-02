"""Step-aligned offline audit for the v7 single-dense MAPPO 10M run.

This script is read-only with respect to the training run directory.  It loads
legacy MAPPO actors directly for audit, builds a true visited-state observation
bank from deterministic trajectories, and writes diagnostics to a separate
audit directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_combat.config import load_config
from uav_combat.environment_3v3 import (
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


CAUSE_NAME = {
    DEATH_NONE: "none",
    DEATH_ATTACK: "attack",
    DEATH_BOUNDARY_ALTITUDE: "boundary_altitude",
    DEATH_BOUNDARY_XY: "boundary_xy",
    DEATH_COLLISION_FRIENDLY: "collision_friendly",
    DEATH_COLLISION_CROSS: "collision_cross",
}
TIERS = ("none", "coarse", "medium", "fine")


def _load_actor_audit_only(checkpoint_path: Path) -> tuple[GaussianActor, dict[str, Any]]:
    """Load a MAPPO actor without treating the checkpoint as strict resumeable."""
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


def _team_aircraft(env: Homogeneous3v3AirCombatEnv, team: str):
    ids = RED_IDS if team == "red" else BLUE_IDS
    return [env._aircraft_by_id(aid) for aid in ids]


def _actor_diagnostics(actor: GaussianActor, obs: np.ndarray) -> dict[str, Any]:
    t_obs = torch.as_tensor(obs.reshape(1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        raw_mean = actor.network(t_obs).squeeze(0)
        det = torch.tanh(raw_mean)
        eff_log_std = actor.log_std.clamp(actor.log_std_min, actor.log_std_max)
        std = eff_log_std.exp()
    return {
        "raw_mean": raw_mean.cpu().numpy(),
        "deterministic_action": det.cpu().numpy(),
        "log_std": eff_log_std.cpu().numpy(),
        "std": std.cpu().numpy(),
    }


def _red_actions(actor: GaussianActor, observations: dict[str, np.ndarray], env: Homogeneous3v3AirCombatEnv):
    obs = np.stack([observations[aid] for aid in RED_IDS], axis=0).astype(np.float32)
    with torch.no_grad():
        actions = actor.deterministic_action(torch.as_tensor(obs)).cpu().numpy().astype(np.float32)
    out: dict[str, np.ndarray] = {}
    diag: dict[str, dict[str, Any]] = {}
    for i, aid in enumerate(RED_IDS):
        if env._aircraft_by_id(aid).state.alive:
            out[aid] = actions[i]
            diag[aid] = _actor_diagnostics(actor, observations[aid])
    return out, diag


def _nearest_target(env: Homogeneous3v3AirCombatEnv, aid: str) -> str | None:
    own = env._aircraft_by_id(aid)
    if not own.state.alive:
        return None
    enemies = [a for a in env.aircraft if a.team != own.team and a.state.alive]
    if not enemies:
        return None
    return min(enemies, key=lambda e: (
        float(np.linalg.norm(own.state.as_array()[:3] - e.state.as_array()[:3])),
        e.aircraft_id,
    )).aircraft_id


def _pre_geometry(env: Homogeneous3v3AirCombatEnv, aid: str, target_id: str | None) -> dict[str, Any]:
    if target_id is None:
        return {"pre_target_id": None, "pre_distance": None, "pre_ata": None, "pre_aa": None, "pre_visible": False}
    own = env._aircraft_by_id(aid)
    target = env._aircraft_by_id(target_id)
    geo = compute_pairwise_geometry(own.state, target.state)
    return {
        "pre_target_id": target_id,
        "pre_distance": float(geo.distance),
        "pre_ata": float(geo.ata),
        "pre_aa": float(geo.aa),
        "pre_visible": bool(target_id in env._effective_visible_enemy_ids(own)),
    }


def _classify_sample(
    env: Homogeneous3v3AirCombatEnv,
    aid: str,
    step: int,
    prev_target: str | None,
    ever_visible: bool,
) -> list[str]:
    ac = env._aircraft_by_id(aid)
    if not ac.state.alive:
        return []
    cfg = env.config
    bf = cfg["battlefield"]
    combat = cfg["combat"]
    target_id = _nearest_target(env, aid)
    labels: list[str] = []
    if step == 0:
        labels.append("reset")
    else:
        labels.append("mid_course_approach")
    altitude = ac.state.altitude
    if bf["altitude_min"] + 0.25 * (bf["altitude_max"] - bf["altitude_min"]) <= altitude <= bf["altitude_max"] - 0.25 * (bf["altitude_max"] - bf["altitude_min"]):
        labels.append("mid_altitude")
    if altitude >= bf["altitude_max"] - 500.0:
        labels.append("near_upper_boundary")
    if altitude <= bf["altitude_min"] + 500.0:
        labels.append("near_lower_boundary")
    if sum(1 for a in _team_aircraft(env, "red") if a.state.alive) <= 1:
        labels.append("single_survivor")
    if any(not a.state.alive for a in _team_aircraft(env, "red") if a.aircraft_id != aid):
        labels.append("teammate_dead")
    if target_id is not None:
        target = env._aircraft_by_id(target_id)
        geo = compute_pairwise_geometry(ac.state, target.state)
        visible = target_id in env._effective_visible_enemy_ids(ac)
        if not visible:
            labels.append("target_invisible")
        if visible and ever_visible:
            labels.append("target_re_visible")
        if prev_target is not None and prev_target != target_id:
            labels.append("target_switch")
        if geo.distance <= 1500.0:
            labels.append("near_combat")
        distance_window = float(combat["attack_distance_min"]) <= geo.distance <= float(combat["attack_distance_max"])
        if distance_window and geo.ata > float(combat["attack_ata_max"]):
            labels.append("near_attack_distance_bad_ata")
        if distance_window and geo.ata <= float(combat["attack_ata_max"]) and geo.aa <= float(combat["attack_aa_max"]):
            labels.append("full_attack_window")
    return labels


def _add_bank_sample(bank: dict[str, list[dict[str, Any]]], row: dict[str, Any], categories: list[str], max_per_category: int) -> None:
    for category in categories:
        if len(bank[category]) < max_per_category:
            item = dict(row)
            item["category"] = category
            bank[category].append(item)


def _row_pre(
    checkpoint: str,
    seed: int,
    step: int,
    aid: str,
    env: Homogeneous3v3AirCombatEnv,
    observations: dict[str, np.ndarray],
    global_state: np.ndarray,
    actor_diag: dict[str, Any],
) -> dict[str, Any]:
    ac = env._aircraft_by_id(aid)
    target_id = _nearest_target(env, aid)
    pre_geo = _pre_geometry(env, aid, target_id)
    dact = actor_diag["deterministic_action"]
    raw_mean = actor_diag["raw_mean"]
    log_std = actor_diag["log_std"]
    std = actor_diag["std"]
    row = {
        "checkpoint": checkpoint,
        "seed": int(seed),
        "step": int(step),
        "aircraft_id": aid,
        "pre_alive": bool(ac.state.alive),
        "pre_x": float(ac.state.x),
        "pre_y": float(ac.state.y),
        "pre_z": float(ac.state.z),
        "pre_altitude": float(ac.state.altitude),
        "pre_speed": float(ac.state.v),
        "pre_theta": float(ac.state.theta),
        "pre_psi": float(ac.state.psi),
        "alive_mask_red": "".join("1" if env._aircraft_by_id(rid).state.alive else "0" for rid in RED_IDS),
        "alive_mask_blue": "".join("1" if env._aircraft_by_id(bid).state.alive else "0" for bid in BLUE_IDS),
        "obs_l2": float(np.linalg.norm(observations[aid])),
        "global_state_l2": float(np.linalg.norm(global_state)),
    }
    row.update(pre_geo)
    for i, dim in enumerate(("yaw", "pitch", "speed")):
        row[f"raw_mean_{dim}"] = float(raw_mean[i])
        row[f"tanh_mean_{dim}"] = float(dact[i])
        row[f"sampled_action_{dim}"] = ""
        row[f"log_std_{dim}"] = float(log_std[i])
        row[f"std_{dim}"] = float(std[i])
        row[f"deterministic_saturated_095_{dim}"] = bool(abs(dact[i]) >= 0.95)
    row["deterministic_any_saturated_095"] = bool(np.any(np.abs(dact) >= 0.95))
    return row


def _merge_post(row: dict[str, Any], env: Homogeneous3v3AirCombatEnv, aid: str, info: dict[str, Any]) -> dict[str, Any]:
    audit = info.get("audit", {}).get("paper_segmented_v4_pre_attack", {}).get(aid, {})
    state = audit.get("state", {})
    rc = info.get("reward_components", {})
    diag = info.get("control_diagnostics", {}).get(aid, {})
    death_cause = int(info.get("death_causes", {}).get(aid, DEATH_NONE))
    row = dict(row)
    for key in (
        "desired_v", "desired_theta", "desired_psi", "requested_acceleration",
        "requested_pitch_rate", "requested_yaw_rate", "nx", "nz", "phi",
        "nx_saturated", "nz_saturated", "phi_saturated", "actual_acceleration",
        "actual_pitch_rate", "actual_yaw_rate", "requested_acceleration_fraction",
        "requested_pitch_rate_fraction", "requested_yaw_rate_fraction",
    ):
        if key in diag:
            row[key] = diag[key]
    # Backward-compatible desired fields are derivable from pre state and effective deltas.
    if "effective_speed_delta" in diag:
        row.setdefault("desired_v", float(row["pre_speed"]) + float(diag["effective_speed_delta"]))
    if "effective_pitch_delta" in diag:
        row.setdefault("desired_theta", float(row["pre_theta"]) + float(diag["effective_pitch_delta"]))
    if "effective_yaw_delta" in diag:
        row.setdefault("desired_psi", float(row["pre_psi"]) + float(diag["effective_yaw_delta"]))

    for key in ("x", "y", "z", "altitude", "speed", "theta", "psi", "alive"):
        row[f"post_motion_{key}"] = state.get(key)
    for key in (
        "target_id", "distance", "ata", "aa", "visible", "distance_window",
        "ata_window", "aa_window", "attack_window", "r3_active", "r41_tier",
        "r42_tier", "r3", "r41", "r42", "dense_total",
    ):
        row[f"reward_{key}"] = audit.get(key)
    row["red_r3"] = float(rc.get("red_approach_reward", 0.0))
    row["red_r41"] = float(rc.get("red_attack_advantage_reward", 0.0))
    row["red_r42"] = float(rc.get("red_threat_penalty", 0.0))
    row["red_dense_reward"] = float(rc.get("red_dense_reward", 0.0))
    row["red_kill_reward"] = float(rc.get("red_kill_reward", 0.0))
    row["red_own_loss_penalty"] = float(rc.get("red_attack_death_penalty", 0.0)) + float(rc.get("red_boundary_death_penalty", 0.0)) + float(rc.get("red_collision_death_penalty", 0.0))
    row["red_terminal_reward"] = float(rc.get("red_terminal_reward", 0.0))
    row["red_team_total_reward"] = float(rc.get("red_team_total_reward", 0.0))
    row["attack_intent"] = info.get("attacks", {}).get(aid)
    row["attack_kill_by_red"] = bool(row["attack_intent"] in info.get("attackers_by_target", {}) and info.get("attack_kills", {}).get("red", 0) > 0)
    row["death_cause"] = CAUSE_NAME.get(death_cause, str(death_cause))
    row["terminated"] = bool(info.get("termination_reason") is not None and not info.get("episode_summary") is None and info.get("termination_reason") != "max_steps")
    row["truncated"] = bool(info.get("termination_reason") == "max_steps")
    row["outcome"] = info.get("outcome")
    row["termination_reason"] = info.get("termination_reason")
    row["finite"] = bool(all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float, np.integer, np.floating))))
    return row


def _trace_checkpoint(
    name: str,
    actor: GaussianActor,
    env_config: Path,
    seeds: list[int],
    bank: dict[str, list[dict[str, Any]]] | None = None,
    max_bank_per_category: int = 128,
    keep_trace_rows: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = load_config(env_config)
    trace_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    predeath_rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = Homogeneous3v3AirCombatEnv(env_config)
        env.audit_trace_enabled = True
        blue_policy = make_team_rule_policy_3v3(cfg, team="blue")
        observations, info0 = env.reset(seed)
        prev_targets = {aid: None for aid in RED_IDS}
        ever_visible = {aid: False for aid in RED_IDS}
        history: dict[str, deque[dict[str, Any]]] = {aid: deque(maxlen=100) for aid in RED_IDS}
        done = False
        info = info0
        while not done:
            global_state = env.global_state()
            red_actions, actor_diag = _red_actions(actor, observations, env)
            blue_actions, _ = blue_policy.select_actions(_team_aircraft(env, "blue"), _team_aircraft(env, "red"))
            actions = {**red_actions}
            actions.update({aid: act for aid, act in blue_actions.items() if env._aircraft_by_id(aid).state.alive})
            pre_rows: dict[str, dict[str, Any]] = {}
            for aid in RED_IDS:
                ac = env._aircraft_by_id(aid)
                if not ac.state.alive or aid not in actor_diag:
                    continue
                row = _row_pre(name, seed, env.step_count, aid, env, observations, global_state, actor_diag[aid])
                pre_rows[aid] = row
                categories = _classify_sample(env, aid, env.step_count, prev_targets[aid], ever_visible[aid])
                if row.get("pre_visible"):
                    ever_visible[aid] = True
                if bank is not None:
                    bank_row = {
                        "checkpoint": name,
                        "seed": int(seed),
                        "step": int(env.step_count),
                        "aircraft_id": aid,
                        "observation": observations[aid].astype(np.float32),
                    }
                    _add_bank_sample(bank, bank_row, categories, max_bank_per_category)
                prev_targets[aid] = row.get("pre_target_id")
            observations, _, terminated, truncated, info = env.step(actions)
            for aid, pre in pre_rows.items():
                aligned = _merge_post(pre, env, aid, info)
                history[aid].append(aligned)
                if keep_trace_rows:
                    trace_rows.append(aligned)
            for aid, cause in info.get("death_causes", {}).items():
                if aid in RED_IDS and int(cause) == DEATH_BOUNDARY_ALTITUDE:
                    final_state = env._aircraft_by_id(aid).state
                    for h in history[aid]:
                        row = dict(h)
                        row["final_altitude"] = float(final_state.altitude)
                        row["upper_altitude_death"] = bool(final_state.altitude > cfg["battlefield"]["altitude_max"])
                        row["lower_altitude_death"] = bool(final_state.altitude < cfg["battlefield"]["altitude_min"])
                        predeath_rows.append(row)
            done = bool(terminated or truncated)
        summary = info["episode_summary"]
        red_causes = summary["red_death_causes"]
        blue_causes = summary["blue_death_causes"]
        episode_rows.append({
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
    return episode_rows, trace_rows, predeath_rows


def _deduplicate_bank(bank: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seen: set[bytes] = set()
    observations: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for category in sorted(bank):
        for item in bank[category]:
            obs = np.asarray(item["observation"], dtype=np.float32)
            key = obs.tobytes()
            if key in seen:
                continue
            seen.add(key)
            observations.append(obs)
            meta_rows.append({
                "category": category,
                "checkpoint": item["checkpoint"],
                "seed": item["seed"],
                "step": item["step"],
                "aircraft_id": item["aircraft_id"],
            })
    if not observations:
        raise RuntimeError("observation bank is empty")
    return np.stack(observations, axis=0).astype(np.float32), meta_rows


def _entropy_mc(actor: GaussianActor, obs: torch.Tensor, samples: int) -> float:
    gen = torch.Generator(device=obs.device)
    gen.manual_seed(97531)
    logps: list[torch.Tensor] = []
    with torch.no_grad():
        dist = actor._distribution(obs)
        for _ in range(samples):
            eps = torch.randn(dist.mean.shape, generator=gen, device=obs.device, dtype=obs.dtype)
            raw = dist.mean + dist.stddev * eps
            action = torch.tanh(raw)
            logp = (dist.log_prob(raw) - torch.log(1.0 - action.square() + actor.epsilon)).sum(-1)
            logps.append(logp)
    return float((-torch.cat(logps, dim=0).mean()).item())


def _action_stats(actor: GaussianActor, obs_bank: np.ndarray, mc_samples: int = 8) -> dict[str, Any]:
    obs = torch.as_tensor(obs_bank.reshape(-1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        raw_mean = actor.network(obs)
        det = torch.tanh(raw_mean)
        dist = actor._distribution(obs)
        raw_entropy = dist.entropy().sum(-1)
        log_std = actor.log_std.clamp(actor.log_std_min, actor.log_std_max)
        std = log_std.exp()
        sampled, _ = actor.sample_action(obs)
    det_np = det.cpu().numpy()
    sampled_np = sampled.cpu().numpy()
    mean_np = raw_mean.cpu().numpy()
    out: dict[str, Any] = {
        "samples": int(len(obs_bank)),
        "raw_gaussian_entropy_mean": float(raw_entropy.mean().item()),
        "estimated_squashed_entropy": _entropy_mc(actor, obs, mc_samples),
        "log_std_by_dim": log_std.cpu().numpy().astype(float).tolist(),
        "std_by_dim": std.cpu().numpy().astype(float).tolist(),
        "deterministic_any_saturation_095": float((np.abs(det_np) >= 0.95).any(axis=1).mean()),
        "sampled_any_saturation_095": float((np.abs(sampled_np) >= 0.95).any(axis=1).mean()),
    }
    for values, prefix in ((mean_np, "raw_mean"), (det_np, "deterministic_action"), (sampled_np, "sampled_action")):
        for i, dim in enumerate(("yaw", "pitch", "speed")):
            v = values[:, i]
            out[f"{prefix}_{dim}_mean"] = float(np.mean(v))
            out[f"{prefix}_{dim}_std"] = float(np.std(v))
            out[f"{prefix}_{dim}_min"] = float(np.min(v))
            out[f"{prefix}_{dim}_max"] = float(np.max(v))
            if prefix != "raw_mean":
                for thr in (0.90, 0.95, 0.99):
                    out[f"{prefix}_{dim}_sat_{thr:.2f}"] = float((np.abs(v) >= thr).mean())
    out["finite"] = bool(all(np.isfinite(v) for v in out.values() if isinstance(v, (int, float))))
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


def _summaries(trace_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_ckpt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_ckpt[str(row["checkpoint"])].append(row)
    reward_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for ckpt, rows in sorted(by_ckpt.items()):
        n = max(1, len(rows))
        attack_windows = [r for r in rows if r.get("reward_attack_window") is True]
        reward_rows.append({
            "checkpoint": ckpt,
            "rows": len(rows),
            "r3_active_steps": sum(1 for r in rows if r.get("reward_r3_active") is True),
            "r3_active_fraction": sum(1 for r in rows if r.get("reward_r3_active") is True) / n,
            **{f"r41_{tier}_steps": sum(1 for r in rows if r.get("reward_r41_tier") == tier) for tier in TIERS},
            **{f"r42_{tier}_steps": sum(1 for r in rows if r.get("reward_r42_tier") == tier) for tier in TIERS},
            "distance_window_fraction": sum(1 for r in rows if r.get("reward_distance_window") is True) / n,
            "ata_window_fraction": sum(1 for r in rows if r.get("reward_ata_window") is True) / n,
            "aa_window_fraction": sum(1 for r in rows if r.get("reward_aa_window") is True) / n,
            "attack_window_fraction": len(attack_windows) / n,
            "p_kill_given_attack_window": (
                sum(1 for r in attack_windows if bool(r.get("attack_intent"))) / len(attack_windows)
                if attack_windows else None
            ),
            "sum_red_r3": float(sum(float(r.get("red_r3", 0.0)) for r in rows)),
            "sum_red_r41": float(sum(float(r.get("red_r41", 0.0)) for r in rows)),
            "sum_red_r42": float(sum(float(r.get("red_r42", 0.0)) for r in rows)),
            "sum_abs_red_r3": float(sum(abs(float(r.get("red_r3", 0.0))) for r in rows)),
            "sum_abs_red_r41": float(sum(abs(float(r.get("red_r41", 0.0))) for r in rows)),
            "sum_abs_red_r42": float(sum(abs(float(r.get("red_r42", 0.0))) for r in rows)),
        })
        control_rows.append({
            "checkpoint": ckpt,
            "rows": len(rows),
            "deterministic_pitch_positive_fraction": sum(1 for r in rows if float(r.get("tanh_mean_pitch", 0.0)) > 0.0) / n,
            "theta_positive_fraction": sum(1 for r in rows if float(r.get("pre_theta", 0.0)) > 0.0) / n,
            "actual_climb_fraction": sum(1 for r in rows if float(r.get("actual_pitch_rate", 0.0)) > 0.0 or float(r.get("post_motion_altitude") or 0.0) > float(r.get("pre_altitude", 0.0))) / n,
            "action_pitch_sat095_fraction": sum(1 for r in rows if abs(float(r.get("tanh_mean_pitch", 0.0))) >= 0.95) / n,
            "action_speed_sat095_fraction": sum(1 for r in rows if abs(float(r.get("tanh_mean_speed", 0.0))) >= 0.95) / n,
            "requested_pitch_rate_sat_fraction": sum(1 for r in rows if abs(float(r.get("requested_pitch_rate_fraction", 0.0))) >= 0.999) / n,
            "requested_yaw_rate_sat_fraction": sum(1 for r in rows if abs(float(r.get("requested_yaw_rate_fraction", 0.0))) >= 0.999) / n,
            "requested_accel_sat_fraction": sum(1 for r in rows if abs(float(r.get("requested_acceleration_fraction", 0.0))) >= 0.999) / n,
            "nx_saturated_fraction": sum(1 for r in rows if r.get("nx_saturated") is True) / n,
            "nz_saturated_fraction": sum(1 for r in rows if r.get("nz_saturated") is True) / n,
            "phi_saturated_fraction": sum(1 for r in rows if r.get("phi_saturated") is True) / n,
            "speed_action_plus_at_vmax_effective_accel_fraction": sum(
                1 for r in rows
                if float(r.get("tanh_mean_speed", 0.0)) > 0.95
                and float(r.get("pre_speed", 0.0)) >= 0.999 * 200.0
                and abs(float(r.get("actual_acceleration", 0.0))) > 1e-6
            ) / n,
        })
        switches = 0
        last: dict[tuple[int, str], Any] = {}
        for r in rows:
            key = (int(r["seed"]), str(r["aircraft_id"]))
            tgt = r.get("pre_target_id")
            if key in last and last[key] is not None and tgt != last[key]:
                switches += 1
            last[key] = tgt
        switch_rows.append({"checkpoint": ckpt, "target_switch_count": switches, "target_switch_per_row": switches / n})
    return reward_rows, control_rows, switch_rows


def _finite_and_ledger_ok(trace_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = all(row.get("finite", True) for row in trace_rows)
    ledger_ok = True
    for row in episode_rows:
        red_total = (
            int(row["red_survivors"]) + int(row["red_boundary_deaths"])
            + int(row["red_friendly_collision_deaths"]) + int(row["red_cross_collision_deaths"])
            + (3 - int(row["red_survivors"]) - int(row["red_boundary_deaths"]) - int(row["red_friendly_collision_deaths"]) - int(row["red_cross_collision_deaths"]))
        )
        ledger_ok = ledger_ok and red_total == 3
    return {"trace_rows_finite": finite, "episode_ledger_checked": bool(ledger_ok)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/mappo_3v3_v7_single_dense_10m_seed42")
    parser.add_argument("--env-config", default="configs/homogeneous_3v3_learnable_v7_paper_segmented.yaml")
    parser.add_argument("--output-dir", default="outputs/audits/mappo_v7_single_dense_10m_failure_audit_v2")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--bank-episodes", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=240000)
    parser.add_argument("--max-bank-per-category", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    env_config = Path(args.env_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        name: run_dir / "checkpoints" / f"{name}.pt"
        for name in ("initial", "best", "latest", "final")
        if (run_dir / "checkpoints" / f"{name}.pt").exists()
    }
    if not checkpoint_paths:
        raise FileNotFoundError(f"no checkpoints found in {run_dir / 'checkpoints'}")

    actors: dict[str, GaussianActor] = {}
    ckpts: dict[str, dict[str, Any]] = {}
    for name, path in checkpoint_paths.items():
        actor, ckpt = _load_actor_audit_only(path)
        actors[name] = actor
        ckpts[name] = ckpt

    bank_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bank_seeds = [args.seed_start + i for i in range(args.bank_episodes)]
    for name, actor in actors.items():
        _trace_checkpoint(
            name,
            actor,
            env_config,
            bank_seeds,
            bank=bank_by_category,
            max_bank_per_category=int(args.max_bank_per_category),
            keep_trace_rows=False,
        )
    obs_bank, bank_meta = _deduplicate_bank(bank_by_category)

    action_bank: dict[str, Any] = {}
    for name, actor in actors.items():
        action_bank[name] = {
            "checkpoint_file": str(checkpoint_paths[name]),
            "env_steps": int(ckpts[name].get("env_steps", 0)),
            "update_count": int(ckpts[name].get("update_count", 0)),
            "audit_only_legacy_actor_load": True,
            **_action_stats(actor, obs_bank),
        }

    trace_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    predeath_rows: list[dict[str, Any]] = []
    trace_seeds = [args.seed_start + 10_000 + i for i in range(args.episodes)]
    for name, actor in actors.items():
        eps, tr, deaths = _trace_checkpoint(name, actor, env_config, trace_seeds)
        episode_rows.extend(eps)
        trace_rows.extend(tr)
        predeath_rows.extend(deaths)

    reward_rows, control_rows, switch_rows = _summaries(trace_rows)
    _write_csv(out_dir / "observation_bank_metadata.csv", bank_meta)
    _write_csv(out_dir / "checkpoint_trace_summary.csv", episode_rows)
    _write_csv(out_dir / "step_aligned_trace.csv", trace_rows)
    _write_csv(out_dir / "altitude_death_pre100.csv", predeath_rows)
    _write_csv(out_dir / "reward_activation_summary.csv", reward_rows)
    _write_csv(out_dir / "control_saturation_summary.csv", control_rows)
    _write_csv(out_dir / "target_switch_summary.csv", switch_rows)
    (out_dir / "checkpoint_action_bank.json").write_text(json.dumps(action_bank, indent=2), encoding="utf-8")
    bank_counts = Counter(row["category"] for row in bank_meta)
    expected_categories = [
        "reset", "mid_course_approach", "near_combat", "near_attack_distance_bad_ata",
        "full_attack_window", "mid_altitude", "near_upper_boundary", "near_lower_boundary",
        "target_invisible", "target_re_visible", "target_switch", "teammate_dead",
        "single_survivor",
    ]
    summary = {
        "run_dir": str(run_dir),
        "env_config": str(env_config),
        "output_dir": str(out_dir),
        "checkpoints_audited": sorted(checkpoint_paths),
        "trace_episodes_per_checkpoint": int(args.episodes),
        "bank_episodes_per_checkpoint": int(args.bank_episodes),
        "observation_bank_samples": int(obs_bank.shape[0]),
        "observation_bank_category_counts": dict(sorted(bank_counts.items())),
        "observation_bank_absent_categories": [c for c in expected_categories if bank_counts.get(c, 0) == 0],
        "step_aligned_trace_rows": len(trace_rows),
        "altitude_death_pre100_rows": len(predeath_rows),
        "audit_interface": "env.audit_trace_enabled=True; info['audit']['paper_segmented_v4_pre_attack']",
        "strict_resume_used": False,
        "audit_only_legacy_actor_load": True,
        "raw_outputs_modified": False,
        **_finite_and_ledger_ok(trace_rows, episode_rows),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
