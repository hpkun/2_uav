"""Audit 3v3 environment contract facts without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv
from uav_combat.mappo.vector_env_3v3 import make_combat_vector_env_3v3
from uav_combat.scenario_3v3 import BLUE_IDS, RED_IDS, Homogeneous3v3Scenario


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ROOT / "configs" / "homogeneous_3v3.yaml",
    ROOT / "configs" / "homogeneous_3v3_reward_v3.yaml",
    ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml",
)
DEFAULT_OUTPUT = ROOT / "outputs" / "3v3_environment_contract_audit.json"
AUDIT_VERSION = "3v3_contract_audit_v4"


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _finite_or_none(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def _set_aircraft(env: Homogeneous3v3AirCombatEnv, aid: str, x: float, y: float,
                  altitude: float = 3000.0, v: float = 150.0,
                  theta: float = 0.0, psi: float = 0.0) -> None:
    ac = env._aircraft_by_id(aid)
    ac.state.x = float(x)
    ac.state.y = float(y)
    ac.state.z = -float(altitude)
    ac.state.v = float(v)
    ac.state.theta = float(theta)
    ac.state.psi = float(psi)
    ac.state.alive = True


def _zero_actions(env: Homogeneous3v3AirCombatEnv) -> dict[str, np.ndarray]:
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _nearest_blue_for_red(red, blues):
    return min(blues, key=lambda b: float(np.linalg.norm(b.state.as_array()[:3] - red.state.as_array()[:3])))


def _radial_closing_speed(relative_position: np.ndarray, relative_velocity: np.ndarray) -> float:
    distance = float(np.linalg.norm(relative_position))
    return float(-np.dot(relative_position, relative_velocity) / max(distance, 1e-8))


def _initial_geometry_audit(config_path: Path, seeds: int = 100) -> dict[str, Any]:
    cfg = load_config(config_path)
    scenario = Homogeneous3v3Scenario(cfg)
    distances: list[float] = []
    closing_speeds: list[float] = []
    symmetric = True
    clean = True
    max_radius = 0.0
    pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
    env = Homogeneous3v3AirCombatEnv(config_path)
    bf = cfg["battlefield"]

    for seed in range(seeds):
        aircraft = scenario.reset(seed)
        by_id = {a.aircraft_id: a for a in aircraft}
        reds = [by_id[aid] for aid in RED_IDS]
        blues = [by_id[aid] for aid in BLUE_IDS]
        for rid, bid in pairs:
            r = by_id[rid].state
            b = by_id[bid].state
            symmetric = symmetric and bool(np.allclose([r.x, r.y, r.z], [-b.x, -b.y, b.z], atol=1e-9))
            symmetric = symmetric and bool(np.isclose(r.v, b.v))
            hdg = abs(r.psi - b.psi)
            hdg = min(hdg, 2.0 * np.pi - hdg)
            symmetric = symmetric and bool(np.isclose(hdg, np.pi, atol=1e-9))
        for ac in aircraft:
            max_radius = max(max_radius, float(np.linalg.norm([ac.state.x, ac.state.y])))
            clean = clean and abs(ac.state.x) <= bf["x_limit"] and abs(ac.state.y) <= bf["y_limit"]
            clean = clean and bf["altitude_min"] <= ac.state.altitude <= bf["altitude_max"]
        for i in range(len(aircraft)):
            for j in range(i + 1, len(aircraft)):
                d = float(np.linalg.norm(aircraft[i].state.as_array()[:3] - aircraft[j].state.as_array()[:3]))
                clean = clean and d > bf["collision_distance"]
                clean = clean and not env.attack_model.can_attack(aircraft[i].state, aircraft[j].state)
                clean = clean and not env.attack_model.can_attack(aircraft[j].state, aircraft[i].state)
        for red in reds:
            blue = _nearest_blue_for_red(red, blues)
            rel_pos = blue.state.as_array()[:3] - red.state.as_array()[:3]
            rel_vel = blue.state.velocity_vector() - red.state.velocity_vector()
            d = float(np.linalg.norm(rel_pos))
            closing = _radial_closing_speed(rel_pos, rel_vel)
            distances.append(d)
            closing_speeds.append(closing)

    closing_arr = np.asarray(closing_speeds, dtype=float)
    closing_stats = _stats(closing_speeds)
    closing_stats["positive_fraction"] = float(np.mean(closing_arr > 0.0))
    closing_stats["nonpositive_fraction"] = float(np.mean(closing_arr <= 0.0))
    return {
        "seeds_checked": seeds,
        "red_blue_center_symmetric": bool(symmetric),
        "initial_no_attack_collision_boundary": bool(clean),
        "observed_initial_max_position_radius": max_radius,
        "nearest_enemy_distance_m": _stats(distances),
        "initial_closing_speed_mps": closing_stats,
    }


def _constructed_geometry_checks(config_path: Path) -> dict[str, bool]:
    env = Homogeneous3v3AirCombatEnv(config_path)
    env.reset(0)
    _set_aircraft(env, "red_0", 0, 0, psi=0.0)
    _set_aircraft(env, "blue_0", 500, 0, psi=0.0)
    for i, aid in enumerate(("red_1", "red_2", "blue_1", "blue_2"), start=1):
        _set_aircraft(env, aid, -6000 + i * 1000, 5000 + i * 500, psi=0.5)
    _, _, _, _, tail_info = env.step(_zero_actions(env))

    env = Homogeneous3v3AirCombatEnv(config_path)
    env.reset(0)
    _set_aircraft(env, "red_0", 0, 0, psi=0.0)
    _set_aircraft(env, "blue_0", 500, 0, psi=np.pi)
    for i, aid in enumerate(("red_1", "red_2", "blue_1", "blue_2"), start=1):
        _set_aircraft(env, aid, -6000 + i * 1000, -5000 - i * 500, psi=-0.5)
    _, _, _, _, head_info = env.step(_zero_actions(env))

    return {
        "tail_chase_attack_occurs": bool(tail_info["attack_kills"]["red"] > 0),
        "head_on_attack_blocked_by_current_aa_limit": bool(head_info["attack_kills"]["red"] == 0),
    }


def _zero_vs_zero_attack_distance_audit(config_path: Path, seeds: int = 100) -> dict[str, Any]:
    env = Homogeneous3v3AirCombatEnv(config_path)
    dt = float(env.config["simulation"]["dt"])
    attack_max = float(env.config["combat"]["attack_distance_max"])
    times: list[float] = []
    not_entered_reasons: dict[str, int] = {}
    total = 0
    entered = 0

    for seed in range(seeds):
        env.reset(seed)
        red_entered = {aid: None for aid in RED_IDS}
        red_failed = {aid: None for aid in RED_IDS}
        while True:
            for rid in RED_IDS:
                red = env._aircraft_by_id(rid)
                if red_entered[rid] is not None or red_failed[rid] is not None:
                    continue
                if not red.state.alive:
                    red_failed[rid] = "red_dead_before_entry"
                    continue
                blues = [env._aircraft_by_id(bid) for bid in BLUE_IDS if env._aircraft_by_id(bid).state.alive]
                if not blues:
                    red_failed[rid] = "no_alive_blue_before_entry"
                    continue
                d = float(np.min([np.linalg.norm(b.state.as_array()[:3] - red.state.as_array()[:3]) for b in blues]))
                if d <= attack_max:
                    red_entered[rid] = env.step_count * dt
            if all(red_entered[aid] is not None or red_failed[aid] is not None for aid in RED_IDS):
                break
            _, _, term, trunc, info = env.step(_zero_actions(env))
            if term or trunc:
                for rid in RED_IDS:
                    if red_entered[rid] is None and red_failed[rid] is None:
                        red_failed[rid] = info.get("termination_reason") or "episode_end_before_entry"
                break
        for rid in RED_IDS:
            total += 1
            if red_entered[rid] is not None:
                entered += 1
                times.append(float(red_entered[rid]))
            else:
                reason = red_failed[rid] or "not_entered"
                not_entered_reasons[reason] = not_entered_reasons.get(reason, 0) + 1

    time_stats = _stats(times)
    time_stats.update({
        "total_samples": total,
        "entered_count": entered,
        "entered_fraction": entered / total if total else 0.0,
        "not_entered_count": total - entered,
        "not_entered_reasons": not_entered_reasons,
    })
    return time_stats


def _run_isolated_control(config_path: Path, speed: float, action: np.ndarray,
                          seconds: float = 90.0) -> dict[str, Any]:
    env = Homogeneous3v3AirCombatEnv(config_path)
    spec = env.scenario.spec
    state = env._aircraft_by_id("red_0").state.copy() if env.aircraft else None
    if state is None:
        env.reset(0)
        state = env._aircraft_by_id("red_0").state.copy()
    state.x = 0.0
    state.y = 0.0
    state.z = -3000.0
    state.v = float(speed)
    state.theta = 0.0
    state.psi = 0.0
    state.alive = True

    dt = float(env.config["simulation"]["dt"])
    steps = int(seconds / dt)
    psi_values = [state.psi]
    theta_values = [state.theta]
    yaw_rates: list[float] = []
    pitch_rates: list[float] = []
    yaw_errors: list[float] = []
    pitch_errors: list[float] = []
    nz_sat: list[bool] = []
    phi_sat: list[bool] = []
    theta_sat: list[bool] = []
    nx_sat: list[bool] = []
    for _ in range(steps):
        target, control = env.controller.control_from_action(state, action, spec)
        diag = env.controller.diagnostics(state, target, control, spec, action)
        deriv = env.dynamics.derivatives(state, control)
        yaw_rates.append(float(deriv[5]))
        pitch_rates.append(float(deriv[4]))
        yaw_errors.append(abs(float(diag["clipped_yaw_rate"]) - float(deriv[5])))
        pitch_errors.append(abs(float(diag["clipped_pitch_rate"]) - float(deriv[4])))
        nz_sat.append(bool(diag["nz_saturated"]))
        phi_sat.append(bool(diag["phi_saturated"]))
        nx_sat.append(bool(diag["nx_saturated"]))
        next_state = env.integrator.step(state, control, env.dynamics, spec)
        theta_sat.append(abs(next_state.theta - spec.theta_min) < 1e-8 or abs(next_state.theta - spec.theta_max) < 1e-8)
        state = next_state
        psi_values.append(state.psi)
        theta_values.append(state.theta)
    unwrapped = np.unwrap(np.asarray(psi_values, dtype=float))
    return {
        "dt": dt,
        "psi_unwrapped": unwrapped,
        "theta": np.asarray(theta_values, dtype=float),
        "yaw_rates": np.asarray(yaw_rates, dtype=float),
        "pitch_rates": np.asarray(pitch_rates, dtype=float),
        "yaw_tracking_errors": np.asarray(yaw_errors, dtype=float),
        "pitch_tracking_errors": np.asarray(pitch_errors, dtype=float),
        "nz_sat": np.asarray(nz_sat, dtype=bool),
        "phi_sat": np.asarray(phi_sat, dtype=bool),
        "theta_sat": np.asarray(theta_sat, dtype=bool),
        "nx_sat": np.asarray(nx_sat, dtype=bool),
    }


def _first_time(mask: np.ndarray, dt: float) -> float | None:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return float(idx[0] * dt)


def _yaw_case(config_path: Path, speed: float, action: np.ndarray) -> dict[str, Any]:
    cfg = load_config(config_path)
    spec = Homogeneous3v3AirCombatEnv(config_path).scenario.spec
    out = _run_isolated_control(config_path, speed, action)
    dt = out["dt"]
    delta = out["psi_unwrapped"] - out["psi_unwrapped"][0]
    direction = 1.0 if action[0] >= 0.0 else -1.0
    progress = direction * delta
    t90 = _first_time(progress >= np.pi / 2.0, dt)
    t180 = _first_time(progress >= np.pi, dt)
    rates = out["yaw_rates"]
    before90 = rates[:max(1, int((t90 or 90.0) / dt))]
    before180 = rates[:max(1, int((t180 or 90.0) / dt))]
    return {
        "configured_yaw_command_rate_limit": float(spec.yaw_rate_max),
        "actual_initial_yaw_rate": float(rates[0]),
        "maximum_actual_yaw_rate": float(np.max(np.abs(rates))),
        "mean_actual_yaw_rate_before_90_deg": float(np.mean(np.abs(before90))),
        "mean_actual_yaw_rate_before_180_deg": float(np.mean(np.abs(before180))),
        "yaw_rate_tracking_absolute_error_mean": float(np.mean(out["yaw_tracking_errors"])),
        "yaw_rate_tracking_absolute_error_max": float(np.max(out["yaw_tracking_errors"])),
        "nz_saturation_fraction": float(np.mean(out["nz_sat"])),
        "phi_saturation_fraction": float(np.mean(out["phi_sat"])),
        "time_to_90_deg": t90,
        "time_to_180_deg": t180,
        "reached_90_deg": t90 is not None,
        "reached_180_deg": t180 is not None,
        "used_unwrapped_heading": True,
        "max_simulation_seconds": 90.0,
    }


def _pitch_case(config_path: Path, speed: float, action: np.ndarray) -> dict[str, Any]:
    spec = Homogeneous3v3AirCombatEnv(config_path).scenario.spec
    out = _run_isolated_control(config_path, speed, action)
    rates = out["pitch_rates"]
    return {
        "configured_pitch_command_rate_limit": float(spec.pitch_rate_max),
        "actual_initial_pitch_rate": float(rates[0]),
        "maximum_actual_pitch_rate": float(np.max(np.abs(rates))),
        "mean_actual_pitch_rate": float(np.mean(rates)),
        "pitch_rate_tracking_absolute_error_mean": float(np.mean(out["pitch_tracking_errors"])),
        "pitch_rate_tracking_absolute_error_max": float(np.max(out["pitch_tracking_errors"])),
        "nz_saturation_fraction": float(np.mean(out["nz_sat"])),
        "theta_saturation_fraction": float(np.mean(out["theta_sat"])),
    }


def _maneuver_audit(config_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speed in (100.0, 150.0, 250.0):
        key = f"{int(speed)}_mps"
        result[key] = {
            "max_positive_yaw": _yaw_case(config_path, speed, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            "max_negative_yaw": _yaw_case(config_path, speed, np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
            "max_climb": _pitch_case(config_path, speed, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            "max_dive": _pitch_case(config_path, speed, np.array([0.0, -1.0, 0.0], dtype=np.float32)),
            "neutral": _pitch_case(config_path, speed, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        }
    return result


def _single_action_response(config_path: Path, speed: float, action: np.ndarray) -> dict[str, Any]:
    env = Homogeneous3v3AirCombatEnv(config_path)
    env.reset(0)
    spec = env.scenario.spec
    state = env._aircraft_by_id("red_0").state.copy()
    state.x = 0.0
    state.y = 0.0
    state.z = -3000.0
    state.v = float(speed)
    state.theta = 0.0
    state.psi = 0.0
    state.alive = True
    target, control = env.controller.control_from_action(state, action, spec)
    diag = env.controller.diagnostics(state, target, control, spec, action)
    deriv = env.dynamics.derivatives(state, control)
    return {
        "action": [float(v) for v in np.asarray(action, dtype=float)],
        "requested_yaw_rate": float(diag["requested_yaw_rate"]),
        "requested_pitch_rate": float(diag["requested_pitch_rate"]),
        "requested_acceleration": float(diag["requested_acceleration"]),
        "actual_initial_yaw_rate": float(deriv[5]),
        "actual_initial_pitch_rate": float(deriv[4]),
        "actual_initial_acceleration": float(deriv[3]),
        "command_yaw_rate_saturated": bool(diag["command_yaw_rate_saturated"]),
        "command_pitch_rate_saturated": bool(diag["command_pitch_rate_saturated"]),
        "command_acceleration_saturated": bool(diag["command_acceleration_saturated"]),
        "nx_saturated": bool(diag["nx_saturated"]),
        "nz_saturated": bool(diag["nz_saturated"]),
        "phi_saturated": bool(diag["phi_saturated"]),
        "action_mapping_mode": str(diag["action_mapping_mode"]),
    }


def _monotonic(values: list[float], tolerance: float = 1e-8) -> bool:
    return all(values[i] <= values[i + 1] + tolerance for i in range(len(values) - 1))


def _unique_count(values: list[float], decimals: int = 8) -> int:
    return len({round(float(v), decimals) for v in values})


def _axis_action(axis: str, magnitude: float) -> np.ndarray:
    arr = np.zeros(3, dtype=np.float32)
    arr[{"yaw": 0, "pitch": 1, "speed": 2}[axis]] = float(magnitude)
    return arr


def _action_mapping_audit(config_path: Path) -> dict[str, Any]:
    magnitudes = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
    result: dict[str, Any] = {
        "magnitudes": magnitudes,
        "speeds": {},
    }
    for speed in (100.0, 150.0, 250.0):
        speed_key = f"{int(speed)}_mps"
        speed_result: dict[str, Any] = {}
        for axis in ("yaw", "pitch", "speed"):
            samples = []
            positive_requested = []
            positive_actual = []
            for sign in (-1.0, 1.0):
                for magnitude in magnitudes:
                    action = _axis_action(axis, sign * magnitude)
                    response = _single_action_response(config_path, speed, action)
                    response["sign"] = int(sign)
                    response["magnitude"] = float(magnitude)
                    samples.append(response)
                    if sign > 0:
                        if axis == "yaw":
                            positive_requested.append(abs(response["requested_yaw_rate"]))
                            positive_actual.append(abs(response["actual_initial_yaw_rate"]))
                        elif axis == "pitch":
                            positive_requested.append(abs(response["requested_pitch_rate"]))
                            positive_actual.append(abs(response["actual_initial_pitch_rate"]))
                        else:
                            positive_requested.append(abs(response["requested_acceleration"]))
                            positive_actual.append(abs(response["actual_initial_acceleration"]))
            speed_result[axis] = {"samples": samples}
            if axis == "yaw":
                speed_result["requested_yaw_rate_monotonic"] = _monotonic(positive_requested)
                speed_result["actual_yaw_rate_monotonic"] = _monotonic(positive_actual)
                speed_result["unique_actual_yaw_response_count"] = _unique_count(positive_actual)
            elif axis == "pitch":
                speed_result["requested_pitch_rate_monotonic"] = _monotonic(positive_requested)
                speed_result["actual_pitch_rate_monotonic"] = _monotonic(positive_actual)
                speed_result["unique_actual_pitch_response_count"] = _unique_count(positive_actual)
            else:
                speed_result["requested_acceleration_monotonic"] = _monotonic(positive_requested)
                speed_result["actual_acceleration_monotonic"] = _monotonic(positive_actual)
                speed_result["unique_actual_acceleration_response_count"] = _unique_count(positive_actual)
        result["speeds"][speed_key] = speed_result
    return result


def _altitude_boundary_case(config_path: Path, speed: float, action: np.ndarray, climb: bool) -> dict[str, Any]:
    env = Homogeneous3v3AirCombatEnv(config_path)
    env.reset(0)
    bf = env.config["battlefield"]
    rv2 = env.config["reward_v2"]
    upper_soft = float(bf["altitude_max"] - rv2["altitude_soft_margin"])
    lower_soft = float(bf["altitude_min"] + rv2["altitude_soft_margin"])
    soft_target = upper_soft if climb else lower_soft
    physical_target = float(bf["altitude_max"] if climb else bf["altitude_min"])
    for aid in RED_IDS + BLUE_IDS:
        _set_aircraft(env, aid, -10000.0 if aid.startswith("red") else 10000.0,
                      1000.0 * (int(aid[-1]) + 1), altitude=3000.0,
                      v=speed, theta=0.0, psi=0.0)
    dt = float(env.config["simulation"]["dt"])
    soft_state_time = None
    physical_state_time = None
    env_death_time = None
    for _ in range(int(120.0 / dt)):
        ac = env._aircraft_by_id("red_0")
        alt = ac.state.altitude
        now = env.step_count * dt
        if soft_state_time is None and ((climb and alt >= soft_target) or ((not climb) and alt <= soft_target)):
            soft_state_time = float(now)
        if physical_state_time is None and ((climb and alt >= physical_target) or ((not climb) and alt <= physical_target)):
            physical_state_time = float(now)
        if not ac.state.alive and env_death_time is None:
            env_death_time = float(now)
            break
        _, _, term, trunc, _ = env.step({a.aircraft_id: (action if a.aircraft_id == "red_0" else np.zeros(3, dtype=np.float32))
                                         for a in env.aircraft if a.state.alive})
        if term or trunc:
            ac = env._aircraft_by_id("red_0")
            if not ac.state.alive and env_death_time is None:
                env_death_time = float(env.step_count * dt)
            break
    return {
        "soft_boundary_altitude": soft_target,
        "physical_boundary_altitude": physical_target,
        "time_to_soft_boundary": soft_state_time,
        "state_crossing_time": physical_state_time,
        "environment_death_time": env_death_time,
        "physical_boundary_time_semantics": "state crossing time and environment death time are both reported",
    }


def _altitude_control_audit(config_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speed in (100.0, 150.0, 250.0):
        result[f"{int(speed)}_mps"] = {
            "climb": {
                "time_to_upper_soft_boundary": _altitude_boundary_case(
                    config_path, speed, np.array([0.0, 1.0, 0.0], dtype=np.float32), True
                ),
                "time_to_upper_physical_boundary": None,
            },
            "dive": {
                "time_to_lower_soft_boundary": _altitude_boundary_case(
                    config_path, speed, np.array([0.0, -1.0, 0.0], dtype=np.float32), False
                ),
                "time_to_lower_physical_boundary": None,
            },
        }
        result[f"{int(speed)}_mps"]["climb"]["time_to_upper_physical_boundary"] = {
            k: result[f"{int(speed)}_mps"]["climb"]["time_to_upper_soft_boundary"][k]
            for k in ("state_crossing_time", "environment_death_time", "physical_boundary_time_semantics")
        }
        result[f"{int(speed)}_mps"]["dive"]["time_to_lower_physical_boundary"] = {
            k: result[f"{int(speed)}_mps"]["dive"]["time_to_lower_soft_boundary"][k]
            for k in ("state_crossing_time", "environment_death_time", "physical_boundary_time_semantics")
        }
    return result


def _run_rule_matchup(config_path: Path, red_mode: str, blue_mode: str, episodes: int,
                      num_envs: int, num_env_workers: int, strict_complete: bool) -> dict[str, Any]:
    mode_map = {"zero": 0, "pursuit": 1}
    vec = make_combat_vector_env_3v3(config_path, num_envs, num_env_workers)
    summaries: list[dict[str, Any]] = []
    all_rewards_finite = True
    all_observations_finite = True
    all_global_states_finite = True
    try:
        obs, gs, _ = vec.reset([{"seed": 2000 + i} for i in range(num_envs)])
        all_observations_finite = all_observations_finite and bool(np.all(np.isfinite(obs)))
        all_global_states_finite = all_global_states_finite and bool(np.all(np.isfinite(gs)))
        next_seed = 2000 + num_envs
        modes = np.full((num_envs, 2), [mode_map[red_mode], mode_map[blue_mode]], dtype=np.int8)
        while len(summaries) < episodes:
            result = vec.step_rules(modes)
            all_observations_finite = all_observations_finite and bool(np.all(np.isfinite(result.observations)))
            all_global_states_finite = all_global_states_finite and bool(np.all(np.isfinite(result.global_states)))
            all_rewards_finite = all_rewards_finite and bool(np.all(np.isfinite(result.team_rewards)))
            all_rewards_finite = all_rewards_finite and bool(np.all(np.isfinite(result.red_reward_components)))
            done_idx = np.where(result.episode_valid)[0]
            for gi in done_idx:
                summaries.append({
                    "red_complete_elimination_success": bool(result.red_complete_elimination_success[gi]),
                    "blue_complete_elimination_success": bool(result.blue_complete_elimination_success[gi]),
                    "red_attack_kills": int(result.episode_red_attack_kills[gi]),
                    "blue_attack_kills": int(result.episode_blue_attack_kills[gi]),
                    "red_attack_deaths": int(result.episode_red_attack_deaths[gi]),
                    "blue_attack_deaths": int(result.episode_blue_attack_deaths[gi]),
                    "red_boundary_deaths": int(result.episode_red_boundary_deaths[gi]),
                    "blue_boundary_deaths": int(result.episode_blue_boundary_deaths[gi]),
                    "red_boundary_altitude_deaths": int(result.episode_red_boundary_altitude_deaths[gi]),
                    "blue_boundary_altitude_deaths": int(result.episode_blue_boundary_altitude_deaths[gi]),
                    "red_boundary_xy_deaths": int(result.episode_red_boundary_xy_deaths[gi]),
                    "blue_boundary_xy_deaths": int(result.episode_blue_boundary_xy_deaths[gi]),
                    "red_friendly_collision_deaths": int(result.episode_red_friendly_collision_deaths[gi]),
                    "blue_friendly_collision_deaths": int(result.episode_blue_friendly_collision_deaths[gi]),
                    "red_cross_collision_deaths": int(result.episode_red_cross_collision_deaths[gi]),
                    "blue_cross_collision_deaths": int(result.episode_blue_cross_collision_deaths[gi]),
                    "red_survivors": int(result.episode_red_survivors[gi]),
                    "blue_survivors": int(result.episode_blue_survivors[gi]),
                    "episode_length": int(result.episode_length[gi]),
                    "termination_reason_code": int(result.termination_reason_codes[gi]),
                    "outcome_code": int(result.outcome_codes[gi]),
                })
            if len(done_idx) > 0:
                seeds = [{"seed": next_seed + i} for i in range(len(done_idx))]
                next_seed += len(done_idx)
                vec.reset_at(done_idx, seeds)
    finally:
        vec.close()

    records = summaries[:episodes]
    if strict_complete and len(records) != episodes:
        raise RuntimeError(f"{red_mode}_vs_{blue_mode}: episodes_completed={len(records)} != episodes_requested={episodes}")
    ledger_ok = True
    boundary_ok = True
    for rec in records:
        for team in ("red", "blue"):
            collision = rec[f"{team}_friendly_collision_deaths"] + rec[f"{team}_cross_collision_deaths"]
            ledger_ok = ledger_ok and rec[f"{team}_survivors"] + rec[f"{team}_attack_deaths"] + rec[f"{team}_boundary_deaths"] + collision == 3
            boundary_ok = boundary_ok and rec[f"{team}_boundary_deaths"] == rec[f"{team}_boundary_altitude_deaths"] + rec[f"{team}_boundary_xy_deaths"]

    def mean(key: str) -> float:
        return float(np.mean([rec[key] for rec in records])) if records else float("nan")

    n = len(records)
    return {
        "episodes_requested": int(episodes),
        "episodes_completed": int(n),
        "red_complete_elimination_success_rate": float(np.mean([rec["red_complete_elimination_success"] for rec in records])) if records else 0.0,
        "blue_complete_elimination_success_rate": float(np.mean([rec["blue_complete_elimination_success"] for rec in records])) if records else 0.0,
        "environment_red_outcome_rate": float(np.mean([rec["outcome_code"] == 1 for rec in records])) if records else 0.0,
        "environment_blue_outcome_rate": float(np.mean([rec["outcome_code"] == 2 for rec in records])) if records else 0.0,
        "draw_rate": float(np.mean([rec["outcome_code"] == 3 for rec in records])) if records else 0.0,
        "max_steps_rate": float(np.mean([rec["termination_reason_code"] == 4 for rec in records])) if records else 0.0,
        "mean_episode_length": mean("episode_length"),
        "mean_red_attack_kills": mean("red_attack_kills"),
        "mean_blue_attack_kills": mean("blue_attack_kills"),
        "mean_red_attack_deaths": mean("red_attack_deaths"),
        "mean_blue_attack_deaths": mean("blue_attack_deaths"),
        "mean_red_boundary_altitude_deaths": mean("red_boundary_altitude_deaths"),
        "mean_blue_boundary_altitude_deaths": mean("blue_boundary_altitude_deaths"),
        "mean_red_boundary_xy_deaths": mean("red_boundary_xy_deaths"),
        "mean_blue_boundary_xy_deaths": mean("blue_boundary_xy_deaths"),
        "red_altitude_boundary_episode_rate": float(np.mean([rec["red_boundary_altitude_deaths"] > 0 for rec in records])) if records else 0.0,
        "blue_altitude_boundary_episode_rate": float(np.mean([rec["blue_boundary_altitude_deaths"] > 0 for rec in records])) if records else 0.0,
        "red_xy_boundary_episode_rate": float(np.mean([rec["red_boundary_xy_deaths"] > 0 for rec in records])) if records else 0.0,
        "blue_xy_boundary_episode_rate": float(np.mean([rec["blue_boundary_xy_deaths"] > 0 for rec in records])) if records else 0.0,
        "mean_red_friendly_collision_deaths": mean("red_friendly_collision_deaths"),
        "mean_blue_friendly_collision_deaths": mean("blue_friendly_collision_deaths"),
        "mean_red_cross_collision_deaths": mean("red_cross_collision_deaths"),
        "mean_blue_cross_collision_deaths": mean("blue_cross_collision_deaths"),
        "mean_red_survivors": mean("red_survivors"),
        "mean_blue_survivors": mean("blue_survivors"),
        "all_rewards_finite": bool(all_rewards_finite),
        "all_observations_finite": bool(all_observations_finite),
        "all_global_states_finite": bool(all_global_states_finite),
        "death_ledger_conserved": bool(ledger_ok),
        "boundary_total_matches_altitude_plus_xy": bool(boundary_ok),
    }


def _scenario_timescale(config_path: Path, cfg: dict[str, Any], initial: dict[str, Any],
                        actual_time: dict[str, Any], maneuver: dict[str, Any],
                        altitude: dict[str, Any], nominal: dict[str, float | None],
                        horizontal_reachable: bool) -> dict[str, Any]:
    tmean = actual_time.get("mean")
    turn90 = maneuver["150_mps"]["max_positive_yaw"]["time_to_90_deg"]
    turn180 = maneuver["150_mps"]["max_positive_yaw"]["time_to_180_deg"]
    return {
        "episode_duration_seconds": float(cfg["simulation"]["dt"] * cfg["simulation"]["max_steps"]),
        "attack_distance_min": float(cfg["combat"]["attack_distance_min"]),
        "attack_distance_max": float(cfg["combat"]["attack_distance_max"]),
        "initial_nearest_enemy_distance_mean": initial["nearest_enemy_distance_m"]["mean"],
        "initial_nearest_enemy_distance_p05": initial["nearest_enemy_distance_m"]["p05"],
        "initial_nearest_enemy_distance_p95": initial["nearest_enemy_distance_m"]["p95"],
        "nominal_centerline_time_to_attack_distance": nominal,
        "actual_time_to_attack_distance_mean": actual_time.get("mean"),
        "actual_time_to_attack_distance_p05": actual_time.get("p05"),
        "actual_time_to_attack_distance_p95": actual_time.get("p95"),
        "actual_90_degree_turn_time_at_150_mps": turn90,
        "actual_180_degree_turn_time_at_150_mps": turn180,
        "time_to_upper_soft_altitude_boundary_at_150_mps": altitude["150_mps"]["climb"]["time_to_upper_soft_boundary"]["time_to_soft_boundary"],
        "time_to_lower_soft_altitude_boundary_at_150_mps": altitude["150_mps"]["dive"]["time_to_lower_soft_boundary"]["time_to_soft_boundary"],
        "initial_distance_to_attack_distance_ratio": initial["nearest_enemy_distance_m"]["mean"] / cfg["combat"]["attack_distance_max"],
        "turn_90_to_first_merge_ratio": turn90 / tmean if turn90 is not None and tmean else None,
        "turn_180_to_first_merge_ratio": turn180 / tmean if turn180 is not None and tmean else None,
        "horizontal_physical_boundary_theoretically_reachable": bool(horizontal_reachable),
    }


def audit_config(config_path: Path, episodes: int, num_envs: int, env_workers: int,
                 strict_complete: bool = True) -> dict[str, Any]:
    cfg = load_config(config_path)
    sim, ac, bf, scen, combat = cfg["simulation"], cfg["aircraft"], cfg["battlefield"], cfg["scenario"], cfg["combat"]
    duration = float(sim["dt"] * sim["max_steps"])
    max_path = float(ac["v_max"] * duration)
    max_slot = max(
        abs(-scen["lateral_spacing"] - scen["opposing_lateral_offset"] / 2.0),
        abs(scen["lateral_spacing"] + scen["opposing_lateral_offset"] / 2.0),
    )
    initial_max_radius = float(np.hypot(scen["separation_max"] / 2.0, max_slot))
    horizontal_reachable = bool(initial_max_radius + max_path > min(bf["x_limit"], bf["y_limit"]))
    nominal = {
        "from_separation_min": max(0.0, (scen["separation_min"] - combat["attack_distance_max"]) / (2.0 * scen["speed_center"])),
        "from_separation_max": max(0.0, (scen["separation_max"] - combat["attack_distance_max"]) / (2.0 * scen["speed_center"])),
    }
    initial = _initial_geometry_audit(config_path)
    actual_time = _zero_vs_zero_attack_distance_audit(config_path)
    maneuver = _maneuver_audit(config_path)
    action_mapping = _action_mapping_audit(config_path)
    altitude = _altitude_control_audit(config_path)
    matchups = {
        "zero_vs_pursuit": _run_rule_matchup(config_path, "zero", "pursuit", episodes, num_envs, env_workers, strict_complete),
        "pursuit_vs_pursuit": _run_rule_matchup(config_path, "pursuit", "pursuit", episodes, num_envs, env_workers, strict_complete),
    }
    return {
        "config_path": str(config_path.resolve()),
        "reward_mode": cfg["combat"]["reward_mode"],
        "dt": float(sim["dt"]),
        "max_steps": int(sim["max_steps"]),
        "episode_duration_seconds": duration,
        "v_min": float(ac["v_min"]),
        "v_max": float(ac["v_max"]),
        "theoretical_max_path_length": max_path,
        "xy_physical_boundary": {"x_limit": float(bf["x_limit"]), "y_limit": float(bf["y_limit"])},
        "horizontal_soft_boundary_start": {
            "x": float(bf["x_limit"] * cfg["reward_v2"]["horizontal_soft_ratio"]),
            "y": float(bf["y_limit"] * cfg["reward_v2"]["horizontal_soft_ratio"]),
        },
        "configured_initial_max_position_radius": initial_max_radius,
        "horizontal_physical_boundary_theoretically_reachable": horizontal_reachable,
        "nominal_centerline_time_to_attack_distance_seconds": nominal,
        "initial_seed_checks": initial,
        "actual_time_to_attack_distance_seconds": actual_time,
        "action_mapping_audit": action_mapping,
        "maneuver_control_audit": maneuver,
        "altitude_control_audit": altitude,
        "constructed_geometry_checks": _constructed_geometry_checks(config_path),
        "rule_matchups": matchups,
        "scenario_timescale_comparison": _scenario_timescale(
            config_path, cfg, initial, actual_time, maneuver, altitude, nominal, horizontal_reachable
        ),
    }


def build_audit(episodes: int, num_envs: int, env_workers: int, output: Path,
                generated_by_test: bool = False, strict_complete: bool = True,
                config_paths: tuple[Path, ...] = DEFAULT_CONFIGS) -> dict[str, Any]:
    audits = {}
    for path in config_paths:
        audits[path.stem] = audit_config(path, episodes, num_envs, env_workers, strict_complete)
    all_complete = all(
        matchup["episodes_requested"] == matchup["episodes_completed"]
        for audit in audits.values()
        for matchup in audit["rule_matchups"].values()
    )
    if strict_complete and not all_complete:
        raise RuntimeError("formal audit did not complete all requested episodes")
    all_finite = all(
        matchup["all_rewards_finite"] and matchup["all_observations_finite"] and matchup["all_global_states_finite"]
        for audit in audits.values()
        for matchup in audit["rule_matchups"].values()
    )
    all_ledgers = all(
        matchup["death_ledger_conserved"] and matchup["boundary_total_matches_altitude_plus_xy"]
        for audit in audits.values()
        for matchup in audit["rule_matchups"].values()
    )
    yaw_below = {
        name: {
            speed: data["max_positive_yaw"]["maximum_actual_yaw_rate"] < data["max_positive_yaw"]["configured_yaw_command_rate_limit"]
            for speed, data in audit["maneuver_control_audit"].items()
        }
        for name, audit in audits.items()
    }
    summary = {
        "configs_successfully_run": list(audits.keys()),
        "all_requested_episodes_completed": bool(all_complete),
        "all_rewards_observations_global_states_finite": bool(all_finite),
        "all_death_ledgers_conserved": bool(all_ledgers),
        "horizontal_boundary_reachable_by_theoretical_max_path": {
            name: audit["horizontal_physical_boundary_theoretically_reachable"]
            for name, audit in audits.items()
        },
        "actual_yaw_rate_below_configured_command_limit": yaw_below,
        "timescale_difference_observed": {
            name: {
                "turn_90_to_first_merge_ratio": audit["scenario_timescale_comparison"]["turn_90_to_first_merge_ratio"],
                "turn_180_to_first_merge_ratio": audit["scenario_timescale_comparison"]["turn_180_to_first_merge_ratio"],
            }
            for name, audit in audits.items()
        },
        "altitude_boundary_reached_by_sustained_actions": {
            name: {
                speed: {
                    "climb_upper_soft": data["climb"]["time_to_upper_soft_boundary"]["time_to_soft_boundary"] is not None,
                    "climb_upper_physical": data["climb"]["time_to_upper_physical_boundary"]["state_crossing_time"] is not None,
                    "dive_lower_soft": data["dive"]["time_to_lower_soft_boundary"]["time_to_soft_boundary"] is not None,
                    "dive_lower_physical": data["dive"]["time_to_lower_physical_boundary"]["state_crossing_time"] is not None,
                }
                for speed, data in audit["altitude_control_audit"].items()
            }
            for name, audit in audits.items()
        },
    }
    return {
        "audit_version": AUDIT_VERSION,
        "generated_by_test": bool(generated_by_test),
        "episodes_requested": int(episodes),
        "num_envs": int(num_envs),
        "env_workers": int(env_workers),
        "output_path": str(output),
        "config_audits": audits,
        "summary": summary,
    }


def write_audit(audit: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--env-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(
        episodes=args.episodes,
        num_envs=args.num_envs,
        env_workers=args.env_workers,
        output=args.output,
        generated_by_test=False,
        strict_complete=True,
    )
    write_audit(audit, args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
