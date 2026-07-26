"""Audit the current 3v3 environment contract without training."""
import json
import os
from pathlib import Path

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv
from uav_combat.mappo.vector_env_3v3 import make_combat_vector_env_3v3
from uav_combat.scenario_3v3 import BLUE_IDS, RED_IDS, Homogeneous3v3Scenario


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "homogeneous_3v3.yaml"
REWARD_CONFIG = ROOT / "configs" / "homogeneous_3v3_reward_v3.yaml"
OUTPUT = ROOT / "outputs" / "3v3_environment_contract_audit.json"


def _set(env, aid, x, y, altitude=3000.0, v=150.0, psi=0.0):
    ac = env._aircraft_by_id(aid)
    ac.state.x = float(x)
    ac.state.y = float(y)
    ac.state.z = -float(altitude)
    ac.state.v = float(v)
    ac.state.theta = 0.0
    ac.state.psi = float(psi)
    ac.state.alive = True


def _zero_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _initial_checks(cfg, seeds=100):
    sc = Homogeneous3v3Scenario(cfg)
    bf = cfg["battlefield"]
    env = Homogeneous3v3AirCombatEnv(ENV_CONFIG)
    pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
    symmetric = True
    clean = True
    max_radius = 0.0
    for seed in range(seeds):
        aircraft = sc.reset(seed)
        by_id = {a.aircraft_id: a for a in aircraft}
        for rid, bid in pairs:
            r = by_id[rid].state
            b = by_id[bid].state
            symmetric = symmetric and np.allclose([r.x, r.y, r.z], [-b.x, -b.y, b.z], atol=1e-9)
            symmetric = symmetric and np.isclose(r.v, b.v)
            hdg = abs(r.psi - b.psi)
            hdg = min(hdg, 2.0 * np.pi - hdg)
            symmetric = symmetric and np.isclose(hdg, np.pi, atol=1e-9)
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
    return {
        "seeds_checked": seeds,
        "red_blue_center_symmetric": bool(symmetric),
        "initial_no_attack_collision_boundary": bool(clean),
        "observed_initial_max_position_radius": max_radius,
    }


def _geometry_checks():
    env = Homogeneous3v3AirCombatEnv(ENV_CONFIG)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)
    _set(env, "blue_0", 500, 0, psi=0.0)
    for i, aid in enumerate(("red_1", "red_2", "blue_1", "blue_2"), start=1):
        _set(env, aid, -6000 + i * 1000, 5000 + i * 500, psi=0.5)
    _, _, _, _, tail_info = env.step(_zero_actions(env))

    env = Homogeneous3v3AirCombatEnv(ENV_CONFIG)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)
    _set(env, "blue_0", 500, 0, psi=np.pi)
    for i, aid in enumerate(("red_1", "red_2", "blue_1", "blue_2"), start=1):
        _set(env, aid, -6000 + i * 1000, -5000 - i * 500, psi=-0.5)
    _, _, _, _, head_info = env.step(_zero_actions(env))

    return {
        "tail_chase_attack_occurs": tail_info["attack_kills"]["red"] > 0,
        "head_on_attack_blocked_by_current_aa_limit": head_info["attack_kills"]["red"] == 0,
    }


def _run_rule_matchup(red_mode, blue_mode, episodes=100, num_envs=8, num_env_workers=4):
    mode_map = {"zero": 0, "pursuit": 1}
    vec = make_combat_vector_env_3v3(ENV_CONFIG, num_envs, num_env_workers)
    try:
        obs, gs, am = vec.reset([{"seed": 2000 + i} for i in range(num_envs)])
        next_seed = 2000 + num_envs
        modes = np.full((num_envs, 2), [mode_map[red_mode], mode_map[blue_mode]], dtype=np.int8)
        summaries = []
        all_finite = bool(np.all(np.isfinite(obs)) and np.all(np.isfinite(gs)))
        while len(summaries) < episodes:
            result = vec.step_rules(modes)
            all_finite = all_finite and bool(np.all(np.isfinite(result.observations)))
            all_finite = all_finite and bool(np.all(np.isfinite(result.global_states)))
            all_finite = all_finite and bool(np.all(np.isfinite(result.team_rewards)))
            all_finite = all_finite and bool(np.all(np.isfinite(result.red_reward_components)))
            done_idx = np.where(result.episode_valid)[0]
            for gi in done_idx:
                summaries.append({
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
                    "red_collision_deaths": int(result.episode_red_friendly_collision_deaths[gi] + result.episode_red_cross_collision_deaths[gi]),
                    "blue_collision_deaths": int(result.episode_blue_friendly_collision_deaths[gi] + result.episode_blue_cross_collision_deaths[gi]),
                    "red_survivors": int(result.episode_red_survivors[gi]),
                    "blue_survivors": int(result.episode_blue_survivors[gi]),
                    "max_steps": bool(result.termination_reason_codes[gi] == 4),
                })
            if len(done_idx) > 0:
                seeds = [{"seed": next_seed + i} for i in range(len(done_idx))]
                next_seed += len(done_idx)
                vec.reset_at(done_idx, seeds)
        records = summaries[:episodes]
    finally:
        vec.close()

    ledger_ok = True
    boundary_ok = True
    for rec in records:
        for team in ("red", "blue"):
            ledger_ok = ledger_ok and (
                rec[f"{team}_survivors"]
                + rec[f"{team}_attack_deaths"]
                + rec[f"{team}_boundary_deaths"]
                + rec[f"{team}_collision_deaths"]
                == 3
            )
            boundary_ok = boundary_ok and (
                rec[f"{team}_boundary_deaths"]
                == rec[f"{team}_boundary_altitude_deaths"] + rec[f"{team}_boundary_xy_deaths"]
            )

    def mean(key):
        return float(np.mean([rec[key] for rec in records]))

    return {
        "episodes": len(records),
        "mean_red_attack_deaths": mean("red_attack_deaths"),
        "mean_blue_attack_deaths": mean("blue_attack_deaths"),
        "mean_red_boundary_altitude_deaths": mean("red_boundary_altitude_deaths"),
        "mean_blue_boundary_altitude_deaths": mean("blue_boundary_altitude_deaths"),
        "mean_red_boundary_xy_deaths": mean("red_boundary_xy_deaths"),
        "mean_blue_boundary_xy_deaths": mean("blue_boundary_xy_deaths"),
        "mean_red_collision_deaths": mean("red_collision_deaths"),
        "mean_blue_collision_deaths": mean("blue_collision_deaths"),
        "mean_red_survivors": mean("red_survivors"),
        "mean_blue_survivors": mean("blue_survivors"),
        "max_steps_rate": mean("max_steps"),
        "death_ledger_conserved": bool(ledger_ok),
        "boundary_total_matches_altitude_plus_xy": bool(boundary_ok),
        "all_rewards_and_states_finite": bool(all_finite),
    }


def main():
    cfg = load_config(ENV_CONFIG)
    reward_cfg = load_config(REWARD_CONFIG)
    sim, ac, bf, scen, combat = cfg["simulation"], cfg["aircraft"], cfg["battlefield"], cfg["scenario"], cfg["combat"]
    duration = float(sim["dt"] * sim["max_steps"])
    max_path = float(ac["v_max"] * duration)
    max_slot = max(abs(-scen["lateral_spacing"] - scen["opposing_lateral_offset"] / 2.0),
                   abs(scen["lateral_spacing"] + scen["opposing_lateral_offset"] / 2.0))
    initial_max_radius = float(np.hypot(scen["separation_max"] / 2.0, max_slot))
    nominal_closing_speed = float(2.0 * scen["speed_center"])
    enter_attack_min = max(0.0, (scen["separation_min"] - combat["attack_distance_max"]) / nominal_closing_speed)
    enter_attack_max = max(0.0, (scen["separation_max"] - combat["attack_distance_max"]) / nominal_closing_speed)
    yaw_rate = float(ac["yaw_rate_max"])

    audit_episodes = int(os.environ.get("UAV_3V3_AUDIT_EPISODES", "100"))
    audit_num_envs = int(os.environ.get("UAV_3V3_AUDIT_NUM_ENVS", "8"))
    audit_workers = int(os.environ.get("UAV_3V3_AUDIT_WORKERS", "4"))
    audit = {
        "config_paths": {
            "environment": str(ENV_CONFIG),
            "reward": str(REWARD_CONFIG),
        },
        "reward_mode": cfg["combat"]["reward_mode"],
        "reward_v2_mode_in_reward_config": reward_cfg["combat"]["reward_mode"],
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
        "horizontal_physical_boundary_theoretically_reachable": bool(initial_max_radius + max_path > min(bf["x_limit"], bf["y_limit"])),
        "nominal_closing_speed": nominal_closing_speed,
        "time_to_enter_1000m_attack_distance_seconds": {
            "from_min_initial_distance": enter_attack_min,
            "from_max_initial_distance": enter_attack_max,
        },
        "minimum_turn_time_seconds": {
            "yaw_90_deg": float((np.pi / 2.0) / yaw_rate),
            "yaw_180_deg": float(np.pi / yaw_rate),
        },
        "initial_seed_checks": _initial_checks(cfg),
        "constructed_geometry_checks": _geometry_checks(),
        "rule_matchups": {
            "zero_vs_pursuit": _run_rule_matchup("zero", "pursuit", episodes=audit_episodes, num_envs=audit_num_envs, num_env_workers=audit_workers),
            "pursuit_vs_pursuit": _run_rule_matchup("pursuit", "pursuit", episodes=audit_episodes, num_envs=audit_num_envs, num_env_workers=audit_workers),
        },
    }
    audit["all_death_ledgers_conserved"] = all(
        item["death_ledger_conserved"] for item in audit["rule_matchups"].values()
    )
    audit["all_rewards_and_states_finite"] = all(
        item["all_rewards_and_states_finite"] for item in audit["rule_matchups"].values()
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
