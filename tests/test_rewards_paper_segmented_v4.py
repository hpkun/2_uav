from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_combat.config import load_config
from uav_combat.environment_3v3 import (
    BLUE_IDS,
    RED_IDS,
    DEATH_ATTACK,
    DEATH_BOUNDARY_XY,
    DEATH_COLLISION_CROSS,
    Homogeneous3v3AirCombatEnv,
)
from uav_combat.mappo.vector_env_3v3 import (
    RED_REWARD_COMPONENT_KEYS_3V3,
    LocalCombatVectorEnv3v3,
    SubprocessCombatVectorEnv3v3,
)
from uav_combat.models import AircraftState
from uav_combat.rewards import paper_segmented_local_reward


ROOT = Path(__file__).parents[1]
CONFIG_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
CONFIG_V5 = ROOT / "configs" / "homogeneous_3v3_learnable_v5_greedy_blue.yaml"
CONFIG_V6 = ROOT / "configs" / "homogeneous_3v3_learnable_v6_task_aligned.yaml"
CONFIG_V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _state(x=0.0, y=0.0, z=-3000.0, psi=0.0, theta=0.0, alive=True):
    return AircraftState(x, y, z, 150.0, theta, psi, alive)


def _polar_target(distance: float, angle: float) -> AircraftState:
    return _state(distance * np.cos(angle), distance * np.sin(angle), psi=angle)


def _cfg():
    cfg = load_config(CONFIG_V7)
    return cfg, cfg["reward_paper_segmented_v4"], cfg["combat"]


def _set(env, aid, x, y, z=-3000.0, psi=0.0, theta=0.0, alive=True):
    ac = env._aircraft_by_id(aid)
    ac.state.x = x
    ac.state.y = y
    ac.state.z = z
    ac.state.psi = psi
    ac.state.theta = theta
    ac.state.alive = alive


def _zero_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _disable_dense_geometry(env):
    for aircraft in env.aircraft:
        aircraft.state.alive = False


def _write_tmp_config(tmp_path, config):
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _assert_red_component_sum(rc):
    subtotal = sum(rc[key] for key in RED_REWARD_COMPONENT_KEYS_3V3[:-1])
    assert np.isclose(subtotal, rc["red_team_total_reward"])


def test_v7_is_isolated_and_history_configs_are_unchanged():
    assert load_config(CONFIG_V4)["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert load_config(CONFIG_V5)["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert load_config(CONFIG_V6)["combat"]["reward_mode"] == "target_consistent_team_v3"
    v7 = load_config(CONFIG_V7)
    v6 = load_config(CONFIG_V6)
    assert v7["combat"]["reward_mode"] == "paper_segmented_team_v4"
    assert v7["combat"]["timeout_outcome_mode"] == "red_failure_blue_win"
    assert v7["blue_rule_policy"] == v6["blue_rule_policy"] == {"mode": "greedy_team_pursuit_v1"}
    assert v7["red_rule_policy"] == v6["red_rule_policy"] == {"mode": "paper_nearest_pursuit_v1"}
    for section in ("simulation", "action", "aircraft", "battlefield", "scenario", "initial_state"):
        assert v7[section] == v6[section]
    for key in ("attack_distance_min", "attack_distance_max", "attack_ata_max", "attack_aa_max"):
        assert v7["combat"][key] == v6["combat"][key]
    assert "reward_v3" not in v7


def test_local_r3_far_guide_uses_current_attack_max_and_exact_gate():
    _, cfg, combat = _cfg()
    own = _state(psi=0.0)
    at_gate = paper_segmented_local_reward(
        own, _state(x=combat["attack_distance_max"], psi=0.0),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert at_gate["guide"] == pytest.approx(0.001)
    off_angle = paper_segmented_local_reward(
        own, _state(x=1200.0, y=900.0, psi=0.0),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert off_angle["guide"] == 0.0
    too_near = paper_segmented_local_reward(
        own, _state(x=999.0, psi=0.0),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert too_near["guide"] == 0.0


@pytest.mark.parametrize("angle_deg,expected", [(4.0, 0.10), (10.0, 0.02), (25.0, 0.01), (40.0, 0.0)])
def test_local_r41_attack_advantage_tiers_are_strict_priority(angle_deg, expected):
    _, cfg, combat = _cfg()
    angle = np.deg2rad(angle_deg)
    local = paper_segmented_local_reward(
        _state(psi=0.0), _polar_target(800.0, angle),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert local["attack_advantage"] == pytest.approx(expected)
    assert local["attack_advantage"] <= 0.10


def test_r41_requires_distance_gate_and_advantage_aspect_gate():
    _, cfg, combat = _cfg()
    too_close = paper_segmented_local_reward(
        _state(), _state(x=50.0, psi=0.0),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    bad_aspect = paper_segmented_local_reward(
        _state(), _state(x=800.0, psi=np.pi),
        combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert too_close["attack_advantage"] == 0.0
    assert bad_aspect["attack_advantage"] == 0.0


@pytest.mark.parametrize("angle_deg,expected", [(4.0, -0.150), (10.0, -0.025), (25.0, -0.015), (40.0, 0.0)])
def test_local_r42_reverse_threat_tiers_are_negative(angle_deg, expected):
    _, cfg, combat = _cfg()
    own = _state(psi=0.0)
    enemy = _state(x=-800.0, y=0.0, psi=np.deg2rad(angle_deg))
    local = paper_segmented_local_reward(
        own, enemy, combat["attack_distance_min"], combat["attack_distance_max"], cfg)
    assert local["threat"] == pytest.approx(expected)
    assert local["dense_total"] == pytest.approx(local["guide"] + local["attack_advantage"] + local["threat"])


def test_nearest_alive_target_id_tie_and_dead_own_contributes_zero():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(1)
    _set(env, "red_0", 0, 0, alive=True)
    _set(env, "red_1", 0, 200, alive=False)
    _set(env, "red_2", 0, -200, alive=False)
    _set(env, "blue_0", 1000, 0, alive=True, psi=0.0)
    _set(env, "blue_1", -1000, 0, alive=True, psi=0.0)
    _set(env, "blue_2", 500, 0, alive=False)
    parts, targets = env._compute_paper_segmented_v4_dense("red", env.config["reward_paper_segmented_v4"])
    assert targets["red_0"] == "blue_0"
    assert targets["red_1"] is None
    assert targets["red_2"] is None
    assert parts["dense_reward"] == pytest.approx(
        parts["approach_reward"] + parts["attack_advantage_reward"] + parts["threat_penalty"])


def test_fixed_team_denominator_and_reward_targets_for_both_teams():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(2)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 800.0, 0.0, psi=0.0, alive=True)
    rewards, rc, targets = env._compute_paper_segmented_v4_rewards(
        {"red": 0, "blue": 0}, {}, False, False, None, None, 1, 1)
    assert targets["red_0"] == "blue_0"
    assert targets["blue_0"] == "red_0"
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_dense_reward"] == pytest.approx(rc["red_approach_reward"] + rc["red_attack_advantage_reward"] + rc["red_threat_penalty"])
    assert rewards["red_0"] == pytest.approx(rc["red_team_total_reward"])
    _assert_red_component_sum(rc)


def test_event_rewards_use_single_negative_loss_penalty_per_death_cause():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(3)
    _disable_dense_geometry(env)
    step_deaths = {
        "red_0": DEATH_ATTACK,
        "red_1": DEATH_BOUNDARY_XY,
        "red_2": DEATH_COLLISION_CROSS,
    }
    _, rc, _ = env._compute_paper_segmented_v4_rewards(
        {"red": 2, "blue": 1}, step_deaths, False, False, None, None, 0, 1)
    assert rc["red_kill_reward"] == 20.0
    assert rc["red_attack_death_penalty"] == -10.0
    assert rc["red_boundary_death_penalty"] == -10.0
    assert rc["red_collision_death_penalty"] == -10.0
    assert rc["red_team_total_reward"] == pytest.approx(rc["red_dense_reward"] - 10.0)
    _assert_red_component_sum(rc)


@pytest.mark.parametrize("step_kills,red_alive,expected_total", [(0, 3, -30.0), (1, 3, -20.0), (2, 3, -10.0)])
def test_terminal_mission_failure_penalizes_surviving_aircraft(step_kills, red_alive, expected_total):
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(4)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = step_kills
    _, rc, _ = env._compute_paper_segmented_v4_rewards(
        {"red": step_kills, "blue": 0}, {}, False, True, "blue", "max_steps", red_alive, 3)
    assert rc["red_terminal_reward"] == pytest.approx(-10.0 * red_alive)
    assert rc["red_team_total_reward"] == pytest.approx(expected_total)
    _assert_red_component_sum(rc)


def test_complete_attack_elimination_has_no_bonus_and_preserves_loss_arithmetic():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(5)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = 3
    _, rc, _ = env._compute_paper_segmented_v4_rewards(
        {"red": 3, "blue": 0},
        {"red_0": DEATH_ATTACK, "red_1": DEATH_BOUNDARY_XY},
        True, False, "red", "red_elimination", red_alive=1, blue_alive=0)
    assert rc["red_terminal_reward"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(30.0 - 20.0)
    _assert_red_component_sum(rc)


def test_mutual_elimination_extra_penalty_and_non_attack_opponent_elimination_is_failure():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(6)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = 3
    _, rc_mutual, _ = env._compute_paper_segmented_v4_rewards(
        {"red": 0, "blue": 0}, {}, True, False, "draw", "mutual_elimination", 0, 0)
    assert rc_mutual["red_terminal_reward"] == -10.0
    assert rc_mutual["red_team_total_reward"] == -10.0

    env._episode_attack_kills["red"] = 0
    _, rc_non_attack, _ = env._compute_paper_segmented_v4_rewards(
        {"red": 0, "blue": 0}, {"blue_0": DEATH_BOUNDARY_XY, "blue_1": DEATH_BOUNDARY_XY, "blue_2": DEATH_BOUNDARY_XY},
        True, False, "red", "red_elimination", 3, 0)
    assert rc_non_attack["red_terminal_reward"] == -30.0
    assert rc_non_attack["red_team_total_reward"] == -30.0


def test_v7_config_validation_rejects_missing_and_invalid_fields(tmp_path):
    cfg = load_config(CONFIG_V7)
    bad = deepcopy(cfg)
    bad["reward_paper_segmented_v4"].pop("fine_angle")
    with pytest.raises(ValueError, match="missing"):
        Homogeneous3v3AirCombatEnv(_write_tmp_config(tmp_path, bad))

    bad = deepcopy(cfg)
    bad["reward_paper_segmented_v4"]["fine_angle"] = bad["reward_paper_segmented_v4"]["medium_angle"]
    with pytest.raises(ValueError, match="fine_angle < medium_angle"):
        Homogeneous3v3AirCombatEnv(_write_tmp_config(tmp_path, bad))

    bad = deepcopy(cfg)
    bad["reward_paper_segmented_v4"]["fine_threat_penalty"] = 0.01
    with pytest.raises(ValueError, match="fine_threat_penalty"):
        Homogeneous3v3AirCombatEnv(_write_tmp_config(tmp_path, bad))


@pytest.mark.parametrize("env_cls,kwargs", [
    (LocalCombatVectorEnv3v3, {"num_envs": 2}),
    (SubprocessCombatVectorEnv3v3, {"num_envs": 2, "num_env_workers": 2}),
])
def test_v7_local_and_worker_step_finite_and_component_consistent(env_cls, kwargs, tmp_path):
    cfg = load_config(CONFIG_V7)
    cfg["simulation"]["max_steps"] = 1
    path = _write_tmp_config(tmp_path, cfg)
    vec = env_cls(path, **kwargs)
    try:
        obs, gs, masks = vec.reset([{"seed": 10}, {"seed": 11}])
        modes = vec.policy_modes()
        assert modes["blue_policy"] == ["greedy_team_pursuit_v1", "greedy_team_pursuit_v1"]
        assert modes["red_policy"] == ["paper_nearest_pursuit_v1", "paper_nearest_pursuit_v1"]
        assert np.isfinite(obs).all()
        assert np.isfinite(gs).all()
        result = vec.step(np.zeros((2, 3, 3), dtype=np.float32))
        assert np.isfinite(result.observations).all()
        assert np.isfinite(result.global_states).all()
        assert np.isfinite(result.team_rewards).all()
        assert np.isfinite(result.red_reward_components).all()
        assert result.red_reward_components.shape == (2, len(RED_REWARD_COMPONENT_KEYS_3V3))
        subtotal = result.red_reward_components[:, :-1].sum(axis=1)
        assert np.allclose(subtotal, result.red_reward_components[:, -1])
        assert result.truncated.all()
        assert result.episode_valid.all()
        ledger = (
            result.episode_red_survivors
            + result.episode_red_attack_deaths
            + result.episode_red_boundary_deaths
            + result.episode_red_friendly_collision_deaths
            + result.episode_red_cross_collision_deaths
        )
        assert np.array_equal(ledger, np.full(2, 3))
    finally:
        vec.close()
