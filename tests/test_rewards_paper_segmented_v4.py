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
    DEATH_BOUNDARY_ALTITUDE,
    DEATH_BOUNDARY_XY,
    DEATH_COLLISION_CROSS,
    Homogeneous3v3AirCombatEnv,
)
from uav_combat.mappo.vector_env_3v3 import (
    RED_REWARD_COMPONENT_KEYS_3V3,
    LocalCombatVectorEnv3v3,
    SubprocessCombatVectorEnv3v3,
)
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
from uav_combat.models import AircraftState
from uav_combat.rewards import paper_segmented_local_reward


ROOT = Path(__file__).parents[1]
CONFIG_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
CONFIG_V5 = ROOT / "configs" / "homogeneous_3v3_learnable_v5_greedy_blue.yaml"
CONFIG_V6 = ROOT / "configs" / "homogeneous_3v3_learnable_v6_task_aligned.yaml"
CONFIG_V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _state(x=0.0, y=0.0, z=-3000.0, psi=0.0, theta=0.0, alive=True, v=150.0):
    return AircraftState(x, y, z, v, theta, psi, alive)


def _polar_target(distance: float, angle: float) -> AircraftState:
    return _state(distance * np.cos(angle), distance * np.sin(angle), psi=angle)


def _cfg():
    cfg = load_config(CONFIG_V7)
    return cfg, cfg["reward_paper_segmented_v4"], cfg["combat"]


def _set(env, aid, x, y, z=-3000.0, psi=0.0, theta=0.0, alive=True, v=150.0):
    ac = env._aircraft_by_id(aid)
    ac.state.x = x
    ac.state.y = y
    ac.state.z = z
    ac.state.v = v
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


def _assert_v7_reward_accounting(rc, team="red"):
    dense = (
        rc[f"{team}_approach_reward"]
        + rc[f"{team}_attack_advantage_reward"]
        + rc[f"{team}_threat_penalty"]
    )
    assert rc[f"{team}_dense_reward"] == pytest.approx(dense, abs=1e-8)

    total = (
        rc[f"{team}_dense_reward"]
        + rc[f"{team}_kill_reward"]
        + rc[f"{team}_attack_death_penalty"]
        + rc[f"{team}_boundary_death_penalty"]
        + rc[f"{team}_collision_death_penalty"]
        + rc[f"{team}_terminal_reward"]
    )
    assert rc[f"{team}_team_total_reward"] == pytest.approx(total, abs=1e-8)


def _assert_old_double_dense_total_rejected(rc, team="red"):
    keys = [f"{team}_{key.removeprefix('red_')}" for key in RED_REWARD_COMPONENT_KEYS_3V3[:-1]]
    old_incorrect_total = sum(rc[key] for key in keys)
    duplicate_dense = (
        rc[f"{team}_approach_reward"]
        + rc[f"{team}_attack_advantage_reward"]
        + rc[f"{team}_threat_penalty"]
    )
    if abs(duplicate_dense) > 1e-12:
        assert old_incorrect_total != pytest.approx(rc[f"{team}_team_total_reward"], abs=1e-8)
        assert old_incorrect_total - rc[f"{team}_team_total_reward"] == pytest.approx(duplicate_dense, abs=1e-8)


def _v7_rewards(env, attack_kills=None, step_deaths=None, terminated=False, truncated=False,
                outcome=None, reason=None, red_alive=3, blue_alive=3):
    dense, targets = env._capture_paper_segmented_v4_pre_attack()
    return env._compute_paper_segmented_v4_rewards(
        attack_kills or {"red": 0, "blue": 0},
        step_deaths or {},
        terminated,
        truncated,
        outcome,
        reason,
        red_alive,
        blue_alive,
        dense,
        targets,
    )


def _prepare_single_pair_env(angle_deg=4.0, distance=500.0):
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(123)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    angle = np.deg2rad(angle_deg)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    # Same velocity as red keeps relative geometry stable through integration.
    _set(env, "blue_0", distance * np.cos(angle), distance * np.sin(angle), psi=0.0, alive=True)
    return env


def _prepare_r42_env(reverse_ata_deg=4.0, distance=500.0):
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(456)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    angle = np.deg2rad(reverse_ata_deg)
    _set(env, "red_0", 0.0, 0.0, psi=np.pi, alive=True)
    _set(env, "blue_0", distance * np.cos(angle), distance * np.sin(angle), psi=np.pi, alive=True)
    return env


def _red_nonzero_fields(rc):
    return {
        "approach": rc["red_approach_reward"],
        "attack": rc["red_attack_advantage_reward"],
        "threat": rc["red_threat_penalty"],
        "dense": rc["red_dense_reward"],
        "kill": rc["red_kill_reward"],
        "attack_death": rc["red_attack_death_penalty"],
        "boundary_death": rc["red_boundary_death_penalty"],
        "collision_death": rc["red_collision_death_penalty"],
        "terminal": rc["red_terminal_reward"],
        "total": rc["red_team_total_reward"],
    }


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
    rewards, rc, targets = _v7_rewards(env, red_alive=1, blue_alive=1)
    assert targets["red_0"] == "blue_0"
    assert targets["blue_0"] == "red_0"
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_dense_reward"] == pytest.approx(rc["red_approach_reward"] + rc["red_attack_advantage_reward"] + rc["red_threat_penalty"])
    assert rewards["red_0"] == pytest.approx(rc["red_team_total_reward"])
    _assert_v7_reward_accounting(rc, "red")
    _assert_v7_reward_accounting(rc, "blue")
    _assert_old_double_dense_total_rejected(rc, "red")


def test_v7_pure_r3_total_equals_dense_not_double_dense():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(210)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 1200.0, 0.0, psi=0.0, alive=True)
    _, rc, _ = _v7_rewards(env, red_alive=1, blue_alive=1)
    assert rc["red_approach_reward"] == pytest.approx(0.001 / 3.0)
    assert rc["red_attack_advantage_reward"] == 0.0
    assert rc["red_threat_penalty"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(rc["red_dense_reward"])
    assert rc["red_team_total_reward"] != pytest.approx(2.0 * rc["red_dense_reward"])
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_v7_pure_r41_fine_total_equals_dense_not_double_dense():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(211)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _, rc, _ = _v7_rewards(env, red_alive=1, blue_alive=1)
    assert rc["red_approach_reward"] == 0.0
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_threat_penalty"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_team_total_reward"] != pytest.approx(2.0 * 0.10 / 3.0)
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_v7_pure_r42_fine_total_equals_dense_not_double_dense():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(212)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=np.pi, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=np.pi, alive=True)
    _, rc, _ = _v7_rewards(env, red_alive=1, blue_alive=1)
    assert rc["red_approach_reward"] == 0.0
    assert rc["red_attack_advantage_reward"] == 0.0
    assert rc["red_threat_penalty"] == pytest.approx(-0.150 / 3.0)
    assert rc["red_team_total_reward"] == pytest.approx(-0.150 / 3.0)
    assert rc["red_team_total_reward"] != pytest.approx(2.0 * -0.150 / 3.0)
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_v7_r3_and_r41_at_attack_max_total_adds_dense_once():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(213)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 1000.0, 0.0, psi=0.0, alive=True)
    _, rc, _ = _v7_rewards(env, red_alive=1, blue_alive=1)
    expected_dense = (0.001 + 0.10) / 3.0
    assert rc["red_approach_reward"] == pytest.approx(0.001 / 3.0)
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_team_total_reward"] == pytest.approx(expected_dense)
    assert rc["red_team_total_reward"] != pytest.approx(2.0 * expected_dense)
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_v7_r41_and_r42_can_coexist_across_team_pairs_without_double_dense():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(214)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _set(env, "red_1", 0.0, 2000.0, psi=np.pi, alive=True)
    _set(env, "blue_1", 500.0, 2000.0, psi=np.pi, alive=True)
    _, rc, _ = _v7_rewards(env, red_alive=2, blue_alive=2)
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_threat_penalty"] == pytest.approx(-0.150 / 3.0)
    assert rc["red_dense_reward"] == pytest.approx((0.10 - 0.150) / 3.0)
    assert rc["red_team_total_reward"] == pytest.approx(rc["red_dense_reward"])
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_event_rewards_use_single_negative_loss_penalty_per_death_cause():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(3)
    _disable_dense_geometry(env)
    step_deaths = {
        "red_0": DEATH_ATTACK,
        "red_1": DEATH_BOUNDARY_XY,
        "red_2": DEATH_COLLISION_CROSS,
    }
    _, rc, _ = _v7_rewards(env, {"red": 2, "blue": 1}, step_deaths, red_alive=0, blue_alive=1)
    assert rc["red_kill_reward"] == 20.0
    assert rc["red_attack_death_penalty"] == -10.0
    assert rc["red_boundary_death_penalty"] == -10.0
    assert rc["red_collision_death_penalty"] == -10.0
    assert rc["red_team_total_reward"] == pytest.approx(rc["red_dense_reward"] - 10.0)
    _assert_v7_reward_accounting(rc)


@pytest.mark.parametrize("step_kills,red_alive,expected_total", [(0, 3, -30.0), (1, 3, -20.0), (2, 3, -10.0)])
def test_terminal_mission_failure_penalizes_surviving_aircraft(step_kills, red_alive, expected_total):
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(4)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = step_kills
    _, rc, _ = _v7_rewards(
        env, {"red": step_kills, "blue": 0}, {}, False, True, "blue", "max_steps", red_alive, 3)
    assert rc["red_terminal_reward"] == pytest.approx(-10.0 * red_alive)
    assert rc["red_team_total_reward"] == pytest.approx(expected_total)
    _assert_v7_reward_accounting(rc)


def test_complete_attack_elimination_has_no_bonus_and_preserves_loss_arithmetic():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(5)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = 3
    _, rc, _ = _v7_rewards(
        env,
        {"red": 3, "blue": 0},
        {"red_0": DEATH_ATTACK, "red_1": DEATH_BOUNDARY_XY},
        True, False, "red", "red_elimination", red_alive=1, blue_alive=0)
    assert rc["red_terminal_reward"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(30.0 - 20.0)
    _assert_v7_reward_accounting(rc)


def test_mutual_elimination_extra_penalty_and_non_attack_opponent_elimination_is_failure():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(6)
    _disable_dense_geometry(env)
    env._episode_attack_kills["red"] = 3
    _, rc_mutual, _ = _v7_rewards(
        env, {"red": 0, "blue": 0}, {}, True, False, "draw", "mutual_elimination", 0, 0)
    assert rc_mutual["red_terminal_reward"] == -10.0
    assert rc_mutual["red_team_total_reward"] == -10.0

    env._episode_attack_kills["red"] = 0
    _, rc_non_attack, _ = _v7_rewards(
        env,
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


@pytest.mark.parametrize("angle_deg,expected", [(4.0, 0.10), (10.0, 0.02), (25.0, 0.01)])
def test_v7_env_step_records_r41_and_kill_same_step(angle_deg, expected):
    env = _prepare_single_pair_env(angle_deg=angle_deg, distance=500.0)
    _, cfg, combat = _cfg()
    before = paper_segmented_local_reward(
        env._aircraft_by_id("red_0").state,
        env._aircraft_by_id("blue_0").state,
        combat["attack_distance_min"],
        combat["attack_distance_max"],
        cfg,
    )
    assert before["attack_advantage"] == pytest.approx(expected)
    assert env.attack_model.can_attack(env._aircraft_by_id("red_0").state, env._aircraft_by_id("blue_0").state)

    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert info["attacks"]["red_0"] == "blue_0"
    assert env._aircraft_by_id("blue_0").state.alive is False
    assert info["attack_kills"]["red"] == 1
    assert rc["red_attack_advantage_reward"] == pytest.approx(expected / 3.0)
    assert rc["red_kill_reward"] == 10.0
    assert info["reward_targets"]["red_0"] == "blue_0"
    _assert_v7_reward_accounting(rc)


def test_v7_r41_plus_kill_nonterminal_adds_dense_once():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(320)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "red_1", -5000.0, 5000.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_1", 5000.0, 5000.0, psi=np.pi, alive=True)
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert not terminated and not truncated
    assert info["attack_kills"]["red"] == 1
    assert env._aircraft_by_id("blue_1").state.alive is True
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_dense_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_kill_reward"] == 10.0
    assert rc["red_terminal_reward"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(10.0 + 0.10 / 3.0)
    assert rc["red_team_total_reward"] != pytest.approx(10.0 + 2.0 * 0.10 / 3.0)
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


@pytest.mark.parametrize("angle_deg,expected", [(5.0, 0.10), (15.0 - 1e-10, 0.02), (30.0, 0.01)])
def test_v7_env_step_r41_angle_boundaries(angle_deg, expected):
    env = _prepare_single_pair_env(angle_deg=angle_deg, distance=500.0)
    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert rc["red_attack_advantage_reward"] == pytest.approx(expected / 3.0)
    assert rc["red_kill_reward"] == 10.0
    assert info["reward_targets"]["red_0"] == "blue_0"
    _assert_v7_reward_accounting(rc)


@pytest.mark.parametrize("distance,expected_guide", [(100.0 + 1e-9, 0.0), (1000.0, 0.001)])
def test_v7_env_step_r41_distance_boundaries_and_r3_at_dmax(distance, expected_guide):
    env = _prepare_single_pair_env(angle_deg=4.0, distance=distance)
    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert rc["red_attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    assert rc["red_approach_reward"] == pytest.approx(expected_guide / 3.0)
    assert rc["red_kill_reward"] == 10.0
    assert info["reward_targets"]["red_0"] == "blue_0"
    _assert_v7_reward_accounting(rc)


@pytest.mark.parametrize("reverse_ata_deg,expected", [(4.0, -0.150), (10.0, -0.025), (25.0, -0.015)])
def test_v7_env_step_records_r42_and_death_same_step(reverse_ata_deg, expected):
    env = _prepare_r42_env(reverse_ata_deg=reverse_ata_deg, distance=500.0)
    _, cfg, combat = _cfg()
    before = paper_segmented_local_reward(
        env._aircraft_by_id("red_0").state,
        env._aircraft_by_id("blue_0").state,
        combat["attack_distance_min"],
        combat["attack_distance_max"],
        cfg,
    )
    assert before["threat"] == pytest.approx(expected)
    assert env.attack_model.can_attack(env._aircraft_by_id("blue_0").state, env._aircraft_by_id("red_0").state)
    assert not env.attack_model.can_attack(env._aircraft_by_id("red_0").state, env._aircraft_by_id("blue_0").state)

    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert info["attacks"]["blue_0"] == "red_0"
    assert env._aircraft_by_id("red_0").state.alive is False
    assert info["attack_kills"]["blue"] == 1
    assert rc["red_threat_penalty"] == pytest.approx(expected / 3.0)
    assert rc["red_attack_death_penalty"] == -10.0
    assert info["reward_targets"]["red_0"] == "blue_0"
    _assert_v7_reward_accounting(rc)


def test_v7_r42_plus_attack_death_nonterminal_adds_dense_once():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(321)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=np.pi, alive=True)
    _set(env, "red_1", -5000.0, 5000.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=np.pi, alive=True)
    _set(env, "blue_1", 5000.0, 5000.0, psi=np.pi, alive=True)
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert not terminated and not truncated
    assert info["attack_kills"]["blue"] == 1
    assert env._aircraft_by_id("red_1").state.alive is True
    assert rc["red_threat_penalty"] == pytest.approx(-0.150 / 3.0)
    assert rc["red_dense_reward"] == pytest.approx(-0.150 / 3.0)
    assert rc["red_attack_death_penalty"] == -10.0
    assert rc["red_terminal_reward"] == 0.0
    assert rc["red_team_total_reward"] == pytest.approx(-10.0 - 0.150 / 3.0)
    assert rc["red_team_total_reward"] != pytest.approx(-10.0 - 2.0 * 0.150 / 3.0)
    _assert_v7_reward_accounting(rc)
    _assert_old_double_dense_total_rejected(rc)


def test_v7_reward_target_preserved_on_kill_step_and_recomputed_next_step():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(700)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_1", 2000.0, 0.0, psi=0.0, alive=True)

    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert not terminated and not truncated
    assert info["reward_targets"]["red_0"] == "blue_0"
    assert env._aircraft_by_id("blue_0").state.alive is False

    _, _, _, _, info2 = env.step(_zero_actions(env))
    assert info2["reward_targets"]["red_0"] == "blue_1"


def test_v7_boundary_and_collision_deaths_do_not_enter_pre_attack_dense():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(701)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, z=-7000.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert info["death_causes"]["red_0"] == DEATH_BOUNDARY_ALTITUDE
    assert info["reward_targets"]["red_0"] is None
    assert rc["red_attack_advantage_reward"] == 0.0
    assert rc["red_threat_penalty"] == 0.0

    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(702)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "red_1", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert info["reward_targets"]["red_0"] is None
    assert info["reward_targets"]["red_1"] is None
    assert rc["red_attack_advantage_reward"] == 0.0
    assert rc["red_collision_death_penalty"] == -20.0
    _assert_v7_reward_accounting(rc)


def test_v7_same_step_cross_team_attack_kills_keep_dense_and_ledger():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V7)
    env.reset(703)
    for aid in RED_IDS + BLUE_IDS:
        _set(env, aid, 9000.0, 9000.0, alive=False)
    _set(env, "red_0", 0.0, 0.0, psi=0.0, alive=True)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0, alive=True)
    _set(env, "red_1", 0.0, 2000.0, z=-3100.0, psi=np.pi, alive=True)
    _set(env, "blue_1", 500.0, 2000.0, z=-3100.0, psi=np.pi, alive=True)
    _, _, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert info["attack_kills"]["red"] == 1
    assert info["attack_kills"]["blue"] == 1
    assert rc["red_attack_advantage_reward"] > 0.0
    assert rc["red_threat_penalty"] < 0.0
    assert rc["red_kill_reward"] == 10.0
    assert rc["red_attack_death_penalty"] == -10.0
    _assert_v7_reward_accounting(rc)


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
        dense = result.red_reward_components[:, 0] + result.red_reward_components[:, 1] + result.red_reward_components[:, 2]
        assert np.allclose(result.red_reward_components[:, 7], dense)
        expected_total = (
            result.red_reward_components[:, 7]
            + result.red_reward_components[:, 8]
            + result.red_reward_components[:, 9]
            + result.red_reward_components[:, 10]
            + result.red_reward_components[:, 11]
            + result.red_reward_components[:, 12]
        )
        assert np.allclose(result.red_reward_components[:, -1], expected_total)
        assert np.allclose(result.team_rewards, result.red_reward_components[:, -1])
        old_incorrect_total = result.red_reward_components[:, :-1].sum(axis=1)
        dense_nonzero = np.abs(dense) > 1e-12
        assert np.allclose(
            old_incorrect_total[dense_nonzero] - result.red_reward_components[dense_nonzero, -1],
            dense[dense_nonzero],
            atol=1e-5,
        )
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


def test_v7_trainer_buffer_receives_single_dense_team_total_without_optimizer(tmp_path):
    env_cfg = load_config(CONFIG_V7)
    env_cfg["simulation"]["max_steps"] = 1
    env_path = _write_tmp_config(tmp_path, env_cfg)
    train_cfg = yaml.safe_load((ROOT / "configs" / "mappo_3v3_fixed_blue.yaml").read_text(encoding="utf-8"))
    train_cfg["experiment"]["device"] = "cpu"
    train_cfg["experiment"]["output_dir"] = str(tmp_path / "unused")
    train_cfg["training"]["num_envs"] = 1
    train_cfg["training"]["num_env_workers"] = 1
    train_cfg["training"]["rollout_steps"] = 1
    train_cfg["training"]["total_env_steps"] = 1
    trainer = FixedBlue3v3MAPPOTrainer(env_path, train_cfg)
    try:
        trainer.collect_rollout(remaining=1)
        means = trainer.last_rollout_reward_means
        dense = (
            means["mean_rollout_red_approach_reward"]
            + means["mean_rollout_red_attack_advantage_reward"]
            + means["mean_rollout_red_threat_penalty"]
        )
        expected_total = (
            means["mean_rollout_red_dense_reward"]
            + means["mean_rollout_red_kill_reward"]
            + means["mean_rollout_red_attack_death_penalty"]
            + means["mean_rollout_red_boundary_death_penalty"]
            + means["mean_rollout_red_collision_death_penalty"]
            + means["mean_rollout_red_terminal_reward"]
        )
        old_incorrect_total = sum(
            means[f"mean_rollout_{key}"] for key in RED_REWARD_COMPONENT_KEYS_3V3[:-1]
        )
        assert means["mean_rollout_red_dense_reward"] == pytest.approx(dense, abs=1e-8)
        assert means["mean_rollout_red_team_total_reward"] == pytest.approx(expected_total, abs=1e-6)
        assert trainer.buffer.team_rewards[0, 0] == pytest.approx(expected_total, abs=1e-6)
        assert old_incorrect_total - means["mean_rollout_red_team_total_reward"] == pytest.approx(dense, abs=1e-6)
        assert np.isfinite(trainer.buffer.team_rewards).all()
        assert np.isfinite(trainer.buffer.advantages).all()
        assert np.isfinite(trainer.buffer.returns).all()
    finally:
        trainer.close()
