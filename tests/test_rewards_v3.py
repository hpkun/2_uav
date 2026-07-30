"""Tests for target_consistent_team_v3 and v6 task semantics."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_combat.config import load_config
from uav_combat.environment_3v3 import (
    BLUE_IDS,
    RED_IDS,
    DEATH_COLLISION_FRIENDLY,
    Homogeneous3v3AirCombatEnv,
)
from uav_combat.mappo.vector_env_3v3 import (
    LocalCombatVectorEnv3v3,
    SubprocessCombatVectorEnv3v3,
    decode_3v3_outcome,
    decode_3v3_termination_reason,
)
from uav_combat.rewards import coupled_attack_advantage

ROOT = Path(__file__).parents[1]
CONFIG_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
CONFIG_V5 = ROOT / "configs" / "homogeneous_3v3_learnable_v5_greedy_blue.yaml"
CONFIG_V6 = ROOT / "configs" / "homogeneous_3v3_learnable_v6_task_aligned.yaml"


def _set(env, aid, x, y, z=-3000.0, v=150.0, theta=0.0, psi=0.0, alive=True):
    ac = env._aircraft_by_id(aid)
    ac.state.x = x
    ac.state.y = y
    ac.state.z = z
    ac.state.v = v
    ac.state.theta = theta
    ac.state.psi = psi
    ac.state.alive = alive


def _zero_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _write_tmp_config(tmp_path, config):
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_v4_v5_history_configs_keep_v2_and_v6_config_diff_is_limited():
    v4 = load_config(CONFIG_V4)
    v5 = load_config(CONFIG_V5)
    v6 = load_config(CONFIG_V6)

    assert v4["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert v5["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert v6["combat"]["reward_mode"] == "target_consistent_team_v3"
    assert v6["combat"]["timeout_outcome_mode"] == "red_failure_blue_win"

    for key in ("attack_distance_min", "attack_distance_max", "attack_ata_max", "attack_aa_max"):
        assert v6["combat"][key] == v5["combat"][key]
    assert v6["action"]["mapping_mode"] == "rate_aligned_v1"
    assert v6["blue_rule_policy"]["mode"] == "greedy_team_pursuit_v1"
    assert v6["red_rule_policy"]["mode"] == "paper_nearest_pursuit_v1"

    v5_norm = dict(v5)
    v6_norm = dict(v6)
    v5_norm.pop("reward_v2", None)
    v6_norm.pop("reward_v3", None)
    v5_norm["combat"] = dict(v5_norm["combat"])
    v6_norm["combat"] = dict(v6_norm["combat"])
    v5_norm["combat"]["reward_mode"] = "__reward_mode__"
    v6_norm["combat"]["reward_mode"] = "__reward_mode__"
    v6_norm["combat"].pop("timeout_outcome_mode", None)
    assert v6_norm == v5_norm

    assert "friendly_separation_weight" not in v6["reward_v3"]
    assert "head_on_risk_weight" not in v6["reward_v3"]


def test_target_consistent_reward_uses_nearest_alive_enemy_for_all_terms():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V6)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0, alive=True)
    _set(env, "red_1", 0, 1000, alive=False)
    _set(env, "red_2", 0, -1000, alive=False)
    _set(env, "blue_0", 800, 0, psi=np.pi, alive=True)
    _set(env, "blue_1", 900, 0, psi=0.0, alive=True)
    _set(env, "blue_2", 2000, 0, alive=False)
    old_states = {a.aircraft_id: a.state.copy() for a in env.aircraft}
    cfg = env.config["reward_v3"]

    score_nearest = coupled_attack_advantage(
        env._aircraft_by_id("red_0").state,
        env._aircraft_by_id("blue_0").state,
        cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    score_farther = coupled_attack_advantage(
        env._aircraft_by_id("red_0").state,
        env._aircraft_by_id("blue_1").state,
        cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert score_farther > score_nearest

    parts, targets = env._compute_target_consistent_dense("red", old_states, cfg)
    assert targets["red_0"] == "blue_0"
    assert np.isclose(parts["attack_advantage_reward"], cfg["attack_advantage_weight"] * score_nearest / 3)
    expected_threat = coupled_attack_advantage(
        env._aircraft_by_id("blue_0").state,
        env._aircraft_by_id("red_0").state,
        cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert np.isclose(parts["threat_penalty"], cfg["threat_weight"] * expected_threat / 3)
    assert parts["approach_reward"] == 0.0


def test_reward_target_tie_break_dead_filter_and_recomputed_each_step():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V6)
    env.reset(1)
    _set(env, "red_0", 0, 0, alive=True)
    _set(env, "red_1", 0, 1000, alive=False)
    _set(env, "red_2", 0, -1000, alive=False)
    _set(env, "blue_0", 1000, 0, alive=True)
    _set(env, "blue_1", -1000, 0, alive=True)
    _set(env, "blue_2", 500, 0, alive=False)
    old_states = {a.aircraft_id: a.state.copy() for a in env.aircraft}
    _, targets = env._compute_target_consistent_dense("red", old_states, env.config["reward_v3"])
    assert targets["red_0"] == "blue_0"

    env._aircraft_by_id("blue_0").state.alive = False
    _, targets = env._compute_target_consistent_dense("red", old_states, env.config["reward_v3"])
    assert targets["red_0"] == "blue_1"


def test_v3_removes_soft_collision_shaping_but_keeps_hard_collision_penalty():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V6)
    env.reset(2)
    _set(env, "red_0", 0, 0, psi=0.0)
    _set(env, "red_1", 10, 0, psi=0.0)
    _set(env, "red_2", 5000, 5000)
    for i, bid in enumerate(BLUE_IDS):
        _set(env, bid, 8000 + i * 500, 8000, psi=np.pi)
    _, rewards, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    assert rc["red_friendly_separation_penalty"] == 0.0
    assert rc["red_head_on_risk_penalty"] == 0.0
    assert info["death_causes"]["red_0"] == DEATH_COLLISION_FRIENDLY
    assert info["death_causes"]["red_1"] == DEATH_COLLISION_FRIENDLY
    assert rc["red_collision_death_penalty"] == 50.0
    assert np.isclose(
        rewards["red_0"],
        rc["red_dense_reward"] + rc["red_kill_reward"]
        - rc["red_attack_death_penalty"] - rc["red_boundary_death_penalty"]
        - rc["red_collision_death_penalty"] + rc["red_terminal_reward"],
    )


def test_v6_max_steps_is_blue_outcome_and_v5_remains_draw(tmp_path):
    v6 = load_config(CONFIG_V6)
    v6["simulation"]["max_steps"] = 1
    v6_path = _write_tmp_config(tmp_path, v6)
    env = Homogeneous3v3AirCombatEnv(v6_path)
    env.reset(3)
    _, rewards, terminated, truncated, info = env.step(_zero_actions(env))
    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "max_steps"
    assert info["outcome"] == "blue"
    assert info["red_complete_elimination_success"] is False
    assert info["blue_complete_elimination_success"] is False
    rc = info["reward_components"]
    assert rc["red_terminal_reward"] == -20.0
    assert rc["blue_terminal_reward"] == 20.0

    v5 = load_config(CONFIG_V5)
    v5["simulation"]["max_steps"] = 1
    v5_path = tmp_path / "v5.yaml"
    v5_path.write_text(yaml.safe_dump(v5, sort_keys=False), encoding="utf-8")
    env5 = Homogeneous3v3AirCombatEnv(v5_path)
    env5.reset(3)
    _, _, terminated5, truncated5, info5 = env5.step(_zero_actions(env5))
    assert terminated5 is False
    assert truncated5 is True
    assert info5["termination_reason"] == "max_steps"
    assert info5["outcome"] == "draw"
    assert info5["reward_components"]["red_terminal_reward"] == -5.0
    assert info5["reward_components"]["blue_terminal_reward"] == -5.0


def test_reward_targets_info_field_only_for_v3():
    env = Homogeneous3v3AirCombatEnv(CONFIG_V6)
    env.reset(4)
    _, _, _, _, info = env.step(_zero_actions(env))
    assert set(info["reward_targets"]) == set(RED_IDS + BLUE_IDS)
    assert all((v is None) or isinstance(v, str) for v in info["reward_targets"].values())

    env5 = Homogeneous3v3AirCombatEnv(CONFIG_V5)
    env5.reset(4)
    _, _, _, _, info5 = env5.step(_zero_actions(env5))
    assert info5["reward_targets"] == {}


@pytest.mark.parametrize("env_cls,kwargs", [
    (LocalCombatVectorEnv3v3, {"num_envs": 2}),
    (SubprocessCombatVectorEnv3v3, {"num_envs": 2, "num_env_workers": 2}),
])
def test_v6_local_and_worker_vector_envs_step_and_report_modes(env_cls, kwargs, tmp_path):
    cfg = load_config(CONFIG_V6)
    cfg["simulation"]["max_steps"] = 1
    path = _write_tmp_config(tmp_path, cfg)
    vec = env_cls(path, **kwargs)
    try:
        obs, gs, masks = vec.reset([{"seed": 10}, {"seed": 11}])
        modes = vec.policy_modes()
        assert modes["blue_policy"] == ["greedy_team_pursuit_v1", "greedy_team_pursuit_v1"]
        assert modes["blue"] == ["rate_aligned_v1", "rate_aligned_v1"]
        assert np.isfinite(obs).all()
        assert np.isfinite(gs).all()
        result = vec.step(np.zeros((2, 3, 3), dtype=np.float32))
        assert np.isfinite(result.observations).all()
        assert np.isfinite(result.global_states).all()
        assert np.isfinite(result.team_rewards).all()
        assert result.truncated.all()
        assert not result.terminated.any()
        assert [decode_3v3_termination_reason(int(c)) for c in result.termination_reason_codes] == ["max_steps", "max_steps"]
        assert [decode_3v3_outcome(int(c)) for c in result.outcome_codes] == ["blue", "blue"]
    finally:
        vec.close()
