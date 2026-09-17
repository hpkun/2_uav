"""Frozen global heterogeneous role reward tests for environment v3.9."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.common.buffer import RolloutBuffer
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import (
    BLUE_IDS, ENTITY_IDS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.reward_role_v37 import uav_distance_reward, uav_speed_reward
from env.reward_role_v39 import (
    attack_gate_indicator, uav_angle_quality, uav_coupled_dense_reward,
    uav_gate_reward,
)


ROOT = Path(__file__).resolve().parents[1]
V36 = ROOT / "configs" / "env_v36.yaml"
V37 = ROOT / "configs" / "env_v37.yaml"
V38 = ROOT / "configs" / "env_v38.yaml"
V39 = ROOT / "configs" / "env_v39.yaml"


def _env(path: Path, seed: int = 41) -> HeterogeneousMAVUAVAirCombatEnv:
    env = HeterogeneousMAVUAVAirCombatEnv(path, randomize=False)
    env.reset(seed=seed)
    return env


def test_v39_contract_and_all_historical_configs_load():
    assert load_environment_config(None)["environment_version"].endswith("v3_5")
    assert load_environment_config(V36)["shaping"]["mode"] == "potential"
    assert load_environment_config(V37)["role_reward"]["mode"] == "heterogeneous_role_v1"
    assert load_environment_config(V38)["role_reward"]["mode"] == "heterogeneous_role_coupled_v1"
    config = load_environment_config(V39)
    assert config["environment_version"] == "heterogeneous_mavuav_4v4_v3_9"
    assert config["role_reward"]["mode"] == "heterogeneous_role_coupled_gate_v1"
    assert config["role_reward"]["uav"] == {"process_mode": "normalized_angle_distance_with_gate"}
    assert _env(V39).reward_mode == "heterogeneous_role_coupled_gate_v1"


def test_v39_rejects_pbrs_and_unfrozen_reward_parameters():
    config = deepcopy(load_environment_config(V39))
    config["shaping"] = {"mode": "potential", "gamma": .99}
    with pytest.raises(ValueError, match="omit PBRS"):
        load_environment_config(config)
    for field in ("angle_weight", "distance_weight", "speed_weight", "process_normalizer",
                  "gate_bonus", "dense_offset"):
        config = deepcopy(load_environment_config(V39))
        config["role_reward"]["uav"][field] = 1.0
        with pytest.raises(ValueError, match="frozen heterogeneous_role_coupled_gate_v1"):
            load_environment_config(config)


def test_v38_v39_config_diff_is_strictly_version_and_role_reward():
    old = deepcopy(load_environment_config(V38)); new = deepcopy(load_environment_config(V39))
    old.pop("environment_version"); new.pop("environment_version")
    old_role = old.pop("role_reward"); new_role = new.pop("role_reward")
    assert old == new
    assert old_role["target_selector"] == new_role["target_selector"]
    assert old_role["mav"] == new_role["mav"]


def test_angle_quality_and_existing_distance_primitive():
    assert uav_angle_quality(0.0, 0.0) == 1.0
    assert uav_angle_quality(np.pi, np.pi) == 0.0
    assert uav_distance_reward(500.0, 1000.0, 3000.0) == pytest.approx(np.exp(-.5))
    assert uav_distance_reward(1000.0, 1000.0, 3000.0) == 1.0
    assert uav_distance_reward(3000.0, 1000.0, 3000.0) == 1.0
    assert uav_distance_reward(6000.0, 1000.0, 3000.0) == pytest.approx(np.exp(-1.0))


@pytest.mark.parametrize("quality,distance,expected", [
    (1.0, 1.0, .5), (0.0, 1.0, -.5), (1.0, 0.0, -.5), (0.0, 0.0, -.5),
])
def test_dense_reward_exact_cases(quality, distance, expected):
    assert uav_coupled_dense_reward(quality, distance) == pytest.approx(expected)


def test_dense_reward_monotonicity_and_far_distance_lower_limit():
    grid = np.linspace(0.0, 1.0, 101)
    for distance in (.01, .25, .8, 1.0):
        values = [uav_coupled_dense_reward(quality, distance) for quality in grid]
        assert np.all(np.diff(values) >= -1e-15)
    for quality in (0.0, .25, .8, 1.0):
        values = [uav_coupled_dense_reward(quality, distance) for distance in grid]
        assert np.all(np.diff(values) >= -1e-15)
    tiny = uav_distance_reward(100000.0, 1000.0, 3000.0)
    assert tiny < 1e-10
    assert uav_coupled_dense_reward(1.0, tiny) == pytest.approx(-.5, abs=1e-10)


def test_exact_combat_gate_boundaries_and_fixed_reward():
    amin, amax = np.deg2rad(30.0), np.deg2rad(90.0)
    gate = lambda d, ata, aa: attack_gate_indicator(d, ata, aa, 1000.0, 3000.0, amin, amax)
    assert gate(1000.0, 0.0, 0.0) == 1.0
    assert gate(3000.0, 0.0, 0.0) == 1.0
    assert gate(999.999, 0.0, 0.0) == 0.0
    assert gate(3000.001, 0.0, 0.0) == 0.0
    assert gate(2000.0, amin, 0.0) == 0.0
    assert gate(2000.0, 0.0, amax) == 0.0
    assert uav_gate_reward(0.0) == 0.0
    assert uav_gate_reward(1.0) == .5
    assert uav_coupled_dense_reward(1.0, 1.0) + uav_gate_reward(1.0) == 1.0


def test_no_target_dead_uav_and_no_alive_blue_contract():
    env = _env(V39)
    for bid in BLUE_IDS:
        env.entities[bid].state.x = 50000.0
        env.entities[bid].state.y = 50000.0
    process, diag = env._role_process_rewards()
    for aid in RED_IDS[1:]:
        assert diag[f"reward_target_{aid}"] is None
        assert diag[f"{aid.lower()}_R_AD"] == -.5
        assert process[aid] == -.5
    env.entities["UAV1"].state.alive = False
    process, _ = env._role_process_rewards()
    assert process["UAV1"] == 0.0
    for bid in BLUE_IDS:
        env.entities[bid].state.alive = False
    process, _ = env._role_process_rewards()
    assert all(process[aid] == 0.0 for aid in RED_IDS[1:])


def test_rv_is_diagnostic_only_with_fixed_quality_distance_and_gate():
    active = uav_coupled_dense_reward(.7, .8) + uav_gate_reward(1.0)
    for blue_speed in (50.0, 200.0, 400.0):
        assert np.isfinite(uav_speed_reward(200.0, blue_speed))
        assert uav_coupled_dense_reward(.7, .8) + uav_gate_reward(1.0) == active


def test_mav_normalization_same_scale_for_one_and_four_identical_blue():
    one = _env(V39); four = _env(V39)
    for env in (one, four):
        mav = env.entities["MAV"].state
        template = env.entities["Blue1"].state
        template.x, template.y, template.h, template.psi = mav.x + 2000.0, mav.y, mav.h, np.pi
        for bid in BLUE_IDS:
            env.entities[bid].state = template.copy()
    for bid in BLUE_IDS[1:]:
        one.entities[bid].state.alive = False
    one_process, one_diag = one._role_process_rewards()
    four_process, four_diag = four._role_process_rewards()
    assert one_diag["mav_R_aspect"] == four_diag["mav_R_aspect"]
    assert one_diag["mav_R_aware"] == four_diag["mav_R_aware"]
    assert one_process["MAV"] == four_process["MAV"]
    assert four_diag["mav_R_aspect_raw_sum"] == 4.0 * one_diag["mav_R_aspect_raw_sum"]
    assert four_diag["mav_R_aware_raw_sum"] == 4.0 * one_diag["mav_R_aware_raw_sum"]


def test_invisible_alive_blue_has_zero_awareness_but_remains_in_denominator():
    env = _env(V39)
    mav = env.entities["MAV"].state
    env.entities["Blue1"].state.x, env.entities["Blue1"].state.y = mav.x + 2000.0, mav.y
    env.entities["Blue2"].state.x, env.entities["Blue2"].state.y = 50000.0, 50000.0
    env.entities["Blue3"].state.alive = env.entities["Blue4"].state.alive = False
    _, diag = env._role_process_rewards()
    assert diag["alive_blue_count"] == 2
    assert diag["team_visible_blue_count"] == 1
    assert diag["mav_R_aware"] == pytest.approx(diag["mav_R_aware_raw_sum"] / 2.0)


def test_v38_v39_fixed_step_invariants_and_expected_role_changes():
    old, new = _env(V38, 47), _env(V39, 47)
    old_obs, old_rewards, old_term, old_trunc, old_info = old.step(np.zeros((4, 3)))
    new_obs, new_rewards, new_term, new_trunc, new_info = new.step(np.zeros((4, 3)))
    for aid in ENTITY_IDS:
        assert np.array_equal(old.entities[aid].state.as_array(), new.entities[aid].state.as_array())
        assert old.entities[aid].state.alive == new.entities[aid].state.alive
    for bid in BLUE_IDS:
        assert old.team_visible(bid) == new.team_visible(bid)
    for aid in RED_IDS[1:]:
        prefix = aid.lower()
        assert old_info[f"reward_target_{aid}"] == new_info[f"reward_target_{aid}"]
        assert old_info[f"target_score_{aid}"] == new_info[f"target_score_{aid}"]
        for field in ("R_A", "R_D", "R_V"):
            assert old_info[f"{prefix}_{field}"] == new_info[f"{prefix}_{field}"]
    assert old_info["event_reward"] == new_info["event_reward"]
    assert old_info["terminal_reward"] == new_info["terminal_reward"]
    assert old_info["safety_reward"] == new_info["safety_reward"]
    assert old._attack_streak == new._attack_streak
    assert old_term == new_term and old_trunc == new_trunc
    for aid in RED_IDS:
        assert np.array_equal(old_obs[aid], new_obs[aid])
    assert np.array_equal(old.global_state(), new.global_state())
    assert any(not np.isclose(old_rewards[aid], new_rewards[aid]) for aid in RED_IDS)
    assert new_info["team_reward"] == pytest.approx(np.mean(list(new_rewards.values())))
    assert all(np.isfinite(value) for value in new_rewards.values())
    assert all(np.isfinite(value).all() and value.shape == (OBS_DIM,) for value in new_obs.values())
    assert np.isfinite(new.global_state()).all() and new.global_state().shape == (GLOBAL_STATE_DIM,)


def test_mav_threat_logic_matches_v38():
    old, new = _env(V38), _env(V39)
    for streak in (0, 1, 2):
        old._attack_streak[("Blue1", "MAV")] = streak
        new._attack_streak[("Blue1", "MAV")] = streak
        assert old._role_process_rewards()[1]["mav_R_threat"] == new._role_process_rewards()[1]["mav_R_threat"]


@pytest.mark.parametrize("outcome,terminal", [("red", 100.0), ("blue", -100.0), ("draw", 0.0)])
def test_shared_event_terminal_safety_match_v38(outcome, terminal, monkeypatch):
    old, new = _env(V38, 53), _env(V39, 53)
    for env in (old, new):
        monkeypatch.setattr(env, "_apply_boundaries", lambda: {})
        monkeypatch.setattr(env, "_resolve_attacks", lambda: ([], {"Blue1": "red_attack", "UAV1": "blue_attack"}))
        monkeypatch.setattr(env, "_termination", lambda: (True, False, outcome))
    _, _, _, _, old_info = old.step(np.zeros((4, 3)))
    obs, rewards, _, _, new_info = new.step(np.zeros((4, 3)))
    assert old_info["event_reward"] == new_info["event_reward"] == 90.0
    assert old_info["terminal_reward"] == new_info["terminal_reward"] == terminal
    assert old_info["safety_reward"] == new_info["safety_reward"]
    assert new_info["team_reward"] == pytest.approx(np.mean(list(rewards.values())))
    assert all(np.isfinite(value) for value in rewards.values())
    assert all(np.isfinite(value).all() for value in obs.values())


def test_rollout_buffer_remains_four_agent_mean():
    buffer = RolloutBuffer(1, 1)
    buffer.insert(np.zeros((1, 4, OBS_DIM)), np.zeros((1, GLOBAL_STATE_DIM)),
                  np.zeros((1, 4, 3)), np.zeros((1, 4)), np.array([[1., 2., 3., 6.]]),
                  np.zeros(1), np.zeros(1, bool), np.zeros(1, bool), np.ones((1, 4)))
    assert buffer.rewards[0, 0] == 3.0


def test_v39_checkpoint_metadata_resume_and_v38_cross_rejection(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": device}
    v39 = HAPPOTrainer(V39, config); same = HAPPOTrainer(V39, config); v38 = HAPPOTrainer(V38, config)
    try:
        v39.train_update()
        path39 = tmp_path / "v39.pt"; v39.save_checkpoint(path39)
        payload = torch.load(path39, map_location="cpu", weights_only=False)
        assert payload["environment_version"] == "heterogeneous_mavuav_4v4_v3_9"
        assert payload["reward_mode"] == "heterogeneous_role_coupled_gate_v1"
        assert payload["reward_shaping_mode"] is None and payload["shaping_gamma"] is None
        validate_checkpoint_contract(payload, v39.environment_config)
        assert same.load_checkpoint(path39) == 1
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v38.load_checkpoint(path39)
        path38 = tmp_path / "v38.pt"; v38.save_checkpoint(path38)
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v39.load_checkpoint(path38)
    finally:
        v39.close(); same.close(); v38.close()
