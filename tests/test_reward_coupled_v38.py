"""Frozen single-variable regression tests for v3.8 coupled UAV reward."""
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
    BLUE_IDS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.reward_role_v37 import (
    target_score, uav_angle_reward, uav_distance_reward, uav_speed_reward,
)
from env.reward_role_v38 import uav_coupled_process_reward


ROOT = Path(__file__).resolve().parents[1]
V37 = ROOT / "configs" / "env_v37.yaml"
V38 = ROOT / "configs" / "env_v38.yaml"


def _paired_envs(seed: int = 23):
    old = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    new = HeterogeneousMAVUAVAirCombatEnv(V38, randomize=False)
    old.reset(seed=seed); new.reset(seed=seed)
    return old, new


def test_v38_config_reward_mode_and_historical_contracts():
    assert load_environment_config(None)["environment_version"].endswith("v3_5")
    assert load_environment_config(ROOT / "configs" / "env_v36.yaml")["shaping"]["mode"] == "potential"
    assert load_environment_config(V37)["role_reward"]["mode"] == "heterogeneous_role_v1"
    config = load_environment_config(V38)
    assert config["environment_version"] == "heterogeneous_mavuav_4v4_v3_8"
    assert config["role_reward"]["mode"] == "heterogeneous_role_coupled_v1"
    assert config["role_reward"]["uav"] == {"process_mode": "angle_distance_product"}
    assert "shaping" not in config
    assert HeterogeneousMAVUAVAirCombatEnv(V38).reward_mode == "heterogeneous_role_coupled_v1"


def test_v37_v38_configs_differ_only_in_frozen_uav_process_contract():
    old = deepcopy(load_environment_config(V37))
    new = deepcopy(load_environment_config(V38))
    old.pop("environment_version"); new.pop("environment_version")
    old_role = old.pop("role_reward"); new_role = new.pop("role_reward")
    assert old == new
    assert old_role["target_selector"] == new_role["target_selector"]
    assert old_role["mav"] == new_role["mav"]
    assert old_role["mode"] == "heterogeneous_role_v1"
    assert new_role["mode"] == "heterogeneous_role_coupled_v1"
    assert old_role["uav"] == {
        "angle_weight": 15.0, "distance_weight": 10.0,
        "speed_weight": 10.0, "process_normalizer": 35.0,
    }
    assert new_role["uav"] == {"process_mode": "angle_distance_product"}


def test_v38_rejects_pbrs_and_obsolete_additive_parameters():
    config = deepcopy(load_environment_config(V38))
    config["shaping"] = {"mode": "potential", "gamma": .99}
    with pytest.raises(ValueError, match="omit PBRS"):
        load_environment_config(config)
    config = deepcopy(load_environment_config(V38))
    config["role_reward"]["uav"]["speed_weight"] = 10.0
    with pytest.raises(ValueError, match="frozen heterogeneous_role_coupled_v1"):
        load_environment_config(config)


@pytest.mark.parametrize("angle,distance,expected", [
    (1.0, 1.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 1.0, -1.0),
    (-.5, .8, -.4), (.8, .2, .16),
])
def test_coupled_process_exact_cases(angle, distance, expected):
    assert uav_coupled_process_reward(angle, distance) == pytest.approx(expected)


def test_rv_is_diagnostic_only_for_active_process():
    angle, distance = -.35, .8
    first = uav_coupled_process_reward(angle, distance)
    for red_speed, blue_speed in ((200, 50), (200, 200), (200, 400)):
        assert np.isfinite(uav_speed_reward(red_speed, blue_speed))
        assert uav_coupled_process_reward(angle, distance) == first


def test_cross_version_fixed_state_only_uav_process_changes():
    old, new = _paired_envs()
    old_process, old_diag = old._role_process_rewards()
    new_process, new_diag = new._role_process_rewards()
    assert old_diag["mav_process_reward"] == new_diag["mav_process_reward"]
    assert old_process["MAV"] == new_process["MAV"]
    changed = []
    for aid in RED_IDS[1:]:
        assert old_diag[f"reward_target_{aid}"] == new_diag[f"reward_target_{aid}"]
        assert old_diag[f"target_score_{aid}"] == new_diag[f"target_score_{aid}"]
        prefix = aid.lower()
        for field in ("R_A", "R_D", "R_V"):
            assert old_diag[f"{prefix}_{field}"] == new_diag[f"{prefix}_{field}"]
        assert new_process[aid] == pytest.approx(
            new_diag[f"{prefix}_R_A"] * new_diag[f"{prefix}_R_D"]
        )
        changed.append(not np.isclose(old_process[aid], new_process[aid], rtol=0.0, atol=1e-15))
    assert any(changed)


def test_v37_v38_primitives_and_target_selector_identical():
    old, new = _paired_envs(29)
    normalization = old.config["normalization"]
    maximum_range = old.config["combat"]["distance"][1]
    for red_id in RED_IDS[1:]:
        red_old, red_new = old.entities[red_id].state, new.entities[red_id].state
        for blue_id in BLUE_IDS:
            blue_old, blue_new = old.entities[blue_id].state, new.entities[blue_id].state
            assert target_score(red_old, blue_old, normalization["relative_altitude_scale"],
                                normalization["relative_velocity_scale"], maximum_range) == target_score(
                                    red_new, blue_new, normalization["relative_altitude_scale"],
                                    normalization["relative_velocity_scale"], maximum_range)
    assert uav_angle_reward(.4, .7) == uav_angle_reward(.4, .7)
    assert uav_distance_reward(2200, 1000, 3000) == uav_distance_reward(2200, 1000, 3000)
    assert uav_speed_reward(275, 240) == uav_speed_reward(275, 240)


@pytest.mark.parametrize("outcome,terminal", [("red", 100.0), ("blue", -100.0), ("draw", 0.0)])
def test_mav_shared_terminal_accounting_and_finite_outputs_match(outcome, terminal, monkeypatch):
    old, new = _paired_envs(31)
    for env in (old, new):
        monkeypatch.setattr(env, "_apply_boundaries", lambda: {})
        monkeypatch.setattr(env, "_resolve_attacks", lambda: ([], {"Blue1": "red_attack", "UAV1": "blue_attack"}))
        monkeypatch.setattr(env, "_termination", lambda: (True, False, outcome))
    old_obs, old_rewards, _, _, old_info = old.step(np.zeros((4, 3)))
    new_obs, new_rewards, _, _, new_info = new.step(np.zeros((4, 3)))
    assert old_info["event_reward"] == new_info["event_reward"] == 90.0
    assert old_info["terminal_reward"] == new_info["terminal_reward"] == terminal
    assert old_info["safety_reward"] == new_info["safety_reward"]
    assert old_info["mav_process_reward"] == new_info["mav_process_reward"]
    assert new_info["team_reward"] == pytest.approx(np.mean(list(new_rewards.values())))
    assert new_info["team_reward"] == pytest.approx(
        new_info["event_reward"] + new_info["terminal_reward"] + new_info["safety_reward"]
        + np.mean([new_info["mav_process_reward"], *[
            new_info[f"{aid.lower()}_process_reward"] for aid in RED_IDS[1:]
        ]])
    )
    assert all(np.isfinite(value) for value in new_rewards.values())
    assert all(value.shape == (OBS_DIM,) and np.isfinite(value).all() for value in new_obs.values())
    assert new.global_state().shape == (GLOBAL_STATE_DIM,) and np.isfinite(new.global_state()).all()


def test_rollout_buffer_still_uses_four_agent_mean():
    buffer = RolloutBuffer(1, 1)
    buffer.insert(np.zeros((1, 4, OBS_DIM)), np.zeros((1, GLOBAL_STATE_DIM)),
                  np.zeros((1, 4, 3)), np.zeros((1, 4)), np.array([[1., 2., 3., 6.]]),
                  np.zeros(1), np.zeros(1, bool), np.zeros(1, bool), np.ones((1, 4)))
    assert buffer.rewards[0, 0] == 3.0


def test_v38_checkpoint_metadata_resume_and_cross_version_rejection(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": device}
    v38 = HAPPOTrainer(V38, config)
    same = HAPPOTrainer(V38, config)
    v37 = HAPPOTrainer(V37, config)
    try:
        v38.train_update()
        path38 = tmp_path / "v38.pt"
        v38.save_checkpoint(path38)
        payload = torch.load(path38, map_location="cpu", weights_only=False)
        assert payload["environment_version"] == "heterogeneous_mavuav_4v4_v3_8"
        assert payload["reward_mode"] == "heterogeneous_role_coupled_v1"
        assert payload["reward_shaping_mode"] is None and payload["shaping_gamma"] is None
        validate_checkpoint_contract(payload, v38.environment_config)
        assert same.load_checkpoint(path38) == 1
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v37.load_checkpoint(path38)
        path37 = tmp_path / "v37.pt"
        v37.save_checkpoint(path37)
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v38.load_checkpoint(path37)
    finally:
        v38.close(); same.close(); v37.close()
