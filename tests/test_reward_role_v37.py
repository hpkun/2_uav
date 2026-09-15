"""Frozen v3.7 role reward, accounting and historical-mode regression tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from algorithm.common.buffer import RolloutBuffer
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.train_happo import _episode_metrics
from env.vector_env import MAVUAVVectorEnv

from env.mavuav import (
    BLUE_IDS, RED_IDS, OBS_DIM, GLOBAL_STATE_DIM,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.models import AircraftState
from env.reward_role_v37 import (
    target_score, uav_angle_reward, uav_speed_reward, uav_distance_reward,
    uav_process_reward, mav_aspect_reward, mav_awareness_reward,
)
from env.vector_env import _environment_state, _restore_environment_state

ROOT = Path(__file__).resolve().parents[1]
V37 = ROOT / "configs" / "env_v37.yaml"


def test_three_version_contracts_and_forbidden_shaping():
    assert load_environment_config(None)["environment_version"].endswith("v3_5")
    assert load_environment_config(ROOT / "configs" / "env_v36.yaml")["shaping"]["mode"] == "potential"
    cfg = load_environment_config(V37)
    assert cfg["environment_version"].endswith("v3_7")
    assert cfg["role_reward"]["mode"] == "heterogeneous_role_v1"
    with pytest.raises(ValueError, match="omit PBRS"):
        modified = deepcopy(cfg); modified["shaping"] = {"mode": "potential", "gamma": .99}
        load_environment_config(modified)


def test_target_score_formula_and_speed_direction():
    red = AircraftState(0, 0, 5000, 275, 0, 0)
    blue = AircraftState(2000, 0, 4500, 250, 0, np.pi)
    score = target_score(red, blue, 10000, 800, 3000)
    from env.geometry import compute_pairwise_geometry
    g = compute_pairwise_geometry(red, blue)
    expected = .35 * (1 - (g.ata + g.aa) / (2 * np.pi)) + .25 + .20 * (500 / 10000) + .20 * np.linalg.norm(red.velocity_vector() - blue.velocity_vector()) / 800
    assert score == pytest.approx(expected)
    assert uav_speed_reward(200, 50) == 1
    assert uav_speed_reward(200, 200) == 0
    assert uav_speed_reward(200, 400) == -1


def test_uav_piecewise_primitives_and_process():
    assert uav_angle_reward(0, 0) == 1
    assert uav_angle_reward(np.pi / 2, np.pi / 2) == pytest.approx(0)
    assert uav_angle_reward(np.pi, np.pi) == -1
    assert uav_distance_reward(500, 1000, 3000) == pytest.approx(np.exp(-.5))
    assert uav_distance_reward(1000, 1000, 3000) == 1
    assert uav_distance_reward(3000, 1000, 3000) == 1
    assert uav_distance_reward(6000, 1000, 3000) == pytest.approx(np.exp(-1))
    assert uav_process_reward(.2, .8, -.1) == pytest.approx((15*.2 + 10*.8 - 1)/35)


def test_mav_aspect_awareness_thresholds():
    assert mav_aspect_reward(0) == -1
    assert mav_aspect_reward(np.pi/4) == 0
    assert mav_awareness_reward(0) == .3
    assert mav_awareness_reward(np.pi/2) == 0


def test_selector_tie_visibility_none_and_accounting():
    env = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    env.reset(seed=7)
    for bid in BLUE_IDS:
        env.entities[bid].state = env.entities["Blue1"].state.copy()
    process, diag = env._role_process_rewards()
    assert diag["reward_target_UAV1"] == "Blue1"
    assert diag["target_score_UAV1"] is not None
    env._reward_target_previous["UAV1"] = "Blue4"
    _, diag = env._role_process_rewards()
    assert diag["reward_target_switch_UAV1"] is True
    assert env._reward_target_switches["UAV1"] == 1
    for bid in BLUE_IDS:
        env.entities[bid].state.x = 50000
    process, diag = env._role_process_rewards()
    assert diag["reward_target_UAV1"] is None
    assert process["UAV1"] == 0
    env.reset(seed=8)
    obs, rewards, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert set(rewards) == set(RED_IDS)
    assert info["team_reward"] == pytest.approx(np.mean(list(rewards.values())))
    role_mean = np.mean([info["mav_process_reward"], *(info[f"{aid.lower()}_process_reward"] for aid in RED_IDS[1:])])
    assert info["team_reward"] == pytest.approx(info["event_reward"] + info["terminal_reward"] + info["safety_reward"] + role_mean)
    assert all(np.isfinite(v) for v in rewards.values())
    assert all(np.isfinite(v).all() and v.shape == (OBS_DIM,) for v in obs.values())
    assert env.global_state().shape == (GLOBAL_STATE_DIM,)


def test_mav_threat_only_true_streak():
    env = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    env.reset(seed=9)
    env._attack_streak[("Blue1", "MAV")] = 0
    assert env._role_process_rewards()[1]["mav_R_threat"] == 0
    env._attack_streak[("Blue1", "MAV")] = 1
    assert env._role_process_rewards()[1]["mav_R_threat"] == -1


def test_vector_state_exact_target_and_reset():
    env = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    env.reset(seed=10)
    env._reward_target_previous["UAV1"] = "Blue3"
    env._reward_target_switches["UAV1"] = 7
    env._role_process_sums["UAV1"] = 2.5
    state = _environment_state(env)
    other = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    other.reset(seed=11)
    _restore_environment_state(other, state)
    assert other._reward_target_previous == env._reward_target_previous
    assert other._reward_target_switches == env._reward_target_switches
    assert other._role_process_sums == env._role_process_sums
    other.reset(seed=12)
    assert other._reward_target_previous["UAV1"] is None
    assert other._reward_target_switches["UAV1"] == 0
    assert other._role_process_sums["UAV1"] == 0


def test_shared_events_terminal_and_distinct_agent_rewards(monkeypatch):
    env = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    env.reset(seed=14)
    monkeypatch.setattr(env, "_apply_boundaries", lambda: {})
    monkeypatch.setattr(env, "_resolve_attacks", lambda: ([], {"Blue1": "red_attack", "UAV1": "blue_attack"}))
    monkeypatch.setattr(env, "_termination", lambda: (False, False, None))
    _, rewards, _, _, info = env.step(np.zeros((4, 3)))
    assert info["event_reward"] == 90.0
    assert len(set(round(value, 8) for value in rewards.values())) > 1
    assert info["team_reward"] == pytest.approx(np.mean(list(rewards.values())))
    assert env.episode_return == pytest.approx(info["team_reward"])
    env.reset(seed=15)
    def kill_mav():
        env.entities["MAV"].state.alive = False
        return [], {"MAV": "blue_attack"}
    monkeypatch.setattr(env, "_resolve_attacks", kill_mav)
    monkeypatch.setattr(env, "_termination", lambda: (True, False, "blue"))
    _, rewards, _, _, info = env.step(np.zeros((4, 3)))
    assert info["event_reward"] == -100.0
    assert info["terminal_reward"] == -100.0
    assert info["mav_process_reward"] == 0.0
    assert info["episode_summary"]["team_reward_sum"] == pytest.approx(env.episode_return)
    assert env.config["reward"]["terminal_red_win"] == 100.0
    assert env.config["reward"]["terminal_draw"] == 0.0


@pytest.mark.parametrize("outcome, expected", [("red", 100.0), ("blue", -100.0), ("draw", 0.0)])
def test_terminal_reward_contract(outcome, expected, monkeypatch):
    env = HeterogeneousMAVUAVAirCombatEnv(V37, randomize=False)
    env.reset(seed=22)
    monkeypatch.setattr(env, "_apply_boundaries", lambda: {})
    monkeypatch.setattr(env, "_resolve_attacks", lambda: ([], {}))
    monkeypatch.setattr(env, "_termination", lambda: (True, False, outcome))
    _, _, _, _, info = env.step(np.zeros((4, 3)))
    assert info["terminal_reward"] == expected


def test_buffer_training_and_evaluation_accounting_fields():
    buffer = RolloutBuffer(1, 1)
    buffer.insert(np.zeros((1, 4, OBS_DIM)), np.zeros((1, GLOBAL_STATE_DIM)),
                  np.zeros((1, 4, 3)), np.zeros((1, 4)), np.array([[1., 2., 3., 6.]]),
                  np.zeros(1), np.zeros(1, bool), np.zeros(1, bool), np.ones((1, 4)))
    assert buffer.rewards[0, 0] == 3.0
    metrics = _episode_metrics([{"outcome": "draw", "episode_return": 4.0,
                                 "mav_survived": True, "red_uav_survivors": 3,
                                 "red_attack_kills": 0, "blue_attack_kills": 0,
                                 "episode_length": 75,
                                 "mav_process_reward_sum": 2.0, "mean_uav_process_reward_sum": 1.0,
                                 "shared_event_reward_sum": 3.0, "shared_terminal_reward_sum": 0.0,
                                 "shared_safety_reward_sum": -1.0}])
    assert metrics["mean_mav_process_reward_sum"] == 2.0
    assert metrics["mean_uav_process_reward_sum"] == 1.0
    assert metrics["mean_shared_event_reward_sum"] == 3.0


def test_v37_vector_auto_reset_clears_role_state():
    config = deepcopy(load_environment_config(V37))
    config["simulation"]["max_decision_steps"] = 1
    vector = MAVUAVVectorEnv(1, config, parallel=False, profile="main")
    try:
        vector.reset(seed=20)
        _, _, rewards, _, truncated, _, infos = vector.step(np.zeros((1, 4, 3)))
        assert truncated[0] and infos[0]["auto_reset"]
        assert rewards.shape == (1, 4)
        assert all(np.isfinite(rewards[0]))
        summary = infos[0]["episode_summary"]
        assert summary["reward_mode"] == "heterogeneous_role_v1"
        state = vector.get_env_states()[0]
        assert state["episode_return"] == 0.0
        assert state["role_process_sums"] == {aid: 0.0 for aid in RED_IDS}
        assert state["reward_target_previous"] == {aid: None for aid in RED_IDS[1:]}
    finally:
        vector.close()


def test_v37_checkpoint_resume_reward_contract_and_legacy_mode(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": device}
    source = HAPPOTrainer(V37, config)
    target = HAPPOTrainer(V37, config)
    legacy = HAPPOTrainer(config=config)
    try:
        source.train_update()
        assert source.env_steps == 1
        source.vector_env.set_env_states([{
            **source.vector_env.get_env_states()[0],
            "reward_target_previous": {"UAV1": "Blue2", "UAV2": None, "UAV3": "Blue1"},
            "reward_target_switches": {"UAV1": 3, "UAV2": 0, "UAV3": 2},
            "role_process_sums": {"MAV": 1.0, "UAV1": 2.0, "UAV2": 3.0, "UAV3": 4.0},
        }], source.vector_env.reset_counts, source.vector_env.base_seed)
        path = tmp_path / "v37.pt"
        source.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["reward_mode"] == "heterogeneous_role_v1"
        assert payload["environment_version"] == load_environment_config(V37)["environment_version"]
        validate_checkpoint_contract(payload, source.environment_config)
        assert target.load_checkpoint(path) == source.env_steps
        assert target.vector_env.get_env_states() == source.vector_env.get_env_states()
        _, metrics = target.train_update()
        assert target.env_steps == 2
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)))
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            legacy.load_checkpoint(path)
        old = legacy.checkpoint_state()
        old.pop("reward_mode")
        validate_checkpoint_contract(old, legacy.environment_config)
    finally:
        source.close(); target.close(); legacy.close()
