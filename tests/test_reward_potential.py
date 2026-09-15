import numpy as np
import pytest

from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo import HAPPOTrainer
from algorithm.train_happo import _episode_metrics
from env import MAVUAVVectorEnv
from env.mavuav import BLUE_IDS, GLOBAL_STATE_DIM, HeterogeneousMAVUAVAirCombatEnv, OBS_DIM, RED_IDS
from env.reward import potential_shaping_reward


def test_potential_transition_values_and_terminal_zeroing():
    effective, shaping = potential_shaping_reward(0.3, 0.5, 0.99, False)
    assert effective == 0.5 and np.isclose(shaping, 0.195)
    assert np.isclose(potential_shaping_reward(0.8, 0.4, 0.99, False)[1], -0.404)
    assert np.isclose(potential_shaping_reward(0.57, 0.57, 0.99, False)[1], -0.0057)
    effective, shaping = potential_shaping_reward(0.57, 0.57, 0.99, True)
    assert effective == 0.0 and np.isclose(shaping, -0.57)


def test_discounted_multistep_potential_telescopes_to_negative_initial_potential():
    gamma = 0.99
    potentials = np.asarray([0.2, 0.35, 0.6, 0.8], dtype=float)
    shaping = [
        potential_shaping_reward(potentials[i], potentials[i + 1], gamma, False)[1]
        for i in range(len(potentials) - 1)
    ]
    shaping.append(potential_shaping_reward(potentials[-1], 0.8, gamma, True)[1])
    discounted = sum(gamma**i * value for i, value in enumerate(shaping))
    assert np.isclose(discounted, -potentials[0], atol=1e-12)


def test_v36_environment_uses_potential_reward_and_exposes_accounting():
    env = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
    observations, reset_info = env.reset(seed=123)
    assert observations["MAV"].shape == (100,)
    assert reset_info["profile"] == "main"
    _, rewards, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert not terminated and not truncated
    assert info["potential_terminal_zeroed"] is False
    assert np.isclose(
        info["team_reward"],
        info["potential_shaping_reward"] + info["event_reward"] + info["terminal_reward"] + info["safety_reward"],
    )
    assert np.isclose(rewards["MAV"], info["team_reward"])
    assert np.isfinite(list(rewards.values())).all()


def test_v35_default_remains_absolute_and_v36_is_explicit():
    old = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    old.reset(seed=7)
    _, _, _, _, old_info = old.step(np.zeros((4, 3)))
    assert old.config["environment_version"].endswith("v3_5")
    assert old.reward_mode == "absolute"
    assert old_info["potential_shaping_reward"] == 0.0

    new = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
    new.reset(seed=7)
    _, _, _, _, new_info = new.step(np.zeros((4, 3)))
    assert new.config["environment_version"].endswith("v3_6")
    assert new.reward_mode == "potential"


def test_potential_gamma_must_match_training_gamma():
    with pytest.raises(ValueError, match="must equal training gamma"):
        HAPPOTrainer(
            "configs/env_v36.yaml",
            {"training": {"gamma": 0.98, "num_envs": 1, "device": "cpu"}},
        )


def test_real_environment_terminal_and_truncation_zero_potential(monkeypatch):
    for terminal in ("red", "blue", "draw"):
        env = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
        env.reset(seed=10)
        if terminal == "red":
            env._red_attack_kills.update(BLUE_IDS)
            for aid in BLUE_IDS:
                env.entities[aid].state.alive = False
        elif terminal == "blue":
            env.entities["MAV"].state.alive = False
        else:
            env.step_count = 74
        _, _, terminated, truncated, info = env.step(np.zeros((len(RED_IDS), 3)))
        assert terminated or truncated
        assert info["potential_next_effective"] == 0.0
        assert info["potential_terminal_zeroed"] is True


def test_real_kill_and_loss_reward_accounting(monkeypatch):
    env = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
    env.reset(seed=11)
    def fake_attack():
        env.entities["Blue1"].state.alive = False
        return ([{"attacker": "MAV", "target": "Blue1"}], {"Blue1": "red_attack"})
    monkeypatch.setattr(env, "_resolve_attacks", fake_attack)
    _, _, _, _, info = env.step(np.zeros((len(RED_IDS), 3)))
    assert np.isclose(info["team_reward"], info["potential_shaping_reward"] + info["event_reward"] + info["terminal_reward"] + info["safety_reward"])
    assert info["event_reward"] == 50.0

    env = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
    env.reset(seed=12)
    def fake_uav_loss():
        env.entities["UAV1"].state.alive = False
        return {"UAV1": "boundary"}
    monkeypatch.setattr(env, "_apply_boundaries", fake_uav_loss)
    _, _, _, _, info = env.step(np.zeros((len(RED_IDS), 3)))
    assert info["event_reward"] == -10.0
    env = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False)
    env.reset(seed=13)
    def fake_mav_loss():
        env.entities["MAV"].state.alive = False
        return {"MAV": "boundary"}
    monkeypatch.setattr(env, "_apply_boundaries", fake_mav_loss)
    _, _, _, _, info = env.step(np.zeros((len(RED_IDS), 3)))
    assert info["event_reward"] == -100.0 and info["terminal_reward"] == -100.0


def test_vector_autoreset_does_not_leak_episode_accounting():
    config = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml", randomize=False).config
    config["simulation"]["max_decision_steps"] = 1
    vector = MAVUAVVectorEnv(1, config, parallel=False, profile="main")
    try:
        vector.reset(seed=20)
        _, _, _, _, truncated, _, infos = vector.step(np.zeros((1, len(RED_IDS), 3)))
        assert truncated[0] and infos[0]["auto_reset"]
        summary = infos[0]["episode_summary"]
        assert summary["potential_shaping_sum"] != 0.0
        state = vector.get_env_states()[0]
        assert state["episode_return"] == 0.0
        assert state["potential_shaping_sum"] == 0.0
    finally:
        vector.close()


def test_evaluator_contract_accepts_matching_versions_and_rejects_cross_version():
    old = HeterogeneousMAVUAVAirCombatEnv().config
    new = HeterogeneousMAVUAVAirCombatEnv("configs/env_v36.yaml").config
    old_payload = {"environment_version": old["environment_version"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "reward_shaping_mode": "absolute"}
    new_payload = {"environment_version": new["environment_version"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "reward_shaping_mode": "potential", "shaping_gamma": 0.99}
    validate_checkpoint_contract(old_payload, old)
    validate_checkpoint_contract(new_payload, new)
    with pytest.raises(RuntimeError): validate_checkpoint_contract(old_payload, new)
    with pytest.raises(RuntimeError): validate_checkpoint_contract(new_payload, old)


def test_training_episode_metrics_persist_r1_reward_decomposition():
    row = {
        "outcome": "draw", "episode_return": 1.0, "mav_survived": True,
        "red_uav_survivors": 3, "red_attack_kills": 0, "blue_attack_kills": 0,
        "episode_length": 75, "potential_shaping_sum": 0.2,
        "absolute_situation_sum": 10.0, "event_reward_sum": 0.0,
        "terminal_reward_sum": 0.0, "safety_reward_sum": -1.0,
    }
    metrics = _episode_metrics([row])
    assert metrics["mean_potential_shaping_sum"] == 0.2
    assert metrics["mean_absolute_situation_sum"] == 10.0
    assert metrics["mean_safety_reward_sum"] == -1.0


def test_checkpoint_versions_and_reward_modes_are_not_cross_compatible(tmp_path):
    v35 = HAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": "cpu"})
    v35_path = tmp_path / "v35.pt"
    v35.save_checkpoint(v35_path)
    v36 = HAPPOTrainer("configs/env_v36.yaml", {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": "cpu"})
    v36_path = tmp_path / "v36.pt"
    v36.save_checkpoint(v36_path)
    try:
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v36.load_checkpoint(v35_path)
        with pytest.raises(RuntimeError, match="incompatible checkpoint contract"):
            v35.load_checkpoint(v36_path)
        restored = HAPPOTrainer("configs/env_v36.yaml", {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": "cpu"})
        try:
            assert restored.load_checkpoint(v36_path) == 0
        finally:
            restored.close()
    finally:
        v35.close()
        v36.close()
