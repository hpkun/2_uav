import numpy as np
import pytest

from algorithm.happo import HAPPOTrainer
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv
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
