from pathlib import Path
import numpy as np
import pytest
from uav_combat.environment import HomogeneousAirCombatEnv

CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def actions():
    return {"red_0": np.zeros(3), "blue_0": np.zeros(3)}


def test_reset_and_step_structure():
    env = HomogeneousAirCombatEnv(CONFIG)
    observations, _ = env.reset(seed=4)
    assert set(observations) == {"red_0", "blue_0"}
    assert env.aircraft[0].spec is env.aircraft[1].spec
    result = env.step(actions())
    assert len(result) == 5 and all(value.shape == (14,) for value in result[0].values())
    assert all(np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1) for value in result[0].values())
    assert np.isclose(result[1]["red_0"], -result[1]["blue_0"])


def test_deterministic_and_synchronous():
    first, second = HomogeneousAirCombatEnv(CONFIG), HomogeneousAirCombatEnv(CONFIG)
    first.reset(7); second.reset(7)
    first.step(actions()); second.aircraft.reverse(); second.step(actions())
    states1 = {a.aircraft_id: a.state.as_array() for a in first.aircraft}
    states2 = {a.aircraft_id: a.state.as_array() for a in second.aircraft}
    assert all(np.allclose(states1[key], states2[key]) for key in states1)


def test_truncation_and_altitude_termination():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset(scenario_name="fixed")
    env.config["simulation"]["max_steps"] = 1
    assert env.step(actions())[3]
    env.reset(scenario_name="fixed"); env.aircraft[0].state.z = -100
    assert env.step(actions())[2]


def test_step_requires_active_episode_and_reset_restores_it():
    env = HomogeneousAirCombatEnv(CONFIG)
    with pytest.raises(RuntimeError):
        env.step(actions())
    _, info = env.reset()
    assert info["termination_reason"] is None
    env.config["simulation"]["max_steps"] = 1
    env.step(actions())
    with pytest.raises(RuntimeError):
        env.step(actions())
    env.reset()
    env.config["simulation"]["max_steps"] = 600
    assert env.step(actions())[2:4] == (False, False)


def test_xy_boundary_and_collision_termination():
    env = HomogeneousAirCombatEnv(CONFIG)
    env.reset(scenario_name="fixed"); env.aircraft[0].state.x = env.config["battlefield"]["x_limit"] + 1
    result = env.step(actions())
    assert result[2] and result[4]["termination_reason"] == "xy_boundary"
    env.reset(scenario_name="fixed")
    env.aircraft[1].state.x = env.aircraft[0].state.x + 10
    env.aircraft[1].state.y = env.aircraft[0].state.y
    env.aircraft[1].state.z = env.aircraft[0].state.z
    env.aircraft[1].state.psi = env.aircraft[0].state.psi
    result = env.step(actions())
    assert result[2] and result[4]["termination_reason"] == "collision"
    assert result[1]["red_0"] < 0 and result[1]["blue_0"] < 0


def test_single_sided_attacks_and_rewards():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset(scenario_name="fixed")
    red, blue = env.aircraft
    red.state.x, red.state.psi = 0, 0
    blue.state.x, blue.state.psi = 500, 0
    observations, rewards, terminated, _, info = env.step(actions())
    assert terminated and info["outcome"] == "red" and info["termination_reason"] == "red_kill"
    assert red.state.alive and not blue.state.alive
    assert rewards["red_0"] > 0 and rewards["blue_0"] < 0
    assert all(value.shape == (14,) for value in observations.values())

    env.reset(scenario_name="fixed"); red, blue = env.aircraft
    red.state.x, red.state.psi = 500, 0
    blue.state.x, blue.state.psi = 0, 0
    _, rewards, terminated, _, info = env.step(actions())
    assert terminated and info["outcome"] == "blue" and info["termination_reason"] == "blue_kill"
    assert not red.state.alive and blue.state.alive
    assert rewards["red_0"] < 0 and rewards["blue_0"] > 0
