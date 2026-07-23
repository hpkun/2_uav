from pathlib import Path
import numpy as np
from uav_combat.environment import HomogeneousAirCombatEnv

CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def states(env):
    return np.asarray([aircraft.state.as_array() for aircraft in env.aircraft])


def test_scenarios_are_reproducible_variable_and_valid():
    for name in ("tail_chase", "offset_head_on", "crossing"):
        first, second, third = HomogeneousAirCombatEnv(CONFIG), HomogeneousAirCombatEnv(CONFIG), HomogeneousAirCombatEnv(CONFIG)
        _, info = first.reset(123, name); second.reset(123, name); third.reset(124, name)
        assert info["scenario_name"] == name
        assert np.array_equal(states(first), states(second))
        assert not np.array_equal(states(first), states(third))
        geometry = first._geometries()["red_0"]
        assert geometry.distance > first.config["combat"]["attack_distance_max"]
        assert geometry.distance > first.config["battlefield"]["collision_distance"]
        for aircraft in first.aircraft:
            assert aircraft.spec.v_min <= aircraft.state.v <= aircraft.spec.v_max
            assert first.config["battlefield"]["altitude_min"] <= aircraft.state.altitude <= first.config["battlefield"]["altitude_max"]
            assert aircraft.spec.theta_min <= aircraft.state.theta <= aircraft.spec.theta_max


def test_tail_chase_roles_are_randomized():
    rear_teams = set()
    for seed in range(20):
        env = HomogeneousAirCombatEnv(CONFIG); env.reset(seed, "tail_chase")
        red, blue = env.aircraft
        relative = blue.state.as_array()[:2] - red.state.as_array()[:2]
        rear_teams.add("red" if np.dot(red.state.velocity_vector()[:2], relative) > 0 else "blue")
    assert rear_teams == {"red", "blue"}


def test_axis_specific_observation_normalization_does_not_saturate_early():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset(scenario_name="fixed")
    red, blue = env.aircraft
    limit = env.config["battlefield"]["x_limit"]
    red.state.x, blue.state.x = -0.9 * limit, 0.9 * limit
    red.state.y = blue.state.y = 0.0; red.state.z = blue.state.z = -3000.0
    observations = env._observations()
    assert np.isclose(observations["red_0"][4], 0.9)
    assert all(value.shape == (13,) and np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1) for value in observations.values())


def test_dense_reward_scale_is_below_terminal_event():
    env = HomogeneousAirCombatEnv(CONFIG)
    assert env.config["simulation"]["max_steps"] * env.config["combat"]["situation_reward_scale"] < env.config["combat"]["terminal_reward"]

