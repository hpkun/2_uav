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


def test_fixed_tail_chase_forces_red_to_rear():
    for seed in range(20):
        env=HomogeneousAirCombatEnv(CONFIG); env.reset(seed,"tail_chase",rear_team="red"); red,blue=env.aircraft
        assert np.dot(red.state.velocity_vector()[:2],blue.state.as_array()[:2]-red.state.as_array()[:2])>0


def test_axis_specific_observation_normalization_does_not_saturate_early():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset(scenario_name="fixed")
    red, blue = env.aircraft
    limit = env.config["battlefield"]["x_limit"]
    red.state.x, blue.state.x = -0.9 * limit, 0.9 * limit
    red.state.y = blue.state.y = 0.0; red.state.z = blue.state.z = -3000.0
    observations = env._observations()
    assert np.isclose(observations["red_0"][3], 0.9)
    assert all(value.shape == (14,) and np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1) for value in observations.values())


def test_ego_observation_rotation_invariance_altitude_and_global_state():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset(8, "tail_chase")
    before = env._observations()["red_0"].copy(); angle = 0.7
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for aircraft in env.aircraft:
        aircraft.state.x, aircraft.state.y = matrix @ np.array([aircraft.state.x, aircraft.state.y])
        aircraft.state.psi += angle
    after = env._observations()["red_0"]
    assert np.allclose(before, after, atol=1e-12)
    red = env.aircraft[0]; red.state.z = -env.config["battlefield"]["altitude_min"]
    assert np.isclose(env._observations()["red_0"][2], -1)
    red.state.z = -env.config["battlefield"]["altitude_max"]
    assert np.isclose(env._observations()["red_0"][2], 1)
    global_state = env.global_state("red")
    assert global_state.shape == (14,) and np.all(np.isfinite(global_state)) and np.all(np.abs(global_state) <= 1)


def test_tail_chase_rear_aircraft_has_speed_advantage_on_average():
    differences = []
    for seed in range(30):
        env = HomogeneousAirCombatEnv(CONFIG); env.reset(seed, "tail_chase"); red, blue = env.aircraft
        red_is_rear = np.dot(red.state.velocity_vector()[:2], blue.state.as_array()[:2] - red.state.as_array()[:2]) > 0
        differences.append((red.state.v - blue.state.v) if red_is_rear else (blue.state.v - red.state.v))
    assert np.mean(differences) > 15


def test_dense_reward_scale_is_below_terminal_event():
    env = HomogeneousAirCombatEnv(CONFIG)
    assert env.config["simulation"]["max_steps"] * env.config["combat"]["situation_reward_scale"] < env.config["combat"]["terminal_reward"]

def test_global_state_is_own_opponent_ordered_and_label_swap_invariant():
    env=HomogeneousAirCombatEnv(CONFIG);env.reset(17,"offset_head_on")
    red_view=env.global_state("red");blue_view=env.global_state("blue")
    assert np.allclose(red_view[:7],blue_view[7:]) and np.allclose(red_view[7:],blue_view[:7])
    red,blue=env.aircraft
    red.state,blue.state=blue.state,red.state
    assert np.allclose(red_view,env.global_state("blue"))
    for view in (red_view,blue_view):
        assert view.shape==(14,) and np.isfinite(view).all() and np.all(np.abs(view)<=1)
