import numpy as np

from uav_combat.models import Aircraft, AircraftState
from uav_combat.environment import HomogeneousAirCombatEnv
from uav_combat.rule_policy import PurePursuitPolicy


def aircraft(identifier, spec, x=0, y=0, z=-3000, v=150):
    return Aircraft(identifier, identifier.split("_")[0], spec, AircraftState(x, y, z, v, 0, 0))


def test_pure_pursuit_action_directions_and_speed(spec):
    own = aircraft("red_0", spec)
    policy = PurePursuitPolicy(np.pi, np.pi / 3, 50)
    assert policy.action(own, aircraft("blue_0", spec, x=500, y=100))[0] > 0
    assert policy.action(own, aircraft("blue_0", spec, x=500, y=-100))[0] < 0
    assert policy.action(own, aircraft("blue_0", spec, x=500, z=-3100))[1] > 0
    action = policy.action(own, aircraft("blue_0", spec, x=500, v=140))
    assert np.isclose(action[2], 0.2)


def test_tail_chase_demo_scenario_ends_in_red_victory():
    from pathlib import Path

    config = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"
    env = HomogeneousAirCombatEnv(config); env.reset()
    red, blue = env.aircraft
    red.state = AircraftState(0, 0, -3000, 170, 0, 0)
    blue.state = AircraftState(1500, 0, -3000, 150, 0, 0)
    action_config = env.config["action"]
    policy = PurePursuitPolicy(action_config["delta_yaw_max"], action_config["delta_pitch_max"], action_config["delta_speed_max"])
    info = {}
    for _ in range(300):
        _, _, terminated, truncated, info = env.step({"red_0": policy.action(red, blue), "blue_0": np.zeros(3)})
        if terminated or truncated:
            break
    assert info["outcome"] == "red" and info["termination_reason"] == "red_kill"
