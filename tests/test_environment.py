from pathlib import Path
import numpy as np
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
    assert len(result) == 5 and all(value.shape == (10,) for value in result[0].values())
    assert result[1] == {"red_0": 0.0, "blue_0": 0.0}


def test_deterministic_and_synchronous():
    first, second = HomogeneousAirCombatEnv(CONFIG), HomogeneousAirCombatEnv(CONFIG)
    first.reset(7); second.reset(7)
    first.step(actions()); second.aircraft.reverse(); second.step(actions())
    states1 = {a.aircraft_id: a.state.as_array() for a in first.aircraft}
    states2 = {a.aircraft_id: a.state.as_array() for a in second.aircraft}
    assert all(np.allclose(states1[key], states2[key]) for key in states1)


def test_truncation_and_altitude_termination():
    env = HomogeneousAirCombatEnv(CONFIG); env.reset()
    env.config["simulation"]["max_steps"] = 1
    assert env.step(actions())[3]
    env.reset(); env.aircraft[0].state.z = -100
    assert env.step(actions())[2]

