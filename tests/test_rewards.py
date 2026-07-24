import numpy as np
from uav_combat.models import AircraftState
from uav_combat.rewards import coupled_difference_rewards, madsac_segmented_reward


def state(x=0, z=-3000, psi=0, theta=0): return AircraftState(x,0,z,150,theta,psi)


def test_madsac_position_angle_levels_and_distance_boundary():
    target=state(x=1000)
    assert madsac_segmented_reward(state(psi=np.deg2rad(-5)),target,"red",None,None)["reward_position"]==0.1
    assert madsac_segmented_reward(state(psi=np.deg2rad(-15)),target,"red",None,None)["reward_position"]==0.02
    assert madsac_segmented_reward(state(psi=np.deg2rad(-30)),target,"red",None,None)["reward_position"]==0.01
    boundary=madsac_segmented_reward(state(),state(x=4000),"red",None,None)
    assert boundary["reward_guide"]==0.001 and boundary["reward_position"]==0.1


def test_madsac_dense_r3_r4_values_are_unchanged():
    guide = madsac_segmented_reward(state(), state(x=5000), "red", None, None)
    close = madsac_segmented_reward(state(), state(x=1000), "red", None, None)
    assert guide["reward_guide"] == 0.001
    assert close["reward_position"] == 0.1
    assert close["reward_threat"] == 0.0


def terminal(team, reason, outcome):
    result = madsac_segmented_reward(state(), state(x=2000), team, reason, outcome)
    return result["reward_terminal"], result["reward_boundary"]


def test_terminal_kill_boundary_collision_and_mutual_kill_semantics():
    assert terminal("red", "red_kill", "red")[0] == 10
    assert terminal("blue", "red_kill", "red")[0] == -10
    assert terminal("red", "blue_kill", "blue")[0] == -10
    assert terminal("blue", "blue_kill", "blue")[0] == 10
    assert terminal("red", "xy_boundary", "blue")[1] == -10
    assert terminal("blue", "xy_boundary", "blue")[1] == 0
    assert terminal("red", "altitude_boundary", "red")[1] == 0
    assert terminal("blue", "altitude_boundary", "red")[1] == -10
    for reason in ("collision", "mutual_kill"):
        assert terminal("red", reason, "draw")[0] == -10
        assert terminal("blue", reason, "draw")[0] == -10
    assert terminal("red", "max_steps", "draw")[0] == 0


def test_coupled_difference_boundary_survivor_gets_no_positive_terminal():
    red_out = coupled_difference_rewards(.8, .2, .1, 10, "xy_boundary", "blue")
    blue_out = coupled_difference_rewards(.8, .2, .1, 10, "altitude_boundary", "red")
    assert red_out["red"]["terminal"] == -10 and red_out["blue"]["terminal"] == 0
    assert blue_out["red"]["terminal"] == 0 and blue_out["blue"]["terminal"] == -10
    assert red_out["red"]["dense"] == .1 * (.8 - .2)
    assert red_out["blue"]["dense"] == -.1 * (.8 - .2)


def test_madsac_threat_and_total_are_finite():
    threatened=madsac_segmented_reward(state(x=1000),state(x=0),"red",None,None)
    assert threatened["reward_threat"]==-0.15
    collision=madsac_segmented_reward(state(),state(x=10),"red","collision","draw")
    for result in (threatened,collision):
        assert np.isfinite(list(result.values())).all()
        assert np.isclose(result["reward_total"],sum(v for k,v in result.items() if k!="reward_total"))
