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

from uav_combat.rewards import crdrl_coupled_reward

def test_crdrl_exact_equation_continuity_and_distance_peak():
    own=state();target=state(x=1000)
    at=crdrl_coupled_reward(own,target,"red",None,None)
    expected=np.exp(-.1)*14-4
    assert np.isclose(at["reward_coupled_dense_raw"],expected)
    left=crdrl_coupled_reward(own,state(x=999.999),"red",None,None)["reward_coupled_dense_raw"]
    right=crdrl_coupled_reward(own,state(x=1000.001),"red",None,None)["reward_coupled_dense_raw"]
    assert np.isclose(left,right,atol=2e-5)
    assert at["reward_coupled_dense_raw"]>crdrl_coupled_reward(own,state(x=900),"red",None,None)["reward_coupled_dense_raw"]
    assert at["reward_coupled_dense_raw"]>crdrl_coupled_reward(own,state(x=1100),"red",None,None)["reward_coupled_dense_raw"]

def test_crdrl_angles_sparse_strict_boundaries_and_terminal_adapter():
    favorable=crdrl_coupled_reward(state(),state(x=100),"red",None,None)
    poor=crdrl_coupled_reward(state(psi=np.pi),state(x=100),"red",None,None)
    assert favorable["reward_coupled_dense_raw"]>poor["reward_coupled_dense_raw"] and favorable["reward_sparse"]==2
    for target in (state(x=50),state(x=150),state(x=100,z=-3020)):
        assert crdrl_coupled_reward(state(),target,"red",None,None)["reward_sparse"]==0
    boundary=crdrl_coupled_reward(state(),state(x=100),"red","xy_boundary","blue")
    kill=crdrl_coupled_reward(state(),state(x=100),"red","red_kill","red")
    assert boundary["reward_boundary"]==-10 and boundary["reward_terminal"]==0
    assert kill["reward_terminal"]==10 and np.isfinite(list(kill.values())).all()
