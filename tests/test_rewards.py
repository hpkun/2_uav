import numpy as np
from uav_combat.models import AircraftState
from uav_combat.rewards import madsac_segmented_reward


def state(x=0, z=-3000, psi=0, theta=0): return AircraftState(x,0,z,150,theta,psi)


def test_madsac_position_angle_levels_and_distance_boundary():
    target=state(x=1000)
    assert madsac_segmented_reward(state(psi=np.deg2rad(-5)),target,"red",None,None)["reward_position"]==0.1
    assert madsac_segmented_reward(state(psi=np.deg2rad(-15)),target,"red",None,None)["reward_position"]==0.02
    assert madsac_segmented_reward(state(psi=np.deg2rad(-30)),target,"red",None,None)["reward_position"]==0.01
    boundary=madsac_segmented_reward(state(),state(x=4000),"red",None,None)
    assert boundary["reward_guide"]==0.001 and boundary["reward_position"]==0.1


def test_madsac_threat_terminal_boundary_and_total():
    threatened=madsac_segmented_reward(state(x=1000),state(x=0),"red",None,None)
    assert threatened["reward_threat"]==-0.15
    assert madsac_segmented_reward(state(),state(x=1000),"red","red_kill","red")["reward_terminal"]==10
    assert madsac_segmented_reward(state(),state(x=1000),"blue","red_kill","red")["reward_terminal"]==-10
    assert madsac_segmented_reward(state(),state(x=1000),"red","xy_boundary","blue")["reward_boundary"]==-10
    assert madsac_segmented_reward(state(),state(x=1000),"red","max_steps","draw")["reward_terminal"]==0
    collision=madsac_segmented_reward(state(),state(x=10),"red","collision","draw"); assert collision["reward_terminal"]==-10
    for result in (threatened,collision):
        assert np.isfinite(list(result.values())).all()
        assert np.isclose(result["reward_total"],sum(v for k,v in result.items() if k!="reward_total"))
