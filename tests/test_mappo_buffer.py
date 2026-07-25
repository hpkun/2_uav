import numpy as np
import pytest
from uav_combat.mappo.buffer import MAPPOBuffer

def test_policy_centric_shapes_and_manual_gae_done_mask():
    b=MAPPOBuffer(3,1)
    for done in (False,True,False):
        b.add(np.zeros((1,14)),np.zeros((1,14)),np.zeros((1,3)),np.zeros(1),np.ones(1),np.zeros(1),np.array([done]),np.zeros(1,dtype=np.int8))
    b.compute_returns_and_advantages(np.zeros(1),1.0,1.0)
    assert b.observations.shape==(3,1,14) and b.actions.shape==(3,1,3)
    assert b.rewards.shape==b.values.shape==b.advantages.shape==b.returns.shape==(3,1)
    assert b.active_teams.shape==(3,1)
    assert np.allclose(b.advantages[:,0],[2,1,1])
    b.clear();assert b.position==0 and not b.dones.any()

def test_buffer_rejects_color_slots_and_wrong_dimensions():
    b=MAPPOBuffer(1,1)
    with pytest.raises(ValueError):
        b.add(np.zeros((1,2,14)),np.zeros((1,14)),np.zeros((1,3)),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,dtype=bool),np.zeros(1,dtype=np.int8))
