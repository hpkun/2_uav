import numpy as np
import pytest
from uav_combat.mappo.buffer import MAPPOBuffer


def test_buffer_shapes_and_manual_gae_done_mask():
    buffer = MAPPOBuffer(3, 1)
    for done in (False, True, False):
        buffer.add(np.zeros((1, 2, 13)), np.zeros((1, 26)), np.zeros((1, 2, 3)), np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 2)), np.array([done]))
    buffer.compute_returns_and_advantages(np.zeros((1, 2)), gamma=1.0, gae_lambda=1.0)
    assert buffer.observations.shape == (3, 1, 2, 13)
    assert np.allclose(buffer.advantages[:, 0, 0], [2, 1, 1])
    assert np.all(np.isfinite(buffer.advantages)) and np.all(np.isfinite(buffer.returns))
    buffer.clear(); assert buffer.position == 0 and not buffer.dones.any()


def test_buffer_rejects_wrong_dimensions():
    buffer = MAPPOBuffer(1, 1)
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 13)), np.zeros((1, 26)), np.zeros((1, 2, 3)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1, dtype=bool))
