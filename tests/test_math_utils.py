import numpy as np
from uav_combat.math_utils import angle_difference, wrap_angle


def test_wrap_angle_range():
    assert all(-np.pi <= wrap_angle(x) < np.pi for x in np.linspace(-20, 20, 1000))


def test_angle_difference_across_boundary():
    assert np.isclose(angle_difference(np.deg2rad(-179), np.deg2rad(179)), np.deg2rad(2))
    assert np.isclose(angle_difference(np.deg2rad(179), np.deg2rad(-179)), np.deg2rad(-2))

