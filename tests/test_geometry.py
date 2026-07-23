import numpy as np

from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.models import AircraftState


def state(x=0, y=0, z=-3000, v=150, theta=0, psi=0):
    return AircraftState(x, y, z, v, theta, psi)


def test_tail_chase_head_on_and_crossing_geometry():
    tail = compute_pairwise_geometry(state(), state(x=500))
    assert np.isclose(tail.ata, 0) and np.isclose(tail.aa, 0)
    head_on = compute_pairwise_geometry(state(), state(x=500, psi=np.pi))
    assert np.isclose(head_on.ata, 0) and np.isclose(head_on.aa, np.pi)
    crossing = compute_pairwise_geometry(state(psi=np.pi / 2), state(x=500))
    assert np.isclose(crossing.ata, np.pi / 2)


def test_los_yaw_and_pitch_signs():
    right_and_high = compute_pairwise_geometry(state(), state(x=500, y=100, z=-3100))
    assert right_and_high.los_yaw > 0 and right_and_high.yaw_error > 0
    assert right_and_high.los_pitch > 0 and right_and_high.pitch_error > 0
    left_and_low = compute_pairwise_geometry(state(), state(x=500, y=-100, z=-2900))
    assert left_and_low.los_yaw < 0 and left_and_low.los_pitch < 0


def test_zero_distance_geometry_is_finite():
    geometry = compute_pairwise_geometry(state(), state())
    scalars = [geometry.distance, geometry.ata, geometry.aa, geometry.los_yaw, geometry.los_pitch, geometry.yaw_error, geometry.pitch_error]
    assert np.all(np.isfinite(scalars))
    assert np.all(np.isfinite(geometry.relative_position)) and np.all(np.isfinite(geometry.relative_velocity))

