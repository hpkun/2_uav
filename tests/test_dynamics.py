import numpy as np
from uav_combat.dynamics import PointMassDynamics
from uav_combat.models import AircraftState, ControlCommand


def test_level_trim_derivatives():
    derivative = PointMassDynamics().derivatives(AircraftState(0, 0, -3000, 150, 0, 0), ControlCommand(0, 1, 0))
    assert np.allclose(derivative[[2, 3, 4, 5]], 0.0)
    assert np.all(np.isfinite(derivative))


def test_denominator_protection_is_finite():
    derivative = PointMassDynamics().derivatives(AircraftState(0, 0, 0, 0, np.pi / 2, 0), ControlCommand(0, 1, 1))
    assert np.all(np.isfinite(derivative))

