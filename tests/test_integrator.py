import numpy as np
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftState, TargetCommand


def run_target(spec, target, steps=600):
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    controller = TargetStateController()
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(0.1)
    for _ in range(steps):
        state = integrator.step(state, controller.compute_control(state, target, spec), dynamics, spec)
    return state


def test_long_level_trim_is_stable(spec):
    state = run_target(spec, TargetCommand(0, 0, 150))
    assert np.all(np.isfinite(state.as_array()))
    assert np.allclose([state.v, state.altitude, state.theta, state.psi], [150, 3000, 0, 0], atol=1e-10)


def test_yaw_step_converges_without_speed_or_pitch_drift(spec):
    target_yaw = np.deg2rad(30)
    state = run_target(spec, TargetCommand(target_yaw, 0, 150))
    assert state.psi > 0 and np.isclose(state.psi, target_yaw, atol=1e-8)
    assert np.isclose(state.v, 150, atol=1e-8)
    assert np.isclose(state.theta, 0, atol=1e-8)


def test_speed_step_converges(spec):
    state = run_target(spec, TargetCommand(0, 0, 170))
    assert np.isclose(state.v, 170, atol=1e-8)
