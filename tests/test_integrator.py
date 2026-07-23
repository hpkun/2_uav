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


def run_action(spec, action, steps=20):
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    controller = TargetStateController()
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(0.1)
    history = []
    for _ in range(steps):
        _, control = controller.control_from_action(state, np.asarray(action), spec)
        state = integrator.step(state, control, dynamics, spec)
        history.append(state.copy())
    return history


def test_complete_action_to_target_chain_has_correct_directions(spec):
    positive_yaw = run_action(spec, [0.05, 0, 0])
    negative_yaw = run_action(spec, [-0.05, 0, 0])
    assert all(b.psi > a.psi for a, b in zip(positive_yaw, positive_yaw[1:]))
    assert all(b.psi < a.psi for a, b in zip(negative_yaw, negative_yaw[1:]))
    assert np.isclose(positive_yaw[-1].psi, -negative_yaw[-1].psi, atol=1e-10)
    speed = run_action(spec, [0, 0, 0.1])
    assert all(b.v > a.v for a, b in zip(speed, speed[1:]))
