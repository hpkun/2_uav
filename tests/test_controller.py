import numpy as np
from uav_combat.controller import TargetStateController
from uav_combat.models import AircraftState


def test_zero_action_is_level_trim(spec):
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    _, control = TargetStateController().control_from_action(state, np.zeros(3), spec)
    assert np.allclose([control.nx, control.nz, control.phi], [0, 1, 0])


def test_yaw_sign_and_limits(spec):
    controller = TargetStateController()
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    assert controller.control_from_action(state, np.array([0.2, 0, 0]), spec)[1].phi > 0
    assert controller.control_from_action(state, np.array([-0.2, 0, 0]), spec)[1].phi < 0
    target, control = controller.control_from_action(state, np.array([9, -9, 9]), spec)
    clipped_target, clipped_control = controller.control_from_action(state, np.array([1, -1, 1]), spec)
    assert target == clipped_target and control == clipped_control
    assert target.desired_theta == spec.theta_min and target.desired_v == 200.0
    assert spec.nx_min <= control.nx <= spec.nx_max
    assert spec.nz_min <= control.nz <= spec.nz_max
    assert spec.phi_min <= control.phi <= spec.phi_max


def test_shortest_yaw_error_across_pi(spec):
    controller = TargetStateController(delta_yaw_max=np.deg2rad(2))
    state = AircraftState(0, 0, -3000, 150, 0, np.deg2rad(179))
    target, control = controller.control_from_action(state, np.array([1, 0, 0]), spec)
    assert target.desired_psi < 0 and control.phi > 0
