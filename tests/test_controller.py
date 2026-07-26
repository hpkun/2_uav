import numpy as np
import pytest
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.math_utils import angle_difference
from uav_combat.models import AircraftState, TargetCommand


def test_zero_action_is_level_trim(spec):
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    _, control = TargetStateController().control_from_action(state, np.zeros(3), spec)
    assert np.allclose([control.nx, control.nz, control.phi], [0, 1, 0])


def test_paper_action_ranges_are_unchanged():
    controller=TargetStateController()
    assert controller.delta_yaw_max==np.pi
    assert controller.delta_pitch_max==np.pi/3
    assert controller.delta_speed_max==50.0
    assert controller.mapping_mode == "legacy_delta"


def test_legacy_delta_outputs_match_existing_values(spec):
    controller = TargetStateController()
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    target = controller.action_to_target(state, np.array([0.5, 0.5, 0.5]), spec)
    assert np.isclose(angle_difference(target.desired_psi, state.psi), (np.pi - 1e-6) * 0.5)
    assert np.isclose(target.desired_theta, np.pi / 6)
    assert np.isclose(target.desired_v, 175.0)


def test_rate_aligned_yaw_pitch_speed_ratios(spec):
    controller = TargetStateController(mapping_mode="rate_aligned_v1")
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    for yaw_action, expected in ((1.0, spec.yaw_rate_max), (-1.0, -spec.yaw_rate_max), (0.5, 0.5 * spec.yaw_rate_max)):
        target, control = controller.control_from_action(state, np.array([yaw_action, 0.0, 0.0]), spec)
        diag = controller.diagnostics(state, target, control, spec, np.array([yaw_action, 0.0, 0.0]))
        assert np.isclose(diag["requested_yaw_rate"], expected)
        assert not diag["command_yaw_rate_saturated"]
    target, control = controller.control_from_action(state, np.array([0.0, 1.0, 1.0]), spec)
    diag = controller.diagnostics(state, target, control, spec, np.array([0.0, 1.0, 1.0]))
    assert np.isclose(diag["requested_pitch_rate"], spec.pitch_rate_max)
    assert np.isclose(diag["requested_acceleration"], spec.acceleration_max)
    assert np.isclose(diag["effective_speed_delta"], spec.acceleration_max / spec.k_speed)
    assert not diag["command_pitch_rate_saturated"]
    assert not diag["command_acceleration_saturated"]


def test_rate_aligned_all_legal_actions_stay_within_command_limits(spec):
    controller = TargetStateController(mapping_mode="rate_aligned_v1")
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    for action in (
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([0.75, -0.25, 0.5]),
    ):
        target, control = controller.control_from_action(state, action, spec)
        diag = controller.diagnostics(state, target, control, spec, action)
        assert diag["requested_yaw_rate_fraction"] <= 1.0 + 1e-8
        assert diag["requested_pitch_rate_fraction"] <= 1.0 + 1e-8
        assert diag["requested_acceleration_fraction"] <= 1.0 + 1e-8
        assert not diag["command_yaw_rate_saturated"]
        assert not diag["command_pitch_rate_saturated"]
        assert not diag["command_acceleration_saturated"]


def test_rate_aligned_can_still_hit_physical_saturation(spec):
    controller = TargetStateController(mapping_mode="rate_aligned_v1")
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    action = np.array([1.0, 0.0, 0.0])
    target, control = controller.control_from_action(state, action, spec)
    diag = controller.diagnostics(state, target, control, spec, action)
    assert diag["nz_saturated"] or diag["phi_saturated"] or diag["nx_saturated"]


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


def test_maximum_yaw_endpoints_keep_opposite_directions(spec):
    controller = TargetStateController()
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    positive_target, positive_control = controller.control_from_action(state, np.array([1, 0, 0]), spec)
    negative_target, negative_control = controller.control_from_action(state, np.array([-1, 0, 0]), spec)
    assert positive_target.desired_psi != negative_target.desired_psi
    assert angle_difference(positive_target.desired_psi, state.psi) > 0
    assert angle_difference(negative_target.desired_psi, state.psi) < 0
    assert positive_control.phi > 0 and negative_control.phi < 0


def test_pure_pitch_commands_do_not_create_yaw(spec):
    controller = TargetStateController()
    dynamics = PointMassDynamics()
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    down = controller.compute_control(state, TargetCommand(0, np.deg2rad(-20), 150), spec)
    down_derivative = dynamics.derivatives(state, down)
    assert down.nz < 0 and abs(down.phi) < 1e-12
    assert down_derivative[4] < 0 and abs(down_derivative[5]) < 1e-12
    up = controller.compute_control(state, TargetCommand(0, np.deg2rad(10), 150), spec)
    up_derivative = dynamics.derivatives(state, up)
    assert abs(up.phi) < 1e-12
    assert up_derivative[4] > 0 and abs(up_derivative[5]) < 1e-12


@pytest.mark.parametrize("yaw_deg,pitch_deg", [
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1),
])
def test_control_inverse_reconstructs_unsaturated_terms(spec, yaw_deg, pitch_deg):
    controller = TargetStateController()
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    target = TargetCommand(np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 150)
    control = controller.compute_control(state, target, spec)
    psi_dot = spec.k_yaw * angle_difference(target.desired_psi, state.psi)
    theta_dot = spec.k_pitch * (target.desired_theta - state.theta)
    a_term = np.cos(state.theta) + state.v / controller.gravity * theta_dot
    b_term = state.v * np.cos(state.theta) / controller.gravity * psi_dot
    assert np.isclose(control.nz * np.cos(control.phi), a_term, atol=1e-12)
    assert np.isclose(control.nz * np.sin(control.phi), b_term, atol=1e-12)


@pytest.mark.parametrize("action", [
    np.zeros(2), np.zeros(4), np.array([0, np.nan, 0]), np.array([0, 0, np.inf]),
])
def test_invalid_actions_raise_value_error(spec, action):
    with pytest.raises(ValueError):
        TargetStateController().action_to_target(AircraftState(0, 0, -3000, 150, 0, 0), action, spec)
