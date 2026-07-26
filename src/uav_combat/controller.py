"""Low-level controller from normalized actions to point-mass controls."""
from __future__ import annotations

import numpy as np

from .math_utils import angle_difference, safe_clip, wrap_angle
from .models import AircraftSpec, AircraftState, ControlCommand, TargetCommand


class TargetStateController:
    """Project controller, not a strict reproduction of a paper controller."""

    def __init__(
        self,
        delta_yaw_max: float = np.pi,
        delta_pitch_max: float = np.pi / 3,
        delta_speed_max: float = 50.0,
        gravity: float = 9.81,
        yaw_endpoint_epsilon: float = 1e-6,
        mapping_mode: str = "legacy_delta",
        epsilon: float = 1e-8,
    ) -> None:
        if not 0.0 < yaw_endpoint_epsilon < np.pi:
            raise ValueError("yaw_endpoint_epsilon must be in (0, pi)")
        if mapping_mode not in ("legacy_delta", "rate_aligned_v1"):
            raise ValueError(f"unknown action mapping_mode: {mapping_mode}")
        self.delta_yaw_max = delta_yaw_max
        self.delta_pitch_max = delta_pitch_max
        self.delta_speed_max = delta_speed_max
        self.gravity = gravity
        self.yaw_endpoint_epsilon = yaw_endpoint_epsilon
        self.mapping_mode = mapping_mode
        self.epsilon = epsilon

    def _mapped_deltas(self, action: np.ndarray, spec: AircraftSpec) -> tuple[np.ndarray, tuple[float, float, float]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with shape (3,)")
        clipped = np.clip(action, -1.0, 1.0)
        if self.mapping_mode == "legacy_delta":
            yaw_scale = min(abs(self.delta_yaw_max), np.pi - self.yaw_endpoint_epsilon)
            deltas = (
                clipped[0] * yaw_scale,
                clipped[1] * self.delta_pitch_max,
                clipped[2] * self.delta_speed_max,
            )
        else:
            deltas = (
                clipped[0] * spec.yaw_rate_max / max(abs(spec.k_yaw), self.epsilon),
                clipped[1] * spec.pitch_rate_max / max(abs(spec.k_pitch), self.epsilon),
                clipped[2] * spec.acceleration_max / max(abs(spec.k_speed), self.epsilon),
            )
        return clipped, tuple(float(v) for v in deltas)

    def action_to_target(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> TargetCommand:
        """Map a normalized action to a target heading, pitch, and speed."""
        _, (yaw_delta, pitch_delta, speed_delta) = self._mapped_deltas(action, spec)
        return TargetCommand(
            wrap_angle(state.psi + yaw_delta),
            safe_clip(state.theta + pitch_delta, spec.theta_min, spec.theta_max),
            safe_clip(state.v + speed_delta, spec.v_min, spec.v_max),
        )

    def compute_control(self, state: AircraftState, target: TargetCommand, spec: AircraftSpec) -> ControlCommand:
        """Invert requested rates to tangential/normal load and roll angle."""
        psi_dot = safe_clip(spec.k_yaw * angle_difference(target.desired_psi, state.psi), -spec.yaw_rate_max, spec.yaw_rate_max)
        theta_dot = safe_clip(spec.k_pitch * (target.desired_theta - state.theta), -spec.pitch_rate_max, spec.pitch_rate_max)
        v_dot = safe_clip(spec.k_speed * (target.desired_v - state.v), -spec.acceleration_max, spec.acceleration_max)
        nx = safe_clip(v_dot / self.gravity + np.sin(state.theta), spec.nx_min, spec.nx_max)
        a_term = np.cos(state.theta) + state.v / self.gravity * theta_dot
        b_term = state.v * np.cos(state.theta) / self.gravity * psi_dot
        magnitude = float(np.hypot(a_term, b_term))
        if a_term >= 0.0:
            nz_raw = magnitude
            phi_raw = float(np.arctan2(b_term, a_term))
        else:
            nz_raw = -magnitude
            phi_raw = float(np.arctan2(-b_term, -a_term))
        nz = safe_clip(nz_raw, spec.nz_min, spec.nz_max)
        phi = safe_clip(phi_raw, spec.phi_min, spec.phi_max)
        return ControlCommand(nx, nz, phi)

    def control_from_action(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> tuple[TargetCommand, ControlCommand]:
        """Return the target command and low-level control for one action."""
        target = self.action_to_target(state, action, spec)
        return target, self.compute_control(state, target, spec)

    def diagnostics(
        self,
        state: AircraftState,
        target: TargetCommand,
        control: ControlCommand,
        spec: AircraftSpec,
        action: np.ndarray | None = None,
    ) -> dict[str, float | bool | str]:
        """Return command-layer and physical-control diagnostics."""
        yaw_error = angle_difference(target.desired_psi, state.psi)
        pitch_error = target.desired_theta - state.theta
        speed_error = target.desired_v - state.v
        raw_yaw = spec.k_yaw * yaw_error
        raw_pitch = spec.k_pitch * pitch_error
        raw_accel = spec.k_speed * speed_error
        clipped_yaw = float(np.clip(raw_yaw, -spec.yaw_rate_max, spec.yaw_rate_max))
        clipped_pitch = float(np.clip(raw_pitch, -spec.pitch_rate_max, spec.pitch_rate_max))
        clipped_accel = float(np.clip(raw_accel, -spec.acceleration_max, spec.acceleration_max))
        if action is None:
            clipped_action = np.zeros(3, dtype=float)
            effective_yaw_delta, effective_pitch_delta, effective_speed_delta = yaw_error, pitch_error, speed_error
        else:
            clipped_action, deltas = self._mapped_deltas(action, spec)
            effective_yaw_delta, effective_pitch_delta, effective_speed_delta = deltas
        tolerance = 1e-8
        return {
            "action_mapping_mode": self.mapping_mode,
            "normalized_action_yaw": float(clipped_action[0]),
            "normalized_action_pitch": float(clipped_action[1]),
            "normalized_action_speed": float(clipped_action[2]),
            "effective_yaw_delta": float(effective_yaw_delta),
            "effective_pitch_delta": float(effective_pitch_delta),
            "effective_speed_delta": float(effective_speed_delta),
            "requested_yaw_rate": clipped_yaw,
            "requested_pitch_rate": clipped_pitch,
            "requested_acceleration": clipped_accel,
            "requested_yaw_rate_fraction": abs(clipped_yaw) / spec.yaw_rate_max if spec.yaw_rate_max > 0 else 0.0,
            "requested_pitch_rate_fraction": abs(clipped_pitch) / spec.pitch_rate_max if spec.pitch_rate_max > 0 else 0.0,
            "requested_acceleration_fraction": abs(clipped_accel) / spec.acceleration_max if spec.acceleration_max > 0 else 0.0,
            "command_yaw_rate_saturated": abs(raw_yaw - clipped_yaw) > tolerance,
            "command_pitch_rate_saturated": abs(raw_pitch - clipped_pitch) > tolerance,
            "command_acceleration_saturated": abs(raw_accel - clipped_accel) > tolerance,
            "yaw_error": yaw_error,
            "pitch_error": pitch_error,
            "speed_error": speed_error,
            "unclipped_yaw_rate": raw_yaw,
            "unclipped_pitch_rate": raw_pitch,
            "unclipped_acceleration": raw_accel,
            "clipped_yaw_rate": clipped_yaw,
            "clipped_pitch_rate": clipped_pitch,
            "clipped_acceleration": clipped_accel,
            "nx": control.nx,
            "nz": control.nz,
            "phi": control.phi,
            "yaw_rate_saturated": abs(raw_yaw - clipped_yaw) > tolerance,
            "pitch_rate_saturated": abs(raw_pitch - clipped_pitch) > tolerance,
            "acceleration_saturated": abs(raw_accel - clipped_accel) > tolerance,
            "nx_saturated": min(abs(control.nx - spec.nx_min), abs(control.nx - spec.nx_max)) < tolerance,
            "nz_saturated": min(abs(control.nz - spec.nz_min), abs(control.nz - spec.nz_max)) < tolerance,
            "phi_saturated": min(abs(control.phi - spec.phi_min), abs(control.phi - spec.phi_max)) < tolerance,
        }
