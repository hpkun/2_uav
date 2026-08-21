"""Trim-centred overload control and fixed-step RK4 3DOF dynamics."""
from __future__ import annotations

import numpy as np

from .models import AircraftSpec, AircraftState, OverloadCommand

GRAVITY = 9.81
THETA_MIN = np.deg2rad(-60.0)
THETA_MAX = np.deg2rad(60.0)


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _piecewise_trim_map(value: float, lower: float, trim: float, upper: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    trim = float(np.clip(trim, lower, upper))
    return trim + value * ((upper - trim) if value >= 0.0 else (trim - lower))


def map_normalized_action(action: np.ndarray, state: AircraftState, spec: AircraftSpec) -> OverloadCommand:
    """Map ``[-1,1]^3`` around the current local equilibrium overload."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"action must have shape (3,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("action must be finite")
    trim = (np.sin(state.theta), np.cos(state.theta), 0.0)
    return OverloadCommand(
        _piecewise_trim_map(values[0], spec.nx[0], trim[0], spec.nx[1]),
        _piecewise_trim_map(values[1], spec.ny[0], trim[1], spec.ny[1]),
        _piecewise_trim_map(values[2], spec.nz[0], trim[2], spec.nz[1]),
    )


def derivatives(values: np.ndarray, command: OverloadCommand, gravity: float = GRAVITY) -> np.ndarray:
    """Evaluate the overload-controlled 3DOF equations."""
    _, _, _, v, theta, psi = np.asarray(values, dtype=np.float64)
    v_safe = max(float(v), 1e-6)
    cos_theta = float(np.cos(theta))
    cos_safe = np.copysign(max(abs(cos_theta), 1e-6), cos_theta if cos_theta != 0.0 else 1.0)
    return np.asarray([
        v * cos_theta * np.cos(psi), v * cos_theta * np.sin(psi), v * np.sin(theta),
        gravity * (command.nx - np.sin(theta)),
        gravity / v_safe * (command.ny - cos_theta),
        gravity / (v_safe * cos_safe) * command.nz,
    ], dtype=np.float64)


def rk4_step(state: AircraftState, command: OverloadCommand, dt: float, spec: AircraftSpec) -> AircraftState:
    """Advance one physics step and apply numerical safety constraints."""
    if dt <= 0.0:
        raise ValueError("physics dt must be positive")
    y = state.as_array()
    k1 = derivatives(y, command)
    k2 = derivatives(y + 0.5 * dt * k1, command)
    k3 = derivatives(y + 0.5 * dt * k2, command)
    k4 = derivatives(y + dt * k3, command)
    result = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    result[3] = np.clip(result[3], spec.v_min, spec.v_max)
    result[4] = np.clip(result[4], THETA_MIN, THETA_MAX)
    result[5] = wrap_angle(float(result[5]))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite aircraft state produced by RK4")
    return AircraftState.from_array(result, alive=state.alive)


def integrate_interval(state: AircraftState, action: np.ndarray, spec: AircraftSpec, physics_dt: float, substeps: int) -> AircraftState:
    """Hold one normalized action over a decision interval."""
    command = map_normalized_action(action, state, spec)
    result = state.copy()
    for _ in range(int(substeps)):
        result = rk4_step(result, command, physics_dt, spec)
    return result


class PointMassDynamics:
    """Small facade around the project dynamics equation."""

    def derivatives(self, state: AircraftState, command: OverloadCommand) -> np.ndarray:
        return derivatives(state.as_array(), command)

