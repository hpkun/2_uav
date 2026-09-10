"""Deterministic privileged-state Blue policy using direct geometric pursuit."""
from __future__ import annotations

from typing import Mapping
import numpy as np

from .dynamics import GRAVITY, THETA_MAX, THETA_MIN, inverse_trim_map, wrap_angle
from .models import Aircraft, OverloadCommand


class BluePolicy:
    """Independent nearest-alive-Red direct-pursuit controller."""

    TARGET_STRATEGY = "nearest_red_aircraft"

    def __init__(self, decision_dt: float, physics_dt: float, battlefield: Mapping[str, tuple[float, float]]) -> None:
        self.decision_dt = float(decision_dt)
        self.physics_dt = float(physics_dt)
        self.battlefield = {axis: tuple(float(v) for v in battlefield[axis]) for axis in ("x", "y", "altitude")}

    def reset(self, rng: np.random.Generator) -> str:
        del rng
        return self.TARGET_STRATEGY

    def select_target(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> Aircraft | None:
        alive_red = [red[aid] for aid in ("MAV", "UAV1", "UAV2", "UAV3") if red[aid].state.alive]
        return min(
            alive_red,
            key=lambda target: (
                (target.state.x - blue.state.x) ** 2
                + (target.state.y - blue.state.y) ** 2
                + (target.state.h - blue.state.h) ** 2
            ),
        ) if alive_red else None

    def _within_battlefield(self, state: object) -> bool:
        return (
            self.battlefield["x"][0] <= state.x <= self.battlefield["x"][1]
            and self.battlefield["y"][0] <= state.y <= self.battlefield["y"][1]
            and self.battlefield["altitude"][0] <= state.h <= self.battlefield["altitude"][1]
        )

    def _altitude_recovery_guard(self, state: object, blue: Aircraft) -> float:
        """Return a conservative guard that also covers steep-flight recovery distance."""
        base_guard = 10.0 * blue.spec.v_max * self.decision_dt
        theta = float(state.theta)
        if theta < 0.0:
            ny_recovery = float(blue.spec.ny[1])
            level_margin = ny_recovery - 1.0
            current_margin = ny_recovery - float(np.cos(theta))
        elif theta > 0.0:
            ny_recovery = float(blue.spec.ny[0])
            level_margin = 1.0 - ny_recovery
            current_margin = float(np.cos(theta)) - ny_recovery
        else:
            return base_guard
        if level_margin <= 0.0 or current_margin <= 0.0:
            return float("inf")
        stopping_distance = blue.spec.v_max ** 2 / GRAVITY * abs(np.log(current_margin / level_margin))
        return max(base_guard, float(stopping_distance))

    def _altitude_recovery_action(self, blue: Aircraft) -> np.ndarray | None:
        lower, upper = self.battlefield["altitude"]
        state = blue.state
        guard = self._altitude_recovery_guard(state, blue)
        if state.theta < 0.0 and state.h < lower + guard:
            return np.asarray((-1.0, 1.0, 0.0), dtype=np.float64)
        if state.theta > 0.0 and state.h > upper - guard:
            return np.asarray((-1.0, -1.0, 0.0), dtype=np.float64)
        return None

    def _horizontal_recovery_active(self, blue: Aircraft) -> bool:
        state = blue.state
        guard = 10.0 * blue.spec.v_max * self.decision_dt
        vx = state.v * np.cos(state.theta) * np.cos(state.psi)
        vy = state.v * np.cos(state.theta) * np.sin(state.psi)
        x_lower, x_upper = self.battlefield["x"]
        y_lower, y_upper = self.battlefield["y"]
        return bool(
            (state.x < x_lower + guard and vx < 0.0)
            or (state.x > x_upper - guard and vx > 0.0)
            or (state.y < y_lower + guard and vy < 0.0)
            or (state.y > y_upper - guard and vy > 0.0)
        )

    def _direct_pursuit_action(self, blue: Aircraft, destination: tuple[float, float, float]) -> np.ndarray:
        state = blue.state
        dx, dy, dh = (
            float(destination[0]) - state.x,
            float(destination[1]) - state.y,
            float(destination[2]) - state.h,
        )
        horizontal_distance = float(np.hypot(dx, dy))
        desired_heading = float(np.arctan2(dy, dx)) if horizontal_distance > 0.0 else float(state.psi)
        desired_pitch = float(np.clip(np.arctan2(dh, horizontal_distance), THETA_MIN, THETA_MAX))
        heading_error = wrap_angle(desired_heading - state.psi)
        pitch_error = desired_pitch - state.theta

        v = max(float(state.v), 1e-6)
        cos_theta = float(np.cos(state.theta))
        theta_rate_limits = (
            GRAVITY / v * (blue.spec.ny[0] - cos_theta),
            GRAVITY / v * (blue.spec.ny[1] - cos_theta),
        )
        psi_scale = GRAVITY / max(v * cos_theta, 1e-6)
        psi_rate_limits = (psi_scale * blue.spec.nz[0], psi_scale * blue.spec.nz[1])
        desired_theta_rate = float(np.clip(pitch_error / self.decision_dt, *theta_rate_limits))
        desired_psi_rate = float(np.clip(heading_error / self.decision_dt, *psi_rate_limits))
        command = OverloadCommand(
            nx=float(np.sin(state.theta)),
            ny=float(np.clip(cos_theta + desired_theta_rate * v / GRAVITY, *blue.spec.ny)),
            nz=float(np.clip(desired_psi_rate * v * cos_theta / GRAVITY, *blue.spec.nz)),
        )
        return inverse_trim_map(command, state, blue.spec)

    def diagnostics(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> dict[str, object]:
        target = self.select_target(blue, red) if blue.state.alive else None
        altitude_recovery = self._altitude_recovery_action(blue) if blue.state.alive else None
        horizontal_recovery = bool(
            blue.state.alive and altitude_recovery is None and self._horizontal_recovery_active(blue)
        )
        return {
            "blue_target_id": None if target is None else target.aircraft_id,
            "blue_target_is_MAV": bool(target is not None and target.aircraft_id == "MAV"),
            "blue_boundary_recovery_active": altitude_recovery is not None,
            "blue_horizontal_recovery_active": horizontal_recovery,
        }

    def action(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> np.ndarray:
        if not blue.state.alive:
            return np.zeros(3, dtype=np.float64)
        target = self.select_target(blue, red)
        if target is None:
            return np.zeros(3, dtype=np.float64)
        recovery = self._altitude_recovery_action(blue)
        if recovery is not None:
            return recovery
        if self._horizontal_recovery_active(blue):
            x_bounds, y_bounds = self.battlefield["x"], self.battlefield["y"]
            destination = (
                0.5 * (x_bounds[0] + x_bounds[1]),
                0.5 * (y_bounds[0] + y_bounds[1]),
                blue.state.h,
            )
        else:
            destination = (target.state.x, target.state.y, target.state.h)
        action = self._direct_pursuit_action(blue, destination)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise FloatingPointError(f"non-finite Blue direct-pursuit action for {blue.aircraft_id}")
        return np.clip(action, -1.0, 1.0)
