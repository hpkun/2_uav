"""Deterministic privileged-state Blue periodic-heading pursuit policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dynamics import GRAVITY, THETA_MAX, THETA_MIN, inverse_trim_map, wrap_angle
from .models import Aircraft, OverloadCommand


BLUE_IDS = ("Blue1", "Blue2", "Blue3", "Blue4")
RED_IDS = ("MAV", "UAV1", "UAV2", "UAV3")


@dataclass
class GuidanceState:
    target_id: str | None = None
    desired_heading: float = 0.0
    desired_pitch: float = 0.0
    last_refresh_step: int = -1
    force_refresh: bool = False


class BluePolicy:
    """Nearest-alive-Red pursuit with a deterministic two-step guidance hold."""

    TARGET_STRATEGY = "nearest_red_aircraft"
    GUIDANCE_MODE = "periodic_heading"

    def __init__(
        self,
        decision_dt: float,
        physics_dt: float,
        battlefield: Mapping[str, tuple[float, float]],
        target_refresh_steps: int,
    ) -> None:
        self.decision_dt = float(decision_dt)
        self.physics_dt = float(physics_dt)
        if isinstance(target_refresh_steps, bool) or not isinstance(target_refresh_steps, (int, np.integer)):
            raise ValueError("target_refresh_steps must be a positive integer")
        if int(target_refresh_steps) <= 0:
            raise ValueError("target_refresh_steps must be a positive integer")
        self.target_refresh_steps = int(target_refresh_steps)
        self.battlefield = {
            axis: tuple(float(value) for value in battlefield[axis])
            for axis in ("x", "y", "altitude")
        }
        self._guidance_state: dict[str, GuidanceState] = {}
        self.reset(np.random.default_rng())

    def reset(self, rng: np.random.Generator) -> str:
        del rng
        self._guidance_state = {blue_id: GuidanceState() for blue_id in BLUE_IDS}
        return self.TARGET_STRATEGY

    def state_dict(self) -> dict[str, Any]:
        return {
            "guidance_mode": self.GUIDANCE_MODE,
            "target_refresh_steps": self.target_refresh_steps,
            "guidance_state": {
                blue_id: {
                    "target_id": state.target_id,
                    "desired_heading": float(state.desired_heading),
                    "desired_pitch": float(state.desired_pitch),
                    "last_refresh_step": int(state.last_refresh_step),
                    "force_refresh": bool(state.force_refresh),
                }
                for blue_id, state in self._guidance_state.items()
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("Blue policy state must be a mapping")
        if set(payload) != {"guidance_mode", "target_refresh_steps", "guidance_state"}:
            raise ValueError("Blue policy state has unknown or missing fields")
        if payload["guidance_mode"] != self.GUIDANCE_MODE:
            raise ValueError("Blue policy guidance mode mismatch")
        if payload["target_refresh_steps"] != self.target_refresh_steps:
            raise ValueError("Blue policy target refresh interval mismatch")
        raw_states = payload["guidance_state"]
        if not isinstance(raw_states, Mapping) or set(raw_states) != set(BLUE_IDS):
            raise ValueError("Blue policy state must contain all four Blue slots")
        restored: dict[str, GuidanceState] = {}
        expected = {
            "target_id", "desired_heading", "desired_pitch", "last_refresh_step", "force_refresh",
        }
        for blue_id in BLUE_IDS:
            raw = raw_states[blue_id]
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise ValueError(f"invalid Blue guidance state for {blue_id}")
            target_id = raw["target_id"]
            if target_id is not None and target_id not in RED_IDS:
                raise ValueError(f"invalid cached Red target for {blue_id}")
            heading = float(raw["desired_heading"])
            pitch = float(raw["desired_pitch"])
            last_refresh = raw["last_refresh_step"]
            force_refresh = raw["force_refresh"]
            if not np.isfinite((heading, pitch)).all() or not THETA_MIN <= pitch <= THETA_MAX:
                raise ValueError(f"non-finite or invalid cached guidance for {blue_id}")
            if isinstance(last_refresh, bool) or not isinstance(last_refresh, (int, np.integer)):
                raise ValueError(f"last_refresh_step must be an integer for {blue_id}")
            if not isinstance(force_refresh, (bool, np.bool_)):
                raise ValueError(f"force_refresh must be boolean for {blue_id}")
            restored[blue_id] = GuidanceState(
                target_id=target_id,
                desired_heading=heading,
                desired_pitch=pitch,
                last_refresh_step=int(last_refresh),
                force_refresh=bool(force_refresh),
            )
        self._guidance_state = restored

    def select_target(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> Aircraft | None:
        alive_red = [red[aid] for aid in RED_IDS if aid in red and red[aid].state.alive]
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

    @staticmethod
    def _guidance_angles(blue: Aircraft, target: Aircraft) -> tuple[float, float]:
        dx = float(target.state.x - blue.state.x)
        dy = float(target.state.y - blue.state.y)
        dh = float(target.state.h - blue.state.h)
        horizontal_distance = float(np.hypot(dx, dy))
        desired_heading = float(np.arctan2(dy, dx)) if horizontal_distance > 0.0 else float(blue.state.psi)
        desired_pitch = float(np.clip(np.arctan2(dh, horizontal_distance), THETA_MIN, THETA_MAX))
        return desired_heading, desired_pitch

    def _refresh_due(self, state: GuidanceState, red: Mapping[str, Aircraft], decision_step: int) -> bool:
        target_valid = (
            state.target_id is not None
            and state.target_id in red
            and red[state.target_id].state.alive
        )
        return bool(state.force_refresh or not target_valid or decision_step % self.target_refresh_steps == 0)

    def _refresh_guidance(
        self, blue: Aircraft, red: Mapping[str, Aircraft], decision_step: int,
    ) -> Aircraft | None:
        target = self.select_target(blue, red)
        state = self._guidance_state[blue.aircraft_id]
        if target is None:
            self._guidance_state[blue.aircraft_id] = GuidanceState()
            return None
        heading, pitch = self._guidance_angles(blue, target)
        state.target_id = target.aircraft_id
        state.desired_heading = heading
        state.desired_pitch = pitch
        state.last_refresh_step = int(decision_step)
        state.force_refresh = False
        return target

    def _track_direction_action(
        self, blue: Aircraft, desired_heading: float, desired_pitch: float,
    ) -> np.ndarray:
        state = blue.state
        heading_error = wrap_angle(float(desired_heading) - state.psi)
        pitch_error = float(desired_pitch) - state.theta
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

    def diagnostics(
        self, blue: Aircraft, red: Mapping[str, Aircraft], decision_step: int,
    ) -> dict[str, object]:
        """Read controller state without refreshing or otherwise mutating guidance."""
        state = self._guidance_state.get(blue.aircraft_id, GuidanceState())
        refresh_due = bool(blue.state.alive and self._refresh_due(state, red, int(decision_step)))
        altitude_recovery = self._altitude_recovery_action(blue) if blue.state.alive else None
        horizontal_recovery = bool(
            blue.state.alive and altitude_recovery is None and self._horizontal_recovery_active(blue)
        )
        return {
            "blue_target_id": state.target_id,
            "blue_target_is_MAV": state.target_id == "MAV",
            "blue_guidance_refresh_due": refresh_due,
            "blue_guidance_age": None if state.target_id is None else int(decision_step) - state.last_refresh_step,
            "blue_desired_heading": None if state.target_id is None else float(state.desired_heading),
            "blue_desired_pitch": None if state.target_id is None else float(state.desired_pitch),
            "blue_boundary_recovery_active": altitude_recovery is not None,
            "blue_horizontal_recovery_active": horizontal_recovery,
        }

    def action(self, blue: Aircraft, red: Mapping[str, Aircraft], decision_step: int) -> np.ndarray:
        if isinstance(decision_step, bool) or not isinstance(decision_step, (int, np.integer)):
            raise ValueError("decision_step must be an integer")
        decision_step = int(decision_step)
        if decision_step < 0:
            raise ValueError("decision_step cannot be negative")
        if not blue.state.alive:
            return np.zeros(3, dtype=np.float64)
        state = self._guidance_state[blue.aircraft_id]
        if self._refresh_due(state, red, decision_step):
            target = self._refresh_guidance(blue, red, decision_step)
            if target is None:
                return np.zeros(3, dtype=np.float64)

        recovery = self._altitude_recovery_action(blue)
        if recovery is not None:
            state.force_refresh = True
            return recovery
        if self._horizontal_recovery_active(blue):
            x_bounds, y_bounds = self.battlefield["x"], self.battlefield["y"]
            state.force_refresh = True
            action = self._track_direction_action(
                blue,
                float(np.arctan2(
                    0.5 * (y_bounds[0] + y_bounds[1]) - blue.state.y,
                    0.5 * (x_bounds[0] + x_bounds[1]) - blue.state.x,
                )),
                0.0,
            )
        else:
            action = self._track_direction_action(blue, state.desired_heading, state.desired_pitch)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise FloatingPointError(f"non-finite Blue periodic-guidance action for {blue.aircraft_id}")
        return np.clip(action, -1.0, 1.0)


__all__ = ["BluePolicy", "GuidanceState"]
