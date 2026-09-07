"""Fixed Blue policy based on one-step situation-reward lookahead."""
from __future__ import annotations

from itertools import product
from typing import Mapping
import numpy as np

from .dynamics import integrate_interval
from .geometry import compute_pairwise_geometry
from .models import Aircraft
from .reward import situation_reward

BLUE_ACTION_CANDIDATES = np.asarray(
    sorted(product((-1.0, 0.0, 1.0), repeat=3), key=lambda action: (sum(v * v for v in action), action)),
    dtype=np.float64,
)


class BluePolicy:
    """Independent nearest-Red-UAV 27-action boundary-safe lookahead."""

    TARGET_STRATEGY = "nearest_red_uav"

    def __init__(self, decision_dt: float, physics_dt: float, battlefield: Mapping[str, tuple[float, float]]) -> None:
        self.physics_dt = float(physics_dt)
        self.substeps = int(round(float(decision_dt) / self.physics_dt))
        self.battlefield = {axis: tuple(float(v) for v in battlefield[axis]) for axis in ("x", "y", "altitude")}

    def reset(self, rng: np.random.Generator) -> str:
        del rng
        return self.TARGET_STRATEGY

    def select_target(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> Aircraft | None:
        alive_uavs = [red[aid] for aid in ("UAV1", "UAV2", "UAV3") if red[aid].state.alive]
        if alive_uavs:
            return min(alive_uavs, key=lambda target: compute_pairwise_geometry(blue.state, target.state).distance)
        mav = red["MAV"]
        if not mav.state.alive:
            return None
        return mav

    def _within_battlefield(self, state: object) -> bool:
        return (
            self.battlefield["x"][0] <= state.x <= self.battlefield["x"][1]
            and self.battlefield["y"][0] <= state.y <= self.battlefield["y"][1]
            and self.battlefield["altitude"][0] <= state.h <= self.battlefield["altitude"][1]
        )

    def action(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> np.ndarray:
        if not blue.state.alive:
            return np.zeros(3, dtype=np.float64)
        target = self.select_target(blue, red)
        if target is None:
            return np.zeros(3, dtype=np.float64)
        best_action, best_score = None, -np.inf
        for candidate in BLUE_ACTION_CANDIDATES:
            predicted = integrate_interval(blue.state, candidate, blue.spec, self.physics_dt, self.substeps)
            if not self._within_battlefield(predicted):
                continue
            score = situation_reward(predicted, target.state)
            if score > best_score:
                best_score, best_action = score, candidate
        if best_action is None:
            raise RuntimeError(f"Blue boundary controller invariant violated: no safe action for {blue.aircraft_id}")
        return best_action.copy()
