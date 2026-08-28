"""Fixed Blue policy based on one-step situation-reward lookahead."""
from __future__ import annotations

from itertools import product
from typing import Mapping
import numpy as np

from .dynamics import integrate_interval
from .geometry import compute_pairwise_geometry
from .models import Aircraft
from .reward import situation_reward

BLUE_ACTION_CANDIDATES = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=3)), dtype=np.float64)


class BluePolicy:
    """Team target mode plus independent 27-action lookahead for each Blue."""

    MODES = ("nearest", "mav_priority", "mixed_episode")

    def __init__(self, mode: str, decision_dt: float, physics_dt: float) -> None:
        if mode not in self.MODES:
            raise ValueError(f"invalid Blue target mode: {mode}")
        self.configured_mode = mode
        self.episode_mode = mode
        self.physics_dt = float(physics_dt)
        self.substeps = int(round(float(decision_dt) / self.physics_dt))

    def reset(self, rng: np.random.Generator) -> str:
        self.episode_mode = str(rng.choice(("nearest", "mav_priority"))) if self.configured_mode == "mixed_episode" else self.configured_mode
        return self.episode_mode

    def select_target(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> Aircraft | None:
        alive = [entity for entity in red.values() if entity.state.alive]
        if not alive:
            return None
        if self.episode_mode == "mav_priority" and red["MAV"].state.alive:
            return red["MAV"]
        return min(alive, key=lambda target: compute_pairwise_geometry(blue.state, target.state).distance)

    def action(self, blue: Aircraft, red: Mapping[str, Aircraft]) -> np.ndarray:
        if not blue.state.alive:
            return np.zeros(3, dtype=np.float64)
        target = self.select_target(blue, red)
        if target is None:
            return np.zeros(3, dtype=np.float64)
        best_action, best_score = BLUE_ACTION_CANDIDATES[0], -np.inf
        for candidate in BLUE_ACTION_CANDIDATES:
            predicted = integrate_interval(blue.state, candidate, blue.spec, self.physics_dt, self.substeps)
            score = situation_reward(predicted, target.state)
            if score > best_score:
                best_score, best_action = score, candidate
        return best_action.copy()
