"""Shared red-combat formation reference for the heterogeneous 4v3 contract."""
from __future__ import annotations

from typing import Any

import numpy as np

from .models import Aircraft


def compute_red_combat_formation_reference(
    support: Aircraft,
    combat_aircraft: list[Aircraft],
    *,
    direction_validity_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Return one stable formation reference shared by obs, reward, and rule policy."""
    alive_combat = [a for a in combat_aircraft if a.state.alive]
    if not alive_combat:
        return {
            "centroid": np.zeros(3, dtype=np.float64),
            "horizontal_direction": np.zeros(2, dtype=np.float64),
            "direction_strength": 0.0,
            "direction_valid": False,
            "support_relative": np.zeros(3, dtype=np.float64),
            "centroid_distance": 0.0,
            "rear_alignment": 0.0,
        }

    positions = np.asarray([a.state.as_array()[:3] for a in alive_combat], dtype=np.float64)
    centroid = positions.mean(axis=0)
    horizontal_velocity = np.asarray(
        [[a.state.v * np.cos(a.state.theta) * np.cos(a.state.psi),
          a.state.v * np.cos(a.state.theta) * np.sin(a.state.psi)] for a in alive_combat],
        dtype=np.float64,
    ).mean(axis=0)
    direction_strength = float(np.linalg.norm(horizontal_velocity))
    direction_valid = direction_strength >= float(direction_validity_threshold)
    direction = horizontal_velocity / direction_strength if direction_valid else np.zeros(2, dtype=np.float64)

    support_relative = support.state.as_array()[:3].astype(np.float64) - centroid
    centroid_distance = float(np.linalg.norm(support_relative[:2]))
    if centroid_distance > 1e-8 and direction_valid:
        backward = -direction
        rear_alignment = float(np.dot(support_relative[:2] / centroid_distance, backward))
    else:
        rear_alignment = 0.0
    return {
        "centroid": centroid,
        "horizontal_direction": direction,
        "direction_strength": direction_strength,
        "direction_valid": bool(direction_valid),
        "support_relative": support_relative,
        "centroid_distance": centroid_distance,
        "rear_alignment": rear_alignment,
    }


__all__ = ["compute_red_combat_formation_reference"]
