"""Numerically safe pairwise air-combat geometry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .models import AircraftState


@dataclass(frozen=True)
class PairwiseGeometry:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    distance: float
    ata: float
    aa: float


def compute_pairwise_geometry(attacker: AircraftState, target: AircraftState) -> PairwiseGeometry:
    relative_position = target.as_array()[:3] - attacker.as_array()[:3]
    attacker_velocity = attacker.velocity_vector()
    target_velocity = target.velocity_vector()
    distance = float(np.linalg.norm(relative_position))
    line = relative_position / max(distance, 1e-9)
    attacker_forward = attacker_velocity / max(float(np.linalg.norm(attacker_velocity)), 1e-9)
    target_forward = target_velocity / max(float(np.linalg.norm(target_velocity)), 1e-9)
    ata = float(np.arccos(np.clip(np.dot(attacker_forward, line), -1.0, 1.0)))
    aa = float(np.arccos(np.clip(np.dot(target_forward, line), -1.0, 1.0)))
    result = PairwiseGeometry(relative_position, target_velocity - attacker_velocity, distance, ata, aa)
    if not np.all(np.isfinite([*relative_position, *result.relative_velocity, distance, ata, aa])):
        raise FloatingPointError("non-finite pairwise geometry")
    return result

