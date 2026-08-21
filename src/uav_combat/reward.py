"""Published segmented air-combat situation reward used by both teams."""
from __future__ import annotations

import numpy as np

from .geometry import compute_pairwise_geometry
from .models import AircraftState

PHI_M = np.deg2rad(30.0)
SITUATION_WEIGHTS = (0.32, 0.43, 0.10, 0.10, 0.05)


def bearing_reward(phi: float) -> float:
    phi = abs(float(phi))
    return float(1.0 - 0.3 * phi / PHI_M) if phi <= PHI_M else float(0.7 * (np.pi - phi) / (np.pi - PHI_M))


def entering_angle_reward(q: float) -> float:
    return float(1.0 - abs(float(q)) / np.pi)


def distance_reward(distance: float) -> float:
    distance = float(distance)
    if distance < 1000.0:
        return 0.0
    if distance <= 3000.0:
        return 1.0
    return float(np.exp((3000.0 - distance) / 5000.0))


def speed_reward(attacker_speed: float, target_speed: float) -> float:
    ratio = float(attacker_speed) / max(float(target_speed), 1e-9)
    if ratio < 0.6:
        return 0.1
    if ratio <= 1.5:
        return -0.5 + ratio
    return 1.0


def height_reward(height_difference: float) -> float:
    dh = float(height_difference)
    if dh < -2000.0:
        return 0.0
    if dh < 2000.0:
        return (2000.0 + dh) / 4000.0
    if dh < 4000.0:
        return (4000.0 - dh) / 2000.0
    return 0.0


def situation_components(attacker: AircraftState, target: AircraftState) -> tuple[float, float, float, float, float]:
    geometry = compute_pairwise_geometry(attacker, target)
    return (
        bearing_reward(geometry.ata), entering_angle_reward(geometry.aa),
        distance_reward(geometry.distance), speed_reward(attacker.v, target.v),
        height_reward(attacker.h - target.h),
    )


def situation_reward(attacker: AircraftState, target: AircraftState) -> float:
    return float(np.dot(SITUATION_WEIGHTS, situation_components(attacker, target)))

