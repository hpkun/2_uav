"""Frozen v3.7 heterogeneous role reward primitives; no training-side logic."""
from __future__ import annotations

import numpy as np

from .geometry import compute_pairwise_geometry


def target_score(red, blue, altitude_scale: float, velocity_scale: float, maximum_range: float) -> float:
    geometry = compute_pairwise_geometry(red, blue)
    relative_velocity = red.velocity_vector() - blue.velocity_vector()
    terms = np.asarray((
        1.0 - (geometry.ata + geometry.aa) / (2.0 * np.pi),
        float(geometry.distance <= maximum_range),
        (red.h - blue.h) / altitude_scale,
        np.linalg.norm(relative_velocity) / velocity_scale,
    ), dtype=np.float64)
    if not np.all(np.isfinite(terms)):
        return float("nan")
    return float(np.dot((0.35, 0.25, 0.20, 0.20), terms))


def uav_angle_reward(ata: float, aa: float) -> float:
    return float(1.0 - (ata + aa) / np.pi)


def uav_speed_reward(red_speed: float, blue_speed: float) -> float:
    red_speed = max(float(red_speed), 1e-9)
    blue_speed = float(blue_speed)
    if blue_speed < 0.5 * red_speed:
        return 1.0
    if blue_speed <= 1.5 * red_speed:
        return float(2.0 - 2.0 * blue_speed / red_speed)
    return -1.0


def uav_distance_reward(distance: float, minimum_range: float, maximum_range: float) -> float:
    distance = float(distance)
    if distance < minimum_range:
        return float(np.exp((distance - minimum_range) / minimum_range))
    if distance <= maximum_range:
        return 1.0
    return float(np.exp((maximum_range - distance) / maximum_range))


def uav_process_reward(angle: float, distance: float, speed: float) -> float:
    return float((15.0 * angle + 10.0 * distance + 10.0 * speed) / 35.0)


def mav_aspect_reward(blue_ata: float) -> float:
    threshold = np.pi / 4.0
    return float(-(1.0 - blue_ata / threshold)) if blue_ata < threshold else 0.0


def mav_awareness_reward(mav_ata: float) -> float:
    threshold = np.pi / 2.0
    return float(0.3 * (1.0 - mav_ata / threshold)) if mav_ata < threshold else 0.0
