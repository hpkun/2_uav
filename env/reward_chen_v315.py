"""Frozen Chen-aligned v3.15 reward; no missiles or substitute shaping.

PAPER_DIRECT: UAV speed/angle/distance, process weights and event scales.
ENV_MAPPING: MAV distances, reverse ATA as TA, bounded contribution constants.
DISABLED_UNAVAILABLE: height, dodge, missile threat, position and awareness.
"""
from __future__ import annotations

import numpy as np

CHEN_VERSION = "heterogeneous_mavuav_4v4_v3_15"
CHEN_MODE = "chen_heterogeneous_v1"
CHEN_CONFIG = {
    "mode": CHEN_MODE,
    "mav": {"danger_distance": 5000.0, "safe_distance": 10000.0,
            "death_penalty": 200.0, "kill_contribution": 50.0, "contribution_cap": 200.0},
    "uav": {"speed_weight": 10.0, "angle_weight": 15.0, "distance_weight": 10.0,
            "kill_reward": 200.0, "combat_loss": -200.0, "boundary_loss": -100.0},
    "target_selector": {"angle_weight": .35, "distance_weight": .25,
                        "altitude_weight": .20, "relative_velocity_weight": .20},
}


def angle_reward(ata: float, aa: float) -> float:
    return float(1.0 - (ata + aa) / np.pi)


def distance_reward(distance_m: float) -> float:
    km = float(distance_m) / 1000.0
    return 1.0 if km <= 5.0 else float(np.exp(-.921 * (km - 5.0))) if km < 10.0 else -1.0


def speed_reward(red_speed: float, blue_speed: float) -> float:
    # Valid live aircraft speeds are >=150; guard only an invalid zero denominator.
    vr = max(float(red_speed), np.finfo(np.float64).tiny)
    vb = float(blue_speed)
    if vb < .5 * vr:
        return 1.0
    if vb <= 1.5 * vr:
        return float(2.0 - 2.0 * vb / vr)
    return -1.0


def mav_distance_reward(distance: float | None, danger: float, safe: float) -> float:
    if distance is None or distance >= safe:
        return .2
    if distance < danger:
        return float(-(1.0 - distance / danger))
    return float(-.5 * (1.0 - (distance - danger) / (safe - danger)))


def situation_score(geometry, red, blue, normalization, weapon_range: float) -> float:
    # Selection only: this score NEVER enters the local reward sum.
    return float(.35 * (1.0 - (geometry.ata + geometry.aa) / (2.0 * np.pi))
                 + .25 * float(geometry.distance <= weapon_range)
                 + .20 * (red.h - blue.h) / normalization["relative_altitude_scale"]
                 + .20 * np.linalg.norm(geometry.relative_velocity) / normalization["relative_velocity_scale"])
