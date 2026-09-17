"""Frozen v3.9 global heterogeneous role reward primitives."""
from __future__ import annotations

import numpy as np


def uav_angle_quality(ata: float, aa: float) -> float:
    """Normalized three-dimensional angle quality in the geometry contract range [0, 1]."""
    return float(1.0 - (float(ata) + float(aa)) / (2.0 * np.pi))


def uav_coupled_dense_reward(angle_quality: float, distance_quality: float) -> float:
    return float(angle_quality * distance_quality - 0.5)


def attack_gate_indicator(
    distance: float,
    ata: float,
    aa: float,
    minimum_range: float,
    maximum_range: float,
    ata_threshold: float,
    aa_threshold: float,
) -> float:
    """Use the exact inclusive-distance/strict-angle combat envelope."""
    inside = (
        minimum_range <= distance <= maximum_range
        and ata < ata_threshold
        and aa < aa_threshold
    )
    return float(inside)


def uav_gate_reward(gate_indicator: float) -> float:
    """Fixed v3.9 definition; the exact combat-envelope term is not tunable."""
    return float(0.5 * gate_indicator)


def mav_normalized_role_reward(
    threat: float,
    aspect_raw_sum: float,
    awareness_raw_sum: float,
    alive_blue_count: int,
) -> tuple[float, float, float]:
    if int(alive_blue_count) <= 0:
        return 0.0, 0.0, 0.0
    aspect_mean = float(aspect_raw_sum / alive_blue_count)
    awareness_mean = float(awareness_raw_sum / alive_blue_count)
    process = float(0.3 * threat + 0.2 * aspect_mean + 0.4 * awareness_mean)
    return process, aspect_mean, awareness_mean
