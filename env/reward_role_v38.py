"""Frozen v3.8 coupled UAV process reward primitive."""
from __future__ import annotations


def uav_coupled_process_reward(angle: float, distance: float) -> float:
    """Couple angle and distance so favorable distance cannot offset poor geometry."""
    return float(angle * distance)
