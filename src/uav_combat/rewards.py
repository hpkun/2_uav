"""Segmented, legacy, and CR-DRL reward functions."""
from typing import Any
import numpy as np
from .geometry import compute_pairwise_geometry
from .models import AircraftState

BOUNDARY_REASONS = {"altitude_boundary", "xy_boundary", "boundary"}


def _empty() -> dict[str, float]:
    return {key: 0.0 for key in ("reward_terminal", "reward_boundary", "reward_guide", "reward_position", "reward_threat", "reward_total")}


def terminal_team_rewards(reason: str | None, outcome: str | None, magnitude: float = 10.0) -> dict[str, float]:
    rewards = {"red": 0.0, "blue": 0.0}
    if reason == "red_kill": return {"red": magnitude, "blue": -magnitude}
    if reason == "blue_kill": return {"red": -magnitude, "blue": magnitude}
    if reason in {"collision", "mutual_kill"}: return {"red": -magnitude, "blue": -magnitude}
    if reason in BOUNDARY_REASONS:
        if outcome == "red": rewards["blue"] = -magnitude
        elif outcome == "blue": rewards["red"] = -magnitude
        else: rewards = {"red": -magnitude, "blue": -magnitude}
    return rewards


def madsac_segmented_reward(own: AircraftState, target: AircraftState, own_team: str, reason: str | None, outcome: str | None) -> dict[str, float]:
    result = _empty(); geometry = compute_pairwise_geometry(own, target); reverse = compute_pairwise_geometry(target, own)
    terminal = terminal_team_rewards(reason, outcome)
    result["reward_boundary" if reason in BOUNDARY_REASONS else "reward_terminal"] = terminal[own_team]
    degrees = np.pi / 180.0; tolerance = 1e-12; pitch = abs(geometry.pitch_error)
    within = lambda value, limit: value <= limit + tolerance
    if geometry.distance >= 4000.0 and within(geometry.ata, 30*degrees) and within(pitch, 30*degrees): result["reward_guide"] = .001
    if geometry.distance <= 4000.0 and within(geometry.aa, 30*degrees):
        if within(geometry.ata, 5*degrees) and within(pitch, 5*degrees): result["reward_position"] = .1
        elif within(geometry.ata, 15*degrees) and within(pitch, 15*degrees): result["reward_position"] = .02
        elif within(geometry.ata, 30*degrees) and within(pitch, 30*degrees): result["reward_position"] = .01
    reverse_pitch = abs(reverse.pitch_error)
    if reverse.distance <= 4000.0 and within(reverse.aa, 30*degrees):
        if within(reverse.ata, 5*degrees) and within(reverse_pitch, 5*degrees): result["reward_threat"] = -.15
        elif within(reverse.ata, 15*degrees) and within(reverse_pitch, 15*degrees): result["reward_threat"] = -.025
        elif within(reverse.ata, 30*degrees) and within(reverse_pitch, 30*degrees): result["reward_threat"] = -.015
    result["reward_total"] = float(sum(v for k,v in result.items() if k != "reward_total"))
    if not np.isfinite(list(result.values())).all(): raise FloatingPointError("non-finite segmented reward")
    return result


def coupled_difference_rewards(red_score: float, blue_score: float, scale: float, terminal_reward: float, reason: str | None, outcome: str | None) -> dict[str, dict[str, float]]:
    dense = scale * (red_score-blue_score); terminal = terminal_team_rewards(reason,outcome,terminal_reward)
    return {team: {"dense": float(dense if team=="red" else -dense), "terminal": terminal[team]} for team in ("red","blue")}


def crdrl_coupled_reward(own: AircraftState, target: AircraftState, own_team: str, reason: str | None, outcome: str | None, dense_scale: float = 1.0, sparse_scale: float = 1.0) -> dict[str, float]:
    """CR-DRL Eq. 9 plus the paper sparse term and project terminal semantics."""
    geometry = compute_pairwise_geometry(own,target)
    distance_km = geometry.distance / 1000.0
    angle_factor = 4*np.cos(geometry.ata/2) + 8*np.exp(-geometry.ata) + 2*np.cos(geometry.aa/2)
    dense_raw = (np.exp(distance_km-1.1) if distance_km <= 1 else np.exp(-.1*distance_km))*angle_factor - 4
    sparse_raw = 2.0 if (geometry.ata < np.deg2rad(10) and geometry.aa < np.deg2rad(10) and 50 < geometry.distance < 150 and abs(own.altitude-target.altitude) < 20) else 0.0
    terminal = terminal_team_rewards(reason,outcome)
    result = {"reward_coupled_dense_raw":float(dense_raw), "reward_coupled_dense":float(dense_scale*dense_raw),
              "reward_sparse":float(sparse_scale*sparse_raw), "reward_terminal":float(0 if reason in BOUNDARY_REASONS else terminal[own_team]),
              "reward_boundary":float(terminal[own_team] if reason in BOUNDARY_REASONS else 0)}
    result["reward_total"] = float(sum(result[k] for k in ("reward_coupled_dense","reward_sparse","reward_terminal","reward_boundary")))
    if not np.isfinite(list(result.values())).all(): raise FloatingPointError("non-finite CR-DRL reward")
    return result
