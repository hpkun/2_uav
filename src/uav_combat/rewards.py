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


# ===================================================================
# Paper-coupled team v2 reward components (3v3)
# ===================================================================


def coupled_attack_advantage(
    own: AircraftState, target: AircraftState,
    preferred_distance: float, distance_sigma: float,
    ata_sigma: float, aa_sigma: float,
) -> float:
    """Gaussian-weighted attack geometry score in [0, 1].

    distance_factor = exp(-0.5 * ((d - preferred) / distance_sigma)^2)
    ata_factor      = exp(-0.5 * (ATA / ata_sigma)^2)
    aa_factor       = exp(-0.5 * (AA / aa_sigma)^2)
    score = distance_factor * ata_factor * aa_factor
    """
    geo = compute_pairwise_geometry(own, target)
    dist_factor = float(np.exp(-0.5 * ((geo.distance - preferred_distance) / distance_sigma) ** 2))
    ata_factor = float(np.exp(-0.5 * (geo.ata / ata_sigma) ** 2))
    aa_factor = float(np.exp(-0.5 * (geo.aa / aa_sigma) ** 2))
    score = dist_factor * ata_factor * aa_factor
    if not np.isfinite(score):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def approach_progress_reward(
    prev_own: AircraftState, curr_own: AircraftState,
    prev_target: AircraftState, curr_target: AircraftState,
    distance_threshold: float, distance_normalizer: float,
) -> float:
    """Reward closing distance when beyond distance_threshold, weighted by heading.

    Returns value in [-1, 1].  Zero if either distance <= distance_threshold.
    """
    prev_geo = compute_pairwise_geometry(prev_own, prev_target)
    curr_geo = compute_pairwise_geometry(curr_own, curr_target)
    if prev_geo.distance <= distance_threshold or curr_geo.distance <= distance_threshold:
        return 0.0
    closing = float(np.clip((prev_geo.distance - curr_geo.distance) / distance_normalizer, -1.0, 1.0))
    heading = float(max(np.cos(curr_geo.ata), 0.0))
    score = closing * heading
    return float(np.clip(score, -1.0, 1.0))


def soft_boundary_risk(
    state: AircraftState,
    x_limit: float, y_limit: float,
    altitude_min: float, altitude_max: float,
    horizontal_soft_ratio: float, altitude_soft_margin: float,
) -> dict[str, float]:
    """Soft boundary risk: 0 in safe zone, rising quadratically to 1 at physical limit.

    Returns {xy_risk, altitude_risk, total_risk}.
    """
    rho_xy = float(max(abs(state.x) / x_limit, abs(state.y) / y_limit))
    if rho_xy <= horizontal_soft_ratio:
        xy_risk = 0.0
    else:
        xy_risk = float(np.clip((rho_xy - horizontal_soft_ratio) / (1.0 - horizontal_soft_ratio), 0.0, 1.0)) ** 2

    alt = state.altitude
    low_soft = altitude_min + altitude_soft_margin
    high_soft = altitude_max - altitude_soft_margin
    if alt <= low_soft:
        alt_risk = float(np.clip((low_soft - alt) / altitude_soft_margin, 0.0, 1.0)) ** 2
    elif alt >= high_soft:
        alt_risk = float(np.clip((alt - high_soft) / altitude_soft_margin, 0.0, 1.0)) ** 2
    else:
        alt_risk = 0.0

    total = float(np.clip(xy_risk + alt_risk, 0.0, 2.0))
    return {"xy_risk": xy_risk, "altitude_risk": alt_risk, "total_risk": total}


def friendly_separation_risk(
    own: AircraftState, teammates: list[AircraftState],
    friendly_safe_distance: float, collision_distance: float,
) -> float:
    """Penalty when nearest alive teammate is too close.

    Returns 0 if nearest >= friendly_safe_distance, else quadratic risk in [0, 1].
    """
    alive_teammates = [t for t in teammates if t.alive]
    if not alive_teammates:
        return 0.0
    min_dist = float(min(np.linalg.norm(own.as_array()[:3] - t.as_array()[:3]) for t in alive_teammates))
    if min_dist >= friendly_safe_distance:
        return 0.0
    denom = friendly_safe_distance - collision_distance
    if denom <= 0:
        return 1.0
    risk = float(np.clip((friendly_safe_distance - min_dist) / denom, 0.0, 1.0)) ** 2
    return risk


def head_on_collision_risk(
    red_state: AircraftState, blue_state: AircraftState,
    head_on_distance: float, head_on_angle: float,
) -> float:
    """Risk when red and blue are on a head-on collision course.

    Only triggers when distance < head_on_distance AND
    red->blue ATA < head_on_angle AND blue->red ATA < head_on_angle.
    Returns quadratic risk in [0, 1].
    """
    dist = float(np.linalg.norm(red_state.as_array()[:3] - blue_state.as_array()[:3]))
    if dist >= head_on_distance:
        return 0.0
    red_to_blue = compute_pairwise_geometry(red_state, blue_state)
    blue_to_red = compute_pairwise_geometry(blue_state, red_state)
    if red_to_blue.ata >= head_on_angle or blue_to_red.ata >= head_on_angle:
        return 0.0
    return float(np.clip(1.0 - dist / head_on_distance, 0.0, 1.0)) ** 2
