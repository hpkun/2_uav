"""Analytic attack-geometry potential shaping for HAPPO training only."""
from __future__ import annotations

import numpy as np


SELF_ALIVE_INDEX = 6
ENEMY_BLOCK_STARTS = (33, 44)
ENEMY_DISTANCE_OFFSET = 3
ENEMY_ATA_OFFSET = 4
ENEMY_AA_OFFSET = 5
ENEMY_ALIVE_OFFSET = 6
ENEMY_DIRECT_OFFSET = 7
ENEMY_DATALINK_OFFSET = 8
ENEMY_KILLED_OFFSET = 10
EXPECTED_OBSERVATION_SHAPE = (3, 55)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-array))


def smooth_distance_gate(distance_m: np.ndarray | float) -> np.ndarray:
    """Return the smooth 1--3 km engagement-distance gate."""
    distance = np.asarray(distance_m, dtype=np.float64)
    below = np.maximum(1000.0 - distance, 0.0) / 500.0
    above = np.maximum(distance - 3000.0, 0.0) / 5000.0
    return np.exp(-(below * below) - (above * above))


def smooth_ata_gate(ata_deg: np.ndarray | float) -> np.ndarray:
    return _sigmoid((30.0 - np.asarray(ata_deg, dtype=np.float64)) / 10.0)


def smooth_aa_gate(aa_deg: np.ndarray | float) -> np.ndarray:
    return _sigmoid((90.0 - np.asarray(aa_deg, dtype=np.float64)) / 15.0)


def pair_potential(
    distance_m: np.ndarray | float,
    ata_deg: np.ndarray | float,
    aa_deg: np.ndarray | float,
) -> np.ndarray:
    """Hierarchical distance -> ATA -> AA potential, bounded in [0, 1]."""
    distance_gate = smooth_distance_gate(distance_m)
    ata_gate = smooth_ata_gate(ata_deg)
    aa_gate = smooth_aa_gate(aa_deg)
    potential = (distance_gate + distance_gate * ata_gate + distance_gate * ata_gate * aa_gate) / 3.0
    return np.clip(potential, 0.0, 1.0)


def team_potential_from_observations(observations: np.ndarray, distance_scale: float) -> np.ndarray:
    """Parse the stable [B, 3, 55] actor-observation contract into team potential."""
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != EXPECTED_OBSERVATION_SHAPE:
        raise ValueError(f"observations must have shape [B, 3, 55], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("observations contain non-finite values")
    if float(distance_scale) <= 0.0:
        raise ValueError("distance_scale must be positive")

    team = np.zeros(values.shape[0], dtype=np.float64)
    for red_index in range(3):
        red_observation = values[:, red_index]
        eligible_potentials = []
        for start in ENEMY_BLOCK_STARTS:
            alive = red_observation[:, start + ENEMY_ALIVE_OFFSET] > 0.5
            visible = (
                (red_observation[:, start + ENEMY_DIRECT_OFFSET] > 0.5)
                | (red_observation[:, start + ENEMY_DATALINK_OFFSET] > 0.5)
            )
            not_killed = red_observation[:, start + ENEMY_KILLED_OFFSET] <= 0.5
            eligible = alive & visible & not_killed
            potential = pair_potential(
                red_observation[:, start + ENEMY_DISTANCE_OFFSET] * float(distance_scale),
                red_observation[:, start + ENEMY_ATA_OFFSET] * 180.0,
                red_observation[:, start + ENEMY_AA_OFFSET] * 180.0,
            )
            eligible_potentials.append(np.where(eligible, potential, 0.0))
        best_blue = np.maximum(eligible_potentials[0], eligible_potentials[1])
        red_alive = red_observation[:, SELF_ALIVE_INDEX] > 0.5
        team += np.where(red_alive, best_blue, 0.0)
    return team / 3.0


def potential_delta(
    current_observations: np.ndarray,
    next_observations: np.ndarray,
    done: np.ndarray,
    distance_scale: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute gamma*Phi(next)-Phi(current), with an absorbing zero terminal potential."""
    current = team_potential_from_observations(current_observations, distance_scale)
    following = team_potential_from_observations(next_observations, distance_scale)
    terminal = np.asarray(done, dtype=bool)
    if terminal.shape != current.shape:
        raise ValueError(f"done must have shape {current.shape}, got {terminal.shape}")
    following = np.where(terminal, 0.0, following)
    return float(gamma) * following - current


def apply_agp(
    base_rewards: np.ndarray,
    current_observations: np.ndarray,
    next_observations: np.ndarray,
    done: np.ndarray,
    distance_scale: float,
    *,
    gamma: float = 0.99,
    agp_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add the same training-only shaping reward to all three Red agents."""
    rewards = np.asarray(base_rewards, dtype=np.float32)
    if rewards.ndim != 2 or rewards.shape[1] != 3:
        raise ValueError(f"base_rewards must have shape [B, 3], got {rewards.shape}")
    raw = potential_delta(current_observations, next_observations, done, distance_scale, gamma)
    shaping = float(agp_lambda) * raw
    return rewards + shaping[:, None].astype(np.float32), raw, shaping
