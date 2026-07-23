"""论文参考分段奖励与项目原有耦合差奖励。"""
from typing import Any
import numpy as np
from .geometry import compute_pairwise_geometry
from .models import AircraftState


def _empty() -> dict[str, float]:
    return {key: 0.0 for key in ("reward_terminal", "reward_boundary", "reward_guide", "reward_position", "reward_threat", "reward_total")}


def madsac_segmented_reward(own: AircraftState, target: AircraftState, own_team: str, reason: str | None, outcome: str | None) -> dict[str, float]:
    """计算 MADSAC 数值参考奖励；pitch_error 近似论文 HA 属项目适配。"""
    result = _empty(); geometry = compute_pairwise_geometry(own, target); reverse = compute_pairwise_geometry(target, own)
    if reason == "collision": result["reward_terminal"] = -10.0
    elif reason in {"red_kill", "blue_kill", "mutual_kill"}:
        result["reward_terminal"] = 10.0 if outcome == own_team else -10.0
    elif reason in {"altitude_boundary", "xy_boundary", "boundary"}:
        result["reward_boundary"] = 10.0 if outcome == own_team else (-10.0 if outcome in {"red", "blue"} else -10.0)
    degrees = np.pi / 180.0; tolerance = 1e-12; pitch = abs(geometry.pitch_error)
    within = lambda value, limit: value <= limit + tolerance
    if geometry.distance >= 4000.0 and within(geometry.ata, 30 * degrees) and within(pitch, 30 * degrees):
        result["reward_guide"] = 0.001
    if geometry.distance <= 4000.0 and within(geometry.aa, 30 * degrees):
        if within(geometry.ata, 5 * degrees) and within(pitch, 5 * degrees): result["reward_position"] = 0.1
        elif within(geometry.ata, 15 * degrees) and within(pitch, 15 * degrees): result["reward_position"] = 0.02
        elif within(geometry.ata, 30 * degrees) and within(pitch, 30 * degrees): result["reward_position"] = 0.01
    reverse_pitch = abs(reverse.pitch_error)
    if reverse.distance <= 4000.0 and within(reverse.aa, 30 * degrees):
        if within(reverse.ata, 5 * degrees) and within(reverse_pitch, 5 * degrees): result["reward_threat"] = -0.15
        elif within(reverse.ata, 15 * degrees) and within(reverse_pitch, 15 * degrees): result["reward_threat"] = -0.025
        elif within(reverse.ata, 30 * degrees) and within(reverse_pitch, 30 * degrees): result["reward_threat"] = -0.015
    result["reward_total"] = float(sum(value for key, value in result.items() if key != "reward_total"))
    if not np.all(np.isfinite(list(result.values()))): raise FloatingPointError("non-finite segmented reward")
    return result


def coupled_difference_rewards(red_score: float, blue_score: float, scale: float, terminal_reward: float, reason: str | None, outcome: str | None) -> dict[str, dict[str, float]]:
    """保留项目原有零和耦合差奖励用于消融。"""
    dense = scale * (red_score - blue_score); terminal = {"red": 0.0, "blue": 0.0}
    if outcome == "red": terminal = {"red": terminal_reward, "blue": -terminal_reward}
    elif outcome == "blue": terminal = {"red": -terminal_reward, "blue": terminal_reward}
    elif reason in {"mutual_kill", "collision"}: terminal = {"red": -terminal_reward, "blue": -terminal_reward}
    return {team: {"dense": float(dense if team == "red" else -dense), "terminal": terminal[team]} for team in ("red", "blue")}
