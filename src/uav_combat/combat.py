"""确定性攻击判定与简化态势评分。"""
import numpy as np

from .geometry import compute_pairwise_geometry
from .models import AircraftState


class SimplifiedAttackModel:
    """一次有效几何攻击即摧毁目标的确定性模型。"""

    def __init__(self, distance_min: float, distance_max: float, ata_max: float, aa_max: float) -> None:
        if distance_min < 0 or distance_max < distance_min or ata_max < 0 or aa_max < 0:
            raise ValueError("invalid attack envelope")
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.ata_max = ata_max
        self.aa_max = aa_max

    def can_attack(self, attacker: AircraftState, target: AircraftState) -> bool:
        """判断目标是否位于唯一的确定性攻击区域内。"""
        geometry = compute_pairwise_geometry(attacker, target)
        return bool(
            self.distance_min <= geometry.distance <= self.distance_max
            and geometry.ata <= self.ata_max
            and geometry.aa <= self.aa_max
        )


def situation_score(attacker: AircraftState, target: AircraftState, preferred_distance: float, distance_scale: float) -> float:
    """计算 project-defined 角度—距离耦合态势评分。"""
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    geometry = compute_pairwise_geometry(attacker, target)
    ata_score = 0.5 * (1.0 + np.cos(geometry.ata))
    aa_score = 0.5 * (1.0 + np.cos(geometry.aa))
    distance_score = np.exp(-((geometry.distance - preferred_distance) / distance_scale) ** 2)
    return float(np.clip(distance_score * ata_score * aa_score, 0.0, 1.0))

