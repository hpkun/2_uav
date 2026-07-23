"""NED 坐标中的 1v1 空战几何。"""
from dataclasses import dataclass

import numpy as np

from .math_utils import angle_difference
from .models import AircraftState


@dataclass(frozen=True)
class PairwiseGeometry:
    """攻击方相对唯一目标的几何量。"""

    relative_position: np.ndarray
    relative_velocity: np.ndarray
    distance: float
    ata: float
    aa: float
    los_yaw: float
    los_pitch: float
    yaw_error: float
    pitch_error: float


def compute_pairwise_geometry(own: AircraftState, target: AircraftState) -> PairwiseGeometry:
    """计算攻击方视角下数值稳定的相对几何。"""
    eps = 1e-8
    relative_position = target.as_array()[:3] - own.as_array()[:3]
    own_velocity = own.velocity_vector()
    target_velocity = target.velocity_vector()
    relative_velocity = target_velocity - own_velocity
    distance = float(np.linalg.norm(relative_position))
    protected_distance = max(distance, eps)
    own_speed = max(float(np.linalg.norm(own_velocity)), eps)
    target_speed = max(float(np.linalg.norm(target_velocity)), eps)
    ata_cosine = np.clip(np.dot(own_velocity, relative_position) / (own_speed * protected_distance), -1.0, 1.0)
    aa_cosine = np.clip(np.dot(target_velocity, relative_position) / (target_speed * protected_distance), -1.0, 1.0)
    los_yaw = float(np.arctan2(relative_position[1], relative_position[0]))
    horizontal_distance = float(np.hypot(relative_position[0], relative_position[1]))
    los_pitch = float(np.arctan2(-relative_position[2], horizontal_distance))
    geometry = PairwiseGeometry(
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        distance=distance,
        ata=float(np.arccos(ata_cosine)),
        aa=float(np.arccos(aa_cosine)),
        los_yaw=los_yaw,
        los_pitch=los_pitch,
        yaw_error=angle_difference(los_yaw, own.psi),
        pitch_error=los_pitch - own.theta,
    )
    scalars = [geometry.distance, geometry.ata, geometry.aa, geometry.los_yaw, geometry.los_pitch, geometry.yaw_error, geometry.pitch_error]
    if not np.all(np.isfinite(relative_position)) or not np.all(np.isfinite(relative_velocity)) or not np.all(np.isfinite(scalars)):
        raise ValueError("aircraft states must produce finite geometry")
    return geometry

