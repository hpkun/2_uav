"""环境使用的基础数据模型。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftState:
    """NED 坐标中的六维点质量状态。"""
    x: float
    y: float
    z: float
    v: float
    theta: float
    psi: float
    alive: bool = True

    def as_array(self) -> np.ndarray:
        """返回动力学六维状态向量。"""
        return np.array([self.x, self.y, self.z, self.v, self.theta, self.psi], dtype=float)

    def copy(self) -> "AircraftState":
        """返回独立状态副本。"""
        return AircraftState(self.x, self.y, self.z, self.v, self.theta, self.psi, self.alive)

    def velocity_vector(self) -> np.ndarray:
        """返回 NED 坐标中的速度向量。"""
        ct = np.cos(self.theta)
        return np.array([self.v * ct * np.cos(self.psi), self.v * ct * np.sin(self.psi), -self.v * np.sin(self.theta)])

    @property
    def altitude(self) -> float:
        """返回海拔高度（向上为正）。"""
        return -self.z


@dataclass(frozen=True)
class AircraftSpec:
    """可配置的飞机性能和控制参数。"""
    v_min: float; v_max: float
    theta_min: float; theta_max: float
    nx_min: float; nx_max: float
    nz_min: float; nz_max: float
    phi_min: float; phi_max: float
    yaw_rate_max: float; pitch_rate_max: float; acceleration_max: float
    k_yaw: float; k_pitch: float; k_speed: float


@dataclass(frozen=True)
class TargetCommand:
    """期望航向、俯仰和速度。"""
    desired_psi: float
    desired_theta: float
    desired_v: float


@dataclass(frozen=True)
class ControlCommand:
    """动力学控制量：切向过载、法向过载和滚转角。"""
    nx: float
    nz: float
    phi: float


@dataclass
class Aircraft:
    """具有标识、阵营、统一规格和状态的飞行实体。"""
    aircraft_id: str
    team: str
    spec: AircraftSpec
    state: AircraftState
    role: str = "combat"
    sensor_range: float = float("inf")
    can_attack: bool = True

    def __post_init__(self) -> None:
        if self.role not in ("support", "combat"):
            raise ValueError(f"unknown aircraft role: {self.role!r}")
