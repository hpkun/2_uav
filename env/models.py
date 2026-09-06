"""Core data models for the MAV/UAV 4v4 environment."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AircraftSpec:
    """Only the performance limits used by the 3DOF model."""

    aircraft_type: str
    v_min: float
    v_max: float
    nx: tuple[float, float]
    ny: tuple[float, float]
    nz: tuple[float, float]

    def __post_init__(self) -> None:
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be smaller than v_max")
        for name in ("nx", "ny", "nz"):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ValueError(f"{name} lower limit must be smaller than upper limit")


@dataclass
class AircraftState:
    """Physical state ``[x, y, h, v, theta, psi]`` with altitude positive up."""

    x: float
    y: float
    h: float
    v: float
    theta: float
    psi: float
    alive: bool = True

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.h, self.v, self.theta, self.psi], dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray, *, alive: bool = True) -> "AircraftState":
        return cls(*(float(v) for v in values), alive=alive)

    def copy(self) -> "AircraftState":
        return AircraftState(*self.as_array(), alive=self.alive)

    def velocity_vector(self) -> np.ndarray:
        ct = np.cos(self.theta)
        return np.asarray(
            [self.v * ct * np.cos(self.psi), self.v * ct * np.sin(self.psi), self.v * np.sin(self.theta)],
            dtype=np.float64,
        )

    @property
    def altitude(self) -> float:
        return self.h


@dataclass(frozen=True)
class OverloadCommand:
    nx: float
    ny: float
    nz: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.nx, self.ny, self.nz], dtype=np.float64)


@dataclass
class Aircraft:
    aircraft_id: str
    team: str
    spec: AircraftSpec
    state: AircraftState
    inactive_cause: str | None = None
