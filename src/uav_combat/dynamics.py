"""三自由度点质量动力学。"""
import numpy as np
from .models import AircraftState, ControlCommand


class PointMassDynamics:
    """严格采用 NED 坐标的六状态导数模型。"""
    def __init__(self, gravity: float = 9.81, eps: float = 1e-8) -> None:
        self.gravity = gravity
        self.eps = eps

    def derivatives(self, state: AircraftState, control: ControlCommand) -> np.ndarray:
        """计算 [x,y,z,v,theta,psi] 的时间导数。"""
        v, theta, psi = state.v, state.theta, state.psi
        ct, st = np.cos(theta), np.sin(theta)
        v_safe = max(abs(v), self.eps)
        ct_safe = np.copysign(max(abs(ct), self.eps), ct if ct != 0.0 else 1.0)
        g = self.gravity
        return np.array([
            v * ct * np.cos(psi), v * ct * np.sin(psi), -v * st,
            g * (control.nx - st),
            g / v_safe * (control.nz * np.cos(control.phi) - ct),
            g * control.nz * np.sin(control.phi) / (v_safe * ct_safe),
        ], dtype=float)

