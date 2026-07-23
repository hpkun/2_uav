"""固定控制量下的经典 RK4 积分。"""
import numpy as np
from .dynamics import PointMassDynamics
from .math_utils import safe_clip, wrap_angle
from .models import AircraftSpec, AircraftState, ControlCommand


class RK4Integrator:
    """在单个固定时间步中执行四阶 Runge-Kutta 积分。"""
    def __init__(self, dt: float = 0.1) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt

    @staticmethod
    def _state(vector: np.ndarray, alive: bool) -> AircraftState:
        return AircraftState(*map(float, vector), alive=alive)

    def step(self, state: AircraftState, control: ControlCommand, dynamics: PointMassDynamics, spec: AircraftSpec) -> AircraftState:
        """积分一步并施加速度、俯仰和航向约束。"""
        y, h = state.as_array(), self.dt
        f = dynamics.derivatives
        k1 = f(self._state(y, state.alive), control)
        k2 = f(self._state(y + h * k1 / 2, state.alive), control)
        k3 = f(self._state(y + h * k2 / 2, state.alive), control)
        k4 = f(self._state(y + h * k3, state.alive), control)
        result = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        result[3] = safe_clip(result[3], spec.v_min, spec.v_max)
        result[4] = safe_clip(result[4], spec.theta_min, spec.theta_max)
        result[5] = wrap_angle(result[5])
        return self._state(result, state.alive)

