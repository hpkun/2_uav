"""从归一化连续动作到动力学控制量的简化低层控制器。"""
import numpy as np
from .math_utils import angle_difference, safe_clip, wrap_angle
from .models import AircraftSpec, AircraftState, ControlCommand, TargetCommand


class TargetStateController:
    """项目自定义控制器，并非对未公开论文控制器的严格复现。"""
    def __init__(self, delta_yaw_max: float = np.pi, delta_pitch_max: float = np.pi / 3, delta_speed_max: float = 50.0, gravity: float = 9.81) -> None:
        self.delta_yaw_max = delta_yaw_max
        self.delta_pitch_max = delta_pitch_max
        self.delta_speed_max = delta_speed_max
        self.gravity = gravity

    def action_to_target(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> TargetCommand:
        """把三维归一化连续动作转换为目标状态。"""
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with shape (3,)")
        a = np.clip(action, -1.0, 1.0)
        return TargetCommand(
            wrap_angle(state.psi + a[0] * self.delta_yaw_max),
            safe_clip(state.theta + a[1] * self.delta_pitch_max, spec.theta_min, spec.theta_max),
            safe_clip(state.v + a[2] * self.delta_speed_max, spec.v_min, spec.v_max),
        )

    def compute_control(self, state: AircraftState, target: TargetCommand, spec: AircraftSpec) -> ControlCommand:
        """由限幅变化率反解切向/法向过载与滚转角。"""
        psi_dot = safe_clip(spec.k_yaw * angle_difference(target.desired_psi, state.psi), -spec.yaw_rate_max, spec.yaw_rate_max)
        theta_dot = safe_clip(spec.k_pitch * (target.desired_theta - state.theta), -spec.pitch_rate_max, spec.pitch_rate_max)
        v_dot = safe_clip(spec.k_speed * (target.desired_v - state.v), -spec.acceleration_max, spec.acceleration_max)
        nx = safe_clip(v_dot / self.gravity + np.sin(state.theta), spec.nx_min, spec.nx_max)
        a_term = np.cos(state.theta) + state.v / self.gravity * theta_dot
        b_term = state.v * np.cos(state.theta) / self.gravity * psi_dot
        nz = safe_clip(float(np.hypot(a_term, b_term)), spec.nz_min, spec.nz_max)
        phi = safe_clip(float(np.arctan2(b_term, a_term)), spec.phi_min, spec.phi_max)
        return ControlCommand(nx, nz, phi)

    def control_from_action(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> tuple[TargetCommand, ControlCommand]:
        """一次完成动作映射和低层控制计算。"""
        target = self.action_to_target(state, action, spec)
        return target, self.compute_control(state, target, spec)

