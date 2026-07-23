"""从归一化连续动作到动力学控制量的简化低层控制器。"""
import numpy as np
from .math_utils import angle_difference, safe_clip, wrap_angle
from .models import AircraftSpec, AircraftState, ControlCommand, TargetCommand


class TargetStateController:
    """项目自定义控制器，并非对未公开论文控制器的严格复现。"""
    def __init__(self, delta_yaw_max: float = np.pi, delta_pitch_max: float = np.pi / 3, delta_speed_max: float = 50.0, gravity: float = 9.81, yaw_endpoint_epsilon: float = 1e-6) -> None:
        if not 0.0 < yaw_endpoint_epsilon < np.pi:
            raise ValueError("yaw_endpoint_epsilon must be in (0, pi)")
        self.delta_yaw_max = delta_yaw_max
        self.delta_pitch_max = delta_pitch_max
        self.delta_speed_max = delta_speed_max
        self.gravity = gravity
        self.yaw_endpoint_epsilon = yaw_endpoint_epsilon

    def action_to_target(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> TargetCommand:
        """把三维归一化连续动作转换为目标状态。"""
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with shape (3,)")
        a = np.clip(action, -1.0, 1.0)
        effective_delta_yaw_max = min(abs(self.delta_yaw_max), np.pi - self.yaw_endpoint_epsilon)
        return TargetCommand(
            wrap_angle(state.psi + a[0] * effective_delta_yaw_max),
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
        magnitude = float(np.hypot(a_term, b_term))
        if a_term >= 0.0:
            nz_raw = magnitude
            phi_raw = float(np.arctan2(b_term, a_term))
        else:
            nz_raw = -magnitude
            phi_raw = float(np.arctan2(-b_term, -a_term))
        nz = safe_clip(nz_raw, spec.nz_min, spec.nz_max)
        phi = safe_clip(phi_raw, spec.phi_min, spec.phi_max)
        return ControlCommand(nx, nz, phi)

    def control_from_action(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> tuple[TargetCommand, ControlCommand]:
        """一次完成动作映射和低层控制计算。"""
        target = self.action_to_target(state, action, spec)
        return target, self.compute_control(state, target, spec)

    def diagnostics(self, state: AircraftState, target: TargetCommand, control: ControlCommand, spec: AircraftSpec) -> dict[str, float | bool]:
        """返回不改变控制行为的误差、限幅前后变化率和饱和诊断。"""
        yaw_error = angle_difference(target.desired_psi, state.psi); pitch_error = target.desired_theta - state.theta; speed_error = target.desired_v - state.v
        raw_yaw = spec.k_yaw * yaw_error; raw_pitch = spec.k_pitch * pitch_error; raw_accel = spec.k_speed * speed_error
        clipped_yaw = float(np.clip(raw_yaw, -spec.yaw_rate_max, spec.yaw_rate_max)); clipped_pitch = float(np.clip(raw_pitch, -spec.pitch_rate_max, spec.pitch_rate_max)); clipped_accel = float(np.clip(raw_accel, -spec.acceleration_max, spec.acceleration_max))
        tolerance = 1e-8
        return {"yaw_error": yaw_error, "pitch_error": pitch_error, "speed_error": speed_error, "unclipped_yaw_rate": raw_yaw, "unclipped_pitch_rate": raw_pitch, "unclipped_acceleration": raw_accel, "clipped_yaw_rate": clipped_yaw, "clipped_pitch_rate": clipped_pitch, "clipped_acceleration": clipped_accel, "nx": control.nx, "nz": control.nz, "phi": control.phi, "yaw_rate_saturated": abs(raw_yaw-clipped_yaw)>tolerance, "pitch_rate_saturated": abs(raw_pitch-clipped_pitch)>tolerance, "acceleration_saturated": abs(raw_accel-clipped_accel)>tolerance, "nx_saturated": min(abs(control.nx-spec.nx_min),abs(control.nx-spec.nx_max))<tolerance, "nz_saturated": min(abs(control.nz-spec.nz_min),abs(control.nz-spec.nz_max))<tolerance, "phi_saturated": min(abs(control.phi-spec.phi_min),abs(control.phi-spec.phi_max))<tolerance}
