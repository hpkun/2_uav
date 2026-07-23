"""用于环境基线验证的单纯视线追击策略。"""
import numpy as np

from .geometry import compute_pairwise_geometry
from .models import Aircraft


class PurePursuitPolicy:
    """不预测、不规避且无状态切换的纯追击策略。"""

    def __init__(self, delta_yaw_max: float, delta_pitch_max: float, delta_speed_max: float, speed_margin: float = 20.0) -> None:
        if delta_pitch_max <= 0 or delta_speed_max <= 0:
            raise ValueError("action scales must be positive")
        self.effective_delta_yaw_max = min(abs(delta_yaw_max), np.pi - 1e-6)
        self.delta_pitch_max = delta_pitch_max
        self.delta_speed_max = delta_speed_max
        self.speed_margin = speed_margin

    def action(self, own: Aircraft, target: Aircraft) -> np.ndarray:
        """根据当前视线误差和目标速度生成三维连续动作。"""
        geometry = compute_pairwise_geometry(own.state, target.state)
        desired_speed = np.clip(target.state.v + self.speed_margin, own.spec.v_min, own.spec.v_max)
        return np.array([
            np.clip(geometry.yaw_error / self.effective_delta_yaw_max, -1.0, 1.0),
            np.clip(geometry.pitch_error / self.delta_pitch_max, -1.0, 1.0),
            np.clip((desired_speed - own.state.v) / self.delta_speed_max, -1.0, 1.0),
        ], dtype=float)

