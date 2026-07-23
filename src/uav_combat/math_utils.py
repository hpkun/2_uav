"""数值稳定的基础数学工具。"""
import numpy as np


def wrap_angle(angle: float) -> float:
    """将弧度角包装到 [-pi, pi)。"""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def angle_difference(target: float, current: float) -> float:
    """返回从当前角到目标角的最短有符号误差。"""
    return wrap_angle(target - current)


def safe_clip(value: float, lower: float, upper: float) -> float:
    """裁剪有限标量，并拒绝非有限输入。"""
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return float(np.clip(value, lower, upper))


def safe_norm(vector: np.ndarray, eps: float = 1e-8) -> float:
    """返回不小于 eps 的欧氏范数。"""
    return max(float(np.linalg.norm(np.asarray(vector, dtype=float))), eps)

