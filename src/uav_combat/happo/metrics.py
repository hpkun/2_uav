"""Small metric helpers for HAPPO."""
from __future__ import annotations

import numpy as np


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y <= 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def finite_numeric_dict(row: dict) -> bool:
    for value in row.values():
        if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(float(value)):
            return False
    return True


__all__ = ["explained_variance", "finite_numeric_dict"]
