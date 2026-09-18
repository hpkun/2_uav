"""Rollout buffer extension for fixed counterfactual reward components."""
from __future__ import annotations

import numpy as np

from algorithm.common.buffer import RolloutBuffer
from .counterfactual_credit import compute_component_lambda_returns


class CreditRolloutBuffer(RolloutBuffer):
    def __init__(self, horizon: int, num_envs: int, component_count: int, **kwargs) -> None:
        super().__init__(horizon, num_envs, **kwargs)
        shape = (self.horizon, self.num_envs, int(component_count))
        self.component_count = int(component_count)
        self.credit_rewards = np.zeros(shape, np.float32)
        self.credit_values = np.zeros(shape, np.float32)
        self.credit_returns = np.zeros(shape, np.float32)

    def insert(self, *args, credit_rewards, **kwargs) -> None:
        index = self.position
        super().insert(*args, **kwargs)
        values = np.asarray(credit_rewards, dtype=np.float32)
        if values.shape != (self.num_envs, self.component_count):
            raise ValueError("credit reward shape mismatch")
        self.credit_rewards[index] = values

    def compute_credit_returns(
        self,
        values: np.ndarray,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("component GAE requires a complete fixed-horizon rollout")
        values = np.asarray(values, dtype=np.float32)
        if values.shape != self.credit_values.shape:
            raise ValueError("credit value shape mismatch")
        self.credit_values[:] = values
        self.credit_returns[:] = compute_component_lambda_returns(
            self.credit_rewards, self.credit_values, last_values,
            self.terminated, self.truncated, gamma, gae_lambda,
        )
