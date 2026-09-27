"""Rollout-baseline-centered loss credit for LSA-RGAA-HAPPO."""
from __future__ import annotations

import torch


LSA_RGAA_METHOD = "lsa_rgaa"
LSA_RGAA_VERSION = 1
LSA_CREDIT_SEMANTICS = "rollout_baseline_centered_loss_advantage_v1"


def compute_loss_advantage(
    loss_returns: torch.Tensor,
    rollout_loss_values: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed on-policy loss advantage without normalization."""
    if loss_returns.shape != rollout_loss_values.shape:
        raise ValueError("loss returns and rollout loss values must have identical shape")
    return loss_returns - rollout_loss_values
