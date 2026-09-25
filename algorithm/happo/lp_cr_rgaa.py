"""Directional conflict gate for LP-CR-RGAA-HAPPO."""
from __future__ import annotations

import torch


LP_CR_RGAA_METHOD = "lp_cr_rgaa"
LP_CR_RGAA_VERSION = 1
LP_CR_GATE_VERSION = "loss_preserving_directional_conflict_v1"


def loss_preserving_directional_fusion(
    team_normalized: torch.Tensor,
    role_normalized: torch.Tensor,
    *,
    role_advantage_coef: float,
    beta: float,
    lambda_floor_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Suppress only locally optimistic, globally harmful role credit.

    Team-positive/role-negative corrections retain their full role coefficient,
    preserving the learned own-loss penalty direction.
    """
    if role_advantage_coef < 0.0 or beta < 0.0:
        raise ValueError("role_advantage_coef and beta must be non-negative")
    if not 0.0 <= lambda_floor_ratio <= 1.0:
        raise ValueError("lambda_floor_ratio must lie in [0,1]")
    consistency = team_normalized * role_normalized
    suppress = (team_normalized < 0.0) & (role_normalized > 0.0)
    conflict_magnitude = torch.where(suppress, -consistency, torch.zeros_like(consistency))
    lambda_min = float(role_advantage_coef) * float(lambda_floor_ratio)
    suppressed_lambda = lambda_min + (
        float(role_advantage_coef) - lambda_min
    ) * torch.exp(-float(beta) * conflict_magnitude)
    adaptive_lambda = torch.where(
        suppress, suppressed_lambda,
        torch.full_like(suppressed_lambda, float(role_advantage_coef)),
    )
    combined = team_normalized + adaptive_lambda * role_normalized
    return combined, adaptive_lambda, consistency
