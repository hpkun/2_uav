"""Conflict-aware relational extensions for CR-RGAA-HAPPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from env.mavuav import OBS_DIM
from .rgaa import RoleValueNetwork


CR_RGAA_METHOD = "cr_rgaa"
CR_RGAA_VERSION = 1
CR_GATE_VERSION = "analytic_consistency_sigmoid_v1"
CR_RELATIONAL_CRITIC_VERSION = 1


def cr_rgaa_auxiliary_seed(training_seed: int) -> int:
    """Derive a stable CR-only seed without relying on Python's hash."""
    return (int(training_seed) + 0x43525247) % (2**63 - 1)


@dataclass(frozen=True)
class RelationalRoleOutput:
    values: torch.Tensor
    residuals: torch.Tensor
    attention_weights: torch.Tensor


class RelationalRoleValueNetwork(nn.Module):
    """Local role values plus one masked relational-attention correction."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        hidden_dim: int = 128,
        relational_dim: int = 64,
        attention_heads: int = 4,
        relational_value_coef: float = 1.0,
    ) -> None:
        super().__init__()
        if relational_dim <= 0 or attention_heads <= 0 or relational_dim % attention_heads:
            raise ValueError("relational_dim must be positive and divisible by attention_heads")
        if relational_value_coef < 0.0:
            raise ValueError("relational_value_coef cannot be negative")
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.relational_dim = int(relational_dim)
        self.attention_heads = int(attention_heads)
        self.relational_value_coef = float(relational_value_coef)

        self.local_mav = RoleValueNetwork(self.observation_dim, self.hidden_dim)
        self.local_uav = RoleValueNetwork(self.observation_dim, self.hidden_dim)
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, self.relational_dim), nn.Tanh(),
        )
        self.role_embedding = nn.Embedding(2, self.relational_dim)
        self.attention = nn.MultiheadAttention(
            self.relational_dim, self.attention_heads, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.relational_dim)
        self.mav_correction = self._correction_head()
        self.uav_correction = self._correction_head()
        self._zero_initialize_corrections()

    def _correction_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * self.relational_dim, self.relational_dim), nn.Tanh(),
            nn.Linear(self.relational_dim, 1),
        )

    def _zero_initialize_corrections(self) -> None:
        for head in (self.mav_correction, self.uav_correction):
            output = head[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        observations: torch.Tensor,
        active_masks: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | RelationalRoleOutput:
        if observations.ndim != 3 or observations.shape[1:] != (4, self.observation_dim):
            raise ValueError(
                f"observations must have shape [B,4,{self.observation_dim}]"
            )
        if active_masks.shape != observations.shape[:2]:
            raise ValueError("active_masks must have shape [B,4]")
        active = active_masks > 0.5
        batch = observations.shape[0]

        mav_local = self.local_mav(observations[:, 0])
        uav_local = self.local_uav(observations[:, 1:].reshape(-1, self.observation_dim)).reshape(
            batch, 3,
        )
        local_values = torch.cat((mav_local[:, None], uav_local), dim=1)

        role_ids = torch.tensor([0, 1, 1, 1], device=observations.device)
        tokens = self.observation_encoder(observations) + self.role_embedding(role_ids)[None]
        key_padding_mask = ~active
        all_inactive = key_padding_mask.all(dim=1)
        safe_key_padding_mask = key_padding_mask.clone()
        if all_inactive.any():
            safe_key_padding_mask[all_inactive, 0] = False
        attended, weights = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=safe_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.attention_norm(tokens + attended)
        context = context * active.unsqueeze(-1)
        features = torch.cat((tokens, context), dim=-1)
        mav_residual = self.mav_correction(features[:, 0]).squeeze(-1)
        uav_residual = self.uav_correction(features[:, 1:].reshape(-1, 2 * self.relational_dim)).reshape(
            batch, 3,
        )
        residuals = torch.cat((mav_residual[:, None], uav_residual), dim=1)
        residuals = residuals * active
        values = (local_values + self.relational_value_coef * residuals) * active
        weights = weights.masked_fill(~active[:, None, :, None], 0.0)
        weights = weights.masked_fill(~active[:, None, None, :], 0.0)
        if return_details:
            return RelationalRoleOutput(values, residuals, weights)
        return values

    def architecture(self) -> dict[str, Any]:
        return {
            "type": "local_plus_relational_residual",
            "observation_dim": self.observation_dim,
            "local_hidden_dim": self.hidden_dim,
            "local_hidden_layers": [self.hidden_dim, self.hidden_dim],
            "local_activation": "tanh",
            "local_sharing": {"MAV": "independent", "UAV1-UAV3": "shared"},
            "relational_dim": self.relational_dim,
            "attention_heads": self.attention_heads,
            "attention_layers": 1,
            "attention_residual_layer_norm": True,
            "role_embeddings": {"MAV": 0, "UAV": 1},
            "correction_hidden_dim": self.relational_dim,
            "correction_sharing": {"MAV": "independent", "UAV1-UAV3": "shared"},
            "correction_zero_initialized": True,
            "relational_value_coef": self.relational_value_coef,
            "output_dim": 4,
        }


def conflict_aware_fusion(
    team_normalized: torch.Tensor,
    role_normalized: torch.Tensor,
    *,
    role_advantage_coef: float,
    beta: float,
    lambda_floor_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the deterministic analytic consistency gate without renormalizing."""
    if role_advantage_coef < 0.0 or beta < 0.0:
        raise ValueError("role_advantage_coef and beta must be non-negative")
    if not 0.0 <= lambda_floor_ratio <= 1.0:
        raise ValueError("lambda_floor_ratio must lie in [0,1]")
    consistency = team_normalized * role_normalized
    gate = torch.sigmoid(float(beta) * consistency)
    adaptive_lambda = float(role_advantage_coef) * (
        float(lambda_floor_ratio) + (1.0 - float(lambda_floor_ratio)) * gate
    )
    combined = team_normalized + adaptive_lambda * role_normalized
    return combined, adaptive_lambda, consistency


def attention_diagnostics(
    attention_weights: torch.Tensor,
    active_masks: torch.Tensor,
) -> dict[str, float]:
    """Summarize head-averaged attention over active queries and keys only."""
    active = active_masks > 0.5
    weights = attention_weights.mean(dim=1)
    entropy_values: list[torch.Tensor] = []
    self_mass: list[torch.Tensor] = []
    mav_to_uav: list[torch.Tensor] = []
    uav_to_mav: list[torch.Tensor] = []
    uav_to_other: list[torch.Tensor] = []
    for row in range(weights.shape[0]):
        valid = torch.nonzero(active[row], as_tuple=False).squeeze(-1)
        valid_count = int(valid.numel())
        if not valid_count:
            continue
        for query_tensor in valid:
            query = int(query_tensor.item())
            probability = weights[row, query, valid]
            probability = probability / probability.sum().clamp_min(1e-12)
            if valid_count > 1:
                entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
                entropy_values.append(entropy / torch.log(probability.new_tensor(float(valid_count))))
            else:
                entropy_values.append(probability.new_zeros(()))
            self_mass.append(weights[row, query, query])
            if query == 0:
                uav_keys = valid[valid > 0]
                if uav_keys.numel():
                    mav_to_uav.append(weights[row, query, uav_keys].sum())
            else:
                if active[row, 0]:
                    uav_to_mav.append(weights[row, query, 0])
                other = valid[(valid > 0) & (valid != query)]
                if other.numel():
                    uav_to_other.append(weights[row, query, other].sum())

    def mean_or_zero(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().item()) if values else 0.0

    return {
        "cr_attention_entropy": mean_or_zero(entropy_values),
        "cr_attention_self_mass": mean_or_zero(self_mass),
        "cr_attention_mav_to_uav_mass": mean_or_zero(mav_to_uav),
        "cr_attention_uav_to_mav_mass": mean_or_zero(uav_to_mav),
        "cr_attention_uav_to_other_uav_mass": mean_or_zero(uav_to_other),
    }
