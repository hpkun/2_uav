"""Heterogeneous Role-Target Attention actor for structured 55D observations.

This optional module is intentionally independent from the vanilla HAPPO
trainer.  It changes actor representation only and preserves the existing
tanh-squashed Gaussian action distribution.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from env.mavuav import OBS_DIM


SELF_SLICE = slice(0, 11)
FRIEND_SLICES = (slice(11, 22), slice(22, 33))
ENEMY_SLICES = (slice(33, 44), slice(44, 55))

SELF_TYPE_SLICE = slice(7, 10)
FRIEND_ALIVE_INDEX = 7
ENEMY_ALIVE_INDEX = 6
ENEMY_DIRECT_VISIBLE_INDEX = 7
ENEMY_DATALINK_VISIBLE_INDEX = 8


class _EntityEncoder(nn.Module):
    def __init__(self, entity_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(11, entity_dim), nn.Tanh(),
            nn.Linear(entity_dim, entity_dim), nn.Tanh(),
        )

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        return self.network(block)


class SelfEncoder(_EntityEncoder):
    """Encode the 11D self block."""


class FriendEncoder(_EntityEncoder):
    """Shared encoder for both 11D friendly blocks."""


class EnemyEncoder(_EntityEncoder):
    """Shared encoder for both 11D Blue blocks."""


def _masked_single_head_attention(
    query: torch.Tensor,
    values: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically safe scaled dot-product pooling over two entities."""
    scores = torch.einsum("...d,...nd->...n", query, values) / math.sqrt(query.shape[-1])
    mask = eligible.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )
    context = torch.einsum("...n,...nd->...d", weights, values)
    return context, weights


class HRTAActor(nn.Module):
    """Structured actor with role-conditioned friend and enemy attention."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        entity_dim: int = 32,
        role_dim: int = 16,
        fusion_hidden_dim: int = 64,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_dim != OBS_DIM:
            raise ValueError(f"HRTAActor requires the existing {OBS_DIM}D observation contract")
        if min(entity_dim, role_dim, fusion_hidden_dim, action_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.entity_dim = int(entity_dim)
        self.self_encoder = SelfEncoder(entity_dim)
        self.friend_encoder = FriendEncoder(entity_dim)
        self.enemy_encoder = EnemyEncoder(entity_dim)
        self.role_embedding = nn.Sequential(nn.Linear(3, role_dim), nn.Tanh())
        self.friend_query = nn.Sequential(nn.Linear(entity_dim + role_dim, entity_dim), nn.Tanh())
        self.enemy_query = nn.Sequential(
            nn.Linear(entity_dim + role_dim + entity_dim, entity_dim), nn.Tanh(),
        )
        fusion_dim = entity_dim + role_dim + entity_dim + entity_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim), nn.Tanh(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.Tanh(),
            nn.Linear(fusion_hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.epsilon = 1e-6

    @staticmethod
    def _blocks(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError(f"observations must have final dimension {OBS_DIM}, got {observations.shape}")
        self_block = observations[..., SELF_SLICE]
        friend_blocks = torch.stack([observations[..., block] for block in FRIEND_SLICES], dim=-2)
        enemy_blocks = torch.stack([observations[..., block] for block in ENEMY_SLICES], dim=-2)
        return self_block, friend_blocks, enemy_blocks

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return fused actor features and pure, per-call attention diagnostics."""
        self_block, friend_blocks, enemy_blocks = self._blocks(observations)
        self_embedding = self.self_encoder(self_block)
        role_embedding = self.role_embedding(self_block[..., SELF_TYPE_SLICE])

        friend_embeddings = self.friend_encoder(friend_blocks)
        friend_query = self.friend_query(torch.cat((self_embedding, role_embedding), dim=-1))
        friend_eligible = friend_blocks[..., FRIEND_ALIVE_INDEX] > 0.5
        friend_context, friend_attention = _masked_single_head_attention(
            friend_query, friend_embeddings, friend_eligible,
        )

        enemy_embeddings = self.enemy_encoder(enemy_blocks)
        enemy_query = self.enemy_query(
            torch.cat((self_embedding, role_embedding, friend_context), dim=-1),
        )
        enemy_alive = enemy_blocks[..., ENEMY_ALIVE_INDEX] > 0.5
        enemy_visible = (
            (enemy_blocks[..., ENEMY_DIRECT_VISIBLE_INDEX] > 0.5)
            | (enemy_blocks[..., ENEMY_DATALINK_VISIBLE_INDEX] > 0.5)
        )
        enemy_context, enemy_attention = _masked_single_head_attention(
            enemy_query, enemy_embeddings, enemy_alive & enemy_visible,
        )
        features = torch.cat(
            (self_embedding, role_embedding, friend_context, enemy_context), dim=-1,
        )
        diagnostics = {
            "enemy_attention": enemy_attention,
            "friend_attention": friend_attention,
        }
        return features, diagnostics

    def forward_features(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.encode(observations)

    def attention_weights(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encode(observations)[1]["enemy_attention"]

    def _distribution(self, observations: torch.Tensor) -> Normal:
        features, _ = self.encode(observations)
        mean = self.action_head(features)
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())

    def sample(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_probs = distribution.log_prob(raw) - torch.log(
            1.0 - actions.square() + self.epsilon,
        )
        return actions, log_probs.sum(dim=-1)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        raw = torch.atanh(clipped)
        distribution = self._distribution(observations)
        log_probs = distribution.log_prob(raw) - torch.log(
            1.0 - clipped.square() + self.epsilon,
        )
        return log_probs.sum(dim=-1), distribution.entropy().sum(dim=-1)


class HRTAIndependentActors(nn.Module):
    """Three parameter-independent HRTA actors for MAV, UAV1, and UAV2."""

    def __init__(self, num_agents: int = 3, **actor_kwargs: Any) -> None:
        super().__init__()
        self.actors = nn.ModuleList([HRTAActor(**actor_kwargs) for _ in range(num_agents)])


__all__ = [
    "HRTAActor", "HRTAIndependentActors", "SelfEncoder", "FriendEncoder", "EnemyEncoder",
]
