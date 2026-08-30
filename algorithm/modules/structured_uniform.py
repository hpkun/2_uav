"""Structured HAPPO actor with pure masked-uniform entity pooling."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from env.mavuav import OBS_DIM
from .hrta import (
    ENEMY_ALIVE_INDEX,
    ENEMY_DATALINK_VISIBLE_INDEX,
    ENEMY_DIRECT_VISIBLE_INDEX,
    ENEMY_SLICES,
    FRIEND_ALIVE_INDEX,
    FRIEND_SLICES,
    SELF_SLICE,
    SELF_TYPE_SLICE,
    EnemyEncoder,
    FriendEncoder,
    SelfEncoder,
)


def masked_uniform_pool(
    embeddings: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool eligible entities and safely return zero for an empty set."""
    if embeddings.ndim < 2 or eligible.shape != embeddings.shape[:-1]:
        raise ValueError(
            "eligible shape must equal embeddings shape without its feature dimension: "
            f"embeddings={tuple(embeddings.shape)}, eligible={tuple(eligible.shape)}"
        )
    mask = eligible.to(dtype=torch.bool)
    mask_values = mask.to(dtype=embeddings.dtype)
    count = mask_values.sum(dim=-1, keepdim=True)
    weights = mask_values / count.clamp_min(1.0)
    safe_embeddings = torch.where(mask.unsqueeze(-1), embeddings, torch.zeros_like(embeddings))
    context = torch.einsum("...n,...nd->...d", weights, safe_embeddings)
    return context, weights


class StructuredUniformActor(nn.Module):
    """HRTA-compatible structured representation without learned attention."""

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
            raise ValueError(
                f"StructuredUniformActor requires the existing {OBS_DIM}D observation contract"
            )
        if min(entity_dim, role_dim, fusion_hidden_dim, action_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.entity_dim = int(entity_dim)
        self.role_dim = int(role_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.self_encoder = SelfEncoder(entity_dim)
        self.friend_encoder = FriendEncoder(entity_dim)
        self.enemy_encoder = EnemyEncoder(entity_dim)
        self.role_embedding = nn.Sequential(nn.Linear(3, role_dim), nn.Tanh())
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
            raise ValueError(
                f"observations must have final dimension {OBS_DIM}, got {observations.shape}"
            )
        self_block = observations[..., SELF_SLICE]
        friend_blocks = torch.stack(
            [observations[..., block] for block in FRIEND_SLICES], dim=-2,
        )
        enemy_blocks = torch.stack(
            [observations[..., block] for block in ENEMY_SLICES], dim=-2,
        )
        return self_block, friend_blocks, enemy_blocks

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self_block, friend_blocks, enemy_blocks = self._blocks(observations)
        self_embedding = self.self_encoder(self_block)
        role_embedding = self.role_embedding(self_block[..., SELF_TYPE_SLICE])

        friend_embeddings = self.friend_encoder(friend_blocks)
        friend_eligible = friend_blocks[..., FRIEND_ALIVE_INDEX] > 0.5
        friend_context, friend_weights = masked_uniform_pool(
            friend_embeddings, friend_eligible,
        )

        enemy_embeddings = self.enemy_encoder(enemy_blocks)
        enemy_alive = enemy_blocks[..., ENEMY_ALIVE_INDEX] > 0.5
        enemy_visible = (
            (enemy_blocks[..., ENEMY_DIRECT_VISIBLE_INDEX] > 0.5)
            | (enemy_blocks[..., ENEMY_DATALINK_VISIBLE_INDEX] > 0.5)
        )
        enemy_context, enemy_weights = masked_uniform_pool(
            enemy_embeddings, enemy_alive & enemy_visible,
        )
        features = torch.cat(
            (self_embedding, role_embedding, friend_context, enemy_context), dim=-1,
        )
        return features, {
            "friend_pooling_weights": friend_weights,
            "enemy_pooling_weights": enemy_weights,
        }

    def forward_features(
        self, observations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.encode(observations)

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


class StructuredUniformIndependentActors(nn.Module):
    """Three independent structured-uniform actors."""

    def __init__(self, num_agents: int = 3, **actor_kwargs: Any) -> None:
        super().__init__()
        self.actors = nn.ModuleList(
            [StructuredUniformActor(**actor_kwargs) for _ in range(num_agents)]
        )


__all__ = [
    "StructuredUniformActor", "StructuredUniformIndependentActors", "masked_uniform_pool",
]
