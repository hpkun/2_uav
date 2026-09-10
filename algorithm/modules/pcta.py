"""Pursuit-Consistent Target Attention actors for the canonical 100D observation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from env.mavuav import BLUE_IDS, OBS_DIM, RED_IDS


SELF_FRIEND_SLICE = slice(0, 44)
ENEMY_SLICES = tuple(slice(44 + 14 * index, 58 + 14 * index) for index in range(len(BLUE_IDS)))
ENEMY_ALIVE_INDEX = 9
ENEMY_DIRECT_VISIBLE_INDEX = 10
ENEMY_DATALINK_VISIBLE_INDEX = 11


def enemy_valid_mask(observations: torch.Tensor) -> torch.Tensor:
    """Return the alive-and-team-visible mask for the four fixed Blue slots."""
    blocks = torch.stack([observations[..., block] for block in ENEMY_SLICES], dim=-2)
    alive = blocks[..., ENEMY_ALIVE_INDEX] > 0.5
    visible = (
        (blocks[..., ENEMY_DIRECT_VISIBLE_INDEX] > 0.5)
        | (blocks[..., ENEMY_DATALINK_VISIBLE_INDEX] > 0.5)
    )
    return alive & visible


class PCTAActor(nn.Module):
    """Target-aware actor with a shared encoder across the four Blue slots."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        context_dim: int = 64,
        enemy_dim: int = 32,
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_dim != OBS_DIM:
            raise ValueError(f"PCTAActor requires the existing {OBS_DIM}D observation contract")
        if min(action_dim, context_dim, enemy_dim, hidden_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.enemy_dim = int(enemy_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(44, context_dim), nn.Tanh(),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(14, enemy_dim), nn.Tanh(),
        )
        self.query = nn.Linear(context_dim, enemy_dim)
        self.action_head = nn.Sequential(
            nn.Linear(context_dim + enemy_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.epsilon = 1e-6

    @staticmethod
    def enemy_blocks(observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError(f"observations must have final dimension {OBS_DIM}, got {observations.shape}")
        return torch.stack([observations[..., block] for block in ENEMY_SLICES], dim=-2)

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError(f"observations must have final dimension {OBS_DIM}, got {observations.shape}")
        context = self.context_encoder(observations[..., SELF_FRIEND_SLICE])
        enemies = self.enemy_blocks(observations)
        embeddings = self.enemy_encoder(enemies)
        query = self.query(context)
        scores = torch.einsum("...d,...nd->...n", query, embeddings) / math.sqrt(self.enemy_dim)
        valid = enemy_valid_mask(observations)
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(masked_scores, dim=-1)
        attention = torch.where(valid, attention, torch.zeros_like(attention))
        denominator = attention.sum(dim=-1, keepdim=True)
        attention = torch.where(
            denominator > 0.0,
            attention / denominator.clamp_min(torch.finfo(attention.dtype).eps),
            torch.zeros_like(attention),
        )
        aggregate = torch.einsum("...n,...nd->...d", attention, embeddings)
        features = torch.cat((context, aggregate), dim=-1)
        return features, {"enemy_attention": attention, "enemy_valid_mask": valid}

    def attention_weights(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encode(observations)[1]["enemy_attention"]

    def _distribution(self, observations: torch.Tensor) -> Normal:
        features, _ = self.encode(observations)
        mean = self.action_head(features)
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())

    def sample(
        self, observations: torch.Tensor, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - actions.square() + self.epsilon)
        return actions, log_probs.sum(dim=-1)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        raw = torch.atanh(clipped)
        distribution = self._distribution(observations)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - clipped.square() + self.epsilon)
        return log_probs.sum(dim=-1), distribution.entropy().sum(dim=-1)


class PCTAIndependentActors(nn.Module):
    """Parameter-independent PCTA actors for all Red aircraft."""

    def __init__(self, num_agents: int = len(RED_IDS), **actor_kwargs: Any) -> None:
        super().__init__()
        self.actors = nn.ModuleList([PCTAActor(**actor_kwargs) for _ in range(num_agents)])


@dataclass
class PCTAConsistencyResult:
    raw_loss: torch.Tensor
    valid_pairs: int
    attention_entropy_sum: float
    target_switches: int


def pursuit_consistency(
    actor: PCTAActor,
    observations: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    active_masks: torch.Tensor,
) -> PCTAConsistencyResult:
    """Compute valid-target temporal consistency over a ``[T,N,Obs]`` rollout."""
    if observations.ndim != 3 or observations.shape[-1] != OBS_DIM:
        raise ValueError("PCTA temporal observations must have shape [T,N,OBS_DIM]")
    if observations.shape[0] < 2:
        zero = actor.log_std.sum() * 0.0
        return PCTAConsistencyResult(zero, 0, 0.0, 0)
    attention = actor.attention_weights(observations)
    valid_targets = enemy_valid_mask(observations)
    previous_attention = attention[:-1].detach()
    current_attention = attention[1:]
    previous_valid = valid_targets[:-1]
    current_valid = valid_targets[1:]
    previous_target = previous_attention.argmax(dim=-1)
    previous_has_target = previous_valid.any(dim=-1)
    previous_target_still_valid = current_valid.gather(-1, previous_target.unsqueeze(-1)).squeeze(-1)
    transition_valid = (
        ~(terminated[:-1].bool() | truncated[:-1].bool())
        & (active_masks[:-1] > 0.5)
        & (active_masks[1:] > 0.5)
        & previous_has_target
        & previous_target_still_valid
    )
    count = int(transition_valid.sum().item())
    if count == 0:
        zero = actor.log_std.sum() * 0.0
        return PCTAConsistencyResult(zero, 0, 0.0, 0)
    differences = (current_attention - previous_attention).square().mean(dim=-1)
    raw_loss = differences[transition_valid].mean()
    selected = current_attention[transition_valid]
    attention_entropy = -(selected * selected.clamp_min(1e-12).log()).sum(dim=-1)
    switches = int((selected.argmax(dim=-1) != previous_target[transition_valid]).sum().item())
    return PCTAConsistencyResult(
        raw_loss=raw_loss,
        valid_pairs=count,
        attention_entropy_sum=float(attention_entropy.detach().sum().item()),
        target_switches=switches,
    )


__all__ = [
    "PCTAActor", "PCTAIndependentActors", "PCTAConsistencyResult",
    "enemy_valid_mask", "pursuit_consistency", "ENEMY_SLICES",
]
