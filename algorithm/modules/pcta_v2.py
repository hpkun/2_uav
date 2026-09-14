"""PCTA-v2 actors with full-observation residual fusion and additive attention."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from env.mavuav import OBS_DIM, RED_IDS
from algorithm.modules.pcta import ENEMY_SLICES, enemy_valid_mask


SELF_FRIEND_SLICE = slice(0, 44)
OWN_ATTACK_STREAK_INDEX = 12


class PCTAv2Actor(nn.Module):
    """Four-head target attention augmenting, never replacing, the raw observation."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        context_dim: int = 64,
        enemy_dim: int = 32,
        target_dim: int = 32,
        attention_heads: int = 4,
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_dim != OBS_DIM:
            raise ValueError(f"PCTAv2Actor requires the existing {OBS_DIM}D observation contract")
        if min(action_dim, context_dim, enemy_dim, target_dim, attention_heads, hidden_dim) <= 0:
            raise ValueError("network dimensions and attention_heads must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.enemy_dim = int(enemy_dim)
        self.target_dim = int(target_dim)
        self.attention_heads = int(attention_heads)
        self.hidden_dim = int(hidden_dim)
        self.policy_fusion_dim = self.observation_dim + self.target_dim + len(ENEMY_SLICES)

        self.context_encoder = nn.Sequential(nn.Linear(44, context_dim), nn.Tanh())
        self.enemy_encoder = nn.Sequential(nn.Linear(14, enemy_dim), nn.Tanh())
        self.query_heads = nn.ModuleList(
            nn.Linear(context_dim, enemy_dim) for _ in range(attention_heads)
        )
        self.key_heads = nn.ModuleList(
            nn.Linear(enemy_dim, enemy_dim) for _ in range(attention_heads)
        )
        self.score_heads = nn.ModuleList(
            nn.Linear(enemy_dim, 1, bias=False) for _ in range(attention_heads)
        )
        self.pursuit_bias_raw = nn.Parameter(torch.zeros(attention_heads))
        self.target_projection = nn.Sequential(
            nn.Linear(attention_heads * enemy_dim, target_dim), nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.policy_fusion_dim, hidden_dim), nn.Tanh(),
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

    @property
    def pursuit_bias_gain(self) -> torch.Tensor:
        return F.softplus(self.pursuit_bias_raw)

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError(f"observations must have final dimension {OBS_DIM}, got {observations.shape}")
        context = self.context_encoder(observations[..., SELF_FRIEND_SLICE])
        enemy_blocks = self.enemy_blocks(observations)
        enemy_embeddings = self.enemy_encoder(enemy_blocks)
        valid = enemy_valid_mask(observations)
        pursuit_progress = enemy_blocks[..., OWN_ATTACK_STREAK_INDEX]
        beta = self.pursuit_bias_gain

        attention_heads = []
        head_features = []
        for head, (query_layer, key_layer, score_layer) in enumerate(
            zip(self.query_heads, self.key_heads, self.score_heads)
        ):
            query = query_layer(context).unsqueeze(-2)
            keys = key_layer(enemy_embeddings)
            base_scores = score_layer(torch.tanh(query + keys)).squeeze(-1)
            scores = base_scores + beta[head] * pursuit_progress
            masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = torch.softmax(masked_scores, dim=-1)
            attention = torch.where(valid, attention, torch.zeros_like(attention))
            denominator = attention.sum(dim=-1, keepdim=True)
            attention = torch.where(
                denominator > 0.0,
                attention / denominator.clamp_min(torch.finfo(attention.dtype).eps),
                torch.zeros_like(attention),
            )
            attention_heads.append(attention)
            head_features.append(torch.einsum("...n,...nd->...d", attention, enemy_embeddings))

        alpha_heads = torch.stack(attention_heads, dim=-2)
        alpha_mean = alpha_heads.mean(dim=-2)
        target_feature = self.target_projection(torch.cat(head_features, dim=-1))
        policy_feature = torch.cat((observations, target_feature, alpha_mean), dim=-1)
        return policy_feature, {
            "enemy_attention": alpha_mean,
            "enemy_attention_heads": alpha_heads,
            "enemy_valid_mask": valid,
            "target_feature": target_feature,
            "policy_feature": policy_feature,
            "pursuit_bias_gain": beta,
        }

    def attention_weights(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encode(observations)[1]["enemy_attention"]

    def _distribution(self, observations: torch.Tensor) -> Normal:
        policy_feature, _ = self.encode(observations)
        mean = self.action_head(policy_feature)
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


class PCTAv2IndependentActors(nn.Module):
    """Parameter-independent PCTA-v2 actors for all Red aircraft."""

    def __init__(self, num_agents: int = len(RED_IDS), **actor_kwargs: Any) -> None:
        super().__init__()
        self.actors = nn.ModuleList([PCTAv2Actor(**actor_kwargs) for _ in range(num_agents)])


@dataclass
class PCTAv2TargetDiagnostics:
    valid_pairs: int
    attention_entropy_sum: float
    target_switches: int
    max_attention_weight_sum: float
    pursuit_bias_mean: float
    valid_target_states: int = 0
    multi_target_states: int = 0
    head_normalized_entropy_sum: float = 0.0
    head_normalized_entropy_count: int = 0
    head_max_attention_sum: float = 0.0
    head_max_attention_count: int = 0
    head_disagreement_sum: float = 0.0
    head_disagreement_count: int = 0


def target_behavior_diagnostics(
    actor: PCTAv2Actor,
    observations: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    active_masks: torch.Tensor,
) -> PCTAv2TargetDiagnostics:
    """Measure static per-head and temporal ensemble behavior without optimization."""
    if observations.ndim != 3 or observations.shape[-1] != OBS_DIM:
        raise ValueError("PCTA-v2 temporal observations must have shape [T,N,OBS_DIM]")
    with torch.no_grad():
        _, encoded = actor.encode(observations)
        attention = encoded["enemy_attention"]
        attention_heads = encoded["enemy_attention_heads"]
        valid_targets = enemy_valid_mask(observations)
        active = active_masks > 0.5
        valid_state = active & valid_targets.any(dim=-1)
        multi_state = active & (valid_targets.sum(dim=-1) >= 2)
        static = {
            "valid_target_states": int(valid_state.sum().item()),
            "multi_target_states": int(multi_state.sum().item()),
            "head_normalized_entropy_sum": 0.0,
            "head_normalized_entropy_count": 0,
            "head_max_attention_sum": 0.0,
            "head_max_attention_count": 0,
            "head_disagreement_sum": 0.0,
            "head_disagreement_count": 0,
        }
        if valid_state.any():
            selected_heads = attention_heads[valid_state]
            selected_valid = valid_targets[valid_state]
            static["head_max_attention_sum"] = float(selected_heads.max(dim=-1).values.sum().item())
            static["head_max_attention_count"] = int(selected_heads.shape[0] * selected_heads.shape[1])
            mean_distribution = selected_heads.mean(dim=-2)
            safe_heads = selected_heads.clamp_min(1e-12)
            kl = safe_heads * (safe_heads.log() - mean_distribution.clamp_min(1e-12).log().unsqueeze(-2))
            kl = kl.masked_fill(~selected_valid[:, None, :], 0.0)
            jsd = kl.sum(dim=-1).mean(dim=-1)
            static["head_disagreement_sum"] = float((jsd / math.log(selected_heads.shape[1])).sum().item())
            static["head_disagreement_count"] = int(selected_heads.shape[0])
        if multi_state.any():
            multi_heads = attention_heads[multi_state]
            multi_valid = valid_targets[multi_state]
            multi_heads = multi_heads.masked_fill(~multi_valid[:, None, :], 0.0)
            entropy = -(multi_heads * multi_heads.clamp_min(1e-12).log()).sum(dim=-1)
            denominator = multi_valid.sum(dim=-1).to(dtype=entropy.dtype).log()
            normalized = entropy / denominator.unsqueeze(-1)
            static["head_normalized_entropy_sum"] = float(normalized.sum().item())
            static["head_normalized_entropy_count"] = int(normalized.numel())
        pursuit_bias_mean = float(actor.pursuit_bias_gain.detach().mean().item())
        if observations.shape[0] < 2:
            return PCTAv2TargetDiagnostics(0, 0.0, 0, 0.0, pursuit_bias_mean, **static)
        previous_attention = attention[:-1]
        current_attention = attention[1:]
        previous_valid = valid_targets[:-1]
        current_valid = valid_targets[1:]
        previous_target = previous_attention.argmax(dim=-1)
        previous_has_target = previous_valid.any(dim=-1)
        previous_target_still_valid = current_valid.gather(
            -1, previous_target.unsqueeze(-1),
        ).squeeze(-1)
        transition_valid = (
            ~(terminated[:-1].bool() | truncated[:-1].bool())
            & (active_masks[:-1] > 0.5)
            & (active_masks[1:] > 0.5)
            & previous_has_target
            & previous_target_still_valid
        )
        count = int(transition_valid.sum().item())
        if count == 0:
            return PCTAv2TargetDiagnostics(0, 0.0, 0, 0.0, pursuit_bias_mean, **static)
        selected = current_attention[transition_valid]
        entropy = -(selected * selected.clamp_min(1e-12).log()).sum(dim=-1)
        switches = int((selected.argmax(dim=-1) != previous_target[transition_valid]).sum().item())
        max_weights = selected.max(dim=-1).values
        return PCTAv2TargetDiagnostics(
            valid_pairs=count,
            attention_entropy_sum=float(entropy.sum().item()),
            target_switches=switches,
            max_attention_weight_sum=float(max_weights.sum().item()),
            pursuit_bias_mean=pursuit_bias_mean,
            **static,
        )


__all__ = [
    "PCTAv2Actor", "PCTAv2IndependentActors", "PCTAv2TargetDiagnostics",
    "target_behavior_diagnostics", "OWN_ATTACK_STREAK_INDEX",
]
