"""Temporal Attention Masked networks for TAM-HAPPO transfer reproduction."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS


class TAMGaussianActor(nn.Module):
    """State-memory-first continuous actor adapted to the current 100D contract."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        recurrent_hidden_dim: int = 128,
        hidden_layers: tuple[int, int] = (256, 128),
        log_std_init: float = -0.25,
        state_memory: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_layers) != 2:
            raise ValueError("TAM actor requires exactly two policy hidden layers")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.state_memory = bool(state_memory)
        self.gru = nn.GRUCell(self.observation_dim, self.recurrent_hidden_dim)
        self.temporal_projection = nn.Linear(self.recurrent_hidden_dim, self.observation_dim)
        self.policy_fc1 = nn.Linear(2 * self.observation_dim, self.hidden_layers[0])
        self.policy_fc2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        self.mean = nn.Linear(self.hidden_layers[1], self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))
        self.epsilon = 1e-6

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        target = device if device is not None else self.log_std.device
        return torch.zeros(int(batch_size), self.recurrent_hidden_dim, device=target)

    def forward_step(
        self, observations: torch.Tensor, hidden: torch.Tensor, recurrent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = recurrent_mask.reshape(observations.shape[0], 1).to(dtype=hidden.dtype)
        next_hidden = self.gru(observations, hidden * mask)
        temporal_hidden = next_hidden if self.state_memory else torch.zeros_like(next_hidden)
        temporal = torch.tanh(self.temporal_projection(temporal_hidden))
        fused = torch.cat((observations, temporal), dim=-1)
        policy = torch.tanh(self.policy_fc1(fused))
        policy = torch.tanh(self.policy_fc2(policy))
        return self.mean(policy), next_hidden

    def _distribution(self, mean: torch.Tensor) -> Normal:
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())

    def sample_step(
        self, observations: torch.Tensor, hidden: torch.Tensor, recurrent_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, next_hidden = self.forward_step(observations, hidden, recurrent_mask)
        distribution = self._distribution(mean)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - actions.square() + self.epsilon)
        return actions, log_probs.sum(dim=-1), next_hidden

    def evaluate_actions_sequence(
        self, observations: torch.Tensor, actions: torch.Tensor,
        initial_hidden: torch.Tensor, recurrent_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = initial_hidden
        log_prob_steps: list[torch.Tensor] = []
        entropy_steps: list[torch.Tensor] = []
        for step in range(observations.shape[1]):
            mean, hidden = self.forward_step(observations[:, step], hidden, recurrent_masks[:, step])
            clipped = actions[:, step].clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
            raw = torch.atanh(clipped)
            distribution = self._distribution(mean)
            log_prob = distribution.log_prob(raw) - torch.log(1.0 - clipped.square() + self.epsilon)
            log_prob_steps.append(log_prob.sum(dim=-1))
            entropy_steps.append(distribution.entropy().sum(dim=-1))
        return torch.stack(log_prob_steps, dim=1), torch.stack(entropy_steps, dim=1), hidden


class TAMIndependentActors(nn.Module):
    """Four fully independent TAM actors (no shared parameters)."""

    def __init__(self, **actor_kwargs) -> None:
        super().__init__()
        self.actors = nn.ModuleList([TAMGaussianActor(**actor_kwargs) for _ in RED_IDS])


class TAMAttentionCritic(nn.Module):
    """Recurrent, entity-masked, single-layer attention state-value function."""

    ENTITY_COUNT = 8
    ENTITY_DIM = 10
    CONTEXT_DIM = 37

    def __init__(
        self,
        state_dim: int = GLOBAL_STATE_DIM,
        recurrent_hidden_dim: int = 128,
        token_dim: int = 128,
        attention_heads: int = 4,
        hidden_layers: tuple[int, int] = (256, 128),
        state_memory: bool = True,
        attention: bool = True,
        inactive_mask: bool = True,
    ) -> None:
        super().__init__()
        if int(state_dim) != self.ENTITY_COUNT * self.ENTITY_DIM + self.CONTEXT_DIM:
            raise ValueError(f"TAM critic requires canonical {GLOBAL_STATE_DIM}D global state")
        if token_dim % attention_heads:
            raise ValueError("TAM critic token dimension must be divisible by attention heads")
        self.state_dim = int(state_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.state_memory = bool(state_memory)
        self.attention_enabled = bool(attention)
        self.inactive_mask = bool(inactive_mask)
        self.gru = nn.GRUCell(self.state_dim, self.recurrent_hidden_dim)
        self.entity_encoder = nn.Sequential(nn.Linear(self.ENTITY_DIM, self.token_dim), nn.Tanh())
        self.context_encoder = nn.Sequential(
            nn.Linear(self.CONTEXT_DIM + self.recurrent_hidden_dim, self.token_dim), nn.Tanh(),
        )
        self.attention = nn.MultiheadAttention(
            self.token_dim, self.attention_heads, batch_first=True, dropout=0.0,
        )
        self.attention_norm = nn.LayerNorm(self.token_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.token_dim, self.hidden_layers[0]), nn.Tanh(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]), nn.Tanh(),
            nn.LayerNorm(self.hidden_layers[1]), nn.Linear(self.hidden_layers[1], 1),
        )
        self.last_attention_key_padding_mask: torch.Tensor | None = None
        self.last_token_count = self.ENTITY_COUNT + 1

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        target = device if device is not None else next(self.parameters()).device
        return torch.zeros(int(batch_size), self.recurrent_hidden_dim, device=target)

    def forward_step(
        self, states: torch.Tensor, hidden: torch.Tensor, recurrent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"expected global state final dimension {self.state_dim}, got {states.shape[-1]}")
        mask = recurrent_mask.reshape(states.shape[0], 1).to(dtype=hidden.dtype)
        next_hidden = self.gru(states, hidden * mask)
        temporal_hidden = next_hidden if self.state_memory else torch.zeros_like(next_hidden)
        entities = states[..., :80].reshape(states.shape[0], self.ENTITY_COUNT, self.ENTITY_DIM)
        alive = entities[..., 6]
        entity_tokens = self.entity_encoder(entities)
        context = states[..., 80:117]
        context_token = self.context_encoder(torch.cat((context, temporal_hidden), dim=-1)).unsqueeze(1)
        tokens = torch.cat((entity_tokens, context_token), dim=1)
        key_padding_mask = torch.cat(
            ((alive <= 0.5) if self.inactive_mask else torch.zeros_like(alive, dtype=torch.bool),
             torch.zeros((states.shape[0], 1), dtype=torch.bool, device=states.device)), dim=1,
        )
        self.last_attention_key_padding_mask = key_padding_mask.detach()
        if self.attention_enabled:
            attended, _ = self.attention(
                tokens, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False,
            )
            tokens = self.attention_norm(tokens + attended)
        valid = (~key_padding_mask).to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.value_head(pooled).squeeze(-1), next_hidden

    def evaluate_values_sequence(
        self, states: torch.Tensor, initial_hidden: torch.Tensor, recurrent_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = initial_hidden
        values: list[torch.Tensor] = []
        for step in range(states.shape[1]):
            value, hidden = self.forward_step(states[:, step], hidden, recurrent_masks[:, step])
            values.append(value)
        return torch.stack(values, dim=1), hidden

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(states.shape[0], states.device)
        values, _ = self.forward_step(states, hidden, torch.zeros(states.shape[0], device=states.device))
        return values

    def architecture(self) -> dict[str, object]:
        return {
            "state_dim": self.state_dim, "entity_count": self.ENTITY_COUNT,
            "entity_dim": self.ENTITY_DIM, "context_dim": self.CONTEXT_DIM,
            "recurrent_hidden_dim": self.recurrent_hidden_dim, "token_dim": self.token_dim,
            "attention_heads": self.attention_heads, "attention_layers": 1,
            "token_count": self.ENTITY_COUNT + 1, "hidden_layers": list(self.hidden_layers),
            "state_only_value": True, "state_memory": self.state_memory,
            "attention": self.attention_enabled, "inactive_mask": self.inactive_mask,
        }


__all__ = ["TAMGaussianActor", "TAMIndependentActors", "TAMAttentionCritic"]
