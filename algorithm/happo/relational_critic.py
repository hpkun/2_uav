"""Relational centralized value function for canonical 4v4 HAPPO."""
from __future__ import annotations

import torch
from torch import nn

from env.mavuav import GLOBAL_STATE_DIM


class RelationalCentralizedCritic(nn.Module):
    """Encode fixed-slot aircraft relations while returning one team value."""

    ENTITY_COUNT = 8
    ENTITY_INPUT_DIM = 10
    ENTITY_EMBED_DIM = 64
    ATTENTION_HEADS = 4
    CONTEXT_INPUT_DIM = 39
    CONTEXT_EMBED_DIM = 64
    VALUE_HIDDEN_DIM = 128

    def __init__(self, state_dim: int = GLOBAL_STATE_DIM) -> None:
        super().__init__()
        if int(state_dim) != self.ENTITY_COUNT * self.ENTITY_INPUT_DIM + self.CONTEXT_INPUT_DIM:
            raise ValueError(f"relational critic requires the canonical {GLOBAL_STATE_DIM}D global state")
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.ENTITY_INPUT_DIM, self.ENTITY_EMBED_DIM), nn.Tanh(),
        )
        self.attention = nn.MultiheadAttention(
            self.ENTITY_EMBED_DIM, self.ATTENTION_HEADS, batch_first=True, dropout=0.0,
        )
        self.layer_norm = nn.LayerNorm(self.ENTITY_EMBED_DIM)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.CONTEXT_INPUT_DIM, self.CONTEXT_EMBED_DIM), nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(
                self.ENTITY_COUNT * self.ENTITY_EMBED_DIM + self.CONTEXT_EMBED_DIM,
                self.VALUE_HIDDEN_DIM,
            ),
            nn.Tanh(),
            nn.Linear(self.VALUE_HIDDEN_DIM, 1),
        )

    @classmethod
    def architecture(cls) -> dict[str, int]:
        return {
            "entity_count": cls.ENTITY_COUNT,
            "entity_input_dim": cls.ENTITY_INPUT_DIM,
            "entity_embed_dim": cls.ENTITY_EMBED_DIM,
            "attention_heads": cls.ATTENTION_HEADS,
            "context_input_dim": cls.CONTEXT_INPUT_DIM,
            "context_embed_dim": cls.CONTEXT_EMBED_DIM,
            "value_hidden_dim": cls.VALUE_HIDDEN_DIM,
        }

    def encode_entities(self, states: torch.Tensor) -> torch.Tensor:
        entities = states[..., :80].reshape(*states.shape[:-1], self.ENTITY_COUNT, self.ENTITY_INPUT_DIM)
        alive = entities[..., 6]
        embeddings = self.entity_encoder(entities)
        key_padding_mask = alive <= 0.5

        # MultiheadAttention has undefined softmax when every key is masked.
        # Temporarily expose one fixed key; the post-mask below still zeros every
        # token for an all-dead row and therefore preserves the intended result.
        safe_key_padding_mask = key_padding_mask.clone()
        all_dead = safe_key_padding_mask.all(dim=-1)
        if torch.any(all_dead):
            safe_key_padding_mask[..., 0] &= ~all_dead
        attended, _ = self.attention(
            embeddings, embeddings, embeddings, key_padding_mask=safe_key_padding_mask,
            need_weights=False,
        )
        return self.layer_norm(embeddings + attended) * alive.unsqueeze(-1)

    def encode_context(self, states: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(states[..., 80:119])

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != GLOBAL_STATE_DIM:
            raise ValueError(f"expected global state final dimension {GLOBAL_STATE_DIM}, got {states.shape[-1]}")
        entity_tokens = self.encode_entities(states)
        flattened = entity_tokens.reshape(*states.shape[:-1], self.ENTITY_COUNT * self.ENTITY_EMBED_DIM)
        fused = torch.cat((flattened, self.encode_context(states)), dim=-1)
        return self.value_head(fused).squeeze(-1)


__all__ = ["RelationalCentralizedCritic"]
