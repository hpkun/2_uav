"""Entity-aware recurrent attention actors and team critic (ERAM-HAPPO)."""
from __future__ import annotations
import torch
from torch import nn
from env.mavuav import OBS_DIM, GLOBAL_STATE_DIM, RED_IDS
from .entity_layout import (
    SELF_DIM, FRIEND_DIM, ENEMY_DIM, AIRCRAFT_DIM, CONTEXT_DIM,
    parse_observation, parse_global_state,
)
from .recurrent import RecurrentGaussianActor
from .tam_buffer import TAMRolloutBuffer

ERAM_DEFAULTS = {
    "eram_entity_dim": 64, "eram_actor_attention_heads": 4,
    "eram_actor_fusion_dim": 128, "eram_actor_recurrent_hidden_dim": 128,
    "eram_critic_token_dim": 128, "eram_critic_attention_heads": 4,
    "eram_critic_recurrent_hidden_dim": 128,
}
ERAM_CONFIG_FIELDS = (*ERAM_DEFAULTS, "recurrent_sequence_length")


def _attention(module, query, tokens, valid, details=False):
    """Empty rows use a zero dummy key, then zero the entire output (bias too)."""
    nonempty = valid.any(-1)
    safe_valid = valid.clone()
    safe_valid[~nonempty, 0] = True
    tokens = tokens * valid[..., None].to(tokens.dtype)
    context, weights = module(query, tokens, tokens, key_padding_mask=~safe_valid,
                              need_weights=details)
    context = context.squeeze(1) * nonempty[:, None].to(context.dtype)
    if details:
        weights = weights.squeeze(1) * valid.to(weights.dtype)
    return context, weights


class EntityRecurrentActor(RecurrentGaussianActor):
    def __init__(self, entity_dim=64, attention_heads=4, fusion_dim=128,
                 recurrent_hidden_dim=128, log_std_init=-0.5):
        nn.Module.__init__(self)
        if entity_dim <= 0 or attention_heads <= 0 or entity_dim % attention_heads:
            raise ValueError("ERAM actor entity dimension must be divisible by attention heads")
        if min(fusion_dim, recurrent_hidden_dim) <= 0:
            raise ValueError("ERAM actor hidden dimensions must be positive")
        self.observation_dim, self.action_dim = OBS_DIM, 3
        self.entity_dim, self.attention_heads = entity_dim, attention_heads
        self.hidden_dim, self.recurrent_hidden_dim = fusion_dim, recurrent_hidden_dim
        self.self_encoder = nn.Sequential(nn.Linear(SELF_DIM, entity_dim), nn.Tanh())
        self.friend_encoder = nn.Sequential(nn.Linear(FRIEND_DIM, entity_dim), nn.Tanh())
        self.enemy_encoder = nn.Sequential(nn.Linear(ENEMY_DIM, entity_dim), nn.Tanh())
        self.ally_attention = nn.MultiheadAttention(entity_dim, attention_heads, dropout=0.0, batch_first=True)
        self.enemy_attention = nn.MultiheadAttention(entity_dim, attention_heads, dropout=0.0, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(3 * entity_dim, fusion_dim), nn.Tanh())
        self.gru = nn.GRUCell(fusion_dim, recurrent_hidden_dim)
        self.head = nn.Linear(recurrent_hidden_dim, fusion_dim)
        self.mean = nn.Linear(fusion_dim, 3)
        self.log_std = nn.Parameter(torch.full((3,), float(log_std_init)))
        self.epsilon = 1e-6

    def architecture(self):
        return {"type": "entity_recurrent", "observation_dim": OBS_DIM, "action_dim": 3,
                "self_dim": SELF_DIM, "friend_dim": FRIEND_DIM, "enemy_dim": ENEMY_DIM,
                "entity_dim": self.entity_dim, "attention_heads": self.attention_heads,
                "fusion_dim": self.hidden_dim, "recurrent_hidden_dim": self.recurrent_hidden_dim,
                "activation": "tanh", "policy_head": [self.recurrent_hidden_dim, self.hidden_dim, 3],
                "log_std": "learned_state_independent_clamp_-5_2"}

    def encode(self, observations, return_details=False):
        parsed = parse_observation(observations)
        own = self.self_encoder(parsed.self_features)
        allies, aw = _attention(self.ally_attention, own[:, None],
                               self.friend_encoder(parsed.friends), parsed.friend_valid, return_details)
        enemies, ew = _attention(self.enemy_attention, own[:, None],
                                self.enemy_encoder(parsed.enemies), parsed.enemy_valid, return_details)
        feature = self.fusion(torch.cat((own, allies, enemies), dim=-1))
        return feature, (parsed, allies, enemies, aw, ew)

    @torch.no_grad()
    def attention_diagnostics(self, observations):
        # No RNG, hidden-state mutation or extra policy sampling.
        _, (p, ally, enemy, aw, ew) = self.encode(observations, return_details=True)
        entropy = lambda w: -(w * w.clamp_min(1e-12).log()).sum(-1)
        return {
            "valid_ally_count": p.friend_valid.sum(-1), "valid_enemy_count": p.enemy_valid.sum(-1),
            "ally_attention_entropy": entropy(aw), "enemy_attention_entropy": entropy(ew),
            "max_ally_attention_weight": aw.max(-1).values,
            "max_enemy_attention_weight": ew.max(-1).values,
            "enemy_direct_mass": (ew * p.enemy_direct).sum(-1),
            "enemy_datalink_only_mass": (ew * (p.enemy_datalink & ~p.enemy_direct)).sum(-1),
        }

    def forward_step(self, observations, hidden, recurrent_mask):
        features, (parsed, *_) = self.encode(observations)
        alive = parsed.alive[:, None].to(hidden.dtype)
        mask = recurrent_mask.reshape(-1, 1).to(hidden.dtype) * alive
        next_hidden = self.gru(features, hidden * mask) * alive
        return self.mean(torch.tanh(self.head(next_hidden))) * alive, next_hidden

    def sample_step(self, observations, hidden, recurrent_mask, deterministic=False):
        actions, log_probs, next_hidden = super().sample_step(observations, hidden, recurrent_mask, deterministic)
        alive = parse_observation(observations).alive
        return actions * alive[:, None], log_probs * alive, next_hidden

    def evaluate_actions_sequence(self, observations, actions, initial_hidden, recurrent_masks):
        log_probs, entropy, hidden = super().evaluate_actions_sequence(observations, actions, initial_hidden, recurrent_masks)
        alive = parse_observation(observations).alive
        return log_probs * alive, entropy * alive, hidden


class EntityIndependentActors(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.actors = nn.ModuleList([EntityRecurrentActor(**kwargs) for _ in RED_IDS])


class EntityAttentionRecurrentCritic(nn.Module):
    def __init__(self, token_dim=128, attention_heads=4, recurrent_hidden_dim=128):
        super().__init__()
        if min(token_dim, attention_heads, recurrent_hidden_dim) <= 0 or token_dim % attention_heads:
            raise ValueError("ERAM critic token dimension must be divisible by attention heads")
        self.token_dim, self.attention_heads = token_dim, attention_heads
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.entity_encoder = nn.Sequential(nn.Linear(AIRCRAFT_DIM, token_dim), nn.Tanh())
        self.gru = nn.GRUCell(GLOBAL_STATE_DIM, recurrent_hidden_dim)
        self.context_encoder = nn.Sequential(nn.Linear(CONTEXT_DIM + recurrent_hidden_dim, token_dim), nn.Tanh())
        self.attention = nn.MultiheadAttention(token_dim, attention_heads, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(token_dim)
        self.value_head = nn.Sequential(nn.Linear(token_dim, 256), nn.LayerNorm(256), nn.Tanh(),
                                        nn.Linear(256, 128), nn.LayerNorm(128), nn.Tanh(), nn.Linear(128, 1))

    def architecture(self):
        return {"type": "entity_attention_recurrent", "state_dim": GLOBAL_STATE_DIM,
                "aircraft_count": 8, "aircraft_dim": AIRCRAFT_DIM, "context_dim": CONTEXT_DIM,
                "token_dim": self.token_dim, "attention_heads": self.attention_heads,
                "recurrent_hidden_dim": self.recurrent_hidden_dim,
                "attention": "residual_layernorm", "pooling": "alive_plus_context_masked_mean",
                "value_head": [self.token_dim, 256, 128, 1], "post_attention_layernorm": [256, 128],
                "value_loss": "mse"}

    def initial_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.recurrent_hidden_dim, device=device or next(self.parameters()).device)

    def forward_step(self, states, hidden, recurrent_mask):
        entities, context, alive = parse_global_state(states)
        next_hidden = self.gru(states, hidden * recurrent_mask.reshape(-1, 1).to(hidden.dtype))
        tokens = torch.cat((self.entity_encoder(entities),
                            self.context_encoder(torch.cat((context, next_hidden), -1))[:, None]), 1)
        valid = torch.cat((alive, torch.ones_like(alive[:, :1])), 1)
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False)
        attended = self.attention_norm(tokens + attended)
        pooled = (attended * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        return self.value_head(pooled).squeeze(-1), next_hidden

    def evaluate_values_sequence(self, states, initial_hidden, recurrent_masks):
        hidden, values = initial_hidden, []
        for step in range(states.shape[1]):
            value, hidden = self.forward_step(states[:, step], hidden, recurrent_masks[:, step])
            values.append(value)
        return torch.stack(values, 1), hidden


class ERAMRolloutBuffer(TAMRolloutBuffer):
    """Reuse generic dual-memory storage; no TAM-specific model/loss semantics."""


def actor_kwargs(c):
    return {"entity_dim": int(c["eram_entity_dim"]), "attention_heads": int(c["eram_actor_attention_heads"]),
            "fusion_dim": int(c["eram_actor_fusion_dim"]),
            "recurrent_hidden_dim": int(c["eram_actor_recurrent_hidden_dim"]),
            "log_std_init": float(c["actor_log_std_init"])}


def critic_kwargs(c):
    return {"token_dim": int(c["eram_critic_token_dim"]), "attention_heads": int(c["eram_critic_attention_heads"]),
            "recurrent_hidden_dim": int(c["eram_critic_recurrent_hidden_dim"])}


def eram_metadata(c):
    return {"algorithm": "eram_happo", "base_algorithm": "happo", "action_dim": 3,
            "entity_dimensions": {"self": SELF_DIM, "friend": FRIEND_DIM, "enemy": ENEMY_DIM,
                                  "aircraft": AIRCRAFT_DIM, "global_context": CONTEXT_DIM},
            "attention_heads": {"actor": int(c["eram_actor_attention_heads"]),
                                "critic": int(c["eram_critic_attention_heads"])},
            "actor_count": 4, "independent_actor_count": 4, "sequential_happo_update": True,
            "training_seed": int(c["seed"]), "entity_layout_version": "v311_named_100_117_v1",
            "actor_visibility_mask": "alive_and_direct_or_datalink",
            "actor_active_mask": "zero_action_loss_entropy_hidden_inactive_ratio_one",
            "critic_entity_mask": "alive_aircraft_context_always_valid_no_visibility_mask",
            "actor_recurrent_configuration": {"hidden_dim": int(c["eram_actor_recurrent_hidden_dim"]), "reset": "episode_or_individual_death"},
            "critic_recurrent_configuration": {"hidden_dim": int(c["eram_critic_recurrent_hidden_dim"]), "reset": "episode_only"},
            "recurrent_sequence_length": int(c["recurrent_sequence_length"])}


def validate_eram_metadata(payload, config):
    for key, expected in eram_metadata(config).items():
        if payload.get(key) != expected:
            raise RuntimeError(f"incompatible ERAM checkpoint contract: {key}")
