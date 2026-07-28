"""Neural networks for the homogeneous 3v3 MADSAC baseline."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class SharedSquashedGaussianActor(nn.Module):
    """Shared red-team tanh-squashed Gaussian actor.

    The same MLP is applied to each homogeneous red aircraft. ``log_std`` is
    state-dependent via a linear head, not a global parameter.
    """

    def __init__(
        self,
        observation_dim: int = 68,
        action_dim: int = 3,
        team_size: int = 3,
        hidden_dim: int = 256,
        log_std_bias_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.team_size = team_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = 1e-6
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.log_std_head.bias, float(log_std_bias_init))

    @property
    def effective_log_std_mean(self) -> float:
        with torch.no_grad():
            zeros = torch.zeros(1, self.observation_dim, device=next(self.parameters()).device)
            return float(self._mean_log_std(zeros)[1].mean().item())

    @property
    def effective_std_mean(self) -> float:
        with torch.no_grad():
            zeros = torch.zeros(1, self.observation_dim, device=next(self.parameters()).device)
            return float(self._mean_log_std(zeros)[1].exp().mean().item())

    def _mean_log_std(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(f"last observation dimension must be {self.observation_dim}")
        hidden = self.trunk(observations)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampled actions and corrected log probabilities.

        Accepts ``[B, 3, 68]`` or ``[B, 68]``. For the 3-agent input, log prob
        shape is ``[B, 3]`` after summing each agent's three action dimensions.
        """
        mean, log_std = self._mean_log_std(observations)
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (
            dist.log_prob(raw)
            - torch.log(1.0 - action.square() + self.epsilon)
        ).sum(dim=-1)
        if not torch.isfinite(action).all() or not torch.isfinite(log_prob).all():
            raise FloatingPointError("non-finite actor output")
        return action, log_prob

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self._mean_log_std(observations)
        action = torch.tanh(mean)
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite deterministic actor output")
        return action


class AttentionCritic(nn.Module):
    """Per-agent centralized attention critic with other-agent attention.

    For each agent i, the query is produced from its own encoded
    observation-action token. Keys and values come only from other alive agents
    j != i. The agent's own embedding is preserved as an explicit self path and
    concatenated with the attention context before the Q head.
    """

    def __init__(
        self,
        observation_dim: int = 68,
        action_dim: int = 3,
        team_size: int = 3,
        hidden_dim: int = 256,
        attention_heads: int = 2,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, alive_masks: torch.Tensor) -> torch.Tensor:
        if observations.shape[-2:] != (self.team_size, self.observation_dim):
            raise ValueError(f"observations must end with ({self.team_size}, {self.observation_dim})")
        if actions.shape[-2:] != (self.team_size, self.action_dim):
            raise ValueError(f"actions must end with ({self.team_size}, {self.action_dim})")
        alive = alive_masks.to(dtype=observations.dtype)
        joint = torch.cat([observations, actions], dim=-1)
        encoded = self.encoder(joint) * alive.unsqueeze(-1)

        batch = encoded.shape[0]
        q = self.query(encoded).view(batch, self.team_size, self.attention_heads, self.head_dim).transpose(1, 2)
        k = self.key(encoded).view(batch, self.team_size, self.attention_heads, self.head_dim).transpose(1, 2)
        v = self.value(encoded).view(batch, self.team_size, self.attention_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        other_mask = torch.ones(self.team_size, self.team_size, dtype=torch.bool, device=observations.device)
        other_mask.fill_diagonal_(False)
        key_alive = alive[:, None, None, :] > 0.5
        valid_keys = key_alive & other_mask[None, None, :, :]
        scores = scores.masked_fill(~valid_keys, -torch.finfo(scores.dtype).max)
        has_other = valid_keys.any(dim=-1, keepdim=True)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(has_other, weights, torch.zeros_like(weights))
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, self.team_size, self.hidden_dim)
        q = self.q_head(torch.cat([encoded, context], dim=-1)).squeeze(-1)
        q = q * alive
        if not torch.isfinite(q).all():
            raise FloatingPointError("non-finite attention critic output")
        return q


class TwinAttentionCritic(nn.Module):
    """Two fully independent attention critics."""

    def __init__(
        self,
        observation_dim: int = 68,
        action_dim: int = 3,
        team_size: int = 3,
        hidden_dim: int = 256,
        attention_heads: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = AttentionCritic(observation_dim, action_dim, team_size, hidden_dim, attention_heads)
        self.q2 = AttentionCritic(observation_dim, action_dim, team_size, hidden_dim, attention_heads)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, alive_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observations, actions, alive_masks), self.q2(observations, actions, alive_masks)


# Backward-compatible name for earlier local imports/tests.
TwinCentralizedQCritic = TwinAttentionCritic
