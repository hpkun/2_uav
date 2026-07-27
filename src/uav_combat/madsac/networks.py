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
    """Per-agent centralized attention critic for three red aircraft."""

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
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
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

        key_padding_mask = alive <= 0.5
        # PyTorch attention returns NaN if every key is masked. Keep one dummy
        # zero token available in all-dead rows; final Q is explicitly zeroed.
        all_dead = key_padding_mask.all(dim=1)
        if bool(all_dead.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_dead, 0] = False

        context, _ = self.attention(
            query=encoded,
            key=encoded,
            value=encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
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
