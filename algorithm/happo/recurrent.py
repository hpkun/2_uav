"""Sequence-aware squashed-Gaussian actors for recurrent HAPPO."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from env.mavuav import OBS_DIM, RED_IDS


class RecurrentGaussianActor(nn.Module):
    """GRU actor whose hidden state is explicitly controlled by recurrent masks."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        hidden_dim: int = 128,
        recurrent_hidden_dim: int = 128,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.encoder = nn.Linear(self.observation_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.recurrent_hidden_dim)
        self.head = nn.Linear(self.recurrent_hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))
        self.epsilon = 1e-6

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        target = device if device is not None else self.log_std.device
        return torch.zeros(int(batch_size), self.recurrent_hidden_dim, device=target)

    def forward_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        recurrent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = recurrent_mask.reshape(observations.shape[0], 1).to(dtype=hidden.dtype)
        encoded = torch.tanh(self.encoder(observations))
        next_hidden = self.gru(encoded, hidden * mask)
        mean = self.mean(torch.tanh(self.head(next_hidden)))
        return mean, next_hidden

    def _distribution(self, mean: torch.Tensor) -> Normal:
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())

    def sample_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        recurrent_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, next_hidden = self.forward_step(observations, hidden, recurrent_mask)
        distribution = self._distribution(mean)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_probs = distribution.log_prob(raw) - torch.log(1.0 - actions.square() + self.epsilon)
        return actions, log_probs.sum(dim=-1), next_hidden

    def evaluate_actions_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        recurrent_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of ordered, equal-length sequences without padding."""
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("sequence observations/actions must have shape [batch, length, features]")
        hidden = initial_hidden
        log_prob_steps: list[torch.Tensor] = []
        entropy_steps: list[torch.Tensor] = []
        for step in range(observations.shape[1]):
            mean, hidden = self.forward_step(observations[:, step], hidden, recurrent_masks[:, step])
            clipped = actions[:, step].clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
            raw = torch.atanh(clipped)
            distribution = self._distribution(mean)
            log_probs = distribution.log_prob(raw) - torch.log(1.0 - clipped.square() + self.epsilon)
            log_prob_steps.append(log_probs.sum(dim=-1))
            entropy_steps.append(distribution.entropy().sum(dim=-1))
        return torch.stack(log_prob_steps, dim=1), torch.stack(entropy_steps, dim=1), hidden


class RecurrentIndependentActors(nn.Module):
    """One independent recurrent actor for each Red aircraft."""

    def __init__(
        self,
        observation_dim: int = OBS_DIM,
        action_dim: int = 3,
        hidden_dim: int = 128,
        recurrent_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.actors = nn.ModuleList([
            RecurrentGaussianActor(
                observation_dim=observation_dim, action_dim=action_dim,
                hidden_dim=hidden_dim, recurrent_hidden_dim=recurrent_hidden_dim,
            )
            for _ in RED_IDS
        ])
