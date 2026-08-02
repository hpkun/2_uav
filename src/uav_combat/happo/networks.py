"""Networks for the project-native homogeneous 3v3 HAPPO baseline."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class HAPPOGaussianActor(nn.Module):
    """One decentralised tanh-squashed Gaussian actor for one agent slot."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.epsilon = 1e-6
        self.network = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.log_std.clamp_(self.log_std_min, self.log_std_max)

    @property
    def effective_log_std_by_dim(self) -> list[float]:
        with torch.no_grad():
            return [float(v) for v in self.log_std.clamp(self.log_std_min, self.log_std_max).detach().cpu().tolist()]

    @property
    def effective_std_by_dim(self) -> list[float]:
        with torch.no_grad():
            return [float(v) for v in self.log_std.clamp(self.log_std_min, self.log_std_max).exp().detach().cpu().tolist()]

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.network(observations)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        return Normal(mean, std)

    def sample_action(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(observations)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (
            dist.log_prob(raw)
            - torch.log(1.0 - action.square() + self.epsilon)
        ).sum(dim=-1)
        if not torch.isfinite(action).all() or not torch.isfinite(log_prob).all():
            raise FloatingPointError("non-finite HAPPO actor sample")
        return action, log_prob

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        raw = torch.atanh(bounded)
        dist = self._distribution(observations)
        log_prob = (
            dist.log_prob(raw)
            - torch.log(1.0 - bounded.square() + self.epsilon)
        ).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("non-finite HAPPO actor evaluation")
        return log_prob, entropy

    def deterministic_action(self, observations: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.network(observations))
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite HAPPO deterministic action")
        return action


class IndependentHAPPOActors(nn.Module):
    """Container of independent actors, one per red agent slot."""

    def __init__(
        self,
        observation_dims: list[int],
        action_dims: list[int],
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if len(observation_dims) != len(action_dims):
            raise ValueError("observation_dims and action_dims must have the same length")
        self.observation_dims = [int(v) for v in observation_dims]
        self.action_dims = [int(v) for v in action_dims]
        self.actors = nn.ModuleList([
            HAPPOGaussianActor(o, a, hidden_dim, log_std_init, log_std_min, log_std_max)
            for o, a in zip(self.observation_dims, self.action_dims)
        ])

    @property
    def team_size(self) -> int:
        return len(self.actors)

    @property
    def effective_log_std_by_dim(self) -> list[float]:
        vals = torch.stack([
            actor.log_std.clamp(actor.log_std_min, actor.log_std_max).detach().cpu()
            for actor in self.actors
        ], dim=0)
        return [float(v) for v in vals.mean(dim=0).tolist()]

    @property
    def effective_std_by_dim(self) -> list[float]:
        vals = torch.stack([
            actor.log_std.clamp(actor.log_std_min, actor.log_std_max).exp().detach().cpu()
            for actor in self.actors
        ], dim=0)
        return [float(v) for v in vals.mean(dim=0).tolist()]

    def sample_actions(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions, log_probs = [], []
        for i, actor in enumerate(self.actors):
            a, lp = actor.sample_action(observations[:, i, : self.observation_dims[i]])
            actions.append(a)
            log_probs.append(lp)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def deterministic_actions(self, observations: torch.Tensor) -> torch.Tensor:
        actions = [
            actor.deterministic_action(observations[:, i, : self.observation_dims[i]])
            for i, actor in enumerate(self.actors)
        ]
        return torch.stack(actions, dim=1)

    def evaluate_agent_actions(
        self, agent_id: int, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actors[agent_id].evaluate_actions(
            observations[:, : self.observation_dims[agent_id]],
            actions[:, : self.action_dims[agent_id]],
        )

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        for actor in self.actors:
            actor.clamp_log_std_()


class CentralizedValueCritic(nn.Module):
    """Centralized value critic V_phi(s), not a Q critic."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        value = self.network(global_states).squeeze(-1)
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite HAPPO critic value")
        return value


__all__ = ["CentralizedValueCritic", "HAPPOGaussianActor", "IndependentHAPPOActors"]
