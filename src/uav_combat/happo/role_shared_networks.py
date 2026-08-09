"""Role-shared actors for the v13 4v3 HAPPO experiments.

This is a lightweight project-native extension: one Support policy and one
parameter-shared Combat policy.  The optional recurrent path uses a single
GRUCell per policy group and intentionally keeps the centralized critic MLP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .networks import HAPPOGaussianActor


@dataclass
class RoleHiddenState:
    support: torch.Tensor
    combat: torch.Tensor

    def clone(self) -> "RoleHiddenState":
        return RoleHiddenState(self.support.clone(), self.combat.clone())


class RecurrentHAPPOGaussianActor(nn.Module):
    """Small observation encoder + GRUCell + squashed Gaussian head."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_init: float = -1.0,
        log_std_min: float = -3.0,
        log_std_max: float = -0.3,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if int(num_layers) != 1:
            raise ValueError("the lightweight v13 recurrent actor supports recurrent_num_layers=1")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.epsilon = 1e-6
        self.encoder = nn.Sequential(nn.Linear(self.observation_dim, self.hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    @property
    def effective_log_std_by_dim(self) -> list[float]:
        with torch.no_grad():
            return [float(v) for v in self.log_std.clamp(self.log_std_min, self.log_std_max).cpu().tolist()]

    @property
    def effective_std_by_dim(self) -> list[float]:
        with torch.no_grad():
            return [float(v) for v in self.log_std.clamp(self.log_std_min, self.log_std_max).exp().cpu().tolist()]

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.log_std.clamp_(self.log_std_min, self.log_std_max)

    def _advance(self, observations: torch.Tensor, hidden: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
        masked_hidden = hidden * reset_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return self.gru(self.encoder(observations), masked_hidden)

    def _distribution(self, hidden: torch.Tensor) -> Normal:
        mean = self.mean_head(hidden)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        return Normal(mean, std)

    def sample_step(
        self, observations: torch.Tensor, hidden: torch.Tensor, reset_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_hidden = self._advance(observations, hidden, reset_mask)
        dist = self._distribution(new_hidden)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1.0 - action.square() + self.epsilon)).sum(-1)
        if not torch.isfinite(action).all() or not torch.isfinite(log_prob).all():
            raise FloatingPointError("non-finite recurrent HAPPO actor sample")
        return action, log_prob, new_hidden

    def deterministic_step(
        self, observations: torch.Tensor, hidden: torch.Tensor, reset_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_hidden = self._advance(observations, hidden, reset_mask)
        action = torch.tanh(self.mean_head(new_hidden))
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite recurrent HAPPO deterministic action")
        return action, new_hidden

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        reset_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("recurrent observations/actions must have shape [batch,time,dim]")
        hidden = initial_hidden
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for step in range(observations.shape[1]):
            hidden = self._advance(observations[:, step], hidden, reset_masks[:, step])
            dist = self._distribution(hidden)
            bounded = actions[:, step].clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
            raw = torch.atanh(bounded)
            log_probs.append((dist.log_prob(raw) - torch.log(1.0 - bounded.square() + self.epsilon)).sum(-1))
            entropies.append(dist.entropy().sum(-1))
        return torch.stack(log_probs, 1), torch.stack(entropies, 1), hidden


class RoleSharedHAPPOActors(nn.Module):
    """Two policy objects deployed to four slots: Support + shared Combat."""

    policy_mapping = ("support", "combat", "combat", "combat")

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        log_std_init: float = -1.0,
        log_std_min: float = -3.0,
        log_std_max: float = -0.3,
        *,
        recurrent: bool = False,
        recurrent_hidden_dim: int = 128,
        recurrent_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.recurrent = bool(recurrent)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.recurrent_num_layers = int(recurrent_num_layers)
        if self.recurrent:
            actor_args = dict(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_dim=self.recurrent_hidden_dim,
                log_std_init=log_std_init,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                num_layers=self.recurrent_num_layers,
            )
            self.support_actor = RecurrentHAPPOGaussianActor(**actor_args)
            self.combat_actor = RecurrentHAPPOGaussianActor(**actor_args)
        else:
            actor_args = dict(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_dim=int(hidden_dim),
                log_std_init=log_std_init,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            )
            self.support_actor = HAPPOGaussianActor(**actor_args)
            self.combat_actor = HAPPOGaussianActor(**actor_args)

    def actor_for_slot(self, slot: int) -> nn.Module:
        if int(slot) == 0:
            return self.support_actor
        if int(slot) in (1, 2, 3):
            return self.combat_actor
        raise IndexError(slot)

    def initial_hidden(self, num_envs: int, device: torch.device | str) -> RoleHiddenState | None:
        if not self.recurrent:
            return None
        return RoleHiddenState(
            support=torch.zeros(int(num_envs), self.recurrent_hidden_dim, device=device),
            combat=torch.zeros(int(num_envs), 3, self.recurrent_hidden_dim, device=device),
        )

    def sample_actions(
        self,
        observations: torch.Tensor,
        alive_masks: torch.Tensor,
        hidden: RoleHiddenState | None = None,
        reset_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoleHiddenState | None]:
        alive = alive_masks.to(dtype=observations.dtype)
        if self.recurrent:
            if hidden is None:
                raise ValueError("recurrent actor requires hidden state")
            reset = alive if reset_masks is None else reset_masks.to(dtype=observations.dtype)
            support_action, support_lp, support_hidden = self.support_actor.sample_step(
                observations[:, 0], hidden.support, reset[:, 0]
            )
            combat_action, combat_lp, combat_hidden = self.combat_actor.sample_step(
                observations[:, 1:4].reshape(-1, self.observation_dim),
                hidden.combat.reshape(-1, self.recurrent_hidden_dim),
                reset[:, 1:4].reshape(-1),
            )
            combat_action = combat_action.reshape(-1, 3, self.action_dim)
            combat_lp = combat_lp.reshape(-1, 3)
            combat_hidden = combat_hidden.reshape(-1, 3, self.recurrent_hidden_dim)
            next_hidden = RoleHiddenState(
                support=support_hidden * alive[:, 0:1],
                combat=combat_hidden * alive[:, 1:4].unsqueeze(-1),
            )
        else:
            support_action, support_lp = self.support_actor.sample_action(observations[:, 0])
            combat_action, combat_lp = self.combat_actor.sample_action(
                observations[:, 1:4].reshape(-1, self.observation_dim)
            )
            combat_action = combat_action.reshape(-1, 3, self.action_dim)
            combat_lp = combat_lp.reshape(-1, 3)
            next_hidden = None
        actions = torch.cat((support_action.unsqueeze(1), combat_action), 1) * alive.unsqueeze(-1)
        log_probs = torch.cat((support_lp.unsqueeze(1), combat_lp), 1) * alive
        return actions, log_probs, next_hidden

    def deterministic_actions(
        self,
        observations: torch.Tensor,
        alive_masks: torch.Tensor,
        hidden: RoleHiddenState | None = None,
        reset_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RoleHiddenState | None]:
        alive = alive_masks.to(dtype=observations.dtype)
        if self.recurrent:
            if hidden is None:
                raise ValueError("recurrent actor requires hidden state")
            reset = alive if reset_masks is None else reset_masks.to(dtype=observations.dtype)
            support_action, support_hidden = self.support_actor.deterministic_step(
                observations[:, 0], hidden.support, reset[:, 0]
            )
            combat_action, combat_hidden = self.combat_actor.deterministic_step(
                observations[:, 1:4].reshape(-1, self.observation_dim),
                hidden.combat.reshape(-1, self.recurrent_hidden_dim),
                reset[:, 1:4].reshape(-1),
            )
            combat_action = combat_action.reshape(-1, 3, self.action_dim)
            next_hidden = RoleHiddenState(
                support=support_hidden * alive[:, 0:1],
                combat=combat_hidden.reshape(-1, 3, self.recurrent_hidden_dim) * alive[:, 1:4].unsqueeze(-1),
            )
        else:
            support_action = self.support_actor.deterministic_action(observations[:, 0])
            combat_action = self.combat_actor.deterministic_action(
                observations[:, 1:4].reshape(-1, self.observation_dim)
            ).reshape(-1, 3, self.action_dim)
            next_hidden = None
        return torch.cat((support_action.unsqueeze(1), combat_action), 1) * alive.unsqueeze(-1), next_hidden

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.support_actor.clamp_log_std_()
        self.combat_actor.clamp_log_std_()


__all__ = ["RecurrentHAPPOGaussianActor", "RoleHiddenState", "RoleSharedHAPPOActors"]
