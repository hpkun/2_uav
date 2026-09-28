"""Dynamic behavioral-mode UAV actors for DBM-RGAA-v1.

The module changes only the three UAV policy mean networks.  It deliberately
inherits the squashed-Gaussian probability implementation from GaussianActor.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from torch import nn

from algorithm.common.networks import GaussianActor
from env.mavuav import OBS_DIM, RED_IDS


DBM_RGAA_METHOD = "dbm_rgaa"
RGAA_WIDE_METHOD = "rgaa_wide"
DBM_RGAA_VERSION = 1
DBM_VERSION = DBM_RGAA_VERSION
DBM_ROLE_COUNT = 2
DBM_RESIDUAL_SCALE = 0.25
DBM_EXPERT_INIT_SCALE = 0.01
DBM_INIT_SCALE = DBM_EXPERT_INIT_SCALE
DBM_INITIALIZATION_SEMANTICS = "zero_router_antisymmetric_experts_v1"
DBM_ACTOR_SEMANTICS = "vanilla_mean_plus_private_two_mode_bounded_residual_v1"
DBM_DIAGNOSTICS_VERSION = 1
DBM_DIAGNOSTICS_SOURCE = "post_update_policy_on_collected_rollout_v1"
RGAA_WIDE_UAV_HIDDEN_DIM = 131


def dbm_auxiliary_seed(training_seed: int, agent_index: int) -> int:
    """Stable seed for one isolated UAV DBM module initialization."""
    return (int(training_seed) + 0x44424D00 + int(agent_index)) % (2**63 - 1)


def wide_auxiliary_seed(training_seed: int, agent_index: int) -> int:
    """Stable seed for one isolated wide-UAV actor initialization."""
    return (int(training_seed) + 0x57494400 + int(agent_index)) % (2**63 - 1)


class DBMMeanNetwork(nn.Module):
    """Vanilla mean plus a two-mode, bounded residual for one UAV."""

    def __init__(
        self,
        base_network: nn.Sequential,
        *,
        hidden_dim: int = 128,
        action_dim: int = 3,
        role_count: int = DBM_ROLE_COUNT,
        residual_scale: float = DBM_RESIDUAL_SCALE,
        expert_init_scale: float = DBM_EXPERT_INIT_SCALE,
    ) -> None:
        super().__init__()
        if int(role_count) != DBM_ROLE_COUNT:
            raise ValueError(f"DBM-RGAA-v1 requires exactly {DBM_ROLE_COUNT} modes")
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.role_count = int(role_count)
        self.residual_scale = float(residual_scale)
        self.expert_init_scale = float(expert_init_scale)
        self.encoder = nn.Sequential(*[deepcopy(module) for module in base_network[:4]])
        self.base_head = deepcopy(base_network[4])
        self.router = nn.Linear(self.hidden_dim, self.role_count)
        self.experts = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.action_dim) for _ in range(self.role_count)
        ])
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.zero_()
            nn.init.normal_(self.experts[0].weight, mean=0.0, std=self.expert_init_scale)
            self.experts[0].bias.zero_()
            self.experts[1].weight.copy_(-self.experts[0].weight)
            self.experts[1].bias.copy_(-self.experts[0].bias)

    def details(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(observations)
        base_mean = self.base_head(hidden)
        router_probabilities = torch.softmax(self.router(hidden), dim=-1)
        expert_outputs = torch.stack(
            [torch.tanh(expert(hidden)) for expert in self.experts], dim=-2,
        )
        weighted_residual = (
            router_probabilities.unsqueeze(-1) * expert_outputs
        ).sum(dim=-2)
        scaled_residual = self.residual_scale * weighted_residual
        return {
            "base_mean": base_mean,
            "router_probabilities": router_probabilities,
            "expert_outputs": expert_outputs,
            "weighted_residual": weighted_residual,
            "scaled_residual": scaled_residual,
            "final_mean": base_mean + scaled_residual,
        }

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.details(observations)["final_mean"]


class DBMGaussianActor(GaussianActor):
    """Gaussian actor whose mean is augmented by the DBM residual module."""

    def __init__(
        self,
        base_actor: GaussianActor,
        *,
        hidden_dim: int = 128,
        action_dim: int = 3,
        role_count: int = DBM_ROLE_COUNT,
        residual_scale: float = DBM_RESIDUAL_SCALE,
        expert_init_scale: float = DBM_EXPERT_INIT_SCALE,
    ) -> None:
        nn.Module.__init__(self)
        self.network = DBMMeanNetwork(
            base_actor.network,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            role_count=role_count,
            residual_scale=residual_scale,
            expert_init_scale=expert_init_scale,
        )
        self.log_std = nn.Parameter(base_actor.log_std.detach().clone())
        self.epsilon = float(base_actor.epsilon)

    def mode_diagnostics(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        details = self.network.details(observations)
        details["log_std"] = self.log_std.expand_as(details["final_mean"])
        return details


class MixedIndependentActors(nn.Module):
    """Four independent actors with method-specific UAV implementations."""

    def __init__(self, actors: list[nn.Module]) -> None:
        super().__init__()
        if len(actors) != len(RED_IDS):
            raise ValueError(f"expected {len(RED_IDS)} actors")
        self.actors = nn.ModuleList(actors)


def build_method_actors(
    *,
    method_variant: str,
    training_seed: int,
    observation_dim: int = OBS_DIM,
    action_dim: int = 3,
    hidden_dim: int = 128,
    log_std_init: float = -0.5,
    role_module_enabled: bool = True,
    dbm_role_count: int = DBM_ROLE_COUNT,
    dbm_residual_scale: float = DBM_RESIDUAL_SCALE,
    dbm_expert_init_scale: float = DBM_EXPERT_INIT_SCALE,
    uav_actor_hidden_dim: int = RGAA_WIDE_UAV_HIDDEN_DIM,
) -> MixedIndependentActors:
    """Build actors while preserving the vanilla/RGAA global RNG sequence."""
    # Constructing this exact baseline first is the RNG contract: MAV, all UAV
    # base policies, and every module constructed after actors see the same main
    # RNG state as ordinary RGAA.
    baseline = [
        GaussianActor(observation_dim, action_dim, hidden_dim, log_std_init)
        for _ in RED_IDS
    ]
    if method_variant == DBM_RGAA_METHOD and role_module_enabled:
        for agent_index in range(1, len(RED_IDS)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    dbm_auxiliary_seed(training_seed, agent_index),
                )
                baseline[agent_index] = DBMGaussianActor(
                    baseline[agent_index],
                    hidden_dim=hidden_dim,
                    action_dim=action_dim,
                    role_count=dbm_role_count,
                    residual_scale=dbm_residual_scale,
                    expert_init_scale=dbm_expert_init_scale,
                )
    elif method_variant == RGAA_WIDE_METHOD:
        for agent_index in range(1, len(RED_IDS)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    wide_auxiliary_seed(training_seed, agent_index),
                )
                baseline[agent_index] = GaussianActor(
                    observation_dim, action_dim, uav_actor_hidden_dim, log_std_init,
                )
    return MixedIndependentActors(baseline)


def dbm_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("method_variant") != DBM_RGAA_METHOD:
        return {}
    enabled = bool(config["role_module_enabled"])
    return {
        "dbm_rgaa_version": DBM_RGAA_VERSION,
        "role_module_enabled": enabled,
        "dbm_role_count": int(config["dbm_role_count"]),
        "dbm_residual_scale": float(config["dbm_residual_scale"]),
        "dbm_init_scale": float(config["dbm_init_scale"]),
        "dbm_initialization_semantics": DBM_INITIALIZATION_SEMANTICS,
        "dbm_actor_semantics": DBM_ACTOR_SEMANTICS,
        "dbm_module_seeds": (
            {RED_IDS[i]: dbm_auxiliary_seed(int(config["seed"]), i) for i in range(1, len(RED_IDS))}
            if enabled else {}
        ),
        "dbm_diagnostics_version": DBM_DIAGNOSTICS_VERSION,
        "dbm_diagnostics_source": DBM_DIAGNOSTICS_SOURCE,
    }


def wide_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("method_variant") != RGAA_WIDE_METHOD:
        return {}
    return {
        "rgaa_wide_version": 1,
        "mav_actor_hidden_dim": int(config["hidden_dim"]),
        "uav_actor_hidden_dim": int(config["uav_actor_hidden_dim"]),
        "uav_actor_initialization_seeds": {
            RED_IDS[i]: wide_auxiliary_seed(int(config["seed"]), i)
            for i in range(1, len(RED_IDS))
        },
    }


def dbm_rollout_diagnostics(
    actors: MixedIndependentActors,
    observations: torch.Tensor,
    active_masks: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> dict[str, float]:
    """Compute active-mask-aware, RNG-free DBM policy diagnostics."""
    if observations.ndim != 4:
        raise ValueError("observations must have [time, env, agent, feature] shape")
    metrics: dict[str, float] = {}
    eps = 1e-8
    boundary = (terminated.bool() | truncated.bool())
    for agent_index in range(1, len(RED_IDS)):
        label = RED_IDS[agent_index]
        actor = actors.actors[agent_index]
        if not isinstance(actor, DBMGaussianActor):
            continue
        details = actor.mode_diagnostics(observations[:, :, agent_index])
        probs = details["router_probabilities"]
        expert_outputs = details["expert_outputs"]
        residual = details["scaled_residual"]
        base_mean = details["base_mean"]
        final_mean = details["final_mean"]
        active = active_masks[:, :, agent_index] > 0.5
        count = int(active.sum().item())
        if count:
            active_probs = probs[active]
            hard = active_probs.argmax(dim=-1)
            entropy = -(active_probs * active_probs.clamp_min(eps).log()).sum(dim=-1)
            active_experts = expert_outputs[active]
            divergence = (active_experts[:, 0] - active_experts[:, 1]).norm(dim=-1)
            residual_mag = residual[active].norm(dim=-1)
            base_mag = base_mean[active].norm(dim=-1)
            final_actions = torch.tanh(final_mean[active])
            for mode in range(DBM_ROLE_COUNT):
                metrics[f"dbm_soft_occupancy_mode{mode + 1}_{label}"] = float(active_probs[:, mode].mean().item())
                metrics[f"dbm_hard_occupancy_mode{mode + 1}_{label}"] = float((hard == mode).float().mean().item())
                metrics[f"dbm_router_variance_mode{mode + 1}_{label}"] = float(
                    active_probs[:, mode].var(unbiased=False).item()
                )
            metrics[f"dbm_router_entropy_{label}"] = float(entropy.mean().item())
            metrics[f"dbm_router_probability_variance_{label}"] = float(active_probs.var(dim=0, unbiased=False).mean().item())
            metrics[f"dbm_expert_divergence_{label}"] = float(divergence.mean().item())
            metrics[f"dbm_scaled_residual_magnitude_{label}"] = float(residual_mag.mean().item())
            metrics[f"dbm_residual_base_ratio_{label}"] = float(
                (residual_mag.mean() / (base_mag.mean() + eps)).item()
            )
            metrics[f"dbm_deterministic_action_saturation_{label}"] = float(
                (final_actions.abs().amax(dim=-1) > 0.95).float().mean().item()
            )
            metrics[f"dbm_max_soft_occupancy_{label}"] = max(
                metrics[f"dbm_soft_occupancy_mode1_{label}"],
                metrics[f"dbm_soft_occupancy_mode2_{label}"],
            )
            metrics[f"dbm_max_hard_occupancy_{label}"] = max(
                metrics[f"dbm_hard_occupancy_mode1_{label}"],
                metrics[f"dbm_hard_occupancy_mode2_{label}"],
            )
        else:
            for mode in range(DBM_ROLE_COUNT):
                metrics[f"dbm_soft_occupancy_mode{mode + 1}_{label}"] = 0.0
                metrics[f"dbm_hard_occupancy_mode{mode + 1}_{label}"] = 0.0
                metrics[f"dbm_router_variance_mode{mode + 1}_{label}"] = 0.0
            for name in (
                "router_entropy", "router_probability_variance", "expert_divergence",
                "scaled_residual_magnitude", "residual_base_ratio",
                "deterministic_action_saturation", "max_soft_occupancy",
                "max_hard_occupancy",
            ):
                metrics[f"dbm_{name}_{label}"] = 0.0
        valid_pairs = (
            active[:-1] & active[1:] & ~boundary[:-1]
        ) if observations.shape[0] > 1 else active[:0]
        pair_count = int(valid_pairs.sum().item())
        metrics[f"dbm_active_sample_count_{label}"] = float(count)
        metrics[f"dbm_valid_switch_pairs_{label}"] = float(pair_count)
        if pair_count:
            previous = probs[:-1][valid_pairs]
            following = probs[1:][valid_pairs]
            metrics[f"dbm_mode_switch_rate_{label}"] = float(
                (previous.argmax(-1) != following.argmax(-1)).float().mean().item()
            )
            metrics[f"dbm_router_l1_movement_{label}"] = float(
                (following - previous).abs().sum(dim=-1).mean().item()
            )
        else:
            metrics[f"dbm_mode_switch_rate_{label}"] = 0.0
            metrics[f"dbm_router_l1_movement_{label}"] = 0.0
    return metrics
