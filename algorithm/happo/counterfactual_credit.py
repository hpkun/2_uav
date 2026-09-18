"""Action-marginal credit primitives for CF/RDC-HAPPO v2."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from env.mavuav import GLOBAL_STATE_DIM, RED_IDS


CF_METHOD = "cf_happo"
RDC_METHOD = "rdc_happo"
CREDIT_METHODS = frozenset((CF_METHOD, RDC_METHOD))
RDC_COMPONENT_NAMES = ("shared", "mav_role", "uav1_role", "uav2_role", "uav3_role")
RDC_COMPONENT_WEIGHTS = np.asarray((1.0, 0.25, 0.25, 0.25, 0.25), dtype=np.float32)


def credit_component_names(method: str) -> tuple[str, ...]:
    if method == CF_METHOD:
        return ("team",)
    if method == RDC_METHOD:
        return RDC_COMPONENT_NAMES
    raise ValueError(f"method does not use counterfactual credit: {method!r}")


def component_weights(method: str) -> np.ndarray:
    if method == CF_METHOD:
        return np.ones(1, dtype=np.float32)
    if method == RDC_METHOD:
        return RDC_COMPONENT_WEIGHTS.copy()
    raise ValueError(f"method does not use counterfactual credit: {method!r}")


def extract_credit_components(
    infos: Sequence[Mapping[str, Any]], rewards: np.ndarray, method: str,
) -> np.ndarray:
    """Read the environment's realized reward accounting; never recompute geometry."""
    per_agent_rewards = np.asarray(rewards, dtype=np.float32)
    team_rewards = per_agent_rewards.mean(axis=-1)
    if method == CF_METHOD:
        return team_rewards[:, None]
    if method != RDC_METHOD:
        raise ValueError(f"method does not use counterfactual credit: {method!r}")
    components = np.asarray([
        [
            float(info["event_reward"]) + float(info["terminal_reward"]) + float(info["safety_reward"]),
            float(info["mav_process_reward"]),
            float(info["uav1_process_reward"]),
            float(info["uav2_process_reward"]),
            float(info["uav3_process_reward"]),
        ]
        for info in infos
    ], dtype=np.float32)
    reconstructed = components @ RDC_COMPONENT_WEIGHTS
    if not np.allclose(reconstructed, team_rewards, rtol=1e-6, atol=1e-5):
        maximum_error = float(np.max(np.abs(reconstructed - team_rewards)))
        raise RuntimeError(
            "v3.9 RDC reward decomposition does not reconstruct team reward "
            f"(maximum error {maximum_error:.8g})"
        )
    return components


def compute_component_lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """GAE-style component returns with the exact RolloutBuffer boundary semantics."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(last_values, dtype=np.float32)
    returns = np.empty_like(rewards)
    gae = np.zeros_like(next_values)
    for step in reversed(range(rewards.shape[0])):
        boundary = np.logical_or(terminated[step], truncated[step]).astype(np.float32)
        continuation = (1.0 - boundary)[:, None]
        delta = rewards[step] + gamma * next_values * continuation - values[step]
        gae = delta + gamma * gae_lambda * continuation * gae
        returns[step] = gae + values[step]
        next_values = values[step]
    return returns


def other_agent_actions(joint_actions: torch.Tensor, agent_index: int) -> torch.Tensor:
    """Return [B,9] actions in RED_IDS order with the current agent removed."""
    if joint_actions.ndim != 3 or joint_actions.shape[1:] != (len(RED_IDS), 3):
        raise ValueError("joint_actions must have shape [batch, 4, 3]")
    if not 0 <= int(agent_index) < len(RED_IDS):
        raise IndexError("agent_index is out of range")
    indices = [index for index in range(len(RED_IDS)) if index != int(agent_index)]
    return joint_actions[:, indices].reshape(joint_actions.shape[0], -1)


def component_residual(component_returns: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
    if component_returns.shape != baselines.shape:
        raise ValueError("component returns and baselines must share shape")
    return component_returns - baselines


def combine_rdc_component_residual(
    component_returns: torch.Tensor, baselines: torch.Tensor,
) -> torch.Tensor:
    """Preserve the full team objective using its exact environment-defined weights."""
    if component_returns.shape[-1] != len(RDC_COMPONENT_NAMES) or component_returns.shape != baselines.shape:
        raise ValueError("RDC component tensors must share shape [..., 5]")
    weights = torch.as_tensor(
        RDC_COMPONENT_WEIGHTS, dtype=component_returns.dtype, device=component_returns.device,
    )
    return ((component_returns - baselines) * weights).sum(dim=-1)


def normalize_credit_advantage(
    advantages: torch.Tensor, active: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Normalize informative active residuals; suppress only numerical degeneracy."""
    normalized = torch.zeros_like(advantages)
    active = active > 0.5
    if not active.any():
        return normalized, True
    active_values = advantages[active]
    std = active_values.std(unbiased=False)
    if std <= 1e-6:
        # The threshold only disables numerically degenerate residual estimators.
        # It does not rescale informative advantages or alter the task reward.
        return normalized, True
    normalized[active] = (active_values - active_values.mean()) / std
    return normalized, False


class ActionMarginalCreditCritic(nn.Module):
    """Component values plus independent B_i(s, a_-i) marginal baselines."""

    def __init__(
        self,
        component_count: int,
        state_dim: int = GLOBAL_STATE_DIM,
        num_agents: int = len(RED_IDS),
        action_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.component_count = int(component_count)
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.other_action_dim = int((num_agents - 1) * action_dim)
        self.hidden_dim = int(hidden_dim)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, component_count),
        )
        self.marginal_baselines = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + self.other_action_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, component_count),
            )
            for _ in range(num_agents)
        ])

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_network(states)

    def baseline_for_agent(
        self, states: torch.Tensor, joint_actions: torch.Tensor, agent_index: int,
    ) -> torch.Tensor:
        others = other_agent_actions(joint_actions, agent_index)
        return self.marginal_baselines[int(agent_index)](torch.cat((states, others), dim=-1))

    def baselines(self, states: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self.baseline_for_agent(states, joint_actions, agent)
            for agent in range(self.num_agents)
        ], dim=1)

    def architecture(self) -> dict[str, int]:
        return {
            "state_dim": self.state_dim,
            "other_action_dim": self.other_action_dim,
            "num_agents": self.num_agents,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "component_count": self.component_count,
            "baseline_type": "agent_specific_action_marginal",
        }
