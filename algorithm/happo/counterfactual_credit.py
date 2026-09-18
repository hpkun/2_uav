"""Continuous-action counterfactual credit primitives for CF/RDC-HAPPO."""
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


def replace_agent_action(
    joint_actions: torch.Tensor, agent_index: int, counterfactual_actions: torch.Tensor,
) -> torch.Tensor:
    """Clone a [B,A,D] joint action and replace exactly one agent's action."""
    if joint_actions.ndim != 3:
        raise ValueError("joint_actions must have shape [batch, agents, action_dim]")
    if counterfactual_actions.shape != joint_actions[:, agent_index].shape:
        raise ValueError("counterfactual action shape mismatch")
    replaced = joint_actions.clone()
    replaced[:, agent_index] = counterfactual_actions.detach()
    return replaced


def component_credit(actual_q: torch.Tensor, counterfactual_q: torch.Tensor) -> torch.Tensor:
    return actual_q - counterfactual_q


def combine_rdc_component_credit(
    actual_q: torch.Tensor, counterfactual_q: torch.Tensor,
) -> torch.Tensor:
    """Preserve the full team objective using its exact environment-defined weights."""
    if actual_q.shape[-1] != len(RDC_COMPONENT_NAMES) or actual_q.shape != counterfactual_q.shape:
        raise ValueError("RDC Q tensors must share shape [..., 5]")
    weights = torch.as_tensor(RDC_COMPONENT_WEIGHTS, dtype=actual_q.dtype, device=actual_q.device)
    return ((actual_q - counterfactual_q) * weights).sum(dim=-1)


class CounterfactualCreditCritic(nn.Module):
    """State-value and continuous joint-action Q heads for fixed reward components."""

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
        self.joint_action_dim = int(num_agents * action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, component_count),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + self.joint_action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, component_count),
        )

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.state_encoder(states))

    def q_values(self, states: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        embedding = self.state_encoder(states)
        flattened = joint_actions.reshape(joint_actions.shape[0], -1)
        if flattened.shape[-1] != self.joint_action_dim:
            raise ValueError("joint action dimension mismatch")
        return self.q_head(torch.cat((embedding, flattened), dim=-1))

    def architecture(self) -> dict[str, int]:
        return {
            "state_dim": self.state_dim,
            "joint_action_dim": self.joint_action_dim,
            "hidden_dim": self.hidden_dim,
            "component_count": self.component_count,
        }
