"""Version-isolated probability and value contracts for the pure v3.15 baseline."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from algorithm.common.buffer import RolloutBuffer

V315_DEFAULTS = {
    "action_probability_protocol": "latent_gaussian_raw_v1",
    "entropy_semantics": "latent_gaussian", "use_valuenorm": True,
    "use_huber_loss": True, "huber_delta": 10.0,
    "use_clipped_value_loss": True, "orthogonal_init": True,
}


class LatentRolloutBuffer(RolloutBuffer):
    def __init__(self, horizon, num_envs):
        super().__init__(horizon, num_envs)
        self.raw_actions = np.zeros_like(self.actions)
        # Keep the critic's collection-time output unchanged for value clipping.
        # self.values remains in raw reward units for team GAE.
        self.old_normalized_values = np.zeros_like(self.values)

    def insert(self, *args, raw_actions, old_normalized_values, **kwargs):
        self.raw_actions[self.position] = raw_actions
        self.old_normalized_values[self.position] = old_normalized_values
        super().insert(*args, **kwargs)


def orthogonal_initialize(network: nn.Module, output_gain: float) -> None:
    linear = [module for module in network.modules() if isinstance(module, nn.Linear)]
    for layer in linear[:-1]:
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linear[-1].weight, gain=output_gain)
    nn.init.zeros_(linear[-1].bias)


def validate_metadata(payload, config=V315_DEFAULTS) -> None:
    saved_config = payload.get("trainer_config", payload.get("config", {}))
    for key in V315_DEFAULTS:
        if payload.get(key) != config[key] or saved_config.get(key) != config[key]:
            raise RuntimeError(f"incompatible v3.15 checkpoint contract: {key}")
    state = payload.get("value_normalizer")
    required = {"running_mean", "running_mean_sq", "debiasing_term"}
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError("v3.15 checkpoint requires complete ValueNorm state")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid v3.15 ValueNorm state: {key}")
    if state["debiasing_term"].item() < 0 or state["debiasing_term"].item() > 1:
        raise RuntimeError("invalid v3.15 ValueNorm debiasing term")


class ValueNorm(nn.Module):
    """Debiased EMA moments (HARL defaults), tracked independently of critic weights.

    Updated once per completed rollout; stats stay fixed during its PPO epochs.
    Float64 statistics avoid cancellation; public values keep the input dtype.
    """
    def __init__(self, beta: float = .99999, epsilon: float = 1e-5):
        super().__init__()
        self.beta, self.epsilon = beta, epsilon
        for key in ("running_mean", "running_mean_sq", "debiasing_term"):
            self.register_buffer(key, torch.zeros((), dtype=torch.float64))

    def moments(self):
        divisor = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / divisor
        variance = (self.running_mean_sq / divisor - mean.square()).clamp(min=1e-2)
        return mean, variance

    @torch.no_grad()
    def update(self, values):
        batch = torch.as_tensor(values, device=self.running_mean.device, dtype=torch.float64)
        if not batch.numel() or not torch.isfinite(batch).all():
            raise ValueError("ValueNorm requires nonempty finite targets")
        self.running_mean.mul_(self.beta).add_(batch.mean() * (1 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch.square().mean() * (1 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, values):
        mean, variance = self.moments()
        return ((values - mean) / variance.sqrt()).to(values.dtype)

    def denormalize(self, values):
        mean, variance = self.moments()
        return (values * variance.sqrt() + mean).to(values.dtype)


def clipped_huber_value_loss(new, old, target, clip: float = .2, delta: float = 10.):
    clipped = old + (new - old).clamp(-clip, clip)
    def huber(error):
        absolute = error.abs()
        return torch.where(absolute <= delta, .5 * error.square(), delta * (absolute - .5 * delta))
    return torch.maximum(huber(target - new), huber(target - clipped)).mean()
