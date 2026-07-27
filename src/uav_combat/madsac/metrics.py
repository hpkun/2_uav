"""Metric aggregation helpers for MADSAC training logs."""
from __future__ import annotations

import math
import numbers
from typing import Any


CRITIC_MEAN_FIELDS = (
    ("critic1_loss_mean", "critic1_loss", "critic1_loss_mean"),
    ("critic2_loss_mean", "critic2_loss", "critic2_loss_mean"),
    ("q1_mean", "q1_mean", "q1_mean"),
    ("q2_mean", "q2_mean", "q2_mean"),
    ("target_q_mean", "target_q_mean", "target_q_mean"),
    ("q1_q2_abs_gap_mean", "q1_q2_abs_gap", "q1_q2_abs_gap_mean"),
    ("td_error_abs_mean", "td_error_abs_mean", "td_error_abs_mean"),
    ("critic1_grad_norm_pre_clip_mean", "critic1_grad_norm_pre_clip", "critic1_grad_norm_pre_clip_mean"),
    ("critic2_grad_norm_pre_clip_mean", "critic2_grad_norm_pre_clip", "critic2_grad_norm_pre_clip_mean"),
    ("critic1_grad_clipped_fraction", "critic1_grad_clipped", "critic1_grad_clipped_fraction"),
    ("critic2_grad_clipped_fraction", "critic2_grad_clipped", "critic2_grad_clipped_fraction"),
)

CRITIC_MAX_FIELDS = (
    ("critic1_loss", "critic1_loss_max"),
    ("critic2_loss", "critic2_loss_max"),
    ("q1_q2_abs_gap_max", "q1_q2_abs_gap_max"),
    ("td_error_abs_max", "td_error_abs_max"),
    ("critic1_grad_norm_pre_clip", "critic1_grad_norm_pre_clip_max"),
    ("critic2_grad_norm_pre_clip", "critic2_grad_norm_pre_clip_max"),
)

ACTOR_MEAN_FIELDS = (
    ("actor_loss_mean", "actor_loss", "actor_loss_mean"),
    ("sampled_log_prob_mean", "sampled_log_prob_mean", "sampled_log_prob_mean"),
    ("deterministic_action_abs_mean", "deterministic_action_abs_mean", "deterministic_action_abs_mean"),
    ("stochastic_action_abs_mean", "stochastic_action_abs_mean", "stochastic_action_abs_mean"),
    ("action_saturation_fraction_mean", "action_saturation_fraction", "action_saturation_fraction_mean"),
    ("actor_grad_norm_pre_clip_mean", "actor_grad_norm_pre_clip", "actor_grad_norm_pre_clip_mean"),
    ("actor_grad_clipped_fraction", "actor_grad_clipped", "actor_grad_clipped_fraction"),
)

ACTOR_MAX_FIELDS = (
    ("action_saturation_fraction", "action_saturation_fraction_max"),
    ("actor_grad_norm_pre_clip", "actor_grad_norm_pre_clip_max"),
)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Real):
        out = float(value)
        if not math.isfinite(out):
            raise FloatingPointError(f"non-finite MADSAC metric: {value!r}")
        return out
    return None


class MADSACMetricAccumulator:
    """Aggregate ``MADSAC3v3Trainer.update`` outputs over one log interval.

    Critic statistics are weighted by the number of critic updates represented
    by each update-call result. Actor statistics are weighted only by real actor
    optimizer steps, so critic-only calls cannot dilute actor metrics with
    synthetic zeros.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.update_calls = 0
        self.critic_updates = 0
        self.actor_updates = 0
        self.target_updates = 0
        self._critic_sums: dict[str, float] = {}
        self._critic_max: dict[str, float] = {}
        self._actor_sums: dict[str, float] = {}
        self._actor_max: dict[str, float] = {}
        self._actor_loss_last: float | None = None

    def add(self, update_metrics: dict[str, Any]) -> None:
        if not update_metrics:
            return
        self.update_calls += 1
        critic_n = int(update_metrics.get("critic_updates_in_call", 0) or 0)
        actor_n = int(update_metrics.get("actor_updates_in_call", 0) or 0)
        target_n = int(update_metrics.get("target_updates_in_call", 0) or 0)
        self.critic_updates += critic_n
        self.actor_updates += actor_n
        self.target_updates += target_n

        for primary, legacy, target in CRITIC_MEAN_FIELDS:
            value = _finite_or_none(update_metrics.get(primary, update_metrics.get(legacy)))
            if value is not None and critic_n > 0:
                self._critic_sums[target] = self._critic_sums.get(target, 0.0) + value * critic_n
        for source, target in CRITIC_MAX_FIELDS:
            value = _finite_or_none(update_metrics.get(target, update_metrics.get(source)))
            if value is not None:
                self._critic_max[target] = max(self._critic_max.get(target, value), value)

        if actor_n > 0:
            for primary, legacy, target in ACTOR_MEAN_FIELDS:
                value = _finite_or_none(update_metrics.get(primary, update_metrics.get(legacy)))
                if value is not None:
                    self._actor_sums[target] = self._actor_sums.get(target, 0.0) + value * actor_n
            for source, target in ACTOR_MAX_FIELDS:
                value = _finite_or_none(update_metrics.get(target, update_metrics.get(source)))
                if value is not None:
                    self._actor_max[target] = max(self._actor_max.get(target, value), value)
            last = _finite_or_none(update_metrics.get("actor_loss_last"))
            if last is not None:
                self._actor_loss_last = last

    def summarize(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {
            "update_calls_in_interval": self.update_calls,
            "critic_updates_in_interval": self.critic_updates,
            "actor_updates_in_interval": self.actor_updates,
            "target_updates_in_interval": self.target_updates,
        }
        for _, _, target in CRITIC_MEAN_FIELDS:
            out[target] = (
                self._critic_sums[target] / self.critic_updates
                if self.critic_updates > 0 and target in self._critic_sums
                else None
            )
        for _, target in CRITIC_MAX_FIELDS:
            out[target] = self._critic_max.get(target)

        for _, _, target in ACTOR_MEAN_FIELDS:
            out[target] = (
                self._actor_sums[target] / self.actor_updates
                if self.actor_updates > 0 and target in self._actor_sums
                else None
            )
        out["actor_loss_last"] = self._actor_loss_last
        for _, target in ACTOR_MAX_FIELDS:
            out[target] = self._actor_max.get(target)

        for value in out.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"non-finite MADSAC interval metric: {out}")
        return out


__all__ = ["MADSACMetricAccumulator"]
