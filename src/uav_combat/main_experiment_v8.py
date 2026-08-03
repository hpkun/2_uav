"""Shared contract helpers for the main-experiment v8 configurations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

V8_REWARD_MODES = {
    "task_aligned_paper_segmented_team_v8",
    "task_aligned_heterogeneous_paper_segmented_team_v8",
}

V8_ALLOWED_DEATH_CAUSES = ("ATTACK", "BOUNDARY_ALTITUDE", "BOUNDARY_XY")

_V8_BEST_SCORE_FIELDS = (
    "red_complete_elimination_success_rate",
    "red_any_attack_kill_rate",
    "mean_red_attack_kills",
    "mean_red_survivors",
    "neg_mean_red_boundary_deaths",
    "neg_max_steps_rate",
    "neg_mean_episode_length",
)

_LEGACY_BEST_SCORE_FIELDS = (
    "red_complete_elimination_success_rate",
    "red_any_attack_kill_rate",
    "mean_red_attack_kills",
    "mean_red_survivors",
    "neg_mean_red_boundary_deaths",
    "neg_mean_red_collision_deaths",
    "neg_max_steps_rate",
    "neg_mean_episode_length",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_main_v8_config(config: dict[str, Any]) -> bool:
    return config.get("combat", {}).get("reward_mode") in V8_REWARD_MODES


def _v8_reward_config(config: dict[str, Any]) -> dict[str, Any]:
    mode = config.get("combat", {}).get("reward_mode")
    if mode == "task_aligned_heterogeneous_paper_segmented_team_v8":
        return config.get("reward_heterogeneous_v8", {})
    return config.get("reward_v8", {})


def validate_main_v8_contract(config: dict[str, Any]) -> None:
    """Validate official v8 configs; no-op for historical non-v8 configs."""
    mode = config.get("combat", {}).get("reward_mode")
    if mode == "task_aligned_heterogeneous_team_v8":
        raise ValueError(
            "legacy continuous v8 reward mode is not allowed; use "
            "task_aligned_heterogeneous_paper_segmented_team_v8"
        )
    if mode not in V8_REWARD_MODES:
        return

    battlefield = config.get("battlefield", {})
    collision_distance = float(battlefield.get("collision_distance", 0.0))
    if abs(collision_distance) > 1e-12:
        raise ValueError(
            "main experiment v8 requires battlefield.collision_distance == 0.0; "
            f"got {collision_distance!r}"
        )

    reward_cfg = _v8_reward_config(config)
    paper_segment_distance = float(reward_cfg.get("paper_segment_distance", 0.0))
    if paper_segment_distance <= 0.0:
        raise ValueError("main experiment v8 requires positive paper_segment_distance")

    combat = config.get("combat", {})
    attack_min = float(combat.get("attack_distance_min", 0.0))
    attack_max = float(combat.get("attack_distance_max", 0.0))
    if not attack_min < attack_max:
        raise ValueError("main experiment v8 requires attack_distance_min < attack_distance_max")
    if not attack_max < paper_segment_distance:
        raise ValueError("main experiment v8 requires attack_distance_max < paper_segment_distance")

    blue_mode = config.get("blue_rule_policy", {}).get("mode", "paper_nearest_pursuit_v1")
    red_mode = config.get("red_rule_policy", {}).get("mode", "paper_nearest_pursuit_v1")
    if mode == "task_aligned_paper_segmented_team_v8":
        expected = "paper_nearest_pursuit_v1"
        if blue_mode != expected or red_mode != expected:
            raise ValueError("homogeneous main v8 requires paper_nearest_pursuit_v1 rule policies")
    else:
        expected = "functional_heterogeneous_nearest_pursuit_v8"
        if blue_mode != expected or red_mode != expected:
            raise ValueError("heterogeneous main v8 requires functional_heterogeneous_nearest_pursuit_v8 rule policies")

    if int(config.get("observation_dim", 68)) != 68:
        raise ValueError("main experiment v8 observation contract must remain 68")


def build_main_v8_contract_metadata(config: dict[str, Any]) -> dict[str, Any]:
    reward_cfg = _v8_reward_config(config)
    combat = config.get("combat", {})
    collision_distance = float(config.get("battlefield", {}).get("collision_distance", 0.0))
    return {
        "reward_mode": combat.get("reward_mode"),
        "collision_enabled": bool(collision_distance > 0.0),
        "allowed_death_causes": list(V8_ALLOWED_DEATH_CAUSES),
        "attack_distance_min": float(combat.get("attack_distance_min", 0.0)),
        "attack_distance_max": float(combat.get("attack_distance_max", 0.0)),
        "paper_segment_distance": float(reward_cfg.get("paper_segment_distance", 0.0)),
        "observation_dim": 68,
    }


def v8_best_score_fields() -> tuple[str, ...]:
    return _V8_BEST_SCORE_FIELDS


def best_score_fields_for_config(config: dict[str, Any]) -> tuple[str, ...]:
    return _V8_BEST_SCORE_FIELDS if is_main_v8_config(config) else _LEGACY_BEST_SCORE_FIELDS


def compute_best_score_for_config(summary: dict[str, Any], config: dict[str, Any]) -> tuple[float, ...]:
    fields = best_score_fields_for_config(config)
    values: list[float] = []
    for field in fields:
        if field == "red_any_attack_kill_rate":
            value = summary.get(field)
            if value is None:
                value = 1.0 if float(summary.get("mean_red_attack_kills", 0.0)) > 0.0 else 0.0
            values.append(float(value))
        elif field.startswith("neg_"):
            values.append(-float(summary.get(field[4:], 600.0 if field == "neg_mean_episode_length" else 0.0)))
        else:
            values.append(float(summary.get(field, 0.0)))
    return tuple(values)


def best_score_values_for_config(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    return {
        name: value
        for name, value in zip(best_score_fields_for_config(config), compute_best_score_for_config(summary, config))
    }


def infer_best_score_schema_for_checkpoint(ckpt: dict[str, Any], config: dict[str, Any]) -> tuple[str, ...]:
    schema = ckpt.get("best_score_schema")
    if schema:
        return tuple(schema)
    return best_score_fields_for_config(config)


def filter_public_metrics_for_config(metrics: Any, config: dict[str, Any]) -> Any:
    """Remove public-facing collision metrics for v8 while preserving raw internals elsewhere."""
    if not is_main_v8_config(config):
        return metrics
    if isinstance(metrics, dict):
        return {
            key: filter_public_metrics_for_config(value, config)
            for key, value in metrics.items()
            if "collision" not in str(key).lower()
        }
    if isinstance(metrics, list):
        return [filter_public_metrics_for_config(value, config) for value in metrics]
    if isinstance(metrics, tuple):
        return tuple(filter_public_metrics_for_config(value, config) for value in metrics)
    return metrics
