"""HAPPO trainer for the functional heterogeneous red 4v3 v9 environment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import load_config
from ..environment_4v3 import GS_DIM_4V3, OBS_DIM_4V3
from ..environment_4v3_v11 import GS_DIM_V11, OBS_DIM_V11
from ..environment_4v3_v12 import GS_DIM_V12, OBS_DIM_V12
from ..mappo.trainer_3v3 import linear_schedule, resolve_device
from ..mappo.vector_env_4v3 import RED_REWARD_COMPONENT_KEYS_4V3, RED_TEAM_SIZE_4V3, make_combat_vector_env_4v3
from ..mappo.vector_env_4v3_v11 import REWARD_COMPONENT_KEYS_V11, make_combat_vector_env_4v3_v11
from ..mappo.vector_env_4v3_v12 import REWARD_COMPONENT_KEYS_V12, make_combat_vector_env_4v3_v12
from ..scenario_4v3 import resolved_reward_contract_4v3
from ..scenario_4v3_v11 import resolved_reward_contract_v11
from ..scenario_4v3_v12 import resolved_reward_contract_v12
from .buffer_3v3 import HAPPORolloutBuffer3v3
from .metrics import explained_variance
from .networks import CentralizedValueCritic, IndependentHAPPOActors
from .trainer_3v3 import (
    happo_preceding_factor_update,
    normalize_advantages_for_agent,
    ppo_clipped_policy_loss,
    sha256_file,
    signature_mismatches,
)

CHECKPOINT_FAMILY_HAPPO_4V3 = "functional_heterogeneous_4v3_v9_happo"
CHECKPOINT_VERSION_HAPPO_4V3 = 1


def _restore_cuda_rng_state(states: Any) -> None:
    """Restore CUDA RNG states as CPU byte tensors required by PyTorch."""
    if isinstance(states, torch.Tensor):
        states = [states]
    normalized = []
    for state in states:
        if isinstance(state, torch.Tensor):
            normalized.append(state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
        else:
            normalized.append(torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous())
    torch.cuda.set_rng_state_all(normalized)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def best_score_fields_4v3(v11: bool = False, v12: bool = False) -> list[str]:
    if v12:
        return [
            "strict_full_elimination_rate", "at_least_two_kill_rate", "task_win_rate",
            "any_kill_rate", "mean_red_kills", "mean_red_combat_survivors",
            "support_assisted_kill_rate", "negative_mean_episode_length",
        ]
    if v11:
        return [
            "task_win_rate", "full_elimination_rate", "at_least_two_kill_rate",
            "any_kill_rate", "mean_red_kills", "support_assisted_kill_rate",
            "mean_red_combat_survivors", "negative_mean_episode_length",
        ]
    return [
        "red_complete_elimination_success_rate",
        "red_at_least_two_attack_kill_rate",
        "red_any_attack_kill_rate",
        "mean_red_attack_kills",
        "support_assisted_kill_rate",
        "mean_red_combat_survivors",
        "negative_timeout_rate",
        "negative_mean_episode_length",
    ]


def compute_best_score_4v3(summary: dict[str, float]) -> tuple[tuple[float, ...], dict[str, float]]:
    if "strict_full_elimination_rate" in summary:
        fields = {
            "strict_full_elimination_rate": float(summary.get("strict_full_elimination_rate", 0.0)),
            "at_least_two_kill_rate": float(summary.get("at_least_two_kill_rate", 0.0)),
            "task_win_rate": float(summary.get("task_win_rate", 0.0)),
            "any_kill_rate": float(summary.get("any_kill_rate", 0.0)),
            "mean_red_kills": float(summary.get("mean_red_kills", 0.0)),
            "mean_red_combat_survivors": float(summary.get("mean_red_combat_survivors", 0.0)),
            "support_assisted_kill_rate": float(summary.get("support_assisted_kill_rate", 0.0)),
            "negative_mean_episode_length": -float(summary.get("mean_episode_length", 0.0)),
        }
        return tuple(fields[key] for key in best_score_fields_4v3(v12=True)), fields
    if "task_win_rate" in summary:
        fields = {
            "task_win_rate": float(summary.get("task_win_rate", 0.0)),
            "full_elimination_rate": float(summary.get("full_elimination_rate", 0.0)),
            "at_least_two_kill_rate": float(summary.get("at_least_two_kill_rate", 0.0)),
            "any_kill_rate": float(summary.get("any_kill_rate", 0.0)),
            "mean_red_kills": float(summary.get("mean_red_kills", 0.0)),
            "support_assisted_kill_rate": float(summary.get("support_assisted_kill_rate", 0.0)),
            "mean_red_combat_survivors": float(summary.get("mean_red_combat_survivors", 0.0)),
            "negative_mean_episode_length": -float(summary.get("mean_episode_length", 0.0)),
        }
        return tuple(fields[key] for key in best_score_fields_4v3(True)), fields
    fields = {
        "red_complete_elimination_success_rate": float(summary.get("red_complete_elimination_success_rate", 0.0)),
        "red_at_least_two_attack_kill_rate": float(summary.get("red_at_least_two_attack_kill_rate", 0.0)),
        "red_any_attack_kill_rate": float(summary.get("red_any_attack_kill_rate", 0.0)),
        "mean_red_attack_kills": float(summary.get("mean_red_attack_kills", 0.0)),
        "support_assisted_kill_rate": float(summary.get("support_assisted_kill_rate", 0.0)),
        "mean_red_combat_survivors": float(summary.get("mean_red_combat_survivors", 0.0)),
        "negative_timeout_rate": -float(summary.get("timeout_rate", 0.0)),
        "negative_mean_episode_length": -float(summary.get("mean_episode_length", 0.0)),
    }
    return tuple(fields[key] for key in best_score_fields_4v3()), fields


def _summarize_v11_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0.0}
    n = len(records)
    mean = lambda key, default=0.0: float(np.mean([float(record.get(key, default)) for record in records]))
    total_kills = sum(int(record.get("red_attack_kills", 0)) for record in records)
    total_assists = sum(int(record.get("support_assisted_kills", 0)) for record in records)
    summary = {
        "episodes": float(n),
        "task_win_rate": mean("task_win"),
        "full_elimination_rate": mean("full_elimination"),
        "timeout_win_rate": mean("timeout_red_win"),
        "timeout_loss_rate": mean("timeout_red_loss"),
        "timeout_draw_rate": mean("timeout_draw"),
        "timeout_rate": float(np.mean([str(record.get("termination_reason", "")).startswith("timeout") for record in records])),
        "mutual_elimination_draw_rate": mean("mutual_elimination_draw"),
        "any_kill_rate": mean("red_any_attack_kill"),
        "at_least_two_kill_rate": float(np.mean([int(record.get("red_attack_kills", 0)) >= 2 for record in records])),
        "mean_red_kills": mean("red_attack_kills"),
        "mean_blue_kills": mean("blue_attack_kills"),
        "support_assisted_kill_rate": float(total_assists / max(1, total_kills)),
        "support_assisted_episode_rate": mean("support_assisted_episode_rate"),
        "mean_red_combat_survivors": mean("red_combat_survivors"),
        "mean_episode_length": mean("episode_length"),
        "mean_return": mean("episode_return"),
        "mean_first_kill_time": float(np.mean([record["first_kill_time"] for record in records if record.get("first_kill_time") is not None])) if any(record.get("first_kill_time") is not None for record in records) else None,
        "lock_episode_rate": mean("lock_episode_rate"),
        "half_lock_episode_rate": mean("half_lock_episode_rate"),
        "mean_max_lock_progress": mean("mean_max_lock_progress"),
        "red_lock_episode_rate": mean("red_lock_episode_rate"),
        "red_half_lock_episode_rate": mean("red_half_lock_episode_rate"),
        "mean_red_max_lock_progress": mean("mean_red_max_lock_progress"),
        "red_lock_active_step_rate": mean("red_lock_active_step_rate"),
        "red_half_lock_active_step_rate": mean("red_half_lock_active_step_rate"),
        "blue_lock_episode_rate": mean("blue_lock_episode_rate"),
        "blue_half_lock_episode_rate": mean("blue_half_lock_episode_rate"),
        "mean_blue_max_lock_progress": mean("mean_blue_max_lock_progress"),
        "blue_lock_active_step_rate": mean("blue_lock_active_step_rate"),
        "blue_half_lock_active_step_rate": mean("blue_half_lock_active_step_rate"),
        "mean_blue_combat_survivors": mean("blue_combat_survivors"),
        "mean_target_switch_count": mean("target_switch_count"),
        "support_cue_rate": mean("support_cue_rate"),
        "support_cue_pair_step_rate": mean("support_cue_pair_step_rate"),
        "mean_support_active_cue_steps": mean("support_active_cue_steps"),
        "mean_support_active_cue_pair_steps": mean("support_active_cue_pair_steps"),
        "mean_support_eligible_steps": mean("support_eligible_steps"),
        "mean_support_cue_update_count": mean("support_cue_update_count"),
        "support_cue_to_direct_rate": mean("support_cue_to_direct_rate"),
        "support_assisted_kill_rate_episode_denominator": mean("support_assisted_episode_rate"),
        "mean_dense_clip_positive_saturation_rate": mean("dense_clip_positive_saturation_rate"),
        "mean_dense_clip_negative_saturation_rate": mean("dense_clip_negative_saturation_rate"),
        "mean_dense_clip_saturation_rate": mean("dense_clip_saturation_rate"),
        "mean_raw_dense_reward": mean("raw_dense_reward_mean"),
        "min_raw_dense_reward": float(np.min([float(record.get("raw_dense_reward_min", 0.0)) for record in records])),
        "max_raw_dense_reward": float(np.max([float(record.get("raw_dense_reward_max", 0.0)) for record in records])),
        # Compatibility aliases used by the existing 4v3 reporting code.
        "red_win_rate": mean("task_win"),
        "red_complete_elimination_success_rate": mean("full_elimination"),
        "red_at_least_two_attack_kill_rate": float(np.mean([int(record.get("red_attack_kills", 0)) >= 2 for record in records])),
        "red_any_attack_kill_rate": mean("red_any_attack_kill"),
        "mean_red_attack_kills": mean("red_attack_kills"),
    }
    for key in REWARD_COMPONENT_KEYS_V11:
        if key != "team_total_reward":
            summary[key] = float(np.mean([record.get("reward_components", {}).get(key, 0.0) for record in records]))
    summary["team_total_reward"] = float(np.mean([record.get("reward_components", {}).get("team_total_reward", record.get("episode_return", 0.0)) for record in records]))
    distribution = {str(index): 0.0 for index in range(4)}
    for record in records:
        distribution[str(min(3, int(record.get("red_attack_kills", 0))))] += 1.0 / n
    summary["red_attack_kill_distribution"] = distribution
    summary["episode_records"] = deepcopy(records)
    return summary


def _summarize_v12_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize v12 while keeping v11-compatible reporting fields."""
    summary = _summarize_v11_episodes(records)
    n = max(1, len(records))
    mean = lambda key, default=0.0: float(np.mean([float(record.get(key, default)) for record in records]))
    summary.update({
        "strict_full_elimination_rate": mean("strict_full_elimination"),
        "full_elimination_rate": mean("strict_full_elimination"),
        "red_complete_elimination_success_rate": mean("strict_full_elimination"),
        "task_win_rate": mean("task_win"),
        "any_kill_rate": mean("red_any_attack_kill"),
        "at_least_two_kill_rate": float(np.mean([int(record.get("red_lock_kills", record.get("red_attack_kills", 0))) >= 2 for record in records])),
        "mean_red_kills": mean("red_lock_kills", mean("red_attack_kills")),
        "mean_red_lock_kills": mean("red_lock_kills"),
        "mean_blue_lock_kills": mean("blue_lock_kills"),
        "full_elimination_consistency_pass": float(np.mean([record.get("full_elimination_consistency_pass", False) for record in records])),
        "non_lock_blue_death_count": float(sum(int(record.get("non_lock_blue_death_count", 0)) for record in records)),
        "non_lock_red_combat_death_count": float(sum(int(record.get("non_lock_red_combat_death_count", 0)) for record in records)),
        "red_boundary_soft_recovery_step_rate": mean("red_boundary_soft_recovery_step_rate"),
        "blue_boundary_soft_recovery_step_rate": mean("blue_boundary_soft_recovery_step_rate"),
        "support_boundary_soft_recovery_step_rate": mean("support_boundary_soft_recovery_step_rate"),
        "support_cue_to_half_lock_rate": mean("support_cue_to_half_lock_rate"),
        "red_boundary_hard_contacts": float(sum(int(record.get("red_boundary_hard_contacts", 0)) for record in records)),
        "blue_boundary_hard_contacts": float(sum(int(record.get("blue_boundary_hard_contacts", 0)) for record in records)),
        "support_boundary_hard_contacts": float(sum(int(record.get("support_boundary_hard_contacts", 0)) for record in records)),
        "mean_red_combat_survivors": mean("red_combat_survivors"),
        "red_attack_kill_distribution": {
            str(kills): float(np.mean([int(record.get("red_lock_kills", record.get("red_attack_kills", 0))) == kills for record in records]))
            for kills in range(4)
        },
    })
    # v12 uses lock kills as the canonical red kill count.
    summary["mean_red_attack_kills"] = summary["mean_red_kills"]
    summary["red_any_attack_kill_rate"] = summary["any_kill_rate"]
    summary["red_at_least_two_attack_kill_rate"] = summary["at_least_two_kill_rate"]
    summary["strict_full_elimination_rate"] = float(summary["full_elimination_rate"])
    return summary


def summarize_4v3_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}
    if records[0].get("environment_variant") == "functional_heterogeneous_4v3_v11_target_lock_support_cue":
        return _summarize_v11_episodes(records)
    if records[0].get("environment_variant") in {
        "functional_heterogeneous_4v3_v12_soft_boundary_combat_aligned",
        "functional_heterogeneous_4v3_v14_mission_aligned_role_credit",
    }:
        return _summarize_v12_episodes(records)
    n = len(records)
    def mean_optional(key: str) -> float | None:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def reward_mean(key: str) -> float:
        return float(np.mean([r.get("reward_components", {}).get(key, 0.0) for r in records]))

    total_red_attack_kills = sum(int(r.get("red_attack_kills", 0)) for r in records)
    total_support_assisted_kills = sum(int(r.get("support_assisted_kills", 0)) for r in records)

    out = {
        "episodes": float(n),
        "red_win_rate": float(np.mean([r.get("red_win", False) for r in records])),
        "red_complete_elimination_success_rate": float(np.mean([r.get("red_complete_elimination_success", False) for r in records])),
        "red_at_least_two_attack_kill_rate": float(np.mean([int(r.get("red_attack_kills", 0)) >= 2 for r in records])),
        "red_any_attack_kill_rate": float(np.mean([r.get("red_any_attack_kill", False) for r in records])),
        "mean_red_attack_kills": float(np.mean([r.get("red_attack_kills", 0) for r in records])),
        "mean_blue_attack_kills": float(np.mean([r.get("blue_attack_kills", 0) for r in records])),
        "support_assisted_kill_rate": float(total_support_assisted_kills / max(1, total_red_attack_kills)),
        "support_assisted_episode_rate": float(np.mean([int(r.get("support_assisted_kills", 0)) > 0 for r in records])),
        "mean_support_assisted_kills": float(np.mean([r.get("support_assisted_kills", 0) for r in records])),
        "mean_red_combat_survivors": float(np.mean([r.get("red_combat_survivors", 0) for r in records])),
        "timeout_rate": float(np.mean([r.get("termination_reason") == "timeout" for r in records])),
        "mutual_combat_elimination_rate": float(np.mean([r.get("mutual_combat_elimination", False) for r in records])),
        "blue_noncombat_elimination_rate": float(np.mean([r.get("blue_noncombat_elimination", False) for r in records])),
        "mean_episode_length": float(np.mean([r.get("episode_length", 0) for r in records])),
        "mean_red_first_attack_kill_step": mean_optional("red_first_attack_kill_step"),
        "mean_red_second_attack_kill_step": mean_optional("red_second_attack_kill_step"),
        "mean_red_third_attack_kill_step": mean_optional("red_third_attack_kill_step"),
        "mean_remaining_steps_after_first_kill": float(np.mean([
            r["episode_length"] - r["red_first_attack_kill_step"]
            for r in records if r.get("red_first_attack_kill_step") is not None
        ])) if any(r.get("red_first_attack_kill_step") is not None for r in records) else None,
        "mean_remaining_steps_after_second_kill": float(np.mean([
            r["episode_length"] - r["red_second_attack_kill_step"]
            for r in records if r.get("red_second_attack_kill_step") is not None
        ])) if any(r.get("red_second_attack_kill_step") is not None for r in records) else None,
        "support_survival_rate": float(np.mean([r.get("support_survived", False) for r in records])),
        "red_all_combat_eliminated_rate": float(np.mean([r.get("red_all_combat_eliminated", False) for r in records])),
        "support_unique_detection_step_rate": float(np.mean([r.get("support_unique_detection_step_rate", 0.0) for r in records])),
        "support_shared_target_step_rate": float(np.mean([r.get("support_shared_target_step_rate", 0.0) for r in records])),
        "support_only_target_step_rate": float(np.mean([r.get("support_only_target_step_rate", 0.0) for r in records])),
        "support_shared_pair_step_rate": float(np.mean([r.get("support_shared_pair_step_rate", 0.0) for r in records])),
        "mean_shared_only_combat_target_pairs": float(np.mean([r.get("mean_shared_only_combat_target_pairs", 0.0) for r in records])),
        "mean_shared_only_pair_ratio": float(np.mean([r.get("mean_shared_only_pair_ratio", 0.0) for r in records])),
        "mean_support_active_steps": float(np.mean([r.get("support_active_steps", 0) for r in records])),
        "mean_share_to_direct_delay": mean_optional("mean_share_to_direct_delay"),
        "mean_share_to_kill_delay": mean_optional("mean_share_to_kill_delay"),
        "share_to_direct_event_count": float(sum(int(r.get("share_to_direct_event_count", 0)) for r in records)),
        "share_to_kill_event_count": float(sum(int(r.get("share_to_kill_event_count", 0)) for r in records)),
        "mean_combat_early_acquisition_events": float(np.mean([r.get("combat_early_acquisition_events", r.get("combat_early_acquisition_steps", 0)) for r in records])),
        "mean_combat_early_acquisition_steps": float(np.mean([r.get("combat_early_acquisition_steps", 0) for r in records])),
        "combat_attack_window_step_rate": float(np.mean([r.get("combat_attack_window_step_rate", 0.0) for r in records])),
        "combat_readiness_mean": float(np.mean([r.get("combat_readiness_mean", 0.0) for r in records])),
        "combat_threat_mean": float(np.mean([r.get("combat_threat_mean", 0.0) for r in records])),
        "mean_support_to_combat_centroid_distance": float(np.mean([r.get("mean_support_to_combat_centroid_distance", 0.0) for r in records])),
        "support_rear_position_rate": float(np.mean([r.get("support_rear_position_rate", 0.0) for r in records])),
        "support_threat_exposure_rate": float(np.mean([r.get("support_threat_exposure_rate", 0.0) for r in records])),
        "mean_dense_clip_positive_saturation_rate": float(np.mean([r.get("dense_clip_positive_saturation_rate", 0.0) for r in records])),
        "mean_dense_clip_negative_saturation_rate": float(np.mean([r.get("dense_clip_negative_saturation_rate", 0.0) for r in records])),
        "mean_dense_clip_saturation_rate": float(np.mean([r.get("dense_clip_saturation_rate", 0.0) for r in records])),
        "mean_raw_dense_reward": float(np.mean([r.get("raw_dense_reward_mean", 0.0) for r in records])),
        "min_raw_dense_reward": float(np.min([r.get("raw_dense_reward_min", 0.0) for r in records])),
        "max_raw_dense_reward": float(np.max([r.get("raw_dense_reward_max", 0.0) for r in records])),
    }
    out.update({key: reward_mean(key) for key in RED_REWARD_COMPONENT_KEYS_4V3})
    kill_distribution = {str(k): 0.0 for k in range(4)}
    for r in records:
        kills = int(r.get("red_attack_kills", 0))
        kill_distribution[str(min(3, max(0, kills)))] += 1.0
    out["red_attack_kill_distribution"] = {k: v / n for k, v in kill_distribution.items()}  # type: ignore[assignment]
    out["mean_support_only_target_steps"] = float(np.mean([r.get("support_only_target_steps", 0) for r in records]))
    out["mean_support_shared_pair_steps"] = float(np.mean([r.get("support_shared_pair_steps", 0) for r in records]))
    out["mean_support_unique_detection_steps"] = float(np.mean([r.get("support_unique_detection_steps", 0) for r in records]))
    out["mean_support_shared_target_steps"] = float(np.mean([r.get("support_shared_target_steps", 0) for r in records]))
    out["mean_share_to_direct_event_count"] = float(np.mean([r.get("share_to_direct_event_count", 0) for r in records]))
    out["mean_share_to_kill_event_count"] = float(np.mean([r.get("share_to_kill_event_count", 0) for r in records]))
    out["mean_combat_early_acquisition_events"] = float(np.mean([
        r.get("combat_early_acquisition_events", r.get("combat_early_acquisition_steps", 0))
        for r in records
    ]))
    out["mean_combat_early_acquisition_steps"] = out["mean_combat_early_acquisition_events"]
    out["death_cause_counts"] = {
        str(cause): int(sum(1 for r in records for value in r.get("death_causes", {}).values() if int(value) == cause))
        for cause in (0, 1, 2, 5)
    }
    return out


class HAPPO4v3Trainer:
    """Four red actors (support + three combat) with the existing HAPPO update."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.env_contract_config = load_config(self.env_config)
        self.config = deepcopy(config)
        self.experiment_variant = self.config["experiment"].get("variant")
        self.reward_contract_version = self.env_contract_config["combat"].get("reward_contract_version")
        self.is_v11 = self.reward_contract_version == "v11_target_lock_support_cue"
        self.is_v12 = self.reward_contract_version == "v12_soft_boundary_combat_aligned"
        self.reward_contract = (
            resolved_reward_contract_v12(self.env_contract_config)
            if self.is_v12 else (
                resolved_reward_contract_v11(self.env_contract_config)
                if self.is_v11 else resolved_reward_contract_4v3(self.env_contract_config)
            )
        )
        self.obs_dim = OBS_DIM_V12 if self.is_v12 else (OBS_DIM_V11 if self.is_v11 else OBS_DIM_4V3)
        self.gs_dim = GS_DIM_V12 if self.is_v12 else (GS_DIM_V11 if self.is_v11 else GS_DIM_4V3)
        self.reward_keys = REWARD_COMPONENT_KEYS_V12 if self.is_v12 else (REWARD_COMPONENT_KEYS_V11 if self.is_v11 else RED_REWARD_COMPONENT_KEYS_4V3)
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "fixed_rule_blue_heterogeneous_4v3_happo":
            raise ValueError("training_mode must be fixed_rule_blue_heterogeneous_4v3_happo")
        if int(t.get("team_size", -1)) != RED_TEAM_SIZE_4V3:
            raise ValueError("4v3 HAPPO requires training.team_size=4")
        self.device = resolve_device(e["device"])
        torch.manual_seed(int(e["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.episode_seed_rng = np.random.default_rng(int(e["seed"]) + 1009)
        self.num_envs = int(t["num_envs"])
        self.rollout_steps = int(t["rollout_steps"])
        if self.rollout_steps <= 0:
            raise ValueError("training.rollout_steps must be positive")
        self.total_env_steps = int(t["total_env_steps"])
        if self.total_env_steps <= 0 or self.total_env_steps % self.num_envs != 0:
            raise ValueError("training.total_env_steps must be a positive multiple of num_envs")
        self.schedule_env_steps = int(t.get("schedule_env_steps", self.total_env_steps))
        if self.schedule_env_steps <= 0:
            raise ValueError("training.schedule_env_steps must be positive")
        evaluation = self.config.get("evaluation", {})
        self.evaluation_seed_base = int(e["seed"]) + int(evaluation.get("selection_seed_offset", evaluation.get("seed_offset", 50000)))
        vector_factory = make_combat_vector_env_4v3_v12 if self.is_v12 else (make_combat_vector_env_4v3_v11 if self.is_v11 else make_combat_vector_env_4v3)
        self.envs = vector_factory(self.env_config, self.num_envs, int(t.get("num_env_workers", 0)), int(e["seed"]))
        self.actors = IndependentHAPPOActors(
            [self.obs_dim] * RED_TEAM_SIZE_4V3,
            [3] * RED_TEAM_SIZE_4V3,
            hidden_dim=int(n["hidden_dim"]),
            log_std_init=float(n["log_std_init"]),
            log_std_min=float(n["log_std_min"]),
            log_std_max=float(n["log_std_max"]),
        ).to(self.device)
        self.critic = CentralizedValueCritic(self.gs_dim, hidden_dim=int(n["hidden_dim"])).to(self.device)
        self.initial_actor_lr = float(t.get("actor_learning_rate", t.get("actor_lr")))
        self.final_actor_lr = float(t.get("actor_learning_rate_final", t.get("actor_lr_final", self.initial_actor_lr * 0.1)))
        self.initial_critic_lr = float(t.get("critic_learning_rate", t.get("critic_lr")))
        self.final_critic_lr = float(t.get("critic_learning_rate_final", t.get("critic_lr_final", self.initial_critic_lr * 0.1)))
        self.initial_entropy_coef = float(t.get("entropy_coef", 0.01))
        self.final_entropy_coef = float(t.get("entropy_coef_final", self.initial_entropy_coef * 0.1))
        self.current_actor_lr = self.initial_actor_lr
        self.current_critic_lr = self.initial_critic_lr
        self.current_entropy_coef = self.initial_entropy_coef
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.current_actor_lr) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.current_critic_lr)
        self.buffer = HAPPORolloutBuffer3v3(self.rollout_steps, self.num_envs, RED_TEAM_SIZE_4V3, self.obs_dim, 3, self.gs_dim)
        self.obs, self.global_states, self.alive_masks = self.envs.reset()
        # VectorEnv.reset() uses the constructor seed plus the global env index.
        # Keep those episode seeds alongside the live environment state so a
        # completed summary is labelled before its replacement seed is drawn.
        self.current_episode_seeds = [int(e["seed"]) + i for i in range(self.num_envs)]
        self.env_steps = 0
        self.vector_steps = 0
        self.update_count = 0
        self.effective_rollout_steps = self.rollout_steps
        self.last_agent_order: list[int] = list(range(RED_TEAM_SIZE_4V3))
        self.best_score: tuple[float, ...] | None = None
        self.best_score_fields: dict[str, float] = {}
        self.best_evaluation: dict[str, Any] | None = None
        self.best_checkpoint_name: str | None = None
        self.best_scheduled_env_steps: int | None = None
        self.best_actual_env_steps: int | None = None
        self.evaluation_history: list[dict[str, Any]] = []
        self.recent_episodes: list[dict[str, Any]] = []
        self.last_update_metrics: dict[str, float] = {}
        self.last_rollout_reward_means: dict[str, float] = {}
        self.seed_manifest: dict[str, Any] = {}
        self.next_evaluation_env_steps: int | None = None
        self.next_checkpoint_env_steps: int | None = None

    def _next_episode_seed(self) -> int:
        return int(self.episode_seed_rng.integers(0, 2**31 - 1))

    def training_signature(self) -> dict[str, Any]:
        t = self.config["training"]
        signature = {
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_4V3,
            "env_config_sha256": sha256_file(self.env_config),
            "reward_contract_sha256": _sha256_json(self.reward_contract),
            "team_size": RED_TEAM_SIZE_4V3,
            "obs_dim": self.obs_dim,
            "state_dim": self.gs_dim,
            "num_envs": int(t["num_envs"]),
            "rollout_steps": int(t["rollout_steps"]),
        }
        # Keep legacy v9 signatures loadable while requiring explicit variant
        # identity for v10 checkpoints.
        if self.experiment_variant is not None:
            signature["variant"] = str(self.experiment_variant)
        if self.reward_contract_version is not None:
            signature["reward_contract_version"] = str(self.reward_contract_version)
        if self.is_v12:
            signature["schedule_env_steps"] = int(self.schedule_env_steps)
        return signature

    @torch.no_grad()
    def _select_actions(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        red_obs = torch.as_tensor(obs[:, :RED_TEAM_SIZE_4V3, :], dtype=torch.float32, device=self.device)
        actions, log_probs = self.actors.sample_actions(red_obs)
        values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device))
        return (
            actions.detach().cpu().numpy().astype(np.float32),
            log_probs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )

    def collect_rollout(self, max_env_steps: int | None = None) -> list[dict[str, Any]]:
        if max_env_steps is None:
            steps = self.rollout_steps
        else:
            remaining = int(max_env_steps)
            if remaining < self.num_envs:
                raise ValueError("max_env_steps must allow at least one vector step")
            steps = min(self.rollout_steps, remaining // self.num_envs)
        if steps <= 0:
            raise ValueError("effective rollout must contain at least one vector step")
        if self.buffer.rollout_steps != steps:
            self.buffer = HAPPORolloutBuffer3v3(steps, self.num_envs, RED_TEAM_SIZE_4V3, self.obs_dim, 3, self.gs_dim)
        self.effective_rollout_steps = steps
        self.buffer.clear()
        episodes: list[dict[str, Any]] = []
        reward_component_sum = np.zeros(len(self.reward_keys), dtype=np.float64)
        reward_component_count = 0
        for _ in range(steps):
            actions, log_probs, values = self._select_actions(self.obs)
            result = self.envs.step(actions)
            reward_component_sum += result.red_reward_components.sum(axis=0)
            reward_component_count += self.num_envs
            self.buffer.add(
                self.obs[:, :RED_TEAM_SIZE_4V3, :],
                self.global_states,
                actions,
                log_probs,
                self.alive_masks[:, :RED_TEAM_SIZE_4V3],
                result.team_rewards,
                values,
                result.terminated | result.truncated,
            )
            self.obs, self.global_states, self.alive_masks = result.observations, result.global_states, result.alive_masks
            for i, summary in enumerate(result.episode_summaries):
                if summary is not None:
                    summary = deepcopy(summary)
                    summary["episode_seed"] = int(self.current_episode_seeds[i])
                    episodes.append(summary)
                    next_seed = self._next_episode_seed()
                    self.obs[i], self.global_states[i], self.alive_masks[i] = self.envs.reset_at(i, next_seed)
                    self.current_episode_seeds[i] = next_seed
            self.vector_steps += 1
            self.env_steps += self.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["training"]["gamma"]), float(self.config["training"]["gae_lambda"]))
        self.recent_episodes.extend(episodes)
        self.recent_episodes = self.recent_episodes[-200:]
        if reward_component_count > 0:
            self.last_rollout_reward_means = {
                f"mean_rollout_{key}": float(value)
                for key, value in zip(self.reward_keys, reward_component_sum / reward_component_count)
            }
        else:
            self.last_rollout_reward_means = {}
        return episodes

    def update(self) -> dict[str, float]:
        t = self.config["training"]
        rollout_steps = int(self.buffer.rollout_steps)
        total = rollout_steps * self.num_envs
        obs = torch.as_tensor(self.buffer.observations.reshape(total, RED_TEAM_SIZE_4V3, self.obs_dim), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(total, self.gs_dim), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(total, RED_TEAM_SIZE_4V3, 3), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(total, RED_TEAM_SIZE_4V3), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(self.buffer.agent_alive_masks.reshape(total, RED_TEAM_SIZE_4V3), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(total), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(total), dtype=torch.float32, device=self.device)

        progress = min(1.0, self.env_steps / max(1, self.schedule_env_steps if self.is_v12 else self.total_env_steps))
        self.current_actor_lr = linear_schedule(self.initial_actor_lr, self.final_actor_lr, progress)
        self.current_critic_lr = linear_schedule(self.initial_critic_lr, self.final_critic_lr, progress)
        self.current_entropy_coef = linear_schedule(self.initial_entropy_coef, self.final_entropy_coef, progress)
        for opt in self.actor_optimizers:
            for group in opt.param_groups:
                group["lr"] = self.current_actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.current_critic_lr

        factor = torch.ones_like(advantages)
        agent_order = [int(v) for v in self.rng.permutation(RED_TEAM_SIZE_4V3)]
        self.last_agent_order = agent_order
        actor_rows: list[dict[str, float | int]] = []
        critic_losses: list[float] = []
        clip_coef = float(t["clip_coef"])
        minibatch_size = int(t["minibatch_size"])
        ppo_epochs = int(t["ppo_epochs"])
        max_grad_norm = float(t["max_grad_norm"])

        for agent_id in agent_order:
            active = masks[:, agent_id] > 0.5
            if int(active.sum().detach().cpu().item()) <= 0:
                factor = factor.detach()
                actor_rows.append({"agent_id": agent_id, "active_samples": 0})
                continue
            normalized_advantages_i = normalize_advantages_for_agent(advantages, active.float())
            for _ in range(ppo_epochs):
                order = self.rng.permutation(total)
                for start in range(0, total, minibatch_size):
                    idx = torch.as_tensor(order[start:start + minibatch_size], dtype=torch.long, device=self.device)
                    idx = idx[active[idx]]
                    if len(idx) == 0:
                        continue
                    new_lp, entropy = self.actors.evaluate_agent_actions(agent_id, obs[idx, agent_id, :], actions[idx, agent_id, :])
                    log_ratio = new_lp - old_log_probs[idx, agent_id]
                    ratio = log_ratio.exp()
                    effective_adv = (factor[idx] * normalized_advantages_i[idx]).detach()
                    policy_loss = ppo_clipped_policy_loss(ratio, effective_adv, clip_coef)
                    loss = policy_loss - self.current_entropy_coef * entropy.mean()
                    opt = self.actor_optimizers[agent_id]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(self.actors.actors[agent_id].parameters(), max_grad_norm)
                    opt.step()
                    self.actors.actors[agent_id].clamp_log_std_()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
                    actor_rows.append({
                        "agent_id": agent_id,
                        "policy_loss": float(policy_loss.item()),
                        "entropy": float(entropy.mean().item()),
                        "approx_kl": float(approx_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "ratio_mean": float(ratio.mean().item()),
                        "ratio_min": float(ratio.min().item()),
                        "ratio_max": float(ratio.max().item()),
                        "factor_mean": float(factor[idx].mean().item()),
                        "factor_min": float(factor[idx].min().item()),
                        "factor_max": float(factor[idx].max().item()),
                        "actor_grad_norm": float(grad),
                        "active_samples": int(len(idx)),
                    })
            with torch.no_grad():
                new_lp_all, _ = self.actors.evaluate_agent_actions(agent_id, obs[:, agent_id, :], actions[:, agent_id, :])
                factor = happo_preceding_factor_update(factor, old_log_probs[:, agent_id], new_lp_all, active.float())

        value_preds_before = self.critic(states).detach().cpu().numpy()
        for _ in range(ppo_epochs):
            order = self.rng.permutation(total)
            for start in range(0, total, minibatch_size):
                idx = torch.as_tensor(order[start:start + minibatch_size], dtype=torch.long, device=self.device)
                values = self.critic(states[idx])
                critic_loss = 0.5 * (returns[idx] - values).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(t["value_loss_coef"]) * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach().cpu()))
        self.actors.clamp_log_std_()
        self.update_count += 1
        nonempty = [r for r in actor_rows if int(r.get("active_samples", 0)) > 0]
        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in nonempty])) if nonempty else 0.0
        log_std_dim = self.actors.effective_log_std_by_dim
        std_dim = self.actors.effective_std_by_dim
        metrics = {
            "policy_loss": mean("policy_loss"),
            "actor_loss": mean("policy_loss"),
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": mean("entropy"),
            "approx_kl": mean("approx_kl"),
            "clip_fraction": mean("clip_fraction"),
            "ratio_mean": mean("ratio_mean"),
            "ratio_min": float(np.min([float(r["ratio_min"]) for r in nonempty])) if nonempty else 1.0,
            "ratio_max": float(np.max([float(r["ratio_max"]) for r in nonempty])) if nonempty else 1.0,
            "factor_mean": mean("factor_mean"),
            "factor_min": float(np.min([float(r["factor_min"]) for r in nonempty])) if nonempty else 1.0,
            "factor_max": float(np.max([float(r["factor_max"]) for r in nonempty])) if nonempty else 1.0,
            "actor_grad_norm": mean("actor_grad_norm"),
            "actor_updates": len(nonempty),
            "agents_updated": len({int(r["agent_id"]) for r in nonempty}),
            "alive_actor_samples": int(sum(int(r.get("active_samples", 0)) for r in actor_rows)),
            "advantage_mean": float(np.mean(self.buffer.advantages)),
            "advantage_std": float(np.std(self.buffer.advantages)),
            "explained_variance": float(explained_variance(value_preds_before, self.buffer.returns.reshape(total))),
            "env_steps": float(self.env_steps),
            "vector_steps": float(self.vector_steps),
            "update_count": float(self.update_count),
            "effective_rollout_steps": float(rollout_steps),
            "total_env_steps": float(self.total_env_steps),
            "unique_alive_actor_samples": int(np.count_nonzero(np.any(self.buffer.agent_alive_masks > 0.5, axis=2))),
            "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef,
            "effective_log_std_yaw": log_std_dim[0],
            "effective_log_std_pitch": log_std_dim[1],
            "effective_log_std_speed": log_std_dim[2],
            "effective_std_yaw": std_dim[0],
            "effective_std_pitch": std_dim[1],
            "effective_std_speed": std_dim[2],
        }
        for agent_id, actor in enumerate(self.actors.actors):
            for dim, name in enumerate(("yaw", "pitch", "speed")):
                metrics[f"actor_{agent_id}_log_std_{name}"] = actor.effective_log_std_by_dim[dim]
                metrics[f"actor_{agent_id}_std_{name}"] = actor.effective_std_by_dim[dim]
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"non-finite HAPPO 4v3 update metrics: {metrics}")
        metrics.update(self.last_rollout_reward_means)
        self.last_update_metrics = metrics
        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        is_best: bool = False,
        scheduled_env_steps: int | None = None,
    ) -> None:
        ckpt = {
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_4V3,
            "variant": self.experiment_variant,
            "reward_contract_version": self.reward_contract_version,
            "training_signature": self.training_signature(),
            "config": deepcopy(self.config),
            "env_config": self.env_config,
            "reward_contract": deepcopy(self.reward_contract),
            "reward_contract_sha256": _sha256_json(self.reward_contract),
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizers": [o.state_dict() for o in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "actual_env_steps": self.env_steps,
            "scheduled_env_steps": scheduled_env_steps,
            "schedule_env_steps": self.schedule_env_steps,
            "evaluation_seed_base": self.evaluation_seed_base,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "last_agent_order": self.last_agent_order,
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "best_evaluation": self.best_evaluation,
            "best_checkpoint_name": self.best_checkpoint_name,
            "best_scheduled_env_steps": self.best_scheduled_env_steps,
            "best_actual_env_steps": self.best_actual_env_steps,
            "evaluation_history": self.evaluation_history,
            "effective_rollout_steps": self.effective_rollout_steps,
            "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef,
            "next_evaluation_env_steps": self.next_evaluation_env_steps,
            "next_checkpoint_env_steps": self.next_checkpoint_env_steps,
            "seed_manifest": deepcopy(self.seed_manifest),
            "recent_episodes": deepcopy(self.recent_episodes),
            "last_update_metrics": deepcopy(self.last_update_metrics),
            "last_rollout_reward_means": deepcopy(self.last_rollout_reward_means),
            "current_episode_seeds": list(self.current_episode_seeds),
            "observations": self.obs,
            "global_states": self.global_states,
            "alive_masks": self.alive_masks,
            "vector_env_state": self.envs.state_dict(),
            "numpy_rng_state": self.rng.bit_generator.state,
            "episode_seed_rng_state": self.episode_seed_rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "is_best": bool(is_best),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as fh:
            torch.save(ckpt, fh)
            fh.flush()
            import os
            os.fsync(fh.fileno())
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_HAPPO_4V3:
            raise ValueError("checkpoint family mismatch")
        diffs = signature_mismatches(ckpt.get("training_signature", {}), self.training_signature())
        if diffs:
            raise ValueError("training signature mismatch:\n" + "\n".join(diffs))
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        for opt, state in zip(self.actor_optimizers, ckpt["actor_optimizers"]):
            opt.load_state_dict(state)
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"])
        if self.is_v12 and int(ckpt.get("schedule_env_steps", -1)) != self.schedule_env_steps:
            raise ValueError(
                "v12 schedule_env_steps mismatch: "
                f"checkpoint={ckpt.get('schedule_env_steps')!r} current={self.schedule_env_steps!r}"
            )
        self.evaluation_seed_base = int(ckpt.get("evaluation_seed_base", self.evaluation_seed_base))
        self.vector_steps = int(ckpt["vector_steps"])
        self.update_count = int(ckpt["update_count"])
        self.last_agent_order = list(ckpt.get("last_agent_order", list(range(RED_TEAM_SIZE_4V3))))
        loaded_best = ckpt.get("best_score")
        self.best_score = tuple(loaded_best) if loaded_best is not None else None
        self.best_score_fields = dict(ckpt.get("best_score_fields", {}))
        self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.best_scheduled_env_steps = ckpt.get("best_scheduled_env_steps")
        self.best_actual_env_steps = ckpt.get("best_actual_env_steps")
        self.evaluation_history = list(ckpt.get("evaluation_history", []))
        self.effective_rollout_steps = int(ckpt.get("effective_rollout_steps", self.rollout_steps))
        self.current_actor_lr = float(ckpt.get("current_actor_lr", self.current_actor_lr))
        self.current_critic_lr = float(ckpt.get("current_critic_lr", self.current_critic_lr))
        self.current_entropy_coef = float(ckpt.get("current_entropy_coef", self.current_entropy_coef))
        self.next_evaluation_env_steps = ckpt.get("next_evaluation_env_steps")
        self.next_checkpoint_env_steps = ckpt.get("next_checkpoint_env_steps")
        self.seed_manifest = deepcopy(ckpt.get("seed_manifest", {}))
        self.recent_episodes = list(ckpt.get("recent_episodes", []))
        self.last_update_metrics = dict(ckpt.get("last_update_metrics", {}))
        self.last_rollout_reward_means = dict(ckpt.get("last_rollout_reward_means", {}))
        loaded_episode_seeds = ckpt.get("current_episode_seeds")
        if loaded_episode_seeds is None:
            loaded_episode_seeds = [int(self.config["experiment"]["seed"]) + i for i in range(self.num_envs)]
        if len(loaded_episode_seeds) != self.num_envs:
            raise ValueError("checkpoint current_episode_seeds length does not match num_envs")
        self.current_episode_seeds = [int(seed) for seed in loaded_episode_seeds]
        if "vector_env_state" in ckpt:
            self.envs.load_state_dict(ckpt["vector_env_state"])
            self.obs = np.asarray(ckpt["observations"], dtype=np.float32)
            self.global_states = np.asarray(ckpt["global_states"], dtype=np.float32)
            self.alive_masks = np.asarray(ckpt["alive_masks"], dtype=np.float32)
        else:
            self.obs, self.global_states, self.alive_masks = self.envs.reset()
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        if "episode_seed_rng_state" in ckpt:
            self.episode_seed_rng.bit_generator.state = ckpt["episode_seed_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu() if hasattr(ckpt["torch_cpu_rng_state"], "cpu") else ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            _restore_cuda_rng_state(ckpt["torch_cuda_rng_state"])

    def write_summary(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "device": str(self.device),
            "variant": self.experiment_variant,
            "reward_contract_version": self.reward_contract_version,
            "schedule_env_steps": self.schedule_env_steps,
            "last_update_metrics": self.last_update_metrics,
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "best_checkpoint_name": self.best_checkpoint_name,
            "best_evaluation": self.best_evaluation,
            "best_scheduled_env_steps": self.best_scheduled_env_steps,
            "best_actual_env_steps": self.best_actual_env_steps,
            "final_actual_env_steps": self.env_steps,
            "evaluation_seed_base": self.evaluation_seed_base,
            "reward_contract": deepcopy(self.reward_contract),
            "reward_contract_sha256": _sha256_json(self.reward_contract),
            "seed_manifest": deepcopy(self.seed_manifest),
            "next_evaluation_env_steps": self.next_evaluation_env_steps,
            "next_checkpoint_env_steps": self.next_checkpoint_env_steps,
            "final_evaluation": self.evaluation_history[-1]["summary"] if self.evaluation_history else None,
            "evaluation_history": self.evaluation_history,
            "best_score_schema": best_score_fields_4v3(self.is_v11, self.is_v12),
            "policy_modes": self.envs.policy_modes(),
        }
        (out / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self.envs.close()


__all__ = [
    "CHECKPOINT_FAMILY_HAPPO_4V3",
    "HAPPO4v3Trainer",
    "best_score_fields_4v3",
    "compute_best_score_4v3",
    "summarize_4v3_episodes",
]
