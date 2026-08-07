"""Configuration and mirrored scenario for the v12 soft-boundary contract."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from .config import aircraft_spec
from .models import Aircraft, AircraftState
from .scenario_4v3_v11 import (
    ALL_IDS_V11,
    BLUE_IDS_V11,
    FIXED_ROLES_V11,
    RED_COMBAT_IDS_V11,
    RED_IDS_V11,
    FunctionalHeterogeneous4v3V11Scenario,
)

RED_IDS_V12 = RED_IDS_V11
RED_COMBAT_IDS_V12 = RED_COMBAT_IDS_V11
BLUE_IDS_V12 = BLUE_IDS_V11
ALL_IDS_V12 = ALL_IDS_V11
FIXED_ROLES_V12 = deepcopy(FIXED_ROLES_V11)

REWARD_COMPONENT_KEYS_V12 = (
    "mission_outcome_reward",
    "blue_kill_event_reward",
    "red_combat_loss_event_penalty",
    "support_loss_event_penalty",
    "boundary_event_penalty",
    "combat_geometry_progress_reward",
    "combat_lock_progress_reward",
    "combat_half_lock_event_reward",
    "support_unique_detection_reward",
    "support_cue_to_direct_reward",
    "support_cue_to_half_lock_reward",
    "support_assisted_kill_reward",
    "support_formation_progress_reward",
    "total_dense_reward",
    "team_total_reward",
)

_PROFILE_KEYS_V12 = (
    "sensor_range", "v_min", "v_max", "theta_min", "theta_max", "yaw_rate_max",
    "pitch_rate_max", "acceleration_max", "lock_distance_min",
    "lock_distance_optimal_max", "lock_distance_fade_max", "lock_ata_fade_max",
    "lock_aa_fade_max", "lock_increment_scale", "lock_decay_per_step",
    "lock_kill_threshold", "target_min_hold_steps", "target_lost_release_steps",
    "target_switch_distance_ratio",
)

_BOUNDARY_KEYS_V12 = (
    "mode", "horizontal_soft_margin", "altitude_soft_margin", "max_recovery_blend",
    "hard_horizontal_buffer", "hard_altitude_buffer", "recovery_heading_error_scale",
    "recovery_pitch_error_scale", "red_hard_contact_penalty",
)


def resolved_reward_contract_v12(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v12_config(config: dict[str, Any]) -> None:
    scenario = config.get("scenario", {})
    if int(scenario.get("red_team_size", -1)) != 4 or int(scenario.get("blue_team_size", -1)) != 3:
        raise ValueError("v12 requires a 4v3 scenario")
    combat = config.get("combat", {})
    if combat.get("reward_mode") != "functional_heterogeneous_4v3_team_v12":
        raise ValueError("v12 requires combat.reward_mode=functional_heterogeneous_4v3_team_v12")
    if combat.get("reward_contract_version") != "v12_soft_boundary_combat_aligned":
        raise ValueError("v12 requires reward_contract_version=v12_soft_boundary_combat_aligned")
    roles = config.get("heterogeneous", {}).get("roles", {})
    if roles != FIXED_ROLES_V12:
        raise ValueError(f"v12 requires fixed roles {FIXED_ROLES_V12}")
    heterogeneous = config.get("heterogeneous", {})
    if heterogeneous.get("can_attack", {}).get("support") is not False:
        raise ValueError("v12 Support must be unarmed")
    if heterogeneous.get("can_attack", {}).get("combat") is not True:
        raise ValueError("v12 Combat must be armed")
    if not heterogeneous.get("information_sharing", {}).get("red_support_to_red_combat", False):
        raise ValueError("v12 requires support-to-combat cue sharing")

    profile = config.get("combat_profile")
    if not isinstance(profile, dict):
        raise KeyError("v12 requires one shared combat_profile")
    missing = [key for key in _PROFILE_KEYS_V12 if key not in profile]
    if missing:
        raise KeyError(f"missing combat_profile fields: {', '.join(missing)}")
    for key in _PROFILE_KEYS_V12:
        if not math.isfinite(float(profile[key])):
            raise ValueError(f"combat_profile.{key} must be finite")
    for key in ("v_min", "v_max", "theta_min", "theta_max", "yaw_rate_max", "pitch_rate_max", "acceleration_max"):
        if not np.isclose(float(config["aircraft"][key]), float(profile[key]), rtol=1e-9, atol=1e-9):
            raise ValueError(f"aircraft.{key} must match combat_profile.{key}")
    if not (0.0 < float(profile["lock_distance_min"]) < float(profile["lock_distance_optimal_max"]) < float(profile["lock_distance_fade_max"])):
        raise ValueError("v12 lock distance ranges must be strictly increasing")
    if float(profile["lock_increment_scale"]) != 0.17 or float(profile["lock_decay_per_step"]) != 0.03:
        raise ValueError("v12 lock scales must match the frozen v11 profile")
    if float(profile["lock_kill_threshold"]) != 1.0:
        raise ValueError("v12 lock_kill_threshold must be 1.0")
    if int(profile["target_min_hold_steps"]) != 30 or int(profile["target_lost_release_steps"]) != 10:
        raise ValueError("v12 target hold/release steps must match v11")
    if float(profile["target_switch_distance_ratio"]) != 0.70:
        raise ValueError("v12 target switch ratio must match v11")

    boundary = config.get("boundary")
    if not isinstance(boundary, dict):
        raise KeyError("missing v12 boundary contract")
    missing = [key for key in _BOUNDARY_KEYS_V12 if key not in boundary]
    if missing:
        raise KeyError(f"missing boundary fields: {', '.join(missing)}")
    if boundary["mode"] != "soft_containment":
        raise ValueError("v12 boundary.mode must be soft_containment")
    for key in _BOUNDARY_KEYS_V12[1:]:
        if not math.isfinite(float(boundary[key])):
            raise ValueError(f"boundary.{key} must be finite")
    if float(boundary["horizontal_soft_margin"]) <= 0.0 or float(boundary["altitude_soft_margin"]) <= 0.0:
        raise ValueError("v12 soft margins must be positive")
    if not 0.0 <= float(boundary["max_recovery_blend"]) <= 1.0:
        raise ValueError("v12 max_recovery_blend must be in [0, 1]")
    if float(boundary["hard_horizontal_buffer"]) < 0.0 or float(boundary["hard_altitude_buffer"]) < 0.0:
        raise ValueError("v12 hard buffers must be non-negative")

    rewards = config.get("rewards")
    if not isinstance(rewards, dict):
        raise KeyError("missing v12 rewards contract")
    sections = {
        "mission": ("red_full_elimination", "red_total_loss", "timeout_red_win", "timeout_red_loss", "timeout_draw", "mutual_elimination_draw"),
        "events": ("blue_combat_killed", "red_combat_killed", "red_support_killed", "red_boundary_hard_contact"),
        "combat_progress": ("geometry_scale", "lock_scale", "half_lock_event"),
        "support_events": ("unique_detection", "cue_to_direct", "cue_to_half_lock", "assisted_kill"),
        "support_formation": ("progress_scale",),
        "dense_clip": ("min", "max"),
    }
    for section, keys in sections.items():
        values = rewards.get(section)
        if not isinstance(values, dict):
            raise KeyError(f"missing rewards.{section}")
        for key in keys:
            if key not in values or not math.isfinite(float(values[key])):
                raise ValueError(f"missing or non-finite rewards.{section}.{key}")
    if float(rewards["dense_clip"]["min"]) >= float(rewards["dense_clip"]["max"]):
        raise ValueError("v12 dense clip min must be less than max")
    if float(heterogeneous.get("sensor_range", {}).get("red_support", 0.0)) != 6000.0:
        raise ValueError("v12 support sensor range must be 6000m")


class FunctionalHeterogeneous4v3V12Scenario(FunctionalHeterogeneous4v3V11Scenario):
    """Reuse the frozen v11 mirrored placement without importing v11 validation."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v12_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "ALL_IDS_V12", "BLUE_IDS_V12", "FIXED_ROLES_V12", "RED_COMBAT_IDS_V12", "RED_IDS_V12",
    "REWARD_COMPONENT_KEYS_V12", "FunctionalHeterogeneous4v3V12Scenario",
    "resolved_reward_contract_v12", "validate_heterogeneous_4v3_v12_config",
]
