"""Simple mirrored 4v3 target-lock scenario used by the v11 contract."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState

RED_IDS_V11 = ("red_0", "red_1", "red_2", "red_3")
RED_COMBAT_IDS_V11 = ("red_1", "red_2", "red_3")
BLUE_IDS_V11 = ("blue_0", "blue_1", "blue_2")
ALL_IDS_V11 = RED_IDS_V11 + BLUE_IDS_V11

FIXED_ROLES_V11 = {
    "red_0": "support",
    "red_1": "combat",
    "red_2": "combat",
    "red_3": "combat",
    "blue_0": "combat",
    "blue_1": "combat",
    "blue_2": "combat",
}

REWARD_COMPONENT_KEYS_V11 = (
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


def resolved_reward_contract_v11(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v11_config(config: dict[str, Any]) -> None:
    scenario = config.get("scenario", {})
    if int(scenario.get("red_team_size", -1)) != 4 or int(scenario.get("blue_team_size", -1)) != 3:
        raise ValueError("v11 requires a 4v3 scenario")
    combat = config.get("combat", {})
    if combat.get("reward_mode") != "functional_heterogeneous_4v3_team_v11":
        raise ValueError("v11 requires combat.reward_mode=functional_heterogeneous_4v3_team_v11")
    if combat.get("reward_contract_version") != "v11_target_lock_support_cue":
        raise ValueError("v11 requires reward_contract_version=v11_target_lock_support_cue")

    roles = config.get("heterogeneous", {}).get("roles", {})
    if roles != FIXED_ROLES_V11:
        raise ValueError(f"v11 requires fixed roles {FIXED_ROLES_V11}")
    can_attack = config.get("heterogeneous", {}).get("can_attack", {})
    if can_attack.get("support") is not False or can_attack.get("combat") is not True:
        raise ValueError("v11 requires an unarmed support and armed combat aircraft")
    if not config.get("heterogeneous", {}).get("information_sharing", {}).get("red_support_to_red_combat", False):
        raise ValueError("v11 requires support-to-combat cue sharing")

    profile = config.get("combat_profile")
    if not isinstance(profile, dict):
        raise KeyError("v11 requires one shared combat_profile")
    required_profile = (
        "sensor_range", "v_min", "v_max", "theta_min", "theta_max", "yaw_rate_max",
        "pitch_rate_max", "acceleration_max", "lock_distance_min",
        "lock_distance_optimal_max", "lock_distance_fade_max", "lock_ata_fade_max",
        "lock_aa_fade_max", "lock_increment_scale", "lock_decay_per_step",
        "lock_kill_threshold", "target_min_hold_steps", "target_lost_release_steps",
        "target_switch_distance_ratio",
    )
    missing = [key for key in required_profile if key not in profile]
    if missing:
        raise KeyError(f"missing combat_profile fields: {', '.join(missing)}")
    if float(profile["sensor_range"]) <= 0.0:
        raise ValueError("combat_profile.sensor_range must be positive")
    for key in required_profile:
        if not math.isfinite(float(profile[key])):
            raise ValueError(f"combat_profile.{key} must be finite")
    for key in ("v_min", "v_max", "theta_min", "theta_max", "yaw_rate_max", "pitch_rate_max", "acceleration_max"):
        if not np.isclose(float(config["aircraft"][key]), float(profile[key]), rtol=1e-9, atol=1e-9):
            raise ValueError(f"aircraft.{key} must match combat_profile.{key} for v11 combat symmetry")
    if not (0.0 < float(profile["lock_distance_min"]) < float(profile["lock_distance_optimal_max"]) < float(profile["lock_distance_fade_max"])):
        raise ValueError("v11 lock distance ranges must be strictly increasing")
    if not (0.0 < float(profile["lock_increment_scale"]) and 0.0 < float(profile["lock_decay_per_step"])):
        raise ValueError("v11 lock increment and decay must be positive")
    if float(profile["lock_kill_threshold"]) != 1.0:
        raise ValueError("v11 lock_kill_threshold must be 1.0")
    if int(profile["target_min_hold_steps"]) <= 0 or int(profile["target_lost_release_steps"]) <= 0:
        raise ValueError("v11 target hold/release steps must be positive")
    if not 0.0 < float(profile["target_switch_distance_ratio"]) < 1.0:
        raise ValueError("v11 target_switch_distance_ratio must be in (0, 1)")

    rewards = config.get("rewards")
    if not isinstance(rewards, dict):
        raise KeyError("missing v11 rewards contract")
    sections = {
        "mission": ("red_full_elimination", "red_total_loss", "timeout_red_win", "timeout_red_loss", "timeout_draw"),
        "events": ("blue_combat_killed", "red_combat_killed", "red_support_killed", "red_boundary_loss"),
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
        raise ValueError("v11 dense clip min must be less than max")
    if float(rewards["support_formation"]["progress_scale"]) > 0.002:
        raise ValueError("v11 support formation progress scale is too large")

    support_range = float(config.get("heterogeneous", {}).get("sensor_range", {}).get("red_support", 0.0))
    if support_range != 6000.0:
        raise ValueError("v11 support sensor range must be 6000m")


class FunctionalHeterogeneous4v3V11Scenario:
    """Mirrored head-on formations with paired speed, altitude and heading jitter."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v11_config(config)
        self.config = config
        self.spec = aircraft_spec(config)

    def _aircraft(self, aircraft_id: str, team: str, state: AircraftState) -> Aircraft:
        role = FIXED_ROLES_V11[aircraft_id]
        sensor_range = (
            float(self.config["heterogeneous"]["sensor_range"]["red_support"])
            if role == "support"
            else float(self.config["combat_profile"]["sensor_range"])
        )
        return Aircraft(
            aircraft_id=aircraft_id,
            team=team,
            spec=self.spec,
            state=state,
            role=role,
            sensor_range=sensor_range,
            can_attack=role == "combat",
        )

    def reset(self, seed: int | None = None) -> list[Aircraft]:
        rng = np.random.default_rng(seed)
        settings = self.config["scenario"]
        battlefield = self.config["battlefield"]
        separation = float(rng.uniform(settings["separation_min"], settings["separation_max"]))
        spacing = float(settings["lateral_spacing"])
        altitude_center = float(settings["altitude_center"])
        altitude_jitter = float(settings["altitude_jitter"])
        speed_center = float(settings["speed_center"])
        speed_jitter = float(settings["speed_jitter"])
        heading_jitter = float(settings["heading_jitter"])
        rotation = float(rng.uniform(-np.pi, np.pi))
        rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        red_center = np.array([-separation / 2.0, 0.0])
        blue_center = np.array([separation / 2.0, 0.0])
        slots = (-spacing, 0.0, spacing)
        pairs = [
            (
                float(np.clip(speed_center + rng.uniform(-speed_jitter, speed_jitter), self.spec.v_min, self.spec.v_max)),
                float(np.clip(altitude_center + rng.uniform(-altitude_jitter, altitude_jitter), battlefield["altitude_min"], battlefield["altitude_max"])),
                float(rng.uniform(-heading_jitter, heading_jitter)),
            )
            for _ in range(3)
        ]

        def state(center: np.ndarray, lateral: float, speed: float, altitude: float, heading: float) -> AircraftState:
            xy = rotation_matrix @ (center + np.array([0.0, lateral]))
            return AircraftState(float(xy[0]), float(xy[1]), -altitude, speed, 0.0, wrap_angle(heading))

        trailing = float(settings["support_trailing_distance"])
        support_xy = rotation_matrix @ (red_center + np.array([-trailing, 0.0]))
        support = AircraftState(
            float(support_xy[0]), float(support_xy[1]),
            -float(np.clip(altitude_center + rng.uniform(-altitude_jitter, altitude_jitter), battlefield["altitude_min"], battlefield["altitude_max"])),
            float(np.clip(speed_center + rng.uniform(-speed_jitter, speed_jitter), self.spec.v_min, self.spec.v_max)),
            0.0,
            wrap_angle(rotation),
        )
        aircraft = [self._aircraft("red_0", "red", support)]
        for index, aid in enumerate(RED_COMBAT_IDS_V11):
            speed, altitude, heading_jitter_value = pairs[index]
            aircraft.append(self._aircraft(aid, "red", state(red_center, slots[index], speed, altitude, rotation + heading_jitter_value)))
        for index, aid in enumerate(BLUE_IDS_V11):
            speed, altitude, heading_jitter_value = pairs[index]
            aircraft.append(self._aircraft(aid, "blue", state(blue_center, slots[index], speed, altitude, np.pi + rotation - heading_jitter_value)))
        return aircraft


__all__ = [
    "ALL_IDS_V11", "BLUE_IDS_V11", "FIXED_ROLES_V11", "RED_COMBAT_IDS_V11", "RED_IDS_V11",
    "REWARD_COMPONENT_KEYS_V11", "FunctionalHeterogeneous4v3V11Scenario",
    "resolved_reward_contract_v11", "validate_heterogeneous_4v3_v11_config",
]
