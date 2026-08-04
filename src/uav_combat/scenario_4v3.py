"""Functional heterogeneous red 4v3 main-experiment scenario."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState

RED_IDS_4V3 = ("red_0", "red_1", "red_2", "red_3")
RED_COMBAT_IDS_4V3 = ("red_1", "red_2", "red_3")
BLUE_IDS_4V3 = ("blue_0", "blue_1", "blue_2")
ALL_IDS_4V3 = RED_IDS_4V3 + BLUE_IDS_4V3

FIXED_ROLES_4V3 = {
    "red_0": "support",
    "red_1": "combat",
    "red_2": "combat",
    "red_3": "combat",
    "blue_0": "combat",
    "blue_1": "combat",
    "blue_2": "combat",
}

REWARD_CONTRACT_REQUIRED_4V3 = {
    "mission": (
        "red_complete_elimination_success",
        "red_all_combat_eliminated",
        "timeout",
        "mutual_combat_elimination",
        "blue_noncombat_elimination",
    ),
    "events": (
        "blue_combat_attack_kill",
        "red_combat_attack_loss",
        "red_support_attack_loss",
        "red_boundary_loss",
        "support_assisted_kill",
    ),
    "combat_dense": (
        "approach_scale",
        "approach_distance_normalizer",
        "readiness_scale",
        "threat_scale",
        "boundary_scale",
        "readiness_fade_distance",
    ),
    "support_dense": (
        "coverage_scale",
        "position_scale",
        "threat_scale",
        "boundary_scale",
    ),
    "dense_clip": ("min", "max"),
    "support_credit": ("assisted_window_steps",),
    "boundary": ("soft_margin",),
}


def resolved_reward_contract_4v3(config: dict[str, Any]) -> dict[str, Any]:
    """Return the validated, resolved reward contract without mutable aliases."""
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_config(config: dict[str, Any]) -> None:
    scenario = config.get("scenario", {})
    if int(scenario.get("red_team_size", -1)) != 4 or int(scenario.get("blue_team_size", -1)) != 3:
        raise ValueError("heterogeneous 4v3 v9 requires scenario.red_team_size=4 and blue_team_size=3")
    if config.get("combat", {}).get("reward_mode") != "functional_heterogeneous_4v3_team_v9":
        raise ValueError("4v3 v9 requires combat.reward_mode=functional_heterogeneous_4v3_team_v9")

    hetero = config.get("heterogeneous", {})
    roles = hetero.get("roles", {})
    if roles != FIXED_ROLES_4V3:
        raise ValueError(f"4v3 v9 requires fixed roles {FIXED_ROLES_4V3}")

    sensor_range = hetero.get("sensor_range", {})
    can_attack = hetero.get("can_attack", {})
    required_sensor = ("red_support", "red_combat", "blue_combat")
    for key in required_sensor:
        if key not in sensor_range:
            raise KeyError(f"missing heterogeneous.sensor_range.{key}")
        if float(sensor_range[key]) <= 0.0:
            raise ValueError(f"heterogeneous.sensor_range.{key} must be positive")
    if can_attack.get("support") is not False or can_attack.get("combat") is not True:
        raise ValueError("4v3 v9 requires support can_attack=false and combat can_attack=true")
    if not bool(hetero.get("information_sharing", {}).get("red_support_to_red_combat", False)):
        raise ValueError("4v3 v9 requires red support-to-combat information sharing enabled")

    formation = config.get("support_formation")
    if not isinstance(formation, dict):
        raise KeyError("missing support_formation configuration")
    required_formation = (
        "initial_trailing_distance",
        "rule_hold_distance",
        "reward_optimal_min",
        "reward_optimal_max",
        "reward_fade_near",
        "reward_fade_far",
        "rear_alignment_threshold",
        "direction_validity_threshold",
    )
    missing = [key for key in required_formation if key not in formation]
    if missing:
        raise KeyError(f"missing support_formation fields: {', '.join(missing)}")
    initial_trailing = float(formation["initial_trailing_distance"])
    rule_hold = float(formation["rule_hold_distance"])
    fade_near = float(formation["reward_fade_near"])
    optimal_min = float(formation["reward_optimal_min"])
    optimal_max = float(formation["reward_optimal_max"])
    fade_far = float(formation["reward_fade_far"])
    if initial_trailing <= 0.0 or rule_hold <= 0.0:
        raise ValueError("support_formation initial_trailing_distance and rule_hold_distance must be positive")
    if not (0.0 < fade_near <= optimal_min <= optimal_max <= fade_far):
        raise ValueError(
            "support_formation distances must satisfy "
            "0 < reward_fade_near <= reward_optimal_min <= reward_optimal_max <= reward_fade_far"
        )
    direction_threshold = float(formation["direction_validity_threshold"])
    if not math.isfinite(direction_threshold) or direction_threshold < 0.0:
        raise ValueError("support_formation.direction_validity_threshold must be finite and non-negative")
    if not (0.0 <= float(formation["rear_alignment_threshold"]) <= 1.0):
        raise ValueError("support_formation.rear_alignment_threshold must be in [0, 1]")

    rewards = config.get("rewards")
    if not isinstance(rewards, dict):
        raise KeyError("missing rewards contract")
    for section, required in REWARD_CONTRACT_REQUIRED_4V3.items():
        values = rewards.get(section)
        if not isinstance(values, dict):
            raise KeyError(f"missing rewards.{section} contract section")
        missing_reward = [key for key in required if key not in values]
        if missing_reward:
            raise KeyError(f"missing rewards.{section} fields: {', '.join(missing_reward)}")
        for key in required:
            value = float(values[key])
            if not math.isfinite(value):
                raise ValueError(f"rewards.{section}.{key} must be finite")
    mission = rewards["mission"]
    events = rewards["events"]
    dense_clip = rewards["dense_clip"]
    if mission["red_complete_elimination_success"] <= 0.0:
        raise ValueError("red_complete_elimination_success mission reward must be positive")
    for key in ("red_all_combat_eliminated", "timeout", "mutual_combat_elimination", "blue_noncombat_elimination"):
        if float(mission[key]) >= 0.0:
            raise ValueError(f"rewards.mission.{key} must be negative")
    if float(events["blue_combat_attack_kill"]) <= 0.0 or float(events["support_assisted_kill"]) <= 0.0:
        raise ValueError("positive event rewards must be positive")
    for key in ("red_combat_attack_loss", "red_support_attack_loss", "red_boundary_loss"):
        if float(events[key]) >= 0.0:
            raise ValueError(f"rewards.events.{key} must be negative")
    for section in ("combat_dense", "support_dense"):
        for key, value in rewards[section].items():
            if key != "readiness_fade_distance" and float(value) <= 0.0:
                raise ValueError(f"rewards.{section}.{key} must be positive")
    if float(rewards["combat_dense"]["readiness_fade_distance"]) <= 0.0:
        raise ValueError("rewards.combat_dense.readiness_fade_distance must be positive")
    if float(dense_clip["min"]) >= float(dense_clip["max"]):
        raise ValueError("rewards.dense_clip.min must be less than max")
    if int(rewards["support_credit"]["assisted_window_steps"]) <= 0:
        raise ValueError("rewards.support_credit.assisted_window_steps must be positive")
    if float(rewards["boundary"]["soft_margin"]) <= 0.0:
        raise ValueError("rewards.boundary.soft_margin must be positive")


class FunctionalHeterogeneous4v3Scenario:
    """Seven-aircraft head-on scenario: red support + 3 combat vs 3 blue combat."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_config(config)
        self.config = config
        self.spec = aircraft_spec(config)

    def _sensor_range(self, aircraft_id: str) -> float:
        sr = self.config["heterogeneous"]["sensor_range"]
        if aircraft_id == "red_0":
            return float(sr["red_support"])
        if aircraft_id.startswith("red_"):
            return float(sr["red_combat"])
        return float(sr["blue_combat"])

    def _aircraft(self, aircraft_id: str, team: str, state: AircraftState) -> Aircraft:
        role = FIXED_ROLES_4V3[aircraft_id]
        return Aircraft(
            aircraft_id=aircraft_id,
            team=team,
            spec=self.spec,
            state=state,
            role=role,
            sensor_range=self._sensor_range(aircraft_id),
            can_attack=(role == "combat"),
        )

    def reset(self, seed: int | None = None) -> list[Aircraft]:
        rng = np.random.default_rng(seed)
        settings = self.config["scenario"]
        battlefield = self.config["battlefield"]

        separation = float(rng.uniform(settings["separation_min"], settings["separation_max"]))
        lateral_spacing = float(settings["lateral_spacing"])
        formation = self.config.get("support_formation", {})
        support_trailing = float(formation["initial_trailing_distance"])
        altitude_center = float(settings["altitude_center"])
        altitude_jitter = float(settings["altitude_jitter"])
        speed_center = float(settings["speed_center"])
        speed_jitter = float(settings["speed_jitter"])
        heading_jitter = float(settings["heading_jitter"])

        red_center = np.array([-separation / 2.0, 0.0])
        blue_center = np.array([separation / 2.0, 0.0])
        red_combat_slots = np.array([-lateral_spacing, 0.0, lateral_spacing])
        blue_slots = np.array([lateral_spacing, 0.0, -lateral_spacing])

        pair_speed = []
        pair_altitude = []
        pair_heading = []
        for _ in range(3):
            pair_speed.append(float(np.clip(speed_center + rng.uniform(-speed_jitter, speed_jitter),
                                            self.spec.v_min, self.spec.v_max)))
            pair_altitude.append(float(np.clip(altitude_center + rng.uniform(-altitude_jitter, altitude_jitter),
                                               battlefield["altitude_min"], battlefield["altitude_max"])))
            pair_heading.append(float(rng.uniform(-heading_jitter, heading_jitter)))

        support_altitude = float(np.clip(altitude_center + rng.uniform(-altitude_jitter, altitude_jitter),
                                         battlefield["altitude_min"], battlefield["altitude_max"]))
        support_speed = float(np.clip(speed_center + rng.uniform(-speed_jitter, speed_jitter),
                                      self.spec.v_min, self.spec.v_max))

        rotation = float(rng.uniform(-np.pi, np.pi))
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

        aircraft: list[Aircraft] = []

        support_xy = matrix @ (red_center + np.array([-support_trailing, 0.0]))
        support_state = AircraftState(
            float(support_xy[0]), float(support_xy[1]), -support_altitude,
            support_speed, 0.0, wrap_angle(rotation)
        )
        aircraft.append(self._aircraft("red_0", "red", support_state))

        for i, aid in enumerate(RED_COMBAT_IDS_4V3):
            xy = matrix @ (red_center + np.array([0.0, red_combat_slots[i]]))
            state = AircraftState(
                float(xy[0]), float(xy[1]), -pair_altitude[i],
                pair_speed[i], 0.0, wrap_angle(rotation + pair_heading[i])
            )
            aircraft.append(self._aircraft(aid, "red", state))

        for i, aid in enumerate(BLUE_IDS_4V3):
            xy = matrix @ (blue_center + np.array([0.0, blue_slots[i]]))
            state = AircraftState(
                float(xy[0]), float(xy[1]), -pair_altitude[i],
                pair_speed[i], 0.0, wrap_angle(np.pi + rotation + pair_heading[i])
            )
            aircraft.append(self._aircraft(aid, "blue", state))

        return aircraft
