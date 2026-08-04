"""Functional heterogeneous red 4v3 main-experiment scenario."""
from __future__ import annotations

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
