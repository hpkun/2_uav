"""Formation head-on 3v3 scenario – truly head-on with paired symmetric init."""
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState

RED_IDS = ("red_0", "red_1", "red_2")
BLUE_IDS = ("blue_0", "blue_1", "blue_2")
ALL_IDS = RED_IDS + BLUE_IDS


class Homogeneous3v3Scenario:
    """Generates a symmetric formation_head_on 3v3 initial condition.

    Red base heading = 0, blue base heading = pi.  After global rotation the
    two headings differ by exactly pi.  Per-slot speed, altitude, and heading
    jitter are sampled once and shared by the paired red_i / blue_i aircraft.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = aircraft_spec(config)
        self.scenario_name = "formation_head_on"
        ts = int(config["scenario"]["team_size"])
        if ts != 3:
            raise ValueError(f"Homogeneous3v3Scenario only supports team_size=3, got {ts}")

    def reset(self, seed: int | None = None) -> list[Aircraft]:
        rng = np.random.default_rng(seed)
        settings = self.config["scenario"]
        battlefield = self.config["battlefield"]
        team_size = 3

        separation = float(rng.uniform(settings["separation_min"], settings["separation_max"]))
        lateral_spacing = float(settings["lateral_spacing"])
        half_span = lateral_spacing * (team_size - 1) / 2.0

        red_centre = np.array([-separation / 2.0, 0.0])
        blue_centre = np.array([separation / 2.0, 0.0])
        slot_offsets = np.linspace(-half_span, half_span, team_size)

        # --- per-slot paired jitter (one sample per slot, shared by red_i and blue_i) ---
        slot_speed = np.zeros(team_size, dtype=float)
        slot_altitude = np.zeros(team_size, dtype=float)
        slot_hdg_jitter = np.zeros(team_size, dtype=float)
        for i in range(team_size):
            slot_speed[i] = float(np.clip(
                settings["speed_center"] + rng.uniform(-settings["speed_jitter"], settings["speed_jitter"]),
                self.spec.v_min, self.spec.v_max))
            slot_altitude[i] = float(np.clip(
                settings["altitude_center"] + rng.uniform(-settings["altitude_jitter"], settings["altitude_jitter"]),
                battlefield["altitude_min"], battlefield["altitude_max"]))
            slot_hdg_jitter[i] = float(rng.uniform(-settings["heading_jitter"], settings["heading_jitter"]))

        # --- global rotation ---
        rotation = float(rng.uniform(-np.pi, np.pi))
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

        aircraft_list: list[Aircraft] = []
        for i in range(team_size):
            # red_i
            rpos = red_centre + np.array([0.0, slot_offsets[i]])
            rxy = matrix @ rpos
            rhdg = wrap_angle(0.0 + slot_hdg_jitter[i] + rotation)
            rstate = AircraftState(float(rxy[0]), float(rxy[1]), -slot_altitude[i],
                                   slot_speed[i], 0.0, rhdg)
            aircraft_list.append(Aircraft(RED_IDS[i], "red", self.spec, rstate))

            # blue_i – heading differs by exactly pi from red_i
            bpos = blue_centre + np.array([0.0, slot_offsets[i]])
            bxy = matrix @ bpos
            bhdg = wrap_angle(np.pi + slot_hdg_jitter[i] + rotation)
            bstate = AircraftState(float(bxy[0]), float(bxy[1]), -slot_altitude[i],
                                   slot_speed[i], 0.0, bhdg)
            aircraft_list.append(Aircraft(BLUE_IDS[i], "blue", self.spec, bstate))

        return aircraft_list
