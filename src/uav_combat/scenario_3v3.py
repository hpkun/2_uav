"""Formation head-on 3v3 scenario for homogeneous air combat."""
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState

RED_IDS = ("red_0", "red_1", "red_2")
BLUE_IDS = ("blue_0", "blue_1", "blue_2")
ALL_IDS = RED_IDS + BLUE_IDS


class Homogeneous3v3Scenario:
    """Generates a formation_head_on 3v3 initial condition."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = aircraft_spec(config)
        self.scenario_name = "formation_head_on"

    def reset(self, seed: int | None = None) -> list[Aircraft]:
        """Seed-reproducible formation_head_on with global rotation."""
        rng = np.random.default_rng(seed)
        settings = self.config["scenario"]
        team_size = int(settings["team_size"])

        separation = float(rng.uniform(settings["separation_min"], settings["separation_max"]))
        lateral_spacing = float(settings["lateral_spacing"])
        half_span = lateral_spacing * (team_size - 1) / 2.0

        # Red team: centre at (-separation/2, 0), heading right (+x)
        # Blue team: centre at (+separation/2, 0), heading left (-x)
        red_centre = np.array([-separation / 2.0, 0.0])
        blue_centre = np.array([separation / 2.0, 0.0])

        # Lateral slot offsets
        slot_offsets = np.linspace(-half_span, half_span, team_size)

        red_headings = {f"red_{i}": 0.0 for i in range(team_size)}
        blue_headings = {f"blue_{i}": 0.0 for i in range(team_size)}

        # Assemble per-aircraft positions before rotation
        raw_positions: dict[str, np.ndarray] = {}
        raw_headings: dict[str, float] = {}
        for i in range(team_size):
            raw_positions[f"red_{i}"] = red_centre + np.array([0.0, slot_offsets[i]])
            raw_positions[f"blue_{i}"] = blue_centre + np.array([0.0, slot_offsets[i]])
            raw_headings[f"red_{i}"] = red_headings[f"red_{i}"]
            raw_headings[f"blue_{i}"] = blue_headings[f"blue_{i}"]

        # Global random rotation
        rotation = float(rng.uniform(-np.pi, np.pi))
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

        battlefield = self.config["battlefield"]
        aircraft_list: list[Aircraft] = []
        for team, ids in (("red", RED_IDS), ("blue", BLUE_IDS)):
            for idx, aircraft_id in enumerate(ids):
                xy = matrix @ raw_positions[aircraft_id]
                heading = wrap_angle(raw_headings[aircraft_id] + rotation + rng.uniform(-settings["heading_jitter"], settings["heading_jitter"]))
                altitude = float(np.clip(
                    settings["altitude_center"] + rng.uniform(-settings["altitude_jitter"], settings["altitude_jitter"]),
                    battlefield["altitude_min"], battlefield["altitude_max"],
                ))
                speed = float(np.clip(
                    settings["speed_center"] + rng.uniform(-settings["speed_jitter"], settings["speed_jitter"]),
                    self.spec.v_min, self.spec.v_max,
                ))
                state = AircraftState(float(xy[0]), float(xy[1]), -altitude, speed, 0.0, heading)
                aircraft_list.append(Aircraft(aircraft_id, team, self.spec, state))

        return aircraft_list
