"""Formation head-on 3v3 scenario – offset slots to avoid head-on collisions."""
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState

RED_IDS = ("red_0", "red_1", "red_2")
BLUE_IDS = ("blue_0", "blue_1", "blue_2")
ALL_IDS = RED_IDS + BLUE_IDS


class Homogeneous3v3Scenario:
    """Symmetric formation_head_on with offset lateral slots.

    Red base heading = 0, blue base heading = pi.
    Red slots are shifted by -opposing_lateral_offset/2,
    blue slots by +opposing_lateral_offset/2.
    Reverse slot pairing: red_i matches blue_(2-i) for symmetric
    speed/altitude/heading jitter.
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
        offset = float(settings.get("opposing_lateral_offset", 200.0))
        half_span = lateral_spacing * (team_size - 1) / 2.0

        red_centre = np.array([-separation / 2.0, 0.0])
        blue_centre = np.array([separation / 2.0, 0.0])

        # base_slots = [-lateral_spacing, 0, lateral_spacing]
        base_slots = np.linspace(-half_span, half_span, team_size)
        red_slots = base_slots - offset / 2.0
        blue_slots = base_slots + offset / 2.0

        # Reverse pairing: red_0<->blue_2, red_1<->blue_1, red_2<->blue_0
        blue_pair_idx = [2, 1, 0]  # blue_i paired with red_{blue_pair_idx[i]}

        # Per-pair shared jitter (3 pairs, based on red index)
        pair_speed = np.zeros(team_size, dtype=float)
        pair_altitude = np.zeros(team_size, dtype=float)
        pair_hdg_jitter = np.zeros(team_size, dtype=float)
        for i in range(team_size):
            pair_speed[i] = float(np.clip(
                settings["speed_center"] + rng.uniform(-settings["speed_jitter"], settings["speed_jitter"]),
                self.spec.v_min, self.spec.v_max))
            pair_altitude[i] = float(np.clip(
                settings["altitude_center"] + rng.uniform(-settings["altitude_jitter"], settings["altitude_jitter"]),
                battlefield["altitude_min"], battlefield["altitude_max"]))
            pair_hdg_jitter[i] = float(rng.uniform(-settings["heading_jitter"], settings["heading_jitter"]))

        rotation = float(rng.uniform(-np.pi, np.pi))
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

        aircraft_list: list[Aircraft] = []
        for i in range(team_size):
            # red_i: slot i, pair i
            rpos = red_centre + np.array([0.0, red_slots[i]])
            rxy = matrix @ rpos
            rhdg = wrap_angle(0.0 + pair_hdg_jitter[i] + rotation)
            rstate = AircraftState(float(rxy[0]), float(rxy[1]), -pair_altitude[i],
                                   pair_speed[i], 0.0, rhdg)
            aircraft_list.append(Aircraft(RED_IDS[i], "red", self.spec, rstate))

            # blue_i: slot i, paired with red_{blue_pair_idx[i]}
            bi = blue_pair_idx[i]
            bpos = blue_centre + np.array([0.0, blue_slots[i]])
            bxy = matrix @ bpos
            # blue_i heading = red_(paired) heading + pi
            bhdg = wrap_angle(np.pi + pair_hdg_jitter[bi] + rotation)
            # Position symmetry: blue_pos = -red_pos (for paired aircraft)
            # red center = (-sep/2, red_slots[i]), blue center = (+sep/2, blue_slots[i])
            # paired: red at (-sep/2, red_slots[bi]), blue at (+sep/2, blue_slots[i])
            # For symmetry: blue should be at -red_pos when rotation=0:
            # red_pos[bi] = (-sep/2, red_slots[bi])
            # -red_pos[bi] = (+sep/2, -red_slots[bi])
            # blue_pos[i] = (+sep/2, blue_slots[i])
            # For symmetry: blue_slots[i] = -red_slots[bi]
            # With our setup: red_slots[bi] = base_slots[bi] - offset/2
            # blue_slots[i] = base_slots[i] + offset/2
            # For i=1,bi=1: base_slots[1]=0, red_slots[1]=-offset/2, blue_slots[1]=+offset/2 ✓
            # For i=0,bi=2: base_slots[0]=-400, red_slots[2]=+400-offset/2=+300, blue_slots[0]=-400+offset/2=-300 ✓
            # For i=2,bi=0: base_slots[2]=+400, red_slots[0]=-400-offset/2=-500, blue_slots[2]=+400+offset/2=+500 ✓
            bstate = AircraftState(float(bxy[0]), float(bxy[1]), -pair_altitude[bi],
                                   pair_speed[bi], 0.0, bhdg)
            aircraft_list.append(Aircraft(BLUE_IDS[i], "blue", self.spec, bstate))

        return aircraft_list
