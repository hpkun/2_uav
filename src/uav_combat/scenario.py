"""同构 1v1 固定与随机场景创建逻辑。"""
from typing import Any

import numpy as np

from .config import aircraft_spec
from .math_utils import wrap_angle
from .models import Aircraft, AircraftState


class HomogeneousScenario:
    """生成三个简化随机模板或原始固定场景。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = aircraft_spec(config)
        self.aircraft: list[Aircraft] = []
        self.scenario_name = "fixed"

    def reset(self, seed: int | None = None, scenario_name: str | None = None) -> list[Aircraft]:
        """按种子与模板生成可复现的 red_0 和 blue_0。"""
        rng = np.random.default_rng(seed)
        templates = self.config["scenario"]["templates"]
        chosen = str(rng.choice(templates)) if scenario_name is None else scenario_name
        if chosen == "fixed":
            states = self._fixed_states()
        elif chosen in templates:
            states = self._random_states(chosen, rng)
        else:
            raise ValueError(f"unknown scenario_name: {chosen}")
        self.scenario_name = chosen
        self.aircraft = [Aircraft(f"{team}_0", team, self.spec, states[team]) for team in ("red", "blue")]
        return self.aircraft

    def _fixed_states(self) -> dict[str, AircraftState]:
        states = {}
        for team in ("red", "blue"):
            item = self.config["initial_state"][team]
            states[team] = AircraftState(item["x"], item["y"], -item["altitude"], item["v"], item["theta"], item["psi"])
        return states

    def _random_states(self, name: str, rng: np.random.Generator) -> dict[str, AircraftState]:
        settings = self.config["scenario"]
        separation = float(rng.uniform(settings["separation_min"], settings["separation_max"]))
        if name == "tail_chase":
            positions = {"red": np.array([-separation / 2, 0.0]), "blue": np.array([separation / 2, 0.0])}
            headings = {"red": 0.0, "blue": 0.0}
            if settings["randomize_roles"] and bool(rng.integers(0, 2)):
                positions["red"], positions["blue"] = positions["blue"], positions["red"]
        elif name == "offset_head_on":
            offset = settings["lateral_offset"] / 2.0
            positions = {"red": np.array([-separation / 2, offset]), "blue": np.array([separation / 2, -offset])}
            headings = {"red": 0.0, "blue": np.pi}
        else:  # crossing
            leg = separation / np.sqrt(2.0)
            positions = {"red": np.array([-leg, 0.0]), "blue": np.array([0.0, -leg])}
            headings = {"red": 0.0, "blue": np.pi / 2.0}

        rotation = float(rng.uniform(-np.pi, np.pi))
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        states: dict[str, AircraftState] = {}
        battlefield = self.config["battlefield"]
        for team in ("red", "blue"):
            xy = matrix @ positions[team]
            altitude = float(np.clip(
                settings["altitude_center"] + rng.uniform(-settings["altitude_jitter"], settings["altitude_jitter"]),
                battlefield["altitude_min"], battlefield["altitude_max"],
            ))
            speed = float(np.clip(
                settings["speed_center"] + rng.uniform(-settings["speed_jitter"], settings["speed_jitter"]),
                self.spec.v_min, self.spec.v_max,
            ))
            heading = wrap_angle(headings[team] + rotation + rng.uniform(-settings["heading_jitter"], settings["heading_jitter"]))
            states[team] = AircraftState(float(xy[0]), float(xy[1]), -altitude, speed, 0.0, heading)
        return states
