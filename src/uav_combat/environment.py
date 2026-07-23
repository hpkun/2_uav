"""不依赖 Gymnasium 的简化同构 1v1 空战环境。"""
from pathlib import Path
from typing import Any

import numpy as np

from .combat import SimplifiedAttackModel, situation_score
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .geometry import PairwiseGeometry, compute_pairwise_geometry
from .integrator import RK4Integrator
from .models import Aircraft, ControlCommand, TargetCommand
from .scenario import HomogeneousScenario


class HomogeneousAirCombatEnv:
    """同步推进并以确定性几何攻击结算的 1v1 环境。"""

    def __init__(self, config_path: str | Path = "configs/homogeneous_1v1.yaml") -> None:
        self.config = load_config(config_path)
        simulation, action, combat = self.config["simulation"], self.config["action"], self.config["combat"]
        self.scenario = HomogeneousScenario(self.config)
        self.dynamics = PointMassDynamics(simulation["gravity"])
        self.integrator = RK4Integrator(simulation["dt"])
        self.controller = TargetStateController(**action, gravity=simulation["gravity"])
        self.attack_model = SimplifiedAttackModel(
            combat["attack_distance_min"], combat["attack_distance_max"], combat["attack_ata_max"], combat["attack_aa_max"]
        )
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False

    def reset(self, seed: int | None = None, scenario_name: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """重置确定性场景，并恢复可运行状态。"""
        self.aircraft = self.scenario.reset(seed, scenario_name)
        self.step_count = 0
        self._running = True
        geometries = self._geometries()
        scores = self._situation_scores()
        info = {
            "step_count": 0,
            "scenario_name": self.scenario.scenario_name,
            "targets": {},
            "controls": {},
            "termination_reason": None,
            "outcome": None,
            "attacks": {"red_0": False, "blue_0": False},
            "geometries": geometries,
            "situation_scores": scores,
            "reward_terms": {aircraft_id: {"dense": 0.0, "terminal": 0.0} for aircraft_id in scores},
        }
        return self._observations(geometries), info

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        """同步积分、同时攻击判定并返回观测、奖励和终止信息。"""
        if not self._running:
            raise RuntimeError("reset() must be called before step(), and after an episode ends")
        alive = [aircraft for aircraft in self.aircraft if aircraft.state.alive]
        missing = {aircraft.aircraft_id for aircraft in alive} - actions.keys()
        if missing:
            raise KeyError(f"missing actions for: {sorted(missing)}")

        targets: dict[str, TargetCommand] = {}
        controls: dict[str, ControlCommand] = {}
        old_states = {aircraft.aircraft_id: aircraft.state.copy() for aircraft in alive}
        for aircraft in alive:
            target, control = self.controller.control_from_action(
                old_states[aircraft.aircraft_id], actions[aircraft.aircraft_id], aircraft.spec
            )
            targets[aircraft.aircraft_id], controls[aircraft.aircraft_id] = target, control
        new_states = {
            aircraft.aircraft_id: self.integrator.step(
                old_states[aircraft.aircraft_id], controls[aircraft.aircraft_id], self.dynamics, aircraft.spec
            )
            for aircraft in alive
        }
        for aircraft in alive:
            aircraft.state = new_states[aircraft.aircraft_id]
        self.step_count += 1

        reason, outcome = self._physical_termination()
        attacks = {"red_0": False, "blue_0": False}
        if reason is None:
            red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
            attacks = {
                "red_0": self.attack_model.can_attack(red.state, blue.state),
                "blue_0": self.attack_model.can_attack(blue.state, red.state),
            }
            if attacks["red_0"] and attacks["blue_0"]:
                red.state.alive = blue.state.alive = False
                reason, outcome = "mutual_kill", "draw"
            elif attacks["red_0"]:
                blue.state.alive = False
                reason, outcome = "red_kill", "red"
            elif attacks["blue_0"]:
                red.state.alive = False
                reason, outcome = "blue_kill", "blue"

        terminated = reason is not None
        truncated = not terminated and self.step_count >= self.config["simulation"]["max_steps"]
        if truncated:
            reason, outcome = "max_steps", "draw"
        scores = self._situation_scores()
        rewards, reward_terms = self._rewards(scores, reason, outcome)
        geometries = self._geometries()
        if terminated or truncated:
            self._running = False
        info = {
            "step_count": self.step_count,
            "scenario_name": self.scenario.scenario_name,
            "targets": targets,
            "controls": controls,
            "termination_reason": reason,
            "outcome": outcome,
            "attacks": attacks,
            "geometries": geometries,
            "situation_scores": scores,
            "reward_terms": reward_terms,
        }
        return self._observations(geometries), rewards, terminated, truncated, info

    def _aircraft_by_id(self, aircraft_id: str) -> Aircraft:
        return next(aircraft for aircraft in self.aircraft if aircraft.aircraft_id == aircraft_id)

    def _geometries(self) -> dict[str, PairwiseGeometry]:
        red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
        return {
            "red_0": compute_pairwise_geometry(red.state, blue.state),
            "blue_0": compute_pairwise_geometry(blue.state, red.state),
        }

    def _situation_scores(self) -> dict[str, float]:
        red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
        combat = self.config["combat"]
        return {
            "red_0": situation_score(red.state, blue.state, combat["preferred_distance"], combat["distance_scale"]),
            "blue_0": situation_score(blue.state, red.state, combat["preferred_distance"], combat["distance_scale"]),
        }

    def _rewards(self, scores: dict[str, float], reason: str | None, outcome: str | None) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        combat = self.config["combat"]
        red_dense = combat["situation_reward_scale"] * (scores["red_0"] - scores["blue_0"])
        terminal = {"red_0": 0.0, "blue_0": 0.0}
        if outcome == "red":
            terminal = {"red_0": combat["terminal_reward"], "blue_0": -combat["terminal_reward"]}
        elif outcome == "blue":
            terminal = {"red_0": -combat["terminal_reward"], "blue_0": combat["terminal_reward"]}
        elif reason in {"mutual_kill", "collision"}:
            terminal = {"red_0": -combat["terminal_reward"], "blue_0": -combat["terminal_reward"]}
        dense = {"red_0": float(red_dense), "blue_0": float(-red_dense)}
        rewards = {aircraft_id: dense[aircraft_id] + terminal[aircraft_id] for aircraft_id in dense}
        terms = {aircraft_id: {"dense": dense[aircraft_id], "terminal": terminal[aircraft_id]} for aircraft_id in dense}
        return rewards, terms

    def _observations(self, geometries: dict[str, PairwiseGeometry] | None = None) -> dict[str, np.ndarray]:
        geometries = geometries or self._geometries()
        battlefield = self.config["battlefield"]
        relative_x_scale = 2.0 * battlefield["x_limit"]
        relative_y_scale = 2.0 * battlefield["y_limit"]
        relative_z_scale = battlefield["altitude_max"] - battlefield["altitude_min"]
        distance_scale = float(np.sqrt(relative_x_scale ** 2 + relative_y_scale ** 2 + relative_z_scale ** 2))
        observations: dict[str, np.ndarray] = {}
        for own in self.aircraft:
            geometry = geometries[own.aircraft_id]
            pitch_scale = max(abs(own.spec.theta_min), abs(own.spec.theta_max))
            speed_normalized = 2.0 * (own.state.v - own.spec.v_min) / (own.spec.v_max - own.spec.v_min) - 1.0
            observation = np.concatenate((
                [speed_normalized, own.state.theta / pitch_scale, np.sin(own.state.psi), np.cos(own.state.psi)],
                geometry.relative_position / np.array([relative_x_scale, relative_y_scale, relative_z_scale]),
                geometry.relative_velocity / (2.0 * own.spec.v_max),
                [geometry.distance / distance_scale, geometry.ata / np.pi, geometry.aa / np.pi],
            ))
            observations[own.aircraft_id] = np.clip(observation, -1.0, 1.0).astype(float)
        return observations

    def _physical_termination(self) -> tuple[str | None, str | None]:
        limits = self.config["battlefield"]
        violations: dict[str, str] = {}
        for aircraft in self.aircraft:
            if not limits["altitude_min"] <= aircraft.state.altitude <= limits["altitude_max"]:
                violations[aircraft.aircraft_id] = "altitude_boundary"
            elif abs(aircraft.state.x) > limits["x_limit"] or abs(aircraft.state.y) > limits["y_limit"]:
                violations[aircraft.aircraft_id] = "xy_boundary"
        if violations:
            for aircraft_id in violations:
                self._aircraft_by_id(aircraft_id).state.alive = False
            outcome = "draw" if len(violations) == 2 else ("blue" if "red_0" in violations else "red")
            reasons = set(violations.values())
            return (next(iter(reasons)) if len(reasons) == 1 else "boundary"), outcome
        red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
        distance = np.linalg.norm(red.state.as_array()[:3] - blue.state.as_array()[:3])
        if distance <= limits["collision_distance"]:
            red.state.alive = blue.state.alive = False
            return "collision", "draw"
        return None, None
