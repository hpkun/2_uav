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
from .math_utils import angle_difference
from .models import Aircraft, ControlCommand, TargetCommand
from .scenario import HomogeneousScenario
from .rewards import coupled_difference_rewards, crdrl_coupled_reward, madsac_segmented_reward


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

    def reset(self, seed: int | None = None, scenario_name: str | None = None, rear_team: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """重置确定性场景，并恢复可运行状态。"""
        self.aircraft = self.scenario.reset(seed, scenario_name, rear_team)
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
            "reward_terms": {aircraft_id: self._empty_reward_terms() for aircraft_id in scores},
            "control_diagnostics": {},
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
        control_diagnostics: dict[str, dict[str, float | bool]] = {}
        old_states = {aircraft.aircraft_id: aircraft.state.copy() for aircraft in alive}
        for aircraft in alive:
            target, control = self.controller.control_from_action(
                old_states[aircraft.aircraft_id], actions[aircraft.aircraft_id], aircraft.spec
            )
            targets[aircraft.aircraft_id], controls[aircraft.aircraft_id] = target, control
            diagnostics = self.controller.diagnostics(
                old_states[aircraft.aircraft_id], target, control, aircraft.spec,
                actions[aircraft.aircraft_id],
            )
            derivatives = self.dynamics.derivatives(old_states[aircraft.aircraft_id], control)
            actual_acceleration, actual_pitch_rate, actual_yaw_rate = map(float, derivatives[3:6])
            diagnostics.update({
                "actual_acceleration": actual_acceleration,
                "actual_pitch_rate": actual_pitch_rate,
                "actual_yaw_rate": actual_yaw_rate,
                "acceleration_tracking_error": diagnostics["clipped_acceleration"] - actual_acceleration,
                "pitch_rate_tracking_error": diagnostics["clipped_pitch_rate"] - actual_pitch_rate,
                "yaw_rate_tracking_error": diagnostics["clipped_yaw_rate"] - actual_yaw_rate,
            })
            diagnostics.update({
                "acceleration_tracking_absolute_error": abs(diagnostics["acceleration_tracking_error"]),
                "pitch_rate_tracking_absolute_error": abs(diagnostics["pitch_rate_tracking_error"]),
                "yaw_rate_tracking_absolute_error": abs(diagnostics["yaw_rate_tracking_error"]),
            })
            clipped_action = np.clip(np.asarray(actions[aircraft.aircraft_id], dtype=float), -1.0, 1.0)
            diagnostics.update({"action_yaw": float(clipped_action[0]), "action_pitch": float(clipped_action[1]), "action_speed": float(clipped_action[2]), "delta_yaw": float(angle_difference(target.desired_psi, old_states[aircraft.aircraft_id].psi)), "delta_pitch": float(target.desired_theta-old_states[aircraft.aircraft_id].theta), "delta_speed": float(target.desired_v-old_states[aircraft.aircraft_id].v)})
            numeric_values = [
                value for value in diagnostics.values()
                if not isinstance(value, (bool, np.bool_, str))
            ]
            if not np.all(np.isfinite(numeric_values)):
                raise FloatingPointError("non-finite control diagnostics")
            control_diagnostics[aircraft.aircraft_id] = diagnostics
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
            "control_diagnostics": control_diagnostics,
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
        mode = combat.get("reward_mode", "coupled_difference")
        if mode == "madsac_segmented":
            red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
            terms = {"red_0": madsac_segmented_reward(red.state, blue.state, "red", reason, outcome), "blue_0": madsac_segmented_reward(blue.state, red.state, "blue", reason, outcome)}
            return {key: value["reward_total"] for key, value in terms.items()}, terms
        if mode == "crdrl_coupled":
            red, blue = self._aircraft_by_id("red_0"), self._aircraft_by_id("blue_0")
            terms = {"red_0": crdrl_coupled_reward(red.state, blue.state, "red", reason, outcome, combat.get("crdrl_dense_scale", 1.0), combat.get("crdrl_sparse_scale", 1.0)), "blue_0": crdrl_coupled_reward(blue.state, red.state, "blue", reason, outcome, combat.get("crdrl_dense_scale", 1.0), combat.get("crdrl_sparse_scale", 1.0))}
            return {key: value["reward_total"] for key, value in terms.items()}, terms
        if mode != "coupled_difference":
            raise ValueError(f"unknown reward_mode: {mode}")
        old_terms = coupled_difference_rewards(scores["red_0"], scores["blue_0"], combat["situation_reward_scale"], combat["terminal_reward"], reason, outcome)
        terms = {f"{team}_0": values for team, values in old_terms.items()}
        return {key: value["dense"] + value["terminal"] for key, value in terms.items()}, terms

    def _empty_reward_terms(self) -> dict[str, float]:
        mode = self.config["combat"].get("reward_mode", "coupled_difference")
        if mode == "madsac_segmented":
            return {key: 0.0 for key in ("reward_terminal", "reward_boundary", "reward_guide", "reward_position", "reward_threat", "reward_total")}
        if mode == "crdrl_coupled":
            return {key: 0.0 for key in ("reward_coupled_dense_raw", "reward_coupled_dense", "reward_sparse", "reward_terminal", "reward_boundary", "reward_total")}
        return {"dense": 0.0, "terminal": 0.0}

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
            altitude_normalized = 2.0 * (own.state.altitude - battlefield["altitude_min"]) / relative_z_scale - 1.0
            cosine, sine = np.cos(own.state.psi), np.sin(own.state.psi)
            dx, dy, dz = geometry.relative_position
            dvx, dvy, dvz = geometry.relative_velocity
            relative_ego = np.array([cosine * dx + sine * dy, -sine * dx + cosine * dy, -dz])
            velocity_ego = np.array([cosine * dvx + sine * dvy, -sine * dvx + cosine * dvy, -dvz])
            observation = np.concatenate((
                [speed_normalized, own.state.theta / pitch_scale, altitude_normalized],
                relative_ego / np.array([relative_x_scale, relative_y_scale, relative_z_scale]),
                velocity_ego / (2.0 * own.spec.v_max),
                [geometry.distance / distance_scale, geometry.yaw_error / np.pi, geometry.pitch_error / (np.pi / 2.0), geometry.ata / np.pi, geometry.aa / np.pi],
            ))
            observations[own.aircraft_id] = np.clip(observation, -1.0, 1.0).astype(float)
        return observations

    def global_state(self, perspective_team: str) -> np.ndarray:
        """Return the normalized absolute state ordered own then opponent."""
        if perspective_team not in {"red", "blue"}:
            raise ValueError("perspective_team must be red or blue")
        battlefield = self.config["battlefield"]
        altitude_span = battlefield["altitude_max"] - battlefield["altitude_min"]
        values: list[float] = []
        opponent_team = "blue" if perspective_team == "red" else "red"
        for aircraft_id in (f"{perspective_team}_0", f"{opponent_team}_0"):
            aircraft = self._aircraft_by_id(aircraft_id)
            state, spec = aircraft.state, aircraft.spec
            altitude = 2.0 * (state.altitude - battlefield["altitude_min"]) / altitude_span - 1.0
            speed = 2.0 * (state.v - spec.v_min) / (spec.v_max - spec.v_min) - 1.0
            pitch = state.theta / max(abs(spec.theta_min), abs(spec.theta_max))
            values.extend([state.x / battlefield["x_limit"], state.y / battlefield["y_limit"], altitude, speed, pitch, np.sin(state.psi), np.cos(state.psi)])
        result = np.clip(np.asarray(values, dtype=float), -1.0, 1.0)
        if not np.all(np.isfinite(result)):
            raise ValueError("global state must be finite")
        return result

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
