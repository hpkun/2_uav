"""不依赖 Gymnasium 的同构空战基础环境。"""
from pathlib import Path
from typing import Any
import numpy as np
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .models import Aircraft, ControlCommand, TargetCommand
from .scenario import HomogeneousScenario


class HomogeneousAirCombatEnv:
    """同步推进所有飞行实体的 1v1 基础动力学环境。"""
    def __init__(self, config_path: str | Path = "configs/homogeneous_1v1.yaml") -> None:
        self.config = load_config(config_path)
        simulation, action = self.config["simulation"], self.config["action"]
        self.scenario = HomogeneousScenario(self.config)
        self.dynamics = PointMassDynamics(simulation["gravity"])
        self.integrator = RK4Integrator(simulation["dt"])
        self.controller = TargetStateController(**action, gravity=simulation["gravity"])
        self.aircraft: list[Aircraft] = []
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """重置场景并返回观测与基础信息。"""
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        return self._observations(), {"step_count": 0, "targets": {}, "controls": {}, "termination_reason": None}

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        """基于同一批旧状态计算控制，再同步积分全部实体。"""
        alive = [a for a in self.aircraft if a.state.alive]
        missing = {a.aircraft_id for a in alive} - actions.keys()
        if missing:
            raise KeyError(f"missing actions for: {sorted(missing)}")
        targets: dict[str, TargetCommand] = {}
        controls: dict[str, ControlCommand] = {}
        old_states = {a.aircraft_id: a.state.copy() for a in alive}
        for aircraft in alive:
            target, control = self.controller.control_from_action(old_states[aircraft.aircraft_id], actions[aircraft.aircraft_id], aircraft.spec)
            targets[aircraft.aircraft_id], controls[aircraft.aircraft_id] = target, control
        new_states = {
            a.aircraft_id: self.integrator.step(old_states[a.aircraft_id], controls[a.aircraft_id], self.dynamics, a.spec)
            for a in alive
        }
        for aircraft in alive:
            aircraft.state = new_states[aircraft.aircraft_id]
        self.step_count += 1
        reason = self._termination_reason()
        terminated = reason is not None
        truncated = self.step_count >= self.config["simulation"]["max_steps"]
        if truncated and reason is None:
            reason = "max_steps"
        # 基础动力学阶段的占位返回值，不可用于强化学习训练。
        rewards = {a.aircraft_id: 0.0 for a in self.aircraft}
        info = {"step_count": self.step_count, "targets": targets, "controls": controls, "termination_reason": reason}
        return self._observations(), rewards, terminated, truncated, info

    def _observations(self) -> dict[str, np.ndarray]:
        observations = {}
        for own in self.aircraft:
            opponents = [a for a in self.aircraft if a.team != own.team]
            if not opponents:
                continue
            target = opponents[0]
            relative_position = target.state.as_array()[:3] - own.state.as_array()[:3]
            relative_velocity = target.state.velocity_vector() - own.state.velocity_vector()
            distance = float(np.linalg.norm(relative_position))
            observations[own.aircraft_id] = np.concatenate(([own.state.v, own.state.theta, own.state.psi], relative_position, relative_velocity, [distance]))
        return observations

    def _termination_reason(self) -> str | None:
        limits = self.config["battlefield"]
        for aircraft in self.aircraft:
            state = aircraft.state
            if not limits["altitude_min"] <= state.altitude <= limits["altitude_max"]:
                return "altitude_boundary"
            if abs(state.x) > limits["x_limit"] or abs(state.y) > limits["y_limit"]:
                return "xy_boundary"
        for i, first in enumerate(self.aircraft):
            for second in self.aircraft[i + 1:]:
                if np.linalg.norm(first.state.as_array()[:3] - second.state.as_array()[:3]) <= limits["collision_distance"]:
                    return "collision"
        return None

