"""v12 4v3 environment with symmetric soft boundary containment.

The v12 environment deliberately reuses the v11 observation, target, cue,
lock, and point-mass mechanics while owning the transition, reward, terminal,
and reporting contract.  This keeps the historical v11 behavior unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .environment_4v3_v11 import (
    BLUE_TEAM_SIZE_V11,
    GS_DIM_V11,
    OBS_DIM_V11,
    RED_TEAM_SIZE_V11,
    DEATH_LOCK_V11,
    combat_potential_v11,
    lock_quality_v11,
    reward_group_totals_v11,
    FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv,
)
from .integrator import RK4Integrator
from .models import Aircraft, AircraftState
from .scenario_4v3_v12 import (
    ALL_IDS_V12,
    BLUE_IDS_V12,
    RED_COMBAT_IDS_V12,
    RED_IDS_V12,
    REWARD_COMPONENT_KEYS_V12,
    FunctionalHeterogeneous4v3V12Scenario,
    resolved_reward_contract_v12,
    validate_heterogeneous_4v3_v12_config,
)
from .math_utils import angle_difference

OBS_DIM_V12 = OBS_DIM_V11
GS_DIM_V12 = GS_DIM_V11
RED_TEAM_SIZE_V12 = RED_TEAM_SIZE_V11
BLUE_TEAM_SIZE_V12 = BLUE_TEAM_SIZE_V11
DEATH_NONE_V12 = 0
DEATH_LOCK_V12 = DEATH_LOCK_V11

COMBAT_EVENT_REWARD_KEYS_V12 = (
    "blue_kill_event_reward",
    "red_combat_loss_event_penalty",
)
SUPPORT_EVENT_REWARD_KEYS_V12 = (
    "support_loss_event_penalty",
    "support_unique_detection_reward",
    "support_cue_to_direct_reward",
    "support_assisted_kill_reward",
)
HALF_LOCK_REWARD_KEYS_V12 = (
    "combat_half_lock_event_reward",
    "support_cue_to_half_lock_reward",
)


def reward_group_totals_v12(components: dict[str, float]) -> dict[str, float]:
    """Return the mutually exclusive v12 reward groups."""
    return {
        "mission": float(components.get("mission_outcome_reward", 0.0)),
        "combat_evt": float(sum(components.get(key, 0.0) for key in COMBAT_EVENT_REWARD_KEYS_V12)),
        "support_evt": float(sum(components.get(key, 0.0) for key in SUPPORT_EVENT_REWARD_KEYS_V12)),
        "half_lock_evt": float(sum(components.get(key, 0.0) for key in HALF_LOCK_REWARD_KEYS_V12)),
        "boundary_evt": float(components.get("boundary_event_penalty", 0.0)),
        "dense": float(components.get("total_dense_reward", 0.0)),
    }


def aggregate_team_reward_v12(components: dict[str, float]) -> float:
    groups = reward_group_totals_v12(components)
    return float(sum(groups.values()))


class FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(
    FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv
):
    """Red support plus three red Combat aircraft against three blue Combat aircraft."""

    variant = "functional_heterogeneous_4v3_v12_soft_boundary_combat_aligned"
    reward_contract_version = "v12_soft_boundary_combat_aligned"

    def __init__(
        self,
        config_path: str | Path = "configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml",
    ) -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        validate_heterogeneous_4v3_v12_config(self.config)
        self.reward_contract = resolved_reward_contract_v12(self.config)
        self.profile = self.config["combat_profile"]
        simulation = self.config["simulation"]
        self.scenario = FunctionalHeterogeneous4v3V12Scenario(self.config)
        self.dynamics = PointMassDynamics(float(simulation.get("gravity", 9.81)))
        self.integrator = RK4Integrator(float(simulation["dt"]))
        self.controller = TargetStateController(
            **self.config["action"], gravity=float(simulation.get("gravity", 9.81))
        )
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False
        self._death_causes: dict[str, int] = {}
        self._attack_kills = {"red": 0, "blue": 0}
        self._kill_steps = {"red": [], "blue": []}
        self.targets: dict[str, str | None] = {}
        self.target_hold_steps: dict[str, int] = {}
        self.target_lost_steps: dict[str, int] = {}
        self.target_switch_count = 0
        self.lock_progress: dict[str, float] = {}
        self.max_lock_progress = 0.0
        self._half_lock_pairs: set[tuple[str, str]] = set()
        self._support_seen: set[str] = set()
        self._cue_pairs: set[tuple[str, str]] = set()
        self._cue_last_step: dict[tuple[str, str], int] = {}
        self._cue_to_direct_pairs: set[tuple[str, str]] = set()
        self._cue_to_half_pairs: set[tuple[str, str]] = set()
        self._support_cues: dict[str, str | None] = {cid: None for cid in RED_COMBAT_IDS_V12}
        self._last_formation_score = 0.0
        self._episode_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
        self._last_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
        self._last_raw_dense_reward = 0.0
        self._episode_metrics: dict[str, float] = {}
        self._episode_return = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # v11 reset only establishes shared mechanics; all v12 counters below
        # are owned by this contract and are reset explicitly.
        observations = super().reset(seed)
        self._episode_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
        self._last_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
        self._episode_metrics.update({
            "red_boundary_soft_recovery_steps": 0.0,
            "blue_boundary_soft_recovery_steps": 0.0,
            "support_boundary_soft_recovery_steps": 0.0,
            "red_boundary_hard_contacts": 0.0,
            "blue_boundary_hard_contacts": 0.0,
            "support_boundary_hard_contacts": 0.0,
            "red_boundary_soft_recovery_steps_step": 0.0,
            "blue_boundary_soft_recovery_steps_step": 0.0,
            "support_boundary_soft_recovery_steps_step": 0.0,
            "red_boundary_hard_contacts_step": 0.0,
            "blue_boundary_hard_contacts_step": 0.0,
            "support_boundary_hard_contacts_step": 0.0,
        })
        return observations

    def _recovery_pitch(self, state: AircraftState) -> float:
        boundary = self.config["boundary"]
        altitude_error = float(self.config["scenario"]["altitude_center"]) - state.altitude
        horizontal_distance = max(float(np.hypot(state.x, state.y)), float(boundary["horizontal_soft_margin"]))
        desired = float(np.arctan2(altitude_error, horizontal_distance))
        return float(np.clip(desired, self.profile["theta_min"], self.profile["theta_max"]))

    def _recovery_heading(self, state: AircraftState) -> float:
        if abs(state.x) + abs(state.y) <= 1e-9:
            return float(state.psi)
        return float(np.arctan2(-state.y, -state.x))

    def _boundary_recovery_action(
        self, aircraft: Aircraft, action: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Blend only yaw/pitch toward the common battlefield center."""
        boundary = self.config["boundary"]
        battlefield = self.config["battlefield"]
        horizontal_distance = min(
            float(battlefield["x_limit"]) - abs(aircraft.state.x),
            float(battlefield["y_limit"]) - abs(aircraft.state.y),
        )
        horizontal_risk = np.clip(
            (float(boundary["horizontal_soft_margin"]) - horizontal_distance)
            / float(boundary["horizontal_soft_margin"]),
            0.0,
            1.0,
        )
        altitude_distance = min(
            aircraft.state.altitude - float(battlefield["altitude_min"]),
            float(battlefield["altitude_max"]) - aircraft.state.altitude,
        )
        altitude_risk = np.clip(
            (float(boundary["altitude_soft_margin"]) - altitude_distance)
            / float(boundary["altitude_soft_margin"]),
            0.0,
            1.0,
        )
        strength = float(max(horizontal_risk, altitude_risk))
        blend = float(np.clip(float(boundary["max_recovery_blend"]) * strength, 0.0, 1.0))
        corrected = np.asarray(action, dtype=np.float32).copy()
        if blend <= 0.0:
            return corrected, 0.0
        desired_heading = self._recovery_heading(aircraft.state)
        recovery_yaw = np.clip(
            angle_difference(desired_heading, aircraft.state.psi)
            / float(boundary["recovery_heading_error_scale"]),
            -1.0,
            1.0,
        )
        recovery_pitch = np.clip(
            (self._recovery_pitch(aircraft.state) - aircraft.state.theta)
            / float(boundary["recovery_pitch_error_scale"]),
            -1.0,
            1.0,
        )
        corrected[0] = (1.0 - blend) * corrected[0] + blend * recovery_yaw
        corrected[1] = (1.0 - blend) * corrected[1] + blend * recovery_pitch
        # corrected[2] intentionally remains the actor/rule speed action.
        return corrected, blend

    def _boundary_team_key(self, aircraft: Aircraft) -> str:
        if aircraft.aircraft_id == "red_0":
            return "support"
        return "red" if aircraft.team == "red" else "blue"

    def _project_hard_boundary(self, aircraft: Aircraft) -> bool:
        boundary = self.config["boundary"]
        battlefield = self.config["battlefield"]
        x_limit = float(battlefield["x_limit"])
        y_limit = float(battlefield["y_limit"])
        altitude_min = float(battlefield["altitude_min"])
        altitude_max = float(battlefield["altitude_max"])
        outside = (
            abs(aircraft.state.x) > x_limit
            or abs(aircraft.state.y) > y_limit
            or aircraft.state.altitude < altitude_min
            or aircraft.state.altitude > altitude_max
        )
        if not outside:
            return False
        horizontal_buffer = float(boundary["hard_horizontal_buffer"])
        altitude_buffer = float(boundary["hard_altitude_buffer"])
        aircraft.state.x = float(np.clip(aircraft.state.x, -x_limit + horizontal_buffer, x_limit - horizontal_buffer))
        aircraft.state.y = float(np.clip(aircraft.state.y, -y_limit + horizontal_buffer, y_limit - horizontal_buffer))
        aircraft.state.z = -float(np.clip(aircraft.state.altitude, altitude_min + altitude_buffer, altitude_max - altitude_buffer))
        aircraft.state.psi = self._recovery_heading(aircraft.state)
        aircraft.state.theta = self._recovery_pitch(aircraft.state)
        key = self._boundary_team_key(aircraft)
        self._episode_metrics[f"{key}_boundary_hard_contacts"] += 1.0
        self._episode_metrics[f"{key}_boundary_hard_contacts_step"] += 1.0
        return True

    def _strict_full_elimination(self) -> bool:
        blue_alive = sum(self._alive(bid) for bid in BLUE_IDS_V12)
        red_alive = sum(self._alive(cid) for cid in RED_COMBAT_IDS_V12)
        if blue_alive != 0:
            return False
        blue_death_causes = [self._death_causes.get(bid, DEATH_NONE_V12) for bid in BLUE_IDS_V12]
        if self._attack_kills["red"] != 3 or any(cause != DEATH_LOCK_V12 for cause in blue_death_causes):
            raise RuntimeError(
                "v12 full-elimination consistency failure: "
                f"blue_alive=0 red_lock_kills={self._attack_kills['red']} "
                f"blue_death_causes={blue_death_causes}"
            )
        return red_alive > 0

    def _terminal_result(self) -> tuple[bool, str, str]:
        red_alive = sum(self._alive(cid) for cid in RED_COMBAT_IDS_V12)
        blue_alive = sum(self._alive(bid) for bid in BLUE_IDS_V12)
        strict_full = self._strict_full_elimination() if blue_alive == 0 else False
        if strict_full:
            return True, "red", "red_full_elimination"
        if red_alive == 0 and blue_alive == 0:
            return True, "draw", "mutual_elimination_draw"
        if red_alive == 0:
            return True, "blue", "red_total_loss"
        if self.step_count >= int(self.config["simulation"]["max_steps"]):
            if red_alive > blue_alive:
                return True, "red", "timeout_red_win"
            if red_alive < blue_alive:
                return True, "blue", "timeout_red_loss"
            return True, "draw", "timeout_draw"
        return False, "", ""

    def _compute_reward(
        self,
        pre_targets: dict[str, str | None],
        pre_potentials: dict[str, float],
        pre_locks: dict[str, float],
        deaths: dict[str, int],
        half_events: set[tuple[str, str]],
        killers: dict[str, str],
        direct: dict[str, set[str]],
    ) -> tuple[float, dict[str, float]]:
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
        events = self.reward_contract["events"]
        for aircraft_id, cause in deaths.items():
            aircraft = self._by_id(aircraft_id)
            if cause == DEATH_LOCK_V12 and aircraft.team == "blue":
                components["blue_kill_event_reward"] += float(events["blue_combat_killed"])
            elif cause == DEATH_LOCK_V12 and aircraft.team == "red" and aircraft.role == "combat":
                components["red_combat_loss_event_penalty"] += float(events["red_combat_killed"])
            elif cause == DEATH_LOCK_V12 and aircraft.aircraft_id == "red_0":
                components["support_loss_event_penalty"] += float(events["red_support_killed"])

        red_assisted = 0
        for target_id, killer_id in killers.items():
            if killer_id in RED_COMBAT_IDS_V12 and target_id in BLUE_IDS_V12:
                last = self._cue_last_step.get((killer_id, target_id))
                if last is not None and self.step_count - last <= 50:
                    red_assisted += 1
        components["support_assisted_kill_reward"] = float(self.reward_contract["support_events"]["assisted_kill"]) * red_assisted

        geometry = 0.0
        for cid in RED_COMBAT_IDS_V12:
            old_target = pre_targets.get(cid)
            new_target = self.targets.get(cid)
            if old_target is None or old_target != new_target or not self._alive(cid) or new_target is None or not self._alive(new_target):
                continue
            current = combat_potential_v11(self._by_id(cid).state, self._by_id(new_target).state, self.profile)
            delta = current - pre_potentials.get(cid, current)
            geometry += float(self.reward_contract["combat_progress"]["geometry_scale"]) * float(np.clip(delta / 0.05, -1.0, 1.0))
        components["combat_geometry_progress_reward"] = geometry / 3.0
        lock_delta = np.mean([
            self.lock_progress.get(cid, 0.0) - pre_locks.get(cid, 0.0)
            for cid in RED_COMBAT_IDS_V12
        ])
        components["combat_lock_progress_reward"] = float(self.reward_contract["combat_progress"]["lock_scale"]) * float(np.clip(lock_delta / 0.10, -1.0, 1.0))
        components["combat_half_lock_event_reward"] = float(self.reward_contract["combat_progress"]["half_lock_event"]) * sum(
            1 for cid, bid in half_events if cid in RED_COMBAT_IDS_V12 and bid in BLUE_IDS_V12
        )
        components["support_unique_detection_reward"] = float(self.reward_contract["support_events"]["unique_detection"]) * self._episode_metrics.get("support_unique_detection_events_step", 0.0)
        components["support_cue_to_direct_reward"] = float(self.reward_contract["support_events"]["cue_to_direct"]) * self._episode_metrics.get("support_cue_to_direct_events_step", 0.0)
        components["support_cue_to_half_lock_reward"] = float(self.reward_contract["support_events"]["cue_to_half_lock"]) * self._episode_metrics.get("support_cue_to_half_lock_events_step", 0.0)
        components["support_formation_progress_reward"] = float(self.reward_contract["support_formation"]["progress_scale"]) * float(np.clip(self._formation_score() - self._last_formation_score, -1.0, 1.0))

        raw_dense = (
            components["combat_geometry_progress_reward"]
            + components["combat_lock_progress_reward"]
            + components["support_formation_progress_reward"]
        )
        clip_min = float(self.reward_contract["dense_clip"]["min"])
        clip_max = float(self.reward_contract["dense_clip"]["max"])
        components["total_dense_reward"] = float(np.clip(raw_dense, clip_min, clip_max))
        self._episode_metrics["dense_steps"] += 1.0
        self._episode_metrics["raw_dense_sum"] += raw_dense
        self._episode_metrics["raw_dense_min"] = raw_dense if self._episode_metrics["dense_steps"] == 1 else min(self._episode_metrics["raw_dense_min"], raw_dense)
        self._episode_metrics["raw_dense_max"] = raw_dense if self._episode_metrics["dense_steps"] == 1 else max(self._episode_metrics["raw_dense_max"], raw_dense)
        self._episode_metrics["dense_positive_saturation_steps"] += float(raw_dense > clip_max)
        self._episode_metrics["dense_negative_saturation_steps"] += float(raw_dense < clip_min)

        red_contacts = self._episode_metrics.get("red_boundary_hard_contacts_step", 0.0) + self._episode_metrics.get("support_boundary_hard_contacts_step", 0.0)
        components["boundary_event_penalty"] = max(
            -0.4,
            float(events["red_boundary_hard_contact"]) * min(4.0, red_contacts),
        )
        done, _, reason = self._terminal_result()
        if done:
            components["mission_outcome_reward"] = float(self.reward_contract["mission"][reason])
        self._last_raw_dense_reward = float(raw_dense)
        components["team_total_reward"] = aggregate_team_reward_v12(components)
        if not np.isfinite(list(components.values())).all():
            raise FloatingPointError("v12 reward components must be finite")
        return float(components["team_total_reward"]), components

    def step(self, red_actions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._running:
            raise RuntimeError("environment must be reset before step")
        self.step_count += 1
        for key in (
            "support_unique_detection_events_step",
            "support_cue_to_direct_events_step",
            "support_cue_to_half_lock_events_step",
            "red_boundary_soft_recovery_steps_step",
            "blue_boundary_soft_recovery_steps_step",
            "support_boundary_soft_recovery_steps_step",
            "red_boundary_hard_contacts_step",
            "blue_boundary_hard_contacts_step",
            "support_boundary_hard_contacts_step",
        ):
            self._episode_metrics[key] = 0.0
        self._record_support_cue_activity()
        direct_pre = self._direct_visible_ids()
        pre_targets = dict(self.targets)
        pre_locks = dict(self.lock_progress)
        pre_potentials = {
            cid: combat_potential_v11(self._by_id(cid).state, self._by_id(target).state, self.profile)
            for cid, target in self.targets.items()
            if cid in RED_COMBAT_IDS_V12 and target is not None and self._alive(cid) and self._alive(target)
        }
        blue_actions = self._blue_rule_actions(direct_pre)
        all_actions = {aid: np.asarray(red_actions.get(aid, np.zeros(3)), dtype=np.float32) for aid in RED_IDS_V12}
        all_actions.update(blue_actions)
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                continue
            corrected, blend = self._boundary_recovery_action(aircraft, all_actions[aircraft.aircraft_id])
            if blend > 0.0:
                key = self._boundary_team_key(aircraft)
                self._episode_metrics[f"{key}_boundary_soft_recovery_steps"] += 1.0
                self._episode_metrics[f"{key}_boundary_soft_recovery_steps_step"] += 1.0
            _, control = self.controller.control_from_action(aircraft.state, corrected, aircraft.spec)
            aircraft.state = self.integrator.step(aircraft.state, control, self.dynamics, aircraft.spec)
            self._project_hard_boundary(aircraft)

        deaths: dict[str, int] = {}
        direct_post = self._direct_visible_ids()
        before_unique = len(self._support_seen)
        before_direct = len(self._cue_to_direct_pairs)
        self._record_support_events(direct_post, self._effective_targets(direct_post))
        self._episode_metrics["support_unique_detection_events_step"] = float(len(self._support_seen) - before_unique)
        self._episode_metrics["support_cue_to_direct_events_step"] = float(len(self._cue_to_direct_pairs) - before_direct)
        before_half = len(self._cue_to_half_pairs)
        half_events, killers = self._update_locks(direct_post)
        self._episode_metrics["support_cue_to_half_lock_events_step"] = float(max(0, len(self._cue_to_half_pairs) - before_half))
        for target_id, killer_id in killers.items():
            if self._alive(target_id):
                self._by_id(target_id).state.alive = False
                deaths[target_id] = DEATH_LOCK_V12
                team = self._by_id(killer_id).team
                self._attack_kills[team] += 1
                self._kill_steps[team].append(self.step_count)
                if killer_id in RED_COMBAT_IDS_V12 and target_id in BLUE_IDS_V12:
                    last_cue = self._cue_last_step.get((killer_id, target_id))
                    if last_cue is not None and self.step_count - last_cue <= 50:
                        self._episode_metrics["support_assisted_kills"] += 1.0
        for aircraft_id, cause in deaths.items():
            if self._death_causes.get(aircraft_id, DEATH_NONE_V12) == DEATH_NONE_V12:
                self._death_causes[aircraft_id] = cause
        reward, components = self._compute_reward(pre_targets, pre_potentials, pre_locks, deaths, half_events, killers, direct_post)
        self._episode_return += reward
        self._last_reward_components = components
        for key, value in components.items():
            self._episode_reward_components[key] += float(value)
        self._last_formation_score = self._formation_score()
        done, outcome, reason = self._terminal_result()
        if done:
            self._running = False
        else:
            direct_next = self._direct_visible_ids()
            self._update_support_cues(direct_next)
            self._refresh_targets(direct_next, self._effective_targets(direct_next), advance_counters=True)
        obs, state, masks = self._observations()
        info = {
            "reward_components": components,
            "reward_groups": reward_group_totals_v12(components),
            "raw_dense_reward": float(self._last_raw_dense_reward),
            "episode_summary": self._episode_summary(outcome, reason) if done else None,
        }
        return obs, state, masks, reward, done, False, info

    def _episode_summary(self, outcome: str | None, reason: str | None) -> dict[str, Any]:
        summary = super()._episode_summary(outcome, reason)
        length = max(1, self.step_count)
        red_alive = sum(self._alive(cid) for cid in RED_COMBAT_IDS_V12)
        blue_alive = sum(self._alive(bid) for bid in BLUE_IDS_V12)
        strict_full = self._strict_full_elimination() if blue_alive == 0 else False
        blue_non_lock = sum(
            1 for bid in BLUE_IDS_V12
            if not self._alive(bid) and self._death_causes.get(bid, DEATH_NONE_V12) != DEATH_LOCK_V12
        )
        red_non_lock = sum(
            1 for cid in RED_COMBAT_IDS_V12
            if not self._alive(cid) and self._death_causes.get(cid, DEATH_NONE_V12) != DEATH_LOCK_V12
        )
        consistency = not (blue_alive == 0 and (self._attack_kills["red"] != 3 or blue_non_lock != 0))
        summary.update({
            "full_elimination": bool(strict_full),
            "red_complete_elimination_success": bool(strict_full),
            "red_full_elimination": bool(strict_full),
            "strict_full_elimination": bool(strict_full),
            "strict_full_elimination_rate": float(strict_full),
            "full_elimination_consistency_pass": bool(consistency),
            "non_lock_blue_death_count": int(blue_non_lock),
            "non_lock_red_combat_death_count": int(red_non_lock),
            "red_lock_kills": int(self._attack_kills["red"]),
            "blue_lock_kills": int(self._attack_kills["blue"]),
            "red_boundary_soft_recovery_steps": float(self._episode_metrics["red_boundary_soft_recovery_steps"]),
            "blue_boundary_soft_recovery_steps": float(self._episode_metrics["blue_boundary_soft_recovery_steps"]),
            "support_boundary_soft_recovery_steps": float(self._episode_metrics["support_boundary_soft_recovery_steps"]),
            "red_boundary_hard_contacts": int(self._episode_metrics["red_boundary_hard_contacts"]),
            "blue_boundary_hard_contacts": int(self._episode_metrics["blue_boundary_hard_contacts"]),
            "support_boundary_hard_contacts": int(self._episode_metrics["support_boundary_hard_contacts"]),
            "red_boundary_soft_recovery_step_rate": float(self._episode_metrics["red_boundary_soft_recovery_steps"] / length),
            "blue_boundary_soft_recovery_step_rate": float(self._episode_metrics["blue_boundary_soft_recovery_steps"] / length),
            "support_boundary_soft_recovery_step_rate": float(self._episode_metrics["support_boundary_soft_recovery_steps"] / length),
            "support_cue_to_half_lock_rate": float(len(self._cue_to_half_pairs) / max(1, len(self._cue_pairs))),
            "mean_red_boundary_hard_contacts": float(self._episode_metrics["red_boundary_hard_contacts"]),
            "mean_blue_boundary_hard_contacts": float(self._episode_metrics["blue_boundary_hard_contacts"]),
            "mean_support_boundary_hard_contacts": float(self._episode_metrics["support_boundary_hard_contacts"]),
            "full_elimination_rate": float(strict_full),
            "any_kill": bool(self._attack_kills["red"] > 0),
            "at_least_two_kill": bool(self._attack_kills["red"] >= 2),
            "red_combat_survivors": int(red_alive),
            "blue_combat_survivors": int(blue_alive),
        })
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({
            "environment_variant": self.variant,
            "reward_contract_version": self.reward_contract_version,
            "boundary_metrics": {
                key: float(value)
                for key, value in self._episode_metrics.items()
                if "boundary_" in key
            },
        })
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_variant = state.get("environment_variant")
        if saved_variant is not None and saved_variant != self.variant:
            raise ValueError(f"v12 environment state variant mismatch: {saved_variant!r}")
        super().load_state_dict(state)
        for key in (
            "red_boundary_soft_recovery_steps", "blue_boundary_soft_recovery_steps",
            "support_boundary_soft_recovery_steps", "red_boundary_hard_contacts",
            "blue_boundary_hard_contacts", "support_boundary_hard_contacts",
            "red_boundary_soft_recovery_steps_step", "blue_boundary_soft_recovery_steps_step",
            "support_boundary_soft_recovery_steps_step", "red_boundary_hard_contacts_step",
            "blue_boundary_hard_contacts_step", "support_boundary_hard_contacts_step",
        ):
            self._episode_metrics.setdefault(key, 0.0)


FunctionalHeterogeneous4v3AirCombatEnvV12 = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv
FunctionalHeterogeneous4v3V12TargetLockSupportCueEnv = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv

__all__ = [
    "BLUE_TEAM_SIZE_V12", "DEATH_LOCK_V12", "GS_DIM_V12", "OBS_DIM_V12",
    "RED_TEAM_SIZE_V12", "REWARD_COMPONENT_KEYS_V12", "aggregate_team_reward_v12",
    "reward_group_totals_v12", "FunctionalHeterogeneous4v3AirCombatEnvV12",
    "FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv",
    "FunctionalHeterogeneous4v3V12TargetLockSupportCueEnv",
]
