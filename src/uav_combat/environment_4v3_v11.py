"""Learnability-first 4v3 environment with deterministic target locks and support cues."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .formation_4v3 import compute_red_combat_formation_reference
from .geometry import compute_pairwise_geometry
from .integrator import RK4Integrator
from .math_utils import angle_difference
from .models import Aircraft, AircraftState
from .scenario_4v3_v11 import (
    ALL_IDS_V11,
    BLUE_IDS_V11,
    RED_COMBAT_IDS_V11,
    RED_IDS_V11,
    REWARD_COMPONENT_KEYS_V11,
    FunctionalHeterogeneous4v3V11Scenario,
    resolved_reward_contract_v11,
    validate_heterogeneous_4v3_v11_config,
)

OBS_DIM_V11 = 118
GS_DIM_V11 = 70
RED_TEAM_SIZE_V11 = 4
BLUE_TEAM_SIZE_V11 = 3
DEATH_NONE_V11 = 0
DEATH_BOUNDARY_V11 = 1
DEATH_LOCK_V11 = 5


def distance_score_v11(distance: float, profile: dict[str, Any]) -> float:
    d = float(distance)
    d_min = float(profile["lock_distance_min"])
    d_opt = float(profile["lock_distance_optimal_max"])
    d_fade = float(profile["lock_distance_fade_max"])
    if d < d_min:
        return float(np.clip(d / max(d_min, 1e-8), 0.0, 1.0))
    if d <= d_opt:
        return 1.0
    if d >= d_fade:
        return 0.0
    return float(np.clip((d_fade - d) / max(d_fade - d_opt, 1e-8), 0.0, 1.0))


def lock_quality_v11(attacker: AircraftState, target: AircraftState, profile: dict[str, Any]) -> float:
    geometry = compute_pairwise_geometry(attacker, target)
    ata_score = float(np.clip(1.0 - geometry.ata / float(profile["lock_ata_fade_max"]), 0.0, 1.0))
    aa_score = float(np.clip(1.0 - geometry.aa / float(profile["lock_aa_fade_max"]), 0.0, 1.0))
    return distance_score_v11(geometry.distance, profile) * ata_score * aa_score


def combat_potential_v11(attacker: AircraftState, target: AircraftState, profile: dict[str, Any]) -> float:
    geometry = compute_pairwise_geometry(attacker, target)
    distance = distance_score_v11(geometry.distance, profile)
    ata = float(np.clip(1.0 - geometry.ata / float(profile["lock_ata_fade_max"]), 0.0, 1.0))
    aa = float(np.clip(1.0 - geometry.aa / float(profile["lock_aa_fade_max"]), 0.0, 1.0))
    return float(0.40 * distance + 0.35 * ata + 0.25 * aa)


def _clip_obs(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.size > OBS_DIM_V11:
        raise ValueError(f"v11 observation too large: {result.size} > {OBS_DIM_V11}")
    if result.size < OBS_DIM_V11:
        result = np.pad(result, (0, OBS_DIM_V11 - result.size))
    return np.clip(result, -1.0, 1.0).astype(np.float32)


class FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv:
    """Red support + three red combat aircraft against three equal blue combat aircraft."""

    variant = "functional_heterogeneous_4v3_v11_target_lock_support_cue"
    reward_contract_version = "v11_target_lock_support_cue"

    def __init__(self, config_path: str | Path = "configs/heterogeneous_4v3_main_v11_target_lock_support_cue.yaml") -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        validate_heterogeneous_4v3_v11_config(self.config)
        self.reward_contract = resolved_reward_contract_v11(self.config)
        self.profile = self.config["combat_profile"]
        simulation = self.config["simulation"]
        self.scenario = FunctionalHeterogeneous4v3V11Scenario(self.config)
        self.dynamics = PointMassDynamics(float(simulation.get("gravity", 9.81)))
        self.integrator = RK4Integrator(float(simulation["dt"]))
        self.controller = TargetStateController(**self.config["action"], gravity=float(simulation.get("gravity", 9.81)))
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
        self._support_cues: dict[str, str | None] = {cid: None for cid in RED_COMBAT_IDS_V11}
        self._last_formation_score = 0.0
        self._episode_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V11}
        self._last_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V11}
        self._episode_metrics: dict[str, float] = {}
        self._episode_return = 0.0

    def _by_id(self, aircraft_id: str) -> Aircraft:
        return next(ac for ac in self.aircraft if ac.aircraft_id == aircraft_id)

    def _alive(self, aircraft_id: str) -> bool:
        return self._by_id(aircraft_id).state.alive

    def _direct_visible_ids(self) -> dict[str, set[str]]:
        return {
            ac.aircraft_id: {
                other.aircraft_id
                for other in self.aircraft
                if other.team != ac.team
                and other.state.alive
                and ac.state.alive
                and float(compute_pairwise_geometry(ac.state, other.state).distance) <= float(ac.sensor_range)
            }
            for ac in self.aircraft
        }

    def _support_cue_ids(self, direct: dict[str, set[str]]) -> dict[str, str | None]:
        if not self._alive("red_0"):
            return {cid: None for cid in RED_COMBAT_IDS_V11}
        visible = [self._by_id(bid) for bid in BLUE_IDS_V11 if bid in direct["red_0"] and self._alive(bid)]
        alive_red = [self._by_id(cid) for cid in RED_COMBAT_IDS_V11 if self._alive(cid)]
        assignments: dict[str, str | None] = {cid: None for cid in RED_COMBAT_IDS_V11}
        held_targets: set[str] = set()
        for combat in alive_red:
            current = self.targets.get(combat.aircraft_id)
            if current in {target.aircraft_id for target in visible} and self.lock_progress.get(combat.aircraft_id, 0.0) >= 0.25:
                assignments[combat.aircraft_id] = current
                held_targets.add(current)
        pairs = sorted(
            ((float(compute_pairwise_geometry(combat.state, target.state).distance), combat.aircraft_id, target.aircraft_id)
             for combat in alive_red for target in visible if target.aircraft_id not in held_targets),
            key=lambda item: (item[0], item[1], item[2]),
        )
        used_combat = {cid for cid, target in assignments.items() if target is not None}
        used_targets = set(held_targets)
        for _, combat_id, target_id in pairs:
            if combat_id not in used_combat and target_id not in used_targets:
                assignments[combat_id] = target_id
                used_combat.add(combat_id)
                used_targets.add(target_id)
        if visible:
            for combat in alive_red:
                if assignments[combat.aircraft_id] is None:
                    target = min(visible, key=lambda item: (compute_pairwise_geometry(combat.state, item.state).distance, item.aircraft_id))
                    assignments[combat.aircraft_id] = target.aircraft_id
        return assignments

    def _update_support_cues(self, direct: dict[str, set[str],], force: bool = False) -> None:
        if not force and self.step_count % 20 != 0:
            return
        new_cues = self._support_cue_ids(direct)
        for combat_id, target_id in new_cues.items():
            if target_id is not None:
                pair = (combat_id, target_id)
                self._cue_pairs.add(pair)
                self._cue_last_step[pair] = self.step_count
        self._support_cues = new_cues
        self._episode_metrics["support_cue_steps"] += 1.0

    def _effective_targets(self, direct: dict[str, set[str]]) -> dict[str, set[str]]:
        effective = {key: set(value) for key, value in direct.items()}
        for cid in RED_COMBAT_IDS_V11:
            cue = self._support_cues.get(cid)
            if cue is not None and self._alive("red_0") and self._alive(cue):
                effective[cid].add(cue)
        return effective

    def _nearest(self, aircraft_id: str, candidates: set[str]) -> str | None:
        own = self._by_id(aircraft_id)
        valid = [self._by_id(cid) for cid in candidates if self._alive(cid)]
        if not valid:
            return None
        return min(valid, key=lambda target: (compute_pairwise_geometry(own.state, target.state).distance, target.aircraft_id)).aircraft_id

    def _switch_target(self, aircraft_id: str, target_id: str | None) -> None:
        old = self.targets.get(aircraft_id)
        if old != target_id:
            if old is not None:
                self.target_switch_count += 1
            self.targets[aircraft_id] = target_id
            self.target_hold_steps[aircraft_id] = 0
            self.target_lost_steps[aircraft_id] = 0
            self.lock_progress[aircraft_id] = 0.0

    def _refresh_targets(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> None:
        profile = self.profile
        min_hold = int(profile["target_min_hold_steps"])
        release_steps = int(profile["target_lost_release_steps"])
        ratio = float(profile["target_switch_distance_ratio"])
        for aircraft_id in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11):
            if not self._alive(aircraft_id):
                self._switch_target(aircraft_id, None)
                continue
            own = self._by_id(aircraft_id)
            visible = direct[aircraft_id] if own.team == "blue" else effective[aircraft_id]
            current = self.targets.get(aircraft_id)
            if current is not None and not self._alive(current):
                current = None
                self._switch_target(aircraft_id, None)
            if current is not None:
                if current not in visible:
                    self.target_lost_steps[aircraft_id] += 1
                else:
                    self.target_lost_steps[aircraft_id] = 0
                self.target_hold_steps[aircraft_id] += 1
            candidates = {cid for cid in visible if self._alive(cid)}
            nearest = self._nearest(aircraft_id, candidates)
            if current is None:
                self._switch_target(aircraft_id, nearest)
                continue
            if self.target_hold_steps[aircraft_id] < min_hold:
                continue
            current_distance = compute_pairwise_geometry(own.state, self._by_id(current).state).distance
            nearest_distance = compute_pairwise_geometry(own.state, self._by_id(nearest).state).distance if nearest else float("inf")
            cue = self._support_cues.get(aircraft_id) if own.team == "red" else None
            cue_distance = compute_pairwise_geometry(own.state, self._by_id(cue).state).distance if cue and self._alive(cue) else float("inf")
            should_release = self.target_lost_steps[aircraft_id] >= release_steps
            should_switch = nearest is not None and nearest != current and nearest_distance < ratio * current_distance
            cue_switch = cue is not None and cue != current and self.lock_progress.get(aircraft_id, 0.0) < 0.25
            if should_release:
                self._switch_target(aircraft_id, nearest)
            elif should_switch:
                self._switch_target(aircraft_id, nearest)
            elif cue_switch and cue_distance < current_distance * ratio:
                self._switch_target(aircraft_id, cue)

    def _formation_score(self) -> float:
        support = self._by_id("red_0")
        combats = [self._by_id(cid) for cid in RED_COMBAT_IDS_V11 if self._alive(cid)]
        if not support.state.alive or not combats:
            return 0.0
        reference = compute_red_combat_formation_reference(support, combats)
        distance = float(reference["centroid_distance"])
        trailing = float(self.config["scenario"]["support_trailing_distance"])
        distance_score = float(np.clip(1.0 - abs(distance - trailing) / max(trailing, 1.0), 0.0, 1.0))
        return distance_score * max(0.0, float(reference["rear_alignment"]))

    def _action_towards(self, own: Aircraft, target_id: str | None) -> np.ndarray:
        if not own.state.alive or target_id is None or not self._alive(target_id):
            return np.zeros(3, dtype=np.float32)
        target = self._by_id(target_id)
        geometry = compute_pairwise_geometry(own.state, target.state)
        return np.asarray([
            np.clip(own.spec.k_yaw * geometry.yaw_error / max(abs(own.spec.yaw_rate_max), 1e-8), -1.0, 1.0),
            np.clip(own.spec.k_pitch * geometry.pitch_error / max(abs(own.spec.pitch_rate_max), 1e-8), -1.0, 1.0),
            np.clip(own.spec.k_speed * (target.state.v - own.state.v) / max(abs(own.spec.acceleration_max), 1e-8), -1.0, 1.0),
        ], dtype=np.float32)

    def _support_action(self) -> np.ndarray:
        support = self._by_id("red_0")
        combats = [self._by_id(cid) for cid in RED_COMBAT_IDS_V11 if self._alive(cid)]
        if not support.state.alive or not combats:
            return np.zeros(3, dtype=np.float32)
        reference = compute_red_combat_formation_reference(support, combats)
        desired = reference["centroid"] - np.pad(reference["horizontal_direction"], (0, 1)) * float(self.config["scenario"]["support_trailing_distance"])
        relative = desired - support.state.as_array()[:3]
        desired_psi = float(np.arctan2(relative[1], relative[0]))
        desired_theta = float(np.arctan2(-relative[2], max(float(np.hypot(relative[0], relative[1])), 1e-6)))
        return np.asarray([
            np.clip(support.spec.k_yaw * angle_difference(desired_psi, support.state.psi) / max(abs(support.spec.yaw_rate_max), 1e-8), -1.0, 1.0),
            np.clip(support.spec.k_pitch * (desired_theta - support.state.theta) / max(abs(support.spec.pitch_rate_max), 1e-8), -1.0, 1.0),
            0.0,
        ], dtype=np.float32)

    def red_rule_actions(self) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        direct = self._direct_visible_ids()
        self._update_support_cues(direct, force=False)
        self._refresh_targets(direct, self._effective_targets(direct))
        actions = {"red_0": self._support_action()}
        for cid in RED_COMBAT_IDS_V11:
            actions[cid] = self._action_towards(self._by_id(cid), self.targets.get(cid))
        return actions, dict(self.targets)

    def _blue_rule_actions(self, direct: dict[str, set[str]]) -> dict[str, np.ndarray]:
        actions = {}
        for bid in BLUE_IDS_V11:
            actions[bid] = self._action_towards(self._by_id(bid), self.targets.get(bid))
        return actions

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        self._running = True
        self._death_causes = {aid: DEATH_NONE_V11 for aid in ALL_IDS_V11}
        self._attack_kills = {"red": 0, "blue": 0}
        self._kill_steps = {"red": [], "blue": []}
        self.targets = {aid: None for aid in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11)}
        self.target_hold_steps = {aid: 0 for aid in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11)}
        self.target_lost_steps = {aid: 0 for aid in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11)}
        self.target_switch_count = 0
        self.lock_progress = {aid: 0.0 for aid in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11)}
        self.max_lock_progress = 0.0
        self._half_lock_pairs = set()
        self._support_seen = set()
        self._cue_pairs = set()
        self._cue_last_step = {}
        self._cue_to_direct_pairs = set()
        self._cue_to_half_pairs = set()
        self._support_cues = {cid: None for cid in RED_COMBAT_IDS_V11}
        self._episode_return = 0.0
        self._episode_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V11}
        self._last_reward_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V11}
        self._episode_metrics = {
            "support_cue_steps": 0.0, "support_unique_detection_events": 0.0,
            "support_cue_to_direct_events": 0.0, "support_cue_to_half_lock_events": 0.0,
            "support_assisted_kills": 0.0, "lock_episode_steps": 0.0, "half_lock_episode_steps": 0.0,
            "max_lock_sum": 0.0, "max_lock_count": 0.0, "dense_positive_saturation_steps": 0.0,
            "dense_negative_saturation_steps": 0.0, "dense_steps": 0.0, "raw_dense_sum": 0.0,
            "raw_dense_min": 0.0, "raw_dense_max": 0.0,
        }
        direct = self._direct_visible_ids()
        self._update_support_cues(direct, force=True)
        self._refresh_targets(direct, self._effective_targets(direct))
        self._last_formation_score = self._formation_score()
        return self._observations()

    def _observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        direct = self._direct_visible_ids()
        effective = self._effective_targets(direct)
        observations = np.stack([self._obs_for(self._by_id(aid), direct, effective) for aid in ALL_IDS_V11])
        states = self._global_state()
        masks = np.asarray([float(self._alive(aid)) for aid in ALL_IDS_V11], dtype=np.float32)
        return observations.astype(np.float32), states.astype(np.float32), masks

    def _obs_for(self, own: Aircraft, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> np.ndarray:
        state = own.state
        values: list[float] = [
            state.x / 20000.0, state.y / 20000.0, state.altitude / 6000.0,
            state.v / 250.0, state.theta / (np.pi / 2), state.psi / np.pi,
            float(state.alive), float(own.role == "support"), float(own.role == "combat"),
            float(own.can_attack), own.sensor_range / 6000.0, float(self._alive("red_0")),
        ]
        for other in self.aircraft:
            if other.aircraft_id == own.aircraft_id:
                continue
            rel = other.state.as_array()[:3] - state.as_array()[:3]
            rel_v = other.state.velocity_vector() - state.velocity_vector()
            values.extend([
                rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                rel_v[0] / 300.0, rel_v[1] / 300.0, rel_v[2] / 300.0,
                float(other.state.alive), float(other.role == "support"),
            ])
        enemy_ids = BLUE_IDS_V11 if own.team == "red" else RED_IDS_V11
        for enemy_id in enemy_ids:
            enemy = self._by_id(enemy_id)
            source = 2 if enemy_id in direct[own.aircraft_id] else (1 if enemy_id in effective[own.aircraft_id] else 0)
            if source == 0:
                values.extend([0.0] * 10)
            else:
                rel = enemy.state.as_array()[:3] - state.as_array()[:3]
                rel_v = enemy.state.velocity_vector() - state.velocity_vector()
                geometry = compute_pairwise_geometry(state, enemy.state)
                values.extend([
                    rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                    rel_v[0] / 300.0, rel_v[1] / 300.0, rel_v[2] / 300.0,
                    geometry.distance / 6000.0, geometry.ata / np.pi, geometry.aa / np.pi,
                    float(source) / 2.0,
                ])
        target_id = self.targets.get(own.aircraft_id)
        target = self._by_id(target_id) if target_id is not None and self._alive(target_id) else None
        if target is None:
            values.extend([0.0] * 7)
        else:
            source_direct = target_id in direct[own.aircraft_id]
            source_cue = own.team == "red" and target_id == self._support_cues.get(own.aircraft_id)
            geometry = compute_pairwise_geometry(state, target.state)
            values.extend([
                float(source_cue), float(source_direct), float(source_cue and not source_direct),
                self.lock_progress.get(own.aircraft_id, 0.0),
                max(0.0, float(self.profile["target_min_hold_steps"]) - self.target_hold_steps.get(own.aircraft_id, 0)) / max(float(self.profile["target_min_hold_steps"]), 1.0),
                float(self._alive("red_0")), geometry.distance / 6000.0,
            ])
        return _clip_obs(values)

    def _global_state(self) -> np.ndarray:
        values: list[float] = []
        for aid in ALL_IDS_V11:
            ac = self._by_id(aid)
            state = ac.state
            values.extend([
                state.x / 20000.0, state.y / 20000.0, state.altitude / 6000.0,
                state.v / 250.0, state.theta / (np.pi / 2), state.psi / np.pi,
                float(ac.team == "red") if ac.role != "support" else 0.5,
                float(ac.role == "support"), float(ac.can_attack), float(state.alive),
            ])
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def _record_support_events(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> None:
        support = self._by_id("red_0")
        if not support.state.alive:
            return
        for blue_id in BLUE_IDS_V11:
            if blue_id in direct["red_0"] and blue_id not in self._support_seen:
                self._support_seen.add(blue_id)
                self._episode_metrics["support_unique_detection_events"] += 1.0
        for cid in RED_COMBAT_IDS_V11:
            for bid in BLUE_IDS_V11:
                pair = (cid, bid)
                if bid in direct[cid] and pair in self._cue_pairs and pair not in self._cue_to_direct_pairs:
                    self._cue_to_direct_pairs.add(pair)
                    self._episode_metrics["support_cue_to_direct_events"] += 1.0

    def _update_locks(self, direct: dict[str, set[str]]) -> tuple[set[tuple[str, str]], dict[str, str]]:
        half_events: set[tuple[str, str]] = set()
        killers: dict[str, str] = {}
        for attacker_id in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11):
            if not self._alive(attacker_id):
                continue
            target_id = self.targets.get(attacker_id)
            old = self.lock_progress.get(attacker_id, 0.0)
            if target_id is None or not self._alive(target_id) or target_id not in direct[attacker_id]:
                new = max(0.0, old - float(self.profile["lock_decay_per_step"]))
            else:
                quality = lock_quality_v11(self._by_id(attacker_id).state, self._by_id(target_id).state, self.profile)
                new = float(np.clip(old + float(self.profile["lock_increment_scale"]) * quality, 0.0, 1.0)) if quality > 0.0 else max(0.0, old - float(self.profile["lock_decay_per_step"]))
            self.lock_progress[attacker_id] = new
            self.max_lock_progress = max(self.max_lock_progress, new)
            if target_id is not None:
                pair = (attacker_id, target_id)
                if new >= 0.5 and old < 0.5 and pair not in self._half_lock_pairs:
                    self._half_lock_pairs.add(pair)
                    half_events.add(pair)
                if attacker_id in RED_COMBAT_IDS_V11 and new >= 0.5 and pair in self._cue_pairs and pair not in self._cue_to_half_pairs:
                    self._cue_to_half_pairs.add(pair)
                    self._episode_metrics["support_cue_to_half_lock_events"] += 1.0
                if new >= float(self.profile["lock_kill_threshold"]) and self._alive(target_id):
                    killers[target_id] = attacker_id
        self._episode_metrics["max_lock_sum"] += max(self.lock_progress.values(), default=0.0)
        self._episode_metrics["max_lock_count"] += 1.0
        self._episode_metrics["lock_episode_steps"] += float(any(value > 0.0 for value in self.lock_progress.values()))
        self._episode_metrics["half_lock_episode_steps"] += float(any(value >= 0.5 for value in self.lock_progress.values()))
        return half_events, killers

    def _terminal_result(self) -> tuple[bool, str, str]:
        red_alive = sum(self._alive(cid) for cid in RED_COMBAT_IDS_V11)
        blue_alive = sum(self._alive(bid) for bid in BLUE_IDS_V11)
        if blue_alive == 0 and red_alive > 0:
            return True, "red", "red_full_elimination"
        if red_alive == 0 and blue_alive == 0:
            return True, "draw", "timeout_draw"
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
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V11}
        events = self.reward_contract["events"]
        for aircraft_id, cause in deaths.items():
            aircraft = self._by_id(aircraft_id)
            if cause == DEATH_LOCK_V11 and aircraft.team == "blue":
                components["blue_kill_event_reward"] += float(events["blue_combat_killed"])
            elif cause == DEATH_LOCK_V11 and aircraft.team == "red" and aircraft.role == "combat":
                components["red_combat_loss_event_penalty"] += float(events["red_combat_killed"])
            elif cause == DEATH_LOCK_V11 and aircraft.aircraft_id == "red_0":
                components["support_loss_event_penalty"] += float(events["red_support_killed"])
            elif cause == DEATH_BOUNDARY_V11 and aircraft.team == "red":
                components["boundary_event_penalty"] += float(events["red_boundary_loss"])
        red_assisted = 0
        for target_id, killer_id in killers.items():
            if killer_id in RED_COMBAT_IDS_V11 and target_id in BLUE_IDS_V11:
                last = self._cue_last_step.get((killer_id, target_id))
                if last is not None and self.step_count - last <= 50:
                    red_assisted += 1
        components["support_assisted_kill_reward"] = float(self.reward_contract["support_events"]["assisted_kill"]) * red_assisted
        components["combat_geometry_progress_reward"] = 0.0
        for cid in RED_COMBAT_IDS_V11:
            old_target = pre_targets.get(cid)
            new_target = self.targets.get(cid)
            if old_target is None or old_target != new_target or not self._alive(cid) or new_target is None or not self._alive(new_target):
                continue
            current = combat_potential_v11(self._by_id(cid).state, self._by_id(new_target).state, self.profile)
            delta = current - pre_potentials.get(cid, current)
            components["combat_geometry_progress_reward"] += float(self.reward_contract["combat_progress"]["geometry_scale"]) * float(np.clip(delta / 0.05, -1.0, 1.0))
        components["combat_geometry_progress_reward"] /= 3.0
        lock_delta = np.mean([
            self.lock_progress.get(cid, 0.0) - pre_locks.get(cid, 0.0)
            for cid in RED_COMBAT_IDS_V11
        ])
        components["combat_lock_progress_reward"] = float(self.reward_contract["combat_progress"]["lock_scale"]) * float(np.clip(lock_delta / 0.10, -1.0, 1.0))
        components["combat_half_lock_event_reward"] = float(self.reward_contract["combat_progress"]["half_lock_event"]) * sum(1 for cid, bid in half_events if cid in RED_COMBAT_IDS_V11 and bid in BLUE_IDS_V11)
        components["support_unique_detection_reward"] = float(self.reward_contract["support_events"]["unique_detection"]) * self._episode_metrics.get("support_unique_detection_events_step", 0.0)
        components["support_cue_to_direct_reward"] = float(self.reward_contract["support_events"]["cue_to_direct"]) * self._episode_metrics.get("support_cue_to_direct_events_step", 0.0)
        components["support_cue_to_half_lock_reward"] = float(self.reward_contract["support_events"]["cue_to_half_lock"]) * self._episode_metrics.get("support_cue_to_half_lock_events_step", 0.0)
        components["support_formation_progress_reward"] = float(self.reward_contract["support_formation"]["progress_scale"]) * float(np.clip(self._formation_score() - self._last_formation_score, -1.0, 1.0))
        raw_dense = (
            components["combat_geometry_progress_reward"] + components["combat_lock_progress_reward"] +
            components["support_formation_progress_reward"]
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
        done, _, reason = self._terminal_result()
        if done:
            mission = self.reward_contract["mission"]
            components["mission_outcome_reward"] = float(mission[reason])
        event_total = sum(components[key] for key in REWARD_COMPONENT_KEYS_V11 if key.endswith("_reward") and key not in {"mission_outcome_reward", "total_dense_reward", "team_total_reward", "combat_geometry_progress_reward", "combat_lock_progress_reward", "combat_half_lock_event_reward", "support_formation_progress_reward"})
        event_total += components["combat_half_lock_event_reward"]
        components["team_total_reward"] = components["mission_outcome_reward"] + event_total + components["total_dense_reward"] + components["combat_geometry_progress_reward"] + components["combat_lock_progress_reward"] + components["support_formation_progress_reward"]
        return float(components["team_total_reward"]), components

    def step(self, red_actions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._running:
            raise RuntimeError("environment must be reset before step")
        self.step_count += 1
        direct_pre = self._direct_visible_ids()
        self._update_support_cues(direct_pre)
        effective_pre = self._effective_targets(direct_pre)
        self._refresh_targets(direct_pre, effective_pre)
        pre_targets = dict(self.targets)
        pre_locks = dict(self.lock_progress)
        pre_potentials = {
            cid: combat_potential_v11(self._by_id(cid).state, self._by_id(target).state, self.profile)
            for cid, target in self.targets.items()
            if cid in RED_COMBAT_IDS_V11 and target is not None and self._alive(cid) and self._alive(target)
        }
        blue_actions = self._blue_rule_actions(direct_pre)
        all_actions = {aid: np.asarray(red_actions.get(aid, np.zeros(3)), dtype=np.float32) for aid in RED_IDS_V11}
        all_actions.update(blue_actions)
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                continue
            _, control = self.controller.control_from_action(aircraft.state, all_actions.get(aircraft.aircraft_id, np.zeros(3)), aircraft.spec)
            aircraft.state = self.integrator.step(aircraft.state, control, self.dynamics, aircraft.spec)

        deaths: dict[str, int] = {}
        battlefield = self.config["battlefield"]
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                continue
            if not (float(battlefield["altitude_min"]) <= aircraft.state.altitude <= float(battlefield["altitude_max"])):
                aircraft.state.alive = False
                deaths[aircraft.aircraft_id] = DEATH_BOUNDARY_V11
            elif abs(aircraft.state.x) > float(battlefield["x_limit"]) or abs(aircraft.state.y) > float(battlefield["y_limit"]):
                aircraft.state.alive = False
                deaths[aircraft.aircraft_id] = DEATH_BOUNDARY_V11

        direct_post = self._direct_visible_ids()
        self._record_support_events(direct_post, self._effective_targets(direct_post))
        # These counters are per-transition, while the episode counters above are cumulative.
        before_direct = len(self._cue_to_direct_pairs)
        before_half = len(self._cue_to_half_pairs)
        half_events, killers = self._update_locks(direct_post)
        for pair in half_events:
            if pair in self._cue_pairs:
                pass
        self._episode_metrics["support_cue_to_direct_events_step"] = 0.0
        self._episode_metrics["support_cue_to_half_lock_events_step"] = float(max(0, len(self._cue_to_half_pairs) - before_half))
        # Direct events are recorded above; count only new events for this transition.
        self._episode_metrics["support_cue_to_direct_events_step"] = float(max(0, len(self._cue_to_direct_pairs) - before_direct))
        for target_id, killer_id in killers.items():
            if self._alive(target_id):
                self._by_id(target_id).state.alive = False
                deaths[target_id] = DEATH_LOCK_V11
                team = self._by_id(killer_id).team
                self._attack_kills[team] += 1
                self._kill_steps[team].append(self.step_count)
        self._record_support_events(direct_post, self._effective_targets(direct_post))
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
            self._refresh_targets(direct_next, self._effective_targets(direct_next))
        obs, state, masks = self._observations()
        info = {"reward_components": components, "raw_dense_reward": float(components["total_dense_reward"]), "episode_summary": self._episode_summary(outcome, reason) if done else None}
        return obs, state, masks, reward, done, False, info

    def _episode_summary(self, outcome: str | None, reason: str | None) -> dict[str, Any]:
        length = max(1, self.step_count)
        red_alive = sum(self._alive(cid) for cid in RED_COMBAT_IDS_V11)
        blue_alive = sum(self._alive(bid) for bid in BLUE_IDS_V11)
        red_full = reason == "red_full_elimination"
        task_win = reason in {"red_full_elimination", "timeout_red_win"}
        return {
            "environment_variant": self.variant,
            "episode_length": int(self.step_count), "episode_return": float(self._episode_return),
            "environment_outcome": outcome, "termination_reason": reason,
            "task_win": bool(task_win), "full_elimination": bool(red_full),
            "red_win": bool(task_win), "red_complete_elimination_success": bool(red_full),
            "red_full_elimination": bool(red_full), "red_total_loss": bool(reason == "red_total_loss"),
            "timeout_red_win": bool(reason == "timeout_red_win"), "timeout_red_loss": bool(reason == "timeout_red_loss"),
            "timeout_draw": bool(reason == "timeout_draw"),
            "red_attack_kills": int(self._attack_kills["red"]), "blue_attack_kills": int(self._attack_kills["blue"]),
            "red_any_attack_kill": bool(self._attack_kills["red"] > 0),
            "red_attack_kill_distribution": {str(k): float(self._attack_kills["red"] == k) for k in range(4)},
            "red_combat_survivors": int(red_alive), "blue_combat_survivors": int(blue_alive),
            "support_survived": bool(self._alive("red_0")), "death_causes": dict(self._death_causes),
            "first_kill_time": self._kill_steps["red"][0] if self._kill_steps["red"] else None,
            "lock_episode_rate": float(self._episode_metrics["lock_episode_steps"] / length),
            "half_lock_episode_rate": float(self._episode_metrics["half_lock_episode_steps"] / length),
            "mean_max_lock_progress": float(self._episode_metrics["max_lock_sum"] / max(1.0, self._episode_metrics["max_lock_count"])),
            "target_switch_count": int(self.target_switch_count),
            "support_cue_rate": float(self._episode_metrics["support_cue_steps"] / length),
            "support_cue_to_direct_rate": float(len(self._cue_to_direct_pairs) / max(1, len(self._cue_pairs))),
            "support_assisted_kills": int(self._episode_metrics["support_assisted_kills"]),
            "support_assisted_kill_rate": float(self._episode_metrics["support_assisted_kills"] / max(1, self._attack_kills["red"])),
            "support_assisted_episode_rate": float(self._episode_metrics["support_assisted_kills"] > 0),
            "support_cue_count": int(len(self._cue_pairs)),
            "reward_components": dict(self._episode_reward_components),
            "dense_clip_positive_saturation_rate": float(self._episode_metrics["dense_positive_saturation_steps"] / max(1.0, self._episode_metrics["dense_steps"])),
            "dense_clip_negative_saturation_rate": float(self._episode_metrics["dense_negative_saturation_steps"] / max(1.0, self._episode_metrics["dense_steps"])),
            "dense_clip_saturation_rate": float((self._episode_metrics["dense_positive_saturation_steps"] + self._episode_metrics["dense_negative_saturation_steps"]) / max(1.0, self._episode_metrics["dense_steps"])),
            "raw_dense_reward_mean": float(self._episode_metrics["raw_dense_sum"] / max(1.0, self._episode_metrics["dense_steps"])),
            "raw_dense_reward_min": float(self._episode_metrics["raw_dense_min"]), "raw_dense_reward_max": float(self._episode_metrics["raw_dense_max"]),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "aircraft": {ac.aircraft_id: {"state": ac.state.as_array().tolist(), "alive": bool(ac.state.alive)} for ac in self.aircraft},
            "step_count": self.step_count, "running": self._running, "death_causes": dict(self._death_causes),
            "attack_kills": dict(self._attack_kills), "kill_steps": deepcopy(self._kill_steps),
            "targets": dict(self.targets), "target_hold_steps": dict(self.target_hold_steps),
            "target_lost_steps": dict(self.target_lost_steps), "target_switch_count": self.target_switch_count,
            "lock_progress": dict(self.lock_progress), "max_lock_progress": self.max_lock_progress,
            "half_lock_pairs": [list(pair) for pair in self._half_lock_pairs], "support_seen": list(self._support_seen),
            "cue_pairs": [list(pair) for pair in self._cue_pairs], "cue_last_step": {"|".join(pair): value for pair, value in self._cue_last_step.items()},
            "cue_to_direct_pairs": [list(pair) for pair in self._cue_to_direct_pairs], "cue_to_half_pairs": [list(pair) for pair in self._cue_to_half_pairs],
            "support_cues": dict(self._support_cues), "last_formation_score": self._last_formation_score,
            "episode_reward_components": dict(self._episode_reward_components), "last_reward_components": dict(self._last_reward_components),
            "episode_metrics": dict(self._episode_metrics), "episode_return": self._episode_return,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self.aircraft:
            self.aircraft = self.scenario.reset(0)
        for aid, saved in state["aircraft"].items():
            ac = self._by_id(aid)
            ac.state = AircraftState(*[float(v) for v in saved["state"]], bool(saved["alive"]))
        self.step_count = int(state["step_count"]); self._running = bool(state["running"])
        self._death_causes = {str(k): int(v) for k, v in state["death_causes"].items()}
        self._attack_kills = {str(k): int(v) for k, v in state["attack_kills"].items()}
        self._kill_steps = {str(k): [int(v) for v in values] for k, values in state["kill_steps"].items()}
        self.targets = {str(k): (str(v) if v is not None else None) for k, v in state["targets"].items()}
        self.target_hold_steps = {str(k): int(v) for k, v in state["target_hold_steps"].items()}
        self.target_lost_steps = {str(k): int(v) for k, v in state["target_lost_steps"].items()}
        self.target_switch_count = int(state["target_switch_count"]); self.lock_progress = {str(k): float(v) for k, v in state["lock_progress"].items()}
        self.max_lock_progress = float(state["max_lock_progress"])
        self._half_lock_pairs = {tuple(pair) for pair in state["half_lock_pairs"]}; self._support_seen = set(state["support_seen"])
        self._cue_pairs = {tuple(pair) for pair in state["cue_pairs"]}
        self._cue_last_step = {tuple(key.split("|")): int(value) for key, value in state["cue_last_step"].items()}
        self._cue_to_direct_pairs = {tuple(pair) for pair in state["cue_to_direct_pairs"]}; self._cue_to_half_pairs = {tuple(pair) for pair in state["cue_to_half_pairs"]}
        self._support_cues = {str(k): (str(v) if v is not None else None) for k, v in state["support_cues"].items()}
        self._last_formation_score = float(state["last_formation_score"])
        self._episode_reward_components = {str(k): float(v) for k, v in state["episode_reward_components"].items()}
        self._last_reward_components = {str(k): float(v) for k, v in state["last_reward_components"].items()}
        self._episode_metrics = {str(k): float(v) for k, v in state["episode_metrics"].items()}; self._episode_return = float(state["episode_return"])


FunctionalHeterogeneous4v3AirCombatEnvV11 = FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv

__all__ = [
    "BLUE_TEAM_SIZE_V11", "DEATH_BOUNDARY_V11", "DEATH_LOCK_V11", "GS_DIM_V11", "OBS_DIM_V11",
    "RED_TEAM_SIZE_V11", "REWARD_COMPONENT_KEYS_V11", "FunctionalHeterogeneous4v3AirCombatEnvV11",
    "FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv", "combat_potential_v11", "distance_score_v11", "lock_quality_v11",
]
