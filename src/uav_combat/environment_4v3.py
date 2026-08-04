"""Functional heterogeneous red 4v3 main-experiment environment (v9)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .combat import SimplifiedAttackModel
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .formation_4v3 import compute_red_combat_formation_reference
from .geometry import compute_pairwise_geometry
from .integrator import RK4Integrator
from .models import Aircraft, AircraftState
from .scenario_4v3 import (
    ALL_IDS_4V3,
    BLUE_IDS_4V3,
    RED_COMBAT_IDS_4V3,
    RED_IDS_4V3,
    resolved_reward_contract_4v3,
    FunctionalHeterogeneous4v3Scenario,
    validate_heterogeneous_4v3_config,
)

DEATH_NONE = 0
DEATH_BOUNDARY_ALTITUDE = 1
DEATH_BOUNDARY_XY = 2
DEATH_ATTACK = 5

OBS_DIM_4V3 = 118
GS_DIM_4V3 = 70
RED_TEAM_SIZE_4V3 = 4
BLUE_TEAM_SIZE_4V3 = 3
RED_REWARD_COMPONENT_KEYS_4V3 = (
    "mission_reward",
    "kill_event_reward",
    "combat_loss_event_penalty",
    "support_loss_event_penalty",
    "boundary_event_penalty",
    "support_assisted_kill_reward",
    "combat_approach_reward",
    "combat_readiness_reward",
    "combat_threat_penalty",
    "combat_boundary_penalty",
    "support_coverage_reward",
    "support_position_reward",
    "support_threat_penalty",
    "support_boundary_penalty",
    "total_dense_reward",
    "team_total_reward",
)


def _clip_obs(values: list[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if out.size > OBS_DIM_4V3:
        raise ValueError(f"4v3 observation too large: {out.size} > {OBS_DIM_4V3}")
    if out.size < OBS_DIM_4V3:
        out = np.pad(out, (0, OBS_DIM_4V3 - out.size))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _f_distance(distance: float, d_min: float, d_attack: float, d_tail: float) -> float:
    d = float(distance)
    if d < d_min:
        return float(np.clip(d / max(d_min, 1e-8), 0.0, 1.0))
    if d <= d_attack:
        return 1.0
    if d >= d_tail:
        return 0.0
    return float(np.clip((d_tail - d) / max(d_tail - d_attack, 1e-8), 0.0, 1.0))


def _attack_readiness(
    attacker: AircraftState,
    target: AircraftState,
    d_min: float,
    d_max: float,
    fade_distance: float = 2000.0,
) -> float:
    g = compute_pairwise_geometry(attacker, target)
    f_ata = max(0.0, 1.0 - float(g.ata) / (np.pi / 6.0))
    f_aa = max(0.0, 1.0 - float(g.aa) / (np.pi / 2.0))
    fd = _f_distance(float(g.distance), d_min, d_max, fade_distance)
    return float(np.clip(fd * f_ata * f_aa, 0.0, 1.0))


def _boundary_risk(ac: Aircraft, limits: dict[str, float], soft_margin: float = 1000.0) -> float:
    if not ac.state.alive:
        return 0.0
    margin_xy = min(float(limits["x_limit"]) - abs(ac.state.x), float(limits["y_limit"]) - abs(ac.state.y))
    margin_alt = min(ac.state.altitude - float(limits["altitude_min"]),
                     float(limits["altitude_max"]) - ac.state.altitude)
    margin = min(margin_xy, margin_alt)
    if margin >= soft_margin:
        return 0.0
    return float(np.clip((soft_margin - margin) / max(soft_margin, 1e-8), 0.0, 1.0))


class FunctionalHeterogeneous4v3AirCombatEnv:
    """Red support + three red combat aircraft vs three fixed-rule blue combat aircraft."""

    def __init__(self, config_path: str | Path = "configs/heterogeneous_4v3_main_v9.yaml") -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        validate_heterogeneous_4v3_config(self.config)
        self.reward_contract = resolved_reward_contract_4v3(self.config)
        sim, action, combat = self.config["simulation"], self.config["action"], self.config["combat"]
        self.scenario = FunctionalHeterogeneous4v3Scenario(self.config)
        self.dynamics = PointMassDynamics(float(sim["gravity"]))
        self.integrator = RK4Integrator(float(sim["dt"]))
        self.controller = TargetStateController(**action, gravity=float(sim["gravity"]))
        self.attack_model = SimplifiedAttackModel(
            float(combat["attack_distance_min"]),
            float(combat["attack_distance_max"]),
            float(combat["attack_ata_max"]),
            float(combat["attack_aa_max"]),
        )
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False
        self._death_causes: dict[str, int] = {}
        self._attack_kills = {"red": 0, "blue": 0}
        self._attack_kill_steps = {"red": [], "blue": []}
        self._prev_target_distance: dict[str, tuple[str, float]] = {}
        self._first_support_only_shared_step: dict[str, dict[str, int]] = {
            cid: {} for cid in RED_COMBAT_IDS_4V3
        }
        self._last_support_only_shared_step: dict[str, dict[str, int]] = {
            cid: {} for cid in RED_COMBAT_IDS_4V3
        }
        self._share_to_direct_recorded: set[tuple[str, str]] = set()
        self._share_to_direct_delays: list[int] = []
        self._share_to_kill_delays: list[int] = []
        self._episode_metrics: dict[str, float] = {}
        self._episode_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}
        self._last_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}
        self._blue_rule_policy = None
        self._red_rule_policy = None

    def _rule_policy(self, team: str):
        from .rule_policy_4v3 import make_rule_policy_4v3

        if team == "blue":
            if self._blue_rule_policy is None:
                self._blue_rule_policy = make_rule_policy_4v3(self.config, "blue")
            return self._blue_rule_policy
        if self._red_rule_policy is None:
            self._red_rule_policy = make_rule_policy_4v3(self.config, "red")
        return self._red_rule_policy

    def formation_reference(self) -> dict[str, Any]:
        return compute_red_combat_formation_reference(
            self._by_id("red_0"),
            [self._by_id(cid) for cid in RED_COMBAT_IDS_4V3],
            direction_validity_threshold=float(self.config["support_formation"]["direction_validity_threshold"]),
        )

    def _by_id(self, aid: str) -> Aircraft:
        return next(a for a in self.aircraft if a.aircraft_id == aid)

    def _support_formation(self) -> dict[str, float]:
        cfg = self.config.get("support_formation", {})
        return {
            "reward_optimal_min": float(cfg.get("reward_optimal_min", 800.0)),
            "reward_optimal_max": float(cfg.get("reward_optimal_max", 1800.0)),
            "reward_fade_near": float(cfg.get("reward_fade_near", 400.0)),
            "reward_fade_far": float(cfg.get("reward_fade_far", 3000.0)),
            "rear_alignment_threshold": float(cfg.get("rear_alignment_threshold", 0.7)),
        }

    def _team(self, team: str) -> list[Aircraft]:
        return [a for a in self.aircraft if a.team == team]

    def _alive_team(self, team: str) -> list[Aircraft]:
        return [a for a in self.aircraft if a.team == team and a.state.alive]

    def _alive_red_combat(self) -> list[Aircraft]:
        return [self._by_id(aid) for aid in RED_COMBAT_IDS_4V3 if self._by_id(aid).state.alive]

    def _alive_blue(self) -> list[Aircraft]:
        return [self._by_id(aid) for aid in BLUE_IDS_4V3 if self._by_id(aid).state.alive]

    def _direct_visible(self, own: Aircraft, target: Aircraft) -> bool:
        if not own.state.alive or not target.state.alive or own.team == target.team:
            return False
        return float(compute_pairwise_geometry(own.state, target.state).distance) <= float(own.sensor_range)

    def _direct_visible_ids(self) -> dict[str, set[str]]:
        return {
            ac.aircraft_id: {e.aircraft_id for e in self.aircraft if e.team != ac.team and self._direct_visible(ac, e)}
            for ac in self.aircraft
        }

    def _effective_visible_ids(self, direct: dict[str, set[str]]) -> dict[str, set[str]]:
        effective = {aid: set(v) for aid, v in direct.items()}
        support = self._by_id("red_0")
        if support.state.alive:
            shared = direct["red_0"] & set(BLUE_IDS_4V3)
            for cid in RED_COMBAT_IDS_4V3:
                combat = self._by_id(cid)
                if combat.state.alive:
                    effective[cid] |= shared
        return effective

    def _info_source(self, own: Aircraft, target: Aircraft, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> int:
        if not target.state.alive:
            return -1
        if target.aircraft_id in direct.get(own.aircraft_id, set()):
            return 2
        if target.aircraft_id in effective.get(own.aircraft_id, set()):
            return 1
        return 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        self._running = True
        self._death_causes = {aid: DEATH_NONE for aid in ALL_IDS_4V3}
        self._attack_kills = {"red": 0, "blue": 0}
        self._attack_kill_steps = {"red": [], "blue": []}
        self._prev_target_distance.clear()
        self._first_support_only_shared_step = {cid: {} for cid in RED_COMBAT_IDS_4V3}
        self._last_support_only_shared_step = {cid: {} for cid in RED_COMBAT_IDS_4V3}
        self._share_to_direct_recorded = set()
        self._share_to_direct_delays = []
        self._share_to_kill_delays = []
        self._episode_metrics = {
            "support_unique_detection_steps": 0,
            "support_shared_target_steps": 0,
            "support_only_target_steps": 0,
            "support_shared_pair_steps": 0,
            "shared_only_combat_target_pairs": 0,
            "shared_only_pair_ratio_sum": 0.0,
            "support_active_steps": 0,
            "combat_early_acquisition_steps": 0,
            "support_assisted_kills": 0,
            "support_to_combat_centroid_distance_sum": 0.0,
            "support_rear_position_steps": 0,
            "support_threat_exposure_steps": 0,
            "combat_attack_window_steps": 0,
            "combat_readiness_sum": 0.0,
            "combat_threat_sum": 0.0,
        }
        self._episode_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}
        self._last_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}
        return self._observations()

    def _observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        direct = self._direct_visible_ids()
        effective = self._effective_visible_ids(direct)
        obs = np.stack([self._obs_for(self._by_id(aid), direct, effective) for aid in ALL_IDS_4V3], axis=0)
        gs = self._global_state()
        mask = np.array([1.0 if self._by_id(aid).state.alive else 0.0 for aid in ALL_IDS_4V3], dtype=np.float32)
        return obs.astype(np.float32), gs.astype(np.float32), mask

    def _obs_for(self, own: Aircraft, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> np.ndarray:
        s = own.state
        values: list[float] = [
            s.x / 20000.0, s.y / 20000.0, s.altitude / 6000.0,
            s.v / 250.0, s.theta / (np.pi / 2), s.psi / np.pi,
            1.0 if s.alive else 0.0,
            1.0 if own.role == "support" else 0.0,
            1.0 if own.role == "combat" else 0.0,
            1.0 if own.can_attack else 0.0,
            own.sensor_range / 6000.0,
            1.0 if self._by_id("red_0").state.alive else 0.0,
        ]
        teammates = [a for a in self._team(own.team) if a.aircraft_id != own.aircraft_id]
        teammate_slots = RED_IDS_4V3 if own.team == "red" else BLUE_IDS_4V3
        for aid in teammate_slots:
            if aid == own.aircraft_id:
                continue
            t = self._by_id(aid)
            rel = t.state.as_array()[:3] - s.as_array()[:3]
            rv = t.state.velocity_vector() - s.velocity_vector()
            values.extend([rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                           rv[0] / 300.0, rv[1] / 300.0, rv[2] / 300.0,
                           1.0 if t.state.alive else 0.0,
                           1.0 if t.role == "support" else 0.0,
                           1.0 if t.can_attack else 0.0])
        enemy_ids = BLUE_IDS_4V3 if own.team == "red" else RED_IDS_4V3
        for eid in enemy_ids:
            e = self._by_id(eid)
            source = self._info_source(own, e, direct, effective)
            if source <= 0:
                values.extend([0.0] * 10 + [float(source) / 2.0])
                continue
            rel = e.state.as_array()[:3] - s.as_array()[:3]
            rv = e.state.velocity_vector() - s.velocity_vector()
            g = compute_pairwise_geometry(s, e.state)
            values.extend([rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                           rv[0] / 300.0, rv[1] / 300.0, rv[2] / 300.0,
                           g.distance / 6000.0, g.ata / np.pi, g.aa / np.pi,
                           1.0 if e.state.alive else 0.0, float(source) / 2.0])
        if own.aircraft_id == "red_0":
            alive_combat = self._alive_red_combat()
            if alive_combat:
                reference = self.formation_reference()
                rel = reference["centroid"] - s.as_array()[:3]
                direction = reference["horizontal_direction"]
                mean_heading = float(np.arctan2(direction[1], direction[0])) if reference["direction_valid"] else 0.0
                nearest_threat = min((compute_pairwise_geometry(b.state, s).distance for b in self._alive_blue()), default=6000.0)
                support_only = self._support_only_target_count(direct)
                values.extend([rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                               mean_heading / np.pi, reference["centroid_distance"] / 6000.0,
                               support_only / 3.0, nearest_threat / 6000.0])
        return _clip_obs(values)

    def _global_state(self) -> np.ndarray:
        vals: list[float] = []
        for aid in ALL_IDS_4V3:
            a = self._by_id(aid)
            s = a.state
            vals.extend([s.x / 20000.0, s.y / 20000.0, s.altitude / 6000.0,
                         s.v / 250.0, s.theta / (np.pi / 2), s.psi / np.pi,
                         1.0 if a.team == "red" else -1.0,
                         1.0 if a.role == "support" else 0.0,
                         1.0 if a.can_attack else 0.0,
                         1.0 if s.alive else 0.0])
        return np.clip(np.asarray(vals, dtype=np.float32), -1.0, 1.0)

    def _support_only_target_count(self, direct: dict[str, set[str]]) -> int:
        if not self._by_id("red_0").state.alive:
            return 0
        count = 0
        for bid in BLUE_IDS_4V3:
            if bid not in direct["red_0"]:
                continue
            if any(self._by_id(cid).state.alive and bid not in direct[cid] for cid in RED_COMBAT_IDS_4V3):
                count += 1
        return count

    def _support_only_pair_count(
        self,
        direct: dict[str, set[str]],
        effective: dict[str, set[str]],
    ) -> int:
        support = self._by_id("red_0")
        if not support.state.alive:
            return 0
        return sum(
            1
            for cid in RED_COMBAT_IDS_4V3
            if self._by_id(cid).state.alive
            for bid in BLUE_IDS_4V3
            if self._by_id(bid).state.alive
            and bid in direct["red_0"]
            and bid not in direct[cid]
            and bid in effective[cid]
        )

    def _support_only_pair_ratio(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> float:
        denominator = max(1, len(self._alive_red_combat()) * len(self._alive_blue()))
        return float(self._support_only_pair_count(direct, effective) / denominator)

    def _support_position_score(self) -> tuple[float, float, float]:
        support = self._by_id("red_0")
        alive_combat = self._alive_red_combat()
        if not support.state.alive or not alive_combat:
            return 0.0, 0.0, 0.0
        formation = self._support_formation()
        reference = self.formation_reference()
        rel = reference["support_relative"]
        dist = float(reference["centroid_distance"])
        behind = max(0.0, float(reference["rear_alignment"])) if reference["direction_valid"] else 0.0
        fade_near = formation["reward_fade_near"]
        optimal_min = formation["reward_optimal_min"]
        optimal_max = formation["reward_optimal_max"]
        fade_far = formation["reward_fade_far"]
        if dist < fade_near:
            band = dist / max(fade_near, 1e-8)
        elif dist < optimal_min:
            band = (dist - fade_near) / max(optimal_min - fade_near, 1e-8)
        elif dist <= optimal_max:
            band = 1.0
        elif dist >= fade_far:
            band = 0.0
        else:
            band = (fade_far - dist) / max(fade_far - optimal_max, 1e-8)
        score = float(np.clip((behind * band) if reference["direction_valid"] else band, 0.0, 1.0))
        rear = 1.0 if reference["direction_valid"] and behind > formation["rear_alignment_threshold"] and optimal_min <= dist <= optimal_max else 0.0
        return score, dist, rear

    def _nearest_effective_target(self, combat: Aircraft, effective: dict[str, set[str]]) -> Aircraft | None:
        candidates = [self._by_id(bid) for bid in BLUE_IDS_4V3 if self._by_id(bid).state.alive and bid in effective[combat.aircraft_id]]
        if not candidates:
            return None
        return min(candidates, key=lambda b: (compute_pairwise_geometry(combat.state, b.state).distance, b.aircraft_id))

    def _compute_reward(self, direct: dict[str, set[str]], effective: dict[str, set[str]], step_deaths: dict[str, int], assisted: int) -> tuple[float, dict[str, float]]:
        combat = self.config["combat"]
        limits = self.config["battlefield"]
        rewards = self.reward_contract
        mission_rewards = rewards["mission"]
        event_rewards = rewards["events"]
        combat_dense = rewards["combat_dense"]
        support_dense = rewards["support_dense"]
        soft_margin = float(rewards["boundary"]["soft_margin"])
        d_min = float(combat["attack_distance_min"])
        d_max = float(combat["attack_distance_max"])
        components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}

        red_combat_alive = len(self._alive_red_combat())
        blue_alive = len(self._alive_blue())

        for aid, cause in step_deaths.items():
            ac = self._by_id(aid)
            if cause == DEATH_ATTACK and ac.team == "blue":
                components["kill_event_reward"] += float(event_rewards["blue_combat_attack_kill"])
            elif cause == DEATH_ATTACK and ac.team == "red" and ac.role == "combat":
                components["combat_loss_event_penalty"] += float(event_rewards["red_combat_attack_loss"])
            elif cause == DEATH_ATTACK and ac.aircraft_id == "red_0":
                components["support_loss_event_penalty"] += float(event_rewards["red_support_attack_loss"])
            elif cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY) and ac.team == "red":
                components["boundary_event_penalty"] += float(event_rewards["red_boundary_loss"])
        components["support_assisted_kill_reward"] += float(event_rewards["support_assisted_kill"]) * float(assisted)

        combat_approach = combat_readiness = combat_threat = combat_boundary = 0.0
        for cid in RED_COMBAT_IDS_4V3:
            c = self._by_id(cid)
            if not c.state.alive:
                continue
            target = self._nearest_effective_target(c, effective)
            if target is not None:
                g = compute_pairwise_geometry(c.state, target.state)
                if g.distance > d_max:
                    prev = self._prev_target_distance.get(cid)
                    if prev is not None and prev[0] == target.aircraft_id:
                        combat_approach += float(combat_dense["approach_scale"]) * float(
                            np.clip((prev[1] - g.distance) / float(combat_dense["approach_distance_normalizer"]), -1.0, 1.0)
                        )
                ready = float(combat_dense["readiness_scale"]) * _attack_readiness(
                    c.state, target.state, d_min, d_max, float(combat_dense["readiness_fade_distance"])
                )
                combat_readiness += ready
            threats = [b for b in self._alive_blue() if c.aircraft_id in direct[b.aircraft_id]]
            if threats:
                combat_threat += float(combat_dense["threat_scale"]) * max(
                    _attack_readiness(b.state, c.state, d_min, d_max, float(combat_dense["readiness_fade_distance"]))
                    for b in threats
                )
            combat_boundary += float(combat_dense["boundary_scale"]) * _boundary_risk(c, limits, soft_margin)
        components["combat_approach_reward"] = combat_approach / 3.0
        components["combat_readiness_reward"] = combat_readiness / 3.0
        components["combat_threat_penalty"] = -combat_threat / 3.0
        components["combat_boundary_penalty"] = -combat_boundary / 3.0
        self._episode_metrics["combat_readiness_sum"] += components["combat_readiness_reward"]
        self._episode_metrics["combat_threat_sum"] += -components["combat_threat_penalty"]

        support = self._by_id("red_0")
        support_only_count = self._support_only_target_count(direct)
        support_only_pairs = self._support_only_pair_count(direct, effective)
        pair_ratio = self._support_only_pair_ratio(direct, effective)
        alive_blue_count = max(1, blue_alive)
        coverage = float(support_dense["coverage_scale"]) * pair_ratio if support.state.alive else 0.0
        pos_score, _, _ = self._support_position_score()
        support_threat = 0.0
        if support.state.alive and self._alive_blue():
            support_threat = float(support_dense["threat_scale"]) * max(
                _attack_readiness(b.state, support.state, d_min, d_max, float(combat_dense["readiness_fade_distance"]))
                for b in self._alive_blue()
            )
        support_boundary = float(support_dense["boundary_scale"]) * _boundary_risk(support, limits, soft_margin) if support.state.alive else 0.0
        components["support_coverage_reward"] = coverage
        components["support_position_reward"] = float(support_dense["position_scale"]) * pos_score
        components["support_threat_penalty"] = -support_threat
        components["support_boundary_penalty"] = -support_boundary

        dense = (
            components["combat_approach_reward"] + components["combat_readiness_reward"] +
            components["combat_threat_penalty"] + components["combat_boundary_penalty"] +
            components["support_coverage_reward"] + components["support_position_reward"] +
            components["support_threat_penalty"] + components["support_boundary_penalty"]
        )
        components["total_dense_reward"] = float(np.clip(dense, float(rewards["dense_clip"]["min"]), float(rewards["dense_clip"]["max"])))

        done, outcome, reason = self._termination()
        if done:
            if reason == "red_complete_elimination_success":
                components["mission_reward"] += float(mission_rewards["red_complete_elimination_success"])
            elif reason == "red_all_combat_eliminated":
                components["mission_reward"] += float(mission_rewards["red_all_combat_eliminated"])
            elif reason == "timeout":
                components["mission_reward"] += float(mission_rewards["timeout"])
            elif reason == "mutual_combat_elimination":
                components["mission_reward"] += float(mission_rewards["mutual_combat_elimination"])
            elif reason == "blue_noncombat_elimination":
                components["mission_reward"] += float(mission_rewards["blue_noncombat_elimination"])

        event = (components["kill_event_reward"] + components["combat_loss_event_penalty"] +
                 components["support_loss_event_penalty"] + components["boundary_event_penalty"] +
                 components["support_assisted_kill_reward"])
        components["team_total_reward"] = components["mission_reward"] + event + components["total_dense_reward"]
        return float(components["team_total_reward"]), components

    def _record_support_share_metrics(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> None:
        support = self._by_id("red_0")
        if not support.state.alive:
            return
        if not self._alive_red_combat():
            return
        self._episode_metrics["support_active_steps"] += 1
        support_only_count = self._support_only_target_count(direct)
        support_only_pairs = self._support_only_pair_count(direct, effective)
        pair_ratio = self._support_only_pair_ratio(direct, effective)
        if support_only_count > 0:
            self._episode_metrics["support_unique_detection_steps"] += 1
            self._episode_metrics["support_only_target_steps"] += 1
        if support_only_pairs > 0:
            self._episode_metrics["support_shared_pair_steps"] += 1
        self._episode_metrics["shared_only_combat_target_pairs"] += support_only_pairs
        self._episode_metrics["shared_only_pair_ratio_sum"] += pair_ratio
        shared_this_step = 0
        for cid in RED_COMBAT_IDS_4V3:
            combat = self._by_id(cid)
            if not combat.state.alive:
                continue
            for bid in BLUE_IDS_4V3:
                if bid in direct["red_0"] and bid not in direct[cid] and bid in effective[cid]:
                    self._first_support_only_shared_step[cid].setdefault(bid, self.step_count)
                    self._last_support_only_shared_step[cid][bid] = self.step_count
                    shared_this_step += 1
        if shared_this_step > 0:
            self._episode_metrics["support_shared_target_steps"] += 1
        pos_score, dist, rear = self._support_position_score()
        self._episode_metrics["support_to_combat_centroid_distance_sum"] += dist
        self._episode_metrics["support_rear_position_steps"] += rear
        if self._alive_blue() and max(
            _attack_readiness(
                b.state,
                support.state,
                float(self.config["combat"]["attack_distance_min"]),
                float(self.config["combat"]["attack_distance_max"]),
                float(self.reward_contract["combat_dense"]["readiness_fade_distance"]),
            )
            for b in self._alive_blue()
        ) > 0.0:
            self._episode_metrics["support_threat_exposure_steps"] += 1

    def _record_direct_observations_after_dynamics(self, direct: dict[str, set[str]]) -> None:
        for cid in RED_COMBAT_IDS_4V3:
            if not self._by_id(cid).state.alive:
                continue
            for bid in BLUE_IDS_4V3:
                if bid not in direct[cid]:
                    continue
                first_share = self._first_support_only_shared_step[cid].get(bid)
                if first_share is None:
                    # Legacy diagnostics may seed only the last-share map;
                    # normal episode state records both maps together.
                    first_share = self._last_support_only_shared_step[cid].get(bid)
                if first_share is None:
                    continue
                key = (cid, bid)
                if key in self._share_to_direct_recorded:
                    continue
                delay = self.step_count - first_share
                if delay >= 0:
                    self._share_to_direct_delays.append(int(delay))
                    self._episode_metrics["combat_early_acquisition_steps"] += 1
                    self._share_to_direct_recorded.add(key)

    def _update_support_metrics(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> None:
        """Compatibility wrapper used by diagnostics and focused tests."""
        self._record_support_share_metrics(direct, effective)
        self._record_direct_observations_after_dynamics(direct)

    def _termination(self) -> tuple[bool, str | None, str | None]:
        red_combat_alive = len(self._alive_red_combat())
        blue_alive = len(self._alive_blue())
        red_all_combat_dead = red_combat_alive == 0
        blue_all_dead = blue_alive == 0
        strict_red_success = (
            self._attack_kills["red"] == BLUE_TEAM_SIZE_4V3
            and all(self._death_causes.get(bid) == DEATH_ATTACK for bid in BLUE_IDS_4V3)
            and red_combat_alive > 0
        )
        if red_all_combat_dead and blue_all_dead:
            return True, "draw", "mutual_combat_elimination"
        if red_all_combat_dead:
            return True, "blue", "red_all_combat_eliminated"
        if strict_red_success:
            return True, "red", "red_complete_elimination_success"
        if blue_all_dead:
            return True, "draw", "blue_noncombat_elimination"
        if self.step_count >= int(self.config["simulation"]["max_steps"]):
            return True, "blue" if blue_alive > 0 else "draw", "timeout"
        return False, None, None

    def red_rule_actions(self) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        """Return deterministic red rule actions using support-shared visibility.

        This helper is for rule-vs-rule reachability checks. RL training still
        supplies red actions externally.
        """
        direct = self._direct_visible_ids()
        effective = self._effective_visible_ids(direct)
        policy = self._rule_policy("red")
        return policy.select_actions(self._team("red"), self._team("blue"), visibility=effective)

    def step(self, red_actions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._running:
            raise RuntimeError("environment must be reset before step")
        self.step_count += 1
        direct_pre = self._direct_visible_ids()
        effective_pre = self._effective_visible_ids(direct_pre)
        self._record_support_share_metrics(direct_pre, effective_pre)

        # Persist previous target distances for progress reward.
        for cid in RED_COMBAT_IDS_4V3:
            c = self._by_id(cid)
            target = self._nearest_effective_target(c, effective_pre) if c.state.alive else None
            if target is not None:
                self._prev_target_distance.setdefault(cid, (target.aircraft_id, compute_pairwise_geometry(c.state, target.state).distance))

        blue_policy = self._rule_policy("blue")
        blue_actions, _ = blue_policy.select_actions(self._team("blue"), self._team("red"), visibility=direct_pre)

        all_actions: dict[str, np.ndarray] = {}
        all_actions.update({aid: np.asarray(red_actions.get(aid, np.zeros(3)), dtype=np.float32) for aid in RED_IDS_4V3})
        all_actions.update(blue_actions)

        for ac in self.aircraft:
            if not ac.state.alive:
                continue
            target_cmd, control = self.controller.control_from_action(ac.state, all_actions.get(ac.aircraft_id, np.zeros(3)), ac.spec)
            ac.state = self.integrator.step(ac.state, control, self.dynamics, ac.spec)

        step_deaths: dict[str, int] = {}
        limits = self.config["battlefield"]
        for ac in self.aircraft:
            if not ac.state.alive:
                continue
            if not (float(limits["altitude_min"]) <= ac.state.altitude <= float(limits["altitude_max"])):
                ac.state.alive = False
                step_deaths[ac.aircraft_id] = DEATH_BOUNDARY_ALTITUDE
            elif abs(ac.state.x) > float(limits["x_limit"]) or abs(ac.state.y) > float(limits["y_limit"]):
                ac.state.alive = False
                step_deaths[ac.aircraft_id] = DEATH_BOUNDARY_XY

        direct = self._direct_visible_ids()
        effective = self._effective_visible_ids(direct)
        # Direct visibility is evaluated again after dynamics and before attack
        # resolution, so a newly acquired target cannot lose its first-direct event.
        self._record_direct_observations_after_dynamics(direct)
        attackers_by_target: dict[str, list[str]] = {}
        for ac in self.aircraft:
            if not ac.state.alive or not ac.can_attack:
                continue
            enemy_ids = BLUE_IDS_4V3 if ac.team == "red" else RED_IDS_4V3
            candidates = [self._by_id(eid) for eid in enemy_ids if self._by_id(eid).state.alive]
            if ac.team == "red":
                candidates = [e for e in candidates if e.aircraft_id in direct[ac.aircraft_id]]
            else:
                candidates = [e for e in candidates if e.aircraft_id in direct[ac.aircraft_id]]
            attackable = [e for e in candidates if self.attack_model.can_attack(ac.state, e.state)]
            if attackable:
                target = min(attackable, key=lambda e: (compute_pairwise_geometry(ac.state, e.state).distance, e.aircraft_id))
                attackers_by_target.setdefault(target.aircraft_id, []).append(ac.aircraft_id)
        if any(aid.startswith("red_") for attackers in attackers_by_target.values() for aid in attackers):
            self._episode_metrics["combat_attack_window_steps"] += 1

        assisted = 0
        assisted_targets: set[str] = set()
        for target_id, attackers in attackers_by_target.items():
            target = self._by_id(target_id)
            if not target.state.alive:
                continue
            target.state.alive = False
            step_deaths.setdefault(target_id, DEATH_ATTACK)
            killing_team = self._by_id(attackers[0]).team
            self._attack_kills[killing_team] += 1
            self._attack_kill_steps[killing_team].append(self.step_count)
            if killing_team == "red" and target_id in BLUE_IDS_4V3:
                red_attackers = sorted(aid for aid in attackers if aid in RED_COMBAT_IDS_4V3)
                if red_attackers:
                    window = int(self.reward_contract["support_credit"]["assisted_window_steps"])
                    eligible: list[tuple[int, float, str]] = []
                    for attacker_id in red_attackers:
                        last = self._last_support_only_shared_step.get(attacker_id, {}).get(target_id)
                        if last is None or self.step_count - last > window:
                            continue
                        distance = float(compute_pairwise_geometry(self._by_id(attacker_id).state, target.state).distance)
                        eligible.append((int(last), distance, attacker_id))
                    if eligible and target_id not in assisted_targets:
                        # Prefer the most recent valid share, then the closest
                        # participating combat aircraft at the kill step.
                        last, _, _ = sorted(eligible, key=lambda item: (-item[0], item[1], item[2]))[0]
                        assisted_targets.add(target_id)
                        assisted += 1
                        self._episode_metrics["support_assisted_kills"] += 1
                        self._share_to_kill_delays.append(int(self.step_count - last))

        for aid, cause in step_deaths.items():
            if self._death_causes.get(aid, DEATH_NONE) == DEATH_NONE:
                self._death_causes[aid] = cause

        reward, components = self._compute_reward(direct, effective, step_deaths, assisted)
        self._last_reward_components = components
        for key, value in components.items():
            self._episode_reward_components[key] += float(value)

        # Update progress baseline after reward calculation.
        self._prev_target_distance.clear()
        for cid in RED_COMBAT_IDS_4V3:
            c = self._by_id(cid)
            target = self._nearest_effective_target(c, effective) if c.state.alive else None
            if target is not None:
                self._prev_target_distance[cid] = (target.aircraft_id, compute_pairwise_geometry(c.state, target.state).distance)

        done, outcome, reason = self._termination()
        if done:
            self._running = False
        obs, gs, mask = self._observations()
        info = {
            "episode_summary": self._episode_summary(outcome, reason) if done else None,
            "reward_components": components,
        }
        return obs, gs, mask, reward, done, False, info

    def _episode_summary(self, outcome: str | None, reason: str | None) -> dict[str, Any]:
        red_survivors = sum(1 for aid in RED_IDS_4V3 if self._by_id(aid).state.alive)
        blue_survivors = sum(1 for aid in BLUE_IDS_4V3 if self._by_id(aid).state.alive)
        red_combat_survivors = len(self._alive_red_combat())
        red_all_combat_eliminated = red_combat_survivors == 0
        support_alive = self._by_id("red_0").state.alive
        length = max(1, int(self.step_count))
        support_active = int(self._episode_metrics["support_active_steps"])
        support_denominator = max(1, support_active)
        summary = {
            "episode_length": int(self.step_count),
            "environment_outcome": outcome,
            "termination_reason": reason,
            "red_win": bool(reason == "red_complete_elimination_success"),
            "red_complete_elimination_success": bool(reason == "red_complete_elimination_success"),
            "mutual_combat_elimination": bool(reason == "mutual_combat_elimination"),
            "blue_noncombat_elimination": bool(reason == "blue_noncombat_elimination"),
            "red_all_combat_eliminated": bool(red_all_combat_eliminated),
            "red_attack_kills": int(self._attack_kills["red"]),
            "blue_attack_kills": int(self._attack_kills["blue"]),
            "red_any_attack_kill": bool(self._attack_kills["red"] > 0),
            "red_survivors": int(red_survivors),
            "blue_survivors": int(blue_survivors),
            "red_combat_survivors": int(red_combat_survivors),
            "support_survived": bool(support_alive),
            "death_causes": dict(self._death_causes),
            "reward_components": dict(self._episode_reward_components),
        }
        for team in ("red", "blue"):
            steps = self._attack_kill_steps[team]
            for i, label in enumerate(("first", "second", "third")):
                summary[f"{team}_{label}_attack_kill_step"] = steps[i] if len(steps) > i else None
        summary.update({
            "support_unique_detection_steps": int(self._episode_metrics["support_unique_detection_steps"]),
            "support_shared_target_steps": int(self._episode_metrics["support_shared_target_steps"]),
            "support_assisted_kills": int(self._episode_metrics["support_assisted_kills"]),
            "support_assisted_kill_rate": float(self._episode_metrics["support_assisted_kills"] / max(1, self._attack_kills["red"])),
            "support_assisted_episode_rate": float(self._episode_metrics["support_assisted_kills"] > 0),
            "support_active_steps": support_active,
            "support_only_target_steps": int(self._episode_metrics["support_only_target_steps"]),
            "support_shared_pair_steps": int(self._episode_metrics["support_shared_pair_steps"]),
            "mean_shared_only_combat_target_pairs": float(self._episode_metrics["shared_only_combat_target_pairs"] / support_denominator),
            "mean_shared_only_pair_ratio": float(self._episode_metrics["shared_only_pair_ratio_sum"] / support_denominator),
            "support_only_target_step_rate": float(self._episode_metrics["support_only_target_steps"] / support_denominator),
            "support_shared_pair_step_rate": float(self._episode_metrics["support_shared_pair_steps"] / support_denominator),
            "share_to_direct_event_count": int(len(self._share_to_direct_delays)),
            "share_to_kill_event_count": int(len(self._share_to_kill_delays)),
            "mean_share_to_direct_delay": float(np.mean(self._share_to_direct_delays)) if self._share_to_direct_delays else None,
            "mean_share_to_kill_delay": float(np.mean(self._share_to_kill_delays)) if self._share_to_kill_delays else None,
            "combat_early_acquisition_events": int(self._episode_metrics["combat_early_acquisition_steps"]),
            "combat_early_acquisition_steps": int(self._episode_metrics["combat_early_acquisition_steps"]),
            "support_unique_detection_step_rate": float(self._episode_metrics["support_unique_detection_steps"] / support_denominator),
            "support_shared_target_step_rate": float(self._episode_metrics["support_shared_target_steps"] / support_denominator),
            "combat_attack_window_step_rate": float(self._episode_metrics["combat_attack_window_steps"] / length),
            "combat_readiness_mean": float(self._episode_metrics["combat_readiness_sum"] / length),
            "combat_threat_mean": float(self._episode_metrics["combat_threat_sum"] / length),
            "mean_support_to_combat_centroid_distance": float(self._episode_metrics["support_to_combat_centroid_distance_sum"] / support_denominator) if support_active else 0.0,
            "support_rear_position_rate": float(self._episode_metrics["support_rear_position_steps"] / support_denominator) if support_active else 0.0,
            "support_threat_exposure_rate": float(self._episode_metrics["support_threat_exposure_steps"] / support_denominator) if support_active else 0.0,
        })
        return summary

    def state_dict(self) -> dict[str, Any]:
        """Serialize all mutable episode state needed for exact checkpoint resume."""
        return {
            "aircraft": {
                ac.aircraft_id: {
                    "state": ac.state.as_array().tolist(),
                    "alive": bool(ac.state.alive),
                }
                for ac in self.aircraft
            },
            "step_count": int(self.step_count),
            "running": bool(self._running),
            "death_causes": dict(self._death_causes),
            "attack_kills": dict(self._attack_kills),
            "attack_kill_steps": {team: list(steps) for team, steps in self._attack_kill_steps.items()},
            "prev_target_distance": {key: [value[0], float(value[1])] for key, value in self._prev_target_distance.items()},
            "first_support_only_shared_step": {
                cid: dict(values) for cid, values in self._first_support_only_shared_step.items()
            },
            "last_support_only_shared_step": {
                cid: dict(values) for cid, values in self._last_support_only_shared_step.items()
            },
            "share_to_direct_recorded": [list(key) for key in self._share_to_direct_recorded],
            "share_to_direct_delays": list(self._share_to_direct_delays),
            "share_to_kill_delays": list(self._share_to_kill_delays),
            "episode_metrics": dict(self._episode_metrics),
            "episode_reward_components": dict(self._episode_reward_components),
            "last_reward_components": dict(self._last_reward_components),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self.aircraft:
            self.aircraft = self.scenario.reset(0)
        for aid, saved in state["aircraft"].items():
            ac = self._by_id(aid)
            values = [float(v) for v in saved["state"]]
            ac.state = AircraftState(*values, bool(saved["alive"]))
        self.step_count = int(state["step_count"])
        self._running = bool(state["running"])
        self._death_causes = {str(k): int(v) for k, v in state["death_causes"].items()}
        self._attack_kills = {str(k): int(v) for k, v in state["attack_kills"].items()}
        self._attack_kill_steps = {str(k): [int(v) for v in values] for k, values in state["attack_kill_steps"].items()}
        self._prev_target_distance = {str(k): (str(v[0]), float(v[1])) for k, v in state["prev_target_distance"].items()}
        self._first_support_only_shared_step = {
            str(cid): {str(bid): int(step) for bid, step in values.items()}
            for cid, values in state["first_support_only_shared_step"].items()
        }
        self._last_support_only_shared_step = {
            str(cid): {str(bid): int(step) for bid, step in values.items()}
            for cid, values in state["last_support_only_shared_step"].items()
        }
        self._share_to_direct_recorded = {tuple(key) for key in state["share_to_direct_recorded"]}
        self._share_to_direct_delays = [int(v) for v in state["share_to_direct_delays"]]
        self._share_to_kill_delays = [int(v) for v in state["share_to_kill_delays"]]
        self._episode_metrics = {str(k): float(v) for k, v in state["episode_metrics"].items()}
        self._episode_reward_components = {str(k): float(v) for k, v in state["episode_reward_components"].items()}
        self._last_reward_components = {str(k): float(v) for k, v in state["last_reward_components"].items()}
