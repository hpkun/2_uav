"""Functional heterogeneous red 4v3 main-experiment environment (v9)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .combat import SimplifiedAttackModel
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .geometry import compute_pairwise_geometry
from .integrator import RK4Integrator
from .models import Aircraft, AircraftState
from .scenario_4v3 import (
    ALL_IDS_4V3,
    BLUE_IDS_4V3,
    RED_COMBAT_IDS_4V3,
    RED_IDS_4V3,
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


def _attack_readiness(attacker: AircraftState, target: AircraftState, d_min: float, d_max: float) -> float:
    g = compute_pairwise_geometry(attacker, target)
    f_ata = max(0.0, 1.0 - float(g.ata) / (np.pi / 6.0))
    f_aa = max(0.0, 1.0 - float(g.aa) / (np.pi / 2.0))
    fd = _f_distance(float(g.distance), d_min, d_max, 2000.0)
    return float(np.clip(fd * f_ata * f_aa, 0.0, 1.0))


def _boundary_risk(ac: Aircraft, limits: dict[str, float]) -> float:
    if not ac.state.alive:
        return 0.0
    margin_xy = min(float(limits["x_limit"]) - abs(ac.state.x), float(limits["y_limit"]) - abs(ac.state.y))
    margin_alt = min(ac.state.altitude - float(limits["altitude_min"]),
                     float(limits["altitude_max"]) - ac.state.altitude)
    margin = min(margin_xy, margin_alt)
    if margin >= 1000.0:
        return 0.0
    return float(np.clip((1000.0 - margin) / 1000.0, 0.0, 1.0))


class FunctionalHeterogeneous4v3AirCombatEnv:
    """Red support + three red combat aircraft vs three fixed-rule blue combat aircraft."""

    def __init__(self, config_path: str | Path = "configs/heterogeneous_4v3_main_v9.yaml") -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        validate_heterogeneous_4v3_config(self.config)
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
        self._last_support_only_shared_step: dict[str, dict[str, int]] = {
            cid: {} for cid in RED_COMBAT_IDS_4V3
        }
        self._share_to_direct_recorded: set[tuple[str, str]] = set()
        self._share_to_direct_delays: list[int] = []
        self._share_to_kill_delays: list[int] = []
        self._episode_metrics: dict[str, float] = {}
        self._episode_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}
        self._last_reward_components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}

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
        self._last_support_only_shared_step = {cid: {} for cid in RED_COMBAT_IDS_4V3}
        self._share_to_direct_recorded = set()
        self._share_to_direct_delays = []
        self._share_to_kill_delays = []
        self._episode_metrics = {
            "support_unique_detection_steps": 0,
            "support_shared_target_steps": 0,
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
                positions = np.array([a.state.as_array()[:3] for a in alive_combat])
                centroid = positions.mean(axis=0)
                rel = centroid - s.as_array()[:3]
                mean_heading = float(np.arctan2(np.mean([np.sin(a.state.psi) for a in alive_combat]),
                                                np.mean([np.cos(a.state.psi) for a in alive_combat])))
                nearest_threat = min((compute_pairwise_geometry(b.state, s).distance for b in self._alive_blue()), default=6000.0)
                support_only = self._support_only_target_count(direct)
                values.extend([rel[0] / 6000.0, rel[1] / 6000.0, rel[2] / 3000.0,
                               mean_heading / np.pi, np.linalg.norm(rel[:2]) / 6000.0,
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

    def _support_position_score(self) -> tuple[float, float, float]:
        support = self._by_id("red_0")
        alive_combat = self._alive_red_combat()
        if not support.state.alive or not alive_combat:
            return 0.0, 0.0, 0.0
        formation = self._support_formation()
        centroid = np.array([a.state.as_array()[:3] for a in alive_combat]).mean(axis=0)
        rel = support.state.as_array()[:3] - centroid
        dist = float(np.linalg.norm(rel[:2]))
        mean_heading = float(np.arctan2(np.mean([np.sin(a.state.psi) for a in alive_combat]),
                                        np.mean([np.cos(a.state.psi) for a in alive_combat])))
        backward = -np.array([np.cos(mean_heading), np.sin(mean_heading)])
        behind = max(0.0, float(np.dot(rel[:2] / max(dist, 1e-8), backward)))
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
        score = float(np.clip(behind * band, 0.0, 1.0))
        rear = 1.0 if behind > formation["rear_alignment_threshold"] and optimal_min <= dist <= optimal_max else 0.0
        return score, dist, rear

    def _nearest_effective_target(self, combat: Aircraft, effective: dict[str, set[str]]) -> Aircraft | None:
        candidates = [self._by_id(bid) for bid in BLUE_IDS_4V3 if self._by_id(bid).state.alive and bid in effective[combat.aircraft_id]]
        if not candidates:
            return None
        return min(candidates, key=lambda b: (compute_pairwise_geometry(combat.state, b.state).distance, b.aircraft_id))

    def _compute_reward(self, direct: dict[str, set[str]], effective: dict[str, set[str]], step_deaths: dict[str, int], assisted: int) -> tuple[float, dict[str, float]]:
        combat = self.config["combat"]
        limits = self.config["battlefield"]
        d_min = float(combat["attack_distance_min"])
        d_max = float(combat["attack_distance_max"])
        components = {k: 0.0 for k in RED_REWARD_COMPONENT_KEYS_4V3}

        red_combat_alive = len(self._alive_red_combat())
        blue_alive = len(self._alive_blue())

        for aid, cause in step_deaths.items():
            ac = self._by_id(aid)
            if cause == DEATH_ATTACK and ac.team == "blue":
                components["kill_event_reward"] += 10.0
            elif cause == DEATH_ATTACK and ac.team == "red" and ac.role == "combat":
                components["combat_loss_event_penalty"] -= 10.0
            elif cause == DEATH_ATTACK and ac.aircraft_id == "red_0":
                components["support_loss_event_penalty"] -= 12.0
            elif cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY) and ac.team == "red":
                components["boundary_event_penalty"] -= 10.0
        components["support_assisted_kill_reward"] += 2.0 * float(assisted)

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
                        combat_approach += 0.003 * float(np.clip((prev[1] - g.distance) / 100.0, -1.0, 1.0))
                ready = 0.02 * _attack_readiness(c.state, target.state, d_min, d_max)
                combat_readiness += ready
            threats = [b for b in self._alive_blue() if c.aircraft_id in direct[b.aircraft_id]]
            if threats:
                combat_threat += 0.015 * max(_attack_readiness(b.state, c.state, d_min, d_max) for b in threats)
            combat_boundary += 0.01 * _boundary_risk(c, limits)
        components["combat_approach_reward"] = combat_approach / 3.0
        components["combat_readiness_reward"] = combat_readiness / 3.0
        components["combat_threat_penalty"] = -combat_threat / 3.0
        components["combat_boundary_penalty"] = -combat_boundary / 3.0
        self._episode_metrics["combat_readiness_sum"] += components["combat_readiness_reward"]
        self._episode_metrics["combat_threat_sum"] += -components["combat_threat_penalty"]

        support = self._by_id("red_0")
        support_only_count = self._support_only_target_count(direct)
        alive_blue_count = max(1, blue_alive)
        coverage = 0.006 * support_only_count / alive_blue_count if support.state.alive else 0.0
        pos_score, _, _ = self._support_position_score()
        support_threat = 0.0
        if support.state.alive and self._alive_blue():
            support_threat = 0.01 * max(_attack_readiness(b.state, support.state, d_min, d_max) for b in self._alive_blue())
        support_boundary = 0.01 * _boundary_risk(support, limits) if support.state.alive else 0.0
        components["support_coverage_reward"] = coverage
        components["support_position_reward"] = 0.004 * pos_score
        components["support_threat_penalty"] = -support_threat
        components["support_boundary_penalty"] = -support_boundary

        dense = (
            components["combat_approach_reward"] + components["combat_readiness_reward"] +
            components["combat_threat_penalty"] + components["combat_boundary_penalty"] +
            components["support_coverage_reward"] + components["support_position_reward"] +
            components["support_threat_penalty"] + components["support_boundary_penalty"]
        )
        components["total_dense_reward"] = float(np.clip(dense, -0.03, 0.03))

        done, outcome, reason = self._termination()
        if done:
            if reason == "red_complete_elimination_success":
                components["mission_reward"] += 30.0
            elif reason == "red_all_combat_eliminated":
                components["mission_reward"] -= 30.0
            elif reason == "timeout":
                components["mission_reward"] -= 10.0
            elif reason == "mutual_combat_elimination":
                components["mission_reward"] -= 15.0

        event = (components["kill_event_reward"] + components["combat_loss_event_penalty"] +
                 components["support_loss_event_penalty"] + components["boundary_event_penalty"] +
                 components["support_assisted_kill_reward"])
        components["team_total_reward"] = components["mission_reward"] + event + components["total_dense_reward"]
        return float(components["team_total_reward"]), components

    def _update_support_metrics(self, direct: dict[str, set[str]], effective: dict[str, set[str]]) -> None:
        support = self._by_id("red_0")
        if not support.state.alive:
            return
        support_only_count = self._support_only_target_count(direct)
        if support_only_count > 0:
            self._episode_metrics["support_unique_detection_steps"] += 1
        shared_this_step = 0
        for cid in RED_COMBAT_IDS_4V3:
            combat = self._by_id(cid)
            if not combat.state.alive:
                continue
            for bid in BLUE_IDS_4V3:
                if bid in direct["red_0"] and bid not in direct[cid] and bid in effective[cid]:
                    self._last_support_only_shared_step[cid][bid] = self.step_count
                    shared_this_step += 1
                if bid in direct[cid] and bid in self._last_support_only_shared_step[cid]:
                    delay = self.step_count - self._last_support_only_shared_step[cid][bid]
                    key = (cid, bid)
                    if delay >= 0 and key not in self._share_to_direct_recorded:
                        self._share_to_direct_delays.append(int(delay))
                        self._episode_metrics["combat_early_acquisition_steps"] += 1
                        self._share_to_direct_recorded.add(key)
        if shared_this_step > 0:
            self._episode_metrics["support_shared_target_steps"] += 1
        pos_score, dist, rear = self._support_position_score()
        self._episode_metrics["support_to_combat_centroid_distance_sum"] += dist
        self._episode_metrics["support_rear_position_steps"] += rear
        if self._alive_blue() and max(_attack_readiness(b.state, support.state, 100.0, 1000.0) for b in self._alive_blue()) > 0.0:
            self._episode_metrics["support_threat_exposure_steps"] += 1

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
        from .rule_policy_4v3 import make_rule_policy_4v3

        direct = self._direct_visible_ids()
        effective = self._effective_visible_ids(direct)
        policy = make_rule_policy_4v3(self.config, "red")
        return policy.select_actions(self._team("red"), self._team("blue"), visibility=effective)

    def step(self, red_actions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._running:
            raise RuntimeError("environment must be reset before step")
        self.step_count += 1
        from .rule_policy_4v3 import make_rule_policy_4v3

        direct_pre = self._direct_visible_ids()
        effective_pre = self._effective_visible_ids(direct_pre)
        self._update_support_metrics(direct_pre, effective_pre)

        # Persist previous target distances for progress reward.
        for cid in RED_COMBAT_IDS_4V3:
            c = self._by_id(cid)
            target = self._nearest_effective_target(c, effective_pre) if c.state.alive else None
            if target is not None:
                self._prev_target_distance.setdefault(cid, (target.aircraft_id, compute_pairwise_geometry(c.state, target.state).distance))

        blue_policy = make_rule_policy_4v3(self.config, "blue")
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
                    killer_id = red_attackers[0]
                    last = self._last_support_only_shared_step.get(killer_id, {}).get(target_id)
                    if last is not None and self.step_count - last <= 50:
                        if target_id not in assisted_targets:
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
            "mean_share_to_direct_delay": float(np.mean(self._share_to_direct_delays)) if self._share_to_direct_delays else None,
            "mean_share_to_kill_delay": float(np.mean(self._share_to_kill_delays)) if self._share_to_kill_delays else None,
            "combat_early_acquisition_steps": int(self._episode_metrics["combat_early_acquisition_steps"]),
            "support_unique_detection_step_rate": float(self._episode_metrics["support_unique_detection_steps"] / length),
            "support_shared_target_step_rate": float(self._episode_metrics["support_shared_target_steps"] / length),
            "combat_attack_window_step_rate": float(self._episode_metrics["combat_attack_window_steps"] / length),
            "combat_readiness_mean": float(self._episode_metrics["combat_readiness_sum"] / length),
            "combat_threat_mean": float(self._episode_metrics["combat_threat_sum"] / length),
            "mean_support_to_combat_centroid_distance": float(self._episode_metrics["support_to_combat_centroid_distance_sum"] / length),
            "support_rear_position_rate": float(self._episode_metrics["support_rear_position_steps"] / length),
            "support_threat_exposure_rate": float(self._episode_metrics["support_threat_exposure_steps"] / length),
        })
        return summary
