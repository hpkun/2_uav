"""Synchronous 3v3 air combat environment with backward-compatible heterogeneity."""
from pathlib import Path
from typing import Any

import numpy as np

from .combat import SimplifiedAttackModel
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .geometry import compute_pairwise_geometry
from .integrator import RK4Integrator
from .math_utils import angle_difference
from .models import Aircraft, AircraftState, ControlCommand, TargetCommand
from .rewards import (
    madsac_segmented_reward,
    coupled_attack_advantage,
    approach_progress_reward,
    soft_boundary_risk,
    friendly_separation_risk,
    head_on_collision_risk,
    paper_segmented_local_reward,
    validate_paper_segmented_v4_config,
)
from .scenario_3v3 import ALL_IDS, BLUE_IDS, RED_IDS, Homogeneous3v3Scenario

DEATH_NONE = 0
DEATH_BOUNDARY_ALTITUDE = 1
DEATH_BOUNDARY_XY = 2
DEATH_COLLISION_FRIENDLY = 3
DEATH_COLLISION_CROSS = 4
DEATH_ATTACK = 5

DEATH_CAUSE_NAMES: dict[int, str] = {
    0: "NONE",
    1: "BOUNDARY_ALTITUDE",
    2: "BOUNDARY_XY",
    3: "COLLISION_FRIENDLY",
    4: "COLLISION_CROSS_TEAM",
    5: "ATTACK",
}

OBS_DIM = 68
GS_DIM = 48


def _normalize_obs(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, -1.0, 1.0).astype(np.float32)


def _make_episode_summary(
    episode_death_causes: dict[str, int],
    episode_attack_kills: dict[str, int],
    red_alive: int, blue_alive: int,
    outcome: str | None, reason: str | None,
    step_count: int,
    heterogeneous_episode_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build authoritative episode_summary with full death-cause breakdown."""

    def _counts(team_ids):
        atk = bdy_alt = bdy_xy = fr_col = cr_col = surv = 0
        per_aircraft = {}
        for aid in team_ids:
            c = episode_death_causes.get(aid, DEATH_NONE)
            per_aircraft[aid] = DEATH_CAUSE_NAMES.get(c, "UNKNOWN")
            if c == DEATH_NONE:
                surv += 1
            elif c == DEATH_ATTACK:
                atk += 1
            elif c == DEATH_BOUNDARY_ALTITUDE:
                bdy_alt += 1
            elif c == DEATH_BOUNDARY_XY:
                bdy_xy += 1
            elif c == DEATH_COLLISION_FRIENDLY:
                fr_col += 1
            elif c == DEATH_COLLISION_CROSS:
                cr_col += 1
        boundary_total = bdy_alt + bdy_xy
        collision_total = fr_col + cr_col
        total = surv + atk + boundary_total + collision_total
        if total != 3:
            raise RuntimeError(
                f"Death ledger mismatch: surv={surv} atk={atk} bdy_alt={bdy_alt} "
                f"bdy_xy={bdy_xy} fr_col={fr_col} cr_col={cr_col} total={total} != 3")
        return {
            "attack_deaths": atk,
            "boundary_altitude_deaths": bdy_alt,
            "boundary_xy_deaths": bdy_xy,
            "friendly_collision_deaths": fr_col,
            "cross_team_collision_deaths": cr_col,
            "boundary_deaths": boundary_total,
            "collision_deaths": collision_total,
            "survivors": surv,
        }, per_aircraft

    red_dc, red_per = _counts(RED_IDS)
    blue_dc, blue_per = _counts(BLUE_IDS)

    # Cross-validate attack kill symmetry
    if episode_attack_kills["red"] != blue_dc["attack_deaths"]:
        raise RuntimeError(
            f"red_attack_kills={episode_attack_kills['red']} != blue.attack_deaths={blue_dc['attack_deaths']}")
    if episode_attack_kills["blue"] != red_dc["attack_deaths"]:
        raise RuntimeError(
            f"blue_attack_kills={episode_attack_kills['blue']} != red.attack_deaths={red_dc['attack_deaths']}")

    red_success = (episode_attack_kills["red"] == 3 and red_alive > 0)
    blue_success = (episode_attack_kills["blue"] == 3 and blue_alive > 0)

    summary = {
        "red_attack_kills": episode_attack_kills["red"],
        "blue_attack_kills": episode_attack_kills["blue"],
        "red_survivors": red_alive,
        "blue_survivors": blue_alive,
        "red_death_causes": red_dc,
        "blue_death_causes": blue_dc,
        "per_aircraft_death_causes": {**red_per, **blue_per},
        "environment_outcome": outcome,
        "red_complete_elimination_success": red_success,
        "blue_complete_elimination_success": blue_success,
        "episode_length": step_count,
        "termination_reason": reason,
    }
    summary.update(heterogeneous_episode_metrics or {
        "red_kills_with_shared_observation": 0,
        "blue_kills_with_shared_observation": 0,
        "red_mean_support_coverage_ratio": 0.0,
        "blue_mean_support_coverage_ratio": 0.0,
        "red_support_survived": False,
        "blue_support_survived": False,
    })
    return summary


def _team_dense(own_team_alive: list[Aircraft], enemy_team_alive: list[Aircraft]) -> tuple[float, dict[str, float]]:
    """Mean madsac_segmented reward over alive own-team aircraft."""
    per_agent = {}
    for own in own_team_alive:
        if enemy_team_alive:
            nearest = min(enemy_team_alive, key=lambda e: float(
                np.linalg.norm(own.state.as_array()[:3] - e.state.as_array()[:3])))
            rd = madsac_segmented_reward(own.state, nearest.state, own.team, None, None)
            per_agent[own.aircraft_id] = rd["reward_total"]
        else:
            per_agent[own.aircraft_id] = 0.0
    mean_val = float(np.mean(list(per_agent.values()))) if per_agent else 0.0
    return mean_val, per_agent


class Homogeneous3v3AirCombatEnv:
    """3v3 synchronous air combat with opt-in functional heterogeneity."""

    def __init__(self, config_path: str | Path = "configs/homogeneous_3v3.yaml") -> None:
        self.config = load_config(config_path)
        sim, act, combat = self.config["simulation"], self.config["action"], self.config["combat"]
        self.scenario = Homogeneous3v3Scenario(self.config)
        self.dynamics = PointMassDynamics(sim["gravity"])
        self.integrator = RK4Integrator(sim["dt"])
        self.controller = TargetStateController(**act, gravity=sim["gravity"])
        self.attack_model = SimplifiedAttackModel(
            combat["attack_distance_min"], combat["attack_distance_max"],
            combat["attack_ata_max"], combat["attack_aa_max"])
        if combat.get("reward_mode") == "paper_segmented_team_v4":
            validate_paper_segmented_v4_config(
                self.config.get("reward_paper_segmented_v4", {}),
                float(combat["attack_distance_min"]),
                float(combat["attack_distance_max"]),
            )
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False
        self._episode_death_causes: dict[str, int] = {}
        self._episode_attack_kills: dict[str, int] = {}
        self._heterogeneous = bool(self.config.get("heterogeneous", {}).get("enabled", False))
        self._episode_support_coverage_sum: dict[str, float] = {}
        self._episode_support_coverage_step_count: dict[str, int] = {}
        self._episode_shared_kills: dict[str, int] = {}

    def _aircraft_by_id(self, aid: str) -> Aircraft:
        return next(a for a in self.aircraft if a.aircraft_id == aid)

    def _alive(self, team: str) -> list[Aircraft]:
        return [a for a in self.aircraft if a.team == team and a.state.alive]

    def _alive_count(self, team: str) -> int:
        return sum(1 for a in self.aircraft if a.team == team and a.state.alive)

    def _enemy_team(self, team: str) -> str:
        return "blue" if team == "red" else "red"

    def _direct_visible(self, own: Aircraft, enemy: Aircraft) -> bool:
        if not own.state.alive or not enemy.state.alive or own.team == enemy.team:
            return False
        distance = float(np.linalg.norm(own.state.as_array()[:3] - enemy.state.as_array()[:3]))
        return distance <= float(own.sensor_range)

    def _team_support(self, team: str) -> Aircraft | None:
        return next((a for a in self.aircraft if a.team == team and a.role == "support"), None)

    def _direct_visible_enemy_ids(self, own: Aircraft) -> set[str]:
        return {
            enemy.aircraft_id
            for enemy in self.aircraft
            if enemy.team != own.team and self._direct_visible(own, enemy)
        }

    def _effective_visible_enemy_ids(self, own: Aircraft) -> set[str]:
        direct = self._direct_visible_enemy_ids(own)
        if not self._heterogeneous or own.role != "combat" or not own.state.alive:
            return direct
        sharing = self.config["heterogeneous"].get("information_sharing", {})
        if not bool(sharing.get("support_to_combat", False)):
            return direct
        support = self._team_support(own.team)
        if support is None or not support.state.alive:
            return direct
        return direct | self._direct_visible_enemy_ids(support)

    def visible_enemy_ids_by_own(self, team: str) -> dict[str, set[str]]:
        """Return a fresh, read-only-by-convention visibility map for rule policies."""
        return {
            own.aircraft_id: set(self._effective_visible_enemy_ids(own))
            for own in self.aircraft
            if own.team == team
        }

    def _visibility_snapshot(self) -> dict[str, dict[str, list[str]]]:
        if not self._heterogeneous:
            return {}
        snapshot: dict[str, dict[str, list[str]]] = {}
        for own in self.aircraft:
            direct = self._direct_visible_enemy_ids(own)
            effective = self._effective_visible_enemy_ids(own)
            snapshot[own.aircraft_id] = {
                "direct": sorted(direct),
                "effective": sorted(effective),
                "shared": sorted(effective - direct),
            }
        return snapshot

    def _support_coverage(self, team: str) -> tuple[float, int]:
        if not self._heterogeneous:
            return 0.0, 0
        support = self._team_support(team)
        combats = [
            a for a in self.aircraft
            if a.team == team and a.role == "combat" and a.state.alive
        ]
        enemies = [
            a for a in self.aircraft
            if a.team == self._enemy_team(team) and a.state.alive
        ]
        if support is None or not support.state.alive or not combats or not enemies:
            return 0.0, 0
        useful = 0
        for enemy in enemies:
            if self._direct_visible(support, enemy) and any(
                not self._direct_visible(combat, enemy) for combat in combats
            ):
                useful += 1
        return float(useful / len(enemies)), useful

    def _heterogeneous_step_metrics(
        self,
        coverage: dict[str, tuple[float, int]] | None = None,
        shared_kills: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if not self._heterogeneous:
            return {}
        coverage = coverage or {team: self._support_coverage(team) for team in ("red", "blue")}
        shared_kills = shared_kills or {"red": 0, "blue": 0}
        red_support = self._team_support("red")
        blue_support = self._team_support("blue")
        return {
            "red_support_coverage_ratio": float(coverage["red"][0]),
            "blue_support_coverage_ratio": float(coverage["blue"][0]),
            "red_support_alive": bool(red_support is not None and red_support.state.alive),
            "blue_support_alive": bool(blue_support is not None and blue_support.state.alive),
            "red_useful_shared_target_count": int(coverage["red"][1]),
            "blue_useful_shared_target_count": int(coverage["blue"][1]),
            "red_kills_with_shared_observation": int(shared_kills.get("red", 0)),
            "blue_kills_with_shared_observation": int(shared_kills.get("blue", 0)),
        }

    def _heterogeneous_episode_summary_metrics(self) -> dict[str, Any]:
        if not self._heterogeneous:
            return {}
        means = {
            team: (
                self._episode_support_coverage_sum[team]
                / self._episode_support_coverage_step_count[team]
                if self._episode_support_coverage_step_count[team] > 0 else 0.0
            )
            for team in ("red", "blue")
        }
        return {
            "red_kills_with_shared_observation": int(self._episode_shared_kills["red"]),
            "blue_kills_with_shared_observation": int(self._episode_shared_kills["blue"]),
            "red_mean_support_coverage_ratio": float(means["red"]),
            "blue_mean_support_coverage_ratio": float(means["blue"]),
            "red_support_survived": bool(
                self._team_support("red") is not None and self._team_support("red").state.alive
            ),
            "blue_support_survived": bool(
                self._team_support("blue") is not None and self._team_support("blue").state.alive
            ),
        }

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        self._running = True
        self._episode_death_causes = {aid: DEATH_NONE for aid in ALL_IDS}
        self._episode_attack_kills = {"red": 0, "blue": 0}
        self._episode_support_coverage_sum = {"red": 0.0, "blue": 0.0}
        self._episode_support_coverage_step_count = {"red": 0, "blue": 0}
        self._episode_shared_kills = {"red": 0, "blue": 0}
        observations = self._all_observations()
        heterogeneous_metrics = self._heterogeneous_step_metrics()
        info = {"step_count": 0, "scenario_name": self.scenario.scenario_name,
                "termination_reason": None, "outcome": None,
                "red_alive_count": 3, "blue_alive_count": 3,
                "attacks": {}, "death_causes": {}, "attack_kills": {"red": 0, "blue": 0},
                "boundary_deaths": {"red": 0, "blue": 0},
                "collision_deaths": {"red": 0, "blue": 0},
                "red_complete_elimination_success": False,
                "blue_complete_elimination_success": False,
                "control_diagnostics": {}, "reward_components": {},
                "reward_targets": {},
                "nearest_enemy_geometry": {}, "episode_summary": None,
                "heterogeneous_metrics": heterogeneous_metrics,
                "visibility": self._visibility_snapshot(),
                "global_state": self.global_state()}
        return observations, info

    def step(self, actions: dict[str, np.ndarray]) -> tuple:
        if not self._running:
            raise RuntimeError("reset() must be called before step()")
        alive = [a for a in self.aircraft if a.state.alive]
        missing = {a.aircraft_id for a in alive} - set(actions.keys())
        if missing:
            raise KeyError(f"missing actions for: {sorted(missing)}")

        # 1-3. Controls
        old_states = {a.aircraft_id: a.state.copy() for a in alive}
        targets: dict[str, TargetCommand] = {}
        controls: dict[str, ControlCommand] = {}
        control_diagnostics: dict[str, dict] = {}
        for aircraft in alive:
            aid = aircraft.aircraft_id
            tgt, ctrl = self.controller.control_from_action(old_states[aid], actions[aid], aircraft.spec)
            targets[aid], controls[aid] = tgt, ctrl
            diag = self.controller.diagnostics(old_states[aid], tgt, ctrl, aircraft.spec, actions[aid])
            deriv = self.dynamics.derivatives(old_states[aid], ctrl)
            aa, apr, ayr = map(float, deriv[3:6])
            diag.update({"actual_acceleration": aa, "actual_pitch_rate": apr, "actual_yaw_rate": ayr,
                         "acceleration_tracking_error": diag["clipped_acceleration"] - aa,
                         "pitch_rate_tracking_error": diag["clipped_pitch_rate"] - apr,
                         "yaw_rate_tracking_error": diag["clipped_yaw_rate"] - ayr})
            for lbl, ek in (("acceleration_tracking_absolute_error", "acceleration_tracking_error"),
                            ("pitch_rate_tracking_absolute_error", "pitch_rate_tracking_error"),
                            ("yaw_rate_tracking_absolute_error", "yaw_rate_tracking_error")):
                diag[lbl] = abs(diag[ek])
            clipped = np.clip(np.asarray(actions[aid], dtype=float), -1.0, 1.0)
            diag.update({"action_yaw": float(clipped[0]), "action_pitch": float(clipped[1]),
                         "action_speed": float(clipped[2]),
                         "delta_yaw": float(angle_difference(tgt.desired_psi, old_states[aid].psi)),
                         "delta_pitch": float(tgt.desired_theta - old_states[aid].theta),
                         "delta_speed": float(tgt.desired_v - old_states[aid].v)})
            control_diagnostics[aid] = diag

        # 4-5. RK4 integrate all alive
        new_states = {}
        for aircraft in alive:
            a_id = aircraft.aircraft_id
            new_states[a_id] = self.integrator.step(
                old_states[a_id], controls[a_id], self.dynamics, aircraft.spec)
        for aircraft in alive:
            aircraft.state = new_states[aircraft.aircraft_id]
        self.step_count += 1

        # 6-8. Boundary + Collision
        limits = self.config["battlefield"]
        step_death_causes: dict[str, int] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive: continue
            s = aircraft.state
            if not (limits["altitude_min"] <= s.altitude <= limits["altitude_max"]):
                s.alive = False; step_death_causes[aircraft.aircraft_id] = DEATH_BOUNDARY_ALTITUDE
            elif abs(s.x) > limits["x_limit"] or abs(s.y) > limits["y_limit"]:
                s.alive = False; step_death_causes[aircraft.aircraft_id] = DEATH_BOUNDARY_XY

        collision_pairs: list[tuple[str, str]] = []
        alive_now = [a for a in self.aircraft if a.state.alive]
        for i in range(len(alive_now)):
            for j in range(i + 1, len(alive_now)):
                a1, a2 = alive_now[i], alive_now[j]
                if float(np.linalg.norm(a1.state.as_array()[:3] - a2.state.as_array()[:3])) <= limits["collision_distance"]:
                    collision_pairs.append((a1.aircraft_id, a2.aircraft_id))
        for aid1, aid2 in collision_pairs:
            a1, a2 = self._aircraft_by_id(aid1), self._aircraft_by_id(aid2)
            if a1.state.alive:
                a1.state.alive = False
                if aid1 not in step_death_causes:
                    step_death_causes[aid1] = DEATH_COLLISION_FRIENDLY if a1.team == a2.team else DEATH_COLLISION_CROSS
            if a2.state.alive:
                a2.state.alive = False
                if aid2 not in step_death_causes:
                    step_death_causes[aid2] = DEATH_COLLISION_FRIENDLY if a1.team == a2.team else DEATH_COLLISION_CROSS

        reward_mode = self.config["combat"].get("reward_mode", "madsac_segmented")
        v7_pre_attack_dense_parts: dict[str, dict[str, float]] | None = None
        v7_pre_attack_reward_targets: dict[str, str | None] | None = None
        if reward_mode == "paper_segmented_team_v4":
            (
                v7_pre_attack_dense_parts,
                v7_pre_attack_reward_targets,
            ) = self._capture_paper_segmented_v4_pre_attack()

        # 9. Nearest-enemy geometry (post-integration, pre-attack, for still-alive)
        nearest_enemy_geom: dict[str, dict[str, Any]] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                nearest_enemy_geom[aircraft.aircraft_id] = {"target_id": None, "distance": 0.0, "ata": 0.0, "aa": 0.0}
                continue
            enemy_team = "blue" if aircraft.team == "red" else "red"
            alive_enemies = [a for a in self.aircraft if a.team == enemy_team and a.state.alive]
            if not alive_enemies:
                nearest_enemy_geom[aircraft.aircraft_id] = {"target_id": None, "distance": 0.0, "ata": 0.0, "aa": 0.0}
                continue
            nearest = min(alive_enemies, key=lambda e: float(
                np.linalg.norm(aircraft.state.as_array()[:3] - e.state.as_array()[:3])))
            geo = compute_pairwise_geometry(aircraft.state, nearest.state)
            nearest_enemy_geom[aircraft.aircraft_id] = {
                "target_id": nearest.aircraft_id, "distance": float(geo.distance),
                "ata": float(geo.ata), "aa": float(geo.aa)}

        coverage = {team: self._support_coverage(team) for team in ("red", "blue")}
        for team in ("red", "blue"):
            self._episode_support_coverage_sum[team] += float(coverage[team][0])
            self._episode_support_coverage_step_count[team] += 1

        direct_visible_ids = {
            aircraft.aircraft_id: self._direct_visible_enemy_ids(aircraft)
            for aircraft in self.aircraft
        }
        effective_visible_ids = {
            aircraft.aircraft_id: self._effective_visible_enemy_ids(aircraft)
            for aircraft in self.aircraft
        }
        shared_attackers: dict[str, set[str]] = {"red": set(), "blue": set()}

        # 10-11. Attack intents
        attack_intents: dict[str, str | None] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive or not aircraft.can_attack:
                attack_intents[aircraft.aircraft_id] = None; continue
            enemy_team = "blue" if aircraft.team == "red" else "red"
            alive_enemies = [a for a in self.aircraft if a.team == enemy_team and a.state.alive]
            if not alive_enemies:
                attack_intents[aircraft.aircraft_id] = None; continue
            visible = effective_visible_ids[aircraft.aircraft_id]
            attackable = [
                e for e in alive_enemies
                if e.aircraft_id in visible and self.attack_model.can_attack(aircraft.state, e.state)
            ]
            if attackable:
                best = min(attackable, key=lambda e: (
                    float(np.linalg.norm(aircraft.state.as_array()[:3] - e.state.as_array()[:3])), e.aircraft_id))
                attack_intents[aircraft.aircraft_id] = best.aircraft_id
                # This diagnostic counts same-step kills where the attacker
                # could not directly sense the target but received it from a
                # live support aircraft. With the default combat sensor range
                # (3000 m) above attack max range (1000 m), it is structurally
                # expected to remain zero unless tests/configs shrink sensors.
                if best.aircraft_id not in direct_visible_ids[aircraft.aircraft_id]:
                    shared_attackers[aircraft.team].add(best.aircraft_id)
            else:
                attack_intents[aircraft.aircraft_id] = None

        attackers_by_target: dict[str, list[str]] = {}
        for atk_id, tgt_id in attack_intents.items():
            if tgt_id is not None:
                attackers_by_target.setdefault(tgt_id, []).append(atk_id)

        attack_kills = {"red": 0, "blue": 0}
        shared_kills = {"red": 0, "blue": 0}
        for tgt_id in attackers_by_target:
            tgt = self._aircraft_by_id(tgt_id)
            if tgt.state.alive:
                tgt.state.alive = False
                if tgt_id not in step_death_causes:
                    step_death_causes[tgt_id] = DEATH_ATTACK
                if tgt.team == "blue":
                    attack_kills["red"] += 1
                    if tgt_id in shared_attackers["red"]:
                        shared_kills["red"] += 1
                else:
                    attack_kills["blue"] += 1
                    if tgt_id in shared_attackers["blue"]:
                        shared_kills["blue"] += 1

        # Accumulate episode
        for aid, cause in step_death_causes.items():
            if self._episode_death_causes.get(aid, DEATH_NONE) == DEATH_NONE:
                self._episode_death_causes[aid] = cause
        self._episode_attack_kills["red"] += attack_kills["red"]
        self._episode_attack_kills["blue"] += attack_kills["blue"]
        self._episode_shared_kills["red"] += shared_kills["red"]
        self._episode_shared_kills["blue"] += shared_kills["blue"]

        # 12. Team elimination
        red_alive = self._alive_count("red")
        blue_alive = self._alive_count("blue")
        terminated = red_alive == 0 or blue_alive == 0
        truncated = not terminated and self.step_count >= self.config["simulation"]["max_steps"]

        outcome: str | None = None
        reason: str | None = None
        if red_alive == 0 and blue_alive == 0:
            outcome, reason = "draw", "mutual_elimination"
        elif red_alive == 0:
            outcome, reason = "blue", "blue_elimination"
        elif blue_alive == 0:
            outcome, reason = "red", "red_elimination"
        elif truncated:
            reason = "max_steps"
            if self.config["combat"].get("timeout_outcome_mode") == "red_failure_blue_win":
                outcome = "blue"
            else:
                outcome = "draw"
        if terminated or truncated:
            self._running = False

        # --- Per-step death tallies (unconditional, needed for info) ---
        red_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red" and c == DEATH_ATTACK)
        red_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        red_bdy_alt_losses = sum(1 for a_id, c in step_death_causes.items()
                                 if self._aircraft_by_id(a_id).team == "red"
                                 and c == DEATH_BOUNDARY_ALTITUDE)
        red_bdy_xy_losses = sum(1 for a_id, c in step_death_causes.items()
                                if self._aircraft_by_id(a_id).team == "red"
                                and c == DEATH_BOUNDARY_XY)
        red_col_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
        blue_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue" and c == DEATH_ATTACK)
        blue_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        blue_bdy_alt_losses = sum(1 for a_id, c in step_death_causes.items()
                                  if self._aircraft_by_id(a_id).team == "blue"
                                  and c == DEATH_BOUNDARY_ALTITUDE)
        blue_bdy_xy_losses = sum(1 for a_id, c in step_death_causes.items()
                                 if self._aircraft_by_id(a_id).team == "blue"
                                 and c == DEATH_BOUNDARY_XY)
        blue_col_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
        bdy_d = {"red": red_bdy_losses, "blue": blue_bdy_losses}
        bdy_alt_d = {"red": red_bdy_alt_losses, "blue": blue_bdy_alt_losses}
        bdy_xy_d = {"red": red_bdy_xy_losses, "blue": blue_bdy_xy_losses}
        col_d = {"red": red_col_losses, "blue": blue_col_losses}

        # 13. Rewards
        reward_targets: dict[str, str | None] = {}
        if reward_mode == "paper_coupled_team_v2":
            rewards, reward_components = self._compute_v2_rewards(
                old_states, attack_kills, step_death_causes, terminated, truncated,
                outcome, reason, red_alive, blue_alive)
        elif reward_mode == "target_consistent_team_v3":
            rewards, reward_components, reward_targets = self._compute_v3_rewards(
                old_states, attack_kills, step_death_causes, terminated, truncated,
                outcome, reason, red_alive, blue_alive)
        elif reward_mode == "paper_segmented_team_v4":
            rewards, reward_components, reward_targets = self._compute_paper_segmented_v4_rewards(
                attack_kills, step_death_causes, terminated, truncated,
                outcome, reason, red_alive, blue_alive,
                v7_pre_attack_dense_parts, v7_pre_attack_reward_targets)
        elif reward_mode == "functional_heterogeneous_team_v1":
            rewards, reward_components, reward_targets = self._compute_heterogeneous_rewards(
                old_states, attack_kills, step_death_causes, terminated, truncated,
                outcome, reason, red_alive, blue_alive, coverage)
        else:
            # Legacy madsac_segmented (kept for baseline comparison)
            red_dense, _ = _team_dense(self._alive("red"), self._alive("blue"))
            blue_dense, _ = _team_dense(self._alive("blue"), self._alive("red"))
            T = self.config["combat"]["terminal_reward"]
            red_total = red_dense + T * attack_kills["red"] - T * red_atk_losses - T * red_bdy_losses - T * red_col_losses
            blue_total = blue_dense + T * attack_kills["blue"] - T * blue_atk_losses - T * blue_bdy_losses - T * blue_col_losses
            rewards = {}
            for aid in RED_IDS: rewards[aid] = float(red_total)
            for aid in BLUE_IDS: rewards[aid] = float(blue_total)
            reward_components = {
                "red_team_dense_reward": red_dense, "blue_team_dense_reward": blue_dense,
                "red_attack_kill_reward": T * attack_kills["red"],
                "blue_attack_kill_reward": T * attack_kills["blue"],
                "red_attack_loss_penalty": -T * red_atk_losses,
                "blue_attack_loss_penalty": -T * blue_atk_losses,
                "red_boundary_penalty": -T * red_bdy_losses,
                "blue_boundary_penalty": -T * blue_bdy_losses,
                "red_collision_penalty": -T * red_col_losses,
                "blue_collision_penalty": -T * blue_col_losses,
                "red_team_total_reward": red_total, "blue_team_total_reward": blue_total,
            }

        observations = self._all_observations()

        episode_summary = None
        if terminated or truncated:
            episode_summary = _make_episode_summary(
                self._episode_death_causes, self._episode_attack_kills,
                red_alive, blue_alive, outcome, reason, self.step_count,
                self._heterogeneous_episode_summary_metrics() if self._heterogeneous else None)

        red_success = episode_summary["red_complete_elimination_success"] if episode_summary else False
        blue_success = episode_summary["blue_complete_elimination_success"] if episode_summary else False

        info = {
            "step_count": self.step_count, "scenario_name": self.scenario.scenario_name,
            "termination_reason": reason, "outcome": outcome,
            "red_alive_count": red_alive, "blue_alive_count": blue_alive,
            "attacks": attack_intents, "death_causes": step_death_causes,
            "attack_kills": attack_kills, "boundary_deaths": bdy_d, "collision_deaths": col_d,
            "boundary_altitude_deaths": bdy_alt_d,
            "boundary_xy_deaths": bdy_xy_d,
            "red_complete_elimination_success": red_success,
            "blue_complete_elimination_success": blue_success,
            "red_survivors": red_alive, "blue_survivors": blue_alive,
            "collision_pairs": collision_pairs, "attackers_by_target": attackers_by_target,
            "control_diagnostics": control_diagnostics, "reward_components": reward_components,
            "reward_targets": reward_targets,
            "nearest_enemy_geometry": nearest_enemy_geom,
            "heterogeneous_metrics": self._heterogeneous_step_metrics(coverage, shared_kills),
            "visibility": self._visibility_snapshot(),
            "blue_targets": {}, "global_state": self.global_state(),
            "episode_summary": episode_summary,
        }
        return observations, rewards, terminated, truncated, info

    # -- paper_coupled_team_v2 rewards -----------------------------------

    def _compute_v2_rewards(
        self, old_states: dict[str, AircraftState], attack_kills: dict[str, int],
        step_death_causes: dict[str, int], terminated: bool, truncated: bool,
        outcome: str | None, reason: str | None, red_alive: int, blue_alive: int,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        cfg = self.config.get("reward_v2", {})
        if not cfg:
            cfg = {
                "team_size": 3, "approach_weight": 0.01, "approach_distance_threshold": 1000.0,
                "approach_distance_normalizer": 30.0, "attack_advantage_weight": 0.04,
                "threat_weight": 0.05, "preferred_distance": 600.0, "distance_sigma": 450.0,
                "ata_sigma": 0.5235987755982988, "aa_sigma": 1.0471975511965976,
                "boundary_weight": 0.05, "horizontal_soft_ratio": 0.8, "altitude_soft_margin": 750.0,
                "friendly_separation_weight": 0.02, "friendly_safe_distance": 200.0,
                "head_on_risk_weight": 0.03, "head_on_distance": 300.0,
                "head_on_angle": 0.5235987755982988, "time_penalty": 0.001,
                "kill_reward": 20.0, "attack_death_penalty": 20.0,
                "boundary_death_penalty": 30.0, "collision_death_penalty": 25.0,
                "complete_elimination_bonus": 20.0, "team_eliminated_penalty": 20.0,
                "mutual_elimination_penalty": 10.0, "max_steps_penalty": 5.0,
                "dense_reward_min": -0.15, "dense_reward_max": 0.05,
            }
        bf = self.config["battlefield"]
        team_size = int(cfg["team_size"])
        alive_reds = [a for a in self.aircraft if a.team == "red" and a.state.alive]
        alive_blues = [a for a in self.aircraft if a.team == "blue" and a.state.alive]

        # --- Per-red-agent local dense ---
        red_approach = 0.0; red_attack = 0.0; red_threat = 0.0
        red_boundary = 0.0; red_friendly = 0.0; red_head_on = 0.0

        for red_ac in self.aircraft:
            if red_ac.team != "red":
                continue
            if not red_ac.state.alive:
                continue  # dead red contributes 0

            aid = red_ac.aircraft_id
            red_state = red_ac.state
            prev_red = old_states.get(aid, red_state)

            # Approach: signed max over alive blues; preserves negative progress.
            approach_values = []
            for blue_ac in alive_blues:
                prev_blue = old_states.get(blue_ac.aircraft_id, blue_ac.state)
                approach_values.append(approach_progress_reward(
                    prev_red, red_state, prev_blue, blue_ac.state,
                    cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"]))
            app = max(approach_values) if approach_values else 0.0
            red_approach += app

            # Attack advantage: max over alive blues
            atk_val = 0.0
            for blue_ac in alive_blues:
                val = coupled_attack_advantage(
                    red_state, blue_ac.state, cfg["preferred_distance"],
                    cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
                if val > atk_val: atk_val = val
            red_attack += atk_val

            # Threat: max over alive blues (blue attacking red)
            thr_val = 0.0
            for blue_ac in alive_blues:
                val = coupled_attack_advantage(
                    blue_ac.state, red_state, cfg["preferred_distance"],
                    cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
                if val > thr_val: thr_val = val
            red_threat += thr_val

            # Boundary
            bdr = soft_boundary_risk(
                red_state, bf["x_limit"], bf["y_limit"],
                bf["altitude_min"], bf["altitude_max"],
                cfg["horizontal_soft_ratio"], cfg["altitude_soft_margin"])
            red_boundary += bdr["total_risk"]

            # Friendly separation
            teammates = [a.state for a in self.aircraft if a.team == "red" and a.aircraft_id != aid]
            red_friendly += friendly_separation_risk(
                red_state, teammates, cfg["friendly_safe_distance"], bf["collision_distance"])

            # Head-on risk
            ho = 0.0
            for blue_ac in alive_blues:
                val = head_on_collision_risk(
                    red_state, blue_ac.state, cfg["head_on_distance"], cfg["head_on_angle"])
                if val > ho: ho = val
            red_head_on += ho

        # Team dense: sum of per-agent / fixed team_size (3), not alive count
        red_dense_raw = (
            cfg["approach_weight"] * red_approach
            + cfg["attack_advantage_weight"] * red_attack
            - cfg["threat_weight"] * red_threat
            - cfg["boundary_weight"] * red_boundary
            - cfg["friendly_separation_weight"] * red_friendly
            - cfg["head_on_risk_weight"] * red_head_on
        ) / team_size - cfg["time_penalty"]

        red_dense = float(np.clip(red_dense_raw, cfg["dense_reward_min"], cfg["dense_reward_max"]))

        # Blue symmetric dense (diagnostic only)
        blue_approach = 0.0; blue_attack = 0.0; blue_threat = 0.0
        blue_boundary = 0.0; blue_friendly = 0.0; blue_head_on = 0.0
        for blue_ac in self.aircraft:
            if blue_ac.team != "blue" or not blue_ac.state.alive:
                continue
            aid = blue_ac.aircraft_id; bs = blue_ac.state; pb = old_states.get(aid, bs)
            approach_values = []
            for red_ac in alive_reds:
                pr = old_states.get(red_ac.aircraft_id, red_ac.state)
                approach_values.append(approach_progress_reward(
                    pb, bs, pr, red_ac.state,
                    cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"]))
            app = max(approach_values) if approach_values else 0.0
            blue_approach += app
            atk_val = 0.0
            for red_ac in alive_reds:
                val = coupled_attack_advantage(bs, red_ac.state, cfg["preferred_distance"],
                                               cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
                if val > atk_val: atk_val = val
            blue_attack += atk_val
            thr_val = 0.0
            for red_ac in alive_reds:
                val = coupled_attack_advantage(red_ac.state, bs, cfg["preferred_distance"],
                                               cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
                if val > thr_val: thr_val = val
            blue_threat += thr_val
            bdr = soft_boundary_risk(bs, bf["x_limit"], bf["y_limit"], bf["altitude_min"], bf["altitude_max"],
                                     cfg["horizontal_soft_ratio"], cfg["altitude_soft_margin"])
            blue_boundary += bdr["total_risk"]
            mates = [a.state for a in self.aircraft if a.team == "blue" and a.aircraft_id != aid]
            blue_friendly += friendly_separation_risk(bs, mates, cfg["friendly_safe_distance"], bf["collision_distance"])
            ho = 0.0
            for red_ac in alive_reds:
                val = head_on_collision_risk(red_ac.state, bs, cfg["head_on_distance"], cfg["head_on_angle"])
                if val > ho: ho = val
            blue_head_on += ho
        blue_dense_raw = (
            cfg["approach_weight"] * blue_approach
            + cfg["attack_advantage_weight"] * blue_attack
            - cfg["threat_weight"] * blue_threat
            - cfg["boundary_weight"] * blue_boundary
            - cfg["friendly_separation_weight"] * blue_friendly
            - cfg["head_on_risk_weight"] * blue_head_on
        ) / team_size - cfg["time_penalty"]
        blue_dense = float(np.clip(blue_dense_raw, cfg["dense_reward_min"], cfg["dense_reward_max"]))

        # --- Step event tally ---
        red_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red" and c == DEATH_ATTACK)
        red_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        red_col_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
        blue_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue" and c == DEATH_ATTACK)
        blue_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        blue_col_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))

        red_event = (
            cfg["kill_reward"] * attack_kills["red"]
            - cfg["attack_death_penalty"] * red_atk_losses
            - cfg["boundary_death_penalty"] * red_bdy_losses
            - cfg["collision_death_penalty"] * red_col_losses
        )
        blue_event = (
            cfg["kill_reward"] * attack_kills["blue"]
            - cfg["attack_death_penalty"] * blue_atk_losses
            - cfg["boundary_death_penalty"] * blue_bdy_losses
            - cfg["collision_death_penalty"] * blue_col_losses
        )

        # --- Terminal reward ---
        red_terminal = 0.0; blue_terminal = 0.0
        if terminated or truncated:
            red_succ = (self._episode_attack_kills["red"] == 3 and red_alive > 0)
            blue_succ = (self._episode_attack_kills["blue"] == 3 and blue_alive > 0)
            if red_succ:
                red_terminal += cfg["complete_elimination_bonus"]
            if blue_succ:
                blue_terminal += cfg["complete_elimination_bonus"]
            if reason == "max_steps":
                red_terminal -= cfg["max_steps_penalty"]
                blue_terminal -= cfg["max_steps_penalty"]
            elif red_alive == 0 and blue_alive == 0:
                red_terminal -= cfg["mutual_elimination_penalty"]
                blue_terminal -= cfg["mutual_elimination_penalty"]
            elif red_alive == 0:
                red_terminal -= cfg["team_eliminated_penalty"]
            elif blue_alive == 0:
                blue_terminal -= cfg["team_eliminated_penalty"]

        red_total = red_dense + red_event + red_terminal
        blue_total = blue_dense + blue_event + blue_terminal

        rewards = {}
        for aid in RED_IDS: rewards[aid] = float(red_total)
        for aid in BLUE_IDS: rewards[aid] = float(blue_total)

        reward_components = {
            "red_approach_reward": cfg["approach_weight"] * red_approach / team_size,
            "red_attack_advantage_reward": cfg["attack_advantage_weight"] * red_attack / team_size,
            "red_threat_penalty": cfg["threat_weight"] * red_threat / team_size,
            "red_soft_boundary_penalty": cfg["boundary_weight"] * red_boundary / team_size,
            "red_friendly_separation_penalty": cfg["friendly_separation_weight"] * red_friendly / team_size,
            "red_head_on_risk_penalty": cfg["head_on_risk_weight"] * red_head_on / team_size,
            "red_time_penalty": cfg["time_penalty"],
            "red_dense_reward": red_dense,
            "red_kill_reward": cfg["kill_reward"] * attack_kills["red"],
            "red_attack_death_penalty": cfg["attack_death_penalty"] * red_atk_losses,
            "red_boundary_death_penalty": cfg["boundary_death_penalty"] * red_bdy_losses,
            "red_collision_death_penalty": cfg["collision_death_penalty"] * red_col_losses,
            "red_terminal_reward": red_terminal,
            "red_team_total_reward": red_total,
            # Blue symmetric
            "blue_approach_reward": cfg["approach_weight"] * blue_approach / team_size,
            "blue_attack_advantage_reward": cfg["attack_advantage_weight"] * blue_attack / team_size,
            "blue_threat_penalty": cfg["threat_weight"] * blue_threat / team_size,
            "blue_soft_boundary_penalty": cfg["boundary_weight"] * blue_boundary / team_size,
            "blue_friendly_separation_penalty": cfg["friendly_separation_weight"] * blue_friendly / team_size,
            "blue_head_on_risk_penalty": cfg["head_on_risk_weight"] * blue_head_on / team_size,
            "blue_time_penalty": cfg["time_penalty"],
            "blue_dense_reward": blue_dense,
            "blue_kill_reward": cfg["kill_reward"] * attack_kills["blue"],
            "blue_attack_death_penalty": cfg["attack_death_penalty"] * blue_atk_losses,
            "blue_boundary_death_penalty": cfg["boundary_death_penalty"] * blue_bdy_losses,
            "blue_collision_death_penalty": cfg["collision_death_penalty"] * blue_col_losses,
            "blue_terminal_reward": blue_terminal,
            "blue_team_total_reward": blue_total,
        }
        return rewards, reward_components

    # -- target_consistent_team_v3 rewards -------------------------------

    def _nearest_alive_enemy(self, own: Aircraft, enemies: list[Aircraft]) -> Aircraft | None:
        alive_enemies = [enemy for enemy in enemies if enemy.state.alive]
        if not alive_enemies:
            return None
        return min(alive_enemies, key=lambda enemy: (
            float(np.linalg.norm(own.state.as_array()[:3] - enemy.state.as_array()[:3])),
            enemy.aircraft_id,
        ))

    def _compute_target_consistent_dense(
        self, team: str, old_states: dict[str, AircraftState], cfg: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, str | None]]:
        bf = self.config["battlefield"]
        own_team = [a for a in self.aircraft if a.team == team]
        enemy_team_name = "blue" if team == "red" else "red"
        enemies = [a for a in self.aircraft if a.team == enemy_team_name]
        team_size = int(cfg["team_size"])

        sums = {"approach": 0.0, "attack": 0.0, "threat": 0.0, "boundary": 0.0}
        targets: dict[str, str | None] = {}

        for own in own_team:
            if not own.state.alive:
                targets[own.aircraft_id] = None
                continue

            target = self._nearest_alive_enemy(own, enemies)
            targets[own.aircraft_id] = target.aircraft_id if target is not None else None

            boundary = soft_boundary_risk(
                own.state, bf["x_limit"], bf["y_limit"],
                bf["altitude_min"], bf["altitude_max"],
                cfg["horizontal_soft_ratio"], cfg["altitude_soft_margin"])
            sums["boundary"] += boundary["total_risk"]

            if target is None:
                continue

            prev_own = old_states.get(own.aircraft_id, own.state)
            prev_target = old_states.get(target.aircraft_id, target.state)
            sums["approach"] += approach_progress_reward(
                prev_own, own.state, prev_target, target.state,
                cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"])
            sums["attack"] += coupled_attack_advantage(
                own.state, target.state, cfg["preferred_distance"],
                cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
            sums["threat"] += coupled_attack_advantage(
                target.state, own.state, cfg["preferred_distance"],
                cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])

        dense_raw = (
            cfg["approach_weight"] * sums["approach"]
            + cfg["attack_advantage_weight"] * sums["attack"]
            - cfg["threat_weight"] * sums["threat"]
            - cfg["boundary_weight"] * sums["boundary"]
        ) / team_size - cfg["time_penalty"]
        dense = float(np.clip(dense_raw, cfg["dense_reward_min"], cfg["dense_reward_max"]))
        components = {
            "approach_reward": cfg["approach_weight"] * sums["approach"] / team_size,
            "attack_advantage_reward": cfg["attack_advantage_weight"] * sums["attack"] / team_size,
            "threat_penalty": cfg["threat_weight"] * sums["threat"] / team_size,
            "soft_boundary_penalty": cfg["boundary_weight"] * sums["boundary"] / team_size,
            "friendly_separation_penalty": 0.0,
            "head_on_risk_penalty": 0.0,
            "time_penalty": cfg["time_penalty"],
            "dense_reward": dense,
        }
        return components, targets

    def _compute_v3_rewards(
        self, old_states: dict[str, AircraftState], attack_kills: dict[str, int],
        step_death_causes: dict[str, int], terminated: bool, truncated: bool,
        outcome: str | None, reason: str | None, red_alive: int, blue_alive: int,
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, str | None]]:
        cfg = self.config.get("reward_v3", {})
        if not cfg:
            raise KeyError("target_consistent_team_v3 requires reward_v3 config")

        red_dense_parts, red_targets = self._compute_target_consistent_dense("red", old_states, cfg)
        blue_dense_parts, blue_targets = self._compute_target_consistent_dense("blue", old_states, cfg)

        red_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red" and c == DEATH_ATTACK)
        red_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        red_col_losses = sum(1 for a_id, c in step_death_causes.items()
                             if self._aircraft_by_id(a_id).team == "red"
                             and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
        blue_atk_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue" and c == DEATH_ATTACK)
        blue_bdy_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        blue_col_losses = sum(1 for a_id, c in step_death_causes.items()
                              if self._aircraft_by_id(a_id).team == "blue"
                              and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))

        red_event = (
            cfg["kill_reward"] * attack_kills["red"]
            - cfg["attack_death_penalty"] * red_atk_losses
            - cfg["boundary_death_penalty"] * red_bdy_losses
            - cfg["collision_death_penalty"] * red_col_losses
        )
        blue_event = (
            cfg["kill_reward"] * attack_kills["blue"]
            - cfg["attack_death_penalty"] * blue_atk_losses
            - cfg["boundary_death_penalty"] * blue_bdy_losses
            - cfg["collision_death_penalty"] * blue_col_losses
        )

        red_terminal = 0.0
        blue_terminal = 0.0
        if terminated or truncated:
            red_succ = (self._episode_attack_kills["red"] == 3 and red_alive > 0)
            blue_succ = (self._episode_attack_kills["blue"] == 3 and blue_alive > 0)
            if red_succ:
                red_terminal += cfg["complete_elimination_bonus"]
            if blue_succ:
                blue_terminal += cfg["complete_elimination_bonus"]
            if reason == "max_steps":
                red_terminal -= cfg["max_steps_red_failure_penalty"]
                blue_terminal += cfg["max_steps_blue_success_bonus"]
            elif red_alive == 0 and blue_alive == 0:
                red_terminal -= cfg["mutual_elimination_penalty"]
                blue_terminal -= cfg["mutual_elimination_penalty"]
            elif red_alive == 0:
                red_terminal -= cfg["team_eliminated_penalty"]
            elif blue_alive == 0:
                blue_terminal -= cfg["team_eliminated_penalty"]

        red_total = red_dense_parts["dense_reward"] + red_event + red_terminal
        blue_total = blue_dense_parts["dense_reward"] + blue_event + blue_terminal

        rewards = {aid: float(red_total) for aid in RED_IDS}
        rewards.update({aid: float(blue_total) for aid in BLUE_IDS})

        reward_components = {
            "red_approach_reward": red_dense_parts["approach_reward"],
            "red_attack_advantage_reward": red_dense_parts["attack_advantage_reward"],
            "red_threat_penalty": red_dense_parts["threat_penalty"],
            "red_soft_boundary_penalty": red_dense_parts["soft_boundary_penalty"],
            "red_friendly_separation_penalty": 0.0,
            "red_head_on_risk_penalty": 0.0,
            "red_time_penalty": red_dense_parts["time_penalty"],
            "red_dense_reward": red_dense_parts["dense_reward"],
            "red_kill_reward": cfg["kill_reward"] * attack_kills["red"],
            "red_attack_death_penalty": cfg["attack_death_penalty"] * red_atk_losses,
            "red_boundary_death_penalty": cfg["boundary_death_penalty"] * red_bdy_losses,
            "red_collision_death_penalty": cfg["collision_death_penalty"] * red_col_losses,
            "red_terminal_reward": red_terminal,
            "red_team_total_reward": red_total,
            "blue_approach_reward": blue_dense_parts["approach_reward"],
            "blue_attack_advantage_reward": blue_dense_parts["attack_advantage_reward"],
            "blue_threat_penalty": blue_dense_parts["threat_penalty"],
            "blue_soft_boundary_penalty": blue_dense_parts["soft_boundary_penalty"],
            "blue_friendly_separation_penalty": 0.0,
            "blue_head_on_risk_penalty": 0.0,
            "blue_time_penalty": blue_dense_parts["time_penalty"],
            "blue_dense_reward": blue_dense_parts["dense_reward"],
            "blue_kill_reward": cfg["kill_reward"] * attack_kills["blue"],
            "blue_attack_death_penalty": cfg["attack_death_penalty"] * blue_atk_losses,
            "blue_boundary_death_penalty": cfg["boundary_death_penalty"] * blue_bdy_losses,
            "blue_collision_death_penalty": cfg["collision_death_penalty"] * blue_col_losses,
            "blue_terminal_reward": blue_terminal,
            "blue_team_total_reward": blue_total,
        }
        if not np.all(np.isfinite(list(reward_components.values()) + list(rewards.values()))):
            raise FloatingPointError("non-finite target_consistent_team_v3 reward")
        reward_targets = {**red_targets, **blue_targets}
        return rewards, reward_components, reward_targets

    # -- paper_segmented_team_v4 rewards --------------------------------

    def _capture_paper_segmented_v4_pre_attack(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, str | None]]:
        """Capture v7 dense terms after motion/boundary/collision and before attack.

        This snapshot is intentionally local to the current step. Aircraft already
        removed by boundary or collision do not contribute; aircraft killed later
        by attack may still contribute their pre-attack dense terms for this step.
        """
        cfg = self.config.get("reward_paper_segmented_v4", {})
        if not cfg:
            raise KeyError("paper_segmented_team_v4 requires reward_paper_segmented_v4 config")
        red_dense_parts, red_targets = self._compute_paper_segmented_v4_dense("red", cfg)
        blue_dense_parts, blue_targets = self._compute_paper_segmented_v4_dense("blue", cfg)
        return (
            {"red": red_dense_parts, "blue": blue_dense_parts},
            {**red_targets, **blue_targets},
        )

    def _compute_paper_segmented_v4_dense(
        self, team: str, cfg: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, str | None]]:
        combat = self.config["combat"]
        own_team = [a for a in self.aircraft if a.team == team]
        enemies = [a for a in self.aircraft if a.team == self._enemy_team(team)]
        team_size = int(cfg["team_size"])
        sums = {"guide": 0.0, "attack_advantage": 0.0, "threat": 0.0}
        targets: dict[str, str | None] = {}

        for own in own_team:
            if not own.state.alive:
                targets[own.aircraft_id] = None
                continue
            target = self._nearest_alive_enemy(own, enemies)
            targets[own.aircraft_id] = target.aircraft_id if target is not None else None
            if target is None:
                continue
            local = paper_segmented_local_reward(
                own.state,
                target.state,
                float(combat["attack_distance_min"]),
                float(combat["attack_distance_max"]),
                cfg,
            )
            sums["guide"] += local["guide"]
            sums["attack_advantage"] += local["attack_advantage"]
            sums["threat"] += local["threat"]

        components = {
            "approach_reward": float(sums["guide"] / team_size),
            "attack_advantage_reward": float(sums["attack_advantage"] / team_size),
            "threat_penalty": float(sums["threat"] / team_size),
        }
        components["dense_reward"] = float(
            components["approach_reward"]
            + components["attack_advantage_reward"]
            + components["threat_penalty"]
        )
        return components, targets

    def _team_step_death_counts(
        self, team: str, step_death_causes: dict[str, int],
    ) -> tuple[int, int, int]:
        attack_losses = sum(
            1 for a_id, cause in step_death_causes.items()
            if self._aircraft_by_id(a_id).team == team and cause == DEATH_ATTACK
        )
        boundary_losses = sum(
            1 for a_id, cause in step_death_causes.items()
            if self._aircraft_by_id(a_id).team == team
            and cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY)
        )
        collision_losses = sum(
            1 for a_id, cause in step_death_causes.items()
            if self._aircraft_by_id(a_id).team == team
            and cause in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS)
        )
        return attack_losses, boundary_losses, collision_losses

    def _paper_segmented_v4_terminal(
        self, team: str, alive_count: int, opponent_alive_count: int, cfg: dict[str, Any],
        terminated: bool, truncated: bool,
    ) -> float:
        if not (terminated or truncated):
            return 0.0
        complete_attack_success = (
            self._episode_attack_kills[team] == int(cfg["team_size"])
            and alive_count > 0
        )
        terminal = 0.0
        if not complete_attack_success:
            terminal -= float(cfg["mission_failure_penalty_per_survivor"]) * alive_count
        if alive_count == 0 and opponent_alive_count == 0:
            terminal -= float(cfg["mutual_elimination_penalty"])
        return float(terminal)

    def _compute_paper_segmented_v4_rewards(
        self, attack_kills: dict[str, int], step_death_causes: dict[str, int],
        terminated: bool, truncated: bool,
        outcome: str | None, reason: str | None, red_alive: int, blue_alive: int,
        pre_attack_dense_parts: dict[str, dict[str, float]] | None,
        pre_attack_reward_targets: dict[str, str | None] | None,
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, str | None]]:
        cfg = self.config.get("reward_paper_segmented_v4", {})
        if not cfg:
            raise KeyError("paper_segmented_team_v4 requires reward_paper_segmented_v4 config")
        if pre_attack_dense_parts is None or pre_attack_reward_targets is None:
            raise ValueError("paper_segmented_team_v4 requires pre-attack dense snapshot")
        red_dense_parts = pre_attack_dense_parts["red"]
        blue_dense_parts = pre_attack_dense_parts["blue"]

        red_atk_losses, red_bdy_losses, red_col_losses = self._team_step_death_counts("red", step_death_causes)
        blue_atk_losses, blue_bdy_losses, blue_col_losses = self._team_step_death_counts("blue", step_death_causes)

        loss = float(cfg["aircraft_loss_penalty"])
        red_terminal = self._paper_segmented_v4_terminal(
            "red", red_alive, blue_alive, cfg, terminated, truncated)
        blue_terminal = self._paper_segmented_v4_terminal(
            "blue", blue_alive, red_alive, cfg, terminated, truncated)

        red_components = {
            "approach_reward": red_dense_parts["approach_reward"],
            "attack_advantage_reward": red_dense_parts["attack_advantage_reward"],
            "threat_penalty": red_dense_parts["threat_penalty"],
            "soft_boundary_penalty": 0.0,
            "friendly_separation_penalty": 0.0,
            "head_on_risk_penalty": 0.0,
            "time_penalty": 0.0,
            "dense_reward": red_dense_parts["dense_reward"],
            "kill_reward": float(cfg["kill_reward"]) * int(attack_kills["red"]),
            "attack_death_penalty": -loss * red_atk_losses,
            "boundary_death_penalty": -loss * red_bdy_losses,
            "collision_death_penalty": -loss * red_col_losses,
            "terminal_reward": red_terminal,
        }
        blue_components = {
            "approach_reward": blue_dense_parts["approach_reward"],
            "attack_advantage_reward": blue_dense_parts["attack_advantage_reward"],
            "threat_penalty": blue_dense_parts["threat_penalty"],
            "soft_boundary_penalty": 0.0,
            "friendly_separation_penalty": 0.0,
            "head_on_risk_penalty": 0.0,
            "time_penalty": 0.0,
            "dense_reward": blue_dense_parts["dense_reward"],
            "kill_reward": float(cfg["kill_reward"]) * int(attack_kills["blue"]),
            "attack_death_penalty": -loss * blue_atk_losses,
            "boundary_death_penalty": -loss * blue_bdy_losses,
            "collision_death_penalty": -loss * blue_col_losses,
            "terminal_reward": blue_terminal,
        }
        red_event = (
            red_components["kill_reward"]
            + red_components["attack_death_penalty"]
            + red_components["boundary_death_penalty"]
            + red_components["collision_death_penalty"]
        )
        blue_event = (
            blue_components["kill_reward"]
            + blue_components["attack_death_penalty"]
            + blue_components["boundary_death_penalty"]
            + blue_components["collision_death_penalty"]
        )
        red_total = float(red_components["dense_reward"] + red_event + red_components["terminal_reward"])
        blue_total = float(blue_components["dense_reward"] + blue_event + blue_components["terminal_reward"])
        red_components["team_total_reward"] = red_total
        blue_components["team_total_reward"] = blue_total

        rewards = {aid: red_total for aid in RED_IDS}
        rewards.update({aid: blue_total for aid in BLUE_IDS})
        reward_components = {
            **{f"red_{key}": value for key, value in red_components.items()},
            **{f"blue_{key}": value for key, value in blue_components.items()},
        }
        if not np.all(np.isfinite(list(reward_components.values()) + list(rewards.values()))):
            raise FloatingPointError("non-finite paper_segmented_team_v4 reward")
        return rewards, reward_components, dict(pre_attack_reward_targets)

    # -- functional_heterogeneous_team_v1 rewards ----------------------

    def _nearest_effective_visible_enemy(self, own: Aircraft, enemies: list[Aircraft]) -> Aircraft | None:
        visible = self._effective_visible_enemy_ids(own)
        candidates = [
            enemy for enemy in enemies
            if enemy.state.alive and enemy.aircraft_id in visible
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda enemy: (
            float(np.linalg.norm(own.state.as_array()[:3] - enemy.state.as_array()[:3])),
            enemy.aircraft_id,
        ))

    def _compute_functional_heterogeneous_dense(
        self,
        team: str,
        old_states: dict[str, AircraftState],
        cfg: dict[str, Any],
        support_coverage: float,
    ) -> tuple[dict[str, float], dict[str, str | None]]:
        bf = self.config["battlefield"]
        own_team = [a for a in self.aircraft if a.team == team]
        enemies = [a for a in self.aircraft if a.team == self._enemy_team(team)]
        combat_count = int(cfg["combat_count"])
        sums = {"approach": 0.0, "attack": 0.0, "threat": 0.0, "combat_boundary": 0.0}
        support_boundary = 0.0
        support_alive = False
        targets: dict[str, str | None] = {}

        for own in own_team:
            if not own.state.alive:
                targets[own.aircraft_id] = None
                continue
            boundary = soft_boundary_risk(
                own.state, bf["x_limit"], bf["y_limit"],
                bf["altitude_min"], bf["altitude_max"],
                cfg["horizontal_soft_ratio"], cfg["altitude_soft_margin"]
            )["total_risk"]

            if own.role == "support":
                support_alive = True
                support_boundary += boundary
                targets[own.aircraft_id] = None
                continue

            sums["combat_boundary"] += boundary
            target = self._nearest_effective_visible_enemy(own, enemies)
            targets[own.aircraft_id] = target.aircraft_id if target is not None else None
            if target is None:
                continue

            prev_own = old_states.get(own.aircraft_id, own.state)
            prev_target = old_states.get(target.aircraft_id, target.state)
            sums["approach"] += approach_progress_reward(
                prev_own, own.state, prev_target, target.state,
                cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"]
            )
            sums["attack"] += coupled_attack_advantage(
                own.state, target.state, cfg["preferred_distance"],
                cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"]
            )
            sums["threat"] += coupled_attack_advantage(
                target.state, own.state, cfg["preferred_distance"],
                cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"]
            )

        combat_dense_raw = (
            cfg["approach_weight"] * sums["approach"]
            + cfg["attack_advantage_weight"] * sums["attack"]
            - cfg["threat_weight"] * sums["threat"]
            - cfg["combat_boundary_weight"] * sums["combat_boundary"]
        ) / combat_count
        support_dense_raw = 0.0
        if support_alive:
            support_dense_raw = (
                cfg["support_information_weight"] * support_coverage
                - cfg["support_boundary_weight"] * support_boundary
            )
        dense_raw = combat_dense_raw + support_dense_raw - cfg["time_penalty"]
        dense = float(np.clip(dense_raw, cfg["dense_reward_min"], cfg["dense_reward_max"]))
        components = {
            "approach_reward": cfg["approach_weight"] * sums["approach"] / combat_count,
            "attack_advantage_reward": cfg["attack_advantage_weight"] * sums["attack"] / combat_count,
            "threat_penalty": cfg["threat_weight"] * sums["threat"] / combat_count,
            "soft_boundary_penalty": (
                cfg["combat_boundary_weight"] * sums["combat_boundary"] / combat_count
                + (cfg["support_boundary_weight"] * support_boundary if support_alive else 0.0)
            ),
            "friendly_separation_penalty": 0.0,
            "head_on_risk_penalty": 0.0,
            "time_penalty": cfg["time_penalty"],
            "dense_reward": dense,
        }
        return components, targets

    def _compute_heterogeneous_rewards(
        self,
        old_states: dict[str, AircraftState],
        attack_kills: dict[str, int],
        step_death_causes: dict[str, int],
        terminated: bool,
        truncated: bool,
        outcome: str | None,
        reason: str | None,
        red_alive: int,
        blue_alive: int,
        coverage: dict[str, tuple[float, int]],
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, str | None]]:
        cfg = self.config.get("reward_heterogeneous_v1", {})
        if not cfg:
            raise KeyError("functional_heterogeneous_team_v1 requires reward_heterogeneous_v1 config")

        red_dense_parts, red_targets = self._compute_functional_heterogeneous_dense(
            "red", old_states, cfg, float(coverage["red"][0])
        )
        blue_dense_parts, blue_targets = self._compute_functional_heterogeneous_dense(
            "blue", old_states, cfg, float(coverage["blue"][0])
        )

        def _losses(team: str, causes: tuple[int, ...]) -> int:
            return sum(
                1 for aid, cause in step_death_causes.items()
                if self._aircraft_by_id(aid).team == team and cause in causes
            )

        red_atk_losses = _losses("red", (DEATH_ATTACK,))
        red_bdy_losses = _losses("red", (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        red_col_losses = _losses("red", (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
        blue_atk_losses = _losses("blue", (DEATH_ATTACK,))
        blue_bdy_losses = _losses("blue", (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
        blue_col_losses = _losses("blue", (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))

        red_event = (
            cfg["kill_reward"] * attack_kills["red"]
            - cfg["attack_death_penalty"] * red_atk_losses
            - cfg["boundary_death_penalty"] * red_bdy_losses
            - cfg["collision_death_penalty"] * red_col_losses
        )
        blue_event = (
            cfg["kill_reward"] * attack_kills["blue"]
            - cfg["attack_death_penalty"] * blue_atk_losses
            - cfg["boundary_death_penalty"] * blue_bdy_losses
            - cfg["collision_death_penalty"] * blue_col_losses
        )

        red_terminal = 0.0
        blue_terminal = 0.0
        if terminated or truncated:
            red_succ = (self._episode_attack_kills["red"] == 3 and red_alive > 0)
            blue_succ = (self._episode_attack_kills["blue"] == 3 and blue_alive > 0)
            if red_succ:
                red_terminal += cfg["complete_elimination_bonus"]
            if blue_succ:
                blue_terminal += cfg["complete_elimination_bonus"]
            if reason == "max_steps":
                red_terminal -= cfg["max_steps_red_failure_penalty"]
                blue_terminal += cfg["max_steps_blue_success_bonus"]
            elif red_alive == 0 and blue_alive == 0:
                red_terminal -= cfg["mutual_elimination_penalty"]
                blue_terminal -= cfg["mutual_elimination_penalty"]
            elif red_alive == 0:
                red_terminal -= cfg["team_eliminated_penalty"]
            elif blue_alive == 0:
                blue_terminal -= cfg["team_eliminated_penalty"]

        red_total = red_dense_parts["dense_reward"] + red_event + red_terminal
        blue_total = blue_dense_parts["dense_reward"] + blue_event + blue_terminal
        rewards = {aid: float(red_total) for aid in RED_IDS}
        rewards.update({aid: float(blue_total) for aid in BLUE_IDS})
        reward_components = {
            "red_approach_reward": red_dense_parts["approach_reward"],
            "red_attack_advantage_reward": red_dense_parts["attack_advantage_reward"],
            "red_threat_penalty": red_dense_parts["threat_penalty"],
            "red_soft_boundary_penalty": red_dense_parts["soft_boundary_penalty"],
            "red_friendly_separation_penalty": 0.0,
            "red_head_on_risk_penalty": 0.0,
            "red_time_penalty": red_dense_parts["time_penalty"],
            "red_dense_reward": red_dense_parts["dense_reward"],
            "red_kill_reward": cfg["kill_reward"] * attack_kills["red"],
            "red_attack_death_penalty": cfg["attack_death_penalty"] * red_atk_losses,
            "red_boundary_death_penalty": cfg["boundary_death_penalty"] * red_bdy_losses,
            "red_collision_death_penalty": cfg["collision_death_penalty"] * red_col_losses,
            "red_terminal_reward": red_terminal,
            "red_team_total_reward": red_total,
            "blue_approach_reward": blue_dense_parts["approach_reward"],
            "blue_attack_advantage_reward": blue_dense_parts["attack_advantage_reward"],
            "blue_threat_penalty": blue_dense_parts["threat_penalty"],
            "blue_soft_boundary_penalty": blue_dense_parts["soft_boundary_penalty"],
            "blue_friendly_separation_penalty": 0.0,
            "blue_head_on_risk_penalty": 0.0,
            "blue_time_penalty": blue_dense_parts["time_penalty"],
            "blue_dense_reward": blue_dense_parts["dense_reward"],
            "blue_kill_reward": cfg["kill_reward"] * attack_kills["blue"],
            "blue_attack_death_penalty": cfg["attack_death_penalty"] * blue_atk_losses,
            "blue_boundary_death_penalty": cfg["boundary_death_penalty"] * blue_bdy_losses,
            "blue_collision_death_penalty": cfg["collision_death_penalty"] * blue_col_losses,
            "blue_terminal_reward": blue_terminal,
            "blue_team_total_reward": blue_total,
        }
        if not np.all(np.isfinite(list(reward_components.values()) + list(rewards.values()))):
            raise FloatingPointError("non-finite functional_heterogeneous_team_v1 reward")
        return rewards, reward_components, {**red_targets, **blue_targets}

    # -- observations ---------------------------------------------------

    def _all_observations(self) -> dict[str, np.ndarray]:
        return {own.aircraft_id: self._agent_observation(own) for own in self.aircraft}

    def _agent_observation(self, own: Aircraft) -> np.ndarray:
        if self._heterogeneous:
            return self._heterogeneous_agent_observation(own)
        if not own.state.alive:
            return np.zeros(OBS_DIM, dtype=np.float32)
        bf = self.config["battlefield"]
        xs, ys, zs = bf["x_limit"], bf["y_limit"], bf["altitude_max"] - bf["altitude_min"]
        ds = float(np.sqrt(xs**2 + ys**2 + zs**2))
        ps = max(abs(own.spec.theta_min), abs(own.spec.theta_max))
        self_block = np.array([
            own.state.x / xs, own.state.y / ys,
            2.0 * (own.state.altitude - bf["altitude_min"]) / zs - 1.0,
            2.0 * (own.state.v - own.spec.v_min) / (own.spec.v_max - own.spec.v_min) - 1.0,
            own.state.theta / ps, np.sin(own.state.psi), np.cos(own.state.psi), 1.0], dtype=np.float32)

        teammates = [a for a in self.aircraft if a.team == own.team and a.aircraft_id != own.aircraft_id]
        teammates.sort(key=lambda a: (
            float(np.linalg.norm(own.state.as_array()[:3] - a.state.as_array()[:3])) if a.state.alive else np.inf,
            a.aircraft_id))
        enemy_team = "blue" if own.team == "red" else "red"
        enemies = [a for a in self.aircraft if a.team == enemy_team]
        enemies.sort(key=lambda a: (
            float(np.linalg.norm(own.state.as_array()[:3] - a.state.as_array()[:3])) if a.state.alive else np.inf,
            a.aircraft_id))
        blocks = [self_block]
        for lst, n in ((teammates[:2], 2), (enemies[:3], 3)):
            for a in lst:
                blocks.append(self._entity_block(own, a, xs, ys, zs, ds))
            while len(blocks) < (1 + (2 if n == 2 else 0) + 2 + n):
                blocks.append(np.zeros(12, dtype=np.float32))
        obs = np.concatenate([self_block] + [self._entity_block(own, a, xs, ys, zs, ds) for a in teammates[:2]]
                             + [np.zeros(12, np.float32)] * max(0, 2 - len(teammates[:2]))
                             + [self._entity_block(own, a, xs, ys, zs, ds) for a in enemies[:3]]
                             + [np.zeros(12, np.float32)] * max(0, 3 - len(enemies[:3])))
        return _normalize_obs(obs)

    def _heterogeneous_agent_observation(self, own: Aircraft) -> np.ndarray:
        if not own.state.alive:
            return np.zeros(OBS_DIM, dtype=np.float32)
        bf = self.config["battlefield"]
        xs, ys, zs = bf["x_limit"], bf["y_limit"], bf["altitude_max"] - bf["altitude_min"]
        ds = float(np.sqrt(xs**2 + ys**2 + zs**2))
        ps = max(abs(own.spec.theta_min), abs(own.spec.theta_max))
        self_block = np.array([
            own.state.x / xs, own.state.y / ys,
            2.0 * (own.state.altitude - bf["altitude_min"]) / zs - 1.0,
            2.0 * (own.state.v - own.spec.v_min) / (own.spec.v_max - own.spec.v_min) - 1.0,
            own.state.theta / ps, np.sin(own.state.psi), np.cos(own.state.psi), 1.0,
        ], dtype=np.float32)
        team_ids = RED_IDS if own.team == "red" else BLUE_IDS
        enemy_ids = BLUE_IDS if own.team == "red" else RED_IDS
        teammates = [self._aircraft_by_id(aid) for aid in team_ids if aid != own.aircraft_id]
        enemies = [self._aircraft_by_id(aid) for aid in enemy_ids]
        visible = self._effective_visible_enemy_ids(own)
        enemy_blocks = [
            self._heterogeneous_enemy_block(own, enemy, xs, ys, zs, ds, enemy.aircraft_id in visible)
            for enemy in enemies
        ]
        obs = np.concatenate(
            [self_block]
            + [self._entity_block(own, mate, xs, ys, zs, ds) for mate in teammates]
            + enemy_blocks
        )
        return _normalize_obs(obs)

    def _heterogeneous_enemy_block(self, own, enemy, xs, ys, zs, ds, visible: bool) -> np.ndarray:
        if not enemy.state.alive:
            block = np.zeros(12, dtype=np.float32)
            block[-1] = -1.0
            return block
        if not visible:
            return np.zeros(12, dtype=np.float32)
        block = self._entity_block(own, enemy, xs, ys, zs, ds)
        block[-1] = 1.0
        return block

    def _entity_block(self, own, other, xs, ys, zs, ds):
        if not other.state.alive:
            return np.zeros(12, dtype=np.float32)
        geo = compute_pairwise_geometry(own.state, other.state)
        dx, dy, dz = geo.relative_position
        dvx, dvy, dvz = geo.relative_velocity
        c, s = np.cos(own.state.psi), np.sin(own.state.psi)
        return np.array([
            (c * dx + s * dy) / xs, (-s * dx + c * dy) / ys, (-dz) / zs,
            (c * dvx + s * dvy) / (2.0 * own.spec.v_max),
            (-s * dvx + c * dvy) / (2.0 * own.spec.v_max),
            (-dvz) / (2.0 * own.spec.v_max),
            geo.distance / ds, geo.yaw_error / np.pi, geo.pitch_error / (np.pi / 2.0),
            geo.ata / np.pi, geo.aa / np.pi, 1.0], dtype=np.float32)

    def global_state(self) -> np.ndarray:
        bf = self.config["battlefield"]
        xs, ys, zs = bf["x_limit"], bf["y_limit"], bf["altitude_max"] - bf["altitude_min"]
        feats = []
        for aid in ALL_IDS:
            a = self._aircraft_by_id(aid)
            ps = max(abs(a.spec.theta_min), abs(a.spec.theta_max))
            feats.extend([a.state.x / xs, a.state.y / ys,
                          2.0 * (a.state.altitude - bf["altitude_min"]) / zs - 1.0,
                          2.0 * (a.state.v - a.spec.v_min) / (a.spec.v_max - a.spec.v_min) - 1.0,
                          a.state.theta / ps, np.sin(a.state.psi), np.cos(a.state.psi),
                          1.0 if a.state.alive else 0.0])
        r = np.clip(np.asarray(feats, dtype=np.float32), -1.0, 1.0)
        if not np.all(np.isfinite(r)): raise ValueError("global state must be finite")
        return r
