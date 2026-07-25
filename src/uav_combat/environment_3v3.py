"""Homogeneous 3v3 air combat environment with synchronous step semantics."""
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

    return {
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
    """3v3 synchronous air combat."""

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
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False
        self._episode_death_causes: dict[str, int] = {}
        self._episode_attack_kills: dict[str, int] = {}

    def _aircraft_by_id(self, aid: str) -> Aircraft:
        return next(a for a in self.aircraft if a.aircraft_id == aid)

    def _alive(self, team: str) -> list[Aircraft]:
        return [a for a in self.aircraft if a.team == team and a.state.alive]

    def _alive_count(self, team: str) -> int:
        return sum(1 for a in self.aircraft if a.team == team and a.state.alive)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        self._running = True
        self._episode_death_causes = {aid: DEATH_NONE for aid in ALL_IDS}
        self._episode_attack_kills = {"red": 0, "blue": 0}
        observations = self._all_observations()
        info = {"step_count": 0, "scenario_name": self.scenario.scenario_name,
                "termination_reason": None, "outcome": None,
                "red_alive_count": 3, "blue_alive_count": 3,
                "attacks": {}, "death_causes": {}, "attack_kills": {"red": 0, "blue": 0},
                "boundary_deaths": {"red": 0, "blue": 0},
                "collision_deaths": {"red": 0, "blue": 0},
                "red_complete_elimination_success": False,
                "blue_complete_elimination_success": False,
                "control_diagnostics": {}, "reward_components": {},
                "nearest_enemy_geometry": {}, "episode_summary": None,
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
            diag = self.controller.diagnostics(old_states[aid], tgt, ctrl, aircraft.spec)
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

        # 10-11. Attack intents
        attack_intents: dict[str, str | None] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                attack_intents[aircraft.aircraft_id] = None; continue
            enemy_team = "blue" if aircraft.team == "red" else "red"
            alive_enemies = [a for a in self.aircraft if a.team == enemy_team and a.state.alive]
            if not alive_enemies:
                attack_intents[aircraft.aircraft_id] = None; continue
            attackable = [e for e in alive_enemies if self.attack_model.can_attack(aircraft.state, e.state)]
            if attackable:
                best = min(attackable, key=lambda e: (
                    float(np.linalg.norm(aircraft.state.as_array()[:3] - e.state.as_array()[:3])), e.aircraft_id))
                attack_intents[aircraft.aircraft_id] = best.aircraft_id
            else:
                attack_intents[aircraft.aircraft_id] = None

        attackers_by_target: dict[str, list[str]] = {}
        for atk_id, tgt_id in attack_intents.items():
            if tgt_id is not None:
                attackers_by_target.setdefault(tgt_id, []).append(atk_id)

        attack_kills = {"red": 0, "blue": 0}
        for tgt_id in attackers_by_target:
            tgt = self._aircraft_by_id(tgt_id)
            if tgt.state.alive:
                tgt.state.alive = False
                if tgt_id not in step_death_causes:
                    step_death_causes[tgt_id] = DEATH_ATTACK
                if tgt.team == "blue": attack_kills["red"] += 1
                else: attack_kills["blue"] += 1

        # Accumulate episode
        for aid, cause in step_death_causes.items():
            if self._episode_death_causes.get(aid, DEATH_NONE) == DEATH_NONE:
                self._episode_death_causes[aid] = cause
        self._episode_attack_kills["red"] += attack_kills["red"]
        self._episode_attack_kills["blue"] += attack_kills["blue"]

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
            outcome, reason = "draw", "max_steps"
        if terminated or truncated:
            self._running = False

        # --- Per-step death tallies (unconditional, needed for info) ---
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
        bdy_d = {"red": red_bdy_losses, "blue": blue_bdy_losses}
        col_d = {"red": red_col_losses, "blue": blue_col_losses}

        # 13. Rewards
        reward_mode = self.config["combat"].get("reward_mode", "madsac_segmented")
        if reward_mode == "paper_coupled_team_v2":
            rewards, reward_components = self._compute_v2_rewards(
                old_states, attack_kills, step_death_causes, terminated, truncated,
                outcome, reason, red_alive, blue_alive)
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
                red_alive, blue_alive, outcome, reason, self.step_count)

        red_success = episode_summary["red_complete_elimination_success"] if episode_summary else False
        blue_success = episode_summary["blue_complete_elimination_success"] if episode_summary else False

        info = {
            "step_count": self.step_count, "scenario_name": self.scenario.scenario_name,
            "termination_reason": reason, "outcome": outcome,
            "red_alive_count": red_alive, "blue_alive_count": blue_alive,
            "attacks": attack_intents, "death_causes": step_death_causes,
            "attack_kills": attack_kills, "boundary_deaths": bdy_d, "collision_deaths": col_d,
            "red_complete_elimination_success": red_success,
            "blue_complete_elimination_success": blue_success,
            "red_survivors": red_alive, "blue_survivors": blue_alive,
            "collision_pairs": collision_pairs, "attackers_by_target": attackers_by_target,
            "control_diagnostics": control_diagnostics, "reward_components": reward_components,
            "nearest_enemy_geometry": nearest_enemy_geom,
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

            # Approach: max over alive blues
            app = 0.0
            for blue_ac in alive_blues:
                prev_blue = old_states.get(blue_ac.aircraft_id, blue_ac.state)
                val = approach_progress_reward(
                    prev_red, red_state, prev_blue, blue_ac.state,
                    cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"])
                if val > app: app = val
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
            app = 0.0
            for red_ac in alive_reds:
                pr = old_states.get(red_ac.aircraft_id, red_ac.state)
                val = approach_progress_reward(pb, bs, pr, red_ac.state,
                                               cfg["approach_distance_threshold"], cfg["approach_distance_normalizer"])
                if val > app: app = val
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

    # -- observations ---------------------------------------------------

    def _all_observations(self) -> dict[str, np.ndarray]:
        return {own.aircraft_id: self._agent_observation(own) for own in self.aircraft}

    def _agent_observation(self, own: Aircraft) -> np.ndarray:
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
