"""Homogeneous 3v3 air combat environment with synchronous step semantics."""
from pathlib import Path
from typing import Any

import numpy as np

from .combat import SimplifiedAttackModel
from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .geometry import PairwiseGeometry, compute_pairwise_geometry
from .integrator import RK4Integrator
from .math_utils import angle_difference
from .models import Aircraft, AircraftSpec, ControlCommand, TargetCommand
from .rewards import madsac_segmented_reward
from .scenario_3v3 import ALL_IDS, BLUE_IDS, RED_IDS, Homogeneous3v3Scenario

# Death cause codes (int8)
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


def _death_summary(death_causes: dict[str, int], team: str, team_ids: tuple[str, ...]) -> dict[str, int]:
    """Return {attack_deaths, boundary_deaths, collision_deaths, survivors} for *team*."""
    attack = 0
    boundary = 0
    collision = 0
    alive = 0
    for aid in team_ids:
        cause = death_causes.get(aid, DEATH_NONE)
        if cause == DEATH_NONE:
            alive += 1
        elif cause == DEATH_ATTACK:
            attack += 1
        elif cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY):
            boundary += 1
        elif cause in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS):
            collision += 1
    return {"attack_deaths": attack, "boundary_deaths": boundary, "collision_deaths": collision, "survivors": alive}


def _validate_death_ledger(team: str, summary: dict[str, int]) -> None:
    total = summary["survivors"] + summary["attack_deaths"] + summary["boundary_deaths"] + summary["collision_deaths"]
    if total != 3:
        raise RuntimeError(
            f"Death ledger mismatch for {team}: "
            f"survivors={summary['survivors']} + attack={summary['attack_deaths']} + "
            f"boundary={summary['boundary_deaths']} + collision={summary['collision_deaths']} = {total} != 3"
        )


class Homogeneous3v3AirCombatEnv:
    """3v3 synchronous air combat with per-aircraft death, boundary, collision, and attack."""

    def __init__(self, config_path: str | Path = "configs/homogeneous_3v3.yaml") -> None:
        self.config = load_config(config_path)
        sim, act, combat = self.config["simulation"], self.config["action"], self.config["combat"]
        self.scenario = Homogeneous3v3Scenario(self.config)
        self.dynamics = PointMassDynamics(sim["gravity"])
        self.integrator = RK4Integrator(sim["dt"])
        self.controller = TargetStateController(**act, gravity=sim["gravity"])
        self.attack_model = SimplifiedAttackModel(
            combat["attack_distance_min"], combat["attack_distance_max"],
            combat["attack_ata_max"], combat["attack_aa_max"],
        )
        self.aircraft: list[Aircraft] = []
        self.step_count = 0
        self._running = False
        # Accumulated death causes and attack kills across all steps in this episode
        self._episode_death_causes: dict[str, int] = {}
        self._episode_attack_kills: dict[str, int] = {}

    # -- helpers --------------------------------------------------------

    def _aircraft_by_id(self, aircraft_id: str) -> Aircraft:
        return next(a for a in self.aircraft if a.aircraft_id == aircraft_id)

    def _alive(self, team: str) -> list[Aircraft]:
        return [a for a in self.aircraft if a.team == team and a.state.alive]

    def _alive_count(self, team: str) -> int:
        return sum(1 for a in self.aircraft if a.team == team and a.state.alive)

    # -- reset ----------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.aircraft = self.scenario.reset(seed)
        self.step_count = 0
        self._running = True
        self._episode_death_causes = {aid: DEATH_NONE for aid in ALL_IDS}
        self._episode_attack_kills = {"red": 0, "blue": 0}
        observations = self._all_observations()
        info = self._empty_info()
        info["global_state"] = self.global_state()
        return observations, info

    def _empty_info(self) -> dict[str, Any]:
        return {
            "step_count": 0,
            "scenario_name": self.scenario.scenario_name,
            "termination_reason": None,
            "outcome": None,
            "red_alive_count": 3,
            "blue_alive_count": 3,
            "attacks": {aid: None for aid in ALL_IDS},
            "death_causes": {aid: DEATH_NONE for aid in ALL_IDS},
            "attack_kills": {"red": 0, "blue": 0},
            "boundary_deaths": {"red": 0, "blue": 0},
            "collision_deaths": {"red": 0, "blue": 0},
            "red_complete_elimination_success": False,
            "blue_complete_elimination_success": False,
            "control_diagnostics": {},
            "reward_components": {},
            "blue_targets": {},
            "episode_summary": None,
        }

    # -- step (synchronous) ---------------------------------------------

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        """Synchronous 3v3 step."""
        if not self._running:
            raise RuntimeError("reset() must be called before step()")

        alive = [a for a in self.aircraft if a.state.alive]
        missing = {a.aircraft_id for a in alive} - set(actions.keys())
        if missing:
            raise KeyError(f"missing actions for: {sorted(missing)}")

        # --- 1. Save old states ---
        old_states = {a.aircraft_id: a.state.copy() for a in alive}

        # --- 2-3. Compute controls for alive aircraft ---
        targets: dict[str, TargetCommand] = {}
        controls: dict[str, ControlCommand] = {}
        control_diagnostics: dict[str, dict[str, float | bool]] = {}
        for aircraft in alive:
            aid = aircraft.aircraft_id
            target, control = self.controller.control_from_action(
                old_states[aid], actions[aid], aircraft.spec)
            targets[aid], controls[aid] = target, control
            diag = self.controller.diagnostics(old_states[aid], target, control, aircraft.spec)
            derivatives = self.dynamics.derivatives(old_states[aid], control)
            aa, apr, ayr = map(float, derivatives[3:6])
            diag.update({
                "actual_acceleration": aa, "actual_pitch_rate": apr, "actual_yaw_rate": ayr,
                "acceleration_tracking_error": diag["clipped_acceleration"] - aa,
                "pitch_rate_tracking_error": diag["clipped_pitch_rate"] - apr,
                "yaw_rate_tracking_error": diag["clipped_yaw_rate"] - ayr,
            })
            for label, err in (("acceleration_tracking_absolute_error", "acceleration_tracking_error"),
                               ("pitch_rate_tracking_absolute_error", "pitch_rate_tracking_error"),
                               ("yaw_rate_tracking_absolute_error", "yaw_rate_tracking_error")):
                diag[label] = abs(diag[err])
            clipped = np.clip(np.asarray(actions[aid], dtype=float), -1.0, 1.0)
            diag.update({
                "action_yaw": float(clipped[0]), "action_pitch": float(clipped[1]), "action_speed": float(clipped[2]),
                "delta_yaw": float(angle_difference(target.desired_psi, old_states[aid].psi)),
                "delta_pitch": float(target.desired_theta - old_states[aid].theta),
                "delta_speed": float(target.desired_v - old_states[aid].v),
            })
            control_diagnostics[aid] = diag

        # --- 4-5. RK4 integrate all alive ---
        new_states = {}
        for aircraft in alive:
            aid = aircraft.aircraft_id
            new_states[aid] = self.integrator.step(
                old_states[aid], controls[aid], self.dynamics, aircraft.spec)
        for aircraft in alive:
            aircraft.state = new_states[aircraft.aircraft_id]
        self.step_count += 1

        # --- 6. Boundary check ---
        limits = self.config["battlefield"]
        step_death_causes: dict[str, int] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                continue
            s = aircraft.state
            if not (limits["altitude_min"] <= s.altitude <= limits["altitude_max"]):
                s.alive = False
                step_death_causes[aircraft.aircraft_id] = DEATH_BOUNDARY_ALTITUDE
            elif abs(s.x) > limits["x_limit"] or abs(s.y) > limits["y_limit"]:
                s.alive = False
                step_death_causes[aircraft.aircraft_id] = DEATH_BOUNDARY_XY

        # --- 7-8. Collision check (collect first, then apply) ---
        collision_pairs: list[tuple[str, str]] = []
        alive_now = [a for a in self.aircraft if a.state.alive]
        for i in range(len(alive_now)):
            for j in range(i + 1, len(alive_now)):
                a1, a2 = alive_now[i], alive_now[j]
                dist = float(np.linalg.norm(a1.state.as_array()[:3] - a2.state.as_array()[:3]))
                if dist <= limits["collision_distance"]:
                    collision_pairs.append((a1.aircraft_id, a2.aircraft_id))

        for aid1, aid2 in collision_pairs:
            a1, a2 = self._aircraft_by_id(aid1), self._aircraft_by_id(aid2)
            if a1.state.alive:
                a1.state.alive = False
                if aid1 not in step_death_causes:
                    step_death_causes[aid1] = (DEATH_COLLISION_FRIENDLY if a1.team == a2.team else DEATH_COLLISION_CROSS)
            if a2.state.alive:
                a2.state.alive = False
                if aid2 not in step_death_causes:
                    step_death_causes[aid2] = (DEATH_COLLISION_FRIENDLY if a1.team == a2.team else DEATH_COLLISION_CROSS)

        # --- 9-11. Attack (only still-alive aircraft can attack) ---
        attack_intents: dict[str, str | None] = {}
        for aircraft in self.aircraft:
            if not aircraft.state.alive:
                attack_intents[aircraft.aircraft_id] = None
                continue
            enemy_team = "blue" if aircraft.team == "red" else "red"
            alive_enemies = [a for a in self.aircraft if a.team == enemy_team and a.state.alive]
            if not alive_enemies:
                attack_intents[aircraft.aircraft_id] = None
                continue
            attackable: list[Aircraft] = []
            for enemy in alive_enemies:
                if self.attack_model.can_attack(aircraft.state, enemy.state):
                    attackable.append(enemy)
            if attackable:
                best = min(attackable, key=lambda e: (
                    float(np.linalg.norm(aircraft.state.as_array()[:3] - e.state.as_array()[:3])),
                    e.aircraft_id,
                ))
                attack_intents[aircraft.aircraft_id] = best.aircraft_id
            else:
                attack_intents[aircraft.aircraft_id] = None

        # Collect attackers per target, then apply kills (each target dies at most once)
        attackers_by_target: dict[str, list[str]] = {}
        for attacker_id, target_id in attack_intents.items():
            if target_id is not None:
                attackers_by_target.setdefault(target_id, []).append(attacker_id)

        attack_kills = {"red": 0, "blue": 0}
        for target_id in attackers_by_target:
            target = self._aircraft_by_id(target_id)
            if target.state.alive:
                target.state.alive = False
                # Attack death only if not already killed by boundary/collision this step
                if target_id not in step_death_causes:
                    step_death_causes[target_id] = DEATH_ATTACK
                if target.team == "blue":
                    attack_kills["red"] += 1
                else:
                    attack_kills["blue"] += 1

        # --- Accumulate episode-level stats ---
        for aid, cause in step_death_causes.items():
            if self._episode_death_causes.get(aid, DEATH_NONE) == DEATH_NONE:
                self._episode_death_causes[aid] = cause
        self._episode_attack_kills["red"] += attack_kills["red"]
        self._episode_attack_kills["blue"] += attack_kills["blue"]

        # --- 12. Team elimination ---
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

        # --- 13. Rewards ---
        rewards, reward_components = self._compute_rewards(attack_kills, step_death_causes)

        # --- 14. Observations ---
        observations = self._all_observations()

        # --- Per-step death tallies ---
        bdy_d = {"red": 0, "blue": 0}
        col_d = {"red": 0, "blue": 0}
        for aid, cause in step_death_causes.items():
            a = self._aircraft_by_id(aid)
            if cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY):
                bdy_d[a.team] += 1
            elif cause in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS):
                col_d[a.team] += 1

        # --- Episode summary (only at episode end) ---
        episode_summary = None
        if terminated or truncated:
            # Build final per-team death ledger
            red_summary = _death_summary(self._episode_death_causes, "red", RED_IDS)
            blue_summary = _death_summary(self._episode_death_causes, "blue", BLUE_IDS)
            _validate_death_ledger("red", red_summary)
            _validate_death_ledger("blue", blue_summary)

            # Paper success (use accumulated attack kills over whole episode, must have survivors)
            red_success = (self._episode_attack_kills["red"] == 3 and red_alive > 0)
            blue_success = (self._episode_attack_kills["blue"] == 3 and blue_alive > 0)

            episode_summary = {
                "red_attack_kills": self._episode_attack_kills["red"],
                "blue_attack_kills": self._episode_attack_kills["blue"],
                "red_survivors": red_alive,
                "blue_survivors": blue_alive,
                "red_death_causes": {
                    "attack_deaths": red_summary["attack_deaths"],
                    "boundary_deaths": red_summary["boundary_deaths"],
                    "collision_deaths": red_summary["collision_deaths"],
                },
                "blue_death_causes": {
                    "attack_deaths": blue_summary["attack_deaths"],
                    "boundary_deaths": blue_summary["boundary_deaths"],
                    "collision_deaths": blue_summary["collision_deaths"],
                },
                "environment_outcome": outcome,
                "red_complete_elimination_success": red_success,
                "blue_complete_elimination_success": blue_success,
                "episode_length": self.step_count,
                "termination_reason": reason,
            }
        else:
            # Per-step success flags (episode_summary has authoritative ones at end)
            red_success = (self._episode_attack_kills["red"] == 3 and red_alive > 0) if (terminated or truncated) else False
            blue_success = (self._episode_attack_kills["blue"] == 3 and blue_alive > 0) if (terminated or truncated) else False

        info = {
            "step_count": self.step_count,
            "scenario_name": self.scenario.scenario_name,
            "termination_reason": reason,
            "outcome": outcome,
            "red_alive_count": red_alive,
            "blue_alive_count": blue_alive,
            "attacks": attack_intents,
            "death_causes": step_death_causes,   # only this step's deaths
            "attack_kills": attack_kills,         # only this step's kills
            "boundary_deaths": bdy_d,
            "collision_deaths": col_d,
            "red_complete_elimination_success": red_success,
            "blue_complete_elimination_success": blue_success,
            "red_survivors": red_alive,
            "blue_survivors": blue_alive,
            "collision_pairs": collision_pairs,
            "attackers_by_target": attackers_by_target,
            "control_diagnostics": control_diagnostics,
            "reward_components": reward_components,
            "blue_targets": {},
            "global_state": self.global_state(),
            "episode_summary": episode_summary,
        }
        return observations, rewards, terminated, truncated, info

    # -- rewards --------------------------------------------------------

    def _compute_rewards(
        self, attack_kills: dict[str, int], death_causes: dict[str, int],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        red_alive_list = self._alive("red")
        blue_alive_list = self._alive("blue")

        per_red_dense = {}
        for red in red_alive_list:
            if blue_alive_list:
                nearest_blue = min(blue_alive_list, key=lambda b: float(
                    np.linalg.norm(red.state.as_array()[:3] - b.state.as_array()[:3])))
                reward_dict = madsac_segmented_reward(
                    red.state, nearest_blue.state, "red", None, None)
                per_red_dense[red.aircraft_id] = reward_dict["reward_total"]
            else:
                per_red_dense[red.aircraft_id] = 0.0

        if red_alive_list:
            team_dense = float(np.mean(list(per_red_dense.values())))
        else:
            team_dense = 0.0

        red_attack_losses = 0
        red_boundary_losses = 0
        red_collision_losses = 0
        for aid, cause in death_causes.items():
            a = self._aircraft_by_id(aid)
            if a.team == "red":
                if cause == DEATH_ATTACK:
                    red_attack_losses += 1
                elif cause in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY):
                    red_boundary_losses += 1
                elif cause in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS):
                    red_collision_losses += 1

        T = self.config["combat"]["terminal_reward"]
        team_total = (
            team_dense
            + T * attack_kills["red"]
            - T * red_attack_losses
            - T * red_boundary_losses
            - T * red_collision_losses
        )

        rewards = {}
        for aid in RED_IDS:
            rewards[aid] = float(team_total)
        for aid in BLUE_IDS:
            blue_attack_losses = sum(1 for a_id, c in death_causes.items()
                                     if self._aircraft_by_id(a_id).team == "blue" and c == DEATH_ATTACK)
            blue_bdy_losses = sum(1 for a_id, c in death_causes.items()
                                  if self._aircraft_by_id(a_id).team == "blue"
                                  and c in (DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY))
            blue_col_losses = sum(1 for a_id, c in death_causes.items()
                                  if self._aircraft_by_id(a_id).team == "blue"
                                  and c in (DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS))
            blue_sym = (team_dense + T * attack_kills["blue"]
                        - T * blue_attack_losses - T * blue_bdy_losses - T * blue_col_losses)
            rewards[aid] = float(blue_sym)

        components = {
            "team_dense_reward": team_dense,
            "per_red_agent_dense_reward": per_red_dense,
            "blue_attack_kill_reward": T * attack_kills["red"],
            "red_attack_loss_penalty": -T * red_attack_losses,
            "red_boundary_penalty": -T * red_boundary_losses,
            "red_collision_penalty": -T * red_collision_losses,
            "team_total_reward": team_total,
        }
        return rewards, components

    # -- observations (68-dim per agent) --------------------------------

    def _all_observations(self) -> dict[str, np.ndarray]:
        observations: dict[str, np.ndarray] = {}
        for own in self.aircraft:
            observations[own.aircraft_id] = self._agent_observation(own)
        return observations

    def _agent_observation(self, own: Aircraft) -> np.ndarray:
        if not own.state.alive:
            return np.zeros(OBS_DIM, dtype=np.float32)

        battlefield = self.config["battlefield"]
        x_scale = battlefield["x_limit"]
        y_scale = battlefield["y_limit"]
        z_scale = battlefield["altitude_max"] - battlefield["altitude_min"]
        dist_scale = float(np.sqrt(x_scale**2 + y_scale**2 + z_scale**2))

        pitch_scale = max(abs(own.spec.theta_min), abs(own.spec.theta_max))
        speed_norm = 2.0 * (own.state.v - own.spec.v_min) / (own.spec.v_max - own.spec.v_min) - 1.0
        alt_norm = 2.0 * (own.state.altitude - battlefield["altitude_min"]) / z_scale - 1.0
        x_norm = own.state.x / x_scale
        y_norm = own.state.y / y_scale

        self_block = np.array([
            x_norm, y_norm, alt_norm, speed_norm,
            own.state.theta / pitch_scale,
            np.sin(own.state.psi), np.cos(own.state.psi),
            1.0,
        ], dtype=np.float32)

        teammates = [a for a in self.aircraft if a.team == own.team and a.aircraft_id != own.aircraft_id]
        teammates.sort(key=lambda a: (
            float(np.linalg.norm(own.state.as_array()[:3] - a.state.as_array()[:3])) if a.state.alive else np.inf,
            a.aircraft_id,
        ))

        enemy_team = "blue" if own.team == "red" else "red"
        enemies = [a for a in self.aircraft if a.team == enemy_team]
        enemies.sort(key=lambda a: (
            float(np.linalg.norm(own.state.as_array()[:3] - a.state.as_array()[:3])) if a.state.alive else np.inf,
            a.aircraft_id,
        ))

        mate_blocks = []
        for a in teammates[:2]:
            mate_blocks.append(self._entity_block(own, a, x_scale, y_scale, z_scale, dist_scale))
        while len(mate_blocks) < 2:
            mate_blocks.append(np.zeros(12, dtype=np.float32))
        enemy_blocks = []
        for a in enemies[:3]:
            enemy_blocks.append(self._entity_block(own, a, x_scale, y_scale, z_scale, dist_scale))
        while len(enemy_blocks) < 3:
            enemy_blocks.append(np.zeros(12, dtype=np.float32))

        observation = np.concatenate([self_block] + mate_blocks + enemy_blocks)
        return _normalize_obs(observation)

    def _entity_block(
        self, own: Aircraft, other: Aircraft,
        x_scale: float, y_scale: float, z_scale: float, dist_scale: float,
    ) -> np.ndarray:
        if not other.state.alive:
            return np.zeros(12, dtype=np.float32)

        geo = compute_pairwise_geometry(own.state, other.state)
        dx, dy, dz = geo.relative_position
        dvx, dvy, dvz = geo.relative_velocity
        cosine, sine = np.cos(own.state.psi), np.sin(own.state.psi)
        rel_x = (cosine * dx + sine * dy) / x_scale
        rel_y = (-sine * dx + cosine * dy) / y_scale
        rel_z = (-dz) / z_scale
        vel_x = (cosine * dvx + sine * dvy) / (2.0 * own.spec.v_max)
        vel_y = (-sine * dvx + cosine * dvy) / (2.0 * own.spec.v_max)
        vel_z = (-dvz) / (2.0 * own.spec.v_max)

        return np.array([
            rel_x, rel_y, rel_z,
            vel_x, vel_y, vel_z,
            geo.distance / dist_scale,
            geo.yaw_error / np.pi,
            geo.pitch_error / (np.pi / 2.0),
            geo.ata / np.pi,
            geo.aa / np.pi,
            1.0,
        ], dtype=np.float32)

    # -- global state (48-dim) ------------------------------------------

    def global_state(self) -> np.ndarray:
        battlefield = self.config["battlefield"]
        x_scale = battlefield["x_limit"]
        y_scale = battlefield["y_limit"]
        z_scale = battlefield["altitude_max"] - battlefield["altitude_min"]

        features = []
        for aid in ALL_IDS:
            a = self._aircraft_by_id(aid)
            pitch_scale = max(abs(a.spec.theta_min), abs(a.spec.theta_max))
            alive = 1.0 if a.state.alive else 0.0
            x_norm = a.state.x / x_scale
            y_norm = a.state.y / y_scale
            alt_norm = 2.0 * (a.state.altitude - battlefield["altitude_min"]) / z_scale - 1.0
            speed_norm = 2.0 * (a.state.v - a.spec.v_min) / (a.spec.v_max - a.spec.v_min) - 1.0
            pitch_norm = a.state.theta / pitch_scale
            features.extend([x_norm, y_norm, alt_norm, speed_norm, pitch_norm, np.sin(a.state.psi), np.cos(a.state.psi), alive])

        result = np.clip(np.asarray(features, dtype=np.float32), -1.0, 1.0)
        if not np.all(np.isfinite(result)):
            raise ValueError("global state must be finite")
        return result
