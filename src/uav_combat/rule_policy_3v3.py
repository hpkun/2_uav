"""Nearest-target pursuit policy for fixed-rule blue team in 3v3."""
import numpy as np

from .geometry import compute_pairwise_geometry
from .math_utils import angle_difference, safe_clip
from .models import Aircraft
from .rule_policy import PurePursuitPolicy


class NearestTargetPursuitPolicy3v3:
    """Per-step nearest-alive-red-target selection with pure pursuit action.

    Each alive blue aircraft selects the closest alive red aircraft by
    3-D Euclidean distance every step.  Ties are broken by ``aircraft_id``
    lexicographic order.  If no red is alive, the action is all zeros.
    """

    policy_name = "paper_nearest_pursuit_v1"

    def __init__(
        self,
        delta_yaw_max: float,
        delta_pitch_max: float,
        delta_speed_max: float,
        mapping_mode: str = "legacy_delta",
        yaw_rate_max: float | None = None,
        pitch_rate_max: float | None = None,
        acceleration_max: float | None = None,
        k_yaw: float | None = None,
        k_pitch: float | None = None,
        k_speed: float | None = None,
    ) -> None:
        if mapping_mode not in ("legacy_delta", "rate_aligned_v1"):
            raise ValueError(f"unknown action mapping_mode: {mapping_mode}")
        self._pursuit = PurePursuitPolicy(delta_yaw_max, delta_pitch_max, delta_speed_max)
        self.mapping_mode = mapping_mode
        self.yaw_rate_max = yaw_rate_max
        self.pitch_rate_max = pitch_rate_max
        self.acceleration_max = acceleration_max
        self.k_yaw = k_yaw
        self.k_pitch = k_pitch
        self.k_speed = k_speed
        self.target_switch_count: dict[str, int] = {}
        self.target_selection_count: dict[str, int] = {}
        self.focus_fire_count: int = 0
        self._prev_targets: dict[str, str | None] = {}

    def _rate_aligned_action(self, own: Aircraft, target: Aircraft) -> np.ndarray:
        """Encode the same pursuit target as normalized rate-aligned actions."""
        yaw_rate_max = own.spec.yaw_rate_max if self.yaw_rate_max is None else self.yaw_rate_max
        pitch_rate_max = own.spec.pitch_rate_max if self.pitch_rate_max is None else self.pitch_rate_max
        acceleration_max = own.spec.acceleration_max if self.acceleration_max is None else self.acceleration_max
        k_yaw = own.spec.k_yaw if self.k_yaw is None else self.k_yaw
        k_pitch = own.spec.k_pitch if self.k_pitch is None else self.k_pitch
        k_speed = own.spec.k_speed if self.k_speed is None else self.k_speed

        eps = 1e-8
        geometry = compute_pairwise_geometry(own.state, target.state)
        desired_speed = safe_clip(
            target.state.v + self._pursuit.speed_margin,
            own.spec.v_min,
            own.spec.v_max,
        )
        return np.array([
            np.clip(k_yaw * geometry.yaw_error / max(abs(yaw_rate_max), eps), -1.0, 1.0),
            np.clip(k_pitch * geometry.pitch_error / max(abs(pitch_rate_max), eps), -1.0, 1.0),
            np.clip(k_speed * (desired_speed - own.state.v) / max(abs(acceleration_max), eps), -1.0, 1.0),
        ], dtype=float)

    def action(self, own: Aircraft, target: Aircraft) -> np.ndarray:
        """Return a pure-pursuit action encoded for the configured action mapping."""
        if self.mapping_mode == "legacy_delta":
            return self._pursuit.action(own, target)
        return self._rate_aligned_action(own, target)

    def select_actions(
        self, blue_aircraft: list[Aircraft], red_aircraft: list[Aircraft],
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        """Return (blue_actions, blue_targets) for one synchronous step."""
        alive_reds = [a for a in red_aircraft if a.state.alive]
        actions: dict[str, np.ndarray] = {}
        targets: dict[str, str | None] = {}
        target_counts: dict[str, int] = {}

        for blue in blue_aircraft:
            if not blue.state.alive:
                actions[blue.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[blue.aircraft_id] = None
                continue

            self.target_selection_count[blue.aircraft_id] = (
                self.target_selection_count.get(blue.aircraft_id, 0) + 1
            )

            if not alive_reds:
                actions[blue.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[blue.aircraft_id] = None
                continue

            # Nearest alive red (ties by aircraft_id)
            best_red = min(alive_reds, key=lambda r: (
                float(np.linalg.norm(blue.state.as_array()[:3] - r.state.as_array()[:3])),
                r.aircraft_id,
            ))
            targets[blue.aircraft_id] = best_red.aircraft_id
            target_counts[best_red.aircraft_id] = target_counts.get(best_red.aircraft_id, 0) + 1

            action = self.action(blue, best_red)
            actions[blue.aircraft_id] = np.asarray(action, dtype=np.float32)

        # Track switches and focus fire
        for blue_id, tgt in targets.items():
            prev = self._prev_targets.get(blue_id)
            if prev is not None and tgt is not None and prev != tgt:
                self.target_switch_count[blue_id] = self.target_switch_count.get(blue_id, 0) + 1
            self._prev_targets[blue_id] = tgt

        for count in target_counts.values():
            if count >= 2:
                self.focus_fire_count += 1

        return actions, targets

    def reset_counters(self) -> None:
        """Reset all per-episode tracking counters."""
        self._prev_targets.clear()
        self.target_switch_count.clear()
        self.target_selection_count.clear()
        self.focus_fire_count = 0


class GreedyTeamPursuitPolicy3v3(NearestTargetPursuitPolicy3v3):
    """Deterministic team-greedy target assignment with pure pursuit actions.

    The policy changes only target assignment: alive aircraft are greedily
    matched one-to-one when possible, then any extra own aircraft are allowed to
    focus the best remaining alive target.  Action generation is inherited from
    the verified pure-pursuit mapping used by the nearest-target policy.
    """

    policy_name = "greedy_team_pursuit_v1"

    def __init__(
        self,
        delta_yaw_max: float,
        delta_pitch_max: float,
        delta_speed_max: float,
        mapping_mode: str = "legacy_delta",
        yaw_rate_max: float | None = None,
        pitch_rate_max: float | None = None,
        acceleration_max: float | None = None,
        k_yaw: float | None = None,
        k_pitch: float | None = None,
        k_speed: float | None = None,
    ) -> None:
        super().__init__(
            delta_yaw_max,
            delta_pitch_max,
            delta_speed_max,
            mapping_mode=mapping_mode,
            yaw_rate_max=yaw_rate_max,
            pitch_rate_max=pitch_rate_max,
            acceleration_max=acceleration_max,
            k_yaw=k_yaw,
            k_pitch=k_pitch,
            k_speed=k_speed,
        )

    def _pair_score(self, own: Aircraft, target: Aircraft) -> tuple[float, str, str]:
        geometry = compute_pairwise_geometry(own.state, target.state)
        return (
            float(geometry.distance),
            own.aircraft_id,
            target.aircraft_id,
        )

    def _assign_targets(
        self,
        own_aircraft: list[Aircraft],
        enemy_aircraft: list[Aircraft],
    ) -> dict[str, Aircraft]:
        alive_own = sorted((a for a in own_aircraft if a.state.alive), key=lambda a: a.aircraft_id)
        alive_enemy = sorted((a for a in enemy_aircraft if a.state.alive), key=lambda a: a.aircraft_id)
        if not alive_own or not alive_enemy:
            return {}

        pairs = sorted(
            ((self._pair_score(own, enemy), own, enemy) for own in alive_own for enemy in alive_enemy),
            key=lambda item: item[0],
        )

        assignments: dict[str, Aircraft] = {}
        assigned_enemies: set[str] = set()
        first_round_limit = min(len(alive_own), len(alive_enemy))
        for _, own, enemy in pairs:
            if own.aircraft_id in assignments or enemy.aircraft_id in assigned_enemies:
                continue
            assignments[own.aircraft_id] = enemy
            assigned_enemies.add(enemy.aircraft_id)
            if len(assignments) >= first_round_limit:
                break

        if len(alive_own) > len(alive_enemy):
            for own in alive_own:
                if own.aircraft_id in assignments:
                    continue
                best_enemy = min(alive_enemy, key=lambda enemy: self._pair_score(own, enemy))
                assignments[own.aircraft_id] = best_enemy

        return assignments

    def select_actions(
        self, own_aircraft: list[Aircraft], enemy_aircraft: list[Aircraft],
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        """Return (actions, targets) for one synchronous team-greedy step."""
        actions: dict[str, np.ndarray] = {}
        targets: dict[str, str | None] = {}
        assignments = self._assign_targets(own_aircraft, enemy_aircraft)

        for own in own_aircraft:
            if not own.state.alive:
                actions[own.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[own.aircraft_id] = None
                continue

            target = assignments.get(own.aircraft_id)
            if target is None:
                actions[own.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[own.aircraft_id] = None
                continue

            targets[own.aircraft_id] = target.aircraft_id
            action = np.asarray(self.action(own, target), dtype=np.float32)
            actions[own.aircraft_id] = np.clip(action, -1.0, 1.0).astype(np.float32)

        return actions, targets

    def reset_counters(self) -> None:
        """Satisfy the vector-env rule-policy interface without state."""
        return None


class FunctionalHeterogeneousTeamPolicy3v3(NearestTargetPursuitPolicy3v3):
    """Role-aware deterministic rule policy for functional heterogeneous 3v3.

    Combat aircraft greedily match only effective-visible enemy targets.
    Support aircraft ignore enemies and hold a point behind alive teammate
    combat aircraft using the same normalized continuous action interface.
    """

    policy_name = "functional_heterogeneous_team_v1"

    def __init__(
        self,
        delta_yaw_max: float,
        delta_pitch_max: float,
        delta_speed_max: float,
        mapping_mode: str = "legacy_delta",
        yaw_rate_max: float | None = None,
        pitch_rate_max: float | None = None,
        acceleration_max: float | None = None,
        k_yaw: float | None = None,
        k_pitch: float | None = None,
        k_speed: float | None = None,
        support_follow_distance: float = 1200.0,
    ) -> None:
        super().__init__(
            delta_yaw_max,
            delta_pitch_max,
            delta_speed_max,
            mapping_mode=mapping_mode,
            yaw_rate_max=yaw_rate_max,
            pitch_rate_max=pitch_rate_max,
            acceleration_max=acceleration_max,
            k_yaw=k_yaw,
            k_pitch=k_pitch,
            k_speed=k_speed,
        )
        self.support_follow_distance = float(support_follow_distance)

    def _point_action(
        self,
        own: Aircraft,
        desired_point: np.ndarray,
        desired_altitude: float,
        desired_speed: float,
    ) -> np.ndarray:
        dx = float(desired_point[0] - own.state.x)
        dy = float(desired_point[1] - own.state.y)
        horizontal_distance = float(np.hypot(dx, dy))
        desired_z = -float(desired_altitude)
        dz = float(desired_z - own.state.z)
        los_yaw = float(np.arctan2(dy, dx))
        los_pitch = float(np.arctan2(-dz, horizontal_distance))
        yaw_error = angle_difference(los_yaw, own.state.psi)
        pitch_error = los_pitch - own.state.theta
        speed_error = safe_clip(desired_speed, own.spec.v_min, own.spec.v_max) - own.state.v

        if self.mapping_mode == "legacy_delta":
            eps = 1e-8
            return np.array([
                np.clip(yaw_error / max(abs(self._pursuit.effective_delta_yaw_max), eps), -1.0, 1.0),
                np.clip(pitch_error / max(abs(self._pursuit.delta_pitch_max), eps), -1.0, 1.0),
                np.clip(speed_error / max(abs(self._pursuit.delta_speed_max), eps), -1.0, 1.0),
            ], dtype=float)

        eps = 1e-8
        yaw_rate_max = own.spec.yaw_rate_max if self.yaw_rate_max is None else self.yaw_rate_max
        pitch_rate_max = own.spec.pitch_rate_max if self.pitch_rate_max is None else self.pitch_rate_max
        acceleration_max = own.spec.acceleration_max if self.acceleration_max is None else self.acceleration_max
        k_yaw = own.spec.k_yaw if self.k_yaw is None else self.k_yaw
        k_pitch = own.spec.k_pitch if self.k_pitch is None else self.k_pitch
        k_speed = own.spec.k_speed if self.k_speed is None else self.k_speed
        return np.array([
            np.clip(k_yaw * yaw_error / max(abs(yaw_rate_max), eps), -1.0, 1.0),
            np.clip(k_pitch * pitch_error / max(abs(pitch_rate_max), eps), -1.0, 1.0),
            np.clip(k_speed * speed_error / max(abs(acceleration_max), eps), -1.0, 1.0),
        ], dtype=float)

    def _support_action(self, support: Aircraft, own_aircraft: list[Aircraft]) -> np.ndarray:
        combats = [
            a for a in own_aircraft
            if a.role == "combat" and a.state.alive
        ]
        if not combats:
            return np.zeros(3, dtype=np.float32)
        positions = np.array([[a.state.x, a.state.y] for a in combats], dtype=float)
        centroid = positions.mean(axis=0)
        directions = np.array([
            [np.cos(a.state.psi) * np.cos(a.state.theta), np.sin(a.state.psi) * np.cos(a.state.theta)]
            for a in combats
        ], dtype=float)
        mean_direction = directions.mean(axis=0)
        norm = float(np.linalg.norm(mean_direction))
        if norm <= 1e-8:
            mean_direction = np.array([np.cos(support.state.psi), np.sin(support.state.psi)], dtype=float)
            norm = float(np.linalg.norm(mean_direction))
        mean_direction = mean_direction / max(norm, 1e-8)
        desired_xy = centroid - self.support_follow_distance * mean_direction
        desired_altitude = float(np.mean([a.state.altitude for a in combats]))
        desired_speed = float(np.mean([a.state.v for a in combats]))
        return np.asarray(
            self._point_action(support, desired_xy, desired_altitude, desired_speed),
            dtype=np.float32,
        )

    def _assign_combat_targets(
        self,
        own_aircraft: list[Aircraft],
        enemy_aircraft: list[Aircraft],
        visible_enemy_ids_by_own: dict[str, set[str]],
    ) -> dict[str, Aircraft]:
        combats = sorted(
            (a for a in own_aircraft if a.role == "combat" and a.state.alive),
            key=lambda a: a.aircraft_id,
        )
        enemies = {a.aircraft_id: a for a in enemy_aircraft if a.state.alive}
        pairs: list[tuple[tuple[float, str, str], Aircraft, Aircraft]] = []
        for own in combats:
            for enemy_id in sorted(visible_enemy_ids_by_own.get(own.aircraft_id, set())):
                enemy = enemies.get(enemy_id)
                if enemy is None:
                    continue
                distance = float(np.linalg.norm(own.state.as_array()[:3] - enemy.state.as_array()[:3]))
                pairs.append(((distance, own.aircraft_id, enemy.aircraft_id), own, enemy))
        pairs.sort(key=lambda item: item[0])
        assignments: dict[str, Aircraft] = {}
        assigned_enemies: set[str] = set()
        distinct_visible_enemies = {enemy.aircraft_id for _, _, enemy in pairs}
        first_round_limit = min(len(combats), len(distinct_visible_enemies))
        for _, own, enemy in pairs:
            if own.aircraft_id in assignments or enemy.aircraft_id in assigned_enemies:
                continue
            assignments[own.aircraft_id] = enemy
            assigned_enemies.add(enemy.aircraft_id)
            if len(assignments) >= first_round_limit:
                break
        if len(combats) > len(distinct_visible_enemies) and distinct_visible_enemies:
            for own in combats:
                if own.aircraft_id in assignments:
                    continue
                legal = [item for item in pairs if item[1].aircraft_id == own.aircraft_id]
                if legal:
                    assignments[own.aircraft_id] = legal[0][2]
        return assignments

    def select_actions(
        self,
        own_aircraft: list[Aircraft],
        enemy_aircraft: list[Aircraft],
        visible_enemy_ids_by_own: dict[str, set[str]] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        actions: dict[str, np.ndarray] = {}
        targets: dict[str, str | None] = {}
        visible = visible_enemy_ids_by_own or {}
        assignments = self._assign_combat_targets(own_aircraft, enemy_aircraft, visible)
        for own in own_aircraft:
            if not own.state.alive:
                actions[own.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[own.aircraft_id] = None
                continue
            if own.role == "support":
                action = self._support_action(own, own_aircraft)
                actions[own.aircraft_id] = np.clip(action, -1.0, 1.0).astype(np.float32)
                targets[own.aircraft_id] = None
                continue
            target = assignments.get(own.aircraft_id)
            if target is None:
                actions[own.aircraft_id] = np.zeros(3, dtype=np.float32)
                targets[own.aircraft_id] = None
                continue
            action = np.asarray(self.action(own, target), dtype=np.float32)
            actions[own.aircraft_id] = np.clip(action, -1.0, 1.0).astype(np.float32)
            targets[own.aircraft_id] = target.aircraft_id
        return actions, targets

    def reset_counters(self) -> None:
        self._prev_targets.clear()
        self.target_switch_count.clear()
        self.target_selection_count.clear()
        self.focus_fire_count = 0


def make_nearest_target_pursuit_policy_3v3(config: dict) -> NearestTargetPursuitPolicy3v3:
    """Create a 3v3 pursuit policy from the environment config."""
    act_cfg = config["action"]
    ac_cfg = config["aircraft"]
    return NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"],
        act_cfg["delta_pitch_max"],
        act_cfg["delta_speed_max"],
        mapping_mode=act_cfg.get("mapping_mode", "legacy_delta"),
        yaw_rate_max=ac_cfg["yaw_rate_max"],
        pitch_rate_max=ac_cfg["pitch_rate_max"],
        acceleration_max=ac_cfg["acceleration_max"],
        k_yaw=ac_cfg["k_yaw"],
        k_pitch=ac_cfg["k_pitch"],
        k_speed=ac_cfg["k_speed"],
    )


def _make_greedy_team_pursuit_policy_3v3(config: dict) -> GreedyTeamPursuitPolicy3v3:
    act_cfg = config["action"]
    ac_cfg = config["aircraft"]
    return GreedyTeamPursuitPolicy3v3(
        act_cfg["delta_yaw_max"],
        act_cfg["delta_pitch_max"],
        act_cfg["delta_speed_max"],
        mapping_mode=act_cfg.get("mapping_mode", "legacy_delta"),
        yaw_rate_max=ac_cfg["yaw_rate_max"],
        pitch_rate_max=ac_cfg["pitch_rate_max"],
        acceleration_max=ac_cfg["acceleration_max"],
        k_yaw=ac_cfg["k_yaw"],
        k_pitch=ac_cfg["k_pitch"],
        k_speed=ac_cfg["k_speed"],
    )


def _make_functional_heterogeneous_team_policy_3v3(config: dict) -> FunctionalHeterogeneousTeamPolicy3v3:
    act_cfg = config["action"]
    ac_cfg = config["aircraft"]
    support_cfg = config.get("heterogeneous", {}).get("support_rule", {})
    return FunctionalHeterogeneousTeamPolicy3v3(
        act_cfg["delta_yaw_max"],
        act_cfg["delta_pitch_max"],
        act_cfg["delta_speed_max"],
        mapping_mode=act_cfg.get("mapping_mode", "legacy_delta"),
        yaw_rate_max=ac_cfg["yaw_rate_max"],
        pitch_rate_max=ac_cfg["pitch_rate_max"],
        acceleration_max=ac_cfg["acceleration_max"],
        k_yaw=ac_cfg["k_yaw"],
        k_pitch=ac_cfg["k_pitch"],
        k_speed=ac_cfg["k_speed"],
        support_follow_distance=float(support_cfg.get("follow_distance", 1200.0)),
    )


def make_team_rule_policy_3v3(config: dict, team: str):
    """Create the configured deterministic 3v3 rule policy for a team."""
    if team not in ("blue", "red"):
        raise ValueError(f"team must be 'blue' or 'red', got {team!r}")
    mode = config.get(f"{team}_rule_policy", {}).get("mode", "paper_nearest_pursuit_v1")
    if mode == "paper_nearest_pursuit_v1":
        return make_nearest_target_pursuit_policy_3v3(config)
    if mode == "greedy_team_pursuit_v1":
        return _make_greedy_team_pursuit_policy_3v3(config)
    if mode == "functional_heterogeneous_team_v1":
        return _make_functional_heterogeneous_team_policy_3v3(config)
    raise ValueError(f"unknown {team}_rule_policy mode: {mode}")
