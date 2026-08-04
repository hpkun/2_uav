"""Deterministic fixed-rule policies for functional heterogeneous 4v3 v9."""
from __future__ import annotations

import numpy as np

from .geometry import compute_pairwise_geometry
from .formation_4v3 import compute_red_combat_formation_reference
from .math_utils import angle_difference
from .models import Aircraft
from .rule_policy_3v3 import NearestTargetPursuitPolicy3v3


class FunctionalHeterogeneous4v3RulePolicy(NearestTargetPursuitPolicy3v3):
    """Simple role-aware pure-pursuit policy for the v9 4v3 environment.

    Combat aircraft pursue the nearest currently effective-visible target.
    The red support aircraft holds a point behind the red combat centroid.
    Blue has no sharing; every blue combat aircraft uses direct visibility only.
    """

    policy_name = "functional_heterogeneous_4v3_nearest_pursuit_v9"

    def __init__(self, *args, team: str = "blue", **kwargs) -> None:
        self.support_hold_distance = float(kwargs.pop("support_hold_distance", 1200.0))
        self._formation = kwargs.pop("formation", {})
        super().__init__(*args, **kwargs)
        if team not in ("red", "blue"):
            raise ValueError("team must be 'red' or 'blue'")
        self.team = team

    def _zero(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def _visible(self, own: Aircraft, target: Aircraft, visibility: dict[str, set[str]] | None) -> bool:
        if visibility is None:
            return True
        return target.aircraft_id in visibility.get(own.aircraft_id, set())

    def _support_hold_action(self, support: Aircraft, own_aircraft: list[Aircraft]) -> np.ndarray:
        alive_combat = [a for a in own_aircraft if a.role == "combat" and a.state.alive]
        if not alive_combat:
            return self._zero()
        reference = compute_red_combat_formation_reference(
            support,
            alive_combat,
            direction_validity_threshold=float(self._formation.get("direction_validity_threshold", 1e-6)),
        )
        centroid = reference["centroid"]
        sx, sy, sz = support.state.as_array()[:3]
        if not reference["direction_valid"]:
            return self._zero()
        desired = centroid - np.pad(reference["horizontal_direction"], (0, 1)) * self.support_hold_distance
        los = desired - np.array([sx, sy, sz])
        desired_psi = float(np.arctan2(los[1], los[0]))
        horizontal = float(np.linalg.norm(los[:2]))
        desired_theta = float(np.arctan2(-los[2], max(horizontal, 1e-6)))
        yaw_error = angle_difference(desired_psi, support.state.psi)
        pitch_error = desired_theta - support.state.theta
        desired_speed = float(np.clip(np.mean([a.state.v for a in alive_combat]), support.spec.v_min, support.spec.v_max))

        if self.mapping_mode == "legacy_delta":
            return np.array([
                np.clip(yaw_error / max(self._pursuit.effective_delta_yaw_max, 1e-8), -1.0, 1.0),
                np.clip(pitch_error / max(self._pursuit.delta_pitch_max, 1e-8), -1.0, 1.0),
                np.clip((desired_speed - support.state.v) / max(self._pursuit.delta_speed_max, 1e-8), -1.0, 1.0),
            ], dtype=np.float32)

        return np.array([
            np.clip(support.spec.k_yaw * yaw_error / max(abs(support.spec.yaw_rate_max), 1e-8), -1.0, 1.0),
            np.clip(support.spec.k_pitch * pitch_error / max(abs(support.spec.pitch_rate_max), 1e-8), -1.0, 1.0),
            np.clip(support.spec.k_speed * (desired_speed - support.state.v) / max(abs(support.spec.acceleration_max), 1e-8), -1.0, 1.0),
        ], dtype=np.float32)

    def select_actions(
        self,
        own_aircraft: list[Aircraft],
        enemy_aircraft: list[Aircraft],
        visibility: dict[str, set[str]] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        actions: dict[str, np.ndarray] = {}
        targets: dict[str, str | None] = {}
        alive_enemies = [e for e in enemy_aircraft if e.state.alive]

        for own in sorted(own_aircraft, key=lambda a: a.aircraft_id):
            if not own.state.alive:
                actions[own.aircraft_id] = self._zero()
                targets[own.aircraft_id] = None
                continue
            if own.role == "support":
                actions[own.aircraft_id] = self._support_hold_action(own, own_aircraft)
                targets[own.aircraft_id] = None
                continue
            candidates = [e for e in alive_enemies if self._visible(own, e, visibility)]
            if not candidates:
                actions[own.aircraft_id] = self._zero()
                targets[own.aircraft_id] = None
                continue
            target = min(candidates, key=lambda e: (
                float(compute_pairwise_geometry(own.state, e.state).distance),
                e.aircraft_id,
            ))
            actions[own.aircraft_id] = np.clip(np.asarray(self.action(own, target), dtype=np.float32), -1.0, 1.0)
            targets[own.aircraft_id] = target.aircraft_id

        return actions, targets


def make_rule_policy_4v3(config: dict, team: str) -> FunctionalHeterogeneous4v3RulePolicy:
    action = config["action"]
    aircraft = config["aircraft"]
    formation = config["support_formation"]
    mode = config.get(f"{team}_rule_policy", {}).get("mode", "functional_heterogeneous_4v3_nearest_pursuit_v9")
    if mode != "functional_heterogeneous_4v3_nearest_pursuit_v9":
        raise ValueError(f"unsupported 4v3 rule policy mode for {team}: {mode!r}")
    return FunctionalHeterogeneous4v3RulePolicy(
        delta_yaw_max=float(action["delta_yaw_max"]),
        delta_pitch_max=float(action["delta_pitch_max"]),
        delta_speed_max=float(action["delta_speed_max"]),
        mapping_mode=str(action.get("mapping_mode", "legacy_delta")),
        yaw_rate_max=float(aircraft["yaw_rate_max"]),
        pitch_rate_max=float(aircraft["pitch_rate_max"]),
        acceleration_max=float(aircraft["acceleration_max"]),
        k_yaw=float(aircraft["k_yaw"]),
        k_pitch=float(aircraft["k_pitch"]),
        k_speed=float(aircraft["k_speed"]),
        support_hold_distance=float(formation.get("rule_hold_distance", 1200.0)),
        formation=formation,
        team=team,
    )
