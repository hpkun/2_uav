"""Nearest-target pursuit policy for fixed-rule blue team in 3v3."""
import numpy as np

from .geometry import compute_pairwise_geometry
from .math_utils import safe_clip
from .models import Aircraft
from .rule_policy import PurePursuitPolicy


class NearestTargetPursuitPolicy3v3:
    """Per-step nearest-alive-red-target selection with pure pursuit action.

    Each alive blue aircraft selects the closest alive red aircraft by
    3-D Euclidean distance every step.  Ties are broken by ``aircraft_id``
    lexicographic order.  If no red is alive, the action is all zeros.
    """

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
