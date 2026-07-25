"""Nearest-target pursuit policy for fixed-rule blue team in 3v3."""
import numpy as np

from .models import Aircraft
from .rule_policy import PurePursuitPolicy


class NearestTargetPursuitPolicy3v3:
    """Per-step nearest-alive-red-target selection with pure pursuit action.

    Each alive blue aircraft selects the closest alive red aircraft by
    3-D Euclidean distance every step.  Ties are broken by ``aircraft_id``
    lexicographic order.  If no red is alive, the action is all zeros.
    """

    def __init__(self, delta_yaw_max: float, delta_pitch_max: float, delta_speed_max: float) -> None:
        self._pursuit = PurePursuitPolicy(delta_yaw_max, delta_pitch_max, delta_speed_max)
        self.target_switch_count: dict[str, int] = {}
        self.target_selection_count: dict[str, int] = {}
        self.focus_fire_count: int = 0
        self._prev_targets: dict[str, str | None] = {}

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

            action = self._pursuit.action(blue, best_red)
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
        """Reset per-episode tracking counters."""
        self._prev_targets.clear()
