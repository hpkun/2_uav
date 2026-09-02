"""Small helpers for the fixed opponent-dynamics curriculum."""
from __future__ import annotations

from typing import Sequence


DEFAULT_CURRICULUM_SCHEDULE = (
    (0.00, 0.00),
    (0.25, 0.25),
    (0.50, 0.75),
    (0.75, 1.00),
)


def normalized_schedule(schedule: Sequence[Sequence[float]] | None) -> tuple[tuple[float, float], ...]:
    values = DEFAULT_CURRICULUM_SCHEDULE if schedule is None else tuple(
        (float(item[0]), float(item[1])) for item in schedule
    )
    if not values or values[0][0] != 0.0:
        raise ValueError("curriculum schedule must start at progress 0")
    if any(len(item) != 2 for item in values):
        raise ValueError("curriculum schedule entries must be [progress, p_nearest]")
    if any(not 0.0 <= boundary <= 1.0 or not 0.0 <= probability <= 1.0 for boundary, probability in values):
        raise ValueError("curriculum schedule values must lie in [0, 1]")
    boundaries = [item[0] for item in values]
    if boundaries != sorted(set(boundaries)):
        raise ValueError("curriculum schedule progress boundaries must be strictly increasing")
    return tuple(values)


def nearest_probability(
    sampled_steps: int,
    total_steps: int,
    schedule: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return the stage probability; a reached boundary starts the new stage."""
    if int(total_steps) <= 0:
        raise ValueError("curriculum_total_steps must be positive")
    progress = min(max(float(sampled_steps) / float(total_steps), 0.0), 1.0)
    selected = 0.0
    for boundary, probability in normalized_schedule(schedule):
        if progress < boundary:
            break
        selected = probability
    return float(selected)
