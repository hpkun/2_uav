"""Shared, environment-free trace and visualization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

HETERO_COMBAT_TRACE_SCHEMA_VERSION = 1
ENTITY_IDS = ("MAV", "UAV1", "UAV2", "Blue1", "Blue2")
FEATURES = ("x", "y", "h", "v", "theta", "psi")
STYLES = {
    "MAV": {"color": "#c5163a", "marker": "diamond", "mpl_marker": "D", "width": 3.0, "dash": "solid"},
    "UAV1": {"color": "#f28e2b", "marker": "circle", "mpl_marker": "o", "width": 2.0, "dash": "solid"},
    "UAV2": {"color": "#b5a000", "marker": "circle", "mpl_marker": "o", "width": 2.0, "dash": "solid"},
    "Blue1": {"color": "#4169e1", "marker": "triangle-up", "mpl_marker": "^", "width": 2.0, "dash": "dash"},
    "Blue2": {"color": "#00a6c7", "marker": "triangle-up", "mpl_marker": "^", "width": 2.0, "dash": "dash"},
}


def shortest_angle_interpolate(a0: np.ndarray, a1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    delta = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return a0 + alpha * delta


def validate_raw_trace(trace: dict[str, np.ndarray]) -> None:
    required = {"kinematics", "alive", "steps", "time_s", "red_actions", "team_reward",
                "team_situation", "event_reward", "terminal_reward",
                "minimum_friendly_red_distance", "red_safe_distance_violation"}
    missing = required - set(trace)
    if missing:
        raise ValueError(f"trace is missing arrays: {sorted(missing)}")
    f = len(trace["time_s"])
    if trace["kinematics"].shape != (f, 5, 6) or trace["alive"].shape != (f, 5):
        raise ValueError("trace kinematics/alive shape violates schema [F,5,6]/[F,5]")
    if trace["red_actions"].shape != (f - 1, 3, 3):
        raise ValueError("trace red_actions shape violates schema [F-1,3,3]")


def load_trace(input_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(input_dir).expanduser().resolve()
    with np.load(directory / "episode_trace.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("trace_schema_version") != HETERO_COMBAT_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported combat trace schema version")
    validate_raw_trace(trace)
    return trace, metadata


def interpolate_trace_for_visualization(
    trace: dict[str, np.ndarray], decision_dt: float, visual_dt: float = 0.1,
) -> dict[str, np.ndarray]:
    validate_raw_trace(trace)
    if visual_dt <= 0 or visual_dt > decision_dt:
        raise ValueError("visual_dt must be > 0 and <= decision_dt")
    raw_t = np.asarray(trace["time_s"], dtype=np.float64)
    end = float(raw_t[-1])
    count = int(np.floor(end / visual_dt + 1e-9))
    times = np.arange(count + 1, dtype=np.float64) * visual_dt
    if not np.isclose(times[-1], end):
        times = np.append(times, end)
    else:
        times[-1] = end
    right = np.searchsorted(raw_t, times, side="right")
    left = np.clip(right - 1, 0, len(raw_t) - 1)
    upper = np.clip(left + 1, 0, len(raw_t) - 1)
    denom = np.maximum(raw_t[upper] - raw_t[left], np.finfo(float).eps)
    alpha = np.where(upper == left, 0.0, (times - raw_t[left]) / denom)[:, None]
    k0, k1 = trace["kinematics"][left], trace["kinematics"][upper]
    output = k0.copy()
    output[:, :, :4] = k0[:, :, :4] + alpha[:, :, None] * (k1[:, :, :4] - k0[:, :, :4])
    for angle_index in (4, 5):
        output[:, :, angle_index] = shortest_angle_interpolate(
            k0[:, :, angle_index], k1[:, :, angle_index], alpha,
        )
    # left is the most recent real decision boundary: no early death/event leakage.
    alive = trace["alive"][left].copy()
    return {"kinematics": output, "alive": alive, "time_s": times,
            "raw_step": trace["steps"][left].astype(np.int64), "raw_frame": left.astype(np.int64)}


def death_records(trace: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for event in metadata.get("events", []):
        if event.get("type") != "death":
            continue
        frame = int(event["trace_frame"]); index = ENTITY_IDS.index(event["entity"])
        records.append({**event, "position": trace["kinematics"][frame, index, :3].tolist()})
    return records


def episode_ranges(kinematics: np.ndarray, alive: np.ndarray) -> dict[str, list[float]]:
    positions = kinematics[:, :, :3] / 1000.0
    ranges: dict[str, list[float]] = {}
    for axis, minimum_span in zip(range(3), (10.0, 10.0, 2.0)):
        values = positions[:, :, axis][alive]
        lo, hi = float(values.min()), float(values.max())
        span = max(hi - lo, minimum_span)
        center = (lo + hi) / 2.0
        ranges[("x", "y", "z")[axis]] = [center - span * 0.625, center + span * 0.625]
    return ranges

