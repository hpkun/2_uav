"""Read-only terminal report for an existing recorded combat episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from tools.combat_visualization import ENTITY_IDS, load_trace


RED_IDS = ENTITY_IDS[:4]
UAV_IDS = RED_IDS[1:]
BLUE_IDS = ENTITY_IDS[4:]


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"min": None, "mean": None, "max": None}
    return {"min": float(array.min()), "mean": float(array.mean()), "max": float(array.max())}


def _death_map(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(event["entity"]): dict(event)
        for event in metadata.get("events", []) if event.get("type") == "death"
    }


def analyze_episode(trace: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    kinematics = np.asarray(trace["kinematics"])
    alive = np.asarray(trace["alive"])
    times = np.asarray(trace["time_s"])
    death = _death_map(metadata)
    attacks = [event for event in metadata.get("events", []) if event.get("type") == "attack"]
    warnings = [event for event in metadata.get("events", []) if event.get("type") == "red_separation_warning"]
    initial_red = kinematics[0, :4, :3].mean(axis=0)
    initial_blue = kinematics[0, 4:, :3].mean(axis=0)
    attack_axis = initial_blue - initial_red
    attack_axis /= max(float(np.linalg.norm(attack_axis)), 1e-12)

    agents: dict[str, Any] = {}
    projections: dict[str, list[float]] = {aid: [] for aid in RED_IDS}
    nearest_series: dict[str, list[float]] = {aid: [] for aid in RED_IDS}
    centroid_series: dict[str, list[float]] = {aid: [] for aid in RED_IDS}
    combat_range_series: dict[str, list[bool]] = {aid: [] for aid in RED_IDS}
    for aid in RED_IDS:
        index = ENTITY_IDS.index(aid)
        death_event = death.get(aid)
        valid_frames = np.flatnonzero(alive[:, index])
        if death_event is not None:
            death_frame = int(death_event["trace_frame"])
            if death_frame not in valid_frames:
                valid_frames = np.append(valid_frames, death_frame)
        nearest: list[float] = []; centroid: list[float] = []; first_combat = None
        for frame in valid_frames:
            blue_indices = [ENTITY_IDS.index(blue) for blue in BLUE_IDS if alive[frame, ENTITY_IDS.index(blue)]]
            if not blue_indices:
                continue
            position = kinematics[frame, index, :3]
            blue_positions = kinematics[frame, blue_indices, :3]
            distance = float(np.linalg.norm(blue_positions - position, axis=1).min())
            nearest.append(distance)
            centroid.append(float(np.linalg.norm(blue_positions.mean(axis=0) - position)))
            combat_range_series[aid].append(distance <= 3000.0)
            if first_combat is None and distance <= 3000.0:
                first_combat = float(times[frame])
            projections[aid].append(float(np.dot(position - initial_red, attack_axis)))
        nearest_series[aid] = nearest; centroid_series[aid] = centroid
        agent_attacks = [event for event in attacks if event.get("attacker") == aid]
        targets = sorted({str(event["target"]) for event in agent_attacks})
        agents[aid] = {
            "alive": bool(alive[-1, index]),
            "death_time_s": float(death_event["time_s"]) if death_event else None,
            "death_cause": death_event.get("cause") if death_event else "alive",
            "initial_position_m": kinematics[0, index, :3].tolist(),
            "final_position_m": kinematics[-1, index, :3].tolist(),
            "altitude_m": _stats(kinematics[valid_frames, index, 2]),
            "speed_mps": _stats(kinematics[valid_frames, index, 3]),
            "minimum_distance_to_alive_blue_m": min(nearest) if nearest else None,
            "mean_distance_to_alive_blue_centroid_m": float(np.mean(centroid)) if centroid else None,
            "first_within_3km_time_s": first_combat,
            "fraction_steps_within_3km": float(np.mean(combat_range_series[aid])) if combat_range_series[aid] else 0.0,
            "attack_event_count": len(agent_attacks), "distinct_attack_targets": targets,
            "first_attack_time_s": float(agent_attacks[0]["time_s"]) if agent_attacks else None,
            "final_attack_time_s": float(agent_attacks[-1]["time_s"]) if agent_attacks else None,
            "mean_attack_axis_projection_m": float(np.mean(projections[aid])) if projections[aid] else None,
        }

    common_frames = []
    mav_behind = []
    for frame in range(len(times)):
        if not alive[frame, 0]:
            continue
        uav_indices = [i for i in range(1, 4) if alive[frame, i]]
        if not uav_indices:
            continue
        mav_projection = float(np.dot(kinematics[frame, 0, :3] - initial_red, attack_axis))
        uav_projection = float(np.mean([
            np.dot(kinematics[frame, i, :3] - initial_red, attack_axis) for i in uav_indices
        ]))
        common_frames.append(frame); mav_behind.append(mav_projection < uav_projection)

    red_attack_events = [e for e in attacks if e.get("attacker") in RED_IDS]
    uav_attack_events = [e for e in red_attack_events if e.get("attacker") in UAV_IDS]
    near_simultaneous = 0
    for event in red_attack_events:
        if any(
            other is not event and other.get("target") == event.get("target")
            and other.get("attacker") != event.get("attacker")
            and abs(float(other["time_s"]) - float(event["time_s"])) <= float(metadata["decision_dt"])
            for other in red_attack_events
        ):
            near_simultaneous += 1
    uav_boundary_events = [e for e in death.values() if e.get("entity") in UAV_IDS and e.get("cause") == "boundary"]
    altitude_lower = 0
    lower = float(metadata["battlefield"]["altitude"][0])
    for event in uav_boundary_events:
        index = ENTITY_IDS.index(event["entity"]); frame = int(event["trace_frame"])
        altitude_lower += int(kinematics[frame, index, 2] < lower)

    snapshots = []
    interval = 10
    for frame in range(0, len(times), interval):
        alive_blues = [i for i in range(4, 8) if alive[frame, i]]
        centroid = kinematics[frame, alive_blues, :3].mean(axis=0) if alive_blues else None
        snapshots.append({
            "time_s": float(times[frame]),
            "alive_red": [RED_IDS[i] for i in range(4) if alive[frame, i]],
            "alive_blue": [ENTITY_IDS[i] for i in alive_blues],
            "mav_to_blue_centroid_m": (
                float(np.linalg.norm(kinematics[frame, 0, :3] - centroid))
                if centroid is not None and alive[frame, 0] else None
            ),
            "mean_uav_to_blue_centroid_m": (
                float(np.mean([np.linalg.norm(kinematics[frame, i, :3] - centroid)
                               for i in range(1, 4) if alive[frame, i]]))
                if centroid is not None and any(alive[frame, 1:4]) else None
            ),
        })

    dbm = None
    if "dbm_router_probabilities" in trace:
        router = np.asarray(trace["dbm_router_probabilities"])
        hard = np.asarray(trace["dbm_hard_mode_proxy"])
        divergence = np.asarray(trace["dbm_expert_divergence"])
        residual = np.asarray(trace["dbm_residual_magnitude"])
        dbm_agents = {}
        red_attack_steps = sorted({
            int(event["trace_frame"]) - 1 for event in attacks
            if event.get("attacker") in RED_IDS
        })
        first_attack_step = min(red_attack_steps) if red_attack_steps else len(router)
        for index, aid in enumerate(UAV_IDS):
            valid = np.isfinite(router[:, index, 0])
            probabilities = router[valid, index]
            hard_values = hard[valid, index]
            switches = int(np.sum(hard_values[1:] != hard_values[:-1])) if len(hard_values) > 1 else 0
            dbm_agents[aid] = {
                "active_steps": int(valid.sum()),
                "mean_router_p1": float(probabilities[:, 0].mean()) if len(probabilities) else None,
                "router_p1_variance": float(probabilities[:, 0].var()) if len(probabilities) else None,
                "hard_mode_proxy_switch_rate": switches / (len(hard_values) - 1) if len(hard_values) > 1 else 0.0,
                "mean_expert_divergence": float(np.nanmean(divergence[:, index])) if valid.any() else None,
                "mean_residual_magnitude": float(np.nanmean(residual[:, index])) if valid.any() else None,
                "pre_first_attack_router_p1_mean": (
                    float(np.nanmean(router[:first_attack_step, index, 0]))
                    if np.isfinite(router[:first_attack_step, index, 0]).any() else None
                ),
                "from_first_attack_router_p1_mean": (
                    float(np.nanmean(router[first_attack_step:, index, 0]))
                    if np.isfinite(router[first_attack_step:, index, 0]).any() else None
                ),
                "router_p1_at_attack_steps_mean": (
                    float(np.nanmean(router[red_attack_steps, index, 0]))
                    if red_attack_steps and np.isfinite(router[red_attack_steps, index, 0]).any() else None
                ),
                "residual_at_attack_steps_mean": (
                    float(np.nanmean(residual[red_attack_steps, index]))
                    if red_attack_steps and np.isfinite(residual[red_attack_steps, index]).any() else None
                ),
            }
        dbm = {
            "agents": dbm_agents,
            "phase_definition": "pre-first-Red-attack versus from-first-Red-attack; descriptive only",
            "red_attack_decision_steps_zero_based": red_attack_steps,
        }

    mav_nearest = agents["MAV"]["minimum_distance_to_alive_blue_m"]
    uav_nearest = [agents[aid]["minimum_distance_to_alive_blue_m"] for aid in UAV_IDS
                   if agents[aid]["minimum_distance_to_alive_blue_m"] is not None]
    return {
        "overview": {
            key: metadata.get(key) for key in (
                "algorithm", "method_variant", "training_seed", "checkpoint_sampled_steps",
                "evaluation_profile", "episode_seed", "action_mode", "action_seed", "outcome",
                "episode_length", "red_attack_kills", "blue_attack_kills", "mav_survived",
                "red_uav_survivors", "episode_return",
            )
        },
        "timeline": list(metadata.get("events", [])), "agents": agents,
        "spatial_roles": {
            "mav_mean_blue_centroid_distance_m": float(np.mean(centroid_series["MAV"])) if centroid_series["MAV"] else None,
            "uav_mean_blue_centroid_distance_m": float(np.mean([x for aid in UAV_IDS for x in centroid_series[aid]])),
            "mav_minimum_enemy_distance_m": mav_nearest,
            "mean_uav_minimum_enemy_distance_m": float(np.mean(uav_nearest)) if uav_nearest else None,
            "mav_behind_mean_alive_uav_fraction": float(np.mean(mav_behind)) if mav_behind else None,
            "mav_attack_axis_projection_mean_m": agents["MAV"]["mean_attack_axis_projection_m"],
            "uav_attack_axis_projection_mean_m": float(np.mean([
                agents[aid]["mean_attack_axis_projection_m"] for aid in UAV_IDS
                if agents[aid]["mean_attack_axis_projection_m"] is not None
            ])),
            "mav_fraction_steps_within_3km": agents["MAV"]["fraction_steps_within_3km"],
            "uav_fraction_steps_within_3km": float(np.mean([
                agents[aid]["fraction_steps_within_3km"] for aid in UAV_IDS
            ])),
        },
        "coordination": {
            "red_attack_event_count": len(red_attack_events),
            "mav_attack_event_count": agents["MAV"]["attack_event_count"],
            "uav_attack_event_count": len(uav_attack_events),
            "uav_attack_share": len(uav_attack_events) / len(red_attack_events) if red_attack_events else 0.0,
            "uav_distinct_targets": sorted({e["target"] for e in uav_attack_events}),
            "near_simultaneous_same_target_event_count": near_simultaneous,
            "uav_boundary_death_count": len(uav_boundary_events),
            "uav_altitude_lower_death_count": altitude_lower,
            "separation_warning_count": len(warnings),
        },
        "phase_snapshots": snapshots, "dbm": dbm,
    }


def format_report(report: Mapping[str, Any]) -> str:
    overview = report["overview"]
    lines = [
        f"{overview['algorithm']} | train seed={overview['training_seed']} | steps={overview['checkpoint_sampled_steps']}",
        f"profile={overview['evaluation_profile']} env_seed={overview['episode_seed']} "
        f"action={overview['action_mode']} action_seed={overview['action_seed']}",
        f"outcome={overview['outcome']} length={overview['episode_length']} "
        f"red_kills={overview['red_attack_kills']} blue_kills={overview['blue_attack_kills']} "
        f"MAV_survived={overview['mav_survived']} UAV_survivors={overview['red_uav_survivors']}",
        "", "Key event timeline:",
    ]
    for event in report["timeline"]:
        if event["type"] == "attack":
            text = f"{event['attacker']} -> {event['target']} ATTACK"
        elif event["type"] == "death":
            text = f"{event['entity']} DESTROYED/LOST [{event['cause']}]"
        else:
            text = f"RED SEPARATION WARNING ({event['minimum_distance']:.1f} m)"
        lines.append(f"t={float(event['time_s']):.0f}s {text}")
    lines.extend(("", "Red aircraft statistics:"))
    for aid, row in report["agents"].items():
        lines.append(
            f"{aid}: alive={row['alive']} death={row['death_cause']}@{row['death_time_s']} "
            f"alt[min/mean/max]={row['altitude_m']} speed[min/mean/max]={row['speed_mps']} "
            f"min_enemy={row['minimum_distance_to_alive_blue_m']}m first_3km={row['first_within_3km_time_s']}s "
            f"attacks={row['attack_event_count']} targets={row['distinct_attack_targets']}"
        )
    lines.extend(("", f"Spatial roles: {report['spatial_roles']}",
                  f"Coordination descriptors: {report['coordination']}"))
    if report["dbm"] is not None:
        lines.append(f"DBM execution diagnostics: {report['dbm']}")
    return "\n".join(lines)


def report_directory(input_dir: Path, *, write_json: bool = True, write_text: bool = True) -> dict[str, Any]:
    trace, metadata = load_trace(input_dir)
    report = analyze_episode(trace, metadata)
    directory = Path(input_dir).resolve()
    if write_json:
        (directory / "combat_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_text:
        (directory / "combat_report.txt").write_text(format_report(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report = report_directory(args.input_dir, write_json=not args.no_write, write_text=not args.no_write)
    print(format_report(report))


if __name__ == "__main__":
    main()
