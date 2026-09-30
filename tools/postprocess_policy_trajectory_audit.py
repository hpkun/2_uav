"""Postprocess an existing RGAA/Wide/DBM trajectory audit without rollout."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from tools.audit_policy_trajectories import checkpoint_sha256, validate_run_matrix
from tools.render_combat_episode import render_episode
from tools.render_combat_episode_interactive import render_interactive

METHODS = ("rgaa", "rgaa_wide", "dbm_rgaa")
RED_IDS = ("MAV", "UAV1", "UAV2", "UAV3")
UAV_IDS = RED_IDS[1:]
BLUE_IDS = ("Blue1", "Blue2", "Blue3", "Blue4")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _number(row: Mapping[str, Any], key: str, cast: type = float) -> Any:
    return cast(row[key])


def load_checkpoint_contracts(audit_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read checkpoint metadata and hashes only; actors and environments are never constructed."""
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    evaluation_profile = summary.get("protocol", {}).get("profile")
    if not evaluation_profile:
        raise RuntimeError("existing audit summary is missing protocol.profile")
    contracts, integrity = [], []
    for label, old in summary["runs"].items():
        checkpoint = Path(old["checkpoint"])
        before = checkpoint_sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("trainer_config", payload.get("config", {}))
        contracts.append({
            "run": label,
            "method_variant": payload.get("method_variant"),
            "training_seed": int(config["seed"]),
            "sampled_steps": int(payload.get("sampled_steps", 0)),
            "training_profile": payload.get("environment_profile"),
            "evaluation_profile": evaluation_profile,
            "environment_version": payload.get("environment_version"),
            "environment_config": payload.get("environment_config"),
        })
        integrity.append({
            "run": label, "method": payload.get("method_variant"),
            "training_seed": int(config["seed"]), "checkpoint_path": str(checkpoint),
            "sha256_before": before,
        })
        del payload
    validate_run_matrix(contracts)
    return contracts, integrity


def select_paired_representative(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if row["action_mode"] == "deterministic" and row["outcome"] == "red"]
    medians = {
        method: float(np.median([float(row["episode_length"]) for row in wins if row["method_variant"] == method]))
        for method in METHODS
    }
    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in wins:
        key = (int(row["training_seed"]), int(row["environment_seed"]))
        method = row["method_variant"]
        if method in grouped[key]:
            raise RuntimeError(f"duplicate paired episode for {key} and {method}")
        grouped[key][method] = row
    candidates = []
    for (training_seed, environment_seed), group in grouped.items():
        if set(group) != set(METHODS):
            continue
        score = sum(abs(float(group[m]["episode_length"]) - medians[m]) for m in METHODS)
        candidates.append((score, training_seed, environment_seed, group))
    if not candidates:
        raise RuntimeError("no deterministic all-red-win paired representative candidate")
    score, training_seed, environment_seed, group = min(candidates, key=lambda value: value[:3])
    return {
        "selection_rule": "same training/environment seed, deterministic all-method Red wins; minimize sum absolute distance to method-specific deterministic Red-win median length; ties by training seed then environment seed",
        "training_seed": training_seed, "environment_seed": environment_seed,
        "score": float(score), "method_win_length_medians": medians,
        "methods": {method: {
            "run": group[method]["run"], "episode_dir": group[method]["episode_dir"],
            "outcome": group[method]["outcome"], "episode_length": int(group[method]["episode_length"]),
            "red_attack_kills": int(group[method]["red_attack_kills"]),
            "blue_attack_kills": int(group[method]["blue_attack_kills"]),
            "mav_survived": int(group[method]["mav_survived"]),
            "uav_survivors": int(group[method]["uav_survivors"]),
        } for method in METHODS},
    }


def multi_target_terminal_events(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    attacks: dict[tuple[int, str], set[str]] = defaultdict(set)
    deaths: dict[int, set[str]] = defaultdict(set)
    times: dict[int, float] = {}
    for event in metadata.get("events", []):
        frame = int(event["trace_frame"])
        times[frame] = float(event["time_s"])
        if event["type"] == "attack" and event.get("attacker") in RED_IDS and event.get("target") in BLUE_IDS:
            attacks[(frame, event["attacker"])].add(event["target"])
        elif event["type"] == "death" and event.get("entity") in BLUE_IDS and event.get("cause") == "red_attack":
            deaths[frame].add(event["entity"])
    output = []
    for (frame, attacker), targets in sorted(attacks.items()):
        terminal_targets = sorted(targets & deaths.get(frame, set()))
        if len(terminal_targets) >= 2:
            output.append({
                "decision_step": frame, "time_s": times[frame], "attacker": attacker,
                "distinct_targets": len(terminal_targets), "target_count": len(terminal_targets),
                "targets": "|".join(terminal_targets),
                "same_frame_red_attack_deaths": len(deaths[frame]),
            })
    return output


def summarize_multi_target(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details, summary = [], []
    episode_records = []
    for row in rows:
        metadata = json.loads((Path(row["episode_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        events = multi_target_terminal_events(metadata)
        enriched = []
        for event in events:
            item = {
                "method": row["method_variant"], "training_seed": int(row["training_seed"]),
                "action_mode": row["action_mode"], "environment_seed": int(row["environment_seed"]),
                "action_seed": "" if row.get("action_seed") in (None, "", "None") else int(row["action_seed"]),
                **event,
            }
            details.append(item); enriched.append(item)
        death_frames: dict[int, set[str]] = defaultdict(set)
        for event in metadata.get("events", []):
            if event["type"] == "death" and event.get("entity") in BLUE_IDS and event.get("cause") == "red_attack":
                death_frames[int(event["trace_frame"])].add(event["entity"])
        qualifying_frames = {event["decision_step"] for event in enriched}
        episode_records.append((
            row, enriched, sum(map(len, death_frames.values())),
            sum(len(death_frames[frame]) for frame in qualifying_frames),
        ))
    for mode in ("deterministic", "stochastic"):
        for method in METHODS:
            selected = [record for record in episode_records if record[0]["action_mode"] == mode and record[0]["method_variant"] == method]
            method_events = [event for _, events, _, _ in selected for event in events]
            all_deaths = sum(value for _, _, value, _ in selected)
            frame_deaths = sum(value for _, _, _, value in selected)
            counts = Counter(event["attacker"] for event in method_events)
            episode_count = len(selected)
            summary.append({
                "action_mode": mode, "method": method, "total_episodes": episode_count,
                "episodes_with_event": sum(bool(events) for _, events, _, _ in selected),
                "episode_rate": sum(bool(events) for _, events, _, _ in selected) / episode_count if episode_count else 0.0,
                "total_decision_steps": len({(e["training_seed"], e["environment_seed"], e["action_seed"], e["decision_step"]) for e in method_events}),
                "all_red_attack_blue_deaths": all_deaths,
                "multi_target_frame_blue_deaths": frame_deaths,
                "blue_death_share": frame_deaths / all_deaths if all_deaths else 0.0,
                **{f"attacker_events_{aid}": counts[aid] for aid in RED_IDS},
                "max_distinct_targets": max((event["distinct_targets"] for event in method_events), default=0),
            })
    return summary, details


def summarize_attack_concentration(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe direct attack-event concentration without assigning causal support roles."""
    records = []
    for row in rows:
        metadata = json.loads((Path(row["episode_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        counts = Counter(
            event["attacker"] for event in metadata.get("events", [])
            if event["type"] == "attack" and event.get("attacker") in RED_IDS
        )
        total = sum(counts.values())
        uav_contributors = sum(counts[aid] > 0 for aid in UAV_IDS)
        records.append((row, counts, total, uav_contributors))
    output = []
    for mode in ("deterministic", "stochastic"):
        for method in METHODS:
            selected = [r for r in records if r[0]["action_mode"] == mode and r[0]["method_variant"] == method]
            if not selected:
                continue
            output.append({
                "action_mode": mode, "method": method, "episodes": len(selected),
                **{f"mean_attack_events_{aid}": float(np.mean([counts[aid] for _, counts, _, _ in selected])) for aid in RED_IDS},
                "mean_contributing_uavs": float(np.mean([contributors for _, _, _, contributors in selected])),
                "single_attacker_episode_rate": float(np.mean([sum(counts[aid] > 0 for aid in RED_IDS) == 1 for _, counts, _, _ in selected])),
                "mean_dominant_attacker_share": float(np.mean([
                    max(counts.values()) / total if total else 0.0 for _, counts, total, _ in selected
                ])),
            })
    return output


def _initial_states(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(episode_dir / "episode_trace.npz") as trace:
        return trace["kinematics"][0].copy(), trace["alive"][0].copy()


def _router_summary(episode_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    red_attacks = [e for e in metadata["events"] if e["type"] == "attack" and e.get("attacker") in RED_IDS]
    first = min((int(e["trace_frame"]) for e in red_attacks), default=None)
    final = max((int(e["trace_frame"]) for e in red_attacks), default=None)
    with np.load(episode_dir / "episode_trace.npz") as trace:
        probabilities = trace["dbm_router_probabilities"]
        p1 = probabilities[:, :, 0]
        p2 = probabilities[:, :, 1]
        hard = trace["dbm_hard_mode_proxy"]
        divergence = trace["dbm_expert_divergence"]
        residual = trace["dbm_residual_magnitude"]
    result: dict[str, Any] = {"agents": {}, "attacks": []}
    for index, aid in enumerate(UAV_IDS):
        valid = np.isfinite(p1[:, index]) & np.isfinite(p2[:, index])
        values_p1 = p1[:, index][valid]
        values_p2 = p2[:, index][valid]
        hard_values = hard[:, index][valid]
        expected_hard = np.argmax(probabilities[:, index][valid], axis=-1) + 1
        if not np.array_equal(hard_values, expected_hard):
            raise RuntimeError(f"DBM hard mode proxy is inconsistent with router probabilities for {aid}")
        phases = {
            "before_first_red_attack": np.arange(len(probabilities)) < (first - 1 if first is not None else len(probabilities)),
            "first_through_final_red_attack": ((np.arange(len(probabilities)) >= first - 1) & (np.arange(len(probabilities)) <= final - 1)) if first is not None else np.zeros(len(probabilities), bool),
            "after_final_red_attack": (np.arange(len(probabilities)) > final - 1) if final is not None else np.zeros(len(probabilities), bool),
        }
        result["agents"][aid] = {
            "mean_p1": float(np.mean(values_p1)), "mean_p2": float(np.mean(values_p2)),
            "variance_p1": float(np.var(values_p1)), "variance_p2": float(np.var(values_p2)),
            "hard_proxy_switch_rate": float(np.mean(hard_values[1:] != hard_values[:-1])) if len(hard_values) > 1 else 0.0,
            "mean_expert_divergence": float(np.nanmean(divergence[:, index])),
            "mean_residual_magnitude": float(np.nanmean(residual[:, index])),
            "phases": {
                name: {
                    "sample_count": int(np.sum(mask & valid)),
                    "mean_p1": (float(np.nanmean(p1[mask, index])) if np.any(mask & valid) else None),
                    "mean_p2": (float(np.nanmean(p2[mask, index])) if np.any(mask & valid) else None),
                } for name, mask in phases.items()
            },
        }
    for event in red_attacks:
        attacker = event["attacker"]
        if attacker not in UAV_IDS:
            continue
        index, frame = UAV_IDS.index(attacker), int(event["trace_frame"])
        step = frame - 1
        result["attacks"].append({
            "time_s": float(event["time_s"]), "attacker": attacker, "target": event["target"],
            "router_p1": float(p1[step, index]), "router_p2": float(p2[step, index]),
            "hard_proxy": int(hard[step, index]),
            "expert_divergence": float(divergence[step, index]),
            "residual_magnitude": float(residual[step, index]),
        })
    return result


def build_paired_comparison(paired: Mapping[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"initial_states_identical": True, "methods": {}}
    initial = None
    for method in METHODS:
        directory = Path(paired["methods"][method]["episode_dir"])
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "combat_report.json").read_text(encoding="utf-8"))
        state, initial_alive = _initial_states(directory)
        if initial is None:
            initial = (state, initial_alive)
        elif not (np.array_equal(initial[0], state) and np.array_equal(initial[1], initial_alive)):
            comparison["initial_states_identical"] = False
        comparison["methods"][method] = {
            "team": report["overview"], "mav": report["agents"]["MAV"],
            "uavs": {aid: report["agents"][aid] for aid in UAV_IDS},
            "timeline": report["timeline"], "ten_second_distances": report["phase_snapshots"],
            "initial_aircraft_states": {
                aid: {"kinematics": state[index].tolist(), "alive": bool(initial_alive[index])}
                for index, aid in enumerate(metadata["entity_ids"])
            },
        }
        if method == "dbm_rgaa":
            comparison["methods"][method]["router"] = _router_summary(directory, metadata)
        if not (directory / "preview.png").is_file():
            render_episode(directory, preview=directory / "preview.png", mp4=False)
        if not (directory / "episode_interactive.html").is_file():
            render_interactive(directory)
    if not comparison["initial_states_identical"]:
        raise RuntimeError("paired representative initial aircraft states are not identical")
    return comparison


def _paired_text(paired: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    lines = [
        "Paired deterministic comparison",
        f"training seed: {paired['training_seed']}", f"environment seed: {paired['environment_seed']}",
        f"initial states identical: {comparison['initial_states_identical']}", "",
    ]
    for method in METHODS:
        data = comparison["methods"][method]; team = data["team"]; mav = data["mav"]
        lines += [f"[{method}]", f"outcome={team['outcome']} length={team['episode_length']} return={team['episode_return']:.6f} red_kills={team['red_attack_kills']} blue_kills={team['blue_attack_kills']}",
                  f"MAV min_blue={mav['minimum_distance_to_alive_blue_m']:.3f} mean_centroid={mav['mean_distance_to_alive_blue_centroid_m']:.3f} within3km={mav['fraction_steps_within_3km']:.6f} attacks={mav['attack_event_count']} alive={mav['alive']} projection={mav['mean_attack_axis_projection_m']:.3f}"]
        for aid, agent in data["uavs"].items():
            lines.append(f"{aid}: attacks={agent['attack_event_count']} targets={agent['distinct_attack_targets']} first={agent['first_attack_time_s']} final={agent['final_attack_time_s']} min_blue={agent['minimum_distance_to_alive_blue_m']:.3f} within3km={agent['fraction_steps_within_3km']:.6f} death={agent['death_cause']} alive={agent['alive']}")
        lines.append("timeline:")
        for event in data["timeline"]:
            lines.append("  " + json.dumps(event, ensure_ascii=False, sort_keys=True))
        lines.append("10-s distances:")
        for snapshot in data["ten_second_distances"]:
            lines.append(f"  t={snapshot['time_s']:.0f} MAV={snapshot['mav_to_blue_centroid_m']:.3f} mean_UAV={snapshot['mean_uav_to_blue_centroid_m']:.3f}")
        if method == "dbm_rgaa":
            router = data["router"]
            lines.append("DBM router (mode order: index 0=p1/mode1, index 1=p2/mode2):")
            for aid, stats in router["agents"].items():
                lines.append(
                    f"  {aid}: mean_p1={stats['mean_p1']:.6f} mean_p2={stats['mean_p2']:.6f} "
                    f"var_p1={stats['variance_p1']:.6f} var_p2={stats['variance_p2']:.6f} "
                    f"switch={stats['hard_proxy_switch_rate']:.6f}"
                )
                for phase, values in stats["phases"].items():
                    p1_text = "N/A" if values["mean_p1"] is None else f"{values['mean_p1']:.6f}"
                    p2_text = "N/A" if values["mean_p2"] is None else f"{values['mean_p2']:.6f}"
                    lines.append(f"    {phase}: n={values['sample_count']} mean_p1={p1_text} mean_p2={p2_text}")
            lines.append("  attack-time router diagnostics:")
            for attack in router["attacks"]:
                lines.append(
                    f"    t={attack['time_s']:.0f} {attack['attacker']}->{attack['target']} "
                    f"p1={attack['router_p1']:.6f} p2={attack['router_p2']:.6f} "
                    f"hard={attack['hard_proxy']} divergence={attack['expert_divergence']:.6f} "
                    f"residual={attack['residual_magnitude']:.6f}"
                )
        lines.append("")
    return "\n".join(lines)


def postprocess(audit_dir: Path) -> dict[str, Any]:
    audit_dir = audit_dir.resolve()
    rows = _read_csv(audit_dir / "episode_index.csv")
    source_summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    source_profile = source_summary.get("protocol", {}).get("profile")
    if not source_profile:
        raise RuntimeError("existing audit summary is missing protocol.profile")
    for row in rows:
        metadata = json.loads((Path(row["episode_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("evaluation_profile") != source_profile:
            raise RuntimeError("episode evaluation profile does not match existing audit protocol.profile")
    contracts, integrity = load_checkpoint_contracts(audit_dir)
    paired = select_paired_representative(rows)
    comparison = build_paired_comparison(paired)
    multi_summary, multi_events = summarize_multi_target(rows)
    attack_concentration = summarize_attack_concentration(rows)
    for record in integrity:
        record["sha256_after"] = checkpoint_sha256(Path(record["checkpoint_path"]))
        record["unchanged"] = record["sha256_before"] == record["sha256_after"]
        if not record["unchanged"]:
            raise RuntimeError(f"checkpoint changed during postprocessing: {record['checkpoint_path']}")
    (audit_dir / "checkpoint_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (audit_dir / "paired_representative.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (audit_dir / "paired_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (audit_dir / "paired_comparison.txt").write_text(_paired_text(paired, comparison), encoding="utf-8")
    _write_csv(audit_dir / "multi_target_terminal_summary.csv", multi_summary)
    _write_csv(audit_dir / "multi_target_terminal_events.csv", multi_events, fields=(
        "method", "training_seed", "action_mode", "environment_seed", "action_seed",
        "decision_step", "time_s", "attacker", "distinct_targets", "target_count", "targets",
        "same_frame_red_attack_deaths",
    ))
    _write_csv(audit_dir / "attack_concentration_summary.csv", attack_concentration)
    result = {"contracts": contracts, "checkpoint_integrity": integrity, "paired": paired,
              "paired_comparison": comparison, "multi_target_summary": multi_summary,
              "attack_concentration": attack_concentration}
    (audit_dir / "postprocess_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-audit-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = postprocess(args.existing_audit_dir)
    print(json.dumps({"paired": result["paired"], "multi_target_summary": result["multi_target_summary"]}, indent=2))


if __name__ == "__main__":
    main()
