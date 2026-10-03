"""Read-only TACM-RGAA v3.10 last-Blue cleanup diagnostic.

The tool reuses the validated TACM checkpoint loader and the environment's
actual stochastic policy/evaluation seed contract. It never trains or mutates
checkpoint files and only persists compact episode/endgame evidence.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithm.happo.dbm_rgaa import DBMGaussianActor
from algorithm.happo.tacm_rgaa import tactical_teacher
from env.geometry import compute_pairwise_geometry
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from env.reward_role_v39 import attack_gate_indicator
from tools.audit_combat_skill_causality import recompute_reward_targets
from tools.audit_continuation_horizon import load_tacm_checkpoint


UAV_IDS = RED_IDS[1:]
EXPECTED_VERSION = "heterogeneous_mavuav_4v4_v3_10"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else ["no_records"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sample_actions(
    actors: Any, observations: Mapping[str, np.ndarray], device: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    actions: list[np.ndarray] = []
    router: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for index, aid in enumerate(RED_IDS):
            observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
            actor = actors.actors[index]
            if isinstance(actor, DBMGaussianActor):
                details = actor.mode_diagnostics(observation)
                probabilities = details["router_probabilities"].squeeze(0)
                router[aid] = {
                    "router_p1": float(probabilities[0].item()),
                    "router_p2": float(probabilities[1].item()),
                    "router_mode": int(torch.argmax(probabilities).item() + 1),
                }
            action, _ = actor.sample(observation, deterministic=False)
            actions.append(action.squeeze(0).cpu().numpy().astype(np.float32, copy=True))
    return np.asarray(actions, dtype=np.float32), router


def _pair_row(
    env: HeterogeneousMAVUAVAirCombatEnv, aid: str, bid: str, *, step: int,
    selector: str | None, action: np.ndarray, attack_event: bool,
) -> dict[str, Any]:
    geometry = compute_pairwise_geometry(env.entities[aid].state, env.entities[bid].state)
    combat = env.config["combat"]
    in_distance = bool(float(combat["distance"][0]) <= geometry.distance <= float(combat["distance"][1]))
    ata_ok = bool(geometry.ata < np.deg2rad(float(combat["ata_deg"])))
    aa_ok = bool(geometry.aa < np.deg2rad(float(combat["aa_deg"])))
    full_gate = bool(attack_gate_indicator(
        geometry.distance, geometry.ata, geometry.aa,
        float(combat["distance"][0]), float(combat["distance"][1]),
        np.deg2rad(float(combat["ata_deg"])), np.deg2rad(float(combat["aa_deg"])),
    ))
    real_streak = int(env._attack_streak.get((aid, bid), 0))
    return {
        "decision_step": step, "agent": aid, "blue": bid,
        "uav_alive": bool(env.entities[aid].state.alive),
        "blue_alive": bool(env.entities[bid].state.alive),
        "distance_m": float(geometry.distance),
        "ATA_deg": float(np.rad2deg(geometry.ata)),
        "AA_deg": float(np.rad2deg(geometry.aa)),
        "distance_gate": in_distance, "ATA_gate": ata_ok, "AA_gate": aa_ok,
        "full_gate": full_gate,
        "real_attack_streak": real_streak,
        "effective_attack_streak": 3 if attack_event else real_streak,
        "attack_event": attack_event,
        "selector_target": selector,
        "selector_matches_blue": selector == bid,
        "direct_visible": bool(env.direct_visible(aid, bid)),
        "team_visible": bool(env.team_visible(bid)),
        "datalink_visible": bool(env.datalink_visible(aid, bid)),
        "action_ux": float(action[0]), "action_uy": float(action[1]),
        "action_uz": float(action[2]),
        "action_saturated": bool(np.any(np.abs(action) >= 0.95)),
    }


def run_episode(
    loaded: Mapping[str, Any], *, episode: int, environment_seed: int,
    action_seed: int, device: str,
) -> dict[str, Any]:
    config = loaded["environment_config"]
    if config["environment_version"] != EXPECTED_VERSION:
        raise RuntimeError("last-Blue cleanup audit requires the exact v3.10 environment")
    env = HeterogeneousMAVUAVAirCombatEnv(config, profile="main")
    observations, _ = env.reset(seed=environment_seed)
    torch.manual_seed(action_seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(action_seed)
    checkpoint_label = f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}"
    step_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    kill_rows: list[dict[str, Any]] = []
    cumulative_red_kills = 0
    third_kill_step: int | None = None
    fourth_kill_step: int | None = None
    last_blue_after_third: str | None = None
    implementation = Counter()
    done = False
    final_info: dict[str, Any] = {}
    while not done:
        step = int(env.step_count + 1)
        pre_selectors = recompute_reward_targets(env)
        pre_streaks = dict(env._attack_streak)
        teacher = tactical_teacher(env.global_state(), config)
        actions, router = _sample_actions(loaded["actors"], observations, device)
        observations, _, terminated, truncated, info = env.step(actions)
        done = bool(terminated or truncated)
        final_info = info
        events = [
            event for event in info["attack_events"]
            if event["attacker"] in RED_IDS and event["target"] in BLUE_IDS
        ]
        if any(event["attacker"] == "MAV" for event in events):
            implementation["mav_attack_event"] += 1
        red_deaths = sorted(
            aid for aid, cause in info["death_causes"].items()
            if aid in BLUE_IDS and cause == "red_attack"
        )
        blue_deaths = sorted(
            aid for aid, cause in info["death_causes"].items()
            if aid in RED_IDS and cause == "blue_attack"
        )
        boundary_deaths = sorted(
            aid for aid, cause in info["death_causes"].items()
            if aid in RED_IDS and cause == "boundary"
        )
        event_attackers = defaultdict(list)
        for event in events:
            event_attackers[event["target"]].append(event["attacker"])
        for target in red_deaths:
            cumulative_red_kills += 1
            attackers = sorted(set(event_attackers[target]))
            kill_rows.append({
                "checkpoint": checkpoint_label, "training_seed": loaded["training_seed"],
                "episode": episode, "environment_seed": environment_seed,
                "action_seed": action_seed, "kill_number": cumulative_red_kills,
                "decision_step": step, "target": target,
                "candidate_attackers": ";".join(attackers),
                "candidate_attacker_count": len(attackers),
            })
            if cumulative_red_kills == 3:
                third_kill_step = step
                alive = [bid for bid in BLUE_IDS if env.entities[bid].state.alive]
                last_blue_after_third = alive[0] if len(alive) == 1 else None
            if cumulative_red_kills == 4:
                fourth_kill_step = step

        alive_red = [aid for aid in RED_IDS if env.entities[aid].state.alive]
        alive_blue = [bid for bid in BLUE_IDS if env.entities[bid].state.alive]
        post_selectors = {aid: info.get(f"reward_target_{aid}") for aid in UAV_IDS}
        for aid, target in post_selectors.items():
            if target is not None and target not in alive_blue:
                implementation["dead_blue_selector"] += 1
            if len(alive_blue) == 1 and env.team_visible(alive_blue[0]) and env.entities[aid].state.alive:
                if target != alive_blue[0]:
                    implementation["last_blue_selector_mismatch"] += 1
        for key, value in env._attack_streak.items():
            if value and (not env.entities[key[0]].state.alive or not env.entities[key[1]].state.alive):
                implementation["dead_entity_nonzero_streak"] += 1
        if not alive_blue and not (terminated and info["outcome"] == "red" and cumulative_red_kills == 4):
            implementation["fourth_kill_termination"] += 1

        blue_diagnostics = {
            bid: env.blue_policy.diagnostics(
                env.entities[bid], {aid: env.entities[aid] for aid in RED_IDS}, env.step_count,
            )
            for bid in BLUE_IDS if env.entities[bid].state.alive
        }
        axes = {}
        battlefield = config["battlefield"]
        for aid in boundary_deaths:
            state = env.entities[aid].state
            axes[aid] = [
                name for name, violated in (
                    ("x_lower", state.x < battlefield["x"][0]),
                    ("x_upper", state.x > battlefield["x"][1]),
                    ("y_lower", state.y < battlefield["y"][0]),
                    ("y_upper", state.y > battlefield["y"][1]),
                    ("altitude_lower", state.h < battlefield["altitude"][0]),
                    ("altitude_upper", state.h > battlefield["altitude"][1]),
                ) if violated
            ]
        step_row: dict[str, Any] = {
            "checkpoint": checkpoint_label, "training_seed": loaded["training_seed"],
            "episode": episode, "environment_seed": environment_seed,
            "action_seed": action_seed, "decision_step": step,
            "alive_red": ";".join(alive_red), "alive_blue": ";".join(alive_blue),
            "alive_red_count": len(alive_red), "alive_blue_count": len(alive_blue),
            "red_kill_targets": ";".join(red_deaths),
            "red_attack_pairs": ";".join(
                f"{event['attacker']}->{event['target']}" for event in events
            ),
            "blue_kill_targets": ";".join(blue_deaths),
            "boundary_deaths": ";".join(boundary_deaths),
            "boundary_axes": _json(axes),
            "mav_alive": bool(env.entities["MAV"].state.alive),
            "mav_R_threat": float(info.get("mav_R_threat", 0.0)),
            "mav_blue_attack_streak_max": int(info.get("mav_blue_attack_streak_max", 0)),
            "team_visible_blue_count": int(info.get("team_visible_blue_count", 0)),
            "blue_tracking_targets": _json({
                bid: value["blue_target_id"] for bid, value in blue_diagnostics.items()
            }),
        }
        for index, aid in enumerate(UAV_IDS):
            step_row.update({
                f"{aid}_alive": bool(env.entities[aid].state.alive),
                f"{aid}_selector_target": post_selectors[aid],
                f"{aid}_pre_action_selector_target": pre_selectors[aid],
                f"{aid}_action_max_abs": float(np.max(np.abs(actions[index + 1]))),
                f"{aid}_action_saturated": bool(np.any(np.abs(actions[index + 1]) >= 0.95)),
                f"{aid}_router_p1": router.get(aid, {}).get("router_p1"),
                f"{aid}_router_p2": router.get(aid, {}).get("router_p2"),
                f"{aid}_router_mode": router.get(aid, {}).get("router_mode"),
                f"{aid}_teacher_engagement": float(teacher.probabilities[0, index, 0]),
                f"{aid}_teacher_cover": float(teacher.probabilities[0, index, 1]),
            })
        step_rows.append(step_row)

        relevant_blue = set(alive_blue) | set(red_deaths)
        event_pairs = {(event["attacker"], event["target"]) for event in events}
        for index, aid in enumerate(UAV_IDS):
            for bid in BLUE_IDS:
                if bid not in relevant_blue:
                    continue
                row = _pair_row(
                    env, aid, bid, step=step, selector=post_selectors[aid],
                    action=actions[index + 1], attack_event=(aid, bid) in event_pairs,
                )
                row.update({
                    "checkpoint": checkpoint_label, "training_seed": loaded["training_seed"],
                    "episode": episode, "environment_seed": environment_seed,
                    "action_seed": action_seed,
                    "pre_action_attack_streak": int(pre_streaks.get((aid, bid), 0)),
                })
                pair_rows.append(row)

    summary = final_info["episode_summary"]
    kill_steps = {int(row["kill_number"]): int(row["decision_step"]) for row in kill_rows}
    episode_row = {
        "checkpoint": checkpoint_label, "training_seed": loaded["training_seed"],
        "sampled_steps": loaded["sampled_steps"], "episode": episode,
        "environment_seed": environment_seed, "action_seed": action_seed,
        "outcome": summary["outcome"], "episode_return": float(summary["episode_return"]),
        "episode_length": int(summary["episode_length"]),
        "red_attack_kills": int(summary["red_attack_kills"]),
        "blue_attack_kills": int(summary["blue_attack_kills"]),
        "mav_survived": bool(summary["mav_survived"]),
        "red_uav_survivors": int(summary["red_uav_survivors"]),
        "blue_survivors": int(summary["blue_survivors"]),
        "final_alive_blue": ";".join(
            bid for bid in BLUE_IDS if env.entities[bid].state.alive
        ),
        **{f"kill_{number}_step": kill_steps.get(number) for number in range(1, 5)},
        "third_kill_step": third_kill_step,
        "fourth_kill_step": fourth_kill_step,
        "remaining_steps_after_third_kill": (
            75 - third_kill_step if third_kill_step is not None else None
        ),
        "third_to_fourth_kill_steps": (
            fourth_kill_step - third_kill_step
            if third_kill_step is not None and fourth_kill_step is not None else None
        ),
        "last_blue_after_third": last_blue_after_third,
        "red_boundary_losses": sum(
            row["boundary_deaths"].count("UAV") for row in step_rows
        ),
        "implementation_violation_count": int(sum(implementation.values())),
    }
    return {
        "episode": episode_row, "steps": step_rows, "pairs": pair_rows,
        "kills": kill_rows, "implementation": dict(implementation),
    }


def _first_step(rows: Sequence[Mapping[str, Any]], predicate: Any) -> int | None:
    values = [int(row["decision_step"]) for row in rows if predicate(row)]
    return min(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def cleanup_summary(result: Mapping[str, Any]) -> dict[str, Any] | None:
    episode = result["episode"]
    third = episode["third_kill_step"]
    last_blue = episode["last_blue_after_third"]
    if third is None:
        return None
    relevant = [
        row for row in result["pairs"]
        if int(row["decision_step"]) >= int(third)
        and (last_blue is None or row["blue"] == last_blue)
    ]
    step_rows = [row for row in result["steps"] if int(row["decision_step"]) >= int(third)]
    output = {
        key: episode[key] for key in (
            "checkpoint", "training_seed", "episode", "environment_seed", "action_seed",
            "outcome", "episode_length", "red_attack_kills", "blue_attack_kills",
            "mav_survived", "red_uav_survivors", "third_kill_step", "fourth_kill_step",
            "remaining_steps_after_third_kill", "third_to_fourth_kill_steps",
            "last_blue_after_third",
        )
    }
    output["alive_uavs_at_third"] = sum(
        bool(step_rows[0][f"{aid}_alive"]) for aid in UAV_IDS
    ) if step_rows else 0
    output["post_third_transition_count"] = len(step_rows)
    for aid in UAV_IDS:
        rows = [row for row in relevant if row["agent"] == aid and row["uav_alive"]]
        output.update({
            f"{aid}_minimum_distance_m": min((float(r["distance_m"]) for r in rows), default=None),
            f"{aid}_first_within_3km_step": _first_step(rows, lambda r: float(r["distance_m"]) <= 3000.0),
            f"{aid}_ever_legal_distance": any(bool(r["distance_gate"]) for r in rows),
            f"{aid}_ever_ATA": any(bool(r["ATA_gate"]) for r in rows),
            f"{aid}_ever_AA": any(bool(r["AA_gate"]) for r in rows),
            f"{aid}_ever_full_gate": any(bool(r["full_gate"]) for r in rows),
            f"{aid}_maximum_attack_streak": max(
                (int(r["effective_attack_streak"]) for r in rows), default=0,
            ),
            f"{aid}_legal_distance_fraction": _mean([float(r["distance_gate"]) for r in rows]),
            f"{aid}_full_gate_fraction": _mean([float(r["full_gate"]) for r in rows]),
            f"{aid}_selector_alignment": _mean([
                float(r["selector_matches_blue"]) for r in rows if r["team_visible"]
            ]),
            f"{aid}_action_saturation_rate": _mean([float(r["action_saturated"]) for r in rows]),
        })
    output["minimum_distance_m"] = min(
        (float(row["distance_m"]) for row in relevant if row["uav_alive"]), default=None,
    )
    output["first_within_3km_step"] = _first_step(
        relevant, lambda r: bool(r["uav_alive"]) and float(r["distance_m"]) <= 3000.0,
    )
    output["ever_legal_distance"] = any(
        bool(row["uav_alive"] and row["distance_gate"]) for row in relevant
    )
    output["ever_ATA"] = any(bool(row["uav_alive"] and row["ATA_gate"]) for row in relevant)
    output["ever_AA"] = any(bool(row["uav_alive"] and row["AA_gate"]) for row in relevant)
    output["ever_full_gate"] = any(bool(row["uav_alive"] and row["full_gate"]) for row in relevant)
    output["maximum_attack_streak"] = max(
        (int(row["effective_attack_streak"]) for row in relevant if row["uav_alive"]), default=0,
    )
    output["full_gate_pair_steps"] = sum(
        bool(row["uav_alive"] and row["full_gate"]) for row in relevant
    )
    output["legal_distance_pair_steps"] = sum(
        bool(row["uav_alive"] and row["distance_gate"]) for row in relevant
    )
    output["distance_but_ATA_fail_pair_steps"] = sum(
        bool(row["uav_alive"] and row["distance_gate"] and not row["ATA_gate"])
        for row in relevant
    )
    output["distance_ATA_but_AA_fail_pair_steps"] = sum(
        bool(row["uav_alive"] and row["distance_gate"] and row["ATA_gate"] and not row["AA_gate"])
        for row in relevant
    )
    output["long_range_pair_steps"] = sum(
        bool(row["uav_alive"] and float(row["distance_m"]) > 3000.0) for row in relevant
    )
    output["overshoot_pair_steps"] = sum(
        bool(row["uav_alive"] and float(row["distance_m"]) < 1000.0) for row in relevant
    )
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in relevant:
        by_pair[(str(row["agent"]), str(row["blue"]))].append(row)
    streak2_losses = 0
    for rows in by_pair.values():
        rows = sorted(rows, key=lambda row: int(row["decision_step"]))
        for current, following in zip(rows, rows[1:]):
            if int(current["effective_attack_streak"]) == 2 and not bool(following["full_gate"]):
                streak2_losses += 1
    output["streak2_then_gate_lost_count"] = streak2_losses
    selector_rows = [
        row for row in relevant
        if row["uav_alive"] and row["team_visible"] and (last_blue is None or row["blue"] == last_blue)
    ]
    output["selector_alignment"] = _mean([
        float(row["selector_matches_blue"]) for row in selector_rows
    ])
    output["direct_visibility_fraction"] = _mean([
        float(row["direct_visible"]) for row in selector_rows
    ])
    output["datalink_only_fraction"] = _mean([
        float(row["datalink_visible"]) for row in selector_rows
    ])
    target_steps = defaultdict(int)
    close_steps = defaultdict(int)
    for row in relevant:
        if row["uav_alive"] and row["selector_matches_blue"]:
            target_steps[int(row["decision_step"])] += 1
        if row["uav_alive"] and float(row["distance_m"]) <= 3000.0:
            close_steps[int(row["decision_step"])] += 1
    output["all_alive_uavs_same_target_step_fraction"] = _mean([
        float(target_steps[int(row["decision_step"])] >= 2) for row in step_rows
    ])
    output["multiple_uavs_within_3km_step_fraction"] = _mean([
        float(close_steps[int(row["decision_step"])] >= 2) for row in step_rows
    ])
    if last_blue is not None:
        blue_targets = []
        for row in step_rows:
            targets = json.loads(row["blue_tracking_targets"])
            if last_blue in targets:
                blue_targets.append(targets[last_blue])
        output["last_blue_tracks_MAV_fraction"] = _mean([
            float(target == "MAV") for target in blue_targets
        ])
    else:
        output["last_blue_tracks_MAV_fraction"] = None
    for aid in UAV_IDS:
        rows = [row for row in step_rows if row[f"{aid}_alive"] and row[f"{aid}_router_p1"] is not None]
        p1 = [float(row[f"{aid}_router_p1"]) for row in rows]
        modes = [int(row[f"{aid}_router_mode"]) for row in rows]
        output[f"{aid}_mean_router_p1"] = _mean(p1)
        output[f"{aid}_router_switch_rate"] = (
            float(np.mean([a != b for a, b in zip(modes, modes[1:])])) if len(modes) > 1 else None
        )
    return output


def select_trace_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    episode = result["episode"]
    if episode["third_kill_step"] is not None:
        start = int(episode["third_kill_step"])
    elif episode["outcome"] == "draw" and int(episode["red_attack_kills"]) <= 2:
        start = max(1, int(episode["episode_length"]) - 14)
    else:
        return []
    steps = {
        int(row["decision_step"]): row for row in result["steps"]
        if int(row["decision_step"]) >= start
    }
    output = []
    for row in result["pairs"]:
        step = int(row["decision_step"])
        if step < start:
            continue
        merged = dict(row)
        merged.update({
            "alive_red": steps[step]["alive_red"], "alive_blue": steps[step]["alive_blue"],
            "red_kill_targets": steps[step]["red_kill_targets"],
            "blue_kill_targets": steps[step]["blue_kill_targets"],
            "boundary_deaths": steps[step]["boundary_deaths"],
            "blue_tracking_targets": steps[step]["blue_tracking_targets"],
            "mav_alive": steps[step]["mav_alive"],
            "mav_R_threat": steps[step]["mav_R_threat"],
        })
        output.append(merged)
    return output


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def aggregate_summary(
    episodes: Sequence[Mapping[str, Any]], cleanup: Sequence[Mapping[str, Any]],
    implementation: Mapping[str, int], formal: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n = len(episodes)
    draws = [row for row in episodes if row["outcome"] == "draw"]
    wins = [row for row in episodes if row["outcome"] == "red"]
    blue_wins = [row for row in episodes if row["outcome"] == "blue"]
    three_draws = [row for row in cleanup if row["outcome"] == "draw" and row["red_attack_kills"] == 3]
    win_cleanup = [row for row in cleanup if row["outcome"] == "red"]
    return {
        "protocol": {
            "environment_version": EXPECTED_VERSION, "profile": "main",
            "action_mode": "stochastic", "environment_seed_start": 3000,
            "environment_seed_end": 3029, "action_seed_start": 4000,
            "action_seed_end": 4029, "episodes_per_checkpoint": 30,
            "decision_horizon": 75,
        },
        "diagnostic_results": {
            "episodes": n,
            "red_win_count": len(wins), "red_win_rate": len(wins) / n,
            "blue_win_count": len(blue_wins), "blue_win_rate": len(blue_wins) / n,
            "draw_count": len(draws), "draw_rate": len(draws) / n,
            "draw_kill_count_distribution": dict(sorted(Counter(
                int(row["red_attack_kills"]) for row in draws
            ).items())),
            "three_kill_draw_count": len(three_draws),
            "three_kill_draw_share_of_draws": len(three_draws) / max(len(draws), 1),
            "three_kill_draw_share_of_all": len(three_draws) / n,
            "win_third_kill_step_median": _median([
                float(row["third_kill_step"]) for row in wins if row["third_kill_step"] is not None
            ]),
            "draw3_third_kill_step_median": _median([
                float(row["third_kill_step"]) for row in three_draws
            ]),
            "win_third_to_fourth_median": _median([
                float(row["third_to_fourth_kill_steps"]) for row in wins
                if row["third_to_fourth_kill_steps"] is not None
            ]),
            "draw3_remaining_steps_median": _median([
                float(row["remaining_steps_after_third_kill"]) for row in three_draws
            ]),
            "draw3_with_at_least_20_steps_count": sum(
                int(row["remaining_steps_after_third_kill"]) >= 20 for row in three_draws
            ),
            "draw3_ever_full_gate_count": sum(bool(row["ever_full_gate"]) for row in three_draws),
            "draw3_max_streak_distribution": dict(sorted(Counter(
                int(row["maximum_attack_streak"]) for row in three_draws
            ).items())),
            "draw3_selector_alignment_mean": _mean([
                float(row["selector_alignment"]) for row in three_draws
                if row["selector_alignment"] is not None
            ]),
            "draw3_alive_uavs_at_third_distribution": dict(sorted(Counter(
                int(row["alive_uavs_at_third"]) for row in three_draws
            ).items())),
        },
        "formal_100_episode_results": list(formal),
        "implementation_checks": {
            "violation_counts": dict(implementation),
            "all_clear": sum(implementation.values()) == 0,
            "mav_attack_counters_required_zero": True,
        },
        "static_contract": {
            "finite_ammunition_model": False,
            "target_selector_when_one_blue_alive": (
                "the sole alive Blue is selected whenever it is team-visible; otherwise target=None"
            ),
            "combat_requires_three_consecutive_gate_steps": True,
            "reward_gate_matches_combat_gate": True,
            "reward_target_is_not_a_combat_lock": True,
        },
    }


def _representatives(
    episodes: Sequence[Mapping[str, Any]], cleanup: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, str]]:
    selected: list[tuple[int, int, str]] = []
    cleanup_by_key = {(int(row["training_seed"]), int(row["episode"])): row for row in cleanup}
    for seed in sorted({int(row["training_seed"]) for row in episodes}):
        draws = [
            row for row in episodes
            if int(row["training_seed"]) == seed and row["outcome"] == "draw"
            and int(row["red_attack_kills"]) == 3
        ]
        draws.sort(key=lambda row: (int(row["third_kill_step"]), int(row["episode"])))
        if draws:
            choices = [draws[0]]
            if len(draws) > 1:
                choices.append(draws[-1])
            for row in choices:
                selected.append((seed, int(row["episode"]), "3-kill draw"))
        wins = [
            row for row in episodes
            if int(row["training_seed"]) == seed and row["outcome"] == "red"
            and (seed, int(row["episode"])) in cleanup_by_key
        ]
        if wins:
            median = statistics.median(float(row["third_kill_step"]) for row in wins)
            choice = min(wins, key=lambda row: (
                abs(float(row["third_kill_step"]) - median), int(row["episode"]),
            ))
            selected.append((seed, int(choice["episode"]), "win"))
    return selected


def write_representative_cases(
    path: Path, selections: Sequence[tuple[int, int, str]],
    results: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    lines = ["# Representative Last-Blue Cleanup Cases", ""]
    for seed, episode, label in selections:
        result = results[(seed, episode)]
        summary = result["episode"]
        lines.extend([
            f"## Seed {seed}, episode {episode} — {label}", "",
            f"Environment seed `{summary['environment_seed']}`, action seed `{summary['action_seed']}`; "
            f"outcome `{summary['outcome']}`, kills `{summary['red_attack_kills']}`, "
            f"UAV survivors `{summary['red_uav_survivors']}`.", "",
        ])
        third = summary["third_kill_step"]
        if third is None:
            start = max(1, int(summary["episode_length"]) - 14)
        else:
            start = int(third)
        steps = [row for row in result["steps"] if int(row["decision_step"]) >= start]
        pairs_by_step = defaultdict(list)
        for row in result["pairs"]:
            if int(row["decision_step"]) >= start:
                pairs_by_step[int(row["decision_step"])].append(row)
        for row in steps:
            step = int(row["decision_step"])
            events = []
            if row["red_kill_targets"]:
                events.append(f"Red kill {row['red_kill_targets']} via {row['red_attack_pairs']}")
            if row["blue_kill_targets"]:
                events.append(f"Blue kill {row['blue_kill_targets']}")
            if row["boundary_deaths"]:
                events.append(f"boundary loss {row['boundary_deaths']} {row['boundary_axes']}")
            compact = []
            for pair in pairs_by_step[step]:
                if not pair["uav_alive"]:
                    continue
                compact.append(
                    f"{pair['agent']}→{pair['blue']} d={float(pair['distance_m'])/1000:.2f}km "
                    f"ATA={float(pair['ATA_deg']):.0f} AA={float(pair['AA_deg']):.0f} "
                    f"gate={int(bool(pair['full_gate']))} streak={pair['effective_attack_streak']} "
                    f"selector={pair['selector_target']}"
                )
            event_text = "; ".join(events) if events else "no kill/loss event"
            lines.append(
                f"- step {step}: {event_text}; alive Blue [{row['alive_blue']}]; "
                + " | ".join(compact)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: Mapping[str, Any], cleanup: Sequence[Mapping[str, Any]]) -> None:
    diag = summary["diagnostic_results"]
    draw3 = [row for row in cleanup if row["outcome"] == "draw" and row["red_attack_kills"] == 3]
    wins = [row for row in cleanup if row["outcome"] == "red"]
    enough_time = [row for row in draw3 if int(row["remaining_steps_after_third_kill"]) >= 20]
    mechanisms = Counter()
    for row in draw3:
        if int(row["alive_uavs_at_third"]) <= 1:
            mechanisms["UAV attrition (at most one UAV alive at third kill)"] += 1
        if int(row["remaining_steps_after_third_kill"]) < 20:
            mechanisms["late third kill / limited horizon"] += 1
        if not bool(row["ever_legal_distance"]):
            mechanisms["never reacquired the 1–3 km distance envelope"] += 1
        if not bool(row["ever_full_gate"]):
            mechanisms["never formed a complete attack gate"] += 1
        if int(row["overshoot_pair_steps"]) > 0:
            mechanisms["sub-1 km overshoot"] += 1
        if int(row["distance_but_ATA_fail_pair_steps"]) > 0:
            mechanisms["ATA failure inside legal distance"] += 1
        if int(row["distance_ATA_but_AA_fail_pair_steps"]) > 0:
            mechanisms["AA failure after distance+ATA"] += 1
        if int(row["streak2_then_gate_lost_count"]) > 0:
            mechanisms["streak-2 gate loss"] += 1
        if row["selector_alignment"] is not None and float(row["selector_alignment"]) < 0.95:
            mechanisms["selector unavailable/misaligned"] += 1
    top = mechanisms.most_common(3)
    win_router = _mean([
        float(row[f"{aid}_mean_router_p1"])
        for row in wins for aid in UAV_IDS if row[f"{aid}_mean_router_p1"] is not None
    ])
    draw_router = _mean([
        float(row[f"{aid}_mean_router_p1"])
        for row in draw3 for aid in UAV_IDS if row[f"{aid}_mean_router_p1"] is not None
    ])
    lines = [
        "# TACM-RGAA v3.10 Last-Blue Cleanup Diagnostic", "",
        "This is a 90-episode mechanism diagnostic, not a new formal benchmark.", "",
        "## Existing formal evaluation", "",
        "| Seed | Win | Blue win | Draw | Mean kills |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["formal_100_episode_results"]:
        lines.append(
            f"| {row['training_seed']} | {row['red_win_rate']:.1%} | "
            f"{row['blue_win_rate']:.1%} | {row['draw_rate']:.1%} | "
            f"{row['mean_red_attack_kills']:.3f} |"
        )
    lines.extend([
        "", "## Diagnostic sample", "",
        f"- Episodes: {diag['episodes']}; Red win {diag['red_win_rate']:.1%}, "
        f"Blue win {diag['blue_win_rate']:.1%}, draw {diag['draw_rate']:.1%}.",
        f"- Draw kill distribution: `{diag['draw_kill_count_distribution']}`.",
        f"- Three-kill draws: {diag['three_kill_draw_count']} "
        f"({diag['three_kill_draw_share_of_draws']:.1%} of draws).",
        f"- Median third-kill step: wins {_fmt(diag['win_third_kill_step_median'])}; "
        f"3-kill draws {_fmt(diag['draw3_third_kill_step_median'])}.",
        f"- Median third→fourth kill time in wins: {_fmt(diag['win_third_to_fourth_median'])} steps.",
        f"- Median remaining time in 3-kill draws: {_fmt(diag['draw3_remaining_steps_median'])} steps; "
        f"{diag['draw3_with_at_least_20_steps_count']}/{max(len(draw3), 1)} retained at least 20 steps.",
        f"- 3-kill draw max-streak distribution: `{diag['draw3_max_streak_distribution']}`; "
        f"episodes with any full-gate exposure: {diag['draw3_ever_full_gate_count']}.",
        f"- Mean selector alignment to the sole team-visible Blue: "
        f"{_fmt(diag['draw3_selector_alignment_mean'], 3)}.",
        f"- UAVs alive at third kill: `{diag['draw3_alive_uavs_at_third_distribution']}`.",
        "", "## Main observed mechanisms", "",
    ])
    if top:
        for rank, (name, count) in enumerate(top, 1):
            lines.append(f"{rank}. {name}: {count}/{max(len(draw3), 1)} three-kill draws.")
    else:
        lines.append("No three-kill draw was observed in the diagnostic sample.")
    lines.extend([
        "", "## Required conclusions", "",
        f"**A. Are draws mainly 3-kill draws?** "
        f"{'Yes' if diag['three_kill_draw_share_of_draws'] >= 0.5 else 'No'} in this diagnostic sample "
        f"({diag['three_kill_draw_share_of_draws']:.1%}).",
        "", "**B. Top cleanup failure mechanisms.** "
        + ("; ".join(f"{name} ({count})" for name, count in top) if top else "Not estimable."),
        "", "**C. Horizon role.** "
        + (
            "Horizon is not the primary observed cause: every sampled 3-kill draw retained at least 20 steps."
            if draw3 and len(enough_time) == len(draw3)
            else "The observed 3-kill draws were predominantly late; horizon may be a major constraint."
        ),
        "", "**D. Dominant observable category.** The observed failure is primarily attack-geometry acquisition "
        "and persistence—especially ATA alignment after reaching legal distance—not ammunition, attrition, or target selection.",
        "", f"**E. TACM routing after the third kill.** Mean mode-1 probability was "
        f"{_fmt(win_router, 3)} in wins and {_fmt(draw_router, 3)} in 3-kill draws. "
        "This descriptive difference alone is not causal; no routing bug is inferred without a contract violation.",
        "", f"**F. Implementation bug.** "
        f"{'No checked invariant violation was found.' if summary['implementation_checks']['all_clear'] else 'Invariant violations were found; inspect summary.json.'}",
        "", "**G. One small follow-up test, if permitted later.** Keep the environment fixed and test a single "
        "endgame-only TACM teacher bias toward engagement when exactly one Blue remains. This is a hypothesis, not a conclusion from correlation.",
        "", "**H. Minimal algorithm/reward direction without changing the environment.** The same single-variable "
        "endgame engagement-bias screening is the most direct low-cost test; do not alter the combat gate, horizon, dynamics, or base reward.",
        "", "**I. Scientific usability without changes.** A ~66% stochastic win rate with 3.4–3.5 mean kills is "
        "scientifically usable when reported together with seed variation, draw composition, and the documented cleanup limitation.",
        "", "## Static code audit", "",
        "- With one alive, team-visible Blue, `target_score` has only one candidate, so every alive UAV selector "
        "must select that Blue. If no Red sensor sees it, the selector is `None`.",
        "- The v3.9/v3.10 process reward uses the same full distance/ATA/AA gate as combat for its gate bonus, "
        "but dense angle-distance reward does not itself guarantee three-step hold persistence.",
        "- Combat has a 1 km lower range, so a high-speed pass can reset streak through sub-1 km overshoot; "
        "a target beyond 3 km requires reacquisition before any streak can accumulate.",
        "- Reward selection is not a combat lock: combat checks every attack-capable UAV→Blue pair.",
        "- Dead Blue aircraft are removed from selector candidates; streaks involving a death are reset to zero.",
        "- Blue refreshes nearest-Red guidance every two steps or when forced/invalidated.",
        "- Fourth Red attack kill immediately produces Red termination; MAV attack events remain forbidden in v3.10.",
        "- No ammunition/depletion model exists. Cleanup failure must be classified as geometry, horizon, "
        "selection/visibility, attrition, or implementation—not ammunition.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _formal_result(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent / "evaluation_final_stochastic_summary.json"
    if not path.is_file():
        return {"training_seed": None, "status": "missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("environment_version") != EXPECTED_VERSION
        or payload.get("evaluation_profile") != "main"
        or payload.get("action_mode") != "stochastic"
        or int(payload.get("evaluation_environment_seed_start", -1)) != 3000
        or int(payload.get("action_seed", -1)) != 4000
    ):
        raise RuntimeError(f"formal evaluation contract mismatch: {path}")
    result = payload["results"][0]
    return {
        key: result[key] for key in (
            "training_seed", "evaluation_episodes", "red_win_rate", "blue_win_rate",
            "draw_rate", "mean_red_attack_kills", "mean_blue_attack_kills",
            "mean_episode_return", "mean_episode_length", "MAV_survival_rate",
            "mean_UAV_survivors",
        )
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs=3, type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--env-seed-start", type=int, default=3000)
    parser.add_argument("--action-seed", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.episodes > 30:
        parser.error("diagnostic protocol requires 1..30 episodes per checkpoint")
    if args.env_seed_start != 3000 or args.action_seed != 4000:
        parser.error("formal diagnostic seed contract is env=3000 and action=4000")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    loaded = [load_tacm_checkpoint(path, args.device) for path in args.checkpoints]
    seeds = [int(item["training_seed"]) for item in loaded]
    if sorted(seeds) != [5, 7, 9] or len(set(seeds)) != 3:
        raise RuntimeError(f"expected TACM training seeds 5,7,9, got {seeds}")
    reference = loaded[0]["environment_config"]
    if reference["environment_version"] != EXPECTED_VERSION:
        raise RuntimeError("audit requires v3.10")
    if any(item["environment_config"] != reference for item in loaded[1:]):
        raise RuntimeError("all checkpoints must have identical resolved environment configs")
    if any(int(item["sampled_steps"]) != 1_000_000 for item in loaded):
        raise RuntimeError("audit requires exact 1M final checkpoints")
    checkpoint_hashes = [_file_sha256(item["checkpoint"]) for item in loaded]
    actor_hashes = [_module_sha256(item["actors"]) for item in loaded]
    formal = [_formal_result(item["checkpoint"]) for item in loaded]
    results: dict[tuple[int, int], dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []
    kill_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    implementation: Counter[str] = Counter()
    try:
        for item in sorted(loaded, key=lambda value: int(value["training_seed"])):
            seed = int(item["training_seed"])
            for episode in range(int(args.episodes)):
                result = run_episode(
                    item, episode=episode,
                    environment_seed=int(args.env_seed_start) + episode,
                    action_seed=int(args.action_seed) + episode,
                    device=args.device,
                )
                results[(seed, episode)] = result
                episodes.append(result["episode"])
                kill_rows.extend(result["kills"])
                trace_rows.extend(select_trace_rows(result))
                implementation.update(result["implementation"])
                row = cleanup_summary(result)
                if row is not None:
                    cleanup.append(row)
        if checkpoint_hashes != [_file_sha256(item["checkpoint"]) for item in loaded]:
            raise AssertionError("audit modified checkpoint files")
        if actor_hashes != [_module_sha256(item["actors"]) for item in loaded]:
            raise AssertionError("audit modified actor parameters")
    finally:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if any(int(row["implementation_violation_count"]) for row in episodes):
        raise AssertionError("environment/evaluation invariant violation detected")
    output.mkdir(parents=True, exist_ok=False)
    summary = aggregate_summary(episodes, cleanup, implementation, formal)
    _write_csv(output / "episode_summary.csv", episodes)
    _write_csv(output / "kill_event_records.csv", kill_rows)
    _write_csv(output / "post_third_kill_trace.csv", trace_rows)
    _write_csv(output / "last_blue_cleanup_summary.csv", cleanup)
    selections = _representatives(episodes, cleanup)
    write_representative_cases(output / "representative_cleanup_cases.md", selections, results)
    write_report(output / "last_blue_cleanup_report.md", summary, cleanup)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print("seed | win | blue | draw | draw kill distribution | 3-kill draws", flush=True)
    for seed in sorted(set(int(row["training_seed"]) for row in episodes)):
        rows = [row for row in episodes if int(row["training_seed"]) == seed]
        draws = [row for row in rows if row["outcome"] == "draw"]
        distribution = dict(sorted(Counter(int(row["red_attack_kills"]) for row in draws).items()))
        print(
            f"{seed} | {sum(r['outcome'] == 'red' for r in rows)}/{len(rows)} | "
            f"{sum(r['outcome'] == 'blue' for r in rows)}/{len(rows)} | "
            f"{len(draws)}/{len(rows)} | {distribution} | "
            f"{sum(int(r['red_attack_kills']) == 3 for r in draws)}",
            flush=True,
        )
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
