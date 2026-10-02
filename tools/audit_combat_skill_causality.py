"""Read-only paired combat-skill causality audit for TACM-RGAA checkpoints."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from env.dynamics import map_normalized_action
from env.geometry import compute_pairwise_geometry
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from env.reward_role_v37 import target_score
from env.reward_role_v39 import attack_gate_indicator
from env.vector_env import _environment_state, _restore_environment_state
from tools.audit_continuation_horizon import load_tacm_checkpoint


CONDITIONS = ("POLICY", "TRIM", "UNIFORM_RANDOM", "UAV_ACTION_CYCLIC")
UAV_IDS = RED_IDS[1:]
# Recipient <- source.  Equivalently U1 -> U2, U2 -> U3, U3 -> U1.
CYCLIC_SOURCE = {"UAV1": "UAV3", "UAV2": "UAV1", "UAV3": "UAV2"}

EPISODE_FIELDS = (
    "checkpoint", "training_seed", "sampled_steps", "condition", "episode",
    "environment_seed", "action_seed", "control_action_seed", "outcome",
    "episode_return", "episode_length", "MAV_survival", "UAV_survivors",
    "red_attack_kills", "blue_attack_kills", "red_gate_entry_count",
    "red_gate_active_pair_steps", "red_gate_pair_exposures", "red_gate_active_fraction",
    "streak2_opportunities", "attack_event_pair_count", "simultaneous_multi_target_red_attack_steps",
    "ambiguous_blue_death_count", "MAV_attack_event_pair_count",
    "MAV_death_candidate_count", "MAV_only_candidate_death_count",
)


def _module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state().clone(), None if cuda is None else [state.clone() for state in cuda]


def _restore_torch_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0].cpu())
    if state[1] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state[1]])


def recompute_reward_targets(env: HeterogeneousMAVUAVAirCombatEnv) -> dict[str, str | None]:
    """Pure reconstruction of the v3.9 UAV selector on the current state."""
    normalization = env.config["normalization"]
    maximum_range = float(env.config["combat"]["distance"][1])
    available = [bid for bid in BLUE_IDS if env.entities[bid].state.alive and env.team_visible(bid)]
    result: dict[str, str | None] = {"MAV": None}
    for aid in UAV_IDS:
        red = env.entities[aid].state
        candidates: list[tuple[float, str]] = []
        if red.alive:
            for bid in available:
                score = target_score(
                    red, env.entities[bid].state,
                    float(normalization["relative_altitude_scale"]),
                    float(normalization["relative_velocity_scale"]), maximum_range,
                )
                if np.isfinite(score):
                    candidates.append((score, bid))
        chosen = max(candidates, key=lambda item: item[0]) if candidates else None
        result[aid] = chosen[1] if chosen else None
    return result


def _pair_geometry(env: HeterogeneousMAVUAVAirCombatEnv, attacker: str, target: str) -> dict[str, Any]:
    red = env.entities[attacker]
    blue = env.entities[target]
    geometry = compute_pairwise_geometry(red.state, blue.state)
    combat = env.config["combat"]
    gate = bool(attack_gate_indicator(
        geometry.distance, geometry.ata, geometry.aa,
        float(combat["distance"][0]), float(combat["distance"][1]),
        np.deg2rad(float(combat["ata_deg"])), np.deg2rad(float(combat["aa_deg"])),
    ))
    line = geometry.relative_position / max(geometry.distance, 1e-9)
    closing = -float(np.dot(geometry.relative_velocity, line))
    return {
        "distance": float(geometry.distance), "ATA_deg": float(np.rad2deg(geometry.ata)),
        "AA_deg": float(np.rad2deg(geometry.aa)), "gate": gate,
        "closing_rate": closing, "attacker_speed": float(red.state.v),
        "target_speed": float(blue.state.v),
    }


def _sample_policy_actions(actors: Any, observations: Mapping[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for index, aid in enumerate(RED_IDS):
            observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
            action, _ = actors.actors[index].sample(observation, deterministic=False)
            actions[aid] = action.squeeze(0).cpu().numpy().astype(np.float32, copy=True)
    return actions


def condition_actions(
    condition: str, actors: Any, observations: Mapping[str, np.ndarray], device: str,
    random_rng: np.random.Generator | None,
) -> dict[str, np.ndarray]:
    if condition == "TRIM":
        return {aid: np.zeros(3, dtype=np.float32) for aid in RED_IDS}
    if condition == "UNIFORM_RANDOM":
        if random_rng is None:
            raise ValueError("UNIFORM_RANDOM requires an explicit RNG")
        return {aid: random_rng.uniform(-1.0, 1.0, 3).astype(np.float32) for aid in RED_IDS}
    sampled = _sample_policy_actions(actors, observations, device)
    if condition == "POLICY":
        return sampled
    if condition == "UAV_ACTION_CYCLIC":
        return {
            "MAV": sampled["MAV"].copy(),
            **{aid: sampled[CYCLIC_SOURCE[aid]].copy() for aid in UAV_IDS},
        }
    raise ValueError(f"unknown condition: {condition}")


def _command_fields(env: HeterogeneousMAVUAVAirCombatEnv, aid: str, action: np.ndarray) -> dict[str, float]:
    command = map_normalized_action(action, env.entities[aid].state, env.entities[aid].spec)
    return {
        "action_ux": float(action[0]), "action_uy": float(action[1]), "action_uz": float(action[2]),
        "command_nx": float(command.nx), "command_ny": float(command.ny), "command_nz": float(command.nz),
    }


def _branch_once(
    env: HeterogeneousMAVUAVAirCombatEnv, snapshot: Mapping[str, Any], actions: Mapping[str, np.ndarray],
    attacker: str, target: str,
) -> dict[str, Any]:
    _restore_environment_state(env, snapshot)
    _, _, _, _, info = env.step(actions)
    pair = (attacker, target)
    event_pairs = {(row["attacker"], row["target"]) for row in info["attack_events"]}
    geometry = _pair_geometry(env, attacker, target)
    return {
        "target_died": info["death_causes"].get(target) == "red_attack",
        "pair_completed": pair in event_pairs,
        "post_action_gate": geometry["gate"],
    }


def intervention_action_sets(
    actions: Mapping[str, np.ndarray], attacker: str,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    branches = [("NORMAL_POLICY", {key: value.copy() for key, value in actions.items()})]
    trim = {key: value.copy() for key, value in actions.items()}
    trim[attacker] = np.zeros(3, np.float32)
    branches.append(("ATTACKER_TRIM_ONE_STEP", trim))
    if attacker in UAV_IDS:
        peer = {key: value.copy() for key, value in actions.items()}
        peer[attacker] = actions[CYCLIC_SOURCE[attacker]].copy()
        branches.append(("ATTACKER_PEER_ACTION_ONE_STEP", peer))
    return branches


def candidate_attackers(
    event_pairs: Sequence[tuple[str, str]], death_targets: Sequence[str],
) -> dict[str, list[str]]:
    """Retain every simultaneous candidate; deliberately do not choose a killer."""
    return {
        target: sorted({attacker for attacker, event_target in event_pairs if event_target == target})
        for target in death_targets
    }


def reward_target_matches(
    attacker: str, target: str, targets: Mapping[str, str | None],
) -> bool:
    return bool(attacker in UAV_IDS and targets.get(attacker) == target)


def red_attack_event_pairs(events: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    return [
        (row["attacker"], row["target"]) for row in events
        if row["attacker"] in RED_IDS and row["target"] in BLUE_IDS
    ]


def evaluate_streak2_opportunity(
    env: HeterogeneousMAVUAVAirCombatEnv, actions: Mapping[str, np.ndarray],
    attacker: str, target: str, state_group_id: str,
) -> list[dict[str, Any]]:
    """One-step paired intervention from one exact pre-action environment state."""
    snapshot = _environment_state(env)
    rng_state = _torch_rng_state()
    branches = intervention_action_sets(actions, attacker)
    rows = []
    try:
        for branch, branch_actions in branches:
            _restore_environment_state(env, snapshot)
            command_fields = _command_fields(env, attacker, branch_actions[attacker])
            outcome = _branch_once(env, snapshot, branch_actions, attacker, target)
            rows.append({
                "state_group_id": state_group_id, "attacker": attacker, "target": target,
                "branch": branch, **outcome, **command_fields,
            })
    finally:
        _restore_environment_state(env, snapshot)
        _restore_torch_rng(rng_state)
    return rows


def _finite_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise FloatingPointError("audit generated a non-finite value")


def run_condition_episode(
    loaded: Mapping[str, Any], *, condition: str, profile: str, episode: int,
    environment_seed: int, action_seed: int, control_action_seed: int,
    device: str, max_counterfactual_opportunities: int,
    collect_counterfactuals: bool,
) -> dict[str, list[dict[str, Any]]]:
    env = HeterogeneousMAVUAVAirCombatEnv(loaded["environment_config"], profile=profile)
    observations, _ = env.reset(seed=environment_seed)
    torch.manual_seed(action_seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(action_seed)
    random_rng = np.random.default_rng(control_action_seed)
    pair_history: dict[tuple[str, str], list[dict[str, Any]]] = {
        (aid, bid): [] for aid in RED_IDS for bid in BLUE_IDS
    }
    counters = {
        "gate_entries": 0, "gate_active": 0, "gate_exposures": 0,
        "streak2": 0, "attack_events": 0, "multi_target_steps": 0,
        "ambiguous_deaths": 0, "mav_events": 0, "mav_candidates": 0,
        "mav_only": 0,
    }
    per_agent = {
        aid: {"gate_active": 0, "exposures": 0, "streak1": 0, "streak2": 0, "events": 0}
        for aid in RED_IDS
    }
    attack_rows: list[dict[str, Any]] = []
    death_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    total_opportunities = audited_opportunities = 0
    done = False
    info: dict[str, Any] = {}
    while not done:
        step = int(env.step_count + 1)
        targets = recompute_reward_targets(env)
        actions = condition_actions(condition, loaded["actors"], observations, device, random_rng)
        command_fields = {aid: _command_fields(env, aid, actions[aid]) for aid in RED_IDS}
        pre_alive_pairs = [
            (aid, bid) for aid in RED_IDS for bid in BLUE_IDS
            if env.entities[aid].state.alive and env.entities[bid].state.alive
        ]
        pre = {}
        for aid, bid in pre_alive_pairs:
            pre[(aid, bid)] = {
                **_pair_geometry(env, aid, bid),
                "attack_streak": int(env._attack_streak.get((aid, bid), 0)),
                "reward_selected_target": targets[aid],
                "target_match": reward_target_matches(aid, bid, targets),
                "direct_visibility": bool(env.direct_visible(aid, bid)),
                "datalink_visibility": bool(env.datalink_visible(aid, bid)),
            }
        opportunities = [pair for pair in pre_alive_pairs if pre[pair]["attack_streak"] == 2]
        counters["streak2"] += len(opportunities); total_opportunities += len(opportunities)
        if collect_counterfactuals:
            for aid, bid in opportunities:
                if audited_opportunities >= max_counterfactual_opportunities:
                    break
                group = f"{loaded['checkpoint'].parent.name}:{episode}:{step}:{aid}:{bid}"
                rows = evaluate_streak2_opportunity(env, actions, aid, bid, group)
                for row in rows:
                    row.update({
                        "checkpoint": f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}",
                        "training_seed": loaded["training_seed"], "episode": episode,
                        "environment_seed": environment_seed, "action_seed": action_seed,
                        "decision_step": step,
                    })
                cf_rows.extend(rows); audited_opportunities += 1
        observations, _, terminated, truncated, info = env.step(actions)
        event_pairs = red_attack_event_pairs(info["attack_events"])
        event_set = set(event_pairs)
        counters["attack_events"] += len(event_pairs)
        event_by_attacker: dict[str, list[str]] = {}
        for aid, bid in event_pairs:
            event_by_attacker.setdefault(aid, []).append(bid)
            per_agent[aid]["events"] += 1
            counters["mav_events"] += int(aid == "MAV")
        counters["multi_target_steps"] += int(any(len(set(values)) > 1 for values in event_by_attacker.values()))
        death_targets = {
            bid for bid, cause in info["death_causes"].items()
            if bid in BLUE_IDS and cause == "red_attack"
        }
        death_candidates = candidate_attackers(event_pairs, sorted(death_targets))
        for bid in death_targets:
            candidates = death_candidates[bid]
            counters["ambiguous_deaths"] += int(len(candidates) > 1)
            counters["mav_candidates"] += int("MAV" in candidates)
            counters["mav_only"] += int(candidates == ["MAV"])
            for aid in candidates:
                kill_geometry = _pair_geometry(env, aid, bid)
                current_kill = {
                    "decision_step": step, **kill_geometry,
                    "attack_streak": int(pre[(aid, bid)]["attack_streak"] + 1),
                    "reward_selected_target": targets[aid],
                    "target_match": reward_target_matches(aid, bid, targets),
                    "direct_visibility": pre[(aid, bid)]["direct_visibility"],
                    "datalink_visibility": pre[(aid, bid)]["datalink_visibility"],
                    **command_fields[aid],
                }
                history = (pair_history[(aid, bid)] + [current_kill])[-3:]
                death_rows.append({
                    "checkpoint": f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}",
                    "condition": condition, "episode": episode, "decision_step": step,
                    "target": bid, "candidate_attacker": aid,
                    "candidate_attacker_count": len(candidates),
                    "reward_selected_target": pre[(aid, bid)]["reward_selected_target"],
                    "reward_target_match": pre[(aid, bid)]["target_match"],
                    "three_step_reward_target_consistent": bool(
                        len(history) == 3 and all(row["target_match"] for row in history)
                    ),
                    **{key: current_kill[key] for key in (
                        "direct_visibility", "datalink_visibility", "distance", "ATA_deg", "AA_deg",
                        "gate", "attack_streak",
                    )},
                })
                for relative, historic in enumerate(
                    (pair_history[(aid, bid)] + [current_kill])[-5:],
                    start=-min(4, len(pair_history[(aid, bid)])),
                ):
                    geometry_rows.append({
                        "checkpoint": f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}",
                        "condition": condition, "episode": episode, "death_step": step,
                        "relative_step": relative, "attacker": aid, "target": bid, **historic,
                    })
        for aid, bid in pre_alive_pairs:
            post = _pair_geometry(env, aid, bid)
            previous_streak = int(pre[(aid, bid)]["attack_streak"])
            resulting_streak = previous_streak + 1 if post["gate"] else 0
            counters["gate_exposures"] += 1; per_agent[aid]["exposures"] += 1
            counters["gate_active"] += int(post["gate"]); per_agent[aid]["gate_active"] += int(post["gate"])
            counters["gate_entries"] += int(post["gate"] and previous_streak == 0)
            per_agent[aid]["streak1"] += int(resulting_streak >= 1)
            per_agent[aid]["streak2"] += int(resulting_streak >= 2)
            history_row = {
                "decision_step": step, **post, "attack_streak": resulting_streak,
                "reward_selected_target": targets[aid],
                "target_match": reward_target_matches(aid, bid, targets),
                "direct_visibility": pre[(aid, bid)]["direct_visibility"],
                "datalink_visibility": pre[(aid, bid)]["datalink_visibility"],
                **command_fields[aid],
            }
            pair_history[(aid, bid)].append(history_row)
            pair_history[(aid, bid)] = pair_history[(aid, bid)][-5:]
            if (aid, bid) in event_set:
                attack_rows.append({
                    "checkpoint": f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}",
                    "condition": condition, "episode": episode, "decision_step": step,
                    "attacker": aid, "target": bid,
                    "reward_selected_target": targets[aid],
                    "reward_target_match": reward_target_matches(aid, bid, targets),
                    **history_row,
                })
        done = bool(terminated or truncated)
    summary = info["episode_summary"]
    episode_row = {
        "checkpoint": f"{loaded['checkpoint'].parent.name}/{loaded['checkpoint'].name}",
        "training_seed": loaded["training_seed"], "sampled_steps": loaded["sampled_steps"],
        "condition": condition, "episode": episode, "environment_seed": environment_seed,
        "action_seed": action_seed if condition in ("POLICY", "UAV_ACTION_CYCLIC") else None,
        "control_action_seed": control_action_seed if condition == "UNIFORM_RANDOM" else None,
        "outcome": summary["outcome"], "episode_return": float(summary["episode_return"]),
        "episode_length": int(summary["episode_length"]),
        "MAV_survival": bool(summary["mav_survived"]),
        "UAV_survivors": int(summary["red_uav_survivors"]),
        "red_attack_kills": int(summary["red_attack_kills"]),
        "blue_attack_kills": int(summary["blue_attack_kills"]),
        "red_gate_entry_count": counters["gate_entries"],
        "red_gate_active_pair_steps": counters["gate_active"],
        "red_gate_pair_exposures": counters["gate_exposures"],
        "red_gate_active_fraction": counters["gate_active"] / max(counters["gate_exposures"], 1),
        "streak2_opportunities": counters["streak2"],
        "attack_event_pair_count": counters["attack_events"],
        "simultaneous_multi_target_red_attack_steps": counters["multi_target_steps"],
        "ambiguous_blue_death_count": counters["ambiguous_deaths"],
        "MAV_attack_event_pair_count": counters["mav_events"],
        "MAV_death_candidate_count": counters["mav_candidates"],
        "MAV_only_candidate_death_count": counters["mav_only"],
        "counterfactual_total_opportunities": total_opportunities,
        "counterfactual_audited_opportunities": audited_opportunities,
    }
    for aid in RED_IDS:
        values = per_agent[aid]
        episode_row.update({
            f"{aid}_gate_active_fraction": values["gate_active"] / max(values["exposures"], 1),
            f"{aid}_streak_ge_1_count": values["streak1"],
            f"{aid}_streak_ge_2_count": values["streak2"],
            f"{aid}_attack_event_pair_count": values["events"],
        })
    all_rows = [episode_row, *attack_rows, *death_rows, *geometry_rows, *cf_rows]
    _finite_rows(all_rows)
    return {
        "episodes": [episode_row], "attacks": attack_rows, "deaths": death_rows,
        "geometry": geometry_rows, "counterfactuals": cf_rows,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0


def summarize_conditions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    checkpoints = list(dict.fromkeys(str(row["checkpoint"]) for row in rows))
    for checkpoint in checkpoints:
        for condition in CONDITIONS:
            current = [row for row in rows if row["checkpoint"] == checkpoint and row["condition"] == condition]
            if not current:
                continue
            result.append({
                "checkpoint": checkpoint, "training_seed": int(current[0]["training_seed"]),
                "condition": condition, "episodes": len(current),
                "red_win_rate": sum(row["outcome"] == "red" for row in current) / len(current),
                "blue_win_rate": sum(row["outcome"] == "blue" for row in current) / len(current),
                "draw_rate": sum(row["outcome"] == "draw" for row in current) / len(current),
                **{f"mean_{field}": _mean(current, field) for field in (
                    "episode_return", "episode_length", "MAV_survival", "UAV_survivors",
                    "red_attack_kills", "blue_attack_kills", "red_gate_active_fraction",
                    "streak2_opportunities", "attack_event_pair_count",
                    "simultaneous_multi_target_red_attack_steps", "ambiguous_blue_death_count",
                    "MAV_attack_event_pair_count", "MAV_death_candidate_count",
                    "MAV_only_candidate_death_count",
                )},
            })
    return result


def paired_condition_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    checkpoints = list(dict.fromkeys(str(row["checkpoint"]) for row in rows))
    metrics = ("red_attack_kills", "episode_return", "red_gate_active_fraction", "streak2_opportunities", "attack_event_pair_count")
    for checkpoint in checkpoints:
        policy = {int(row["episode"]): row for row in rows if row["checkpoint"] == checkpoint and row["condition"] == "POLICY"}
        for condition in CONDITIONS[1:]:
            control = {int(row["episode"]): row for row in rows if row["checkpoint"] == checkpoint and row["condition"] == condition}
            shared = sorted(set(policy) & set(control))
            result.append({
                "checkpoint": checkpoint, "control_condition": condition, "paired_episodes": len(shared),
                "policy_minus_control_red_win_rate": float(np.mean([
                    float(policy[e]["outcome"] == "red") - float(control[e]["outcome"] == "red") for e in shared
                ])),
                **{
                    f"policy_minus_control_mean_{field}": float(np.mean([
                        float(policy[e][field]) - float(control[e][field]) for e in shared
                    ])) for field in metrics
                },
            })
    return result


def summarize_prekill(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for relative in sorted({int(row["relative_step"]) for row in rows}):
        current = [row for row in rows if int(row["relative_step"]) == relative]
        result.append({
            "relative_step": relative, "candidate_pair_samples": len(current),
            "mean_distance": _mean(current, "distance"), "mean_ATA_deg": _mean(current, "ATA_deg"),
            "mean_AA_deg": _mean(current, "AA_deg"), "gate_occupancy": _mean(current, "gate"),
            "positive_closing_fraction": sum(float(row["closing_rate"]) > 0 for row in current) / len(current),
            "mean_attack_streak": _mean(current, "attack_streak"),
        })
    return result


def summarize_counterfactuals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    summaries = []
    for checkpoint in dict.fromkeys(str(row["checkpoint"]) for row in rows):
        current = [row for row in rows if row["checkpoint"] == checkpoint]
        groups: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in current:
            groups.setdefault(str(row["state_group_id"]), {})[str(row["branch"])] = row
        summary: dict[str, Any] = {"checkpoint": checkpoint, "opportunities": len(groups)}
        for branch in ("NORMAL_POLICY", "ATTACKER_TRIM_ONE_STEP", "ATTACKER_PEER_ACTION_ONE_STEP"):
            available = [group[branch] for group in groups.values() if branch in group]
            summary[f"{branch}_opportunities"] = len(available)
            summary[f"{branch}_completion_rate"] = _mean(available, "pair_completed")
        paired_trim = [g for g in groups.values() if "NORMAL_POLICY" in g and "ATTACKER_TRIM_ONE_STEP" in g]
        paired_peer = [g for g in groups.values() if "NORMAL_POLICY" in g and "ATTACKER_PEER_ACTION_ONE_STEP" in g]
        summary.update({
            "normal_success_trim_failure_rate": float(np.mean([
                g["NORMAL_POLICY"]["pair_completed"] and not g["ATTACKER_TRIM_ONE_STEP"]["pair_completed"] for g in paired_trim
            ])) if paired_trim else None,
            "trim_success_normal_failure_rate": float(np.mean([
                g["ATTACKER_TRIM_ONE_STEP"]["pair_completed"] and not g["NORMAL_POLICY"]["pair_completed"] for g in paired_trim
            ])) if paired_trim else None,
            "normal_success_peer_failure_rate": float(np.mean([
                g["NORMAL_POLICY"]["pair_completed"] and not g["ATTACKER_PEER_ACTION_ONE_STEP"]["pair_completed"] for g in paired_peer
            ])) if paired_peer else None,
            "peer_success_normal_failure_rate": float(np.mean([
                g["ATTACKER_PEER_ACTION_ONE_STEP"]["pair_completed"] and not g["NORMAL_POLICY"]["pair_completed"] for g in paired_peer
            ])) if paired_peer else None,
        })
        summaries.append(summary)
    return summaries


def add_alignment_summaries(
    condition_rows: list[dict[str, Any]], attack_rows: Sequence[Mapping[str, Any]],
    death_rows: Sequence[Mapping[str, Any]],
) -> None:
    for summary in condition_rows:
        checkpoint, condition = summary["checkpoint"], summary["condition"]
        attacks = [row for row in attack_rows if row["checkpoint"] == checkpoint
                   and row["condition"] == condition and row["attacker"] in UAV_IDS]
        deaths = [row for row in death_rows if row["checkpoint"] == checkpoint
                  and row["condition"] == condition and row["candidate_attacker"] in UAV_IDS]
        all_deaths = [row for row in death_rows if row["checkpoint"] == checkpoint
                      and row["condition"] == condition]
        death_events = {
            (int(row["episode"]), int(row["decision_step"]), str(row["target"])):
            int(row["candidate_attacker_count"])
            for row in all_deaths
        }
        summary.update({
            "uav_attack_event_candidates": len(attacks),
            "attack_event_reward_target_match_rate": _mean(attacks, "reward_target_match"),
            "uav_death_candidates": len(deaths),
            "death_candidate_reward_target_match_rate": _mean(deaths, "reward_target_match"),
            "three_step_reward_target_consistency_rate": _mean(
                deaths, "three_step_reward_target_consistent",
            ),
            "blue_death_events": len(death_events),
            **{
                f"candidate_attacker_count_{count}_share": (
                    sum(value == count for value in death_events.values()) / len(death_events)
                    if death_events else 0.0
                ) for count in range(1, 5)
            },
        })


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows); fieldnames = list(fields or (rows[0].keys() if rows else ("no_records",)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_outputs(output: Path, all_rows: Mapping[str, list[dict[str, Any]]], protocol: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    episodes = all_rows["episodes"]; attacks = all_rows["attacks"]
    deaths = all_rows["deaths"]; geometry = all_rows["geometry"]; cf = all_rows["counterfactuals"]
    condition_summary = summarize_conditions(episodes)
    add_alignment_summaries(condition_summary, attacks, deaths)
    paired = paired_condition_comparisons(episodes)
    geometry_summary = summarize_prekill(geometry)
    cf_summary = summarize_counterfactuals(cf)
    _write_csv(output / "condition_episode_records.csv", episodes, (*EPISODE_FIELDS, *[
        field for field in episodes[0].keys() if field not in EPISODE_FIELDS
    ]))
    _write_csv(output / "condition_summary.csv", condition_summary)
    _write_csv(output / "paired_condition_comparison.csv", paired)
    _write_csv(output / "attack_event_records.csv", attacks)
    _write_csv(output / "death_candidate_records.csv", deaths)
    _write_csv(output / "prekill_geometry_records.csv", geometry)
    _write_csv(output / "prekill_geometry_summary.csv", geometry_summary)
    _write_csv(output / "streak2_counterfactual_records.csv", cf)
    _write_csv(output / "streak2_counterfactual_summary.csv", cf_summary)
    summary = {
        "audit": "combat_skill_causality_audit_v1", "protocol": dict(protocol),
        "FACTS FROM CODE": {
            "attack_gate": "distance 1-3 km inclusive; ATA<30 deg; AA<90 deg; 3 consecutive decision steps",
            "explicit_fire_or_lock_required": False,
            "combat_requires_reward_target_or_visibility": False,
            "pairwise_resolver": True,
            "unique_killer_attribution": False,
        },
        "DESCRIPTIVE BEHAVIORAL EVIDENCE": {
            "condition_summary": condition_summary, "paired_condition_comparison": paired,
            "attack_event_records": len(attacks), "death_candidate_records": len(deaths),
            "prekill_geometry_summary": geometry_summary,
        },
        "PAIRED INTERVENTION EVIDENCE": cf_summary,
        "LIMITATIONS": [
            "The audit tests operational state-action dependence, not subjective intent or understanding.",
            "One-step streak-2 interventions isolate immediate gate completion, not long-horizon tactical causality.",
            "Candidate attack pairs are retained; no unique killer is fabricated for simultaneous candidates.",
        ],
    }
    (output / "combat_skill_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-seed", type=int, default=4000)
    parser.add_argument("--env-seed-start", type=int, default=3000)
    parser.add_argument("--uniform-random-seed", type=int, default=14000)
    parser.add_argument("--max-counterfactual-opportunities", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_counterfactual_opportunities < 0:
        raise ValueError("episodes must be positive and max opportunities nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    outer_rng = _torch_rng_state()
    loaded = [load_tacm_checkpoint(path, args.device) for path in args.checkpoints]
    actor_hashes = [_module_sha256(item["actors"]) for item in loaded]
    checkpoint_hashes = [_file_sha256(item["checkpoint"]) for item in loaded]
    reference_config = loaded[0]["environment_config"]
    if any(item["environment_config"] != reference_config for item in loaded[1:]):
        raise RuntimeError("paired combat audit requires identical resolved environment configs")
    all_rows = {name: [] for name in ("episodes", "attacks", "deaths", "geometry", "counterfactuals")}
    try:
        for item in loaded:
            audited_so_far = 0
            for episode in range(args.episodes):
                for condition in CONDITIONS:
                    remaining = max(args.max_counterfactual_opportunities - audited_so_far, 0)
                    rows = run_condition_episode(
                        item, condition=condition, profile=args.profile, episode=episode,
                        environment_seed=args.env_seed_start + episode,
                        action_seed=args.action_seed + episode,
                        control_action_seed=args.uniform_random_seed + episode,
                        device=args.device, max_counterfactual_opportunities=remaining,
                        collect_counterfactuals=(condition == "POLICY"),
                    )
                    for name in all_rows:
                        all_rows[name].extend(rows[name])
                    if condition == "POLICY":
                        audited_so_far += int(rows["episodes"][0]["counterfactual_audited_opportunities"])
        if [_module_sha256(item["actors"]) for item in loaded] != actor_hashes:
            raise AssertionError("audit modified actor parameters")
        if [_file_sha256(item["checkpoint"]) for item in loaded] != checkpoint_hashes:
            raise AssertionError("audit modified checkpoint files")
    finally:
        _restore_torch_rng(outer_rng)
    protocol = {
        "profile": args.profile, "episodes_per_checkpoint_condition": args.episodes,
        "environment_seed_start": args.env_seed_start,
        "environment_seed_end": args.env_seed_start + args.episodes - 1,
        "action_mode": "stochastic", "action_seed_start": args.action_seed,
        "action_seed_end": args.action_seed + args.episodes - 1,
        "uniform_random_seed_start": args.uniform_random_seed,
        "uniform_random_seed_end": args.uniform_random_seed + args.episodes - 1,
        "horizon": int(reference_config["simulation"]["max_decision_steps"]),
        "conditions": list(CONDITIONS),
        "max_counterfactual_opportunities_per_checkpoint": args.max_counterfactual_opportunities,
        "checkpoints": [{
            "path": str(item["checkpoint"]), "training_seed": item["training_seed"],
            "sampled_steps": item["sampled_steps"],
        } for item in loaded],
    }
    write_outputs(args.output_dir.expanduser().resolve(), all_rows, protocol)
    print("checkpoint | condition | red_win | red_kills | gate_fraction | attack_pairs", flush=True)
    for row in summarize_conditions(all_rows["episodes"]):
        print(f"{row['checkpoint']} | {row['condition']} | {row['red_win_rate']:.3f} | "
              f"{row['mean_red_attack_kills']:.3f} | {row['mean_red_gate_active_fraction']:.4f} | "
              f"{row['mean_attack_event_pair_count']:.3f}", flush=True)
    print(f"outputs: {args.output_dir.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
