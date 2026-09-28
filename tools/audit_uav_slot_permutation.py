"""Read-only realized-initial-state UAV slot permutation audit.

The audit changes only which reset-realized UAV ``AircraftState`` is assigned
to UAV1/UAV2/UAV3 before the first action.  Actor identities, aircraft objects,
specifications, rewards, and all non-UAV states remain unchanged.  Results are
descriptive and do not by themselves establish actor or slot causality.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import fields
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithm.happo.evaluation import summarize_records
from env.mavuav import BLUE_IDS, OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from env.models import AircraftState
from tools.audit_same_role_policy_divergence import (
    LoadedRun, _parse_mapping, assert_actor_state_unchanged, load_run,
)


UAV_IDS = RED_IDS[1:]
SOURCE_SLOTS = ("S1", "S2", "S3")
SOURCE_ID_BY_SLOT = dict(zip(SOURCE_SLOTS, UAV_IDS))
CAUSES = ("alive", "boundary", "blue_attack", "other")
AGENT_FIELDS = (
    "run", "method", "training_seed", "sampled_steps", "permutation",
    "UAV1_source_slot", "UAV2_source_slot", "UAV3_source_slot", "episode",
    "env_seed", "action_seed", "actor_id", "source_slot", "final_death_cause",
    "alive", "boundary", "blue_attack", "other", "team_outcome",
    "episode_return", "red_attack_kills", "blue_attack_kills", "episode_length",
    "mav_survived", "uav_survivors",
)
TEAM_FIELDS = (
    "run", "method", "training_seed", "sampled_steps", "permutation",
    "UAV1_source_slot", "UAV2_source_slot", "UAV3_source_slot", "episode",
    "env_seed", "action_seed", "outcome", "episode_return", "red_attack_kills",
    "blue_attack_kills", "episode_length", "mav_survived", "uav_survivors",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_mapping, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--profile", choices=("learnability", "main"), default="learnability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("--run labels must be unique")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def slot_permutations() -> list[dict[str, str]]:
    """Return all mappings ``controlled UAV ID -> realized source slot``."""
    return [dict(zip(UAV_IDS, ordering)) for ordering in permutations(SOURCE_SLOTS)]


def permutation_label(mapping: Mapping[str, str]) -> str:
    if set(mapping) != set(UAV_IDS) or set(mapping.values()) != set(SOURCE_SLOTS):
        raise ValueError("slot mapping must be a bijection from UAV IDs to S1/S2/S3")
    return "__".join(
        f"U{index}_{mapping[aid]}" for index, aid in enumerate(UAV_IDS, start=1)
    )


def is_identity_permutation(mapping: Mapping[str, str]) -> bool:
    return all(mapping[aid] == slot for aid, slot in zip(UAV_IDS, SOURCE_SLOTS))


def copy_aircraft_state(state: AircraftState) -> AircraftState:
    """Copy every dataclass field without sharing the mutable state object."""
    values = {field.name: deepcopy(getattr(state, field.name)) for field in fields(state)}
    return type(state)(**values)


def snapshot_realized_uav_slots(
    env: HeterogeneousMAVUAVAirCombatEnv,
) -> dict[str, AircraftState]:
    return {
        slot: copy_aircraft_state(env.entities[aid].state)
        for slot, aid in zip(SOURCE_SLOTS, UAV_IDS)
    }


def apply_realized_slot_permutation(
    env: HeterogeneousMAVUAVAirCombatEnv,
    snapshots: Mapping[str, AircraftState],
    mapping: Mapping[str, str],
) -> None:
    if set(snapshots) != set(SOURCE_SLOTS):
        raise ValueError("snapshots must contain S1/S2/S3")
    permutation_label(mapping)  # Validate the bijection before mutating the env.
    for aid in UAV_IDS:
        env.entities[aid].state = copy_aircraft_state(snapshots[mapping[aid]])
    state_ids = [id(env.entities[aid].state) for aid in UAV_IDS]
    if len(set(state_ids)) != len(state_ids):
        raise AssertionError("permuted UAV states alias one another")


def reset_with_realized_permutation(
    env: HeterogeneousMAVUAVAirCombatEnv,
    seed: int,
    mapping: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, AircraftState]]:
    reset_observations, _ = env.reset(seed=int(seed))
    snapshots = snapshot_realized_uav_slots(env)
    apply_realized_slot_permutation(env, snapshots, mapping)
    observations = env._observations()
    if is_identity_permutation(mapping):
        for aid in RED_IDS:
            if not np.array_equal(observations[aid], reset_observations[aid]):
                raise AssertionError("identity slot permutation changed reset observations")
    return observations, snapshots


def validate_cross_run_environment_contract(loaded_runs: Sequence[LoadedRun]) -> None:
    if not loaded_runs:
        raise RuntimeError("slot permutation audit requires at least one run")
    expected_version = loaded_runs[0].environment_version
    expected_config = loaded_runs[0].env_config
    for run in loaded_runs:
        if run.observation_dim != OBS_DIM:
            raise RuntimeError(f"slot permutation audit requires observation_dim={OBS_DIM}")
        if run.environment_version != expected_version or run.env_config != expected_config:
            raise RuntimeError(
                "UAV slot permutation audit requires identical resolved environment configs"
            )


def categorize_death(cause: str | None) -> str:
    if cause is None:
        return "alive"
    return cause if cause in ("boundary", "blue_attack") else "other"


def rollout_permuted_episode(
    actors: Any,
    env_config: Mapping[str, Any],
    profile: str,
    mapping: Mapping[str, str],
    *,
    episode: int,
    env_seed: int,
    action_seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run one independent stochastic episode with fixed actor-to-ID mapping."""
    env = HeterogeneousMAVUAVAirCombatEnv(env_config, profile=profile)
    observations, _ = reset_with_realized_permutation(env, env_seed, mapping)
    torch.manual_seed(int(action_seed))
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(action_seed))
    death_causes: dict[str, str] = {}
    done = False
    while not done:
        actions: list[np.ndarray] = []
        with torch.no_grad():
            # This order is the formal feed-forward evaluator contract.
            for actor_index, aid in enumerate(RED_IDS):
                observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
                action, _ = actors.actors[actor_index].sample(observation, deterministic=False)
                actions.append(action.squeeze(0).cpu().numpy())
        observations, _, terminated, truncated, info = env.step(np.asarray(actions))
        for aid, cause in info.get("death_causes", {}).items():
            if aid in RED_IDS:
                death_causes[aid] = categorize_death(str(cause))
        done = bool(terminated or truncated)
    summary = dict(info["episode_summary"])
    final_causes = {
        aid: death_causes.get(aid, "alive") for aid in RED_IDS
    }
    return summary, final_causes


def records_for_episode(
    loaded: LoadedRun,
    mapping: Mapping[str, str],
    episode: int,
    env_seed: int,
    action_seed: int,
    profile: str,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, causes = rollout_permuted_episode(
        loaded.actors, loaded.env_config, profile, mapping, episode=episode,
        env_seed=env_seed, action_seed=action_seed, device=device,
    )
    label = permutation_label(mapping)
    mapping_fields = {f"{aid}_source_slot": mapping[aid] for aid in UAV_IDS}
    base = {
        "run": loaded.label, "method": loaded.method,
        "training_seed": loaded.seed, "sampled_steps": loaded.sampled_steps,
        "permutation": label, **mapping_fields, "episode": episode,
        "env_seed": env_seed, "action_seed": action_seed,
        "episode_return": float(summary["episode_return"]),
        "red_attack_kills": int(summary["red_attack_kills"]),
        "blue_attack_kills": int(summary["blue_attack_kills"]),
        "episode_length": int(summary["episode_length"]),
        "mav_survived": bool(summary["mav_survived"]),
        "uav_survivors": int(summary["red_uav_survivors"]),
    }
    team = {**base, "outcome": summary["outcome"]}
    agents: list[dict[str, Any]] = []
    for aid in UAV_IDS:
        cause = causes[aid]
        agents.append({
            **base, "actor_id": aid, "source_slot": mapping[aid],
            "final_death_cause": cause, **{name: int(cause == name) for name in CAUSES},
            "team_outcome": summary["outcome"],
        })
    return team, agents


def summarize_permutations(team_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(str(row["run"]), str(row["permutation"])) for row in team_rows})
    for run, label in keys:
        rows = [row for row in team_rows if row["run"] == run and row["permutation"] == label]
        mapping = {aid: str(rows[0][f"{aid}_source_slot"]) for aid in UAV_IDS}
        formal_rows = [{
            "outcome": row["outcome"], "episode_return": row["episode_return"],
            "mav_survived": row["mav_survived"],
            "red_uav_survivors": row["uav_survivors"],
            "red_attack_kills": row["red_attack_kills"],
            "blue_attack_kills": row["blue_attack_kills"],
            "episode_length": row["episode_length"],
        } for row in rows]
        stats = summarize_records(formal_rows)
        result.append({
            "run": run, "method": rows[0]["method"],
            "training_seed": rows[0]["training_seed"],
            "sampled_steps": rows[0]["sampled_steps"], "permutation": label,
            **{f"{aid}_source_slot": mapping[aid] for aid in UAV_IDS},
            "identity": int(is_identity_permutation(mapping)), "episodes": len(rows),
            "win_rate": stats["red_win_rate"], "blue_win_rate": stats["blue_win_rate"],
            "draw_rate": stats["draw_rate"], "mean_return": stats["mean_episode_return"],
            "mean_red_kills": stats["mean_red_attack_kills"],
            "mav_survival": stats["MAV_survival_rate"],
            "mean_uav_survivors": stats["mean_UAV_survivors"],
            "mean_blue_attack_kills": stats["mean_blue_attack_kills"],
            "mean_episode_length": stats["mean_episode_length"],
        })
    return result


def _death_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    counts = {cause: sum(int(row[cause]) for row in rows) for cause in CAUSES}
    return {
        "sample_count": count,
        "boundary_count": counts["boundary"],
        "boundary_rate": counts["boundary"] / count,
        "blue_attack_count": counts["blue_attack"],
        "blue_attack_rate": counts["blue_attack"] / count,
        "survival_count": counts["alive"],
        "survival_rate": counts["alive"] / count,
        "other_count": counts["other"],
        "total_loss_rate": (count - counts["alive"]) / count,
    }


def build_actor_slot_matrix(
    agent_rows: Sequence[Mapping[str, Any]], episodes: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in sorted({str(row["run"]) for row in agent_rows}):
        for aid in UAV_IDS:
            for slot in SOURCE_SLOTS:
                rows = [
                    row for row in agent_rows
                    if row["run"] == run and row["actor_id"] == aid
                    and row["source_slot"] == slot
                ]
                if len(rows) != 2 * int(episodes):
                    raise AssertionError(
                        f"actor×slot coverage mismatch for {run}/{aid}/{slot}: "
                        f"{len(rows)} != {2 * int(episodes)}"
                    )
                output.append({"run": run, "actor_id": aid, "source_slot": slot,
                               **_death_stats(rows)})
    return output


def build_marginals(
    agent_rows: Sequence[Mapping[str, Any]], field: str, values: Iterable[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in sorted({str(row["run"]) for row in agent_rows}):
        for value in values:
            rows = [row for row in agent_rows if row["run"] == run and row[field] == value]
            output.append({"run": run, field: value, **_death_stats(rows)})
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields_: Sequence[str] | None = None) -> None:
    fieldnames = list(fields_ or [])
    if not fieldnames:
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = str(args.device)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a tiny CPU audit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    loaded_runs: list[LoadedRun] = []
    try:
        loaded_runs = [load_run(label, path.resolve(), device) for label, path in args.run]
        validate_cross_run_environment_contract(loaded_runs)
        team_rows: list[dict[str, Any]] = []
        agent_rows: list[dict[str, Any]] = []
        all_permutations = slot_permutations()
        for loaded in loaded_runs:
            for mapping in all_permutations:
                for episode in range(int(args.episodes)):
                    team, agents = records_for_episode(
                        loaded, mapping, episode,
                        int(args.env_seed) + episode, int(args.action_seed) + episode,
                        args.profile, device,
                    )
                    team_rows.append(team)
                    agent_rows.extend(agents)
        permutation_summary = summarize_permutations(team_rows)
        actor_slot = build_actor_slot_matrix(agent_rows, args.episodes)
        actor_marginal = build_marginals(agent_rows, "actor_id", UAV_IDS)
        slot_marginal = build_marginals(agent_rows, "source_slot", SOURCE_SLOTS)
        _write_csv(output / "episode_agent_records.csv", agent_rows, AGENT_FIELDS)
        _write_csv(output / "episode_team_records.csv", team_rows, TEAM_FIELDS)
        _write_csv(output / "permutation_summary.csv", permutation_summary)
        _write_csv(output / "actor_slot_matrix.csv", actor_slot)
        _write_csv(output / "actor_marginal.csv", actor_marginal)
        _write_csv(output / "slot_marginal.csv", slot_marginal)
        summary = {
            "audit": "realized_uav_initial_slot_permutation", "read_only": True,
            "profile": args.profile, "episodes_per_permutation": int(args.episodes),
            "permutation_count": len(all_permutations),
            "env_seed_start": int(args.env_seed),
            "env_seed_end": int(args.env_seed) + int(args.episodes) - 1,
            "action_seed_start": int(args.action_seed),
            "action_seed_end": int(args.action_seed) + int(args.episodes) - 1,
            "actor_mapping": {"0": "MAV", "1": "UAV1", "2": "UAV2", "3": "UAV3"},
            "runs": {
                run.label: {
                    "method": run.method, "training_seed": run.seed,
                    "sampled_steps": run.sampled_steps, "checkpoint": str(run.checkpoint),
                    "checkpoint_sha256": run.checkpoint_digest,
                } for run in loaded_runs
            },
            "interpretation_limit": (
                "Descriptive paired intervention only. Actor persistence, slot tracking, and "
                "actor×slot interaction are compatible patterns, not automatic causal verdicts."
            ),
        }
        (output / "audit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (output / "README.txt").write_text(
            "Realized UAV initial-slot permutation audit\n"
            "===========================================\n"
            "Each permutation replays the same reset seed and action seed, then moves the full "
            "reset-realized AircraftState among UAV1/UAV2/UAV3 before the first action.\n"
            "Pattern A (actor persistence): one actor is pathological across S1/S2/S3.\n"
            "Pattern B (slot tracking): one source slot is pathological across actors.\n"
            "Pattern C (interaction): only a specific actor×slot cell is pathological.\n"
            "Pattern D (broad degradation): non-identity permutations broadly harm task performance.\n"
            + summary["interpretation_limit"] + "\n",
            encoding="utf-8",
        )
        for run in loaded_runs:
            assert_actor_state_unchanged(run)
            print(f"{run.label}: method={run.method} seed={run.seed} steps={run.sampled_steps}")
            identity = next(row for row in permutation_summary if row["run"] == run.label and row["identity"])
            print(
                f"  identity: W={identity['win_rate']:.1%} return={identity['mean_return']:.3f} "
                f"kills={identity['mean_red_kills']:.3f} UAV={identity['mean_uav_survivors']:.3f}"
            )
            print("  actor×slot boundary matrix:")
            print("             S1       S2       S3")
            for aid in UAV_IDS:
                rates = [next(row["boundary_rate"] for row in actor_slot
                              if row["run"] == run.label and row["actor_id"] == aid
                              and row["source_slot"] == slot) for slot in SOURCE_SLOTS]
                print(f"    {aid:<4} " + " ".join(f"{rate:8.1%}" for rate in rates))
            print("  actor marginal boundary: " + " / ".join(
                f"{aid}={next(row['boundary_rate'] for row in actor_marginal if row['run'] == run.label and row['actor_id'] == aid):.1%}"
                for aid in UAV_IDS
            ))
            print("  slot marginal boundary: " + " / ".join(
                f"{slot}={next(row['boundary_rate'] for row in slot_marginal if row['run'] == run.label and row['source_slot'] == slot):.1%}"
                for slot in SOURCE_SLOTS
            ))
            for row in (item for item in permutation_summary if item["run"] == run.label):
                print(
                    f"  {row['permutation']}: W={row['win_rate']:.1%} "
                    f"return={row['mean_return']:.3f} kills={row['mean_red_kills']:.3f} "
                    f"UAV={row['mean_uav_survivors']:.3f}"
                )
        return summary
    finally:
        for loaded in loaded_runs:
            assert_actor_state_unchanged(loaded)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_rng])


def main(argv: Sequence[str] | None = None) -> None:
    run_audit(parse_args(argv))


if __name__ == "__main__":
    main()
