"""Read-only paired continuation-horizon audit for TACM-RGAA-v1 checkpoints."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
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

from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.dbm_rgaa import DBM_RGAA_METHOD, build_method_actors, dbm_metadata
from algorithm.happo.tacm_rgaa import TACM_RGAA_METHOD, tacm_metadata
from env.mavuav import (
    BLUE_IDS, RED_IDS, ROLE_REWARD_MODES, HeterogeneousMAVUAVAirCombatEnv,
    load_environment_config,
)


SUPPORTED_ENVIRONMENT_VERSIONS = frozenset((
    "heterogeneous_mavuav_4v4_v3_9",
    "heterogeneous_mavuav_4v4_v3_10",
))
SUPPORTED_REWARD_MODE = "heterogeneous_role_coupled_gate_v1"
DEFAULT_HORIZONS = (75, 100, 125, 150, 200)
JITTER_FIELDS = (
    "team_xy_jitter", "slot_xy_jitter", "altitude_jitter",
    "speed_jitter", "heading_jitter_deg",
)

EPISODE_FIELDS = (
    "checkpoint", "checkpoint_path", "method_variant", "training_seed", "sampled_steps",
    "episode", "environment_seed", "action_seed", "action_mode", "environment_profile",
    "environment_version", "observation_horizon", "audit_max_horizon", "horizon",
    "outcome", "resolved_by_horizon", "actual_terminal_step", "actual_terminal_outcome",
    "mav_survived", "red_uav_survivors", "blue_survivors", "red_attack_kills",
    "blue_attack_kills", "cumulative_team_return", "blue_remaining_to_eliminate",
    "cutoff_or_terminal_length", "outcome_at_75", "outcome_conversion_from_75",
    "action_trace_sha256",
)


class ContinuationAuditEnv(HeterogeneousMAVUAVAirCombatEnv):
    """Extend timeout only while preserving the checkpoint observation clock."""

    def __init__(self, config: Mapping[str, Any], *, audit_max_decision_steps: int,
                 profile: str = "main") -> None:
        super().__init__(config, profile=profile)
        self.observation_horizon = int(self.max_decision_steps)
        self.audit_max_decision_steps = int(audit_max_decision_steps)
        if self.audit_max_decision_steps < self.observation_horizon:
            raise ValueError("audit horizon cannot be shorter than observation horizon")

    def _termination(self) -> tuple[bool, bool, str | None]:
        # The parent owns terminal priority and all terminal semantics.  Only
        # its timeout threshold is temporarily substituted during this call.
        observation_horizon = self.max_decision_steps
        try:
            self.max_decision_steps = self.audit_max_decision_steps
            return super()._termination()
        finally:
            self.max_decision_steps = observation_horizon


def _resolved_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _checkpoint_label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def validate_audit_contract(payload: Mapping[str, Any], env_config: Mapping[str, Any]) -> int:
    validate_checkpoint_contract(dict(payload), dict(env_config))
    if env_config["environment_version"] not in SUPPORTED_ENVIRONMENT_VERSIONS:
        raise RuntimeError("continuation audit only supports the frozen v3.9/v3.10 environments")
    reward_mode = payload.get("reward_mode")
    if reward_mode != SUPPORTED_REWARD_MODE or reward_mode not in ROLE_REWARD_MODES:
        raise RuntimeError("continuation audit requires the v3.9/v3.10 role-reward contract")
    if float(env_config["reward"]["terminal_draw"]) != 0.0:
        raise RuntimeError("single-pass continuation audit requires terminal_draw == 0")
    if str(env_config.get("shaping", {}).get("mode", "absolute")) == "potential":
        raise RuntimeError("single-pass continuation audit rejects timeout-dependent potential shaping")
    observation_horizon = int(env_config["simulation"]["max_decision_steps"])
    if observation_horizon != 75:
        raise RuntimeError("supported v3.9/v3.10 continuation audit requires observation horizon 75")
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    method = payload.get("method_variant", trainer_config.get("method_variant"))
    if method != TACM_RGAA_METHOD or payload.get("algorithm") != "tacm_rgaa_happo":
        raise RuntimeError("continuation audit requires a TACM-RGAA-v1 checkpoint")
    return observation_horizon


def load_tacm_checkpoint(checkpoint: Path, device: str) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "environment_config" not in payload:
        raise RuntimeError("TACM checkpoint is missing resolved environment_config")
    raw_environment = deepcopy(payload["environment_config"])
    env_config = load_environment_config(raw_environment)
    observation_horizon = validate_audit_contract(payload, env_config)
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    if payload.get("actor_variant", trainer_config.get("actor_variant")) != "vanilla":
        raise RuntimeError("TACM continuation audit requires vanilla actor_variant metadata")
    if payload.get("critic_variant", trainer_config.get("critic_variant")) != "mlp":
        raise RuntimeError("TACM continuation audit requires mlp critic metadata")
    dbm_config = dict(trainer_config)
    dbm_config["method_variant"] = DBM_RGAA_METHOD
    for field, expected in dbm_metadata(dbm_config).items():
        if payload.get(field) != expected:
            raise RuntimeError(f"incompatible TACM DBM actor contract: {field}")
    for field, expected in tacm_metadata(trainer_config).items():
        if payload.get(field) != expected:
            raise RuntimeError(f"incompatible TACM checkpoint contract: {field}")
    actors = build_method_actors(
        method_variant=DBM_RGAA_METHOD,
        training_seed=int(trainer_config["seed"]),
        hidden_dim=int(trainer_config["hidden_dim"]),
        log_std_init=float(trainer_config.get("actor_log_std_init", -0.5)),
        role_module_enabled=bool(trainer_config.get("role_module_enabled", True)),
        dbm_role_count=int(trainer_config.get("dbm_role_count", 2)),
        dbm_residual_scale=float(trainer_config.get("dbm_residual_scale", 0.25)),
        dbm_expert_init_scale=float(trainer_config.get("dbm_init_scale", 0.01)),
        uav_actor_hidden_dim=int(trainer_config.get("uav_actor_hidden_dim", 131)),
    ).to(device)
    actors.load_state_dict(payload["actors"], strict=True)
    actors.eval()
    if payload["environment_config"] != raw_environment:
        raise AssertionError("checkpoint environment_config was mutated during loading")
    return {
        "checkpoint": checkpoint, "payload": payload, "environment_config": env_config,
        "observation_horizon": observation_horizon, "actors": actors,
        "training_seed": int(trainer_config["seed"]),
        "sampled_steps": int(payload.get("sampled_steps", 0)),
    }


def _snapshot(env: ContinuationAuditEnv) -> dict[str, Any]:
    return {
        "mav_survived": bool(env.entities["MAV"].state.alive),
        "red_uav_survivors": int(sum(env.entities[aid].state.alive for aid in RED_IDS[1:])),
        "blue_survivors": int(sum(env.entities[aid].state.alive for aid in BLUE_IDS)),
        "red_attack_kills": int(len(env._red_attack_kills)),
        "blue_attack_kills": int(len(env._blue_attack_kills)),
        "cumulative_team_return": float(env.episode_return),
    }


def project_horizon_records(
    snapshots: Mapping[int, Mapping[str, Any]], horizons: Sequence[int], *,
    actual_terminal_step: int | None, actual_terminal_outcome: str | None,
    metadata: Mapping[str, Any], action_trace_sha256: str,
) -> list[dict[str, Any]]:
    horizons = tuple(sorted(int(value) for value in horizons))
    base_horizon = int(metadata["observation_horizon"])
    if base_horizon not in horizons:
        raise ValueError("horizons must include the checkpoint observation horizon")
    outcomes = {
        horizon: (
            actual_terminal_outcome
            if actual_terminal_step is not None and actual_terminal_step <= horizon
            else "draw"
        )
        for horizon in horizons
    }
    base_outcome = outcomes[base_horizon]
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        snapshot = snapshots[horizon]
        resolved = actual_terminal_step is not None and actual_terminal_step <= horizon
        outcome = outcomes[horizon]
        rows.append({
            **metadata,
            "horizon": horizon,
            "outcome": outcome,
            "resolved_by_horizon": bool(resolved),
            "actual_terminal_step": actual_terminal_step,
            "actual_terminal_outcome": actual_terminal_outcome,
            **snapshot,
            "blue_remaining_to_eliminate": int(snapshot["blue_survivors"]) if outcome == "draw" else 0,
            "cutoff_or_terminal_length": int(actual_terminal_step if resolved else horizon),
            "outcome_at_75": base_outcome,
            "outcome_conversion_from_75": bool(outcome != base_outcome),
            "action_trace_sha256": action_trace_sha256,
        })
    return rows


def run_continuation_episode(
    actors: Any, env_config: Mapping[str, Any], *, profile: str, episode: int,
    env_seed: int, action_mode: str, action_seed: int | None, device: str,
    horizons: Sequence[int], metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    audit_max = max(horizons)
    env = ContinuationAuditEnv(env_config, audit_max_decision_steps=audit_max, profile=profile)
    observations, _ = env.reset(seed=env_seed)
    deterministic = action_mode == "deterministic"
    if not deterministic and action_seed is not None:
        torch.manual_seed(int(action_seed))
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(int(action_seed))
    digest = hashlib.sha256()
    snapshots: dict[int, dict[str, Any]] = {}
    actual_step: int | None = None
    actual_outcome: str | None = None
    final_snapshot: dict[str, Any] | None = None
    while env.step_count < audit_max:
        actions = []
        with torch.no_grad():
            for index, aid in enumerate(RED_IDS):
                observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
                action, _ = actors.actors[index].sample(observation, deterministic=deterministic)
                actions.append(action.squeeze(0).cpu().numpy())
        action_array = np.asarray(actions, dtype=np.float32)
        digest.update(action_array.tobytes())
        observations, _, terminated, truncated, info = env.step(action_array)
        if env.step_count in horizons:
            snapshots[env.step_count] = _snapshot(env)
        if terminated:
            actual_step = int(env.step_count)
            actual_outcome = str(info["outcome"])
            final_snapshot = _snapshot(env)
            break
        if truncated:
            # Audit-limit truncation is unresolved, not an actual combat terminal.
            final_snapshot = _snapshot(env)
            break
    if final_snapshot is None:
        final_snapshot = _snapshot(env)
    for horizon in horizons:
        if horizon not in snapshots:
            snapshots[horizon] = deepcopy(final_snapshot)
    return project_horizon_records(
        snapshots, horizons, actual_terminal_step=actual_step,
        actual_terminal_outcome=actual_outcome,
        metadata={**metadata, "episode": int(episode), "environment_seed": int(env_seed),
                  "action_seed": action_seed},
        action_trace_sha256=digest.hexdigest(),
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0


def summarize_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    checkpoint = str(rows[0]["checkpoint"])
    horizons = sorted({int(row["horizon"]) for row in rows})
    base = [row for row in rows if int(row["horizon"]) == int(row["observation_horizon"])]
    base_draw_episodes = {int(row["episode"]) for row in base if row["outcome"] == "draw"}
    base_win = sum(r["outcome"] == "red" for r in base) / len(base)
    base_draw = sum(r["outcome"] == "draw" for r in base) / len(base)
    draw75_kills = [int(row["red_attack_kills"]) for row in base if row["outcome"] == "draw"]
    for horizon in horizons:
        current = [row for row in rows if int(row["horizon"]) == horizon]
        red_rate = sum(row["outcome"] == "red" for row in current) / len(current)
        blue_rate = sum(row["outcome"] == "blue" for row in current) / len(current)
        draw_rate = sum(row["outcome"] == "draw" for row in current) / len(current)
        base_draw_rows = [row for row in current if int(row["episode"]) in base_draw_episodes]
        red_converted = sum(row["outcome"] == "red" for row in base_draw_rows)
        blue_converted = sum(row["outcome"] == "blue" for row in base_draw_rows)
        completion = [int(row["actual_terminal_step"]) for row in base_draw_rows
                      if row["outcome"] == "red" and row["actual_terminal_step"] is not None]
        results.append({
            "checkpoint": checkpoint,
            "training_seed": int(current[0]["training_seed"]),
            "sampled_steps": int(current[0]["sampled_steps"]),
            "horizon": horizon, "episodes": len(current),
            "red_win_rate": red_rate, "blue_win_rate": blue_rate, "draw_rate": draw_rate,
            "MAV_survival_rate": _mean(current, "mav_survived"),
            "mean_red_attack_kills": _mean(current, "red_attack_kills"),
            "mean_red_uav_survivors": _mean(current, "red_uav_survivors"),
            "mean_blue_survivors": _mean(current, "blue_survivors"),
            "mean_cutoff_or_terminal_length": _mean(current, "cutoff_or_terminal_length"),
            "red_win_gain_pp_vs_75": 100.0 * (red_rate - base_win),
            "draw_reduction_pp_vs_75": 100.0 * (base_draw - draw_rate),
            "draw75_count": len(base_draw_episodes),
            "draw75_to_red_rate": red_converted / len(base_draw_episodes) if base_draw_episodes else 0.0,
            "draw75_to_blue_rate": blue_converted / len(base_draw_episodes) if base_draw_episodes else 0.0,
            "draw75_still_unresolved_rate": (
                sum(row["outcome"] == "draw" for row in base_draw_rows) / len(base_draw_episodes)
                if base_draw_episodes else 0.0
            ),
            **{f"draw75_{kills}_kill_count": draw75_kills.count(kills) for kills in range(5)},
            "draw75_3kill_rate": draw75_kills.count(3) / len(draw75_kills) if draw75_kills else 0.0,
            "new_red_win_terminal_step_mean": float(np.mean(completion)) if completion else None,
            "new_red_win_terminal_step_median": float(np.median(completion)) if completion else None,
            "new_red_win_terminal_step_min": min(completion) if completion else None,
            "new_red_win_terminal_step_max": max(completion) if completion else None,
        })
    return results


def aggregate_checkpoint_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        current = [row for row in rows if int(row["horizon"]) == horizon]
        fields = (
            "red_win_rate", "blue_win_rate", "draw_rate", "MAV_survival_rate",
            "mean_red_attack_kills", "mean_red_uav_survivors", "mean_blue_survivors",
            "mean_cutoff_or_terminal_length",
        )
        aggregate: dict[str, Any] = {"horizon": horizon, "checkpoint_count": len(current)}
        for field in fields:
            values = [float(row[field]) for row in current]
            aggregate[f"seed_mean_{field}"] = float(np.mean(values))
            aggregate[f"seed_sample_sd_{field}"] = statistics.stdev(values) if len(values) > 1 else None
        # Rates pooled over equally protocolled episode rows, not training-seed uncertainty.
        total_episodes = sum(int(row["episodes"]) for row in current)
        aggregate.update({
            "pooled_episodes": total_episodes,
            "pooled_red_win_rate": sum(float(r["red_win_rate"]) * int(r["episodes"]) for r in current) / total_episodes,
            "pooled_blue_win_rate": sum(float(r["blue_win_rate"]) * int(r["episodes"]) for r in current) / total_episodes,
            "pooled_draw_rate": sum(float(r["draw_rate"]) * int(r["episodes"]) for r in current) / total_episodes,
            **{
                f"pooled_{field}": sum(float(r[field]) * int(r["episodes"]) for r in current) / total_episodes
                for field in (
                    "MAV_survival_rate", "mean_red_attack_kills", "mean_red_uav_survivors",
                    "mean_blue_survivors", "mean_cutoff_or_terminal_length",
                )
            },
        })
        results.append(aggregate)
    return results


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError(f"cannot write empty audit table: {path.name}")
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_outputs(output_dir: Path, episode_rows: list[dict[str, Any]],
                  checkpoint_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]],
                  protocol: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    conversions = [row for row in episode_rows if row["outcome_at_75"] == "draw"]
    _write_csv(output_dir / "episode_horizon_records.csv", episode_rows, EPISODE_FIELDS)
    _write_csv(output_dir / "checkpoint_horizon_summary.csv", checkpoint_rows)
    _write_csv(output_dir / "cross_checkpoint_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "draw75_conversions.csv", conversions, EPISODE_FIELDS)
    summary = {
        "audit": "continuation_horizon_audit_v1",
        "protocol": dict(protocol),
        "files": {
            "episode_records": "episode_horizon_records.csv",
            "checkpoint_summary": "checkpoint_horizon_summary.csv",
            "cross_checkpoint_aggregate": "cross_checkpoint_aggregate.csv",
            "draw75_conversions": "draw75_conversions.csv",
        },
        "checkpoint_horizon_summary": checkpoint_rows,
        "cross_checkpoint_aggregate": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="stochastic")
    parser.add_argument("--action-seed", type=int, default=4000)
    parser.add_argument("--env-seed-start", type=int, default=3000)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons must be positive")
    device = _resolved_device(args.device)
    loaded = [load_tacm_checkpoint(path, device) for path in args.checkpoints]
    labels = [_checkpoint_label(item["checkpoint"]) for item in loaded]
    if len(set(labels)) != len(labels):
        raise RuntimeError("checkpoint identifiers must be unique within one continuation audit")
    reference_config = loaded[0]["environment_config"]
    observation_horizon = loaded[0]["observation_horizon"]
    for item in loaded[1:]:
        if item["environment_config"] != reference_config:
            raise RuntimeError("paired continuation audit requires identical environment configs")
        if item["observation_horizon"] != observation_horizon:
            raise RuntimeError("paired continuation audit requires identical observation horizons")
    if observation_horizon not in horizons or max(horizons) < observation_horizon:
        raise ValueError("horizons must include the checkpoint observation horizon")
    episode_rows: list[dict[str, Any]] = []
    for item in loaded:
        label = _checkpoint_label(item["checkpoint"])
        for episode in range(args.episodes):
            episode_rows.extend(run_continuation_episode(
                item["actors"], item["environment_config"], profile=args.profile,
                episode=episode, env_seed=args.env_seed_start + episode,
                action_mode=args.action_mode,
                action_seed=(None if args.action_mode == "deterministic" else args.action_seed + episode),
                device=device, horizons=horizons,
                metadata={
                    "checkpoint": label, "checkpoint_path": str(item["checkpoint"]),
                    "method_variant": TACM_RGAA_METHOD,
                    "training_seed": item["training_seed"], "sampled_steps": item["sampled_steps"],
                    "action_mode": args.action_mode, "environment_profile": args.profile,
                    "environment_version": reference_config["environment_version"],
                    "observation_horizon": observation_horizon, "audit_max_horizon": max(horizons),
                },
            ))
    checkpoint_rows: list[dict[str, Any]] = []
    for label in dict.fromkeys(row["checkpoint"] for row in episode_rows):
        checkpoint_rows.extend(summarize_checkpoint_rows(
            [row for row in episode_rows if row["checkpoint"] == label],
        ))
    aggregate_rows = aggregate_checkpoint_summaries(checkpoint_rows)
    protocol = {
        "observation_horizon": observation_horizon, "audit_max_horizon": max(horizons),
        "horizons": list(horizons), "environment_profile": args.profile,
        "environment_seed_start": args.env_seed_start,
        "environment_seed_end": args.env_seed_start + args.episodes - 1,
        "action_mode": args.action_mode,
        "action_seed_start": None if args.action_mode == "deterministic" else args.action_seed,
        "action_seed_end": None if args.action_mode == "deterministic" else args.action_seed + args.episodes - 1,
        "environment_version": reference_config["environment_version"],
        "method_variant": TACM_RGAA_METHOD, "episodes_per_checkpoint": args.episodes,
        "checkpoints": [{"path": str(item["checkpoint"]), "training_seed": item["training_seed"],
                         "sampled_steps": item["sampled_steps"]} for item in loaded],
    }
    write_outputs(args.output_dir.expanduser().resolve(), episode_rows, checkpoint_rows, aggregate_rows, protocol)
    print("checkpoint | horizon | red | blue | draw | red_kills | UAV_survivors", flush=True)
    for row in checkpoint_rows:
        print(f"{row['checkpoint']} | {row['horizon']} | {row['red_win_rate']:.3f} | "
              f"{row['blue_win_rate']:.3f} | {row['draw_rate']:.3f} | "
              f"{row['mean_red_attack_kills']:.3f} | {row['mean_red_uav_survivors']:.3f}", flush=True)
    print(f"outputs: {args.output_dir.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
