"""Read-only boundary and execution-mode audit for RGAA-family policies.

This tool supports ``rgaa``, ``rgaa_wide`` and ``dbm_rgaa`` checkpoints.  It
performs a paired realized-initial-slot intervention and, for DBM checkpoints,
records RNG-free mode diagnostics at the exact execution-time observation.
The audit never updates a model or an environment configuration.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithm.happo.dbm_rgaa import (
    DBMGaussianActor, DBM_RGAA_METHOD, RGAA_WIDE_METHOD,
    build_method_actors, dbm_metadata, wide_metadata,
)
from algorithm.happo.evaluation import evaluate_actors, summarize_records
from algorithm.happo.networks import IndependentActors
from env.mavuav import OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from tools.audit_uav_slot_permutation import (
    SOURCE_SLOTS, UAV_IDS, build_actor_slot_matrix, build_marginals,
    is_identity_permutation, permutation_label, reset_with_realized_permutation,
    slot_permutations,
)


SUPPORTED_METHODS = frozenset(("rgaa", RGAA_WIDE_METHOD, DBM_RGAA_METHOD))
CAUSES = ("alive", "boundary", "blue_attack", "other")
BOUNDARY_AXES = (
    "x_lower", "x_upper", "y_lower", "y_upper",
    "altitude_lower", "altitude_upper",
)
VECTOR_KEYS = (
    "expert1_output", "expert2_output", "base_mean", "scaled_residual",
    "final_mean", "tanh_base_mean", "tanh_final_mean", "actual_sampled_action",
)


@dataclass
class LoadedAuditRun:
    label: str
    checkpoint: Path
    actors: Any
    env_config: dict[str, Any]
    trainer_config: dict[str, Any]
    method: str
    training_seed: int
    sampled_steps: int
    environment_version: str
    checkpoint_sha256: str
    initial_actor_state: dict[str, torch.Tensor]


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR_OR_CHECKPOINT")
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=RUN_DIR_OR_CHECKPOINT")
    return label.strip(), Path(raw).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_mapping, required=True)
    parser.add_argument("--profile", choices=("learnability",), default="learnability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--permutation-episodes", type=int, default=50)
    parser.add_argument("--identity-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("--run labels must be unique")
    if args.permutation_episodes <= 0 or args.identity_episodes <= 0:
        parser.error("episode counts must be positive")
    return args


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    candidate = path / "checkpoint_final.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"checkpoint not found: {path}")


def _clone_actor_state(actors: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in actors.state_dict().items()}


def assert_read_only(run: LoadedAuditRun) -> None:
    if file_sha256(run.checkpoint) != run.checkpoint_sha256:
        raise RuntimeError(f"read-only audit mutated checkpoint: {run.label}")
    state = run.actors.state_dict()
    if state.keys() != run.initial_actor_state.keys() or any(
        not torch.equal(state[name].detach().cpu(), expected)
        for name, expected in run.initial_actor_state.items()
    ):
        raise RuntimeError(f"read-only audit mutated actor parameters: {run.label}")


def _validate_metadata(payload: Mapping[str, Any], config: Mapping[str, Any], method: str) -> None:
    if str(payload.get("actor_variant", config.get("actor_variant", "vanilla"))) != "vanilla":
        raise RuntimeError("DBM boundary audit requires vanilla actor_variant")
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, Mapping):
        raise RuntimeError("checkpoint is missing actor_architecture metadata")
    # Legacy/plain RGAA architecture metadata records hidden/action dimensions;
    # the authoritative observation dimension is the checkpoint top-level field.
    if int(architecture.get("action_dim", -1)) != 3:
        raise RuntimeError("checkpoint actor architecture violates action contract")
    if method == DBM_RGAA_METHOD:
        expected = dbm_metadata(config)
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(f"incompatible DBM-RGAA checkpoint contract: {field}")
    elif method == RGAA_WIDE_METHOD:
        expected = wide_metadata(config)
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(f"incompatible RGAA-Wide checkpoint contract: {field}")
    else:
        for field in ("role_advantage_coef", "role_critic_architecture", "role_aux_reward_mode"):
            if field not in payload:
                raise RuntimeError(f"RGAA checkpoint is missing metadata: {field}")


def load_audit_run(label: str, path: Path, device: str) -> LoadedAuditRun:
    checkpoint = _checkpoint_path(path)
    digest = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(payload.get("trainer_config", payload.get("config", {})))
    method = str(payload.get("method_variant", config.get("method_variant", "")))
    if method not in SUPPORTED_METHODS:
        raise RuntimeError(f"unsupported method_variant for {label}: {method!r}")
    if int(payload.get("observation_dim", -1)) != OBS_DIM:
        raise RuntimeError(f"checkpoint observation_dim mismatch for {label}")
    if "environment_config" not in payload or "actors" not in payload:
        raise RuntimeError(f"checkpoint for {label} lacks resolved environment_config or actors")
    env_config = load_environment_config(payload["environment_config"])
    version = payload.get("environment_version")
    if version != env_config.get("environment_version"):
        raise RuntimeError(f"checkpoint/environment version mismatch for {label}")
    _validate_metadata(payload, config, method)
    if method in (DBM_RGAA_METHOD, RGAA_WIDE_METHOD):
        actors = build_method_actors(
            method_variant=method,
            training_seed=int(config["seed"]),
            hidden_dim=int(config["hidden_dim"]),
            log_std_init=float(config.get("actor_log_std_init", -0.5)),
            role_module_enabled=bool(config.get("role_module_enabled", True)),
            dbm_role_count=int(config.get("dbm_role_count", 2)),
            dbm_residual_scale=float(config.get("dbm_residual_scale", 0.25)),
            dbm_expert_init_scale=float(config.get("dbm_init_scale", 0.01)),
            uav_actor_hidden_dim=int(config.get("uav_actor_hidden_dim", 131)),
        ).to(device)
    else:
        actors = IndependentActors(
            hidden_dim=int(config["hidden_dim"]),
            log_std_init=float(config.get("actor_log_std_init", -0.5)),
        ).to(device)
    actors.load_state_dict(payload["actors"], strict=True)
    actors.eval()
    with torch.no_grad():
        probe = torch.zeros((1, OBS_DIM), device=device)
        for actor in actors.actors:
            if tuple(actor.log_std.shape) != (3,) or tuple(actor.network(probe).shape) != (1, 3):
                raise RuntimeError(f"checkpoint for {label} violates the 3D Gaussian action contract")
    return LoadedAuditRun(
        label=label, checkpoint=checkpoint, actors=actors, env_config=env_config,
        trainer_config=config, method=method, training_seed=int(config.get("seed", -1)),
        sampled_steps=int(payload.get("sampled_steps", 0)),
        environment_version=str(version), checkpoint_sha256=digest,
        initial_actor_state=_clone_actor_state(actors),
    )


def validate_run_contracts(runs: Sequence[LoadedAuditRun]) -> None:
    if not runs:
        raise RuntimeError("audit requires at least one run")
    expected = runs[0].env_config
    for run in runs:
        if run.environment_version != expected.get("environment_version") or run.env_config != expected:
            raise RuntimeError("DBM boundary audit requires identical resolved environment configs")


def ensure_empty_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"audit output directory must be missing or empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


def categorize_death(cause: str | None) -> str:
    if cause is None:
        return "alive"
    return cause if cause in ("boundary", "blue_attack") else "other"


def boundary_axes(state: Any, battlefield: Mapping[str, Sequence[float]]) -> list[str]:
    axes: list[str] = []
    if state.x < battlefield["x"][0]: axes.append("x_lower")
    if state.x > battlefield["x"][1]: axes.append("x_upper")
    if state.y < battlefield["y"][0]: axes.append("y_lower")
    if state.y > battlefield["y"][1]: axes.append("y_upper")
    if state.h < battlefield["altitude"][0]: axes.append("altitude_lower")
    if state.h > battlefield["altitude"][1]: axes.append("altitude_upper")
    return axes


def state_and_margins(env: HeterogeneousMAVUAVAirCombatEnv, aid: str) -> dict[str, float]:
    state = env.entities[aid].state
    bounds = env.config["battlefield"]
    margins = {
        "x_lower_margin": float(state.x - bounds["x"][0]),
        "x_upper_margin": float(bounds["x"][1] - state.x),
        "y_lower_margin": float(state.y - bounds["y"][0]),
        "y_upper_margin": float(bounds["y"][1] - state.y),
        "altitude_lower_margin": float(state.h - bounds["altitude"][0]),
        "altitude_upper_margin": float(bounds["altitude"][1] - state.h),
    }
    return {
        "position_x": float(state.x), "position_y": float(state.y),
        "altitude": float(state.h), "speed": float(state.v),
        "heading": float(state.psi), "pitch": float(state.theta),
        **margins, "nearest_boundary_margin": float(min(margins.values())),
    }


def _details_row(details: Mapping[str, torch.Tensor], action: torch.Tensor) -> dict[str, Any]:
    def array(key: str) -> np.ndarray:
        return details[key].squeeze(0).detach().cpu().numpy()
    probabilities = array("router_probabilities")
    experts = array("expert_outputs")
    base = array("base_mean"); residual = array("scaled_residual"); final = array("final_mean")
    result: dict[str, Any] = {
        "router_p1": float(probabilities[0]), "router_p2": float(probabilities[1]),
        "router_entropy": float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()),
        "hard_mode_proxy": int(np.argmax(probabilities) + 1),
        "expert_divergence": float(np.linalg.norm(experts[0] - experts[1])),
        "residual_norm": float(np.linalg.norm(residual)),
        "residual_to_base_ratio": float(np.linalg.norm(residual) / (np.linalg.norm(base) + 1e-8)),
        "deterministic_action_delta_norm": float(np.linalg.norm(np.tanh(final) - np.tanh(base))),
    }
    vectors = {
        "expert1_output": experts[0], "expert2_output": experts[1],
        "base_mean": base, "scaled_residual": residual, "final_mean": final,
        "tanh_base_mean": np.tanh(base), "tanh_final_mean": np.tanh(final),
        "actual_sampled_action": action.squeeze(0).detach().cpu().numpy(),
    }
    for key, values in vectors.items():
        for dimension, value in zip(("x", "y", "z"), values):
            result[f"{key}_{dimension}"] = float(value)
    return result


def rollout_episode(
    run: LoadedAuditRun,
    mapping: Mapping[str, str],
    *,
    episode: int,
    env_seed: int,
    action_seed: int,
    profile: str,
    device: str,
    diagnostics: bool = False,
    capture_actions: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    env = HeterogeneousMAVUAVAirCombatEnv(run.env_config, profile=profile)
    observations, _ = reset_with_realized_permutation(env, env_seed, mapping)
    torch.manual_seed(int(action_seed))
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(action_seed))
    death: dict[str, dict[str, Any]] = {
        aid: {"cause": "alive", "step": None, "axes": []} for aid in RED_IDS
    }
    router_rows: list[dict[str, Any]] = []
    execution_trace: list[dict[str, Any]] = []
    done = False
    while not done:
        decision_step = int(env.step_count)
        actions: list[np.ndarray] = []
        step_actions: list[list[float]] = []
        with torch.no_grad():
            # Formal evaluator contract: sample all four actors, in ID order,
            # including inactive agents.  env.step alone masks inactive actions.
            for index, aid in enumerate(RED_IDS):
                observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
                actor = run.actors.actors[index]
                details = None
                active = bool(env.entities[aid].state.alive)
                if diagnostics and active and aid in UAV_IDS and isinstance(actor, DBMGaussianActor):
                    details = actor.mode_diagnostics(observation)
                action, _ = actor.sample(observation, deterministic=False)
                action_np = action.squeeze(0).cpu().numpy()
                actions.append(action_np); step_actions.append(action_np.tolist())
                if details is not None:
                    router_rows.append({
                        "run": run.label, "method": run.method,
                        "training_seed": run.training_seed, "sampled_steps": run.sampled_steps,
                        "episode": episode, "decision_step": decision_step,
                        "agent_id": aid, "agent_active": 1,
                        "environment_seed": env_seed, "action_seed": action_seed,
                        **_details_row(details, action), **state_and_margins(env, aid),
                    })
        observations, rewards, terminated, truncated, info = env.step(np.asarray(actions))
        if capture_actions:
            # A CPU-only equality signature for both the full action sequence
            # and every externally observable env.step result.
            execution_trace.append({
                "actions": step_actions,
                "observations": {aid: observations[aid].tolist() for aid in RED_IDS},
                "rewards": {aid: float(rewards[aid]) for aid in RED_IDS},
                "terminated": bool(terminated), "truncated": bool(truncated),
                "active_masks": np.asarray(info["active_masks"]).tolist(),
                "outcome": info.get("outcome"),
                "death_causes": dict(info.get("death_causes", {})),
                "episode_summary": dict(info["episode_summary"]) if "episode_summary" in info else None,
            })
        for aid, raw_cause in info.get("death_causes", {}).items():
            if aid not in death or death[aid]["cause"] != "alive":
                continue
            cause = categorize_death(str(raw_cause))
            death[aid] = {
                "cause": cause, "step": int(env.step_count),
                "axes": boundary_axes(env.entities[aid].state, env.config["battlefield"])
                if cause == "boundary" else [],
            }
        done = bool(terminated or truncated)
    summary = dict(info["episode_summary"])
    for row in router_rows:
        item = death[row["agent_id"]]
        row["final_death_cause"] = item["cause"]
        row["death_step"] = item["step"]
    return summary, death, router_rows, execution_trace


def episode_rows(
    run: LoadedAuditRun, mapping: Mapping[str, str], episode: int,
    env_seed: int, action_seed: int, profile: str, device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, death, _, _ = rollout_episode(
        run, mapping, episode=episode, env_seed=env_seed, action_seed=action_seed,
        profile=profile, device=device,
    )
    return episode_output_rows(run, mapping, episode, env_seed, action_seed, summary, death)


def episode_output_rows(
    run: LoadedAuditRun, mapping: Mapping[str, str], episode: int,
    env_seed: int, action_seed: int, summary: Mapping[str, Any],
    death: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serialize exact rollout summary/death accounting without another rollout."""
    mapping_fields = {f"{aid}_source_slot": mapping[aid] for aid in UAV_IDS}
    base = {
        "run": run.label, "method": run.method, "training_seed": run.training_seed,
        "sampled_steps": run.sampled_steps, "permutation": permutation_label(mapping),
        **mapping_fields, "episode": episode, "environment_seed": env_seed,
        "action_seed": action_seed, "episode_return": float(summary["episode_return"]),
        "red_attack_kills": int(summary["red_attack_kills"]),
        "blue_attack_kills": int(summary["blue_attack_kills"]),
        "episode_length": int(summary["episode_length"]),
        "mav_survived": bool(summary["mav_survived"]),
        "uav_survivors": int(summary["red_uav_survivors"]),
    }
    team = {**base, "outcome": summary["outcome"]}
    agents = []
    for aid in UAV_IDS:
        item = death[aid]; cause = item["cause"]
        agents.append({
            **base, "actor_id": aid, "source_slot": mapping[aid],
            "final_death_cause": cause, "death_step": item["step"],
            "boundary_crossing_axes": "|".join(item["axes"]),
            **{axis: int(axis in item["axes"]) for axis in BOUNDARY_AXES},
            **{name: int(cause == name) for name in CAUSES},
            "team_outcome": summary["outcome"],
        })
    return team, agents


def summarize_identity_deaths(
    team_rows: Sequence[Mapping[str, Any]], agent_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate exact identity-rollout UAV death causes and boundary axes."""
    actor_output: list[dict[str, Any]] = []
    for run, aid in sorted({(str(row["run"]), str(row["actor_id"])) for row in agent_rows}):
        selected = [row for row in agent_rows if row["run"] == run and row["actor_id"] == aid]
        episodes = len(selected)
        death_steps = [float(row["death_step"]) for row in selected if row.get("death_step") not in (None, "")]
        record: dict[str, Any] = {
            "run": run, "actor_id": aid, "episodes": episodes,
        }
        for cause in CAUSES:
            count = sum(int(row[cause]) for row in selected)
            record[f"{cause}_count"] = count
            record[f"{cause}_rate"] = count / episodes if episodes else 0.0
        for axis in BOUNDARY_AXES:
            record[f"{axis}_count"] = sum(int(row[axis]) for row in selected)
        record["mean_death_step"] = float(np.mean(death_steps)) if death_steps else None
        record["median_death_step"] = float(np.median(death_steps)) if death_steps else None
        actor_output.append(record)

    run_output: list[dict[str, Any]] = []
    for run in sorted({str(row["run"]) for row in team_rows}):
        teams = [row for row in team_rows if row["run"] == run]
        agents = [row for row in agent_rows if row["run"] == run]
        episodes = len(teams); exposures = episodes * len(UAV_IDS)
        alive = sum(int(row["alive"]) for row in agents)
        boundary = sum(int(row["boundary"]) for row in agents)
        blue_attack = sum(int(row["blue_attack"]) for row in agents)
        other = sum(int(row["other"]) for row in agents)
        if alive + boundary + blue_attack + other != exposures:
            raise AssertionError(f"identity UAV death accounting is not conserved for {run}")
        altitude_lower = sum(int(row["altitude_lower"]) for row in agents)
        losses = exposures - alive
        run_output.append({
            "run": run, "episodes": episodes, "total_uav_exposures": exposures,
            "uav_alive_count": alive, "uav_boundary_count": boundary,
            "uav_blue_attack_count": blue_attack, "uav_other_death_count": other,
            "altitude_lower_count": altitude_lower,
            "boundary_share_of_uav_losses": boundary / losses if losses else 0.0,
            "altitude_lower_share_of_uav_losses": altitude_lower / losses if losses else 0.0,
            "altitude_lower_share_of_boundary_losses": altitude_lower / boundary if boundary else 0.0,
            "mean_uav_survivors": float(np.mean([float(row["uav_survivors"]) for row in teams])) if teams else 0.0,
        })
    return actor_output, run_output


def _formal_rows(team_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "outcome": row["outcome"], "episode_return": row["episode_return"],
        "mav_survived": row["mav_survived"],
        "red_uav_survivors": row["uav_survivors"],
        "red_attack_kills": row["red_attack_kills"],
        "blue_attack_kills": row["blue_attack_kills"],
        "episode_length": row["episode_length"],
    } for row in team_rows]


def summarize_permutations(team_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run, permutation in sorted({(r["run"], r["permutation"]) for r in team_rows}):
        rows = [r for r in team_rows if r["run"] == run and r["permutation"] == permutation]
        stats = summarize_records(_formal_rows(rows))
        output.append({
            "run": run, "method": rows[0]["method"], "training_seed": rows[0]["training_seed"],
            "sampled_steps": rows[0]["sampled_steps"], "permutation": permutation,
            "identity": int(permutation == permutation_label(dict(zip(UAV_IDS, SOURCE_SLOTS)))),
            "episodes": len(rows), **stats,
        })
    return output


def _descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "variance": None, "min": None, "max": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "std": float(array.std(ddof=0)), "variance": float(array.var(ddof=0)),
        "min": float(array.min()), "max": float(array.max()),
    }


def add_death_step_statistics(
    summary_rows: Sequence[Mapping[str, Any]], agent_rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Retain legacy death rates and append observed loss-step statistics."""
    output: list[dict[str, Any]] = []
    for summary in summary_rows:
        matching = [
            row for row in agent_rows
            if all(row[field] == summary[field] for field in group_fields)
        ]
        death_steps = [float(row["death_step"]) for row in matching if row.get("death_step") not in (None, "")]
        output.append({
            **summary,
            "mean_death_step": float(np.mean(death_steps)) if death_steps else None,
            "observed_death_step_count": len(death_steps),
        })
    return output


def execution_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run, aid in sorted({(r["run"], r["agent_id"]) for r in rows}):
        selected = [r for r in rows if r["run"] == run and r["agent_id"] == aid]
        by_episode: dict[int, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_episode.setdefault(int(row["episode"]), []).append(row)
        movements: list[float] = []; switches = 0; pairs = 0
        for trajectory in by_episode.values():
            trajectory.sort(key=lambda row: int(row["decision_step"]))
            for left, right in zip(trajectory, trajectory[1:]):
                if int(right["decision_step"]) != int(left["decision_step"]) + 1:
                    continue
                movements.append(abs(float(right["router_p1"]) - float(left["router_p1"])) +
                                 abs(float(right["router_p2"]) - float(left["router_p2"])))
                switches += int(right["hard_mode_proxy"] != left["hard_mode_proxy"]); pairs += 1
        row = {
            "run": run, "agent_id": aid, "active_steps": len(selected),
            "valid_temporal_pairs": pairs,
            "hard_mode_proxy_switch_rate": switches / pairs if pairs else 0.0,
            "router_l1_movement_mean": float(np.mean(movements)) if movements else 0.0,
        }
        for field in (
            "router_p1", "router_entropy", "expert_divergence", "residual_norm",
            "deterministic_action_delta_norm", "nearest_boundary_margin", "speed", "heading", "pitch",
        ):
            stats = _descriptive(float(item[field]) for item in selected)
            row.update({f"{field}_{key}": value for key, value in stats.items()})
        output.append(row)
    return output


def pre_boundary_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["run"]), int(row["episode"]), str(row["agent_id"])), []).append(row)
    trajectory_windows: list[dict[str, Any]] = []
    fields = (
        "router_p1", "router_entropy", "residual_norm", "deterministic_action_delta_norm",
        "actual_sampled_action_x", "actual_sampled_action_y", "actual_sampled_action_z",
        "nearest_boundary_margin", "speed", "heading", "pitch",
    )
    for window in (5, 10):
        for (run, episode, aid), trajectory in sorted(groups.items()):
            cause = str(trajectory[0]["final_death_cause"])
            if cause not in CAUSES:
                cause = "other"
            trajectory.sort(key=lambda row: int(row["decision_step"]))
            chosen = trajectory[-window:]
            row: dict[str, Any] = {
                "run": run, "episode": episode, "agent_id": aid,
                "final_death_cause": cause, "requested_window": window,
                "available_steps": len(chosen),
            }
            for field in fields:
                stats = _descriptive(float(item[field]) for item in chosen)
                row.update({f"{field}_{key}": value for key, value in stats.items()})
                if len(chosen) > 1:
                    row[f"{field}_change"] = float(chosen[-1][field]) - float(chosen[0][field])
                else:
                    row[f"{field}_change"] = 0.0
            trajectory_windows.append(row)
    output: list[dict[str, Any]] = []
    group_keys = sorted({
        (row["run"], row["agent_id"], row["final_death_cause"], row["requested_window"])
        for row in trajectory_windows
    })
    for run, aid, cause, window in group_keys:
        selected = [row for row in trajectory_windows if (
            row["run"], row["agent_id"], row["final_death_cause"], row["requested_window"]
        ) == (run, aid, cause, window)]
        result: dict[str, Any] = {
            "run": run, "agent_id": aid, "final_death_cause": cause,
            "requested_window": window, "trajectory_count": len(selected),
            "available_step_count": sum(int(row["available_steps"]) for row in selected),
            "mean_available_steps": float(np.mean([row["available_steps"] for row in selected])),
        }
        for field in fields:
            for statistic in ("mean", "change"):
                values = [float(row[f"{field}_{statistic}"]) for row in selected]
                stats = _descriptive(values)
                result.update({
                    f"{field}_{statistic}_{key}": value for key, value in stats.items()
                })
        output.append(result)
    return output


def verify_identity_against_evaluator(
    run: LoadedAuditRun, episodes: int, profile: str, env_seed: int,
    action_seed: int, device: str, diagnostics_writer: csv.DictWriter | None = None,
) -> tuple[
    dict[str, float], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    identity = dict(zip(UAV_IDS, SOURCE_SLOTS))
    audited_records: list[dict[str, Any]] = []
    all_router_rows: list[dict[str, Any]] = []
    identity_team_rows: list[dict[str, Any]] = []
    identity_agent_rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        kwargs = dict(
            episode=episode, env_seed=env_seed + episode, action_seed=action_seed + episode,
            profile=profile, device=device,
        )
        summary, death, router_rows, actions_on = rollout_episode(
            run, identity, diagnostics=run.method == DBM_RGAA_METHOD,
            capture_actions=True, **kwargs,
        )
        if run.method == DBM_RGAA_METHOD:
            summary_off, _, _, actions_off = rollout_episode(
                run, identity, diagnostics=False, capture_actions=True, **kwargs,
            )
            if actions_on != actions_off or summary != summary_off:
                raise AssertionError("DBM diagnostics changed actions or episode result")
        audited_records.append(summary)
        team_row, agent_rows = episode_output_rows(
            run, identity, episode, int(kwargs["env_seed"]), int(kwargs["action_seed"]),
            summary, death,
        )
        identity_team_rows.append(team_row); identity_agent_rows.extend(agent_rows)
        all_router_rows.extend(router_rows)
        if diagnostics_writer is not None:
            diagnostics_writer.writerows(router_rows)
    formal = evaluate_actors(
        run.actors, run.env_config, episodes, profile, seed=env_seed, device=device,
        deterministic=False, action_seed=action_seed,
    )
    fields = (
        "red_win_rate", "mean_episode_return", "mean_red_attack_kills",
        "mean_UAV_survivors", "MAV_survival_rate", "mean_episode_length",
    )
    audited_stats = summarize_records(audited_records); formal_stats = summarize_records(formal)
    for field in fields:
        if not np.isclose(audited_stats[field], formal_stats[field], rtol=0.0, atol=0.0):
            raise AssertionError(f"identity/evaluator mismatch for {run.label}: {field}")
    return audited_stats, audited_records, all_router_rows, identity_team_rows, identity_agent_rows


def _csv_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = _csv_fields(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _router_fieldnames() -> list[str]:
    fields = [
        "run", "method", "training_seed", "sampled_steps", "episode", "decision_step",
        "agent_id", "agent_active", "environment_seed", "action_seed", "router_p1",
        "router_p2", "router_entropy", "hard_mode_proxy", "expert_divergence",
        "residual_norm", "residual_to_base_ratio", "deterministic_action_delta_norm",
    ]
    fields.extend(f"{key}_{axis}" for key in VECTOR_KEYS for axis in ("x", "y", "z"))
    fields.extend((
        "position_x", "position_y", "altitude", "speed", "heading", "pitch",
        "x_lower_margin", "x_upper_margin", "y_lower_margin", "y_upper_margin",
        "altitude_lower_margin", "altitude_upper_margin", "nearest_boundary_margin",
        "final_death_cause", "death_step",
    ))
    return fields


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = str(args.device)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = ensure_empty_output(args.output)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    runs: list[LoadedAuditRun] = []
    try:
        runs = [load_audit_run(label, path, device) for label, path in args.run]
        validate_run_contracts(runs)
        for run in runs:
            if str(run.trainer_config.get("environment_profile")) != str(args.profile):
                raise RuntimeError(
                    f"audit profile mismatch for {run.label}: checkpoint="
                    f"{run.trainer_config.get('environment_profile')!r}, audit={args.profile!r}"
                )
        team_rows: list[dict[str, Any]] = []; agent_rows: list[dict[str, Any]] = []
        for run in runs:
            for mapping in slot_permutations():
                for episode in range(int(args.permutation_episodes)):
                    team, agents = episode_rows(
                        run, mapping, episode, int(args.env_seed) + episode,
                        int(args.action_seed) + episode, args.profile, device,
                    )
                    team_rows.append(team); agent_rows.extend(agents)
        permutation_summary = summarize_permutations(team_rows)
        actor_slot = add_death_step_statistics(
            build_actor_slot_matrix(agent_rows, int(args.permutation_episodes)),
            agent_rows, ("run", "actor_id", "source_slot"),
        )
        actor_marginals = add_death_step_statistics(
            build_marginals(agent_rows, "actor_id", UAV_IDS),
            agent_rows, ("run", "actor_id"),
        )
        slot_marginals = add_death_step_statistics(
            build_marginals(agent_rows, "source_slot", SOURCE_SLOTS),
            agent_rows, ("run", "source_slot"),
        )
        write_csv(output / "permutation_episode_team.csv", team_rows)
        write_csv(output / "permutation_episode_agent.csv", agent_rows)
        write_csv(output / "permutation_summary.csv", permutation_summary)
        write_csv(output / "actor_slot_boundary_matrix.csv", actor_slot)
        write_csv(output / "actor_boundary_marginals.csv", actor_marginals)
        write_csv(output / "slot_boundary_marginals.csv", slot_marginals)

        identity_stats: dict[str, Any] = {}; router_rows: list[dict[str, Any]] = []
        identity_team_rows: list[dict[str, Any]] = []
        identity_agent_rows: list[dict[str, Any]] = []
        router_path = output / "execution_router_steps.csv"
        with router_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_router_fieldnames())
            writer.writeheader()
            for run in runs:
                stats, _, rows, team_identity, agent_identity = verify_identity_against_evaluator(
                    run, int(args.identity_episodes), args.profile, int(args.env_seed),
                    int(args.action_seed), device, writer,
                )
                identity_stats[run.label] = stats
                router_rows.extend(rows)
                identity_team_rows.extend(team_identity)
                identity_agent_rows.extend(agent_identity)
        identity_death_summary, identity_run_death_summary = summarize_identity_deaths(
            identity_team_rows, identity_agent_rows,
        )
        write_csv(output / "identity_episode_team.csv", identity_team_rows)
        write_csv(output / "identity_episode_agent.csv", identity_agent_rows)
        write_csv(output / "identity_death_summary.csv", identity_death_summary)
        write_csv(output / "identity_run_death_summary.csv", identity_run_death_summary)
        execution_summary = execution_summaries(router_rows)
        pre_boundary = pre_boundary_summaries(router_rows)
        write_csv(output / "execution_router_summary.csv", execution_summary)
        write_csv(output / "pre_boundary_window_summary.csv", pre_boundary)

        metadata = {
            "audit": "dbm_rgaa_boundary_modes", "read_only": True,
            "profile": args.profile, "action_mode": "stochastic",
            "permutation_episodes": int(args.permutation_episodes),
            "identity_episodes": int(args.identity_episodes),
            "environment_seed_start": int(args.env_seed),
            "permutation_environment_seed_end": int(args.env_seed) + int(args.permutation_episodes) - 1,
            "identity_environment_seed_end": int(args.env_seed) + int(args.identity_episodes) - 1,
            "action_seed_start": int(args.action_seed),
            "permutation_action_seed_end": int(args.action_seed) + int(args.permutation_episodes) - 1,
            "identity_action_seed_end": int(args.action_seed) + int(args.identity_episodes) - 1,
            "environment_version": runs[0].environment_version,
            "runs": {run.label: {
                "checkpoint": str(run.checkpoint), "checkpoint_sha256": run.checkpoint_sha256,
                "method": run.method, "training_seed": run.training_seed,
                "sampled_steps": run.sampled_steps,
            } for run in runs},
            "statistics": {
                "permutation": "six paired reset-realized UAV AircraftState permutations",
                "router": "active UAV execution steps only; soft router probabilities",
                "switch": "adjacent active steps within the same episode only",
                "boundary_axis": "all strict post-step battlefield violations at boundary death",
            },
        }
        summary = {
            **metadata, "identity_evaluator_consistency": identity_stats,
            "identity_death_summary": identity_death_summary,
            "identity_run_death_summary": identity_run_death_summary,
            "diagnostics_action_and_result_invariant": True,
            "checkpoint_and_actor_immutability": True,
        }
        (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "README.txt").write_text(
            "DBM-RGAA boundary-mode audit (read only)\n"
            "========================================\n"
            "Initial-slot permutation is a descriptive intervention; actor/slot associations do not "
            "automatically establish causality. Router/boundary associations likewise do not prove "
            "that a router mode causes boundary loss. execution_router_steps.csv contains only active "
            "DBM UAV decisions and records soft router probabilities before the one and only policy "
            "sample for that decision. Identity sample size and permutation sample size are distinct.\n",
            encoding="utf-8",
        )
        for run in runs:
            assert_read_only(run)
        return summary
    finally:
        for run in runs:
            assert_read_only(run)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_rng])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runs": list(summary["runs"]),
        "diagnostics_invariant": summary["diagnostics_action_and_result_invariant"],
    }, indent=2))


if __name__ == "__main__":
    main()
