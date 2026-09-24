"""Read-only audit of completed RGAA and CR-RGAA HAPPO runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithm.happo.evaluation import summarize_records
from algorithm.happo.networks import IndependentActors
from env.mavuav import (
    GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv,
    load_environment_config,
)


SUPPORTED_METHODS = frozenset(("rgaa", "cr_rgaa"))
CAUSES = ("alive", "boundary", "blue_attack", "other")
PHASES = (
    ("phase_0_750k", 0, 750_000),
    ("phase_750k_1250k", 750_000, 1_250_000),
    ("phase_1250k_1600k", 1_250_000, 1_600_000),
    ("phase_1600k_2000k", 1_600_000, 2_000_000),
    ("late_1500k_2000k", 1_500_000, 2_000_000),
)
CR_CONTINUOUS_FIELDS = (
    "cr_lambda_mean", "cr_conflict_rate",
    *(f"cr_lambda_mean_{aid}" for aid in RED_IDS),
    *(f"cr_conflict_rate_{aid}" for aid in RED_IDS),
    *(f"cr_lambda_on_conflict_mean_{aid}" for aid in RED_IDS),
    *(f"cr_lambda_on_agreement_mean_{aid}" for aid in RED_IDS),
    *(f"cr_relational_residual_abs_mean{suffix}" for suffix in ("", "_MAV", "_UAV")),
    "cr_attention_entropy", "cr_attention_self_mass",
    "cr_attention_mav_to_uav_mass", "cr_attention_uav_to_mav_mass",
    "cr_attention_uav_to_other_uav_mass", "cr_role_critic_total_loss",
    "mav_role_critic_loss", "uav_role_critic_loss",
)
EVENT_PREFIXES = ("own_loss_count", "own_boundary_loss_count", "own_blue_attack_loss_count")
EPISODE_FIELDS = (
    "episode", "environment_seed", "action_seed", "outcome", "episode_length",
    "episode_return", "red_attack_kills", "blue_attack_kills", "mav_survived",
    "red_uav_survivors", *(f"{aid}_death_cause" for aid in RED_IDS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--profile", choices=("learnability", "main"), default="learnability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="stochastic")
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolved_device(requested: str) -> str:
    return "cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested


def validate_checkpoint_contract(payload: Mapping[str, Any], env_config: Mapping[str, Any]) -> dict[str, Any]:
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    expected = (env_config["environment_version"], OBS_DIM, GLOBAL_STATE_DIM)
    actual = (
        payload.get("environment_version"), payload.get("observation_dim"),
        payload.get("global_state_dim"),
    )
    if actual != expected:
        raise RuntimeError(f"incompatible checkpoint environment contract: {actual!r} != {expected!r}")
    actor_variant = payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla"))
    critic_variant = payload.get("critic_variant", trainer_config.get("critic_variant", "mlp"))
    method_variant = payload.get("method_variant", trainer_config.get("method_variant"))
    if actor_variant != "vanilla":
        raise RuntimeError("role-guided audit requires actor_variant='vanilla'")
    if critic_variant != "mlp":
        raise RuntimeError("role-guided audit requires critic_variant='mlp'")
    if method_variant not in SUPPORTED_METHODS:
        raise RuntimeError(f"unsupported role-guided method_variant: {method_variant!r}")
    if "hidden_dim" not in trainer_config:
        raise RuntimeError("checkpoint trainer config is missing hidden_dim")
    return {
        "method_variant": method_variant,
        "actor_variant": actor_variant,
        "critic_variant": critic_variant,
        "hidden_dim": int(trainer_config["hidden_dim"]),
    }


def load_actor_only_checkpoint(
    checkpoint: Path, device: str,
) -> tuple[IndependentActors, dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "environment_config" not in payload:
        raise RuntimeError("checkpoint is missing resolved environment_config")
    env_config = load_environment_config(payload["environment_config"])
    contract = validate_checkpoint_contract(payload, env_config)
    actors = IndependentActors(hidden_dim=contract["hidden_dim"]).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    return actors, payload, env_config, contract


def _categorize_cause(cause: str) -> str:
    return cause if cause in ("boundary", "blue_attack") else "other"


def validate_death_accounting(record: Mapping[str, Any]) -> None:
    final_alive = int(bool(record["mav_survived"])) + int(record["red_uav_survivors"])
    recorded_deaths = sum(record[f"{aid}_death_cause"] != "alive" for aid in RED_IDS)
    if final_alive != int(bool(record["mav_survived"])) + int(record["red_uav_survivors"]):
        raise AssertionError("Red final alive accounting is inconsistent")
    if recorded_deaths != len(RED_IDS) - final_alive:
        raise AssertionError(
            f"Red death accounting mismatch: causes={recorded_deaths}, expected={len(RED_IDS)-final_alive}"
        )


def audit_episodes(
    actors: Any,
    env_config: str | Path | Mapping[str, Any],
    episodes: int,
    profile: str,
    *,
    env_seed: int = 1000,
    device: str = "cpu",
    deterministic: bool = False,
    action_seed: int | None = 2000,
) -> list[dict[str, Any]]:
    """Run the formal actor sequence while additionally recording Red deaths."""
    records: list[dict[str, Any]] = []
    env = HeterogeneousMAVUAVAirCombatEnv(env_config, profile=profile)
    for episode in range(int(episodes)):
        episode_env_seed = int(env_seed) + episode
        observations, _ = env.reset(seed=episode_env_seed)
        episode_action_seed: int | None = None
        if not deterministic and action_seed is not None:
            episode_action_seed = int(action_seed) + episode
            torch.manual_seed(episode_action_seed)
            if torch.device(device).type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_action_seed)
        death_causes: dict[str, str] = {}
        done = False
        while not done:
            actions = []
            with torch.no_grad():
                # This order intentionally matches evaluate_actors exactly.
                for index, aid in enumerate(env.red_ids):
                    actor_observation = torch.as_tensor(
                        observations[aid], device=device,
                    ).unsqueeze(0)
                    action, _ = actors.actors[index].sample(
                        actor_observation, deterministic=deterministic,
                    )
                    actions.append(action.squeeze(0).cpu().numpy())
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            for aid, cause in info.get("death_causes", {}).items():
                if aid in RED_IDS:
                    death_causes[aid] = _categorize_cause(str(cause))
            done = bool(terminated or truncated)
        summary = info["episode_summary"]
        row = {
            "episode": episode,
            "environment_seed": episode_env_seed,
            "action_seed": episode_action_seed,
            **{key: summary[key] for key in (
                "outcome", "episode_length", "episode_return", "red_attack_kills",
                "blue_attack_kills", "mav_survived", "red_uav_survivors",
            )},
            **{f"{aid}_death_cause": death_causes.get(aid, "alive") for aid in RED_IDS},
        }
        validate_death_accounting(row)
        records.append(row)
    return records


def death_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    formal = summarize_records([dict(record) for record in records])
    summary: dict[str, Any] = {
        "episodes": len(records),
        "win_rate": formal.get("red_win_rate", 0.0),
        "blue_win_rate": formal.get("blue_win_rate", 0.0),
        "draw_rate": formal.get("draw_rate", 0.0),
        "mean_return": formal.get("mean_episode_return", 0.0),
        "mean_red_attack_kills": formal.get("mean_red_attack_kills", 0.0),
        "mean_blue_attack_kills": formal.get("mean_blue_attack_kills", 0.0),
        "MAV_survival_rate": formal.get("MAV_survival_rate", 0.0),
        "mean_UAV_survivors": formal.get("mean_UAV_survivors", 0.0),
        "mean_episode_length": formal.get("mean_episode_length", 0.0),
        "by_agent": {},
    }
    for aid in RED_IDS:
        counts = {cause: 0 for cause in CAUSES}
        for record in records:
            counts[str(record[f"{aid}_death_cause"])] += 1
        summary["by_agent"][aid] = counts
    uav = {cause: sum(summary["by_agent"][aid][cause] for aid in RED_IDS[1:]) for cause in CAUSES}
    uav_deaths = uav["boundary"] + uav["blue_attack"] + uav["other"]
    n = max(len(records), 1)
    summary.update({
        "uav_alive_total": uav["alive"],
        "uav_boundary_total": uav["boundary"],
        "uav_blue_attack_total": uav["blue_attack"],
        "uav_other_total": uav["other"],
        "mean_uav_boundary_losses_per_episode": uav["boundary"] / n,
        "mean_uav_blue_attack_losses_per_episode": uav["blue_attack"] / n,
        "uav_boundary_share_of_uav_deaths": uav["boundary"] / uav_deaths if uav_deaths else 0.0,
        "uav_blue_attack_share_of_uav_deaths": uav["blue_attack"] / uav_deaths if uav_deaths else 0.0,
        "mav_boundary_rate": summary["by_agent"]["MAV"]["boundary"] / n,
        "mav_blue_attack_rate": summary["by_agent"]["MAV"]["blue_attack"] / n,
    })
    return summary


def _finite_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _descriptive(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "min": float(array.min()), "max": float(array.max()),
    }


def _phase_rows(rows: Sequence[Mapping[str, Any]], lower: int, upper: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if lower < int(float(row["sampled_steps"])) <= upper]


def _event_value(row: Mapping[str, Any], prefix: str, aid: str) -> float:
    return _finite_number(row.get(f"{prefix}_{aid}")) or 0.0


def analyze_training_rows(
    raw_rows: Sequence[Mapping[str, Any]], method_variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = sorted((dict(row) for row in raw_rows), key=lambda row: int(float(row["sampled_steps"])))
    previous_completed = 0
    for row in rows:
        completed = int(float(row.get("completed_episodes", 0) or 0))
        delta = completed - previous_completed
        if delta < 0:
            raise ValueError("completed_episodes must be cumulative and non-decreasing")
        row["_episodes_in_update"] = delta
        previous_completed = completed
    phase_json: dict[str, Any] = {}
    phase_csv: list[dict[str, Any]] = []
    for phase_name, lower, upper in PHASES:
        selected = _phase_rows(rows, lower, upper)
        episodes = sum(int(row["_episodes_in_update"]) for row in selected)
        continuous: dict[str, Any] = {}
        flat: dict[str, Any] = {"phase": phase_name, "lower_exclusive": lower, "upper_inclusive": upper,
                                "updates": len(selected), "episodes": episodes}
        for field in CR_CONTINUOUS_FIELDS:
            values = [value for row in selected if (value := _finite_number(row.get(field))) is not None]
            stats = _descriptive(values) if method_variant == "cr_rgaa" else None
            continuous[field] = stats
            for statistic in ("mean", "std", "min", "max"):
                flat[f"{field}_{statistic}"] = None if stats is None else stats[statistic]

        events: dict[str, Any] = {}
        for aid in RED_IDS:
            own = sum(_event_value(row, "own_loss_count", aid) for row in selected)
            boundary = sum(_event_value(row, "own_boundary_loss_count", aid) for row in selected)
            blue = sum(_event_value(row, "own_blue_attack_loss_count", aid) for row in selected)
            item = {
                "own_loss_events": own, "boundary_events": boundary,
                "blue_attack_events": blue, "other_or_mismatch_events": own - boundary - blue,
                "own_loss_events_per_1000_episodes": own * 1000.0 / episodes if episodes else 0.0,
                "boundary_events_per_1000_episodes": boundary * 1000.0 / episodes if episodes else 0.0,
                "blue_attack_events_per_1000_episodes": blue * 1000.0 / episodes if episodes else 0.0,
            }
            events[aid] = item
            for key, value in item.items():
                flat[f"{aid}_{key}"] = value
        uav_events = {
            key: sum(events[aid][key] for aid in RED_IDS[1:])
            for key in events["UAV1"]
        }
        events["UAV_aggregate"] = uav_events
        for key, value in uav_events.items():
            flat[f"UAV_aggregate_{key}"] = value
        phase_json[phase_name] = {
            "range": {"lower_exclusive": lower, "upper_inclusive": upper},
            "updates": len(selected), "episodes": episodes,
            "mechanism_metrics": continuous, "death_events": events,
        }
        phase_csv.append(flat)
    return phase_csv, phase_json


def pearson_or_none(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 20 or len(y) != len(x):
        return None
    xa = np.asarray(x, dtype=np.float64); ya = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[valid]; ya = ya[valid]
    if len(xa) < 20 or np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def exploratory_correlations(
    raw_rows: Sequence[Mapping[str, Any]], method_variant: str,
) -> dict[str, float | None]:
    names = {
        "mav_residual_vs_mav_boundary": ("cr_relational_residual_abs_mean_MAV", ("MAV",)),
        "uav_residual_vs_uav_boundary": ("cr_relational_residual_abs_mean_UAV", RED_IDS[1:]),
        "conflict_rate_vs_uav_boundary": ("cr_conflict_rate", RED_IDS[1:]),
        "uav_to_mav_attention_vs_uav_boundary": ("cr_attention_uav_to_mav_mass", RED_IDS[1:]),
    }
    result: dict[str, float | None] = {name: None for name in names}
    if method_variant != "cr_rgaa":
        return result
    rows = [row for row in raw_rows if 750_000 < int(float(row["sampled_steps"])) <= 2_000_000]
    for name, (metric, agents) in names.items():
        x: list[float] = []; y: list[float] = []
        for row in rows:
            value = _finite_number(row.get(metric))
            if value is None:
                continue
            x.append(value)
            y.append(sum(_event_value(row, "own_boundary_loss_count", aid) for aid in agents))
        result[name] = pearson_or_none(x, y)
    return result


def read_training_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ensure_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_outputs(
    output: Path,
    episode_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    ensure_output_directory(output)
    with (output / "death_audit_episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader(); writer.writerows(episode_rows)
    phase_fields = list(phase_rows[0].keys()) if phase_rows else ["phase"]
    with (output / "training_phase_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=phase_fields)
        writer.writeheader(); writer.writerows(phase_rows)
    with (output / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)


def _compact_print(summary: Mapping[str, Any], output: Path) -> None:
    metadata = summary["checkpoint_metadata"]; formal = summary["formal_death_audit"]
    agents = formal["by_agent"]; late = summary["training_phase_summary"]["late_1500k_2000k"]
    metrics = late["mechanism_metrics"]; events = late["death_events"]
    def mean(field: str) -> str:
        value = metrics.get(field)
        return "n/a" if value is None else f"{value['mean']:.4f}"
    print("ROLE-GUIDED RUN AUDIT")
    print(f"seed={metadata['seed']} method={metadata['method_variant']} steps={metadata['sampled_steps']}")
    print(f"Formal: W={formal['win_rate']:.1%} Return={formal['mean_return']:.2f} "
          f"Kills={formal['mean_red_attack_kills']:.2f} MAV={formal['MAV_survival_rate']:.1%} "
          f"UAV={formal['mean_UAV_survivors']:.2f}")
    print(f"Death causes: MAV boundary={agents['MAV']['boundary']} blue={agents['MAV']['blue_attack']} | "
          f"UAV boundary={formal['uav_boundary_total']} blue={formal['uav_blue_attack_total']} "
          f"other={formal['uav_other_total']} | boundary share={formal['uav_boundary_share_of_uav_deaths']:.1%}")
    print(f"Late CR: lambda={mean('cr_lambda_mean')} conflict={mean('cr_conflict_rate')} "
          f"residual total/M/U={mean('cr_relational_residual_abs_mean')}/"
          f"{mean('cr_relational_residual_abs_mean_MAV')}/"
          f"{mean('cr_relational_residual_abs_mean_UAV')}")
    print("         att H/self/M->U/U->M/U->U="
          f"{mean('cr_attention_entropy')}/{mean('cr_attention_self_mass')}/"
          f"{mean('cr_attention_mav_to_uav_mass')}/"
          f"{mean('cr_attention_uav_to_mav_mass')}/"
          f"{mean('cr_attention_uav_to_other_uav_mass')}")
    print("Training death rate late: "
          f"MAV boundary/1000ep={events['MAV']['boundary_events_per_1000_episodes']:.2f} | "
          f"UAV boundary/1000ep={events['UAV_aggregate']['boundary_events_per_1000_episodes']:.2f} | "
          f"UAV blue/1000ep={events['UAV_aggregate']['blue_attack_events_per_1000_episodes']:.2f}")
    print(f"Files: {output}")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    training_path = run_dir / "training.csv"
    checkpoint = run_dir / "checkpoint_final.pt"
    for required in (training_path, checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    ensure_output_directory(output)
    device = resolved_device(args.device)
    actors, payload, env_config, contract = load_actor_only_checkpoint(checkpoint, device)
    episode_rows = audit_episodes(
        actors, env_config, args.episodes, args.profile,
        env_seed=args.env_seed, device=device,
        deterministic=args.action_mode == "deterministic",
        action_seed=None if args.action_mode == "deterministic" else args.action_seed,
    )
    training_rows = read_training_csv(training_path)
    phase_rows, phase_summary = analyze_training_rows(training_rows, contract["method_variant"])
    resolved = None
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.is_file():
        with resolved_path.open(encoding="utf-8") as stream:
            resolved = yaml.safe_load(stream)
    summary = {
        "checkpoint_metadata": {
            "checkpoint": str(checkpoint), "sampled_steps": int(payload.get("sampled_steps", 0)),
            "seed": int(payload.get("trainer_config", {}).get("seed", 0)),
            "environment_version": payload.get("environment_version"),
            "environment_profile": payload.get("environment_profile"),
            **contract, "device": device, "action_mode": args.action_mode,
            "action_seed": None if args.action_mode == "deterministic" else args.action_seed,
            "environment_seed_start": args.env_seed,
            "environment_seed_end": args.env_seed + args.episodes - 1,
            "resolved_config_present": resolved is not None,
        },
        "formal_death_audit": death_summary(episode_rows),
        "training_phase_summary": phase_summary,
        "exploratory_correlation": exploratory_correlations(training_rows, contract["method_variant"]),
    }
    # The directory was proven empty above; write_outputs performs the same
    # guard for direct library callers, so remove no files and write directly.
    with (output / "death_audit_episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader(); writer.writerows(episode_rows)
    phase_fields = list(phase_rows[0].keys()) if phase_rows else ["phase"]
    with (output / "training_phase_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=phase_fields)
        writer.writeheader(); writer.writerows(phase_rows)
    with (output / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
    _compact_print(summary, output)


if __name__ == "__main__":
    main()
