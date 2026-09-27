"""Read-only same-role UAV Gaussian-policy divergence audit.

The audit provides descriptive association evidence only.  It does not infer
that policy divergence causes boundary attrition.
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

from algorithm.happo.networks import IndependentActors
from env.mavuav import OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config


UAV_IDS = RED_IDS[1:]
UAV_ACTOR_INDICES = (1, 2, 3)
PAIR_SPECS = (("UAV1", "UAV2"), ("UAV1", "UAV3"), ("UAV2", "UAV3"))
SUPPORTED_METHODS = frozenset(("rgaa", "cr_rgaa", "lp_cr_rgaa", "ls_rgaa", "lsa_rgaa"))
PAIR_METRICS = (
    "skl_total", "skl_mean", "skl_scale", "raw_mean_l2", "deterministic_action_l2",
)
OUTLIER_METRICS = ("outlier_total", "outlier_mean_component", "outlier_scale_component")


@dataclass
class LoadedRun:
    label: str
    run_dir: Path
    checkpoint: Path
    actors: IndependentActors
    env_config: dict[str, Any]
    method: str
    seed: int
    sampled_steps: int
    checkpoint_digest: str
    initial_actor_state: dict[str, torch.Tensor]


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_mapping, required=True)
    parser.add_argument("--healthy-label", action="append", default=[])
    parser.add_argument("--death-audit", action="append", type=_parse_mapping, default=[])
    parser.add_argument("--probe-episodes", type=int, default=20)
    parser.add_argument("--profile", choices=("learnability", "main"), default="learnability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--max-probes-per-run-agent", type=int, default=512)
    parser.add_argument("--probe-seed", type=int, default=53001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("--run labels must be unique")
    unknown_healthy = sorted(set(args.healthy_label) - set(labels))
    if unknown_healthy:
        parser.error(f"unknown healthy labels: {unknown_healthy}")
    if args.probe_episodes <= 0 or args.max_probes_per_run_agent <= 0:
        parser.error("probe counts must be positive")
    return args


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_actor_state(actors: IndependentActors) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in actors.state_dict().items()}


def assert_actor_state_unchanged(loaded: LoadedRun) -> None:
    current = loaded.actors.state_dict()
    if current.keys() != loaded.initial_actor_state.keys() or any(
        not torch.equal(current[name].detach().cpu(), expected)
        for name, expected in loaded.initial_actor_state.items()
    ):
        raise RuntimeError(f"read-only audit mutated actors for {loaded.label}")
    if file_sha256(loaded.checkpoint) != loaded.checkpoint_digest:
        raise RuntimeError(f"read-only audit mutated checkpoint for {loaded.label}")


def load_run(label: str, run_dir: Path, device: str) -> LoadedRun:
    checkpoint = run_dir / "checkpoint_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint_final.pt for {label}: {checkpoint}")
    digest = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("trainer_config", payload.get("config", {}))
    method = str(payload.get("method_variant", config.get("method_variant", "")))
    if method not in SUPPORTED_METHODS:
        raise RuntimeError(f"unsupported method_variant for {label}: {method!r}")
    if payload.get("actor_variant", config.get("actor_variant", "vanilla")) != "vanilla":
        raise RuntimeError("same-role audit requires vanilla independent Gaussian actors")
    if "environment_config" not in payload:
        raise RuntimeError(f"checkpoint for {label} is missing environment_config")
    if "actors" not in payload:
        raise RuntimeError(f"checkpoint for {label} is missing actors")
    hidden_dim = int(config["hidden_dim"])
    actors = IndependentActors(
        hidden_dim=hidden_dim,
        log_std_init=float(config.get("actor_log_std_init", -0.5)),
    ).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    env_config = load_environment_config(payload["environment_config"])
    return LoadedRun(
        label=label, run_dir=run_dir, checkpoint=checkpoint, actors=actors,
        env_config=env_config, method=method, seed=int(config.get("seed", payload.get("seed", -1))),
        sampled_steps=int(payload.get("sampled_steps", 0)), checkpoint_digest=digest,
        initial_actor_state=clone_actor_state(actors),
    )


def _categorize_death(cause: str) -> str:
    return cause if cause in ("boundary", "blue_attack") else "other"


def _stable_seed(base_seed: int, *parts: str) -> int:
    material = "\0".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "little")


def collect_probes(
    loaded: LoadedRun,
    episodes: int,
    profile: str,
    *,
    env_seed: int,
    action_seed: int,
    device: str,
) -> list[dict[str, Any]]:
    """Collect active-UAV pre-action observations with formal stochastic seeding."""
    records: list[dict[str, Any]] = []
    env = HeterogeneousMAVUAVAirCombatEnv(loaded.env_config, profile=profile)
    for episode in range(int(episodes)):
        observations, _ = env.reset(seed=int(env_seed) + episode)
        torch.manual_seed(int(action_seed) + episode)
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(action_seed) + episode)
        episode_rows: list[dict[str, Any]] = []
        death_causes: dict[str, str] = {}
        done = False
        step = 0
        while not done:
            active_masks = np.asarray(env.active_masks, dtype=np.float32)
            actions: list[np.ndarray] = []
            with torch.no_grad():
                # Sampling order intentionally matches the formal evaluator.
                for index, aid in enumerate(RED_IDS):
                    if index > 0 and active_masks[index] > 0.5:
                        episode_rows.append({
                            "source_run": loaded.label,
                            "source_method": loaded.method,
                            "source_seed": loaded.seed,
                            "source_episode": episode,
                            "source_step": step,
                            "source_agent": aid,
                            "source_agent_active": True,
                            "observation": np.asarray(observations[aid], dtype=np.float32).copy(),
                        })
                    tensor = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
                    action, _ = loaded.actors.actors[index].sample(tensor, deterministic=False)
                    actions.append(action.squeeze(0).cpu().numpy())
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            for aid, cause in info.get("death_causes", {}).items():
                if aid in UAV_IDS:
                    death_causes[aid] = _categorize_death(str(cause))
            done = bool(terminated or truncated)
            step += 1
        for row in episode_rows:
            row["source_final_death_cause"] = death_causes.get(row["source_agent"], "alive")
        records.extend(episode_rows)
    return records


def balanced_subsample(
    probes: Sequence[Mapping[str, Any]], max_per_agent: int, probe_seed: int, run_label: str,
) -> list[dict[str, Any]]:
    """Deterministically retain an equal number of observations for each UAV."""
    groups = {aid: [dict(row) for row in probes if row["source_agent"] == aid] for aid in UAV_IDS}
    if any(not group for group in groups.values()):
        raise RuntimeError(f"run {run_label} produced no probes for at least one UAV")
    target = min(int(max_per_agent), *(len(group) for group in groups.values()))
    selected: list[dict[str, Any]] = []
    for aid in UAV_IDS:
        group = groups[aid]
        rng = np.random.default_rng(_stable_seed(probe_seed, run_label, aid))
        indices = np.sort(rng.choice(len(group), size=target, replace=False))
        selected.extend(group[int(index)] for index in indices)
    selected.sort(key=lambda row: (
        row["source_run"], int(row["source_episode"]), int(row["source_step"]),
        UAV_IDS.index(str(row["source_agent"])),
    ))
    return selected


def build_shared_probe_pool(
    probes_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    max_per_agent: int,
    probe_seed: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for label in sorted(probes_by_run):
        pool.extend(balanced_subsample(probes_by_run[label], max_per_agent, probe_seed, label))
    for index, row in enumerate(pool):
        row["probe_id"] = index
    return pool


def diagonal_gaussian_symmetric_kl(
    mean_i: np.ndarray,
    log_std_i: np.ndarray,
    mean_j: np.ndarray,
    log_std_j: np.ndarray,
) -> dict[str, np.ndarray]:
    mean_i = np.asarray(mean_i, dtype=np.float64)
    mean_j = np.asarray(mean_j, dtype=np.float64)
    var_i = np.exp(2.0 * np.asarray(log_std_i, dtype=np.float64))
    var_j = np.exp(2.0 * np.asarray(log_std_j, dtype=np.float64))
    mean_component_dim = 0.25 * (mean_i - mean_j) ** 2 * (1.0 / var_i + 1.0 / var_j)
    scale_component_dim = 0.25 * (var_i / var_j + var_j / var_i - 2.0)
    mean_component = mean_component_dim.mean(axis=-1)
    scale_component = scale_component_dim.mean(axis=-1)
    return {
        "skl_total": mean_component + scale_component,
        "skl_mean": mean_component,
        "skl_scale": scale_component,
        "raw_mean_l2": np.linalg.norm(mean_i - mean_j, axis=-1),
        "deterministic_action_l2": np.linalg.norm(np.tanh(mean_i) - np.tanh(mean_j), axis=-1),
    }


def compute_actor_outliers(pair_values: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    d12, d13, d23 = pair_values["UAV1-UAV2"], pair_values["UAV1-UAV3"], pair_values["UAV2-UAV3"]
    sources = {
        "UAV1": (d12, d13), "UAV2": (d12, d23), "UAV3": (d13, d23),
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    for aid, (first, second) in sources.items():
        result[aid] = {
            "outlier_total": 0.5 * (first["skl_total"] + second["skl_total"]),
            "outlier_mean_component": 0.5 * (first["skl_mean"] + second["skl_mean"]),
            "outlier_scale_component": 0.5 * (first["skl_scale"] + second["skl_scale"]),
        }
    return result


def evaluate_on_probe_pool(
    loaded: LoadedRun, probes: Sequence[Mapping[str, Any]], device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = np.stack([np.asarray(row["observation"], dtype=np.float32) for row in probes])
    tensor = torch.as_tensor(observations, device=device)
    means: dict[str, np.ndarray] = {}
    log_stds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for aid, actor_index in zip(UAV_IDS, UAV_ACTOR_INDICES):
            actor = loaded.actors.actors[actor_index]
            means[aid] = actor.network(tensor).cpu().numpy()
            log_std = actor.log_std.clamp(-5.0, 2.0).detach().cpu().numpy()
            log_stds[aid] = np.broadcast_to(log_std, means[aid].shape)
    pairs: dict[str, dict[str, np.ndarray]] = {}
    pair_rows: list[dict[str, Any]] = []
    for aid_i, aid_j in PAIR_SPECS:
        name = f"{aid_i}-{aid_j}"
        values = diagonal_gaussian_symmetric_kl(
            means[aid_i], log_stds[aid_i], means[aid_j], log_stds[aid_j],
        )
        pairs[name] = values
        for index, probe in enumerate(probes):
            pair_rows.append({
                "evaluated_run": loaded.label, "probe_id": int(probe["probe_id"]),
                "source_run": probe["source_run"], "source_method": probe["source_method"],
                "source_seed": probe["source_seed"], "source_episode": probe["source_episode"],
                "source_step": probe["source_step"], "source_agent": probe["source_agent"],
                "source_final_death_cause": probe["source_final_death_cause"], "pair": name,
                **{metric: float(values[metric][index]) for metric in PAIR_METRICS},
            })
    outliers = compute_actor_outliers(pairs)
    outlier_rows: list[dict[str, Any]] = []
    for aid in UAV_IDS:
        for index, probe in enumerate(probes):
            outlier_rows.append({
                "evaluated_run": loaded.label, "probe_id": int(probe["probe_id"]),
                "source_run": probe["source_run"], "source_agent": probe["source_agent"],
                "source_final_death_cause": probe["source_final_death_cause"], "actor": aid,
                **{metric: float(outliers[aid][metric][index]) for metric in OUTLIER_METRICS},
            })
    return pair_rows, outlier_rows


def descriptive(values: Iterable[float], *, actor: bool = False) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None
    result = {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)), "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)), "max": float(array.max()),
    }
    if not actor:
        result.update({"std": float(array.std(ddof=0)), "p50": result["median"]})
    return result


def summarize_pairwise(rows: Sequence[Mapping[str, Any]], pool: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in sorted({str(row["evaluated_run"]) for row in rows}):
        run_rows = [row for row in rows if row["evaluated_run"] == run]
        if pool == "own":
            run_rows = [row for row in run_rows if row["source_run"] == run]
        for pair in ("UAV1-UAV2", "UAV1-UAV3", "UAV2-UAV3"):
            selected = [row for row in run_rows if row["pair"] == pair]
            for metric in PAIR_METRICS:
                stats = descriptive(float(row[metric]) for row in selected)
                output.append({"run": run, "pool": pool, "pair": pair, "metric": metric, **(stats or {})})
    return output


def summarize_outliers(rows: Sequence[Mapping[str, Any]], pool: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in sorted({str(row["evaluated_run"]) for row in rows}):
        run_rows = [row for row in rows if row["evaluated_run"] == run]
        if pool == "own":
            run_rows = [row for row in run_rows if row["source_run"] == run]
        for aid in UAV_IDS:
            selected = [row for row in run_rows if row["actor"] == aid]
            for metric in OUTLIER_METRICS:
                stats = descriptive((float(row[metric]) for row in selected), actor=True)
                output.append({"run": run, "pool": pool, "actor": aid, "metric": metric, **(stats or {})})
    return output


def conditioned_summaries(
    pair_rows: Sequence[Mapping[str, Any]], outlier_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in sorted({str(row["evaluated_run"]) for row in pair_rows}):
        for condition_field, values in (
            ("source_agent", UAV_IDS),
            ("source_final_death_cause", ("alive", "boundary", "blue_attack")),
        ):
            for condition_value in values:
                conditioned_pairs = [
                    row for row in pair_rows
                    if row["evaluated_run"] == run and row[condition_field] == condition_value
                ]
                for pair in ("UAV1-UAV2", "UAV1-UAV3", "UAV2-UAV3"):
                    selected = [row for row in conditioned_pairs if row["pair"] == pair]
                    for metric in ("skl_total", "skl_mean", "skl_scale"):
                        stats = descriptive(float(row[metric]) for row in selected)
                        output.append({
                            "run": run, "kind": "pair", "item": pair,
                            "condition": condition_field, "condition_value": condition_value,
                            "metric": metric, **(stats or {}),
                        })
                conditioned_outliers = [
                    row for row in outlier_rows
                    if row["evaluated_run"] == run and row[condition_field] == condition_value
                ]
                for aid in UAV_IDS:
                    selected = [row for row in conditioned_outliers if row["actor"] == aid]
                    for metric in OUTLIER_METRICS:
                        stats = descriptive((float(row[metric]) for row in selected), actor=True)
                        output.append({
                            "run": run, "kind": "actor", "item": aid,
                            "condition": condition_field, "condition_value": condition_value,
                            "metric": metric, **(stats or {}),
                        })
    return output


def healthy_reference(
    pair_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
    healthy_labels: Sequence[str],
) -> dict[str, Any]:
    if not healthy_labels:
        return {
            "healthy_labels": [], "healthy_pairwise_q90": None,
            "healthy_pairwise_q95": None, "healthy_pairwise_q99": None,
            "healthy_actor_outlier_q90": None, "healthy_actor_outlier_q95": None,
            "healthy_actor_outlier_q99": None, "delta_candidate": None,
            "interpretation": "No empirical healthy reference was requested.",
        }
    label_set = set(healthy_labels)
    pair = np.asarray([
        float(row["skl_total"]) for row in pair_rows if row["evaluated_run"] in label_set
    ])
    actor = np.asarray([
        float(row["outlier_total"]) for row in outlier_rows if row["evaluated_run"] in label_set
    ])
    result = {"healthy_labels": list(healthy_labels)}
    for quantile, name in ((0.90, "q90"), (0.95, "q95"), (0.99, "q99")):
        result[f"healthy_pairwise_{name}"] = float(np.quantile(pair, quantile))
        result[f"healthy_actor_outlier_{name}"] = float(np.quantile(actor, quantile))
    result["delta_candidate"] = result["healthy_pairwise_q95"]
    result["interpretation"] = "Empirical descriptive reference, not a statistically proven threshold."
    return result


def excess_diagnostics(
    pair_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    pair_threshold = reference.get("healthy_pairwise_q95")
    actor_threshold = reference.get("healthy_actor_outlier_q95")
    output: dict[str, Any] = {}
    runs = sorted({str(row["evaluated_run"]) for row in pair_rows})
    for run in runs:
        pair_rates: dict[str, float | None] = {}
        actor_rates: dict[str, float | None] = {}
        actor_means: dict[str, float] = {}
        for pair in ("UAV1-UAV2", "UAV1-UAV3", "UAV2-UAV3"):
            values = np.asarray([
                float(row["skl_total"]) for row in pair_rows
                if row["evaluated_run"] == run and row["pair"] == pair
            ])
            pair_rates[pair] = None if pair_threshold is None else float((values > pair_threshold).mean())
        for aid in UAV_IDS:
            values = np.asarray([
                float(row["outlier_total"]) for row in outlier_rows
                if row["evaluated_run"] == run and row["actor"] == aid
            ])
            actor_means[aid] = float(values.mean())
            actor_rates[aid] = None if actor_threshold is None else float((values > actor_threshold).mean())
        max_mean_agent = max(actor_means, key=actor_means.get)
        median_mean = float(np.median(list(actor_means.values())))
        valid_rates = {aid: value for aid, value in actor_rates.items() if value is not None}
        max_rate_agent = max(valid_rates, key=valid_rates.get) if valid_rates else None
        output[run] = {
            "pair_excess_rate_q95": pair_rates,
            "actor_outlier_excess_rate_q95": actor_rates,
            "actor_outlier_mean": actor_means,
            "max_actor_outlier_mean": actor_means[max_mean_agent],
            "max_actor_outlier_agent": max_mean_agent,
            "max_actor_outlier_excess_rate": valid_rates[max_rate_agent] if max_rate_agent else None,
            "max_actor_outlier_excess_rate_agent": max_rate_agent,
            "outlier_ratio": actor_means[max_mean_agent] / median_mean if median_mean > 0.0 else None,
        }
    return output


def load_death_audit(path: Path) -> dict[str, dict[str, float | None]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    formal = data.get("formal_death_audit", data)
    by_agent = formal.get("by_agent", {})
    result: dict[str, dict[str, float | None]] = {}
    for aid in UAV_IDS:
        item = by_agent.get(aid)
        if not isinstance(item, Mapping):
            result[aid] = {"boundary_rate": None, "blue_attack_rate": None, "survival_rate": None}
            continue
        if all(key in item for key in ("boundary_rate", "blue_attack_rate", "survival_rate")):
            result[aid] = {key: float(item[key]) for key in ("boundary_rate", "blue_attack_rate", "survival_rate")}
        else:
            counts = {key: float(item.get(key, 0.0)) for key in ("alive", "boundary", "blue_attack", "other")}
            total = sum(counts.values())
            result[aid] = {
                "boundary_rate": counts["boundary"] / total if total else None,
                "blue_attack_rate": counts["blue_attack"] / total if total else None,
                "survival_rate": counts["alive"] / total if total else None,
            }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = str(args.device)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly for a CPU tiny audit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    loaded_runs: list[LoadedRun] = []
    try:
        loaded_runs = [load_run(label, path.resolve(), device) for label, path in args.run]
        raw_probes = {
            run.label: collect_probes(
                run, args.probe_episodes, args.profile, env_seed=args.env_seed,
                action_seed=args.action_seed, device=device,
            )
            for run in loaded_runs
        }
        pool = build_shared_probe_pool(
            raw_probes, args.max_probes_per_run_agent, args.probe_seed,
        )
        all_pairs: list[dict[str, Any]] = []
        all_outliers: list[dict[str, Any]] = []
        for run in loaded_runs:
            pair_rows, outlier_rows = evaluate_on_probe_pool(run, pool, device)
            all_pairs.extend(pair_rows)
            all_outliers.extend(outlier_rows)
        pair_summary = summarize_pairwise(all_pairs, "shared") + summarize_pairwise(all_pairs, "own")
        actor_summary = summarize_outliers(all_outliers, "shared") + summarize_outliers(all_outliers, "own")
        conditioned = conditioned_summaries(all_pairs, all_outliers)
        reference = healthy_reference(all_pairs, all_outliers, args.healthy_label)
        excess = excess_diagnostics(all_pairs, all_outliers, reference)
        death_paths = dict(args.death_audit)
        death = {
            run.label: load_death_audit(death_paths[run.label].resolve())
            if run.label in death_paths else {
                aid: {"boundary_rate": None, "blue_attack_rate": None, "survival_rate": None}
                for aid in UAV_IDS
            }
            for run in loaded_runs
        }
        manifest_rows = [{
            key: (json.dumps(np.asarray(value).tolist()) if key == "observation" else value)
            for key, value in row.items()
        } for row in pool]
        _write_csv(output / "probe_manifest.csv", manifest_rows)
        _write_csv(output / "shared_pairwise_samples.csv", all_pairs)
        _write_csv(output / "pairwise_summary.csv", pair_summary)
        _write_csv(output / "actor_outlier_summary.csv", actor_summary)
        _write_csv(output / "conditioned_summary.csv", conditioned)
        (output / "healthy_reference.json").write_text(
            json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        run_metadata = {
            run.label: {
                "method": run.method, "training_seed": run.seed,
                "sampled_steps": run.sampled_steps, "checkpoint": str(run.checkpoint),
                "probe_count": sum(row["source_run"] == run.label for row in pool),
                "death_audit": death[run.label], **excess[run.label],
            }
            for run in loaded_runs
        }
        summary = {
            "audit": "same_role_uav_policy_divergence",
            "read_only": True, "profile": args.profile,
            "probe_episodes": args.probe_episodes, "env_seed_start": args.env_seed,
            "env_seed_end": args.env_seed + args.probe_episodes - 1,
            "action_seed_start": args.action_seed,
            "action_seed_end": args.action_seed + args.probe_episodes - 1,
            "probe_seed": args.probe_seed, "shared_probe_count": len(pool),
            "healthy_reference": reference, "runs": run_metadata,
            "interpretation_limit": (
                "Descriptive association only: policy-function outlier behavior may co-occur "
                "with boundary-heavy attrition; no causal conclusion is supported."
            ),
            "observation_contract_caveat": (
                "UAV self-block semantics match and the first teammate block is MAV for every UAV; "
                "the other two blocks are same-role UAVs, but fixed slot ordering remains. Same-input "
                "comparison is a function-level diagnostic, not a permutation-invariant equivalence test."
            ),
        }
        (output / "audit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        readme = (
            "Same-role UAV policy divergence audit\n"
            "=====================================\n"
            "All checkpoints are evaluated on the identical balanced shared probe pool.\n"
            "The Gaussian divergence is analytic; no sampled-action divergence estimate is used.\n"
            "Healthy quantiles are empirical references, not statistically proven thresholds.\n"
            + summary["interpretation_limit"] + "\n" + summary["observation_contract_caveat"] + "\n"
        )
        (output / "README.txt").write_text(readme, encoding="utf-8")
        for run in loaded_runs:
            assert_actor_state_unchanged(run)
            shared_pairs = {
                metric: {
                    pair: next(row["mean"] for row in pair_summary if row["run"] == run.label
                               and row["pool"] == "shared" and row["pair"] == pair
                               and row["metric"] == metric)
                    for pair in ("UAV1-UAV2", "UAV1-UAV3", "UAV2-UAV3")
                }
                for metric in ("skl_total", "skl_mean", "skl_scale")
            }
            means = excess[run.label]["actor_outlier_mean"]
            print(f"{run.label}: method={run.method} seed={run.seed} steps={run.sampled_steps}")
            for metric, label in (("skl_total", "total"), ("skl_mean", "mean component"),
                                  ("skl_scale", "scale component")):
                print(f"  pair SKL {label}: " + " / ".join(
                    f"{shared_pairs[metric][pair]:.6g}" for pair in shared_pairs[metric]
                ))
            print("  actor outlier mean: " + " / ".join(f"{means[aid]:.6g}" for aid in UAV_IDS))
            if reference["healthy_pairwise_q95"] is not None:
                print(f"  empirical healthy pair q95={reference['healthy_pairwise_q95']:.6g}")
                pair_rates = excess[run.label]["pair_excess_rate_q95"]
                actor_rates = excess[run.label]["actor_outlier_excess_rate_q95"]
                print("  pair excess q95: " + " / ".join(
                    f"{pair_rates[pair]:.1%}" for pair in pair_rates
                ))
                print("  actor excess q95: " + " / ".join(
                    f"{actor_rates[aid]:.1%}" for aid in UAV_IDS
                ))
            if any(death[run.label][aid]["boundary_rate"] is not None for aid in UAV_IDS):
                print("  UAV boundary / outlier / excess:")
                for aid in UAV_IDS:
                    boundary = death[run.label][aid]["boundary_rate"]
                    actor_excess = excess[run.label]["actor_outlier_excess_rate_q95"][aid]
                    boundary_text = "n/a" if boundary is None else f"{boundary:.1%}"
                    excess_text = "n/a" if actor_excess is None else f"{actor_excess:.1%}"
                    print(f"    {aid}: {boundary_text} / {means[aid]:.6g} / {excess_text}")
        return summary
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_rng])


def main(argv: Sequence[str] | None = None) -> None:
    run_audit(parse_args(argv))


if __name__ == "__main__":
    main()
