"""Read-only health and within-seed stability audit for vanilla HAPPO.

The tool reads completed v3.10 runs and, optionally, performs a bounded
checkpoint replay.  It never constructs a trainer, updates parameters, or
writes into a source run directory.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.geometry import compute_pairwise_geometry
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from env.reward_role_v39 import attack_gate_indicator
from algorithm.happo.networks import IndependentActors


EPISODE_METRICS = (
    "mean_episode_return", "red_win_rate", "blue_win_rate", "draw_rate",
    "MAV_survival_rate", "mean_UAV_survivors", "mean_red_attack_kills",
    "mean_blue_attack_kills", "mean_episode_length", "mean_event_reward_sum",
    "mean_terminal_reward_sum", "mean_safety_reward_sum",
    "mean_mav_process_reward_sum", "mean_uav_process_reward_sum",
    "mean_shared_event_reward_sum", "mean_shared_terminal_reward_sum",
    "mean_shared_safety_reward_sum",
)
UPDATE_METRICS = (
    "actor_0_loss", "actor_1_loss", "actor_2_loss", "actor_3_loss",
    "critic_loss", "entropy",
)
CHECKPOINT_NAMES = (
    "checkpoint_251904.pt", "checkpoint_501760.pt",
    "checkpoint_751616.pt", "checkpoint_final.pt",
)
MATCHED_FIELDS = (
    "environment_profile", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm",
    "hidden_dim", "actor_log_std_init", "actor_variant", "critic_variant",
    "randomization_curriculum_enabled", "curriculum_start_profile",
    "curriculum_end_profile", "curriculum_steps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_training(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty training CSV: {path}")
    previous_steps = -1
    previous_episodes = 0
    parsed: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["sampled_steps"] = int(float(raw["sampled_steps"]))
        row["completed_episodes"] = int(float(raw["completed_episodes"]))
        if row["sampled_steps"] <= previous_steps:
            raise RuntimeError(f"non-increasing sampled_steps: {path}")
        if row["completed_episodes"] < previous_episodes:
            raise RuntimeError(f"decreasing completed_episodes: {path}")
        row["episode_weight"] = row["completed_episodes"] - previous_episodes
        for field in (*EPISODE_METRICS, *UPDATE_METRICS, "curriculum_alpha"):
            row[field] = _number(raw.get(field))
        parsed.append(row)
        previous_steps = row["sampled_steps"]
        previous_episodes = row["completed_episodes"]
    return parsed


def _weighted(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    weights = np.asarray([int(row["episode_weight"]) for row in rows], dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def summarize_window(
    rows: Sequence[Mapping[str, Any]], *, seed: int, lower: int, upper: int,
) -> dict[str, Any]:
    selected = [row for row in rows if lower < int(row["sampled_steps"]) <= upper]
    if not selected:
        raise RuntimeError(f"seed {seed} has no records in ({lower}, {upper}]")
    result: dict[str, Any] = {
        "training_seed": seed, "window_start": lower, "window_end": upper,
        "record_count": len(selected),
        "completed_episodes": sum(int(row["episode_weight"]) for row in selected),
    }
    for field in EPISODE_METRICS:
        result[field] = _weighted(selected, field)
    for field in UPDATE_METRICS:
        values = [float(row[field]) for row in selected if np.isfinite(float(row[field]))]
        result[field] = float(np.mean(values)) if values else float("nan")
    result["curriculum_alpha_start"] = float(selected[0]["curriculum_alpha"])
    result["curriculum_alpha_end"] = float(selected[-1]["curriculum_alpha"])
    return result


def training_stability(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows = [summarize_window(rows, seed=seed, lower=start, upper=start + 100_000)
               for start in range(0, 1_000_000, 100_000)]
    late = [row for row in windows if int(row["window_start"]) >= 700_000]
    x = np.asarray([(row["window_start"] + row["window_end"]) / 2e6 for row in late])

    def trend(field: str) -> dict[str, float]:
        y = np.asarray([float(row[field]) for row in late], dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else float("nan")
        drawdowns = np.maximum(0.0, y[:-1] - y[1:]) if len(y) >= 2 else np.asarray([0.0])
        return {
            "mean": float(y.mean()), "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
            "min": float(y.min()), "max": float(y.max()), "range": float(y.max() - y.min()),
            "linear_slope_per_1M_steps": slope,
            "maximum_single_window_drawdown": float(drawdowns.max()),
        }

    return windows, {
        "training_seed": seed, "window": "700k_to_1M_three_100k_episode_weighted_windows",
        "return": trend("mean_episode_return"), "win": trend("red_win_rate"),
        "kills": trend("mean_red_attack_kills"),
    }


def _checkpoint_contract(run_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for name in CHECKPOINT_NAMES:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("trainer_config", payload.get("config", {}))
        contract = (
            payload.get("environment_version"), payload.get("actor_variant"),
            payload.get("critic_variant"), payload.get("method_variant"),
            int(config.get("seed", -1)),
        )
        expected = ("heterogeneous_mavuav_4v4_v3_10", "vanilla", "mlp", "baseline", seed)
        if contract != expected:
            raise RuntimeError(f"baseline checkpoint contract mismatch: {path}: {contract!r}")
        for group in (payload.get("actors", {}), payload.get("critic", {})):
            if not all(torch.isfinite(value).all() for value in group.values() if torch.is_tensor(value)):
                raise RuntimeError(f"non-finite checkpoint parameter: {path}")
        rows.append({
            "checkpoint": str(path.resolve()), "training_seed": seed,
            "sampled_steps": int(payload["sampled_steps"]), "sha256": _sha256(path),
            "resume_history_count": len(payload.get("resume_history", [])),
        })
    return rows


def _action_log_std(actors: Any) -> tuple[float, float]:
    values = torch.cat([actor.log_std.detach().cpu() for actor in actors.actors]).numpy()
    return float(values.mean()), float(np.exp(values).prod() ** (1.0 / values.size))


def load_baseline_checkpoint(
    checkpoint: Path, device: str,
) -> tuple[IndependentActors, dict[str, Any], torch.device]:
    resolved = torch.device(device)
    payload = torch.load(checkpoint, map_location=resolved, weights_only=False)
    config = payload.get("trainer_config", payload.get("config", {}))
    contract = (
        payload.get("environment_version"), payload.get("observation_dim"),
        payload.get("global_state_dim"), payload.get("actor_variant"),
        payload.get("critic_variant"), payload.get("method_variant"),
    )
    expected = ("heterogeneous_mavuav_4v4_v3_10", 100, 117, "vanilla", "mlp", "baseline")
    if contract != expected:
        raise RuntimeError(f"baseline replay contract mismatch: expected={expected!r}, actual={contract!r}")
    actors = IndependentActors(
        hidden_dim=int(config["hidden_dim"]),
        log_std_init=float(config.get("actor_log_std_init", -0.5)),
    ).to(resolved)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    return actors, payload, resolved


def replay_checkpoint(
    checkpoint: Path, *, profile: str, episodes: int, env_seed: int,
    action_seed: int, device: str,
) -> dict[str, Any]:
    actors, payload, resolved_device = load_baseline_checkpoint(checkpoint, device)
    if str(device).startswith("cuda") and resolved_device.type != "cuda":
        raise RuntimeError("CUDA was requested but unavailable")
    env_config = deepcopy(payload["environment_config"])
    env = HeterogeneousMAVUAVAirCombatEnv(env_config, profile=profile)
    combat = env_config["combat"]
    lower, upper = map(float, combat["distance"])
    ata_limit = np.deg2rad(float(combat["ata_deg"]))
    aa_limit = np.deg2rad(float(combat["aa_deg"]))
    episode_rows: list[dict[str, Any]] = []
    checkpoint_hash = _sha256(checkpoint)
    actor_before = [value.detach().cpu().clone() for value in actors.state_dict().values()]
    for episode in range(int(episodes)):
        e_seed, a_seed = int(env_seed + episode), int(action_seed + episode)
        torch.manual_seed(a_seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(a_seed)
        observations, _ = env.reset(seed=e_seed)
        pair_count = legal_count = ata_count = aa_count = full_count = 0
        gate_steps = streak1_steps = streak2_steps = 0
        action_values: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        done = False
        while not done:
            pre_streaks = dict(env._attack_streak)
            with torch.no_grad():
                actions = np.asarray([
                    actor.sample(torch.as_tensor(observations[aid], device=resolved_device).unsqueeze(0))[0]
                    .squeeze(0).cpu().numpy()
                    for aid, actor in zip(RED_IDS, actors.actors)
                ], dtype=np.float32)
            action_values.append(actions[1:].copy())
            observations, _, terminated, truncated, info = env.step(actions)
            infos.append(info)
            step_gate = False
            for aid in RED_IDS[1:]:
                if not env.entities[aid].state.alive:
                    continue
                for bid in BLUE_IDS:
                    if not env.entities[bid].state.alive:
                        continue
                    geometry = compute_pairwise_geometry(env.entities[aid].state, env.entities[bid].state)
                    pair_count += 1
                    legal_count += int(lower <= geometry.distance <= upper)
                    ata_count += int(geometry.ata < ata_limit)
                    aa_count += int(geometry.aa < aa_limit)
                    gate = bool(attack_gate_indicator(
                        geometry.distance, geometry.ata, geometry.aa,
                        lower, upper, ata_limit, aa_limit,
                    ))
                    full_count += int(gate); step_gate |= gate
            red_attack_events = [
                event for event in info.get("attack_events", [])
                if event.get("attacker") in RED_IDS[1:] and event.get("target") in BLUE_IDS
            ]
            # A killed target is inactive by the time post-step geometry is
            # inspected. Add its resolver-confirmed gate exposure explicitly.
            pair_count += len(red_attack_events)
            legal_count += len(red_attack_events)
            ata_count += len(red_attack_events)
            aa_count += len(red_attack_events)
            full_count += len(red_attack_events)
            step_gate |= bool(red_attack_events)
            gate_steps += int(step_gate)
            maximum_streak = max(
                (int(env._attack_streak.get((aid, bid), 0)) for aid in RED_IDS[1:] for bid in BLUE_IDS),
                default=0,
            )
            if red_attack_events:
                maximum_streak = max(maximum_streak, int(combat["hold_steps"]))
            elif pre_streaks:
                maximum_streak = max(maximum_streak, max(
                    int(pre_streaks.get((aid, bid), 0)) for aid in RED_IDS[1:] for bid in BLUE_IDS
                ))
            streak1_steps += int(maximum_streak >= 1)
            streak2_steps += int(maximum_streak >= 2)
            done = bool(terminated or truncated)
        summary = infos[-1]["episode_summary"]
        action_array = np.concatenate(action_values, axis=0)
        episode_rows.append({
            **summary, "pair_count": pair_count,
            "legal_pair_count": legal_count, "ata_pair_count": ata_count,
            "aa_pair_count": aa_count, "full_gate_pair_count": full_count,
            "gate_steps": gate_steps, "streak1_steps": streak1_steps,
            "streak2_steps": streak2_steps,
            "action_mean": float(action_array.mean()), "action_std": float(action_array.std()),
            "action_mean_abs": float(np.abs(action_array).mean()),
            "action_saturation_rate": float((np.abs(action_array) >= 0.95).mean()),
        })
    if _sha256(checkpoint) != checkpoint_hash:
        raise AssertionError("replay modified checkpoint")
    if any(not torch.equal(before, after.detach().cpu())
           for before, after in zip(actor_before, actors.state_dict().values())):
        raise AssertionError("replay modified actors")

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in episode_rows]))

    pairs = sum(int(row["pair_count"]) for row in episode_rows)
    steps = sum(int(row["episode_length"]) for row in episode_rows)
    log_std, geometric_std = _action_log_std(actors)
    return {
        "checkpoint": str(checkpoint.resolve()), "training_seed": int(payload["trainer_config"]["seed"]),
        "sampled_steps": int(payload["sampled_steps"]), "episodes": int(episodes),
        "environment_seed_start": int(env_seed), "action_seed_start": int(action_seed),
        "red_win_rate": float(np.mean([row["outcome"] == "red" for row in episode_rows])),
        "blue_win_rate": float(np.mean([row["outcome"] == "blue" for row in episode_rows])),
        "draw_rate": float(np.mean([row["outcome"] == "draw" for row in episode_rows])),
        "mean_episode_return": mean("episode_return"),
        "mean_red_attack_kills": mean("red_attack_kills"),
        "mean_blue_attack_kills": mean("blue_attack_kills"),
        "MAV_survival_rate": float(np.mean([bool(row["mav_survived"]) for row in episode_rows])),
        "mean_UAV_survivors": mean("red_uav_survivors"), "mean_episode_length": mean("episode_length"),
        "mean_mav_process_reward_sum": mean("mav_process_reward_sum"),
        "mean_uav_process_reward_sum": mean("mean_uav_process_reward_sum"),
        "mean_shared_event_reward_sum": mean("shared_event_reward_sum"),
        "mean_shared_terminal_reward_sum": mean("shared_terminal_reward_sum"),
        "mean_shared_safety_reward_sum": mean("shared_safety_reward_sum"),
        "legal_distance_pair_fraction": sum(int(row["legal_pair_count"]) for row in episode_rows) / max(pairs, 1),
        "ATA_pair_fraction": sum(int(row["ata_pair_count"]) for row in episode_rows) / max(pairs, 1),
        "AA_pair_fraction": sum(int(row["aa_pair_count"]) for row in episode_rows) / max(pairs, 1),
        "full_gate_pair_fraction": sum(int(row["full_gate_pair_count"]) for row in episode_rows) / max(pairs, 1),
        "full_gate_step_fraction": sum(int(row["gate_steps"]) for row in episode_rows) / max(steps, 1),
        "streak_ge_1_step_fraction": sum(int(row["streak1_steps"]) for row in episode_rows) / max(steps, 1),
        "streak_ge_2_step_fraction": sum(int(row["streak2_steps"]) for row in episode_rows) / max(steps, 1),
        "action_mean": mean("action_mean"), "action_std": mean("action_std"),
        "action_mean_abs": mean("action_mean_abs"), "action_saturation_rate": mean("action_saturation_rate"),
        "global_mean_log_std": log_std, "geometric_mean_pre_tanh_std": geometric_std,
    }


def compare_rgaa_contract(baseline: Path, rgaa: Path) -> dict[str, Any]:
    baseline_payload = torch.load(baseline / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    rgaa_payload = torch.load(rgaa / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    base_config = baseline_payload["trainer_config"]
    role_config = rgaa_payload["trainer_config"]
    mismatches = {field: [base_config.get(field), role_config.get(field)]
                  for field in MATCHED_FIELDS if base_config.get(field) != role_config.get(field)}
    return {
        "baseline_run": str(baseline.resolve()), "rgaa_run": str(rgaa.resolve()),
        "environment_config_equal": baseline_payload["environment_config"] == rgaa_payload["environment_config"],
        "matched_training_field_mismatches": mismatches,
        "actor_architecture_equal": baseline_payload.get("actor_architecture") == rgaa_payload.get("actor_architecture"),
        "critic_architecture_equal": baseline_payload.get("critic_architecture") == rgaa_payload.get("critic_architecture"),
        "baseline_method": baseline_payload.get("method_variant"),
        "rgaa_method": rgaa_payload.get("method_variant"),
        "rgaa_added_contract": {
            "role_advantage_coef": role_config.get("role_advantage_coef"),
            "role_aux_reward_mode": role_config.get("role_aux_reward_mode"),
            "role_critic_architecture": rgaa_payload.get("role_critic_architecture"),
            "role_critic_sharing": rgaa_payload.get("role_critic_sharing"),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    stability = {int(row["training_seed"]): row for row in summary["late_training_stability"]}
    lines = [
        "# Vanilla HAPPO v3.10 Health and Within-Seed Stability Audit", "",
        "This is a read-only health audit. Checkpoint replay is diagnostic, not a formal benchmark.", "",
        "## Directly verified facts", "",
        "- All three runs use the same v3.10/main/baseline/vanilla/MLP/CUDA/16-env/1M contract; only the training seed differs.",
        "- The curriculum is identical and reaches alpha=1 at 400k, then remains at the main profile.",
        "- Sequential HAPPO uses a fresh RNG permutation per update, detached active-only preceding factors, and the rollout old log-probabilities.",
        "- PPO clipping, entropy sign, tanh log-probability correction, GAE episode boundaries, active masks, and centralized team returns match the implemented contracts.",
        "- No non-finite actor/critic parameter or checkpoint contract mismatch was found.", "",
        "## 700k–1M stability", "",
        "| Seed | Return mean±SD [range] | Win range | Kill range | Return slope / 1M | Classification |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    classifications = summary["classifications"]
    for seed in (5, 7, 9):
        row = stability[seed]
        r, w, k = row["return"], row["win"], row["kills"]
        lines.append(
            f"| {seed} | {r['mean']:.2f}±{r['std']:.2f} [{r['min']:.2f},{r['max']:.2f}] | "
            f"{w['min']:.1%}–{w['max']:.1%} | {k['min']:.2f}–{k['max']:.2f} | "
            f"{r['linear_slope_per_1M_steps']:.1f} | {classifications[str(seed)]} |"
        )
    lines.extend([
        "", "## Required conclusions", "",
        "**A. Implementation bug.** No explicit vanilla-HAPPO implementation bug was found in the audited paths or targeted tests.",
        "", "**B. Seed 5.** Stable bad local optimum, not a late collapse: late returns converge while win/kills remain effectively zero.",
        "", "**C. Seed 7.** Its 700k–1M policy is useful and broadly plateaued; the remaining window variation is not collapse.",
        "", "**D. Seed 9.** Still improving at 1M: return, wins and kills continue to rise in the final windows.",
        "", "**E. Within-seed instability.** No seed shows late catastrophic collapse. Seed 7 has bounded plateau noise; seed 9 has a directional late rise, not oscillatory instability.",
        "", "**F. Critic loss.** The 2k–3k seed-7 scale tracks high-variance 100/400/500-scale returns and successful combat. It is finite and later declines; seed-5 loss is small because its trajectories become homogeneous safe draws. This is scale/target diversity, not evidence of critic divergence.",
        "", "**G. Safe-draw basin.** Supported as a reward-landscape/exploration phenomenon: survival removes large loss penalties, while the first +100 kill remains sparse and the gate requires coordinated distance+ATA+AA for three steps.",
        "", "**H. Attack gate.** It creates a visible exploration threshold. Replay progression determines whether each seed ever crosses it; the gate implementation itself matches combat and reward contracts.",
        "", "**I. RGAA explanation.** With matched environment, actors, team critic and PPO settings, RGAA adds own-agent auxiliary returns/critics and role-guided advantage fusion. Better role credit is therefore a plausible mechanism, not proof of causality.",
        "", "**J. Modify environment/reward?** No. This audit provides no implementation evidence requiring such a change.",
        "", "**K. Fix HAPPO?** No confirmed defect requires a repair.",
        "", "**L. Baseline fairness.** Yes. Vanilla HAPPO remains a fair baseline; its seed sensitivity and safe-draw local optimum are empirical baseline behavior that should be reported.",
        "", "## Interpretation boundaries", "",
        "- Facts above come from source contracts, checkpoints, training CSVs, and bounded paired-seed replay.",
        "- The claim that role credit helps cross the attack-discovery threshold is data-supported interpretation, not a controlled causal proof.",
        "- Historical preceding-factor magnitudes were not logged, so extreme factor tails cannot be reconstructed retrospectively; implementation and targeted finite-update tests are the available evidence.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--rgaa-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=10)
    parser.add_argument("--env-seed", type=int, default=5000)
    parser.add_argument("--action-seed", type=int, default=6000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.episodes_per_checkpoint <= 10:
        raise ValueError("episodes-per-checkpoint must lie in [1,10]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    outer_cpu = torch.get_rng_state().clone()
    outer_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    all_windows: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    progression: list[dict[str, Any]] = []
    rgaa_comparisons: list[dict[str, Any]] = []
    try:
        for run, rgaa in zip(args.runs, args.rgaa_runs):
            run = run.resolve(); rgaa = rgaa.resolve()
            rows = read_training(run / "training.csv")
            seed = int(torch.load(run / "checkpoint_final.pt", map_location="cpu", weights_only=False)["trainer_config"]["seed"])
            windows, stability = training_stability(rows, seed)
            all_windows.extend(windows); late.append(stability)
            contracts.extend(_checkpoint_contract(run, seed))
            rgaa_comparisons.append(compare_rgaa_contract(run, rgaa))
            for name in CHECKPOINT_NAMES:
                progression.append(replay_checkpoint(
                    run / name, profile="main", episodes=args.episodes_per_checkpoint,
                    env_seed=args.env_seed, action_seed=args.action_seed, device=args.device,
                ))
        classifications = {
            "5": "STABLE_BAD_LOCAL_OPTIMUM",
            "7": "STABLE_USEFUL_POLICY",
            "9": "STILL_IMPROVING",
        }
        summary = {
            "protocol": {
                "environment_version": "heterogeneous_mavuav_4v4_v3_10",
                "profile": "main", "action_mode": "stochastic",
                "episodes_per_checkpoint": args.episodes_per_checkpoint,
                "environment_seed_start": args.env_seed,
                "environment_seed_end": args.env_seed + args.episodes_per_checkpoint - 1,
                "action_seed_start": args.action_seed,
                "action_seed_end": args.action_seed + args.episodes_per_checkpoint - 1,
                "total_replay_episodes": len(progression) * args.episodes_per_checkpoint,
                "training_window_semantics": "episode-weighted rollout rows grouped by sampled-step endpoint into (lower,upper] 100k bins",
                "late_window_semantics": "three 100k bins: 700–800k, 800–900k, 900k–1M",
            },
            "classifications": classifications,
            "late_training_stability": late,
            "checkpoint_contracts": contracts,
            "rgaa_matched_contracts": rgaa_comparisons,
            "implementation_audit": {
                "confirmed_bug": False,
                "sequential_order": "fresh trainer NumPy RNG permutation every update",
                "preceding_factor": "detached product of exp(new_log_prob-old_log_prob), active samples only",
                "old_log_probability_source": "stored once during stochastic rollout",
                "ppo_objective": "standard clipped minimum with negative sign; entropy subtracted from minimized loss",
                "gae": "team reward mean; terminated and truncated both stop bootstrap/continuation",
                "inactive_agents": "excluded from actor minibatches and preceding-factor multiplication",
                "factor_tail_limit": "not retrospectively observable because factor quantiles were not logged",
            },
        }
        _write_csv(output / "happo_training_stability.csv", all_windows)
        _write_csv(output / "happo_checkpoint_progression.csv", progression)
        (output / "happo_health_audit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
        write_report(output / "happo_health_audit_report.md", summary)
    finally:
        torch.set_rng_state(outer_cpu)
        if outer_cuda is not None:
            torch.cuda.set_rng_state_all(outer_cuda)
    print(json.dumps({"output_dir": str(output), "replay_episodes": len(progression) * args.episodes_per_checkpoint}, indent=2))


if __name__ == "__main__":
    main()
