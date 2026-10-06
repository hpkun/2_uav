"""Read-only v3.10 baseline/TACM audit; replay is opt-in (no training)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.dbm_rgaa import build_method_actors
from algorithm.happo.evaluation import evaluate_actors, summarize_records
from algorithm.happo.networks import IndependentActors
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from tools.audit_role_guided_run import validate_death_accounting

VERSION = "heterogeneous_mavuav_4v4_v3_10"
PHASES = (("curriculum", 0, 400000), ("400k_1M", 400000, 1000000),
          ("1M_1p5M", 1000000, 1500000), ("1p5M_2M", 1500000, 2000000),
          ("early_100352", 0, 100352))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(obj):
    if isinstance(obj, torch.Tensor):
        return bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(finite_tree(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(finite_tree(value) for value in obj)
    if isinstance(obj, float):
        return bool(np.isfinite(obj))
    return True


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_valid_checkpoint(run):
    candidates = []
    invalid = []
    for path in Path(run).glob("checkpoint*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not all(key in payload for key in ("actors", "critic", "trainer_config", "environment_config", "sampled_steps", "actor_optimizer_states", "critic_optimizer_state", "trainer_numpy_rng", "torch_rng", "rollout_state")):
                raise RuntimeError("incomplete training checkpoint")
            if not finite_tree(payload):
                raise RuntimeError("nonfinite checkpoint")
            validate_checkpoint_contract(payload, load_environment_config(payload["environment_config"]))
            candidates.append((int(payload["sampled_steps"]), path.name == "checkpoint_final.pt", path, payload))
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    if not candidates:
        raise RuntimeError(f"no valid training checkpoint in {run}: {invalid}")
    _, _, path, payload = max(candidates, key=lambda item: item[:2])
    return path, payload, invalid


def load_actors(payload, device):
    env_config = load_environment_config(payload["environment_config"])
    validate_checkpoint_contract(payload, env_config)
    config = payload["trainer_config"]
    method = payload.get("method_variant", config["method_variant"])
    if env_config["environment_version"] != VERSION or method not in ("baseline", "tacm_rgaa"):
        raise RuntimeError("audit requires frozen v3.10 baseline or tacm_rgaa")
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith("cuda") else []):
        if method == "tacm_rgaa":
            actors = build_method_actors(
                method_variant="dbm_rgaa", training_seed=int(config["seed"]),
                hidden_dim=int(config["hidden_dim"]),
                log_std_init=float(config.get("actor_log_std_init", -0.5)),
                role_module_enabled=bool(config.get("role_module_enabled", True)),
                dbm_role_count=int(config.get("dbm_role_count", 2)),
                dbm_residual_scale=float(config.get("dbm_residual_scale", 0.25)),
                dbm_expert_init_scale=float(config.get("dbm_init_scale", 0.01)),
            )
        else:
            actors = IndependentActors(hidden_dim=int(config["hidden_dim"]),
                                       log_std_init=float(config.get("actor_log_std_init", -0.5)))
        actors = actors.to(device)
        actors.load_state_dict(payload["actors"], strict=True)
    actors.eval()
    return actors, env_config


def replay(actors, config, episodes, device, env_seed=1000, action_seed=2000):
    records = []
    env = HeterogeneousMAVUAVAirCombatEnv(config, profile="main")
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith("cuda") else []):
        for episode in range(episodes):
            obs, _ = env.reset(seed=env_seed + episode)
            torch.manual_seed(action_seed + episode)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(action_seed + episode)
            deaths = {}
            steps = {}
            done = False
            while not done:
                actions = []
                with torch.no_grad():
                    for index, aid in enumerate(env.red_ids):
                        action, _ = actors.actors[index].sample(
                            torch.as_tensor(obs[aid], device=device).unsqueeze(0), deterministic=False)
                        actions.append(action.squeeze(0).cpu().numpy())
                obs, _, terminated, truncated, info = env.step(np.asarray(actions))
                for aid, cause in info.get("death_causes", {}).items():
                    if aid in RED_IDS:
                        if aid in deaths:
                            raise AssertionError("duplicate Red death event")
                        deaths[aid] = str(cause)
                        steps[aid] = env.step_count
                done = bool(terminated or truncated)
            row = {"episode": episode, "environment_seed": env_seed + episode,
                   "action_seed": action_seed + episode, **info["episode_summary"],
                   "blue_survivors": sum(env.entities[bid].state.alive for bid in BLUE_IDS),
                   **{f"{aid}_death_cause": deaths.get(aid, "alive") for aid in RED_IDS},
                   **{f"{aid}_death_step": steps.get(aid) for aid in RED_IDS}}
            validate_death_accounting(row)
            records.append(row)
    return records


def death_and_cleanup(records):
    death_rows = []
    for outcome in ("all", "red", "draw", "blue"):
        selected = [r for r in records if outcome == "all" or r["outcome"] == outcome]
        for aid in RED_IDS:
            causes = [r[f"{aid}_death_cause"] for r in selected]
            steps = [r[f"{aid}_death_step"] for r in selected if r[f"{aid}_death_step"] is not None]
            counts = {cause: causes.count(cause) for cause in ("alive", "boundary", "blue_attack")}
            counts["other"] = len(causes) - sum(counts.values())
            death_rows.append({"outcome": outcome, "agent": aid, "episodes": len(selected),
                               **{f"{key}_count": value for key, value in counts.items()},
                               **{f"{key}_rate": value / len(selected) if selected else None for key, value in counts.items()},
                               "raw_causes": json.dumps({c: causes.count(c) for c in sorted(set(causes))}),
                               "mean_death_step": float(np.mean(steps)) if steps else None,
                               **{f"death_step_{label}": float(np.quantile(steps, q)) if steps else None
                                  for label, q in (("p10", .1), ("median", .5), ("p90", .9))}})
    draws = [r for r in records if r["outcome"] == "draw"]
    cleanup = [r for r in draws if r["red_attack_kills"] == 3 and r["blue_survivors"] == 1]
    all_uav = [r[f"{aid}_death_cause"] for r in records for aid in RED_IDS[1:]]
    loss_count = sum(c != "alive" for c in all_uav)
    fraction = len(cleanup) / len(draws) if draws else None
    return death_rows, {
        "uav_deaths": loss_count, "uav_deaths_per_episode": loss_count / len(records),
        "boundary_share_of_uav_deaths": all_uav.count("boundary") / loss_count if loss_count else None,
        "blue_attack_share_of_uav_deaths": all_uav.count("blue_attack") / loss_count if loss_count else None,
        "draw_count": len(draws), "draw_blue_survivors": {str(n): sum(r["blue_survivors"] == n for r in draws) for n in range(1, 5)},
        "draw_red_kills": {str(n): sum(r["red_attack_kills"] == n for r in draws) for n in range(4)},
        "three_kills_one_blue_count": len(cleanup), "share_of_draws": fraction,
        "cleanup_draw_uav_survivors": {str(n): sum(r["red_uav_survivors"] == n for r in cleanup) for n in range(4)},
        "cleanup_draw_MAV_alive_count": sum(bool(r["mav_survived"]) for r in cleanup),
        "last_blue_hypothesis": "INCONCLUSIVE" if len(records) != 200 or fraction is None else
            ("VERIFIED" if fraction > .5 else "PARTIALLY_SUPPORTED" if fraction > 0 else "REFUTED"),
    }


def training_analysis(rows):
    if not rows:
        raise RuntimeError("empty training.csv")
    prepared = []
    previous_steps = previous_episodes = 0
    for row in rows:
        step, completed = int(row["sampled_steps"]), int(row["completed_episodes"])
        if step <= previous_steps or completed < previous_episodes:
            raise RuntimeError("nonmonotonic training data")
        prepared.append((row, step, completed - previous_episodes, step - previous_steps))
        previous_steps, previous_episodes = step, completed
    phases = []
    for label, lo, hi in PHASES:
        selected = [item for item in prepared if lo < item[1] <= hi]
        if not selected:
            continue
        result = {"phase": label, "first_step": selected[0][1], "last_step": selected[-1][1],
                  "completed_episode_weight": sum(item[2] for item in selected), "updates": len(selected)}
        for field in rows[0]:
            if field in ("sampled_steps", "completed_episodes", "method_variant", "reward_mode", "environment_version"):
                continue
            values = [(float(item[0][field]), item) for item in selected if item[0].get(field) not in (None, "")]
            if not values:
                continue
            if field.startswith("own_") and "loss_count" in field:
                result[field] = sum(value for value, _ in values)
            else:
                episode_field = field.startswith("mean_") or field.endswith("_rate") or field == "MAV_survival_rate"
                weights = [item[2] if episode_field and field in (
                    "mean_episode_return", "mean_red_attack_kills", "mean_blue_attack_kills", "mean_UAV_survivors",
                    "mean_episode_length", "red_win_rate", "blue_win_rate", "draw_rate", "MAV_survival_rate") else item[3] for _, item in values]
                result[field] = float(np.average([v for v, _ in values], weights=weights)) if sum(weights) else None
        phases.append(result)
    early = next((p for p in phases if p["phase"] == "early_100352"), {})
    total = sum(early.get(f"own_loss_count_{aid}", 0) for aid in RED_IDS[1:])
    boundary = sum(early.get(f"own_boundary_loss_count_{aid}", 0) for aid in RED_IDS[1:])
    early_hypothesis = {"status": "INCONCLUSIVE" if not total else "VERIFIED" if boundary / total > .5 else "REFUTED",
                        "uav_own_losses": total, "uav_boundary_losses": boundary,
                        "boundary_share": boundary / total if total else None}
    late_bins = []
    for lo in range(1500000, 2000000, 100000):
        selected = [i for i in prepared if lo < i[1] <= lo + 100000]
        if selected and sum(i[2] for i in selected):
            late_bins.append({"start": lo, "end": lo + 100000,
                              **{key: float(np.average([float(i[0][key]) for i in selected], weights=[i[2] for i in selected]))
                                 for key in ("red_win_rate", "mean_red_attack_kills", "mean_episode_return")}})
    return {"metric_type": "training_window_not_evaluation", "weighting": "episode metrics: increments of completed_episodes; update diagnostics: sampled transitions; loss counts: sum",
            "last_observed_training_step": previous_steps, "phases": phases,
            "late_100k_bins": late_bins, "early_boundary_hypothesis": early_hypothesis}


def diagnosis(run, checkpoint, payload, invalid, training, log_paths):
    sources = {str(p): Path(p).read_text(encoding="utf-8", errors="replace") for p in log_paths if Path(p).exists()}
    text = "\n".join(sources.values())
    required = ("actors", "critic", "actor_optimizer_states", "critic_optimizer_state", "trainer_numpy_rng", "torch_rng", "cuda_rng", "rollout_state", "environment_config", "trainer_config")
    if payload.get("method_variant") == "tacm_rgaa":
        required += ("role_critic_mav", "role_critic_uav", "role_critic_mav_optimizer_state", "role_critic_uav_optimizer_state", "rgaa_numpy_rng")
    primary_log = (Path(run) / "run.log").read_text(encoding="utf-8", errors="replace")
    has_traceback = "Traceback (most recent call last)" in primary_log
    has_cuda_oom = bool(re.search(r"CUDA.*out of memory|OutOfMemoryError", primary_log, re.I))
    has_nonfinite = bool(re.search(r"\b(?:NaN|Inf)\b", primary_log, re.I))
    return {"run": str(run), "observed_training_csv_step": training["last_observed_training_step"],
            "last_complete_log_step": max([int(s.replace(",", "")) for s in re.findall(r"\[TRAIN\]\s+([\d,]+)", primary_log)] or [0]),
            "last_complete_log_time": "unavailable: progress lines contain no wall-clock timestamp",
            "checkpoint": str(checkpoint), "checkpoint_sampled_steps": int(payload["sampled_steps"]),
            "checkpoint_finite": finite_tree(payload), "checkpoint_state_fields": {k: k in payload for k in required},
            "invalid_checkpoints": invalid,
            "traceback": has_traceback,
            "cuda_oom_text": has_cuda_oom,
            "nonfinite_error_text": has_nonfinite,
            "last_logged_training_elapsed": (re.findall(r"elapsed\s+(\d\d:\d\d:\d\d)", primary_log) or [None])[-1],
            "system_oom_evidence": "unavailable: interrupted process ran in another WSL/server session; local kernel logs cannot establish its cause",
            "shell_termination_in_saved_logs": [line for line in text.splitlines() if "Terminated" in line],
            "shell_termination_user_report": "[1]+ Terminated / Terminated",
            "root_cause_status": "CUDA_OOM" if has_cuda_oom else "PYTHON_INTERNAL_EXCEPTION" if has_traceback else "UNRESOLVED_EXTERNAL_TERMINATION",
            "evidence": list(sources),
            "uncertainty": "Reported shell termination suggests external signal; exact signal, sender, Linux OOM and host shutdown cannot be distinguished from available artifacts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--master-log", type=Path)
    parser.add_argument("--training-log", type=Path)
    args = parser.parse_args()
    if args.episodes < 0 or (args.episodes and (args.device != "cuda" or not torch.cuda.is_available())):
        raise RuntimeError("replay requires CUDA; CPU fallback forbidden")
    run = args.run_dir.resolve()
    output = args.output.resolve()
    if output == run or run in output.parents:
        raise RuntimeError("audit output must be outside original run")
    if output.exists():
        raise FileExistsError("use a fresh audit output directory")
    checkpoint, payload, invalid = latest_valid_checkpoint(run)
    original_hash = digest(checkpoint)
    rows = read_rows(run / "training.csv")
    for field in ("environment_version", "method_variant", "reward_mode"):
        values = {r[field] for r in rows if r.get(field)}
        if values and values != {str(payload[field])}:
            raise RuntimeError(f"training/checkpoint contract mismatch: {field}")
    training = training_analysis(rows)
    if int(payload["sampled_steps"]) > training["last_observed_training_step"]:
        raise RuntimeError("checkpoint newer than training CSV")
    output.mkdir(parents=True)
    write_json(output / "training_audit.json", training)
    write_csv(output / "training_phases.csv", training["phases"])
    write_csv(output / "late_training_bins.csv", training["late_100k_bins"])
    logs = [run / "run.log", *[p for p in (args.master_log, args.training_log) if p]]
    diag = diagnosis(run, checkpoint, payload, invalid, training, logs)
    # A complete run is not an interrupted run.
    if int(payload["sampled_steps"]) == 2000000 and (run / "summary.json").exists():
        diag["root_cause_status"] = "COMPLETE"
        diag["shell_termination_user_report"] = None
        diag["uncertainty"] = "No termination evidence for this completed run."
    write_json(output / "diagnosis.json", diag)
    (output / "diagnosis.md").write_text("# Run diagnosis\n\n```json\n" + json.dumps(diag, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    if args.episodes:
        actors, config = load_actors(payload, args.device)
        before = {k: v.detach().clone() for k, v in actors.state_dict().items()}
        records = replay(actors, config, args.episodes, args.device, args.env_seed, args.action_seed)
        formal = summarize_records(records)
        # Same seeds, same sample order, same dtype: exact comparison, not tolerance.
        fidelity_episodes = min(args.episodes, 5)
        with torch.random.fork_rng(devices=[0]):
            reference = summarize_records(evaluate_actors(actors, config, fidelity_episodes, "main", seed=args.env_seed,
                                          device=args.device, deterministic=False, action_seed=args.action_seed))
        audit_reference = summarize_records(records[:fidelity_episodes])
        if audit_reference != reference:
            write_json(output / "fidelity_failure.json", {"audit": audit_reference, "evaluator": reference})
            raise AssertionError("replay fidelity failed; audit results must not be used")
        if not all(torch.equal(value, actors.state_dict()[key]) for key, value in before.items()):
            raise AssertionError("actor parameters changed")
        if args.episodes == 200 and args.env_seed == 1000 and args.action_seed == 2000 and int(payload["sampled_steps"]) == 2000000:
            existing = json.loads((run / "summary.json").read_text(encoding="utf-8"))["final_evaluations"][-1]
            expected_protocol = {"evaluation_profile": "main", "action_mode": "stochastic",
                                 "episodes": 200, "evaluation_environment_seed_start": 1000,
                                 "action_seed": 2000}
            if any(existing.get(key) != value for key, value in expected_protocol.items()):
                raise AssertionError("existing final evaluation protocol mismatch; audit unusable")
            for key in ("red_win_rate", "blue_win_rate", "draw_rate", "mean_episode_return", "mean_red_attack_kills", "MAV_survival_rate"):
                if not np.isclose(formal[key], existing[key], rtol=0, atol=1e-6):
                    raise AssertionError(f"existing final evaluation fidelity mismatch: {key}; audit unusable")
        deaths, cleanup = death_and_cleanup(records)
        write_csv(output / "episodes.csv", records)
        write_csv(output / "death_summary.csv", deaths)
        write_json(output / "cleanup_summary.json", cleanup)
        write_json(output / "evaluation_summary.json", {
            "checkpoint_step": int(payload["sampled_steps"]), "observed_training_step": training["last_observed_training_step"],
            "metric_type": "checkpoint_stochastic_evaluation", "protocol": {"profile": "main", "environment_version": VERSION,
            "episodes": args.episodes, "env_seed_start": args.env_seed, "env_seed_end": args.env_seed + args.episodes - 1,
            "action_seed_start": args.action_seed, "action_seed_end": args.action_seed + args.episodes - 1},
            "fidelity": "EXACT_PASS", "fidelity_reference_episodes": fidelity_episodes, "metrics": formal})
    if digest(checkpoint) != original_hash:
        raise AssertionError("checkpoint SHA changed")
    print(json.dumps({"diagnosis": diag, "early_deaths": training["early_boundary_hypothesis"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
