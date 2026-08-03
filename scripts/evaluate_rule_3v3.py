"""Parallel rule baseline evaluation for 3v3."""
import argparse, json, time
from pathlib import Path
import numpy as np
from uav_combat.config import load_config
from uav_combat.main_experiment_v8 import (
    build_main_v8_contract_metadata,
    filter_public_metrics_for_config,
    is_main_v8_config,
    sha256_file,
)
from uav_combat.mappo.evaluation_3v3 import evaluate_rule_matchup_3v3


def _classification(result):
    red_one = float(result.get("red_at_least_one_attack_kill_rate", 0.0))
    blue_one = float(result.get("blue_at_least_one_attack_kill_rate", 0.0))
    red_two = float(result.get("red_at_least_two_attack_kill_rate", 0.0))
    blue_two = float(result.get("blue_at_least_two_attack_kill_rate", 0.0))
    complete = max(
        float(result.get("red_complete_elimination_success_rate", 0.0)),
        float(result.get("blue_complete_elimination_success_rate", 0.0)),
    )
    max_steps = float(result.get("max_steps_rate", 0.0))
    boundary = max(
        float(result.get("mean_red_boundary_deaths", 0.0)),
        float(result.get("mean_blue_boundary_deaths", 0.0)),
    )
    if boundary >= 1.0:
        return "C"
    if max(red_two, blue_two) > 0.05 or complete > 0.0:
        return "A2"
    if max(red_one, blue_one) > 0.05 and max_steps > 0.5:
        return "A1"
    if max(red_one, blue_one) <= 0.05 and max_steps > 0.5:
        return "B"
    return "A1"


def _markdown_report(payload):
    result = payload["result"]
    lines = [
        "# Homogeneous v8 rule-vs-rule attackability report",
        "",
        f"- config: `{payload['env_config']}`",
        f"- config_sha256: `{payload['env_config_sha256']}`",
        f"- episodes: {payload['episodes']}",
        f"- seed_start: {payload['seed_start']}",
        f"- num_envs/env_workers: {payload['num_envs']} / {payload['env_workers']}",
        f"- red_policy: `{payload['red_policy']}`",
        f"- blue_policy: `{payload['blue_policy']}`",
        f"- classification: **{payload['classification']}**",
        "",
        "## Outcomes",
        "",
        f"- environment_red_outcome_rate: {result.get('environment_red_outcome_rate')}",
        f"- environment_blue_outcome_rate: {result.get('environment_blue_outcome_rate')}",
        f"- draw_rate: {result.get('draw_rate')}",
        f"- neutral_rule_red_win_rate: {result.get('neutral_rule_red_win_rate')}",
        f"- neutral_rule_blue_win_rate: {result.get('neutral_rule_blue_win_rate')}",
        f"- neutral_rule_draw_rate: {result.get('neutral_rule_draw_rate')}",
        "",
        "## Attack statistics",
        "",
        f"- red_any_attack_kill_rate: {result.get('red_any_attack_kill_rate')}",
        f"- blue_any_attack_kill_rate: {result.get('blue_any_attack_kill_rate')}",
        f"- red_attack_kill_count_distribution: `{result.get('red_attack_kill_count_distribution')}`",
        f"- blue_attack_kill_count_distribution: `{result.get('blue_attack_kill_count_distribution')}`",
        f"- mean_red_first_attack_kill_step: {result.get('mean_red_first_attack_kill_step')}",
        f"- mean_blue_first_attack_kill_step: {result.get('mean_blue_first_attack_kill_step')}",
        f"- mean_red_remaining_steps_after_first_kill: {result.get('mean_red_remaining_steps_after_first_kill')}",
        f"- mean_blue_remaining_steps_after_first_kill: {result.get('mean_blue_remaining_steps_after_first_kill')}",
        "",
        "## Tactical-window rates",
        "",
        f"- red_r3_active_step_rate: {result.get('red_r3_active_step_rate')}",
        f"- blue_r3_active_step_rate: {result.get('blue_r3_active_step_rate')}",
        f"- red_r41_active_step_rate: {result.get('red_r41_active_step_rate')}",
        f"- blue_r41_active_step_rate: {result.get('blue_r41_active_step_rate')}",
        f"- red_r42_active_step_rate: {result.get('red_r42_active_step_rate')}",
        f"- blue_r42_active_step_rate: {result.get('blue_r42_active_step_rate')}",
        f"- red_attack_window_step_rate: {result.get('red_attack_window_step_rate')}",
        f"- blue_attack_window_step_rate: {result.get('blue_attack_window_step_rate')}",
        "",
        "This report is generated from the JSON payload. It diagnoses attackability only; it does not justify changing rewards, attack ranges, angle gates, or max_steps.",
    ]
    return "\n".join(lines) + "\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--env-workers", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--output", default="outputs/3v3_env_audit/rule_baselines.json")
    p.add_argument(
        "--matchup",
        choices=("all", "pursuit_vs_pursuit", "zero_vs_pursuit", "pursuit_vs_zero"),
        default="all",
    )
    p.add_argument("--include-episode-details", action="store_true")
    args = p.parse_args()

    matchups = [("pursuit_vs_pursuit","pursuit","pursuit"),
                ("zero_vs_pursuit","zero","pursuit"),
                ("pursuit_vs_zero","pursuit","zero")]
    if args.matchup != "all":
        matchups = [m for m in matchups if m[0] == args.matchup]
    all_results = {}
    env_cfg = load_config(args.env_config)
    for name, rm, bm in matchups:
        t0 = time.perf_counter()
        r = evaluate_rule_matchup_3v3(args.env_config, rm, bm, args.episodes,
                                        args.num_envs, args.env_workers, seed_start=args.seed_start,
                                        include_episode_details=args.include_episode_details)
        r["wall_seconds"] = time.perf_counter() - t0
        all_results[name] = r
        print(f"{name}: neutral_red={r.get('neutral_rule_red_win_rate',0):.3f} "
              f"neutral_blue={r.get('neutral_rule_blue_win_rate',0):.3f} "
              f"neutral_draw={r.get('neutral_rule_draw_rate',0):.3f} "
              f"red_atk={r.get('mean_red_attack_kills',0):.2f} blue_atk={r.get('mean_blue_attack_kills',0):.2f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.matchup != "all":
        result = filter_public_metrics_for_config(all_results[args.matchup], env_cfg)
        payload = {
            "environment_contract": build_main_v8_contract_metadata(env_cfg) if is_main_v8_config(env_cfg) else None,
            "env_config": args.env_config,
            "env_config_sha256": sha256_file(args.env_config),
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "num_envs": args.num_envs,
            "env_workers": args.env_workers,
            "red_policy": "paper_nearest_pursuit_v1" if "pursuit" in args.matchup else "zero",
            "blue_policy": "paper_nearest_pursuit_v1" if "pursuit" in args.matchup else "zero",
            "classification": _classification(result),
            "result": result,
        }
    else:
        payload = {name: filter_public_metrics_for_config(result, env_cfg) for name, result in all_results.items()}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if out.suffix.lower() == ".json" and args.matchup != "all":
        out.with_suffix(".md").write_text(_markdown_report(payload), encoding="utf-8")
    print(f"Saved to {out}")

if __name__ == "__main__": main()
