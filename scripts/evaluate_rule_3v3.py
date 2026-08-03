"""Parallel rule baseline evaluation for 3v3."""
import argparse, json, time
from pathlib import Path
import numpy as np
from uav_combat.mappo.evaluation_3v3 import evaluate_rule_matchup_3v3

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
    for name, rm, bm in matchups:
        t0 = time.perf_counter()
        r = evaluate_rule_matchup_3v3(args.env_config, rm, bm, args.episodes,
                                        args.num_envs, args.env_workers, seed_start=args.seed_start,
                                        include_episode_details=args.include_episode_details)
        r["wall_seconds"] = time.perf_counter() - t0
        all_results[name] = r
        print(f"{name}: red_succ={r.get('red_complete_elimination_success_rate',0):.3f} "
              f"blue_succ={r.get('blue_complete_elimination_success_rate',0):.3f} "
              f"red_atk={r.get('mean_red_attack_kills',0):.2f} draw={r.get('draw_rate',0):.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = all_results[args.matchup] if args.matchup != "all" else all_results
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved to {out}")

if __name__ == "__main__": main()
