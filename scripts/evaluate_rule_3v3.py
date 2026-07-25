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
    args = p.parse_args()

    matchups = [("pursuit_vs_pursuit","pursuit","pursuit"),
                ("zero_vs_pursuit","zero","pursuit"),
                ("pursuit_vs_zero","pursuit","zero")]
    all_results = {}
    for name, rm, bm in matchups:
        t0 = time.perf_counter()
        r = evaluate_rule_matchup_3v3(args.env_config, rm, bm, args.episodes,
                                        args.num_envs, args.env_workers, seed_start=1000)
        r["wall_seconds"] = time.perf_counter() - t0
        all_results[name] = r
        print(f"{name}: red_succ={r.get('red_complete_elimination_success_rate',0):.3f} "
              f"blue_succ={r.get('blue_complete_elimination_success_rate',0):.3f} "
              f"red_atk={r.get('mean_red_attack_kills',0):.2f} draw={r.get('draw_rate',0):.3f}")

    out = Path("outputs/3v3_env_audit"); out.mkdir(parents=True, exist_ok=True)
    (out / "rule_baselines.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Saved to {out / 'rule_baselines.json'}")

if __name__ == "__main__": main()
