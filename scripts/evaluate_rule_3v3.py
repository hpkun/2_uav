"""Rule baseline evaluation for 3v3: pursuit vs pursuit, zero vs pursuit, pursuit vs zero."""
import argparse, json
from pathlib import Path
import numpy as np
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS
from uav_combat.rule_policy_3v3 import NearestTargetPursuitPolicy3v3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--env-workers", type=int, default=1)
    args = p.parse_args()

    env = Homogeneous3v3AirCombatEnv(args.env_config)
    cfg = env.config["action"]
    pursuit = NearestTargetPursuitPolicy3v3(cfg["delta_yaw_max"], cfg["delta_pitch_max"], cfg["delta_speed_max"])

    matchups = [
        ("pursuit_vs_pursuit", "pursuit", "pursuit"),
        ("zero_vs_pursuit", "zero", "pursuit"),
        ("pursuit_vs_zero", "pursuit", "zero"),
    ]
    all_results = {}

    for name, red_type, blue_type in matchups:
        records = []
        for ep in range(args.episodes):
            env.reset(seed=1000 + ep)
            while True:
                reds = [a for a in env.aircraft if a.team == "red"]
                blues = [a for a in env.aircraft if a.team == "blue"]
                alive_reds = [r for r in reds if r.state.alive]
                alive_blues = [b for b in blues if b.state.alive]

                # Red actions
                red_actions = {}
                if red_type == "zero":
                    red_actions = {r.aircraft_id: np.zeros(3, dtype=np.float32) for r in alive_reds}
                else:
                    ra, _ = pursuit.select_actions(reds, blues)
                    red_actions = {aid: ra.get(aid, np.zeros(3, dtype=np.float32)) for aid in RED_IDS}

                # Blue actions
                blue_actions = {}
                if blue_type == "zero":
                    blue_actions = {b.aircraft_id: np.zeros(3, dtype=np.float32) for b in alive_blues}
                else:
                    ba, _ = pursuit.select_actions(blues, reds)
                    blue_actions = {aid: ba.get(aid, np.zeros(3, dtype=np.float32)) for aid in BLUE_IDS}

                actions = {}
                for a in env.aircraft:
                    if a.team == "red":
                        actions[a.aircraft_id] = red_actions.get(a.aircraft_id, np.zeros(3, dtype=np.float32))
                    else:
                        actions[a.aircraft_id] = blue_actions.get(a.aircraft_id, np.zeros(3, dtype=np.float32))

                obs, rewards, term, trunc, info = env.step(actions)
                if term or trunc:
                    records.append(info)
                    break

            if (ep + 1) % 10 == 0:
                print(f"  {name}: {ep+1}/{args.episodes}", flush=True)

        n = len(records)
        results = {
            "episodes": n,
            "red_complete_elimination_success_rate": sum(1 for r in records if r.get("red_complete_elimination_success")) / n,
            "blue_complete_elimination_success_rate": sum(1 for r in records if r.get("blue_complete_elimination_success")) / n,
            "red_outcome_rate": sum(1 for r in records if r["outcome"] == "red") / n,
            "blue_outcome_rate": sum(1 for r in records if r["outcome"] == "blue") / n,
            "draw_rate": sum(1 for r in records if r["outcome"] == "draw") / n,
            "mean_red_attack_kills": float(np.mean([r["attack_kills"]["red"] for r in records])),
            "mean_blue_attack_kills": float(np.mean([r["attack_kills"]["blue"] for r in records])),
            "mean_red_survivors": float(np.mean([r["red_survivors"] for r in records])),
            "mean_blue_survivors": float(np.mean([r["blue_survivors"] for r in records])),
            "mean_red_boundary_deaths": float(np.mean([r["boundary_deaths"]["red"] for r in records])),
            "mean_blue_boundary_deaths": float(np.mean([r["boundary_deaths"]["blue"] for r in records])),
            "mean_red_collision_deaths": float(np.mean([r["collision_deaths"]["red"] for r in records])),
            "mean_blue_collision_deaths": float(np.mean([r["collision_deaths"]["blue"] for r in records])),
            "max_steps_rate": sum(1 for r in records if r["termination_reason"] == "max_steps") / n,
            "mean_episode_length": float(np.mean([r["step_count"] for r in records])),
            "blue_focus_fire_count": pursuit.focus_fire_count,
        }
        all_results[name] = results
        print(f"  {name}: red_success={results['red_complete_elimination_success_rate']:.3f} "
              f"blue_success={results['blue_complete_elimination_success_rate']:.3f} draw={results['draw_rate']:.3f}")

    output_dir = Path("outputs/3v3_env_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rule_baselines.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {output_dir / 'rule_baselines.json'}")

if __name__ == "__main__":
    main()
