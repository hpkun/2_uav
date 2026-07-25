"""Rule baseline evaluation for 3v3 using episode_summary from info."""
import argparse, json
from pathlib import Path
import numpy as np
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS, DEATH_CAUSE_NAMES
from uav_combat.rule_policy_3v3 import NearestTargetPursuitPolicy3v3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--episodes", type=int, default=100)
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
        episode_summaries = []
        for ep in range(args.episodes):
            env.reset(seed=1000 + ep)
            while True:
                reds = [a for a in env.aircraft if a.team == "red"]
                blues = [a for a in env.aircraft if a.team == "blue"]
                alive_reds = [r for r in reds if r.state.alive]
                alive_blues = [b for b in blues if b.state.alive]

                red_actions_dict = {}
                if red_type == "zero":
                    red_actions_dict = {r.aircraft_id: np.zeros(3, dtype=np.float32) for r in alive_reds}
                else:
                    ra, _ = pursuit.select_actions(reds, blues)
                    red_actions_dict = {aid: ra.get(aid, np.zeros(3, dtype=np.float32)) for aid in RED_IDS}

                blue_actions_dict = {}
                if blue_type == "zero":
                    blue_actions_dict = {b.aircraft_id: np.zeros(3, dtype=np.float32) for b in alive_blues}
                else:
                    ba, _ = pursuit.select_actions(blues, reds)
                    blue_actions_dict = {aid: ba.get(aid, np.zeros(3, dtype=np.float32)) for aid in BLUE_IDS}

                actions = {}
                for a in env.aircraft:
                    if a.team == "red":
                        actions[a.aircraft_id] = red_actions_dict.get(a.aircraft_id, np.zeros(3, dtype=np.float32))
                    else:
                        actions[a.aircraft_id] = blue_actions_dict.get(a.aircraft_id, np.zeros(3, dtype=np.float32))

                obs, rewards, term, trunc, info = env.step(actions)
                if term or trunc:
                    es = info.get("episode_summary")
                    if es is None:
                        raise RuntimeError("episode_summary missing from info at episode end")
                    # Validate death ledger
                    for team, side in (("red", RED_IDS), ("blue", BLUE_IDS)):
                        dc = es[f"{team}_death_causes"]
                        total = es[f"{team}_survivors"] + dc["attack_deaths"] + dc["boundary_deaths"] + dc["collision_deaths"]
                        if total != 3:
                            raise RuntimeError(f"Death ledger mismatch in {name} ep={ep} {team}: {total} != 3")
                    # Validate attack kill symmetry
                    if es["red_attack_kills"] != es["blue_death_causes"]["attack_deaths"]:
                        raise RuntimeError(f"red_attack_kills={es['red_attack_kills']} != blue.attack_deaths={es['blue_death_causes']['attack_deaths']}")
                    if es["blue_attack_kills"] != es["red_death_causes"]["attack_deaths"]:
                        raise RuntimeError(f"blue_attack_kills={es['blue_attack_kills']} != red.attack_deaths={es['red_death_causes']['attack_deaths']}")
                    episode_summaries.append(es)
                    break

            if (ep + 1) % 25 == 0:
                print(f"  {name}: {ep+1}/{args.episodes}", flush=True)

        n = len(episode_summaries)
        results = {
            "episodes": n,
            "red_complete_elimination_success_rate": sum(1 for s in episode_summaries if s["red_complete_elimination_success"]) / n,
            "blue_complete_elimination_success_rate": sum(1 for s in episode_summaries if s["blue_complete_elimination_success"]) / n,
            "red_outcome_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "red") / n,
            "blue_outcome_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "blue") / n,
            "draw_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "draw") / n,
            "mean_red_attack_kills": float(np.mean([s["red_attack_kills"] for s in episode_summaries])),
            "mean_blue_attack_kills": float(np.mean([s["blue_attack_kills"] for s in episode_summaries])),
            "mean_red_survivors": float(np.mean([s["red_survivors"] for s in episode_summaries])),
            "mean_blue_survivors": float(np.mean([s["blue_survivors"] for s in episode_summaries])),
            "mean_red_boundary_deaths": float(np.mean([s["red_death_causes"]["boundary_deaths"] for s in episode_summaries])),
            "mean_blue_boundary_deaths": float(np.mean([s["blue_death_causes"]["boundary_deaths"] for s in episode_summaries])),
            "mean_red_collision_deaths": float(np.mean([s["red_death_causes"]["collision_deaths"] for s in episode_summaries])),
            "mean_blue_collision_deaths": float(np.mean([s["blue_death_causes"]["collision_deaths"] for s in episode_summaries])),
            "mean_episode_length": float(np.mean([s["episode_length"] for s in episode_summaries])),
            "max_steps_rate": sum(1 for s in episode_summaries if s["termination_reason"] == "max_steps") / n,
        }
        all_results[name] = results
        print(f"  {name}: red_succ={results['red_complete_elimination_success_rate']:.3f} "
              f"blue_succ={results['blue_complete_elimination_success_rate']:.3f} "
              f"red_out={results['red_outcome_rate']:.3f} draw={results['draw_rate']:.3f} "
              f"red_atk_kills={results['mean_red_attack_kills']:.2f}")

    output_dir = Path("outputs/3v3_env_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rule_baselines.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {output_dir / 'rule_baselines.json'}")

if __name__ == "__main__":
    main()
