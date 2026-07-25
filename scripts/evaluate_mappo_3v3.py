"""Evaluate a 3v3 MAPPO checkpoint against fixed-rule blue."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer_3v3 import CHECKPOINT_FAMILY, CHECKPOINT_VERSION_3V3, OBS_DIM, resolve_device
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS, DEATH_ATTACK
from uav_combat.rule_policy_3v3 import NearestTargetPursuitPolicy3v3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--env-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY:
        raise RuntimeError(f"Expected {CHECKPOINT_FAMILY}, got {ckpt.get('checkpoint_family')}")
    n_cfg = ckpt["config"]["network"]
    actor = GaussianActor(OBS_DIM, 3, n_cfg["hidden_dim"], n_cfg["log_std_init"]).to(device)
    actor.load_state_dict(ckpt["shared_red_actor"])
    actor.eval()

    env = Homogeneous3v3AirCombatEnv(args.env_config)
    act_cfg = env.config["action"]
    blue_policy = NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])

    records = []
    for ep in range(args.episodes):
        env.reset(seed=10000 + ep)
        ep_returns = 0.0
        while True:
            reds = [a for a in env.aircraft if a.team == "red"]
            blues = [a for a in env.aircraft if a.team == "blue"]
            blue_refs = blues
            red_refs = reds
            alive_reds = [r for r in reds if r.state.alive]
            alive_blues = [b for b in blues if b.state.alive]

            # Red actions
            red_obs_list = [env._agent_observation(r) for r in reds]
            red_obs = np.stack(red_obs_list, axis=0)
            with torch.no_grad():
                red_actions = actor.deterministic_action(
                    torch.as_tensor(red_obs, device=device)).cpu().numpy()

            # Blue actions
            blue_actions_dict, _ = blue_policy.select_actions(blues, reds)

            actions = {}
            for j, aid in enumerate(RED_IDS):
                actions[aid] = red_actions[j].astype(np.float32)
            for aid in BLUE_IDS:
                actions[aid] = blue_actions_dict.get(aid, np.zeros(3, dtype=np.float32))

            obs, rewards, term, trunc, info = env.step(actions)
            ep_returns += rewards["red_0"]
            if term or trunc:
                records.append({
                    **info,
                    "episode_return": float(ep_returns),
                    "episode_length": info["step_count"],
                })
                break

    n = len(records)
    def rate(key):
        return sum(1 for r in records if r.get(key)) / n if n else 0.0

    results = {
        "episodes": n,
        "red_complete_elimination_success_rate": rate("red_complete_elimination_success"),
        "blue_complete_elimination_success_rate": rate("blue_complete_elimination_success"),
        "environment_red_outcome_rate": sum(1 for r in records if r["outcome"] == "red") / n,
        "environment_blue_outcome_rate": sum(1 for r in records if r["outcome"] == "blue") / n,
        "draw_rate": sum(1 for r in records if r["outcome"] == "draw") / n,
        "mean_red_attack_kills": np.mean([r["attack_kills"]["red"] for r in records]),
        "mean_blue_attack_kills": np.mean([r["attack_kills"]["blue"] for r in records]),
        "mean_red_survivors": np.mean([r["red_survivors"] for r in records]),
        "mean_blue_survivors": np.mean([r["blue_survivors"] for r in records]),
        "red_kd_numerator": sum(r["attack_kills"]["red"] for r in records),
        "red_kd_denominator": sum(r["attack_kills"]["blue"] for r in records),
        "red_kd_ratio": (sum(r["attack_kills"]["red"] for r in records) / sum(r["attack_kills"]["blue"] for r in records)) if sum(r["attack_kills"]["blue"] for r in records) > 0 else None,
        "red_boundary_death_rate": np.mean([r["boundary_deaths"]["red"] for r in records]) / 3,
        "blue_boundary_death_rate": np.mean([r["boundary_deaths"]["blue"] for r in records]) / 3,
        "friendly_collision_rate": sum(1 for r in records for a1, a2 in r.get("collision_pairs", []) if a1.split("_")[0] == a2.split("_")[0]) / n if n else 0,
        "cross_team_collision_rate": sum(1 for r in records for a1, a2 in r.get("collision_pairs", []) if a1.split("_")[0] != a2.split("_")[0]) / n if n else 0,
        "max_steps_rate": sum(1 for r in records if r["termination_reason"] == "max_steps") / n,
        "mean_episode_length": np.mean([r["episode_length"] for r in records]),
        "mean_team_return": np.mean([r["episode_return"] for r in records]),
    }
    print(json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x))

if __name__ == "__main__":
    main()
