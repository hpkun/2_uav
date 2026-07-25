"""Evaluate a 3v3 MAPPO checkpoint against fixed-rule blue."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer_3v3 import CHECKPOINT_FAMILY, OBS_DIM, resolve_device
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS
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

    episode_summaries = []
    for ep in range(args.episodes):
        env.reset(seed=10000 + ep)
        ep_return = 0.0
        while True:
            reds = [a for a in env.aircraft if a.team == "red"]
            blues = [a for a in env.aircraft if a.team == "blue"]

            red_obs_list = [env._agent_observation(r) for r in reds]
            red_obs = np.stack(red_obs_list, axis=0)
            with torch.no_grad():
                red_actions = actor.deterministic_action(
                    torch.as_tensor(red_obs, device=device)).cpu().numpy()
            blue_actions_dict, _ = blue_policy.select_actions(blues, reds)

            actions = {}
            for j, aid in enumerate(RED_IDS):
                actions[aid] = red_actions[j].astype(np.float32)
            for aid in BLUE_IDS:
                actions[aid] = blue_actions_dict.get(aid, np.zeros(3, dtype=np.float32))

            obs, rewards, term, trunc, info = env.step(actions)
            ep_return += rewards["red_0"]
            if term or trunc:
                es = info.get("episode_summary")
                if es is None:
                    raise RuntimeError("episode_summary missing")
                # Validate
                for team in ("red", "blue"):
                    dc = es[f"{team}_death_causes"]
                    total = es[f"{team}_survivors"] + dc["attack_deaths"] + dc["boundary_deaths"] + dc["collision_deaths"]
                    if total != 3:
                        raise RuntimeError(f"Death ledger mismatch ep={ep} {team}: {total} != 3")
                es["episode_return"] = float(ep_return)
                episode_summaries.append(es)
                break

    n = len(episode_summaries)
    def rate(key):
        return sum(1 for s in episode_summaries if s.get(key)) / n if n else 0.0

    red_kills = sum(s["red_attack_kills"] for s in episode_summaries)
    blue_kills = sum(s["blue_attack_kills"] for s in episode_summaries)

    results = {
        "episodes": n,
        "red_complete_elimination_success_rate": rate("red_complete_elimination_success"),
        "blue_complete_elimination_success_rate": rate("blue_complete_elimination_success"),
        "environment_red_outcome_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "red") / n,
        "environment_blue_outcome_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "blue") / n,
        "draw_rate": sum(1 for s in episode_summaries if s["environment_outcome"] == "draw") / n,
        "mean_red_attack_kills": float(np.mean([s["red_attack_kills"] for s in episode_summaries])),
        "mean_blue_attack_kills": float(np.mean([s["blue_attack_kills"] for s in episode_summaries])),
        "mean_red_survivors": float(np.mean([s["red_survivors"] for s in episode_summaries])),
        "mean_blue_survivors": float(np.mean([s["blue_survivors"] for s in episode_summaries])),
        "red_kd_numerator": red_kills,
        "red_kd_denominator": blue_kills,
        "red_kd_ratio": red_kills / blue_kills if blue_kills > 0 else None,
        "mean_red_boundary_deaths": float(np.mean([s["red_death_causes"]["boundary_deaths"] for s in episode_summaries])),
        "mean_blue_boundary_deaths": float(np.mean([s["blue_death_causes"]["boundary_deaths"] for s in episode_summaries])),
        "mean_red_collision_deaths": float(np.mean([s["red_death_causes"]["collision_deaths"] for s in episode_summaries])),
        "mean_blue_collision_deaths": float(np.mean([s["blue_death_causes"]["collision_deaths"] for s in episode_summaries])),
        "max_steps_rate": sum(1 for s in episode_summaries if s["termination_reason"] == "max_steps") / n,
        "mean_episode_length": float(np.mean([s["episode_length"] for s in episode_summaries])),
        "mean_team_return": float(np.mean([s["episode_return"] for s in episode_summaries])),
    }
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
