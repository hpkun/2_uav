"""Run a finite random-action audit over the auto-reset vector environment."""
from __future__ import annotations

import argparse
import numpy as np
from uav_combat import MAVUAVVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--num-envs", type=int, default=4); parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed)
    env = MAVUAVVectorEnv(args.num_envs, seed=args.seed); observations, states, masks, _ = env.reset()
    episodes = 0
    for _ in range(args.steps):
        observations, states, rewards, terminated, truncated, masks, infos = env.step(rng.uniform(-1.0, 1.0, (args.num_envs, 3, 3)))
        assert np.all(np.isfinite(observations)) and np.all(np.isfinite(states)) and np.all(np.isfinite(rewards))
        episodes += int(np.logical_or(terminated, truncated).sum())
    print({"vector_steps": args.steps, "sampled_environment_steps": args.steps * args.num_envs, "completed_episodes": episodes, "finite": True})


if __name__ == "__main__": main()

