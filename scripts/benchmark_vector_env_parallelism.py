"""Compare serial and multi-process MAV/UAV environment throughput."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from uav_combat import MAVUAVVectorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--warmup-vector-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.sample_steps <= 0 or args.warmup_vector_steps < 0:
        raise ValueError("num-envs and sample-steps must be positive; warmup must be non-negative")
    if args.sample_steps % args.num_envs:
        raise ValueError("sample-steps must be divisible by num-envs")
    vector_steps = args.sample_steps // args.num_envs
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(
        -1.0, 1.0,
        (vector_steps + args.warmup_vector_steps, args.num_envs, 3, 3),
    )
    results: dict[str, object] = {
        "logical_cpu_count": os.cpu_count(),
        "num_envs": args.num_envs,
        "sampled_steps": args.sample_steps,
    }
    timings: dict[str, dict[str, object]] = {}
    for label, parallel in (("serial", False), ("multiprocess", True)):
        with MAVUAVVectorEnv(args.num_envs, seed=args.seed, parallel=parallel) as env:
            env.reset()
            for action in actions[:args.warmup_vector_steps]:
                env.step(action)
            start = time.perf_counter()
            for action in actions[args.warmup_vector_steps:]:
                env.step(action)
            elapsed = time.perf_counter() - start
            timings[label] = {
                "elapsed_seconds": elapsed,
                "sampled_steps_per_second": args.sample_steps / elapsed,
                "worker_pids": env.worker_pids,
                "start_method": env.start_method,
            }
    results["timings"] = timings
    results["speedup"] = (
        float(timings["serial"]["elapsed_seconds"])
        / float(timings["multiprocess"]["elapsed_seconds"])
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
