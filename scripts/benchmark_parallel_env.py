"""Benchmark LocalCombatVectorEnv vs SubprocessCombatVectorEnv throughput.

Reports wall time, env steps/s, and result consistency (not speedup assertions).
"""
import argparse
import time
from pathlib import Path
import numpy as np

from uav_combat.mappo.vector_env import (
    LocalCombatVectorEnv,
    SubprocessCombatVectorEnv,
)


def _generate_specs(num_envs: int, seed: int = 42):
    """Deterministic reset specs cycling scenarios and rear teams."""
    scenarios = ["tail_chase", "offset_head_on", "crossing"]
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(num_envs):
        scenario = scenarios[i % 3]
        rear = "red" if i % 2 == 0 else "blue" if scenario == "tail_chase" else None
        specs.append(
            {"seed": int(rng.integers(0, 2**31 - 1)), "scenario": scenario, "rear_team": rear}
        )
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_1v1.yaml")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--env-workers", type=int, default=4)
    p.add_argument("--vector-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    config_path = Path(args.env_config)
    num_envs = args.num_envs
    num_workers = args.env_workers
    num_steps = args.vector_steps
    seed = args.seed

    print(
        f"num_envs={num_envs} env_workers={num_workers} vector_steps={num_steps}"
    )

    rng = np.random.default_rng(seed)

    # --- Local (sequential) ---
    print("\n--- LocalCombatVectorEnv (sequential baseline) ---")
    local_env = LocalCombatVectorEnv(str(config_path), num_envs)
    try:
        specs = _generate_specs(num_envs, seed)
        t0 = time.perf_counter()
        obs, gs = local_env.reset(specs)
        t_reset_local = time.perf_counter() - t0

        t_step = 0.0
        local_episodes = 0
        for step_idx in range(num_steps):
            actions = rng.uniform(-1, 1, (num_envs, 2, 3)).astype(np.float32)
            t1 = time.perf_counter()
            (
                obs,
                gs,
                rewards,
                terminated,
                truncated,
                step_counts,
                attacks,
                geometry,
                control_diag,
                reason_codes,
                outcome_codes,
            ) = local_env.step(actions)
            t_step += time.perf_counter() - t1
            done_mask = terminated | truncated
            local_episodes += int(done_mask.sum())
            if done_mask.any():
                done_indices = np.where(done_mask)[0]
                new_specs = _generate_specs(len(done_indices), seed + 1000 + step_idx)
                new_obs, new_gs = local_env.reset_at(done_indices, new_specs)
                obs[done_indices] = new_obs
                gs[done_indices] = new_gs
        local_wall = time.perf_counter() - t0
        local_steps_total = num_envs * num_steps
        local_sps = local_steps_total / local_wall if local_wall > 0 else 0.0

        # Save local results for consistency check
        local_obs = obs.copy()
        local_rewards = rewards.copy()
        local_dones = terminated | truncated
        local_reason = reason_codes.copy()
        local_outcome = outcome_codes.copy()
    finally:
        local_env.close()

    print(
        f"  wall={local_wall:.2f}s  env_steps/s={local_sps:.1f}  "
        f"reset={t_reset_local:.3f}s  step={t_step:.2f}s  "
        f"completed_episodes={local_episodes}"
    )

    # --- Subprocess (parallel) ---
    print(f"\n--- SubprocessCombatVectorEnv ({num_workers} workers) ---")
    parallel_env = SubprocessCombatVectorEnv(str(config_path), num_envs, num_workers)
    try:
        specs = _generate_specs(num_envs, seed)
        t0 = time.perf_counter()
        obs_p, gs_p = parallel_env.reset(specs)
        t_reset_parallel = time.perf_counter() - t0

        t_step_p = 0.0
        parallel_episodes = 0
        for step_idx in range(num_steps):
            actions = rng.uniform(-1, 1, (num_envs, 2, 3)).astype(np.float32)
            t1 = time.perf_counter()
            (
                obs_p,
                gs_p,
                rewards_p,
                terminated_p,
                truncated_p,
                step_counts_p,
                attacks_p,
                geometry_p,
                control_diag_p,
                reason_codes_p,
                outcome_codes_p,
            ) = parallel_env.step(actions)
            t_step_p += time.perf_counter() - t1
            done_mask_p = terminated_p | truncated_p
            parallel_episodes += int(done_mask_p.sum())
            if done_mask_p.any():
                done_indices_p = np.where(done_mask_p)[0]
                new_specs_p = _generate_specs(
                    len(done_indices_p), seed + 1000 + step_idx
                )
                new_obs_p, new_gs_p = parallel_env.reset_at(done_indices_p, new_specs_p)
                obs_p[done_indices_p] = new_obs_p
                gs_p[done_indices_p] = new_gs_p
        parallel_wall = time.perf_counter() - t0
        parallel_sps = (
            num_envs * num_steps / parallel_wall if parallel_wall > 0 else 0.0
        )

        # Consistency check
        obs_match = np.allclose(local_obs, obs_p, atol=1e-5)
        reward_match = np.allclose(local_rewards, rewards_p, atol=1e-5)
        dones_match = np.array_equal(local_dones, terminated_p | truncated_p)
        reason_match = np.array_equal(local_reason, reason_codes_p)
        outcome_match = np.array_equal(local_outcome, outcome_codes_p)
        all_match = obs_match and reward_match and dones_match and reason_match and outcome_match
    finally:
        parallel_env.close()

    speedup = parallel_sps / local_sps if local_sps > 0 else 0.0
    print(
        f"  wall={parallel_wall:.2f}s  env_steps/s={parallel_sps:.1f}  "
        f"reset={t_reset_parallel:.3f}s  step={t_step_p:.2f}s  "
        f"completed_episodes={parallel_episodes}"
    )

    print(f"\n--- Summary ---")
    print(f"  local   : {local_sps:.1f} env steps/s  ({local_wall:.2f}s)")
    print(f"  parallel: {parallel_sps:.1f} env steps/s  ({parallel_wall:.2f}s)")
    print(f"  speedup : {speedup:.2f}x")
    print(
        f"  consistency: obs={obs_match} reward={reward_match} "
        f"dones={dones_match} reason={reason_match} outcome={outcome_match}"
    )
    if speedup < 1.0:
        print(
            "  NOTE: No speedup on this machine. Overhead dominates or CPU cores "
            "are fully loaded. This is expected on some hardware; "
            "the result is reported honestly."
        )


if __name__ == "__main__":
    main()
