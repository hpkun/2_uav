"""Evaluation helpers for functional heterogeneous red 4v3 HAPPO."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..environment_4v3 import FunctionalHeterogeneous4v3AirCombatEnv
from ..mappo.trainer_3v3 import resolve_device
from ..mappo.vector_env_4v3 import RED_TEAM_SIZE_4V3, make_combat_vector_env_4v3
from .networks import IndependentHAPPOActors
from .trainer_4v3 import summarize_4v3_episodes


@torch.no_grad()
def evaluate_happo_fixed_blue_4v3(
    actors: IndependentHAPPOActors,
    env_config: str | Path,
    *,
    episodes: int = 8,
    num_envs: int = 4,
    num_env_workers: int = 0,
    device: str | torch.device = "cpu",
    seed: int = 10000,
) -> dict[str, float]:
    dev = resolve_device(str(device))
    was_training = actors.training
    actors.eval()
    vec = make_combat_vector_env_4v3(env_config, num_envs, num_env_workers, seed)
    obs, _, _ = vec.reset()
    seed_rng = np.random.default_rng(int(seed) + 1)
    records: list[dict[str, Any]] = []
    try:
        while len(records) < int(episodes):
            red_obs = torch.as_tensor(obs[:, :RED_TEAM_SIZE_4V3, :], dtype=torch.float32, device=dev)
            actions = actors.deterministic_actions(red_obs).cpu().numpy().astype(np.float32)
            result = vec.step(actions)
            obs = result.observations
            for i, summary in enumerate(result.episode_summaries):
                if summary is not None:
                    records.append(summary)
                    obs[i], _, _ = vec.reset_at(i, int(seed_rng.integers(0, 2**31 - 1)))
                    if len(records) >= int(episodes):
                        break
        return summarize_4v3_episodes(records[: int(episodes)])
    finally:
        vec.close()
        if was_training:
            actors.train()


def evaluate_rule_vs_rule_4v3(env_config: str | Path, *, episodes: int = 100, seed: int = 20000) -> dict[str, float]:
    records: list[dict[str, Any]] = []
    for ep in range(int(episodes)):
        env = FunctionalHeterogeneous4v3AirCombatEnv(env_config)
        env.reset(seed + ep)
        done = False
        while not done:
            red_actions, _ = env.red_rule_actions()
            _, _, _, _, done, _, info = env.step(red_actions)
        records.append(info["episode_summary"])
    return summarize_4v3_episodes(records)


__all__ = ["evaluate_happo_fixed_blue_4v3", "evaluate_rule_vs_rule_4v3"]
