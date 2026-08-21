"""Deterministic HAPPO evaluation for one fixed Blue target mode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from ..mavuav import HeterogeneousMAVUAVAirCombatEnv
from .networks import IndependentActors
from ..mappo.evaluation import summarize_records


def evaluate_actors(actors: IndependentActors, env_config: str | Path | Mapping[str, Any] | None, episodes: int, blue_target_mode: str, seed: int = 1000, device: str = "cpu") -> list[dict[str, Any]]:
    records = []; env = HeterogeneousMAVUAVAirCombatEnv(env_config, blue_target_mode=blue_target_mode)
    for episode in range(int(episodes)):
        observations, _ = env.reset(seed=seed + episode); done = False
        while not done:
            actions = []
            with torch.no_grad():
                for index, aid in enumerate(env.red_ids):
                    action, _ = actors.actors[index].sample(torch.as_tensor(observations[aid], device=device).unsqueeze(0), deterministic=True)
                    actions.append(action.squeeze(0).cpu().numpy())
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            done = terminated or truncated
        records.append(info["episode_summary"])
    return records
