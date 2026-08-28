"""Deterministic HAPPO evaluation for one fixed Blue target mode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from env.mavuav import HeterogeneousMAVUAVAirCombatEnv
from .networks import IndependentActors


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"completed_episodes": 0}
    n = len(records)
    return {
        "completed_episodes": n,
        "mean_episode_return": float(np.mean([r["episode_return"] for r in records])),
        "red_win_rate": sum(r["outcome"] == "red" for r in records) / n,
        "blue_win_rate": sum(r["outcome"] == "blue" for r in records) / n,
        "draw_rate": sum(r["outcome"] == "draw" for r in records) / n,
        "MAV_survival_rate": float(np.mean([r["mav_survived"] for r in records])),
        "mean_UAV_survivors": float(np.mean([r["red_uav_survivors"] for r in records])),
        "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
        "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
    }


def evaluate_actors(actors: IndependentActors, env_config: str | Path | Mapping[str, Any] | None, episodes: int, blue_target_mode: str, profile: str, seed: int = 1000, device: str = "cpu") -> list[dict[str, Any]]:
    records = []; env = HeterogeneousMAVUAVAirCombatEnv(env_config, blue_target_mode=blue_target_mode, profile=profile)
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
