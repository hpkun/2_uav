"""Deterministic HAPPO evaluation for one fixed Blue target mode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from algorithm.modules.hrta import ENEMY_SLICES, FRIEND_SLICES


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


def evaluate_actors(actors: Any, env_config: str | Path | Mapping[str, Any] | None, episodes: int, blue_target_mode: str, profile: str, seed: int = 1000, device: str = "cpu", attention_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = []; env = HeterogeneousMAVUAVAirCombatEnv(env_config, blue_target_mode=blue_target_mode, profile=profile)
    for episode in range(int(episodes)):
        observations, _ = env.reset(seed=seed + episode); done = False; decision_step = 0
        attention_start = len(attention_records) if attention_records is not None else 0
        while not done:
            actions = []
            with torch.no_grad():
                for index, aid in enumerate(env.red_ids):
                    actor_observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
                    actor = actors.actors[index]
                    if attention_records is not None:
                        _, diagnostics = actor.encode(actor_observation)
                        enemy = diagnostics["enemy_attention"].squeeze(0).cpu().numpy()
                        friend = diagnostics["friend_attention"].squeeze(0).cpu().numpy()
                        observation = observations[aid]
                        row = {
                            "episode": episode, "decision_step": decision_step,
                            "blue_mode": blue_target_mode, "agent": aid, "outcome": "",
                        }
                        row.update({f"friend_attention_friend{i + 1}": float(value) for i, value in enumerate(friend)})
                        for i, blue_id in enumerate(BLUE_IDS):
                            start = ENEMY_SLICES[i].start
                            row[f"enemy_attention_{blue_id}"] = float(enemy[i])
                            row[f"{blue_id}_alive"] = int(observation[start + 9] > 0.5)
                            row[f"{blue_id}_direct_or_datalink_visible"] = int(observation[start + 10] > 0.5 or observation[start + 11] > 0.5)
                        attention_records.append(row)
                    action, _ = actor.sample(actor_observation, deterministic=True)
                    actions.append(action.squeeze(0).cpu().numpy())
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            done = terminated or truncated
            decision_step += 1
        if attention_records is not None:
            for row in attention_records[attention_start:]:
                row["outcome"] = info["episode_summary"]["outcome"]
        records.append(info["episode_summary"])
    return records


def evaluate_recurrent_actors(
    actors: Any,
    env_config: str | Path | Mapping[str, Any] | None,
    episodes: int,
    blue_target_mode: str,
    profile: str,
    seed: int = 1000,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Deterministically evaluate recurrent actors with episode-safe hidden masks."""
    records: list[dict[str, Any]] = []
    env = HeterogeneousMAVUAVAirCombatEnv(
        env_config, blue_target_mode=blue_target_mode, profile=profile,
    )
    for episode in range(int(episodes)):
        observations, _ = env.reset(seed=seed + episode)
        hidden = [actor.initial_hidden(1, device=device) for actor in actors.actors]
        recurrent_masks = torch.zeros((len(RED_IDS), 1), device=device)
        done = False
        while not done:
            actions: list[np.ndarray] = []
            next_hidden: list[torch.Tensor] = []
            with torch.no_grad():
                for index, aid in enumerate(env.red_ids):
                    action, _, actor_hidden = actors.actors[index].sample_step(
                        torch.as_tensor(observations[aid], device=device).unsqueeze(0),
                        hidden[index], recurrent_masks[index], deterministic=True,
                    )
                    actions.append(action.squeeze(0).cpu().numpy())
                    next_hidden.append(actor_hidden)
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            done = terminated or truncated
            if not done:
                active = torch.as_tensor(info["active_masks"], device=device)
                recurrent_masks = active[:, None]
                hidden = [state * recurrent_masks[index] for index, state in enumerate(next_hidden)]
        records.append(info["episode_summary"])
    return records
