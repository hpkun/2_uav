"""Vanilla shared-actor MAPPO baseline for the MAV/UAV environment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

from env.vector_env import MAVUAVVectorEnv
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM
from algorithm.common.buffer import RolloutBuffer
from algorithm.common.networks import CentralizedCritic, GaussianActor


DEFAULTS = {
    "environment_profile": "main", "seed": 1, "device": "cpu", "num_envs": 16, "rollout_steps": 128,
    "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 4, "minibatch_size": 256,
    "clip_coef": 0.2, "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
    "entropy_coef": 0.01, "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 128,
}


class MAPPOTrainer:
    def __init__(self, env_config: str | Path | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.config = deepcopy(DEFAULTS)
        if config: self.config.update(dict(config.get("training", config)))
        c = self.config
        if c["environment_profile"] not in ("learnability", "main"):
            raise ValueError("environment_profile must be 'learnability' or 'main'")
        self.device = torch.device(c["device"])
        torch.manual_seed(int(c["seed"]))
        self.rng = np.random.default_rng(int(c["seed"]))
        self.vector_env = MAVUAVVectorEnv(
            int(c["num_envs"]), env_config, seed=int(c["seed"]), profile=c["environment_profile"],
        )
        self.actor = GaussianActor(OBS_DIM, 3, int(c["hidden_dim"])).to(self.device)
        self.critic = CentralizedCritic(GLOBAL_STATE_DIM, int(c["hidden_dim"])).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(c["actor_learning_rate"]))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(c["critic_learning_rate"]))
        self.buffer = RolloutBuffer(int(c["rollout_steps"]), int(c["num_envs"]))
        self.observations, self.global_states, self.active_masks, _ = self.vector_env.reset()
        self.env_steps = 0
        self.completed_episodes: list[dict[str, Any]] = []

    def collect_rollout(self) -> list[dict[str, Any]]:
        self.buffer.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(self.buffer.horizon):
            obs_tensor = torch.as_tensor(self.observations, device=self.device)
            state_tensor = torch.as_tensor(self.global_states, device=self.device)
            with torch.no_grad():
                actions, log_probs = self.actor.sample(obs_tensor.reshape(-1, OBS_DIM))
                values = self.critic(state_tensor)
            actions_np = actions.reshape(self.buffer.num_envs, 3, 3).cpu().numpy()
            log_probs_np = log_probs.reshape(self.buffer.num_envs, 3).cpu().numpy()
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(actions_np)
            self.buffer.insert(self.observations, self.global_states, actions_np, log_probs_np, rewards, values.cpu().numpy(), terminated, truncated, self.active_masks)
            for info in infos:
                if "episode_summary" in info: completed.append(info["episode_summary"])
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]))
        self.completed_episodes.extend(completed)
        return completed

    def update(self) -> dict[str, float]:
        c = self.config
        obs = torch.as_tensor(self.buffer.observations.reshape(-1, OBS_DIM), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1), device=self.device)
        active = torch.as_tensor(self.buffer.active_masks.reshape(-1), device=self.device) > 0.5
        advantages_team = np.repeat(self.buffer.advantages[..., None], 3, axis=-1).reshape(-1)
        advantages = torch.as_tensor(advantages_team, device=self.device)
        if active.any():
            mean, std = advantages[active].mean(), advantages[active].std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - mean) / std
        states = torch.as_tensor(self.buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device)
        actor_losses, critic_losses, entropies = [], [], []
        batch_size, mini = len(returns), int(c["minibatch_size"])
        for _ in range(int(c["ppo_epochs"])):
            order = self.rng.permutation(batch_size)
            for start in range(0, batch_size, mini):
                team_idx = order[start:start + mini]
                agent_idx = np.concatenate([3 * team_idx + i for i in range(3)])
                idx = torch.as_tensor(agent_idx, device=self.device)
                idx = idx[active[idx]]
                if len(idx):
                    new_log_probs, entropy = self.actor.evaluate_actions(obs[idx], actions[idx])
                    ratio = (new_log_probs - old_log_probs[idx]).exp()
                    clipped = ratio.clamp(1.0 - float(c["clip_coef"]), 1.0 + float(c["clip_coef"]))
                    policy_loss = -torch.minimum(ratio * advantages[idx], clipped * advantages[idx]).mean()
                    actor_loss = policy_loss - float(c["entropy_coef"]) * entropy.mean()
                    self.actor_optimizer.zero_grad(); actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), float(c["max_grad_norm"])); self.actor_optimizer.step()
                    actor_losses.append(float(policy_loss.item())); entropies.append(float(entropy.mean().item()))
                tidx = torch.as_tensor(team_idx, device=self.device)
                value_loss = (self.critic(states[tidx]) - returns[tidx]).square().mean()
                self.critic_optimizer.zero_grad(); (float(c["value_loss_coef"]) * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"])); self.critic_optimizer.step()
                critic_losses.append(float(value_loss.item()))
        metrics = {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)), "entropy": float(np.mean(entropies))}
        if not all(np.isfinite(list(metrics.values()))): raise FloatingPointError("non-finite MAPPO update")
        return metrics

    def train_update(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        episodes = self.collect_rollout()
        return episodes, self.update()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"environment_version": ENVIRONMENT_VERSION, "environment_profile": self.config["environment_profile"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": self.config}, path)

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim")) != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM):
            raise RuntimeError("incompatible MAPPO checkpoint environment contract")
        if data.get("environment_profile") != self.config["environment_profile"]:
            raise RuntimeError(
                f"incompatible MAPPO checkpoint environment profile: {data.get('environment_profile')!r} "
                f"(expected {self.config['environment_profile']!r})"
            )
        self.actor.load_state_dict(data["actor"]); self.critic.load_state_dict(data["critic"])

    def close(self) -> None:
        self.vector_env.close()
