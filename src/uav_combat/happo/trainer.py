"""Vanilla HAPPO with sequential actor updates and importance factors."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

from ..mappo.buffer import RolloutBuffer
from ..mappo.networks import CentralizedCritic
from ..vector_env import MAVUAVVectorEnv
from .networks import IndependentActors


DEFAULTS = {
    "seed": 1, "device": "cpu", "num_envs": 4, "rollout_steps": 128,
    "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 4, "minibatch_size": 256,
    "clip_coef": 0.2, "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
    "entropy_coef": 0.01, "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 128,
}


def preceding_factor_update(factor: torch.Tensor, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    return (factor * torch.where(active > 0.5, ratio, torch.ones_like(ratio))).detach()


class HAPPOTrainer:
    def __init__(self, env_config: str | Path | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.config = deepcopy(DEFAULTS)
        if config: self.config.update(dict(config.get("training", config)))
        c = self.config
        self.device = torch.device(c["device"])
        torch.manual_seed(int(c["seed"]))
        self.rng = np.random.default_rng(int(c["seed"]))
        self.vector_env = MAVUAVVectorEnv(int(c["num_envs"]), env_config, seed=int(c["seed"]))
        self.actors = IndependentActors(hidden_dim=int(c["hidden_dim"])).to(self.device)
        self.critic = CentralizedCritic(40, int(c["hidden_dim"])).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=float(c["actor_learning_rate"])) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(c["critic_learning_rate"]))
        self.buffer = RolloutBuffer(int(c["rollout_steps"]), int(c["num_envs"]))
        self.observations, self.global_states, self.active_masks, _ = self.vector_env.reset()
        self.env_steps = 0
        self.completed_episodes: list[dict[str, Any]] = []

    def collect_rollout(self) -> list[dict[str, Any]]:
        self.buffer.reset(); completed = []
        for _ in range(self.buffer.horizon):
            actions, log_probs = [], []
            with torch.no_grad():
                for agent, actor in enumerate(self.actors.actors):
                    action, log_prob = actor.sample(torch.as_tensor(self.observations[:, agent], device=self.device))
                    actions.append(action.cpu().numpy()); log_probs.append(log_prob.cpu().numpy())
                values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
            action_array = np.stack(actions, axis=1); log_prob_array = np.stack(log_probs, axis=1)
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(action_array)
            self.buffer.insert(self.observations, self.global_states, action_array, log_prob_array, rewards, values, terminated, truncated, self.active_masks)
            completed.extend(info["episode_summary"] for info in infos if "episode_summary" in info)
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad(): last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]))
        self.completed_episodes.extend(completed)
        return completed

    def update(self) -> dict[str, Any]:
        c = self.config
        observations = torch.as_tensor(self.buffer.observations.reshape(-1, 3, 40), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, 3, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1, 3), device=self.device)
        active_masks = torch.as_tensor(self.buffer.active_masks.reshape(-1, 3), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(-1, 40), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device)
        factor = torch.ones_like(advantages)
        order = [int(v) for v in self.rng.permutation(3)]
        actor_losses: list[list[float]] = [[], [], []]; entropies: list[float] = []
        clip = float(c["clip_coef"]); mini = int(c["minibatch_size"]); total = len(advantages)
        for agent in order:
            active = active_masks[:, agent] > 0.5
            normalized = advantages.clone()
            if active.any():
                normalized = (advantages - advantages[active].mean()) / advantages[active].std(unbiased=False).clamp_min(1e-8)
            for _ in range(int(c["ppo_epochs"])):
                sample_order = self.rng.permutation(total)
                for start in range(0, total, mini):
                    idx = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                    idx = idx[active[idx]]
                    if not len(idx): continue
                    new_log_prob, entropy = self.actors.actors[agent].evaluate_actions(observations[idx, agent], actions[idx, agent])
                    ratio = torch.exp(new_log_prob - old_log_probs[idx, agent])
                    effective = (factor[idx] * normalized[idx]).detach()
                    policy_loss = -torch.minimum(ratio * effective, ratio.clamp(1.0 - clip, 1.0 + clip) * effective).mean()
                    loss = policy_loss - float(c["entropy_coef"]) * entropy.mean()
                    optimizer = self.actor_optimizers[agent]
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.actors.actors[agent].parameters(), float(c["max_grad_norm"])); optimizer.step()
                    actor_losses[agent].append(float(policy_loss.item())); entropies.append(float(entropy.mean().item()))
            with torch.no_grad():
                new_all, _ = self.actors.actors[agent].evaluate_actions(observations[:, agent], actions[:, agent])
                factor = preceding_factor_update(factor, old_log_probs[:, agent], new_all, active_masks[:, agent])
        critic_losses = []
        for _ in range(int(c["ppo_epochs"])):
            sample_order = self.rng.permutation(total)
            for start in range(0, total, mini):
                idx = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                value_loss = (self.critic(states[idx]) - returns[idx]).square().mean()
                self.critic_optimizer.zero_grad(); (float(c["value_loss_coef"]) * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"])); self.critic_optimizer.step()
                critic_losses.append(float(value_loss.item()))
        metrics: dict[str, Any] = {f"actor_{i}_loss": float(np.mean(actor_losses[i])) if actor_losses[i] else 0.0 for i in range(3)}
        metrics.update({"actor_loss": float(np.mean([v for rows in actor_losses for v in rows])), "critic_loss": float(np.mean(critic_losses)), "entropy": float(np.mean(entropies)), "agent_update_order": order})
        if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, float)): raise FloatingPointError("non-finite HAPPO update")
        return metrics

    def train_update(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        episodes = self.collect_rollout()
        return episodes, self.update()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actors": self.actors.state_dict(), "critic": self.critic.state_dict(), "config": self.config}, path)

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actors.load_state_dict(data["actors"]); self.critic.load_state_dict(data["critic"])

    def close(self) -> None:
        self.vector_env.close()

