"""Vanilla HAPPO with sequential actor updates and importance factors."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

from algorithm.common.buffer import RolloutBuffer
from algorithm.common.networks import CentralizedCritic
from env.vector_env import MAVUAVVectorEnv
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from .networks import IndependentActors


DEFAULTS = {
    "environment_profile": "main", "seed": 1, "device": "cpu", "num_envs": 16, "rollout_steps": 128,
    "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 4, "minibatch_size": 256,
    "clip_coef": 0.2, "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
    "entropy_coef": 0.01, "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 128, "actor_variant": "vanilla",
    "hrta_entity_dim": 32, "hrta_role_dim": 16, "hrta_fusion_hidden_dim": 64,
}

RESUME_CONFIG_FIELDS = (
    "environment_profile", "seed", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm", "hidden_dim",
)


def preceding_factor_update(factor: torch.Tensor, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    return (factor * torch.where(active > 0.5, ratio, torch.ones_like(ratio))).detach()


def _restore_cuda_rng_state(states: list[torch.Tensor] | None) -> None:
    """Restore exact CUDA RNG bytes from CPU tensors required by PyTorch."""
    if states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in states])


class HAPPOTrainer:
    def __init__(self, env_config: str | Path | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.config = deepcopy(DEFAULTS)
        if config: self.config.update(dict(config.get("training", config)))
        c = self.config
        if c["environment_profile"] not in ("learnability", "main"):
            raise ValueError("environment_profile must be 'learnability' or 'main'")
        self.device = torch.device(c["device"])
        torch.manual_seed(int(c["seed"]))
        self.rng = np.random.default_rng(int(c["seed"]))
        self.environment_config = load_environment_config(env_config)
        self.vector_env = MAVUAVVectorEnv(
            int(c["num_envs"]), self.environment_config, seed=int(c["seed"]), profile=c["environment_profile"],
        )
        if c["actor_variant"] == "vanilla":
            self.actors = IndependentActors(hidden_dim=int(c["hidden_dim"])).to(self.device)
        elif c["actor_variant"] == "hrta":
            self.actors = HRTAIndependentActors(
                entity_dim=int(c["hrta_entity_dim"]),
                role_dim=int(c["hrta_role_dim"]),
                fusion_hidden_dim=int(c["hrta_fusion_hidden_dim"]),
                action_dim=3,
            ).to(self.device)
        elif c["actor_variant"] == "structured_uniform":
            self.actors = StructuredUniformIndependentActors(
                entity_dim=int(c["hrta_entity_dim"]),
                role_dim=int(c["hrta_role_dim"]),
                fusion_hidden_dim=int(c["hrta_fusion_hidden_dim"]),
                action_dim=3,
            ).to(self.device)
        else:
            raise ValueError("actor_variant must be 'vanilla', 'hrta' or 'structured_uniform'")
        self.critic = CentralizedCritic(GLOBAL_STATE_DIM, int(c["hidden_dim"])).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=float(c["actor_learning_rate"])) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(c["critic_learning_rate"]))
        self.buffer = RolloutBuffer(int(c["rollout_steps"]), int(c["num_envs"]))
        self.observations, self.global_states, self.active_masks, _ = self.vector_env.reset()
        self.env_steps = 0
        self.completed_episodes: list[dict[str, Any]] = []

    @property
    def actor_architecture(self) -> dict[str, int]:
        if self.config["actor_variant"] in ("hrta", "structured_uniform"):
            return {
                "entity_dim": int(self.config["hrta_entity_dim"]),
                "role_dim": int(self.config["hrta_role_dim"]),
                "fusion_hidden_dim": int(self.config["hrta_fusion_hidden_dim"]),
                "action_dim": 3,
            }
        return {"hidden_dim": int(self.config["hidden_dim"]), "action_dim": 3}

    @property
    def actor_parameter_counts(self) -> dict[str, Any]:
        per_agent = [sum(parameter.numel() for parameter in actor.parameters()) for actor in self.actors.actors]
        return {"per_agent": per_agent, "total": sum(per_agent)}

    def _validate_actor_architecture(self, data: Mapping[str, Any]) -> None:
        checkpoint_variant = data.get("actor_variant", data.get("trainer_config", data.get("config", {})).get("actor_variant", "vanilla"))
        checkpoint_architecture = data.get("actor_architecture")
        if checkpoint_variant != self.config["actor_variant"]:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_variant!r} "
                f"current={self.config['actor_variant']!r}"
            )
        if checkpoint_variant in ("hrta", "structured_uniform") and checkpoint_architecture != self.actor_architecture:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_architecture!r} "
                f"current={self.actor_architecture!r}"
            )

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
        observations = torch.as_tensor(self.buffer.observations.reshape(-1, 3, OBS_DIM), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, 3, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1, 3), device=self.device)
        active_masks = torch.as_tensor(self.buffer.active_masks.reshape(-1, 3), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device)
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
        torch.save({"environment_version": ENVIRONMENT_VERSION, "environment_profile": self.config["environment_profile"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "actor_variant": self.config["actor_variant"], "actor_architecture": self.actor_architecture, "actors": self.actors.state_dict(), "critic": self.critic.state_dict(), "config": self.config}, path)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return all state required for an exact continuation of training."""
        return {
            "format": "happo_training_checkpoint_v1",
            "sampled_steps": int(self.env_steps),
            "environment_version": ENVIRONMENT_VERSION,
            "environment_profile": self.config["environment_profile"],
            "observation_dim": OBS_DIM,
            "global_state_dim": GLOBAL_STATE_DIM,
            "actor_variant": self.config["actor_variant"],
            "actor_architecture": self.actor_architecture,
            "environment_config": deepcopy(self.environment_config),
            "trainer_config": deepcopy(self.config),
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer_states": [optimizer.state_dict() for optimizer in self.actor_optimizers],
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "trainer_numpy_rng": deepcopy(self.rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rollout_state": {
                "observations": self.observations.copy(),
                "global_states": self.global_states.copy(),
                "active_masks": self.active_masks.copy(),
                "environment_states": self.vector_env.get_env_states(),
                "vector_reset_counts": self.vector_env.reset_counts.copy(),
                "vector_base_seed": self.vector_env.base_seed,
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path: str | Path) -> int:
        data = torch.load(path, map_location=self.device, weights_only=False)
        expected = (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM)
        actual = (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim"))
        if actual != expected:
            raise RuntimeError("incompatible HAPPO checkpoint environment contract")
        self._validate_actor_architecture(data)
        saved_config = data.get("trainer_config", data.get("config", {}))
        for field in RESUME_CONFIG_FIELDS:
            checkpoint_value = saved_config.get(field)
            current_value = self.config.get(field)
            if checkpoint_value != current_value:
                raise RuntimeError(
                    f"resume config mismatch: {field} checkpoint={checkpoint_value!r} current={current_value!r}"
                )
        if data.get("environment_config") != self.environment_config:
            raise RuntimeError("resume environment config mismatch: resolved content differs from checkpoint")
        self.actors.load_state_dict(data["actors"])
        self.critic.load_state_dict(data["critic"])
        if "actor_optimizer_states" not in data or "rollout_state" not in data:
            raise RuntimeError("checkpoint contains weights only and cannot resume training")
        for optimizer, state in zip(self.actor_optimizers, data["actor_optimizer_states"]):
            optimizer.load_state_dict(state)
        self.critic_optimizer.load_state_dict(data["critic_optimizer_state"])
        self.rng.bit_generator.state = deepcopy(data["trainer_numpy_rng"])
        torch.set_rng_state(data["torch_rng"].cpu())
        _restore_cuda_rng_state(data.get("cuda_rng"))
        rollout = data["rollout_state"]
        self.observations = np.asarray(rollout["observations"], dtype=np.float32)
        self.global_states = np.asarray(rollout["global_states"], dtype=np.float32)
        self.active_masks = np.asarray(rollout["active_masks"], dtype=np.float32)
        self.vector_env.set_env_states(
            rollout["environment_states"],
            np.asarray(rollout["vector_reset_counts"], dtype=np.int64),
            rollout.get("vector_base_seed"),
        )
        self.env_steps = int(data["sampled_steps"])
        return self.env_steps

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim")) != (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM):
            raise RuntimeError("incompatible HAPPO checkpoint environment contract")
        self._validate_actor_architecture(data)
        if data.get("environment_profile") != self.config["environment_profile"]:
            raise RuntimeError(
                f"incompatible HAPPO checkpoint environment profile: {data.get('environment_profile')!r} "
                f"(expected {self.config['environment_profile']!r})"
            )
        self.actors.load_state_dict(data["actors"]); self.critic.load_state_dict(data["critic"])

    def close(self) -> None:
        self.vector_env.close()
