"""HAPPO trainer with isolated feed-forward and recurrent actor paths."""
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
from .recurrent import RecurrentIndependentActors
from .recurrent_buffer import RecurrentRolloutBuffer
from .agp import apply_agp
from .curriculum import DEFAULT_CURRICULUM_SCHEDULE, nearest_probability, normalized_schedule


DEFAULTS = {
    "environment_profile": "main", "seed": 1, "device": "cpu", "num_envs": 16, "rollout_steps": 128,
    "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 4, "minibatch_size": 256,
    "clip_coef": 0.2, "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
    "entropy_coef": 0.01, "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 128, "actor_variant": "vanilla", "method_variant": "baseline",
    "agp_lambda": 0.5, "curriculum_schedule": DEFAULT_CURRICULUM_SCHEDULE,
    "curriculum_total_steps": None,
    "hrta_entity_dim": 32, "hrta_role_dim": 16, "hrta_fusion_hidden_dim": 64,
    "recurrent_hidden_dim": 128, "recurrent_sequence_length": 16,
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
        if c["method_variant"] not in ("baseline", "agp", "curriculum", "agp_curriculum"):
            raise ValueError("invalid method_variant")
        if c["actor_variant"] != "vanilla" and c["method_variant"] != "baseline":
            raise ValueError("AGP and curriculum methods require actor_variant='vanilla'")
        self.agp_enabled = c["method_variant"] in ("agp", "agp_curriculum")
        self.curriculum_enabled = c["method_variant"] in ("curriculum", "agp_curriculum")
        self.curriculum_schedule = normalized_schedule(c.get("curriculum_schedule"))
        self.curriculum_total_steps = c.get("curriculum_total_steps")
        if self.curriculum_enabled and (
            self.curriculum_total_steps is None or int(self.curriculum_total_steps) <= 0
        ):
            raise ValueError("curriculum methods require positive curriculum_total_steps")
        if float(c["agp_lambda"]) < 0.0:
            raise ValueError("agp_lambda cannot be negative")
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
        elif c["actor_variant"] == "recurrent":
            self.actors = RecurrentIndependentActors(
                observation_dim=OBS_DIM, action_dim=3, hidden_dim=int(c["hidden_dim"]),
                recurrent_hidden_dim=int(c["recurrent_hidden_dim"]),
            ).to(self.device)
        else:
            raise ValueError("actor_variant must be 'vanilla', 'hrta', 'structured_uniform' or 'recurrent'")
        self.critic = CentralizedCritic(GLOBAL_STATE_DIM, int(c["hidden_dim"])).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=float(c["actor_learning_rate"])) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(c["critic_learning_rate"]))
        self.buffer = self.make_buffer(int(c["rollout_steps"]))
        initial_probability = self._curriculum_probability(0)
        self.observations, self.global_states, self.active_masks, reset_infos = self.vector_env.reset(
            nearest_probability=initial_probability,
        )
        if self.is_recurrent:
            self.actor_hidden_states = np.zeros(
                (int(c["num_envs"]), 3, int(c["recurrent_hidden_dim"])), dtype=np.float32,
            )
            self.actor_recurrent_masks = np.zeros((int(c["num_envs"]), 3), dtype=np.float32)
        self.env_steps = 0
        self.completed_episodes: list[dict[str, Any]] = []
        self.current_blue_modes = [str(info["blue_target_mode"]) for info in reset_infos]
        self.mode_transition_counts = {"nearest": 0, "mav_priority": 0}
        self.mode_episode_counts = {"nearest": 0, "mav_priority": 0}
        self.last_rollout_metrics = self._empty_rollout_metrics(initial_probability)

    def _curriculum_probability(self, sampled_steps: int) -> float | None:
        if not self.curriculum_enabled:
            return None
        return nearest_probability(
            sampled_steps,
            int(self.curriculum_total_steps),
            self.curriculum_schedule,
        )

    def _empty_rollout_metrics(self, probability: float | None) -> dict[str, Any]:
        return {
            "method_variant": self.config["method_variant"],
            "p_nearest": probability,
            "agp_raw_mean": 0.0,
            "agp_raw_mean_abs": 0.0,
            "agp_shaping_mean": 0.0,
            "agp_shaping_mean_abs": 0.0,
            "transitions_nearest": int(self.mode_transition_counts.get("nearest", 0)) if hasattr(self, "mode_transition_counts") else 0,
            "transitions_mav_priority": int(self.mode_transition_counts.get("mav_priority", 0)) if hasattr(self, "mode_transition_counts") else 0,
            "episodes_nearest": int(self.mode_episode_counts.get("nearest", 0)) if hasattr(self, "mode_episode_counts") else 0,
            "episodes_mav_priority": int(self.mode_episode_counts.get("mav_priority", 0)) if hasattr(self, "mode_episode_counts") else 0,
        }

    @property
    def actor_architecture(self) -> dict[str, int]:
        if self.config["actor_variant"] == "recurrent":
            return {
                "observation_dim": OBS_DIM,
                "encoder_dim": int(self.config["hidden_dim"]),
                "recurrent_hidden_dim": int(self.config["recurrent_hidden_dim"]),
                "head_dim": int(self.config["hidden_dim"]),
                "action_dim": 3,
            }
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

    @property
    def is_recurrent(self) -> bool:
        return self.config["actor_variant"] == "recurrent"

    def make_buffer(self, horizon: int) -> RolloutBuffer:
        if self.is_recurrent:
            return RecurrentRolloutBuffer(
                horizon, int(self.config["num_envs"]), int(self.config["recurrent_hidden_dim"]),
            )
        return RolloutBuffer(horizon, int(self.config["num_envs"]))

    def _validate_actor_architecture(self, data: Mapping[str, Any]) -> None:
        checkpoint_variant = data.get("actor_variant", data.get("trainer_config", data.get("config", {})).get("actor_variant", "vanilla"))
        checkpoint_architecture = data.get("actor_architecture")
        if checkpoint_variant != self.config["actor_variant"]:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_variant!r} "
                f"current={self.config['actor_variant']!r}"
            )
        if checkpoint_variant in ("hrta", "structured_uniform", "recurrent") and checkpoint_architecture != self.actor_architecture:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_architecture!r} "
                f"current={self.actor_architecture!r}"
            )

    def collect_rollout(self) -> list[dict[str, Any]]:
        if self.is_recurrent:
            return self._collect_recurrent_rollout()
        self.buffer.reset(); completed = []
        raw_terms: list[np.ndarray] = []
        shaping_terms: list[np.ndarray] = []
        update_probability = self._curriculum_probability(self.env_steps)
        for _ in range(self.buffer.horizon):
            actions, log_probs = [], []
            with torch.no_grad():
                for agent, actor in enumerate(self.actors.actors):
                    action, log_prob = actor.sample(torch.as_tensor(self.observations[:, agent], device=self.device))
                    actions.append(action.cpu().numpy()); log_probs.append(log_prob.cpu().numpy())
                values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
            action_array = np.stack(actions, axis=1); log_prob_array = np.stack(log_probs, axis=1)
            reset_probability = self._curriculum_probability(self.env_steps + self.buffer.num_envs)
            transition_modes = tuple(self.current_blue_modes)
            for mode in transition_modes:
                self.mode_transition_counts[mode] += 1
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(
                action_array, reset_nearest_probability=reset_probability,
            )
            done = np.logical_or(terminated, truncated)
            if self.agp_enabled:
                training_rewards, raw, shaping = apply_agp(
                    rewards,
                    self.observations,
                    next_obs,
                    done,
                    float(self.environment_config["normalization"]["distance_scale"]),
                    gamma=float(self.config["gamma"]),
                    agp_lambda=float(self.config["agp_lambda"]),
                )
            else:
                training_rewards = rewards
                raw = np.zeros(self.buffer.num_envs, dtype=np.float64)
                shaping = np.zeros(self.buffer.num_envs, dtype=np.float64)
            raw_terms.append(raw)
            shaping_terms.append(shaping)
            self.buffer.insert(self.observations, self.global_states, action_array, log_prob_array, training_rewards, values, terminated, truncated, self.active_masks)
            completed.extend(info["episode_summary"] for info in infos if "episode_summary" in info)
            for index, info in enumerate(infos):
                if done[index]:
                    self.mode_episode_counts[transition_modes[index]] += 1
                if info.get("auto_reset"):
                    self.current_blue_modes[index] = str(info["reset_info"]["blue_target_mode"])
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad(): last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]))
        raw_values = np.concatenate(raw_terms) if raw_terms else np.zeros(1)
        shaping_values = np.concatenate(shaping_terms) if shaping_terms else np.zeros(1)
        self.last_rollout_metrics = {
            "method_variant": self.config["method_variant"],
            "p_nearest": update_probability,
            "agp_raw_mean": float(np.mean(raw_values)),
            "agp_raw_mean_abs": float(np.mean(np.abs(raw_values))),
            "agp_shaping_mean": float(np.mean(shaping_values)),
            "agp_shaping_mean_abs": float(np.mean(np.abs(shaping_values))),
            "transitions_nearest": int(self.mode_transition_counts["nearest"]),
            "transitions_mav_priority": int(self.mode_transition_counts["mav_priority"]),
            "episodes_nearest": int(self.mode_episode_counts["nearest"]),
            "episodes_mav_priority": int(self.mode_episode_counts["mav_priority"]),
        }
        self.completed_episodes.extend(completed)
        return completed

    def _collect_recurrent_rollout(self) -> list[dict[str, Any]]:
        """Collect a rollout while preserving hidden state across rollout boundaries."""
        if not isinstance(self.buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent actor requires RecurrentRolloutBuffer")
        self.buffer.reset()
        completed: list[dict[str, Any]] = []
        update_probability = self._curriculum_probability(self.env_steps)
        for _ in range(self.buffer.horizon):
            actions: list[np.ndarray] = []
            log_probs: list[np.ndarray] = []
            next_hidden = np.empty_like(self.actor_hidden_states)
            with torch.no_grad():
                for agent, actor in enumerate(self.actors.actors):
                    action, log_prob, hidden = actor.sample_step(
                        torch.as_tensor(self.observations[:, agent], device=self.device),
                        torch.as_tensor(self.actor_hidden_states[:, agent], device=self.device),
                        torch.as_tensor(self.actor_recurrent_masks[:, agent], device=self.device),
                    )
                    actions.append(action.cpu().numpy())
                    log_probs.append(log_prob.cpu().numpy())
                    next_hidden[:, agent] = hidden.cpu().numpy()
                values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
            action_array = np.stack(actions, axis=1)
            log_prob_array = np.stack(log_probs, axis=1)
            transition_modes = tuple(self.current_blue_modes)
            for mode in transition_modes:
                self.mode_transition_counts[mode] += 1
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(
                action_array, reset_nearest_probability=None,
            )
            done = np.logical_or(terminated, truncated)
            next_recurrent_masks = next_masks.astype(np.float32) * (~done)[:, None].astype(np.float32)
            next_hidden *= next_recurrent_masks[:, :, None]
            self.buffer.insert(
                self.observations, self.global_states, action_array, log_prob_array, rewards, values,
                terminated, truncated, self.active_masks, self.actor_hidden_states,
                self.actor_recurrent_masks, next_hidden,
            )
            completed.extend(info["episode_summary"] for info in infos if "episode_summary" in info)
            for index, info in enumerate(infos):
                if done[index]:
                    self.mode_episode_counts[transition_modes[index]] += 1
                if info.get("auto_reset"):
                    self.current_blue_modes[index] = str(info["reset_info"]["blue_target_mode"])
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.actor_hidden_states = next_hidden
            self.actor_recurrent_masks = next_recurrent_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(
            last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]),
        )
        self.last_rollout_metrics = {
            "method_variant": self.config["method_variant"], "p_nearest": update_probability,
            "agp_raw_mean": 0.0, "agp_raw_mean_abs": 0.0,
            "agp_shaping_mean": 0.0, "agp_shaping_mean_abs": 0.0,
            "transitions_nearest": int(self.mode_transition_counts["nearest"]),
            "transitions_mav_priority": int(self.mode_transition_counts["mav_priority"]),
            "episodes_nearest": int(self.mode_episode_counts["nearest"]),
            "episodes_mav_priority": int(self.mode_episode_counts["mav_priority"]),
        }
        self.completed_episodes.extend(completed)
        return completed

    def update(self) -> dict[str, Any]:
        if self.is_recurrent:
            return self._update_recurrent()
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
        metrics.update(self.last_rollout_metrics)
        if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, float)): raise FloatingPointError("non-finite HAPPO update")
        return metrics

    def _recurrent_sequence_tensors(
        self, agent: int, specs: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer) or not specs:
            raise ValueError("a non-empty recurrent sequence batch is required")
        observations = np.stack([buffer.observations[start:end, env, agent] for env, start, end in specs])
        actions = np.stack([buffer.actions[start:end, env, agent] for env, start, end in specs])
        initial_hidden = np.stack([buffer.actor_hidden_states[start, env, agent] for env, start, _ in specs])
        masks = np.stack([buffer.recurrent_masks[start:end, env, agent] for env, start, end in specs])
        return tuple(torch.as_tensor(value, device=self.device) for value in (
            observations, actions, initial_hidden, masks,
        ))

    def _recurrent_log_probs_all(self, agent: int) -> torch.Tensor:
        """Re-evaluate every transition in ordered TBPTT chunks without padding or drops."""
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent log-prob evaluation requires RecurrentRolloutBuffer")
        result = torch.empty((buffer.horizon, buffer.num_envs), device=self.device)
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for spec in buffer.chunks(int(self.config["recurrent_sequence_length"])):
            groups.setdefault(spec[2] - spec[1], []).append(spec)
        actor = self.actors.actors[agent]
        with torch.no_grad():
            for specs in groups.values():
                obs, actions, initial_hidden, masks = self._recurrent_sequence_tensors(agent, specs)
                log_probs, _, _ = actor.evaluate_actions_sequence(obs, actions, initial_hidden, masks)
                for index, (env, start, end) in enumerate(specs):
                    result[start:end, env] = log_probs[index]
        return result

    def _update_recurrent(self) -> dict[str, Any]:
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent update requires RecurrentRolloutBuffer")
        c = self.config
        old_log_probs = torch.as_tensor(buffer.log_probs, device=self.device)
        active_masks = torch.as_tensor(buffer.active_masks, device=self.device)
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        factor = torch.ones_like(advantages)
        order = [int(value) for value in self.rng.permutation(3)]
        actor_losses: list[list[float]] = [[], [], []]
        entropies: list[float] = []
        clip = float(c["clip_coef"])
        mini = int(c["minibatch_size"])
        sequence_length = int(c["recurrent_sequence_length"])
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for spec in buffer.chunks(sequence_length):
            groups.setdefault(spec[2] - spec[1], []).append(spec)
        self.last_recurrent_factor_history = [factor.detach().cpu().numpy().copy()]
        for agent in order:
            active = active_masks[:, :, agent] > 0.5
            normalized = advantages.clone()
            if active.any():
                normalized = (
                    advantages - advantages[active].mean()
                ) / advantages[active].std(unbiased=False).clamp_min(1e-8)
            actor = self.actors.actors[agent]
            optimizer = self.actor_optimizers[agent]
            for _ in range(int(c["ppo_epochs"])):
                for length, all_specs in groups.items():
                    chunks_per_batch = max(1, mini // length)
                    shuffled = self.rng.permutation(len(all_specs))
                    for start_index in range(0, len(all_specs), chunks_per_batch):
                        specs = [all_specs[int(index)] for index in shuffled[start_index:start_index + chunks_per_batch]]
                        obs, action_batch, initial_hidden, masks = self._recurrent_sequence_tensors(agent, specs)
                        new_log_prob, entropy, _ = actor.evaluate_actions_sequence(
                            obs, action_batch, initial_hidden, masks,
                        )
                        old = torch.stack([old_log_probs[start:end, env, agent] for env, start, end in specs])
                        batch_active = torch.stack([active_masks[start:end, env, agent] for env, start, end in specs]) > 0.5
                        batch_advantage = torch.stack([normalized[start:end, env] for env, start, end in specs])
                        batch_factor = torch.stack([factor[start:end, env] for env, start, end in specs])
                        if not batch_active.any():
                            continue
                        ratio = torch.exp(new_log_prob[batch_active] - old[batch_active])
                        effective = (batch_factor[batch_active] * batch_advantage[batch_active]).detach()
                        policy_loss = -torch.minimum(
                            ratio * effective,
                            ratio.clamp(1.0 - clip, 1.0 + clip) * effective,
                        ).mean()
                        loss = policy_loss - float(c["entropy_coef"]) * entropy[batch_active].mean()
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), float(c["max_grad_norm"]))
                        optimizer.step()
                        actor_losses[agent].append(float(policy_loss.item()))
                        entropies.append(float(entropy[batch_active].mean().item()))
            new_all = self._recurrent_log_probs_all(agent)
            factor = preceding_factor_update(
                factor, old_log_probs[:, :, agent], new_all, active_masks[:, :, agent],
            )
            self.last_recurrent_factor_history.append(factor.detach().cpu().numpy().copy())

        states = torch.as_tensor(buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device)
        returns = torch.as_tensor(buffer.returns.reshape(-1), device=self.device)
        total = len(returns)
        critic_losses: list[float] = []
        for _ in range(int(c["ppo_epochs"])):
            sample_order = self.rng.permutation(total)
            for start in range(0, total, mini):
                indices = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                value_loss = (self.critic(states[indices]) - returns[indices]).square().mean()
                self.critic_optimizer.zero_grad()
                (float(c["value_loss_coef"]) * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"]))
                self.critic_optimizer.step()
                critic_losses.append(float(value_loss.item()))
        flat_actor_losses = [value for rows in actor_losses for value in rows]
        metrics: dict[str, Any] = {
            f"actor_{agent}_loss": float(np.mean(actor_losses[agent])) if actor_losses[agent] else 0.0
            for agent in range(3)
        }
        metrics.update({
            "actor_loss": float(np.mean(flat_actor_losses)) if flat_actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "agent_update_order": order,
        })
        metrics.update(self.last_rollout_metrics)
        if not all(np.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError("non-finite recurrent HAPPO update")
        return metrics

    def train_update(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        episodes = self.collect_rollout()
        return episodes, self.update()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"environment_version": ENVIRONMENT_VERSION, "environment_profile": self.config["environment_profile"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "actor_variant": self.config["actor_variant"], "method_variant": self.config["method_variant"], "actor_architecture": self.actor_architecture, "actors": self.actors.state_dict(), "critic": self.critic.state_dict(), "config": self.config}, path)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return all state required for an exact continuation of training."""
        state = {
            "format": "happo_training_checkpoint_v1",
            "sampled_steps": int(self.env_steps),
            "environment_version": ENVIRONMENT_VERSION,
            "environment_profile": self.config["environment_profile"],
            "observation_dim": OBS_DIM,
            "global_state_dim": GLOBAL_STATE_DIM,
            "actor_variant": self.config["actor_variant"],
            "method_variant": self.config["method_variant"],
            "agp_lambda": float(self.config["agp_lambda"]),
            "curriculum_schedule": [list(item) for item in self.curriculum_schedule],
            "curriculum_total_steps": self.curriculum_total_steps,
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
                "current_blue_modes": list(self.current_blue_modes),
                "mode_transition_counts": deepcopy(self.mode_transition_counts),
                "mode_episode_counts": deepcopy(self.mode_episode_counts),
            },
        }
        if self.is_recurrent:
            state["rollout_state"]["actor_hidden_states"] = self.actor_hidden_states.copy()
            state["rollout_state"]["actor_recurrent_masks"] = self.actor_recurrent_masks.copy()
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path: str | Path) -> int:
        data = torch.load(path, map_location=self.device, weights_only=False)
        expected = (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM)
        actual = (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim"))
        if actual != expected:
            raise RuntimeError("incompatible checkpoint contract for HAPPO environment")
        self._validate_actor_architecture(data)
        saved_config = data.get("trainer_config", data.get("config", {}))
        checkpoint_method = data.get("method_variant", saved_config.get("method_variant", "baseline"))
        if checkpoint_method != self.config["method_variant"]:
            raise RuntimeError(
                f"resume method mismatch: checkpoint={checkpoint_method!r} "
                f"current={self.config['method_variant']!r}"
            )
        if self.agp_enabled:
            checkpoint_lambda = data.get("agp_lambda", saved_config.get("agp_lambda"))
            if checkpoint_lambda is None or float(checkpoint_lambda) != float(self.config["agp_lambda"]):
                raise RuntimeError("resume AGP lambda mismatch")
        if self.curriculum_enabled:
            checkpoint_schedule = data.get("curriculum_schedule", saved_config.get("curriculum_schedule"))
            checkpoint_total = data.get("curriculum_total_steps", saved_config.get("curriculum_total_steps"))
            if checkpoint_schedule is None or normalized_schedule(checkpoint_schedule) != self.curriculum_schedule:
                raise RuntimeError("resume curriculum schedule mismatch")
            if checkpoint_total is None or int(checkpoint_total) != int(self.curriculum_total_steps):
                raise RuntimeError("resume curriculum total steps mismatch")
        for field in RESUME_CONFIG_FIELDS:
            checkpoint_value = saved_config.get(field)
            current_value = self.config.get(field)
            if checkpoint_value != current_value:
                raise RuntimeError(
                    f"resume config mismatch: {field} checkpoint={checkpoint_value!r} current={current_value!r}"
                )
        if self.is_recurrent:
            for field in ("recurrent_hidden_dim", "recurrent_sequence_length"):
                if saved_config.get(field) != self.config.get(field):
                    raise RuntimeError(
                        f"resume config mismatch: {field} checkpoint={saved_config.get(field)!r} "
                        f"current={self.config.get(field)!r}"
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
        if self.is_recurrent:
            if "actor_hidden_states" not in rollout or "actor_recurrent_masks" not in rollout:
                raise RuntimeError("recurrent checkpoint is missing hidden-state continuation data")
            self.actor_hidden_states = np.asarray(rollout["actor_hidden_states"], dtype=np.float32)
            self.actor_recurrent_masks = np.asarray(rollout["actor_recurrent_masks"], dtype=np.float32)
            expected_hidden = (
                int(self.config["num_envs"]), 3, int(self.config["recurrent_hidden_dim"]),
            )
            if self.actor_hidden_states.shape != expected_hidden:
                raise RuntimeError("checkpoint recurrent hidden-state shape mismatch")
            if self.actor_recurrent_masks.shape != (int(self.config["num_envs"]), 3):
                raise RuntimeError("checkpoint recurrent mask shape mismatch")
        self.vector_env.set_env_states(
            rollout["environment_states"],
            np.asarray(rollout["vector_reset_counts"], dtype=np.int64),
            rollout.get("vector_base_seed"),
        )
        if "current_blue_modes" in rollout:
            modes = [str(mode) for mode in rollout["current_blue_modes"]]
        else:
            modes = [str(state["blue_episode_mode"]) for state in rollout["environment_states"]]
        if len(modes) != int(self.config["num_envs"]):
            raise RuntimeError("checkpoint current_blue_modes shape mismatch")
        self.current_blue_modes = modes
        exposure_fields = ("mode_transition_counts", "mode_episode_counts")
        if self.config["method_variant"] != "baseline" and any(field not in rollout for field in exposure_fields):
            raise RuntimeError("method checkpoint is missing mode exposure state")
        self.mode_transition_counts = {
            mode: int(rollout.get("mode_transition_counts", {}).get(mode, 0))
            for mode in ("nearest", "mav_priority")
        }
        self.mode_episode_counts = {
            mode: int(rollout.get("mode_episode_counts", {}).get(mode, 0))
            for mode in ("nearest", "mav_priority")
        }
        self.env_steps = int(data["sampled_steps"])
        self.last_rollout_metrics = self._empty_rollout_metrics(self._curriculum_probability(self.env_steps))
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
