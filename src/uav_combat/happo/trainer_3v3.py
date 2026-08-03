"""Project-native HAPPO trainer for homogeneous 3v3 fixed-blue experiments."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..environment_3v3 import GS_DIM, OBS_DIM
from ..config import load_config
from ..mappo.trainer_3v3 import compute_best_score, compute_best_score_fields, linear_schedule, resolve_device
from ..main_experiment_v8 import (
    best_score_fields_for_config,
    compute_best_score_for_config,
    infer_best_score_schema_for_checkpoint,
)
from ..mappo.vector_env_3v3 import (
    RED_REWARD_COMPONENT_KEYS_3V3,
    VectorStepResult3v3,
    decode_3v3_outcome,
    decode_3v3_termination_reason,
    make_combat_vector_env_3v3,
)
from .buffer_3v3 import HAPPORolloutBuffer3v3
from .metrics import explained_variance
from .networks import CentralizedValueCritic, IndependentHAPPOActors

CHECKPOINT_FAMILY_HAPPO_3V3 = "homogeneous_3v3_fixed_blue_happo"
CHECKPOINT_VERSION_HAPPO_3V3 = 1


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature_mismatches(checkpoint: Any, current: Any, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(checkpoint, dict) and isinstance(current, dict):
        for key in sorted(set(checkpoint) | set(current)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in checkpoint:
                diffs.append(f"- {child}: checkpoint=<missing> current={current[key]!r}")
            elif key not in current:
                diffs.append(f"- {child}: checkpoint={checkpoint[key]!r} current=<missing>")
            else:
                diffs.extend(signature_mismatches(checkpoint[key], current[key], child))
    elif checkpoint != current:
        diffs.append(f"- {prefix}: checkpoint={checkpoint!r} current={current!r}")
    return diffs


def happo_preceding_factor_update(
    factor: torch.Tensor,
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """HAPPO preceding-policy-ratio factor update.

    Inactive/dead samples contribute ratio 1, and the returned factor is
    detached so gradients cannot cross agent-update boundaries.
    """
    ratio = torch.exp(new_log_prob - old_log_prob)
    ratio = torch.where(active_mask > 0.5, ratio, torch.ones_like(ratio))
    return (factor * ratio).detach()


def ppo_clipped_policy_loss(ratio: torch.Tensor, advantage: torch.Tensor, clip_coef: float) -> torch.Tensor:
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def normalize_advantages_for_agent(advantages: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Normalize team GAE advantages using only one agent's active samples."""
    active = active_mask > 0.5
    if int(active.sum().detach().cpu().item()) <= 0:
        out = advantages.clone()
    else:
        active_advantages = advantages[active]
        out = (advantages - active_advantages.mean()) / (active_advantages.std(unbiased=False) + 1e-8)
    if not torch.isfinite(out).all():
        raise FloatingPointError("non-finite per-agent normalized advantages")
    return out


def validate_episode_accounting_3v3(record: dict[str, Any], env_index: int) -> None:
    """Validate the same 3v3 episode death-ledger invariants used by MAPPO."""
    for team in ("red", "blue"):
        fields = {
            "survivors": int(record[f"{team}_survivors"]),
            "attack_deaths": int(record[f"{team}_attack_deaths"]),
            "boundary_deaths": int(record[f"{team}_boundary_deaths"]),
            "friendly_collision_deaths": int(record[f"{team}_friendly_collision_deaths"]),
            "cross_collision_deaths": int(record[f"{team}_cross_collision_deaths"]),
            "boundary_altitude_deaths": int(record[f"{team}_boundary_altitude_deaths"]),
            "boundary_xy_deaths": int(record[f"{team}_boundary_xy_deaths"]),
        }
        total = (
            fields["survivors"]
            + fields["attack_deaths"]
            + fields["boundary_deaths"]
            + fields["friendly_collision_deaths"]
            + fields["cross_collision_deaths"]
        )
        if total != 3:
            raise RuntimeError(f"Death ledger mismatch for {team} env={env_index}: {fields} total={total} != 3")
        if fields["boundary_deaths"] != fields["boundary_altitude_deaths"] + fields["boundary_xy_deaths"]:
            raise RuntimeError(f"Boundary death mismatch for {team} env={env_index}: {fields}")
    if int(record["red_attack_kills"]) != int(record["blue_attack_deaths"]):
        raise RuntimeError(
            "Attack ledger mismatch env="
            f"{env_index}: red_attack_kills={record['red_attack_kills']} "
            f"blue_attack_deaths={record['blue_attack_deaths']}"
        )
    if int(record["blue_attack_kills"]) != int(record["red_attack_deaths"]):
        raise RuntimeError(
            "Attack ledger mismatch env="
            f"{env_index}: blue_attack_kills={record['blue_attack_kills']} "
            f"red_attack_deaths={record['red_attack_deaths']}"
        )


class HAPPO3v3Trainer:
    """Three independent red actors with sequential HAPPO updates."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.env_contract_config = load_config(self.env_config)
        self.best_score_schema = best_score_fields_for_config(self.env_contract_config)
        self.config = deepcopy(config)
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "fixed_rule_blue_3v3_happo":
            raise ValueError("training_mode must be fixed_rule_blue_3v3_happo")
        self.device = resolve_device(e["device"])
        torch.manual_seed(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.team_size = int(t.get("team_size", 3))
        self.num_envs = int(t["num_envs"])
        self.num_env_workers = int(t.get("num_env_workers", 4))
        self.rollout_steps = int(t["rollout_steps"])
        self.total_env_steps = int(t["total_env_steps"])
        self.observation_dims = [int(v) for v in t.get("observation_dims", [OBS_DIM] * self.team_size)]
        self.action_dims = [int(v) for v in t.get("action_dims", [3] * self.team_size)]
        if self.team_size != 3:
            raise ValueError("current 3v3 wrapper requires team_size=3")

        self.actors = IndependentHAPPOActors(
            self.observation_dims,
            self.action_dims,
            hidden_dim=int(n["hidden_dim"]),
            log_std_init=float(n["log_std_init"]),
            log_std_min=float(n.get("log_std_min", -5.0)),
            log_std_max=float(n.get("log_std_max", 2.0)),
        ).to(self.device)
        self.critic = CentralizedValueCritic(GS_DIM, int(n["hidden_dim"])).to(self.device)
        self.initial_actor_lr = float(t["actor_learning_rate"])
        self.final_actor_lr = float(t.get("actor_learning_rate_final", self.initial_actor_lr * 0.1))
        self.initial_critic_lr = float(t["critic_learning_rate"])
        self.final_critic_lr = float(t.get("critic_learning_rate_final", self.initial_critic_lr * 0.1))
        self.initial_entropy_coef = float(t.get("entropy_coef", 0.01))
        self.final_entropy_coef = float(t.get("entropy_coef_final", self.initial_entropy_coef * 0.1))
        self.current_actor_lr = self.initial_actor_lr
        self.current_critic_lr = self.initial_critic_lr
        self.current_entropy_coef = self.initial_entropy_coef
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.current_actor_lr)
            for actor in self.actors.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.current_critic_lr)
        self.vector_env = make_combat_vector_env_3v3(self.env_config, self.num_envs, self.num_env_workers)
        self.buffer = HAPPORolloutBuffer3v3(self.rollout_steps, self.num_envs)

        self.current_observations = np.empty((self.num_envs, 6, OBS_DIM), np.float32)
        self.current_global_states = np.empty((self.num_envs, GS_DIM), np.float32)
        self.current_alive_masks = np.empty((self.num_envs, 6), np.float32)
        self.episode_returns = np.zeros(self.num_envs, np.float64)
        self.episode_lengths = np.zeros(self.num_envs, np.int32)
        self.env_steps = 0
        self.vector_steps = 0
        self.update_count = 0
        self.last_agent_order: list[int] = []
        self.best_score: tuple[float, ...] | None = None
        self.best_evaluation: dict[str, Any] | None = None
        self.best_checkpoint_name: str | None = None
        self.evaluation_history: list[dict[str, Any]] = []
        self.last_rollout_reward_means: dict[str, float] = {}
        self.rule_policy_mapping_modes = self.vector_env.policy_modes()
        self.environment_metadata = self._environment_metadata()
        self.reset_environments()

    def _environment_metadata(self) -> dict[str, Any]:
        env_cfg = load_config(self.env_config)
        heterogeneous = env_cfg.get("heterogeneous", {})
        return {
            "env_config_sha256": sha256_file(self.env_config),
            "heterogeneous_enabled": bool(heterogeneous.get("enabled", False)),
            "role_mapping": deepcopy(heterogeneous.get("roles", {})),
            "sensor_range": deepcopy(heterogeneous.get("sensor_range", {})),
            "can_attack": deepcopy(heterogeneous.get("can_attack", {})),
            "support_rule": deepcopy(heterogeneous.get("support_rule", {})),
        }

    def reset_environments(self) -> None:
        specs = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in range(self.num_envs)]
        obs, gs, am = self.vector_env.reset(specs)
        self.current_observations = obs
        self.current_global_states = gs
        self.current_alive_masks = am
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)

    def _select_actions(self) -> tuple[np.ndarray, np.ndarray]:
        red_obs = torch.as_tensor(self.current_observations[:, :3, :], device=self.device)
        with torch.no_grad():
            actions, log_probs = self.actors.sample_actions(red_obs)
        actions_np = actions.cpu().numpy().astype(np.float32)
        log_probs_np = log_probs.cpu().numpy().astype(np.float32)
        actions_np *= self.current_alive_masks[:, :3, None]
        return actions_np, log_probs_np

    def collect_rollout(self, remaining: int | None = None) -> list[dict[str, Any]]:
        steps = self.rollout_steps if remaining is None else min(self.rollout_steps, remaining // self.num_envs)
        if steps <= 0:
            raise ValueError("remaining too small for one vector step")
        if self.buffer.rollout_steps != steps:
            self.buffer = HAPPORolloutBuffer3v3(steps, self.num_envs)
        self.buffer.clear()
        completed: list[dict[str, Any]] = []
        reward_component_sum = np.zeros(len(RED_REWARD_COMPONENT_KEYS_3V3), dtype=np.float64)
        reward_component_count = 0
        action_sum = np.zeros(3, dtype=np.float64)
        action_sat_sum = np.zeros(3, dtype=np.float64)
        action_count = 0

        for _ in range(steps):
            red_obs = self.current_observations[:, :3, :].copy()
            red_alive = self.current_alive_masks[:, :3].copy()
            with torch.no_grad():
                value = self.critic(torch.as_tensor(self.current_global_states, device=self.device)).cpu().numpy()
            actions, log_probs = self._select_actions()
            alive_actions = actions[red_alive.astype(bool)]
            if alive_actions.size:
                action_sum += alive_actions.sum(axis=0)
                action_sat_sum += (np.abs(alive_actions) >= 0.95).sum(axis=0)
                action_count += alive_actions.shape[0]
            r: VectorStepResult3v3 = self.vector_env.step(actions)
            reward_component_sum += r.red_reward_components.sum(axis=0)
            reward_component_count += self.num_envs
            self.episode_returns += r.team_rewards
            self.episode_lengths += 1
            done = r.terminated | r.truncated
            done_idx = np.where(done)[0]
            for idx in np.sort(done_idx):
                if r.episode_valid[idx]:
                    rec = {
                        "episode_return": float(self.episode_returns[idx]),
                        "episode_length": int(self.episode_lengths[idx]),
                        "red_complete_elimination_success": bool(r.red_complete_elimination_success[idx]),
                        "blue_complete_elimination_success": bool(r.blue_complete_elimination_success[idx]),
                        "environment_outcome": decode_3v3_outcome(int(r.outcome_codes[idx])),
                        "termination_reason": decode_3v3_termination_reason(int(r.termination_reason_codes[idx])),
                        "red_attack_kills": int(r.episode_red_attack_kills[idx]),
                        "blue_attack_kills": int(r.episode_blue_attack_kills[idx]),
                        "red_survivors": int(r.episode_red_survivors[idx]),
                        "blue_survivors": int(r.episode_blue_survivors[idx]),
                        "red_attack_deaths": int(r.episode_red_attack_deaths[idx]),
                        "blue_attack_deaths": int(r.episode_blue_attack_deaths[idx]),
                        "red_boundary_deaths": int(r.episode_red_boundary_deaths[idx]),
                        "blue_boundary_deaths": int(r.episode_blue_boundary_deaths[idx]),
                        "red_boundary_altitude_deaths": int(r.episode_red_boundary_altitude_deaths[idx]),
                        "blue_boundary_altitude_deaths": int(r.episode_blue_boundary_altitude_deaths[idx]),
                        "red_boundary_xy_deaths": int(r.episode_red_boundary_xy_deaths[idx]),
                        "blue_boundary_xy_deaths": int(r.episode_blue_boundary_xy_deaths[idx]),
                        "red_friendly_collision_deaths": int(r.episode_red_friendly_collision_deaths[idx]),
                        "blue_friendly_collision_deaths": int(r.episode_blue_friendly_collision_deaths[idx]),
                        "red_cross_collision_deaths": int(r.episode_red_cross_collision_deaths[idx]),
                        "blue_cross_collision_deaths": int(r.episode_blue_cross_collision_deaths[idx]),
                        "red_kills_with_shared_observation": int(r.episode_red_kills_with_shared_observation[idx]),
                        "blue_kills_with_shared_observation": int(r.episode_blue_kills_with_shared_observation[idx]),
                        "red_mean_support_coverage_ratio": float(r.episode_red_mean_support_coverage_ratio[idx]),
                        "blue_mean_support_coverage_ratio": float(r.episode_blue_mean_support_coverage_ratio[idx]),
                        "red_support_survived": bool(r.episode_red_support_survived[idx]),
                        "blue_support_survived": bool(r.episode_blue_support_survived[idx]),
                        "red_any_attack_kill": bool(r.episode_red_any_attack_kill[idx]),
                        "blue_any_attack_kill": bool(r.episode_blue_any_attack_kill[idx]),
                        "red_first_attack_kill_step": None if int(r.episode_red_first_attack_kill_step[idx]) < 0 else int(r.episode_red_first_attack_kill_step[idx]),
                        "blue_first_attack_kill_step": None if int(r.episode_blue_first_attack_kill_step[idx]) < 0 else int(r.episode_blue_first_attack_kill_step[idx]),
                        "red_second_attack_kill_step": None if int(r.episode_red_second_attack_kill_step[idx]) < 0 else int(r.episode_red_second_attack_kill_step[idx]),
                        "blue_second_attack_kill_step": None if int(r.episode_blue_second_attack_kill_step[idx]) < 0 else int(r.episode_blue_second_attack_kill_step[idx]),
                        "red_third_attack_kill_step": None if int(r.episode_red_third_attack_kill_step[idx]) < 0 else int(r.episode_red_third_attack_kill_step[idx]),
                        "blue_third_attack_kill_step": None if int(r.episode_blue_third_attack_kill_step[idx]) < 0 else int(r.episode_blue_third_attack_kill_step[idx]),
                        "red_r3_active_steps": int(r.episode_red_r3_active_steps[idx]),
                        "blue_r3_active_steps": int(r.episode_blue_r3_active_steps[idx]),
                        "red_r41_active_steps": int(r.episode_red_r41_active_steps[idx]),
                        "blue_r41_active_steps": int(r.episode_blue_r41_active_steps[idx]),
                        "red_r42_active_steps": int(r.episode_red_r42_active_steps[idx]),
                        "blue_r42_active_steps": int(r.episode_blue_r42_active_steps[idx]),
                        "red_attack_window_steps": int(r.episode_red_attack_window_steps[idx]),
                        "blue_attack_window_steps": int(r.episode_blue_attack_window_steps[idx]),
                    }
                    validate_episode_accounting_3v3(rec, int(idx))
                    completed.append(rec)
                self.episode_returns[idx] = 0.0
                self.episode_lengths[idx] = 0
            if len(done_idx) > 0:
                seeds = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in done_idx]
                no, ng, na = self.vector_env.reset_at(done_idx, seeds)
                r.observations[done_idx] = no
                r.global_states[done_idx] = ng
                r.alive_masks[done_idx] = na
            self.buffer.add(
                red_obs,
                self.current_global_states.copy(),
                actions,
                log_probs,
                red_alive,
                r.team_rewards,
                value,
                done,
            )
            self.current_observations = r.observations
            self.current_global_states = r.global_states
            self.current_alive_masks = r.alive_masks
            self.env_steps += self.num_envs
            self.vector_steps += 1

        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.current_global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(
            last_values, float(self.config["training"]["gamma"]), float(self.config["training"]["gae_lambda"])
        )
        if reward_component_count > 0:
            means = {
                key: float(value)
                for key, value in zip(RED_REWARD_COMPONENT_KEYS_3V3, reward_component_sum / reward_component_count)
            }
            self.last_rollout_reward_means = {
                **means,
                **{f"mean_rollout_{key}": value for key, value in means.items()},
                "mean_rollout_approach_reward": means.get("red_approach_reward", 0.0),
                "mean_rollout_attack_advantage_reward": means.get("red_attack_advantage_reward", 0.0),
                "mean_rollout_threat_penalty": means.get("red_threat_penalty", 0.0),
                "mean_rollout_dense_reward": means.get("red_dense_reward", 0.0),
                "mean_rollout_event_reward": means.get("red_event_reward", 0.0),
                "mean_rollout_terminal_reward": means.get("red_terminal_reward", 0.0),
                "mean_rollout_total_step_reward": means.get("red_team_total_reward", 0.0),
            }
            if action_count > 0:
                action_mean = action_sum / action_count
                action_sat = action_sat_sum / action_count
                for dim, name in enumerate(("yaw", "pitch", "speed")):
                    self.last_rollout_reward_means[f"sampled_action_mean_{name}"] = float(action_mean[dim])
                    self.last_rollout_reward_means[f"sampled_action_saturation_rate_{name}"] = float(action_sat[dim])
        else:
            self.last_rollout_reward_means = {}
        return completed

    def _set_schedules(self) -> None:
        t = self.config["training"]
        progress = float(np.clip(self.env_steps / max(1, self.total_env_steps), 0.0, 1.0))
        self.current_actor_lr = linear_schedule(self.initial_actor_lr, self.final_actor_lr, progress)
        self.current_critic_lr = linear_schedule(self.initial_critic_lr, self.final_critic_lr, progress)
        self.current_entropy_coef = linear_schedule(self.initial_entropy_coef, self.final_entropy_coef, progress)
        for opt in self.actor_optimizers:
            for group in opt.param_groups:
                group["lr"] = self.current_actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.current_critic_lr

    def update(self) -> dict[str, Any]:
        self._set_schedules()
        t = self.config["training"]
        flat_obs = torch.as_tensor(self.buffer.observations.reshape(-1, self.team_size, OBS_DIM), device=self.device)
        flat_states = torch.as_tensor(self.buffer.global_states.reshape(-1, GS_DIM), device=self.device)
        flat_actions = torch.as_tensor(self.buffer.actions.reshape(-1, self.team_size, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1, self.team_size), device=self.device)
        alive = torch.as_tensor(self.buffer.agent_alive_masks.reshape(-1, self.team_size), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device)
        factor = torch.ones_like(advantages)
        agent_order = [int(v) for v in self.rng.permutation(self.team_size)]
        self.last_agent_order = agent_order

        actor_rows: list[dict[str, float | int]] = []
        clip_coef = float(t["clip_coef"])
        minibatch_size = int(t["minibatch_size"])
        ppo_epochs = int(t["ppo_epochs"])
        max_grad_norm = float(t["max_grad_norm"])
        total_samples = len(advantages)

        for agent_id in agent_order:
            active = alive[:, agent_id] > 0.5
            if int(active.sum().item()) <= 0:
                factor = factor.detach()
                actor_rows.append({"agent_id": agent_id, "active_samples": 0})
                continue
            normalized_advantages_i = normalize_advantages_for_agent(advantages, active.float())
            for _ in range(ppo_epochs):
                order = self.rng.permutation(total_samples)
                for start in range(0, total_samples, minibatch_size):
                    idx_np = order[start:start + minibatch_size]
                    idx = torch.as_tensor(idx_np, device=self.device)
                    idx = idx[active[idx]]
                    if len(idx) == 0:
                        continue
                    new_lp, entropy = self.actors.evaluate_agent_actions(
                        agent_id, flat_obs[idx, agent_id], flat_actions[idx, agent_id]
                    )
                    log_ratio = new_lp - old_log_probs[idx, agent_id]
                    ratio = log_ratio.exp()
                    effective_adv = (factor[idx] * normalized_advantages_i[idx]).detach()
                    policy_loss = ppo_clipped_policy_loss(ratio, effective_adv, clip_coef)
                    loss = policy_loss - self.current_entropy_coef * entropy.mean()
                    opt = self.actor_optimizers[agent_id]
                    opt.zero_grad()
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(self.actors.actors[agent_id].parameters(), max_grad_norm)
                    opt.step()
                    self.actors.actors[agent_id].clamp_log_std_()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
                    actor_rows.append({
                        "agent_id": agent_id,
                        "policy_loss": float(policy_loss.item()),
                        "entropy": float(entropy.mean().item()),
                        "approx_kl": float(approx_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "ratio_mean": float(ratio.mean().item()),
                        "ratio_min": float(ratio.min().item()),
                        "ratio_max": float(ratio.max().item()),
                        "factor_mean": float(factor[idx].mean().item()),
                        "factor_min": float(factor[idx].min().item()),
                        "factor_max": float(factor[idx].max().item()),
                        "actor_grad_norm": float(grad),
                        "active_samples": int(len(idx)),
                    })
            with torch.no_grad():
                new_lp_all, _ = self.actors.evaluate_agent_actions(
                    agent_id, flat_obs[:, agent_id], flat_actions[:, agent_id]
                )
                factor = happo_preceding_factor_update(
                    factor, old_log_probs[:, agent_id], new_lp_all, active.float()
                )

        critic_losses: list[float] = []
        value_preds_before = self.critic(flat_states).detach().cpu().numpy()
        for _ in range(ppo_epochs):
            order = self.rng.permutation(total_samples)
            for start in range(0, total_samples, minibatch_size):
                idx = torch.as_tensor(order[start:start + minibatch_size], device=self.device)
                values = self.critic(flat_states[idx])
                value_loss = (values - returns[idx]).square().mean()
                loss = float(t["value_loss_coef"]) * value_loss
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(float(value_loss.item()))

        self.update_count += 1
        metrics = self._summarize_update(actor_rows, critic_losses, value_preds_before, returns.detach().cpu().numpy())
        metrics.update(self.last_rollout_reward_means)
        if not all(np.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float, np.integer, np.floating))):
            raise FloatingPointError(f"non-finite HAPPO metrics: {metrics}")
        return metrics

    def _summarize_update(
        self,
        actor_rows: list[dict[str, float | int]],
        critic_losses: list[float],
        value_preds_before: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, Any]:
        nonempty = [r for r in actor_rows if int(r.get("active_samples", 0)) > 0]
        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in nonempty])) if nonempty else 0.0
        log_std_dim = self.actors.effective_log_std_by_dim
        std_dim = self.actors.effective_std_by_dim
        return {
            "update": self.update_count,
            "agent_update_order": list(self.last_agent_order),
            "actor_updates": len(nonempty),
            "agents_updated": len({int(r["agent_id"]) for r in nonempty}),
            "policy_loss": mean("policy_loss"),
            "entropy": mean("entropy"),
            "approx_kl": mean("approx_kl"),
            "clip_fraction": mean("clip_fraction"),
            "ratio_mean": mean("ratio_mean"),
            "ratio_min": float(np.min([float(r["ratio_min"]) for r in nonempty])) if nonempty else 1.0,
            "ratio_max": float(np.max([float(r["ratio_max"]) for r in nonempty])) if nonempty else 1.0,
            "factor_mean": mean("factor_mean"),
            "factor_min": float(np.min([float(r["factor_min"]) for r in nonempty])) if nonempty else 1.0,
            "factor_max": float(np.max([float(r["factor_max"]) for r in nonempty])) if nonempty else 1.0,
            "actor_grad_norm": mean("actor_grad_norm"),
            "alive_actor_samples": int(sum(int(r.get("active_samples", 0)) for r in actor_rows)),
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "value_mean": float(np.mean(value_preds_before)),
            "explained_variance": explained_variance(value_preds_before, returns),
            "advantage_mean": float(np.mean(self.buffer.advantages)),
            "advantage_std": float(np.std(self.buffer.advantages)),
            "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef,
            "effective_log_std_yaw": log_std_dim[0],
            "effective_log_std_pitch": log_std_dim[1],
            "effective_log_std_speed": log_std_dim[2],
            "effective_std_yaw": std_dim[0],
            "effective_std_pitch": std_dim[1],
            "effective_std_speed": std_dim[2],
        }

    def training_signature(self) -> dict[str, Any]:
        t, n = self.config["training"], self.config["network"]
        return {
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_3V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_3V3,
            "env_config_sha256": sha256_file(self.env_config),
            "observation_dims": list(self.observation_dims),
            "action_dims": list(self.action_dims),
            "team_size": self.team_size,
            "global_state_dim": GS_DIM,
            "network": deepcopy(n),
            "training": {
                key: t[key] for key in (
                    "rollout_steps", "num_envs", "ppo_epochs", "minibatch_size",
                    "gamma", "gae_lambda", "clip_coef", "actor_learning_rate",
                    "critic_learning_rate", "value_loss_coef", "entropy_coef",
                    "max_grad_norm",
                )
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_3V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_3V3,
            "training_signature": self.training_signature(),
            "config": self.config,
            "env_config": self.env_config,
            "actors": [actor.state_dict() for actor in self.actors.actors],
            "critic": self.critic.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "last_agent_order": self.last_agent_order,
            "best_score": self.best_score,
            "best_score_schema": list(self.best_score_schema),
            "best_evaluation": self.best_evaluation,
            "best_checkpoint_name": self.best_checkpoint_name,
            "evaluation_history": self.evaluation_history,
            "rule_policy_mapping_modes": self.rule_policy_mapping_modes,
            "environment_metadata": self.environment_metadata,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_HAPPO_3V3:
            raise RuntimeError(f"Expected {CHECKPOINT_FAMILY_HAPPO_3V3}, got {ckpt.get('checkpoint_family')}")
        diffs = signature_mismatches(ckpt.get("training_signature"), self.training_signature())
        if diffs:
            raise RuntimeError("checkpoint signature mismatch:\n" + "\n".join(diffs))
        for actor, state in zip(self.actors.actors, ckpt["actors"]):
            actor.load_state_dict(state)
        self.critic.load_state_dict(ckpt["critic"])
        for opt, state in zip(self.actor_optimizers, ckpt["actor_optimizers"]):
            opt.load_state_dict(state)
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"])
        self.vector_steps = int(ckpt["vector_steps"])
        self.update_count = int(ckpt["update_count"])
        self.last_agent_order = [int(v) for v in ckpt.get("last_agent_order", [])]
        self.best_score_schema = infer_best_score_schema_for_checkpoint(ckpt, self.env_contract_config)
        self.best_score = ckpt.get("best_score")
        self.best_evaluation = ckpt.get("best_evaluation")
        if self.best_evaluation is not None:
            current_schema = best_score_fields_for_config(self.env_contract_config)
            if tuple(self.best_score_schema) != tuple(current_schema) or self.best_score is None:
                self.best_score_schema = current_schema
                self.best_score = compute_best_score_for_config(self.best_evaluation, self.env_contract_config)
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.evaluation_history = ckpt.get("evaluation_history", [])
        self.reset_environments()
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])

    def close(self) -> None:
        self.vector_env.close()


__all__ = [
    "CHECKPOINT_FAMILY_HAPPO_3V3",
    "CHECKPOINT_VERSION_HAPPO_3V3",
    "HAPPO3v3Trainer",
    "compute_best_score",
    "compute_best_score_fields",
    "happo_preceding_factor_update",
    "normalize_advantages_for_agent",
    "ppo_clipped_policy_loss",
    "signature_mismatches",
    "validate_episode_accounting_3v3",
]
