"""Homogeneous 3v3 MADSAC trainer against fixed-rule blue."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..environment_3v3 import OBS_DIM
from ..mappo.trainer_3v3 import compute_best_score, resolve_device
from ..mappo.vector_env_3v3 import (
    decode_3v3_outcome,
    decode_3v3_termination_reason,
    make_combat_vector_env_3v3,
)
from .networks import SharedSquashedGaussianActor, TwinAttentionCritic
from .replay_buffer import MADSACReplayBuffer

CHECKPOINT_FAMILY_MADSAC_3V3 = "homogeneous_3v3_fixed_blue_madsac"
CHECKPOINT_VERSION_MADSAC_3V3 = 1


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    denom = mask.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return None
    return (values * mask).sum() / denom.clamp_min(1.0)


def soft_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def set_requires_grad_(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(value)


class MADSAC3v3Trainer:
    """Shared-red-actor MADSAC using the existing 3v3 vector env."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.config = deepcopy(config)
        e, n, t = self.config["experiment"], self.config["network"], self.config["training"]
        if t.get("training_mode") not in ("fixed_rule_blue_3v3_madsac", "madsac_fixed_rule_blue_3v3"):
            raise ValueError("training_mode must be fixed_rule_blue_3v3_madsac")
        self.device = resolve_device(e["device"])
        torch.manual_seed(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.num_envs = int(t["num_envs"])
        self.num_env_workers = int(t.get("num_env_workers", 4))
        self.total_env_steps = int(t["total_env_steps"])
        self.batch_size = int(t["batch_size"])
        self.learning_starts = int(t["learning_starts"])
        self.gradient_steps = int(t.get("gradient_steps", t.get("updates_per_step", 1)))
        self.gamma = float(t["gamma"])
        self.tau = float(t["tau"])
        self.alpha = float(t["alpha"])
        self.policy_delay = int(t.get("policy_delay", 2))
        self.max_actor_grad_norm = float(t.get("max_actor_grad_norm", t.get("max_grad_norm", 10.0)))
        self.max_critic_grad_norm = float(t.get("max_critic_grad_norm", t.get("max_grad_norm", 10.0)))

        actor_hidden = int(n.get("actor_hidden_dim", n.get("hidden_dim", 256)))
        critic_hidden = int(n.get("critic_hidden_dim", n.get("hidden_dim", 256)))
        heads = int(n.get("attention_heads", 2))
        log_std_bias = float(n.get("log_std_bias_init", n.get("log_std_init", -0.5)))
        self.actor = SharedSquashedGaussianActor(
            OBS_DIM, 3, 3, actor_hidden, log_std_bias,
            float(n.get("log_std_min", -5.0)), float(n.get("log_std_max", 2.0)),
        ).to(self.device)
        self.target_actor = SharedSquashedGaussianActor(
            OBS_DIM, 3, 3, actor_hidden, log_std_bias,
            float(n.get("log_std_min", -5.0)), float(n.get("log_std_max", 2.0)),
        ).to(self.device)
        self.critic = TwinAttentionCritic(OBS_DIM, 3, 3, critic_hidden, heads).to(self.device)
        self.target_critic = TwinAttentionCritic(OBS_DIM, 3, 3, critic_hidden, heads).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        set_requires_grad_(self.target_actor, False)
        set_requires_grad_(self.target_critic, False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(t["actor_learning_rate"]))
        self.critic1_optimizer = torch.optim.Adam(self.critic.q1.parameters(), lr=float(t["critic_learning_rate"]))
        self.critic2_optimizer = torch.optim.Adam(self.critic.q2.parameters(), lr=float(t["critic_learning_rate"]))
        self.replay = MADSACReplayBuffer(int(t.get("replay_capacity", t.get("replay_size"))), OBS_DIM, 3, 3)
        self.vector_env = make_combat_vector_env_3v3(self.env_config, self.num_envs, self.num_env_workers)
        self.rule_policy_mapping_modes = self.vector_env.policy_modes()

        self.current_observations = np.empty((self.num_envs, 6, OBS_DIM), np.float32)
        self.current_alive_masks = np.empty((self.num_envs, 6), np.float32)
        self.env_steps = 0
        self.vector_steps = 0
        self.critic_update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0
        self.best_score: tuple[float, ...] | None = None
        self.best_evaluation: dict[str, Any] | None = None
        self.best_checkpoint_name: str | None = None
        self.evaluation_history: list[dict[str, Any]] = []
        self.replay_restored = False
        self.episode_returns = np.zeros(self.num_envs, np.float64)
        self.episode_lengths = np.zeros(self.num_envs, np.int32)
        self.last_metrics: dict[str, float | bool] = {}
        self.reset_environments()

    def reset_environments(self) -> None:
        specs = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in range(self.num_envs)]
        obs, _, am = self.vector_env.reset(specs)
        self.current_observations = obs
        self.current_alive_masks = am
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)

    def _select_actions(self) -> np.ndarray:
        alive = self.current_alive_masks[:, :3].astype(np.float32)
        if self.env_steps < self.learning_starts:
            actions = self.rng.uniform(-1.0, 1.0, size=(self.num_envs, 3, 3)).astype(np.float32)
        else:
            obs_t = torch.as_tensor(self.current_observations[:, :3, :], device=self.device)
            with torch.no_grad():
                actions, _ = self.actor.sample(obs_t)
            actions = actions.cpu().numpy().astype(np.float32)
        actions *= alive[:, :, None]
        return actions

    def step_environment(self) -> list[dict[str, Any]]:
        red_obs = self.current_observations[:, :3, :].copy()
        red_alive = self.current_alive_masks[:, :3].copy()
        actions = self._select_actions()
        result = self.vector_env.step(actions)
        next_red_obs = result.observations[:, :3, :].copy()
        next_red_alive = result.alive_masks[:, :3].copy()
        self.replay.add_batch(
            red_obs, actions, result.team_rewards, next_red_obs, red_alive, next_red_alive,
            result.terminated, result.truncated,
        )
        self.episode_returns += result.team_rewards
        self.episode_lengths += 1
        completed: list[dict[str, Any]] = []
        done_idx = np.where(result.terminated | result.truncated)[0]
        for idx in done_idx:
            if result.episode_valid[idx]:
                reason = decode_3v3_termination_reason(int(result.termination_reason_codes[idx]))
                outcome = decode_3v3_outcome(int(result.outcome_codes[idx]))
                rec = {
                    "episode_return": float(self.episode_returns[idx]),
                    "episode_length": int(self.episode_lengths[idx]),
                    "red_complete_elimination_success": bool(result.red_complete_elimination_success[idx]),
                    "blue_complete_elimination_success": bool(result.blue_complete_elimination_success[idx]),
                    "environment_outcome": outcome,
                    "termination_reason": reason,
                    "red_attack_kills": int(result.episode_red_attack_kills[idx]),
                    "blue_attack_kills": int(result.episode_blue_attack_kills[idx]),
                    "red_survivors": int(result.episode_red_survivors[idx]),
                    "blue_survivors": int(result.episode_blue_survivors[idx]),
                    "red_boundary_deaths": int(result.episode_red_boundary_deaths[idx]),
                    "red_boundary_altitude_deaths": int(result.episode_red_boundary_altitude_deaths[idx]),
                    "red_boundary_xy_deaths": int(result.episode_red_boundary_xy_deaths[idx]),
                    "red_friendly_collision_deaths": int(result.episode_red_friendly_collision_deaths[idx]),
                    "red_cross_collision_deaths": int(result.episode_red_cross_collision_deaths[idx]),
                }
                completed.append(rec)
            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0
        if len(done_idx) > 0:
            seeds = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in done_idx]
            no, _, na = self.vector_env.reset_at(done_idx, seeds)
            result.observations[done_idx] = no
            result.alive_masks[done_idx] = na
        self.current_observations = result.observations
        self.current_alive_masks = result.alive_masks
        self.env_steps += self.num_envs
        self.vector_steps += 1
        return completed

    def compute_td_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probs = self.target_actor.sample(batch["next_observations"])
            next_actions = next_actions * batch["next_alive_masks"].unsqueeze(-1)
            next_log_probs = next_log_probs * batch["next_alive_masks"]
            tq1, tq2 = self.target_critic(batch["next_observations"], next_actions, batch["next_alive_masks"])
            target_q = torch.minimum(tq1, tq2)
            done = batch["done_for_bootstrap"].float().unsqueeze(-1)
            y = (
                batch["team_rewards"].unsqueeze(-1)
                + self.gamma
                * (1.0 - done)
                * batch["next_alive_masks"]
                * (target_q - self.alpha * next_log_probs)
            )
        return y.detach()

    def update(self) -> dict[str, float | bool]:
        if self.replay.size < self.batch_size or self.env_steps < self.learning_starts:
            return {}
        rows: list[dict[str, float | bool]] = []
        for _ in range(self.gradient_steps):
            batch = self.replay.sample(self.batch_size, self.rng, self.device)
            alive = batch["alive_masks"]
            if float(alive.sum().detach().cpu()) <= 0.0:
                continue
            y = self.compute_td_target(batch)
            q1, q2 = self.critic(batch["observations"], batch["actions"], alive)
            critic1_loss = masked_mean((q1 - y).square(), alive)
            critic2_loss = masked_mean((q2 - y).square(), alive)
            if critic1_loss is None or critic2_loss is None:
                continue
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            critic1_grad = nn.utils.clip_grad_norm_(self.critic.q1.parameters(), self.max_critic_grad_norm)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            critic2_grad = nn.utils.clip_grad_norm_(self.critic.q2.parameters(), self.max_critic_grad_norm)
            self.critic2_optimizer.step()
            self.critic_update_count += 1

            actor_updated = False
            target_updated = False
            actor_loss_value = 0.0
            actor_grad_value = 0.0
            sampled_log_prob_mean = 0.0
            deterministic_abs_mean = 0.0
            stochastic_abs_mean = 0.0
            saturation_fraction = 0.0
            if self.critic_update_count % self.policy_delay == 0:
                set_requires_grad_(self.critic, False)
                actions_pi, log_probs_pi = self.actor.sample(batch["observations"])
                actions_pi = actions_pi * alive.unsqueeze(-1)
                log_probs_pi = log_probs_pi * alive
                q1_pi, q2_pi = self.critic(batch["observations"], actions_pi, alive)
                q_pi = torch.minimum(q1_pi, q2_pi)
                actor_loss = masked_mean(self.alpha * log_probs_pi - q_pi, alive)
                if actor_loss is not None:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_actor_grad_norm)
                    self.actor_optimizer.step()
                    actor_updated = True
                    self.actor_update_count += 1
                    actor_loss_value = float(actor_loss.item())
                    actor_grad_value = float(actor_grad)
                    sampled_log_prob_mean = float(masked_mean(log_probs_pi, alive).item())
                    with torch.no_grad():
                        det = self.actor.deterministic(batch["observations"]) * alive.unsqueeze(-1)
                        deterministic_abs_mean = float((det.abs() * alive.unsqueeze(-1)).sum().item() / (alive.sum().item() * 3.0))
                        stochastic_abs_mean = float((actions_pi.abs() * alive.unsqueeze(-1)).sum().item() / (alive.sum().item() * 3.0))
                        saturation_fraction = float(((actions_pi.abs() > 0.98).float() * alive.unsqueeze(-1)).sum().item() / (alive.sum().item() * 3.0))
                    soft_update_(self.target_actor, self.actor, self.tau)
                    soft_update_(self.target_critic, self.critic, self.tau)
                    self.target_update_count += 1
                    target_updated = True
                set_requires_grad_(self.critic, True)

            td_error = (torch.minimum(q1, q2) - y).abs()
            td_err_mean = masked_mean(td_error, alive)
            q_gap = masked_mean((q1 - q2).abs(), alive)
            q1_mean = masked_mean(q1, alive)
            q2_mean = masked_mean(q2, alive)
            target_mean = masked_mean(y, alive)
            row = {
                "critic1_loss": float(critic1_loss.item()),
                "critic2_loss": float(critic2_loss.item()),
                "actor_loss": actor_loss_value,
                "q1_mean": float(q1_mean.item()) if q1_mean is not None else 0.0,
                "q2_mean": float(q2_mean.item()) if q2_mean is not None else 0.0,
                "target_q_mean": float(target_mean.item()) if target_mean is not None else 0.0,
                "q1_q2_abs_gap": float(q_gap.item()) if q_gap is not None else 0.0,
                "td_error_abs_mean": float(td_err_mean.item()) if td_err_mean is not None else 0.0,
                "sampled_log_prob_mean": sampled_log_prob_mean,
                "deterministic_action_abs_mean": deterministic_abs_mean,
                "stochastic_action_abs_mean": stochastic_abs_mean,
                "action_saturation_fraction": saturation_fraction,
                "actor_grad_norm": actor_grad_value,
                "critic1_grad_norm": float(critic1_grad),
                "critic2_grad_norm": float(critic2_grad),
                "alpha": self.alpha,
                "actor_updated": actor_updated,
                "target_updated": target_updated,
                "replay_size": float(self.replay.size),
            }
            numeric = [float(v) for v in row.values() if isinstance(v, (int, float, np.floating))]
            if not all(np.isfinite(v) for v in numeric):
                raise FloatingPointError(f"non-finite MADSAC update metrics: {row}")
            rows.append(row)
        if not rows:
            return {}
        out: dict[str, float | bool] = {}
        for key in rows[0]:
            vals = [r[key] for r in rows]
            if isinstance(vals[0], bool):
                out[key] = bool(any(vals))
            else:
                out[key] = float(np.mean(vals))
        self.last_metrics = out
        return out

    def train_until(self, total_env_steps: int | None = None) -> list[dict[str, Any]]:
        target = self.total_env_steps if total_env_steps is None else int(total_env_steps)
        completed: list[dict[str, Any]] = []
        while self.env_steps < target:
            completed.extend(self.step_environment())
            self.update()
        return completed

    def training_signature(self) -> dict[str, Any]:
        n, t = self.config["network"], self.config["training"]
        return {
            "checkpoint_family": CHECKPOINT_FAMILY_MADSAC_3V3,
            "checkpoint_version": CHECKPOINT_VERSION_MADSAC_3V3,
            "observation_dim": OBS_DIM,
            "action_dim": 3,
            "team_size": 3,
            "network": {
                "actor_hidden_dim": n.get("actor_hidden_dim", n.get("hidden_dim", 256)),
                "critic_hidden_dim": n.get("critic_hidden_dim", n.get("hidden_dim", 256)),
                "attention_heads": n.get("attention_heads", 2),
                "log_std_min": n.get("log_std_min", -5.0),
                "log_std_max": n.get("log_std_max", 2.0),
                "log_std_bias_init": n.get("log_std_bias_init", n.get("log_std_init", -0.5)),
            },
            "hyperparameters": {
                "gamma": t["gamma"],
                "tau": t["tau"],
                "alpha": t["alpha"],
                "policy_delay": t.get("policy_delay", 2),
                "batch_size": t["batch_size"],
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_family": CHECKPOINT_FAMILY_MADSAC_3V3,
            "checkpoint_version": CHECKPOINT_VERSION_MADSAC_3V3,
            "training_signature": self.training_signature(),
            "config": self.config,
            "env_config": self.env_config,
            "online_actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "online_critic1": self.critic.q1.state_dict(),
            "online_critic2": self.critic.q2.state_dict(),
            "target_critic1": self.target_critic.q1.state_dict(),
            "target_critic2": self.target_critic.q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "critic_update_count": self.critic_update_count,
            "actor_update_count": self.actor_update_count,
            "target_update_count": self.target_update_count,
            "best_score": self.best_score,
            "best_evaluation": self.best_evaluation,
            "best_checkpoint_name": self.best_checkpoint_name,
            "evaluation_history": self.evaluation_history,
            "rule_policy_mapping_modes": self.rule_policy_mapping_modes,
            "replay_metadata": self.replay.metadata(include_full_replay=False),
            "replay_restored": self.replay_restored,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_MADSAC_3V3:
            raise RuntimeError(f"Expected {CHECKPOINT_FAMILY_MADSAC_3V3}, got {ckpt.get('checkpoint_family')}")
        if ckpt.get("training_signature") != self.training_signature():
            raise RuntimeError("MADSAC checkpoint signature mismatch")
        self.actor.load_state_dict(ckpt["online_actor"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.critic.q1.load_state_dict(ckpt["online_critic1"])
        self.critic.q2.load_state_dict(ckpt["online_critic2"])
        self.target_critic.q1.load_state_dict(ckpt["target_critic1"])
        self.target_critic.q2.load_state_dict(ckpt["target_critic2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(ckpt["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(ckpt["critic2_optimizer"])
        self.env_steps = int(ckpt["env_steps"])
        self.vector_steps = int(ckpt["vector_steps"])
        self.critic_update_count = int(ckpt["critic_update_count"])
        self.actor_update_count = int(ckpt["actor_update_count"])
        self.target_update_count = int(ckpt.get("target_update_count", 0))
        self.best_score = ckpt.get("best_score")
        self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.evaluation_history = ckpt.get("evaluation_history", [])
        self.replay_restored = False
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
        self.reset_environments()

    def close(self) -> None:
        self.vector_env.close()


__all__ = [
    "CHECKPOINT_FAMILY_MADSAC_3V3",
    "CHECKPOINT_VERSION_MADSAC_3V3",
    "MADSAC3v3Trainer",
    "compute_best_score",
    "masked_mean",
    "soft_update_",
]
