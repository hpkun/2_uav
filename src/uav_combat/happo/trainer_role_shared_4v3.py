"""Role-shared Support/Combat HAPPO trainer for the frozen v12 4v3 task."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import load_config
from ..environment_4v3_v12 import GS_DIM_V12, OBS_DIM_V12
from ..mappo.trainer_3v3 import linear_schedule, resolve_device
from ..mappo.vector_env_4v3_v12 import REWARD_COMPONENT_KEYS_V12, make_combat_vector_env_4v3_v12
from ..scenario_4v3_v12 import resolved_reward_contract_v12
from .metrics import explained_variance
from .networks import CentralizedValueCritic
from .role_shared_buffer import RoleSharedRolloutBuffer4v3, SequenceChunk
from .role_shared_networks import RoleHiddenState, RoleSharedHAPPOActors
from .trainer_3v3 import ppo_clipped_policy_loss, sha256_file, signature_mismatches
from .trainer_4v3 import compute_best_score_4v3

CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3 = "functional_heterogeneous_4v3_role_shared_happo"
CHECKPOINT_VERSION_ROLE_SHARED_HAPPO_4V3 = 1
ROLE_POLICY_MAPPING = {"red_0": "support", "red_1": "combat_shared", "red_2": "combat_shared", "red_3": "combat_shared"}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def combat_joint_log_probability(log_probs: torch.Tensor, alive_masks: torch.Tensor) -> torch.Tensor:
    if log_probs.shape[-1] != 3 or alive_masks.shape != log_probs.shape:
        raise ValueError("Combat log probabilities and masks must have shape [...,3]")
    return (log_probs * alive_masks.to(log_probs.dtype)).sum(dim=-1)


def combat_alive_mean_entropy(entropy: torch.Tensor, alive_masks: torch.Tensor) -> torch.Tensor:
    if entropy.shape != alive_masks.shape or entropy.shape[-1] != 3:
        raise ValueError("Combat entropy and masks must have shape [...,3]")
    alive = alive_masks.to(entropy.dtype)
    return (entropy * alive).sum(-1) / alive.sum(-1).clamp_min(1.0)


def role_group_factor_update(
    factor: torch.Tensor,
    old_group_log_prob: torch.Tensor,
    new_group_log_prob: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    ratio = torch.exp(new_group_log_prob - old_group_log_prob)
    ratio = torch.where(active_mask > 0.5, ratio, torch.ones_like(ratio))
    return (factor * ratio).detach()


def _normalize_group_advantage(advantages: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    selected = advantages[active > 0.5]
    if selected.numel() == 0:
        return torch.zeros_like(advantages)
    result = (advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite role-group advantage")
    return result


class RoleSharedHAPPO4v3Trainer:
    """One Support policy and one shared Combat policy with HAPPO ordering."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.env_contract_config = load_config(self.env_config)
        self.config = deepcopy(config)
        if self.env_contract_config.get("combat", {}).get("reward_contract_version") != "v12_soft_boundary_combat_aligned":
            raise ValueError("v13 role-shared HAPPO requires the frozen v12 environment contract")
        self.reward_contract_version = "v12_soft_boundary_combat_aligned"
        self.reward_contract = resolved_reward_contract_v12(self.env_contract_config)
        self.experiment_variant = str(self.config["experiment"]["variant"])
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "fixed_rule_blue_heterogeneous_4v3_role_shared_happo":
            raise ValueError("invalid v13 training_mode")
        if not bool(t.get("share_combat_actor", False)):
            raise ValueError("v13 requires share_combat_actor=true")
        if int(t.get("team_size", -1)) != 4:
            raise ValueError("v13 requires team_size=4")
        self.recurrent = bool(t.get("recurrent_actor", False))
        self.mask_inactive_hidden = bool(t.get("mask_inactive_hidden", True))
        self.recurrent_hidden_dim = int(t.get("recurrent_hidden_dim", 128))
        self.recurrent_num_layers = int(t.get("recurrent_num_layers", 1))
        self.sequence_chunk_length = int(t.get("sequence_chunk_length", 32))
        self.obs_dim = OBS_DIM_V12
        self.gs_dim = GS_DIM_V12
        self.reward_keys = REWARD_COMPONENT_KEYS_V12
        self.device = resolve_device(str(e["device"]))
        torch.manual_seed(int(e["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.episode_seed_rng = np.random.default_rng(int(e["seed"]) + 1009)
        self.num_envs = int(t["num_envs"])
        self.rollout_steps = int(t["rollout_steps"])
        self.total_env_steps = int(t["total_env_steps"])
        self.schedule_env_steps = int(t.get("schedule_env_steps", self.total_env_steps))
        self.envs = make_combat_vector_env_4v3_v12(
            self.env_config, self.num_envs, int(t.get("num_env_workers", 0)), int(e["seed"])
        )
        self.actors = RoleSharedHAPPOActors(
            self.obs_dim,
            3,
            hidden_dim=int(n["hidden_dim"]),
            log_std_init=float(n["log_std_init"]),
            log_std_min=float(n["log_std_min"]),
            log_std_max=float(n["log_std_max"]),
            recurrent=self.recurrent,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            recurrent_num_layers=self.recurrent_num_layers,
        ).to(self.device)
        self.critic = CentralizedValueCritic(self.gs_dim, hidden_dim=int(n["hidden_dim"])).to(self.device)
        self.initial_actor_lr = float(t["actor_lr"])
        self.final_actor_lr = float(t.get("actor_lr_final", self.initial_actor_lr))
        self.initial_critic_lr = float(t["critic_lr"])
        self.final_critic_lr = float(t.get("critic_lr_final", self.initial_critic_lr))
        self.initial_entropy_coef = float(t["entropy_coef"])
        self.final_entropy_coef = float(t.get("entropy_coef_final", self.initial_entropy_coef))
        self.current_actor_lr = self.initial_actor_lr
        self.current_critic_lr = self.initial_critic_lr
        self.current_entropy_coef = self.initial_entropy_coef
        self.support_optimizer = torch.optim.Adam(self.actors.support_actor.parameters(), lr=self.current_actor_lr)
        self.combat_optimizer = torch.optim.Adam(self.actors.combat_actor.parameters(), lr=self.current_actor_lr)
        self.actor_optimizers = {"support": self.support_optimizer, "combat": self.combat_optimizer}
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.current_critic_lr)
        self.buffer = self._new_buffer(self.rollout_steps)
        self.obs, self.global_states, self.alive_masks = self.envs.reset()
        self.hidden = self.actors.initial_hidden(self.num_envs, self.device)
        self.hidden_reset_masks = np.zeros((self.num_envs, 4), np.float32)
        self.current_episode_seeds = [int(e["seed"]) + i for i in range(self.num_envs)]
        self.env_steps = 0
        self.vector_steps = 0
        self.update_count = 0
        self.last_group_order = ["support", "combat"]
        self.last_update_metrics: dict[str, Any] = {}
        self.last_rollout_reward_means: dict[str, float] = {}
        self.recent_episodes: list[dict[str, Any]] = []
        self.seed_manifest: dict[str, Any] = {}
        self.evaluation_history: list[dict[str, Any]] = []
        self.best_score: tuple[float, ...] | None = None
        self.best_score_fields: dict[str, float] = {}
        self.best_evaluation: dict[str, Any] | None = None
        self.best_checkpoint_name: str | None = None
        self.best_scheduled_env_steps: int | None = None
        self.best_actual_env_steps: int | None = None
        self.next_evaluation_env_steps: int | None = None
        self.next_checkpoint_env_steps: int | None = None
        self.effective_rollout_steps = self.rollout_steps

    def _new_buffer(self, steps: int) -> RoleSharedRolloutBuffer4v3:
        return RoleSharedRolloutBuffer4v3(
            steps, self.num_envs, self.obs_dim, self.gs_dim,
            recurrent=self.recurrent, recurrent_hidden_dim=self.recurrent_hidden_dim,
        )

    def training_signature(self) -> dict[str, Any]:
        return {
            "checkpoint_family": CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_ROLE_SHARED_HAPPO_4V3,
            "algorithm_variant": self.experiment_variant,
            "env_config_sha256": sha256_file(self.env_config),
            "reward_contract_sha256": _sha256_json(self.reward_contract),
            "role_policy_mapping": ROLE_POLICY_MAPPING,
            "share_combat_actor": True,
            "recurrent_actor": self.recurrent,
            "recurrent_hidden_dim": self.recurrent_hidden_dim if self.recurrent else 0,
            "recurrent_num_layers": self.recurrent_num_layers if self.recurrent else 0,
            "sequence_chunk_length": self.sequence_chunk_length if self.recurrent else 0,
            "mask_inactive_hidden": self.mask_inactive_hidden,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "obs_dim": self.obs_dim,
            "state_dim": self.gs_dim,
        }

    def _next_episode_seed(self) -> int:
        return int(self.episode_seed_rng.integers(0, 2**31 - 1))

    def reset_hidden_at(self, indices: int | list[int] | np.ndarray) -> None:
        selected = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        self.hidden_reset_masks[selected] = 0.0
        if self.hidden is not None:
            index = torch.as_tensor(selected, dtype=torch.long, device=self.device)
            self.hidden.support[index] = 0.0
            self.hidden.combat[index] = 0.0

    def _hidden_numpy(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.hidden is None:
            return None, None
        return (
            self.hidden.support.detach().cpu().numpy().astype(np.float32),
            self.hidden.combat.detach().cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def _select_actions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, RoleHiddenState | None]:
        obs = torch.as_tensor(self.obs[:, :4], dtype=torch.float32, device=self.device)
        alive = torch.as_tensor(self.alive_masks[:, :4], dtype=torch.float32, device=self.device)
        reset = torch.as_tensor(self.hidden_reset_masks, dtype=torch.float32, device=self.device)
        actions, log_probs, next_hidden = self.actors.sample_actions(obs, alive, self.hidden, reset)
        values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device))
        return (
            actions.cpu().numpy().astype(np.float32),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
            next_hidden,
        )

    def collect_rollout(self, max_env_steps: int | None = None) -> list[dict[str, Any]]:
        steps = self.rollout_steps if max_env_steps is None else min(self.rollout_steps, int(max_env_steps) // self.num_envs)
        if steps <= 0:
            raise ValueError("effective rollout must contain at least one vector step")
        if self.buffer.rollout_steps != steps:
            self.buffer = self._new_buffer(steps)
        self.effective_rollout_steps = steps
        self.buffer.clear()
        episodes: list[dict[str, Any]] = []
        component_sum = np.zeros(len(self.reward_keys), np.float64)
        for _ in range(steps):
            support_hidden, combat_hidden = self._hidden_numpy()
            actions, log_probs, values, next_hidden = self._select_actions()
            current_alive = self.alive_masks[:, :4].copy()
            result = self.envs.step(actions)
            dones = result.terminated | result.truncated
            self.buffer.add(
                self.obs[:, :4], self.global_states, actions, log_probs, current_alive,
                self.hidden_reset_masks, result.team_rewards, values, dones,
                support_hidden_before=support_hidden, combat_hidden_before=combat_hidden,
            )
            component_sum += result.red_reward_components.sum(0)
            self.obs, self.global_states, self.alive_masks = result.observations, result.global_states, result.alive_masks
            continuation = current_alive * self.alive_masks[:, :4] * (~dones).astype(np.float32)[:, None]
            self.hidden = next_hidden
            if self.hidden is not None and self.mask_inactive_hidden:
                mask_t = torch.as_tensor(continuation, dtype=torch.float32, device=self.device)
                self.hidden.support.mul_(mask_t[:, 0:1])
                self.hidden.combat.mul_(mask_t[:, 1:4].unsqueeze(-1))
            self.hidden_reset_masks = continuation.astype(np.float32)
            for env_index, summary in enumerate(result.episode_summaries):
                if summary is None:
                    continue
                item = deepcopy(summary)
                item["episode_seed"] = int(self.current_episode_seeds[env_index])
                episodes.append(item)
                seed = self._next_episode_seed()
                self.obs[env_index], self.global_states[env_index], self.alive_masks[env_index] = self.envs.reset_at(env_index, seed)
                self.current_episode_seeds[env_index] = seed
                self.reset_hidden_at(env_index)
            self.vector_steps += 1
            self.env_steps += self.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, self.config["training"]["gamma"], self.config["training"]["gae_lambda"])
        self.recent_episodes = (self.recent_episodes + episodes)[-200:]
        denom = float(steps * self.num_envs)
        self.last_rollout_reward_means = {
            f"mean_rollout_{key}": float(value / denom) for key, value in zip(self.reward_keys, component_sum)
        }
        return episodes

    def _schedule(self) -> None:
        progress = min(1.0, self.env_steps / max(1, self.schedule_env_steps))
        self.current_actor_lr = linear_schedule(self.initial_actor_lr, self.final_actor_lr, progress)
        self.current_critic_lr = linear_schedule(self.initial_critic_lr, self.final_critic_lr, progress)
        self.current_entropy_coef = linear_schedule(self.initial_entropy_coef, self.final_entropy_coef, progress)
        for optimizer in self.actor_optimizers.values():
            optimizer.param_groups[0]["lr"] = self.current_actor_lr
        self.critic_optimizer.param_groups[0]["lr"] = self.current_critic_lr

    def _mlp_group_log_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        support_lp, support_entropy = self.actors.support_actor.evaluate_actions(obs[:, 0], actions[:, 0])
        combat_lp, combat_entropy = self.actors.combat_actor.evaluate_actions(
            obs[:, 1:4].reshape(-1, self.obs_dim), actions[:, 1:4].reshape(-1, 3)
        )
        return support_lp, support_entropy, combat_lp.reshape(-1, 3), combat_entropy.reshape(-1, 3)

    def _update_mlp_actors(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_lp: torch.Tensor,
        masks: torch.Tensor,
        advantages: torch.Tensor,
        factor: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        t = self.config["training"]
        total = obs.shape[0]
        rows: list[dict[str, Any]] = []
        groups = ["support", "combat"] if int(self.rng.integers(0, 2)) == 0 else ["combat", "support"]
        self.last_group_order = groups
        for group in groups:
            active = masks[:, 0] if group == "support" else (masks[:, 1:4].sum(-1) > 0).float()
            normalized = _normalize_group_advantage(advantages, active)
            if int(active.sum().item()) == 0:
                rows.append({"group": group, "active_samples": 0, "optimizer_steps": 0})
                continue
            optimizer = self.actor_optimizers[group]
            optimizer_steps = 0
            for _ in range(int(t["ppo_epochs"])):
                epoch_order = self.rng.permutation(total)
                for start in range(0, total, int(t["minibatch_size"])):
                    order = epoch_order[start:start + int(t["minibatch_size"])]
                    idx = torch.as_tensor(order, dtype=torch.long, device=self.device)
                    idx = idx[active[idx] > 0.5]
                    if idx.numel() == 0:
                        continue
                    support_lp, support_ent, combat_lp, combat_ent = self._mlp_group_log_probs(obs[idx], actions[idx])
                    if group == "support":
                        new_group_lp = support_lp
                        old_group_lp = old_lp[idx, 0]
                        entropy = support_ent
                    else:
                        combat_alive = masks[idx, 1:4]
                        new_group_lp = combat_joint_log_probability(combat_lp, combat_alive)
                        old_group_lp = combat_joint_log_probability(old_lp[idx, 1:4], combat_alive)
                        entropy = combat_alive_mean_entropy(combat_ent, combat_alive)
                    log_ratio = new_group_lp - old_group_lp
                    ratio = log_ratio.exp()
                    policy_loss = ppo_clipped_policy_loss(ratio, (factor[idx] * normalized[idx]).detach(), float(t["clip_coef"]))
                    loss = policy_loss - self.current_entropy_coef * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    params = self.actors.support_actor.parameters() if group == "support" else self.actors.combat_actor.parameters()
                    grad = nn.utils.clip_grad_norm_(params, float(t["max_grad_norm"]))
                    optimizer.step()
                    self.actors.clamp_log_std_()
                    optimizer_steps += 1
                    row = {
                        "group": group, "policy_loss": float(policy_loss.detach()), "entropy": float(entropy.mean().detach()),
                        "approx_kl": float(((ratio - 1.0) - log_ratio).mean().detach()),
                        "clip_fraction": float(((ratio - 1.0).abs() > float(t["clip_coef"])).float().mean().detach()),
                        "ratio_mean": float(ratio.mean().detach()), "grad_norm": float(grad), "active_samples": int(idx.numel()),
                    }
                    if group == "combat":
                        for slot in range(3):
                            slot_active = masks[idx, slot + 1] > 0.5
                            if slot_active.any():
                                slot_log_ratio = combat_lp[:, slot][slot_active] - old_lp[idx, slot + 1][slot_active]
                                slot_ratio = slot_log_ratio.exp()
                                row[f"slot_{slot + 1}_kl"] = float(((slot_ratio - 1.0) - slot_log_ratio).mean().detach())
                                row[f"slot_{slot + 1}_clip"] = float(((slot_ratio - 1.0).abs() > float(t["clip_coef"])).float().mean().detach())
                    rows.append(row)
            with torch.no_grad():
                support_lp, _, combat_lp, _ = self._mlp_group_log_probs(obs, actions)
                if group == "support":
                    factor = role_group_factor_update(factor, old_lp[:, 0], support_lp, active)
                else:
                    old_joint = combat_joint_log_probability(old_lp[:, 1:4], masks[:, 1:4])
                    new_joint = combat_joint_log_probability(combat_lp, masks[:, 1:4])
                    factor = role_group_factor_update(factor, old_joint, new_joint, active)
            rows.append({"group": group, "active_samples": int(active.sum().item()), "optimizer_steps": optimizer_steps, "summary": True})
        return rows, factor

    def _evaluate_recurrent_batch(self, batch: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        reset = torch.as_tensor(batch["reset_masks"], dtype=torch.float32, device=self.device)
        support_h = torch.as_tensor(batch["support_initial_hidden"], dtype=torch.float32, device=self.device)
        combat_h = torch.as_tensor(batch["combat_initial_hidden"], dtype=torch.float32, device=self.device)
        support_lp, support_ent, _ = self.actors.support_actor.evaluate_sequence(obs[:, :, 0], actions[:, :, 0], support_h, reset[:, :, 0])
        b, length = obs.shape[:2]
        combat_obs = obs[:, :, 1:4].permute(0, 2, 1, 3).reshape(b * 3, length, self.obs_dim)
        combat_actions = actions[:, :, 1:4].permute(0, 2, 1, 3).reshape(b * 3, length, 3)
        combat_reset = reset[:, :, 1:4].permute(0, 2, 1).reshape(b * 3, length)
        combat_lp, combat_ent, _ = self.actors.combat_actor.evaluate_sequence(
            combat_obs, combat_actions, combat_h.reshape(b * 3, self.recurrent_hidden_dim), combat_reset
        )
        combat_lp = combat_lp.reshape(b, 3, length).permute(0, 2, 1)
        combat_ent = combat_ent.reshape(b, 3, length).permute(0, 2, 1)
        return support_lp, support_ent, combat_lp, combat_ent

    @torch.no_grad()
    def _recurrent_log_probs_all(self, chunks: list[SequenceChunk]) -> torch.Tensor:
        total = self.buffer.rollout_steps * self.num_envs
        output = torch.zeros((total, 4), dtype=torch.float32, device=self.device)
        for start in range(0, len(chunks), 64):
            batch = self.buffer.padded_chunk_batch(chunks[start:start + 64], self.sequence_chunk_length)
            support_lp, _, combat_lp, _ = self._evaluate_recurrent_batch(batch)
            indices = torch.as_tensor(batch["factor_indices"], dtype=torch.long, device=self.device)
            valid = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=self.device)
            output[indices[valid], 0] = support_lp[valid]
            output[indices[valid], 1:4] = combat_lp[valid]
        return output

    def _update_recurrent_actors(
        self, old_lp_flat: torch.Tensor, masks_flat: torch.Tensor, advantages_flat: torch.Tensor, factor: torch.Tensor
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        t = self.config["training"]
        chunks = self.buffer.sequence_chunks(self.sequence_chunk_length)
        chunks_per_batch = max(1, int(t["minibatch_size"]) // self.sequence_chunk_length)
        groups = ["support", "combat"] if int(self.rng.integers(0, 2)) == 0 else ["combat", "support"]
        self.last_group_order = groups
        rows: list[dict[str, Any]] = []
        group_active_flat = {
            "support": masks_flat[:, 0],
            "combat": (masks_flat[:, 1:4].sum(-1) > 0).float(),
        }
        normalized = {group: _normalize_group_advantage(advantages_flat, active) for group, active in group_active_flat.items()}
        for group in groups:
            if int(group_active_flat[group].sum().item()) == 0:
                rows.append({"group": group, "active_samples": 0, "optimizer_steps": 0})
                continue
            optimizer = self.actor_optimizers[group]
            optimizer_steps = 0
            for _ in range(int(t["ppo_epochs"])):
                order = self.rng.permutation(len(chunks))
                for start in range(0, len(chunks), chunks_per_batch):
                    selected = [chunks[int(i)] for i in order[start:start + chunks_per_batch]]
                    batch = self.buffer.padded_chunk_batch(selected, self.sequence_chunk_length)
                    support_lp, support_ent, combat_lp, combat_ent = self._evaluate_recurrent_batch(batch)
                    old = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=self.device)
                    alive = torch.as_tensor(batch["alive_masks"], dtype=torch.float32, device=self.device)
                    valid = torch.as_tensor(batch["valid_mask"], dtype=torch.float32, device=self.device)
                    indices = torch.as_tensor(batch["factor_indices"], dtype=torch.long, device=self.device)
                    safe_indices = indices.clamp_min(0)
                    batch_factor = factor[safe_indices]
                    batch_adv = normalized[group][safe_indices]
                    if group == "support":
                        active = valid * alive[:, :, 0]
                        new_group_lp, old_group_lp, entropy = support_lp, old[:, :, 0], support_ent
                    else:
                        active = valid * (alive[:, :, 1:4].sum(-1) > 0).float()
                        new_group_lp = combat_joint_log_probability(combat_lp, alive[:, :, 1:4])
                        old_group_lp = combat_joint_log_probability(old[:, :, 1:4], alive[:, :, 1:4])
                        entropy = combat_alive_mean_entropy(combat_ent, alive[:, :, 1:4])
                    selected_mask = active > 0.5
                    if not selected_mask.any():
                        continue
                    log_ratio = new_group_lp[selected_mask] - old_group_lp[selected_mask]
                    ratio = log_ratio.exp()
                    effective_adv = (batch_factor[selected_mask] * batch_adv[selected_mask]).detach()
                    policy_loss = ppo_clipped_policy_loss(ratio, effective_adv, float(t["clip_coef"]))
                    loss = policy_loss - self.current_entropy_coef * entropy[selected_mask].mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    params = self.actors.support_actor.parameters() if group == "support" else self.actors.combat_actor.parameters()
                    grad = nn.utils.clip_grad_norm_(params, float(t["max_grad_norm"]))
                    optimizer.step()
                    self.actors.clamp_log_std_()
                    optimizer_steps += 1
                    rows.append({
                        "group": group, "policy_loss": float(policy_loss.detach()),
                        "entropy": float(entropy[selected_mask].mean().detach()),
                        "approx_kl": float(((ratio - 1.0) - log_ratio).mean().detach()),
                        "clip_fraction": float(((ratio - 1.0).abs() > float(t["clip_coef"])).float().mean().detach()),
                        "ratio_mean": float(ratio.mean().detach()), "grad_norm": float(grad),
                        "active_samples": int(selected_mask.sum().item()),
                    })
                    if group == "combat":
                        row = rows[-1]
                        for slot in range(3):
                            slot_active = (valid * alive[:, :, slot + 1]) > 0.5
                            if slot_active.any():
                                slot_log_ratio = combat_lp[:, :, slot][slot_active] - old[:, :, slot + 1][slot_active]
                                slot_ratio = slot_log_ratio.exp()
                                row[f"slot_{slot + 1}_kl"] = float(((slot_ratio - 1.0) - slot_log_ratio).mean().detach())
                                row[f"slot_{slot + 1}_clip"] = float(((slot_ratio - 1.0).abs() > float(t["clip_coef"])).float().mean().detach())
            new_lp_flat = self._recurrent_log_probs_all(chunks)
            if group == "support":
                factor = role_group_factor_update(factor, old_lp_flat[:, 0], new_lp_flat[:, 0], group_active_flat[group])
            else:
                factor = role_group_factor_update(
                    factor,
                    combat_joint_log_probability(old_lp_flat[:, 1:4], masks_flat[:, 1:4]),
                    combat_joint_log_probability(new_lp_flat[:, 1:4], masks_flat[:, 1:4]),
                    group_active_flat[group],
                )
            rows.append({"group": group, "active_samples": int(group_active_flat[group].sum().item()), "optimizer_steps": optimizer_steps, "summary": True})
        return rows, factor

    def update(self) -> dict[str, Any]:
        self._schedule()
        t = self.config["training"]
        steps = self.buffer.rollout_steps
        total = steps * self.num_envs
        obs = torch.as_tensor(self.buffer.observations.reshape(total, 4, self.obs_dim), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(total, self.gs_dim), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(total, 4, 3), dtype=torch.float32, device=self.device)
        old_lp = torch.as_tensor(self.buffer.log_probs.reshape(total, 4), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(self.buffer.agent_alive_masks.reshape(total, 4), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(total), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(total), dtype=torch.float32, device=self.device)
        factor = torch.ones(total, dtype=torch.float32, device=self.device)
        if self.recurrent:
            actor_rows, factor = self._update_recurrent_actors(old_lp, masks, advantages, factor)
        else:
            actor_rows, factor = self._update_mlp_actors(obs, actions, old_lp, masks, advantages, factor)

        before = self.critic(states).detach().cpu().numpy()
        critic_losses: list[float] = []
        for _ in range(int(t["ppo_epochs"])):
            order = self.rng.permutation(total)
            for start in range(0, total, int(t["minibatch_size"])):
                idx = torch.as_tensor(order[start:start + int(t["minibatch_size"])], dtype=torch.long, device=self.device)
                values = self.critic(states[idx])
                loss = 0.5 * (returns[idx] - values).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(t["value_loss_coef"]) * loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(t["max_grad_norm"]))
                self.critic_optimizer.step()
                critic_losses.append(float(loss.detach()))
        self.update_count += 1
        usable = [row for row in actor_rows if "policy_loss" in row]
        group_rows = {g: [r for r in usable if r["group"] == g] for g in ("support", "combat")}
        summaries = {g: next((r for r in reversed(actor_rows) if r.get("group") == g and r.get("summary")), {"optimizer_steps": 0, "active_samples": 0}) for g in ("support", "combat")}
        def mean(group: str, key: str) -> float:
            rows = group_rows[group]
            return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0
        metrics: dict[str, Any] = {
            "policy_loss": float(np.mean([float(r["policy_loss"]) for r in usable])) if usable else 0.0,
            "actor_loss": float(np.mean([float(r["policy_loss"]) for r in usable])) if usable else 0.0,
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean([float(r["entropy"]) for r in usable])) if usable else 0.0,
            "approx_kl": float(np.mean([float(r["approx_kl"]) for r in usable])) if usable else 0.0,
            "clip_fraction": float(np.mean([float(r["clip_fraction"]) for r in usable])) if usable else 0.0,
            "actor_grad_norm": float(np.mean([float(r["grad_norm"]) for r in usable])) if usable else 0.0,
            "support_policy_loss": mean("support", "policy_loss"), "combat_policy_loss": mean("combat", "policy_loss"),
            "support_kl": mean("support", "approx_kl"), "combat_joint_kl": mean("combat", "approx_kl"),
            "support_clip_fraction": mean("support", "clip_fraction"), "combat_joint_clip_fraction": mean("combat", "clip_fraction"),
            "support_entropy": mean("support", "entropy"), "combat_entropy": mean("combat", "entropy"),
            "support_grad_norm": mean("support", "grad_norm"), "combat_grad_norm": mean("combat", "grad_norm"),
            "support_optimizer_steps": int(summaries["support"].get("optimizer_steps", 0)),
            "combat_optimizer_steps": int(summaries["combat"].get("optimizer_steps", 0)),
            "support_active_samples": int(masks[:, 0].sum().item()),
            "combat_active_time_env_samples": int((masks[:, 1:4].sum(-1) > 0).sum().item()),
            "combat_active_slot_count": int(masks[:, 1:4].sum().item()),
            "group_update_order": ">".join(self.last_group_order),
            "factor_mean": float(factor.mean().detach()), "factor_min": float(factor.min().detach()), "factor_max": float(factor.max().detach()),
            "advantage_mean": float(self.buffer.advantages.mean()), "advantage_std": float(self.buffer.advantages.std()),
            "explained_variance": float(explained_variance(before, self.buffer.returns.reshape(total))),
            "env_steps": float(self.env_steps), "vector_steps": float(self.vector_steps), "update_count": float(self.update_count),
            "effective_rollout_steps": float(steps), "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr, "current_entropy_coef": self.current_entropy_coef,
            "support_std": float(np.mean(self.actors.support_actor.effective_std_by_dim)),
            "combat_shared_std": float(np.mean(self.actors.combat_actor.effective_std_by_dim)),
            "recurrent_hidden_activity": float(max(
                0.0,
                float(np.max(np.abs(self.buffer.support_hidden_before))) if self.buffer.support_hidden_before is not None else 0.0,
                float(np.max(np.abs(self.buffer.combat_hidden_before))) if self.buffer.combat_hidden_before is not None else 0.0,
            )),
            "hidden_reset_zero_count": int(np.count_nonzero(self.buffer.hidden_reset_masks <= 0.0)),
        }
        for slot in (1, 2, 3):
            slot_rows = [r for r in group_rows["combat"] if f"slot_{slot}_kl" in r]
            metrics[f"combat_slot_{slot}_kl"] = float(np.mean([r[f"slot_{slot}_kl"] for r in slot_rows])) if slot_rows else 0.0
            metrics[f"combat_slot_{slot}_clip_fraction"] = float(np.mean([r[f"slot_{slot}_clip"] for r in slot_rows])) if slot_rows else 0.0
        numeric = [float(v) for v in metrics.values() if isinstance(v, (int, float, np.number))]
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"non-finite role-shared update metrics: {metrics}")
        metrics.update(self.last_rollout_reward_means)
        self.last_update_metrics = metrics
        return metrics

    def save_checkpoint(self, path: str | Path, *, is_best: bool = False, scheduled_env_steps: int | None = None) -> None:
        support_hidden, combat_hidden = self._hidden_numpy()
        payload = {
            "checkpoint_family": CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_ROLE_SHARED_HAPPO_4V3,
            "algorithm_variant": self.experiment_variant,
            "role_policy_mapping": ROLE_POLICY_MAPPING,
            "recurrent_actor": self.recurrent,
            "recurrent_hidden_dim": self.recurrent_hidden_dim,
            "recurrent_num_layers": self.recurrent_num_layers,
            "training_signature": self.training_signature(),
            "config": deepcopy(self.config), "env_config": self.env_config,
            "reward_contract": deepcopy(self.reward_contract),
            "actors": self.actors.state_dict(), "critic": self.critic.state_dict(),
            "support_optimizer": self.support_optimizer.state_dict(),
            "combat_optimizer": self.combat_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps, "vector_steps": self.vector_steps, "update_count": self.update_count,
            "scheduled_env_steps": scheduled_env_steps, "schedule_env_steps": self.schedule_env_steps,
            "last_group_order": list(self.last_group_order), "last_update_metrics": deepcopy(self.last_update_metrics),
            "last_rollout_reward_means": deepcopy(self.last_rollout_reward_means),
            "recent_episodes": deepcopy(self.recent_episodes), "evaluation_history": deepcopy(self.evaluation_history),
            "best_score": self.best_score, "best_score_fields": self.best_score_fields,
            "best_evaluation": self.best_evaluation, "best_checkpoint_name": self.best_checkpoint_name,
            "best_scheduled_env_steps": self.best_scheduled_env_steps, "best_actual_env_steps": self.best_actual_env_steps,
            "next_evaluation_env_steps": self.next_evaluation_env_steps, "next_checkpoint_env_steps": self.next_checkpoint_env_steps,
            "seed_manifest": deepcopy(self.seed_manifest), "current_episode_seeds": list(self.current_episode_seeds),
            "observations": self.obs, "global_states": self.global_states, "alive_masks": self.alive_masks,
            "hidden_reset_masks": self.hidden_reset_masks, "support_online_hidden": support_hidden,
            "combat_online_hidden": combat_hidden, "vector_env_state": self.envs.state_dict(),
            "numpy_rng_state": self.rng.bit_generator.state,
            "episode_seed_rng_state": self.episode_seed_rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "current_actor_lr": self.current_actor_lr, "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef, "is_best": bool(is_best),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as fh:
            torch.save(payload, fh)
            fh.flush(); os.fsync(fh.fileno())
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3:
            raise ValueError("checkpoint family mismatch: v12/other HAPPO checkpoints cannot load into v13")
        diffs = signature_mismatches(ckpt.get("training_signature", {}), self.training_signature())
        if diffs:
            raise ValueError("training signature mismatch:\n" + "\n".join(diffs))
        self.actors.load_state_dict(ckpt["actors"]); self.critic.load_state_dict(ckpt["critic"])
        self.support_optimizer.load_state_dict(ckpt["support_optimizer"])
        self.combat_optimizer.load_state_dict(ckpt["combat_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"]); self.vector_steps = int(ckpt["vector_steps"]); self.update_count = int(ckpt["update_count"])
        self.last_group_order = list(ckpt.get("last_group_order", self.last_group_order))
        self.last_update_metrics = dict(ckpt.get("last_update_metrics", {})); self.last_rollout_reward_means = dict(ckpt.get("last_rollout_reward_means", {}))
        self.recent_episodes = list(ckpt.get("recent_episodes", [])); self.evaluation_history = list(ckpt.get("evaluation_history", []))
        loaded_best = ckpt.get("best_score"); self.best_score = tuple(loaded_best) if loaded_best is not None else None
        self.best_score_fields = dict(ckpt.get("best_score_fields", {})); self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name"); self.best_scheduled_env_steps = ckpt.get("best_scheduled_env_steps")
        self.best_actual_env_steps = ckpt.get("best_actual_env_steps"); self.next_evaluation_env_steps = ckpt.get("next_evaluation_env_steps")
        self.next_checkpoint_env_steps = ckpt.get("next_checkpoint_env_steps"); self.seed_manifest = deepcopy(ckpt.get("seed_manifest", {}))
        self.current_episode_seeds = [int(v) for v in ckpt["current_episode_seeds"]]
        self.envs.load_state_dict(ckpt["vector_env_state"])
        self.obs = np.asarray(ckpt["observations"], np.float32); self.global_states = np.asarray(ckpt["global_states"], np.float32)
        self.alive_masks = np.asarray(ckpt["alive_masks"], np.float32); self.hidden_reset_masks = np.asarray(ckpt["hidden_reset_masks"], np.float32)
        if self.recurrent:
            self.hidden = RoleHiddenState(
                torch.as_tensor(ckpt["support_online_hidden"], dtype=torch.float32, device=self.device),
                torch.as_tensor(ckpt["combat_online_hidden"], dtype=torch.float32, device=self.device),
            )
        else:
            self.hidden = None
        self.current_actor_lr = float(ckpt.get("current_actor_lr", self.current_actor_lr))
        self.current_critic_lr = float(ckpt.get("current_critic_lr", self.current_critic_lr))
        self.current_entropy_coef = float(ckpt.get("current_entropy_coef", self.current_entropy_coef))
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        self.episode_seed_rng.bit_generator.state = ckpt["episode_seed_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            states = [state.detach().cpu().to(torch.uint8) for state in ckpt["torch_cuda_rng_state"]]
            torch.cuda.set_rng_state_all(states)

    def write_summary(self, output_dir: str | Path) -> None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm_variant": self.experiment_variant, "role_policy_mapping": ROLE_POLICY_MAPPING,
            "recurrent_actor": self.recurrent, "env_steps": self.env_steps, "vector_steps": self.vector_steps,
            "update_count": self.update_count, "device": str(self.device), "last_update_metrics": self.last_update_metrics,
            "best_score": self.best_score, "best_score_fields": self.best_score_fields,
            "best_checkpoint_name": self.best_checkpoint_name, "best_evaluation": self.best_evaluation,
            "final_evaluation": self.evaluation_history[-1]["summary"] if self.evaluation_history else None,
            "evaluation_history": self.evaluation_history, "seed_manifest": self.seed_manifest,
            "reward_contract_version": self.reward_contract_version, "reward_contract": self.reward_contract,
        }
        (out / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self.envs.close()


__all__ = [
    "CHECKPOINT_FAMILY_ROLE_SHARED_HAPPO_4V3", "ROLE_POLICY_MAPPING", "RoleSharedHAPPO4v3Trainer",
    "combat_alive_mean_entropy", "combat_joint_log_probability", "role_group_factor_update",
]
