"""Mission-aligned v14A/v14B role-shared HAPPO trainers."""
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
from ..environment_4v3_v14 import (
    AGENT_REWARD_COMPONENT_KEYS_V14,
    GS_DIM_V14,
    OBS_DIM_V14,
    REWARD_COMPONENT_KEYS_V14,
)
from ..mappo.trainer_3v3 import linear_schedule, resolve_device
from ..mappo.vector_env_4v3_v14 import make_combat_vector_env_4v3_v14
from ..scenario_4v3_v14 import (
    REWARD_CONTRACT_VERSION_V14,
    resolved_reward_contract_v14,
)
from ..scenario_4v3_v15 import (
    AGENT_REWARD_COMPONENT_KEYS_V15,
    REWARD_COMPONENT_KEYS_V15,
    REWARD_CONTRACT_VERSION_V15,
    resolved_reward_contract_v15,
)
from ..scenario_4v3_v16 import (
    AGENT_REWARD_COMPONENT_KEYS_V16,
    REWARD_COMPONENT_KEYS_V16,
    REWARD_CONTRACT_VERSION_V16,
    resolved_reward_contract_v16,
)
from .metrics import explained_variance
from .networks import CentralizedValueCritic
from .role_credit_buffer import (
    AgentCreditRolloutBuffer4v3,
    normalize_role_advantages,
)
from .role_credit_networks import RoleSharedCentralizedCritics4v3
from .role_shared_buffer import RoleSharedRolloutBuffer4v3
from .role_shared_networks import RoleSharedHAPPOActors
from .trainer_3v3 import ppo_clipped_policy_loss, sha256_file, signature_mismatches
from .trainer_role_shared_4v3 import (
    ROLE_POLICY_MAPPING,
    RoleSharedHAPPO4v3Trainer,
    combat_joint_log_probability,
    role_group_factor_update,
)

CHECKPOINT_FAMILY_V14_HAPPO_4V3 = (
    "functional_heterogeneous_4v3_mission_aligned_role_shared_happo"
)
CHECKPOINT_VERSION_V14_HAPPO_4V3 = 1
CREDIT_MODE_TEAM = "team"
CREDIT_MODE_ROLE_LOCAL = "role_local"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def combat_local_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    alive_masks: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    """Per-slot surrogate, averaged once over all alive shared-Combat samples."""
    if not (
        new_log_probs.shape
        == old_log_probs.shape
        == advantages.shape
        == alive_masks.shape
    ) or new_log_probs.shape[-1] != 3:
        raise ValueError("Combat local surrogate inputs must have shape [...,3]")
    active = alive_masks > 0.5
    if not active.any():
        return new_log_probs.sum() * 0.0
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.minimum(unclipped, clipped)[active].mean()


class MissionAlignedRoleSharedHAPPO4v3Trainer(RoleSharedHAPPO4v3Trainer):
    """Unified v14A team-credit and v14B role/local-credit trainer."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.env_contract_config = load_config(self.env_config)
        self.config = deepcopy(config)
        contract_version = self.env_contract_config.get("combat", {}).get(
            "reward_contract_version"
        )
        if contract_version == REWARD_CONTRACT_VERSION_V14:
            self.reward_contract = resolved_reward_contract_v14(
                self.env_contract_config
            )
            self.reward_keys = REWARD_COMPONENT_KEYS_V14
            self.agent_reward_keys = AGENT_REWARD_COMPONENT_KEYS_V14
        elif contract_version == REWARD_CONTRACT_VERSION_V15:
            self.reward_contract = resolved_reward_contract_v15(
                self.env_contract_config
            )
            self.reward_keys = REWARD_COMPONENT_KEYS_V15
            self.agent_reward_keys = AGENT_REWARD_COMPONENT_KEYS_V15
        elif contract_version == REWARD_CONTRACT_VERSION_V16:
            self.reward_contract = resolved_reward_contract_v16(
                self.env_contract_config
            )
            self.reward_keys = REWARD_COMPONENT_KEYS_V16
            self.agent_reward_keys = AGENT_REWARD_COMPONENT_KEYS_V16
        else:
            raise ValueError("role-credit HAPPO requires a v14, v15, or v16 environment")
        self.reward_contract_version = str(contract_version)
        self.experiment_variant = str(self.config["experiment"]["variant"])
        t = self.config["training"]
        n = self.config["network"]
        e = self.config["experiment"]
        expected_mode = (
            "fixed_rule_blue_heterogeneous_4v3_v16_happo"
            if contract_version == REWARD_CONTRACT_VERSION_V16
            else
            "fixed_rule_blue_heterogeneous_4v3_v15_happo"
            if contract_version == REWARD_CONTRACT_VERSION_V15
            else "fixed_rule_blue_heterogeneous_4v3_v14_happo"
        )
        if t.get("training_mode") != expected_mode:
            raise ValueError(f"invalid {self.reward_contract_version} training_mode")
        if not bool(t.get("share_combat_actor", False)):
            raise ValueError("v14 requires share_combat_actor=true")
        if bool(t.get("recurrent_actor", False)):
            raise ValueError("v14A/v14B require recurrent_actor=false")
        if int(t.get("team_size", -1)) != 4:
            raise ValueError("v14 requires team_size=4")
        self.credit_mode = str(t.get("credit_mode", ""))
        if self.credit_mode not in {CREDIT_MODE_TEAM, CREDIT_MODE_ROLE_LOCAL}:
            raise ValueError("v14 credit_mode must be team or role_local")
        self.recurrent = False
        self.mask_inactive_hidden = True
        self.recurrent_hidden_dim = 0
        self.recurrent_num_layers = 0
        self.sequence_chunk_length = 0
        self.obs_dim = OBS_DIM_V14
        self.gs_dim = GS_DIM_V14
        if contract_version in {
            REWARD_CONTRACT_VERSION_V15,
            REWARD_CONTRACT_VERSION_V16,
        }:
            if str(t.get("credit_mode")) != CREDIT_MODE_ROLE_LOCAL:
                raise ValueError("v15/v16 requires credit_mode=role_local")
            if str(t.get("team_reward_usage")) != "reporting_only":
                raise ValueError("v15/v16 requires team_reward_usage=reporting_only")
        self.device = resolve_device(str(e["device"]))
        torch.manual_seed(int(e["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.episode_seed_rng = np.random.default_rng(int(e["seed"]) + 1009)
        self.num_envs = int(t["num_envs"])
        self.rollout_steps = int(t["rollout_steps"])
        self.total_env_steps = int(t["total_env_steps"])
        self.schedule_env_steps = int(
            t.get("schedule_env_steps", self.total_env_steps)
        )
        self.envs = make_combat_vector_env_4v3_v14(
            self.env_config,
            self.num_envs,
            int(t.get("num_env_workers", 0)),
            int(e["seed"]),
        )
        self.actors = RoleSharedHAPPOActors(
            self.obs_dim,
            3,
            hidden_dim=int(n["hidden_dim"]),
            log_std_init=float(n["log_std_init"]),
            log_std_min=float(n["log_std_min"]),
            log_std_max=float(n["log_std_max"]),
            recurrent=False,
        ).to(self.device)
        if self.credit_mode == CREDIT_MODE_TEAM:
            self.critic: nn.Module = CentralizedValueCritic(
                self.gs_dim, hidden_dim=int(n["hidden_dim"])
            ).to(self.device)
        else:
            self.critic = RoleSharedCentralizedCritics4v3(
                self.gs_dim, self.obs_dim, hidden_dim=int(n["hidden_dim"])
            ).to(self.device)
        self.initial_actor_lr = float(t["actor_lr"])
        self.final_actor_lr = float(t.get("actor_lr_final", self.initial_actor_lr))
        self.initial_critic_lr = float(t["critic_lr"])
        self.final_critic_lr = float(
            t.get("critic_lr_final", self.initial_critic_lr)
        )
        self.initial_entropy_coef = float(t["entropy_coef"])
        self.final_entropy_coef = float(
            t.get("entropy_coef_final", self.initial_entropy_coef)
        )
        self.current_actor_lr = self.initial_actor_lr
        self.current_critic_lr = self.initial_critic_lr
        self.current_entropy_coef = self.initial_entropy_coef
        self.support_optimizer = torch.optim.Adam(
            self.actors.support_actor.parameters(), lr=self.current_actor_lr
        )
        self.combat_optimizer = torch.optim.Adam(
            self.actors.combat_actor.parameters(), lr=self.current_actor_lr
        )
        self.actor_optimizers = {
            "support": self.support_optimizer,
            "combat": self.combat_optimizer,
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.current_critic_lr
        )
        self.buffer = self._new_buffer(self.rollout_steps)
        self.obs, self.global_states, self.alive_masks = self.envs.reset()
        self.hidden = None
        self.hidden_reset_masks = np.zeros((self.num_envs, 4), np.float32)
        self.current_episode_seeds = [
            int(e["seed"]) + i for i in range(self.num_envs)
        ]
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

    @property
    def role_critics(self) -> RoleSharedCentralizedCritics4v3:
        if self.credit_mode != CREDIT_MODE_ROLE_LOCAL:
            raise AttributeError("v14A has a scalar team critic")
        assert isinstance(self.critic, RoleSharedCentralizedCritics4v3)
        return self.critic

    def _new_buffer(self, steps: int):
        if self.credit_mode == CREDIT_MODE_TEAM:
            return RoleSharedRolloutBuffer4v3(
                steps, self.num_envs, self.obs_dim, self.gs_dim, recurrent=False
            )
        return AgentCreditRolloutBuffer4v3(
            steps, self.num_envs, self.obs_dim, self.gs_dim
        )

    def training_signature(self) -> dict[str, Any]:
        signature = {
            "checkpoint_family": CHECKPOINT_FAMILY_V14_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_V14_HAPPO_4V3,
            "algorithm_variant": self.experiment_variant,
            "credit_mode": self.credit_mode,
            "env_config_sha256": sha256_file(self.env_config),
            "reward_contract_sha256": _sha256_json(self.reward_contract),
            "role_policy_mapping": ROLE_POLICY_MAPPING,
            "critic_design": (
                "scalar_team"
                if self.credit_mode == CREDIT_MODE_TEAM
                else "support_plus_shared_combat"
            ),
            "share_combat_actor": True,
            "recurrent_actor": False,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "obs_dim": self.obs_dim,
            "state_dim": self.gs_dim,
        }
        # Preserve the exact historical v14 checkpoint signature. The v15-only
        # fields make the reporting-only scalar contract explicit without
        # invalidating existing v14A/v14B checkpoints.
        if self.reward_contract_version == REWARD_CONTRACT_VERSION_V15:
            signature.update(
                {
                    "reward_contract_version": self.reward_contract_version,
                    "team_reward_usage": "reporting_only",
                }
            )
        elif self.reward_contract_version == REWARD_CONTRACT_VERSION_V16:
            signature.update(
                {
                    "reward_contract_version": self.reward_contract_version,
                    "observation_contract": self.env_contract_config["combat"][
                        "observation_contract"
                    ],
                    "team_reward_usage": "reporting_only",
                }
            )
        return signature

    @torch.no_grad()
    def _select_actions(self):
        obs = torch.as_tensor(
            self.obs[:, :4], dtype=torch.float32, device=self.device
        )
        alive = torch.as_tensor(
            self.alive_masks[:, :4], dtype=torch.float32, device=self.device
        )
        actions, log_probs, _ = self.actors.sample_actions(
            obs, alive, None, None
        )
        states = torch.as_tensor(
            self.global_states, dtype=torch.float32, device=self.device
        )
        values = (
            self.critic(states)
            if self.credit_mode == CREDIT_MODE_TEAM
            else self.role_critics(states, obs)
        )
        return (
            actions.cpu().numpy().astype(np.float32),
            log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
            None,
        )

    def collect_rollout(self, max_env_steps: int | None = None):
        steps = (
            self.rollout_steps
            if max_env_steps is None
            else min(self.rollout_steps, int(max_env_steps) // self.num_envs)
        )
        if steps <= 0:
            raise ValueError("effective rollout must contain at least one vector step")
        if self.buffer.rollout_steps != steps:
            self.buffer = self._new_buffer(steps)
        self.effective_rollout_steps = steps
        self.buffer.clear()
        episodes: list[dict[str, Any]] = []
        component_sum = np.zeros(len(self.reward_keys), np.float64)
        agent_reward_sum = np.zeros(4, np.float64)
        agent_component_sum = np.zeros(
            (4, len(self.agent_reward_keys)), np.float64
        )
        combat_state_index = (
            self.agent_reward_keys.index("combat_state_reward")
            if "combat_state_reward" in self.agent_reward_keys
            else None
        )
        combat_state_min = float("inf")
        combat_state_max = float("-inf")
        combat_state_finite = True
        for _ in range(steps):
            actions, log_probs, values, _ = self._select_actions()
            current_alive = self.alive_masks[:, :4].copy()
            result = self.envs.step(actions)
            dones = result.terminated | result.truncated
            if self.credit_mode == CREDIT_MODE_TEAM:
                self.buffer.add(
                    self.obs[:, :4],
                    self.global_states,
                    actions,
                    log_probs,
                    current_alive,
                    self.hidden_reset_masks,
                    result.team_rewards,
                    values,
                    dones,
                )
            else:
                self.buffer.add(
                    self.obs[:, :4],
                    self.global_states,
                    actions,
                    log_probs,
                    current_alive,
                    result.team_rewards,
                    result.agent_rewards,
                    values,
                    dones,
                )
            component_sum += result.red_reward_components.sum(0)
            agent_reward_sum += result.agent_rewards.sum(0)
            agent_component_sum += result.red_agent_reward_components.sum(0)
            if combat_state_index is not None:
                combat_state = result.red_agent_reward_components[
                    :, 1:4, combat_state_index
                ]
                combat_state_finite = bool(
                    combat_state_finite and np.isfinite(combat_state).all()
                )
                combat_state_min = min(combat_state_min, float(combat_state.min()))
                combat_state_max = max(combat_state_max, float(combat_state.max()))
            self.obs = result.observations
            self.global_states = result.global_states
            self.alive_masks = result.alive_masks
            continuation = (
                current_alive
                * self.alive_masks[:, :4]
                * (~dones).astype(np.float32)[:, None]
            )
            self.hidden_reset_masks = continuation.astype(np.float32)
            for env_index, summary in enumerate(result.episode_summaries):
                if summary is None:
                    continue
                item = deepcopy(summary)
                item["episode_seed"] = int(self.current_episode_seeds[env_index])
                episodes.append(item)
                seed = self._next_episode_seed()
                (
                    self.obs[env_index],
                    self.global_states[env_index],
                    self.alive_masks[env_index],
                ) = self.envs.reset_at(env_index, seed)
                self.current_episode_seeds[env_index] = seed
                self.hidden_reset_masks[env_index] = 0.0
            self.vector_steps += 1
            self.env_steps += self.num_envs
        with torch.no_grad():
            states = torch.as_tensor(
                self.global_states, dtype=torch.float32, device=self.device
            )
            if self.credit_mode == CREDIT_MODE_TEAM:
                last_values = self.critic(states).cpu().numpy()
            else:
                observations = torch.as_tensor(
                    self.obs[:, :4], dtype=torch.float32, device=self.device
                )
                last_values = self.role_critics(states, observations).cpu().numpy()
        self.buffer.compute_returns_and_advantages(
            last_values,
            self.config["training"]["gamma"],
            self.config["training"]["gae_lambda"],
        )
        self.recent_episodes = (self.recent_episodes + episodes)[-200:]
        denom = float(steps * self.num_envs)
        self.last_rollout_reward_means = {
            f"mean_rollout_{key}": float(value / denom)
            for key, value in zip(self.reward_keys, component_sum)
        }
        for slot, value in enumerate(agent_reward_sum):
            self.last_rollout_reward_means[
                f"mean_rollout_red_{slot}_agent_reward"
            ] = float(value / denom)
        for slot in range(4):
            for key, value in zip(
                self.agent_reward_keys, agent_component_sum[slot]
            ):
                self.last_rollout_reward_means[
                    f"mean_rollout_red_{slot}_{key}"
                ] = float(value / denom)
        if "support_state_reward" in self.agent_reward_keys:
            self.last_rollout_reward_means["mean_rollout_support_state_reward"] = (
                self.last_rollout_reward_means[
                    "mean_rollout_red_0_support_state_reward"
                ]
            )
            self.last_rollout_reward_means["mean_rollout_combat_state_reward"] = (
                float(
                    np.mean(
                        [
                            self.last_rollout_reward_means[
                                f"mean_rollout_red_{slot}_combat_state_reward"
                            ]
                            for slot in (1, 2, 3)
                        ]
                    )
                )
            )
        if combat_state_index is not None:
            scale = float(self.reward_contract.get("combat_state", {}).get("scale", 0.02))
            if self.reward_contract_version == REWARD_CONTRACT_VERSION_V16:
                quality_bounds = [
                    combat_state_min / scale,
                    combat_state_max / scale,
                ]
            else:
                quality_bounds = [
                    (combat_state_min / scale + 1.0) / 2.0,
                    (combat_state_max / scale + 1.0) / 2.0,
                ]
            self.last_rollout_reward_means.update(
                {
                    "min_rollout_combat_state_reward": combat_state_min,
                    "max_rollout_combat_state_reward": combat_state_max,
                    "combat_lock_quality_finite": float(
                        combat_state_finite
                        and np.isfinite(quality_bounds).all()
                    ),
                }
            )
        return episodes

    def _update_credit_actors(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_lp: torch.Tensor,
        masks: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        """Local slot surrogate; joint Combat ratio is used only for HAPPO factor."""
        t = self.config["training"]
        total = obs.shape[0]
        support_np, combat_np = normalize_role_advantages(
            advantages.detach().cpu().numpy(), masks.detach().cpu().numpy()
        )
        support_adv = torch.as_tensor(
            support_np, dtype=torch.float32, device=self.device
        )
        combat_adv = torch.as_tensor(
            combat_np, dtype=torch.float32, device=self.device
        )
        factor = torch.ones(total, dtype=torch.float32, device=self.device)
        rows: list[dict[str, Any]] = []
        groups = (
            ["support", "combat"]
            if int(self.rng.integers(0, 2)) == 0
            else ["combat", "support"]
        )
        self.last_group_order = groups
        for group in groups:
            group_active = (
                masks[:, 0] > 0.5
                if group == "support"
                else masks[:, 1:4].sum(-1) > 0.5
            )
            if not group_active.any():
                rows.append(
                    {
                        "group": group,
                        "active_samples": 0,
                        "optimizer_steps": 0,
                        "summary": True,
                    }
                )
                continue
            optimizer = self.actor_optimizers[group]
            optimizer_steps = 0
            for _ in range(int(t["ppo_epochs"])):
                order = self.rng.permutation(total)
                for start in range(0, total, int(t["minibatch_size"])):
                    idx = torch.as_tensor(
                        order[start : start + int(t["minibatch_size"])],
                        dtype=torch.long,
                        device=self.device,
                    )
                    idx = idx[group_active[idx]]
                    if idx.numel() == 0:
                        continue
                    support_lp, support_ent, combat_lp, combat_ent = (
                        self._mlp_group_log_probs(obs[idx], actions[idx])
                    )
                    if group == "support":
                        log_ratio = support_lp - old_lp[idx, 0]
                        ratio = log_ratio.exp()
                        effective_advantage = (
                            factor[idx] * support_adv[idx]
                        ).detach()
                        policy_loss = ppo_clipped_policy_loss(
                            ratio,
                            effective_advantage,
                            float(t["clip_coef"]),
                        )
                        entropy = support_ent.mean()
                        local_kl = ((ratio - 1.0) - log_ratio).mean()
                        local_clip = (
                            (ratio - 1.0).abs() > float(t["clip_coef"])
                        ).float().mean()
                        joint_kl, joint_clip = local_kl, local_clip
                        active_slots = int(idx.numel())
                    else:
                        alive = masks[idx, 1:4]
                        local_advantage = (
                            factor[idx, None] * combat_adv[idx]
                        ).detach()
                        policy_loss = combat_local_clipped_policy_loss(
                            combat_lp,
                            old_lp[idx, 1:4],
                            local_advantage,
                            alive,
                            float(t["clip_coef"]),
                        )
                        slot_active = alive > 0.5
                        entropy = combat_ent[slot_active].mean()
                        slot_log_ratio = combat_lp - old_lp[idx, 1:4]
                        slot_ratio = slot_log_ratio.exp()
                        local_kl = (
                            ((slot_ratio - 1.0) - slot_log_ratio)[slot_active]
                        ).mean()
                        local_clip = (
                            (slot_ratio - 1.0).abs() > float(t["clip_coef"])
                        )[slot_active].float().mean()
                        new_joint = combat_joint_log_probability(combat_lp, alive)
                        old_joint = combat_joint_log_probability(
                            old_lp[idx, 1:4], alive
                        )
                        joint_log_ratio = new_joint - old_joint
                        joint_ratio = joint_log_ratio.exp()
                        joint_kl = (
                            (joint_ratio - 1.0) - joint_log_ratio
                        ).mean()
                        joint_clip = (
                            (joint_ratio - 1.0).abs() > float(t["clip_coef"])
                        ).float().mean()
                        active_slots = int(slot_active.sum().item())
                    loss = policy_loss - self.current_entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    params = (
                        self.actors.support_actor.parameters()
                        if group == "support"
                        else self.actors.combat_actor.parameters()
                    )
                    grad = nn.utils.clip_grad_norm_(
                        params, float(t["max_grad_norm"])
                    )
                    optimizer.step()
                    self.actors.clamp_log_std_()
                    optimizer_steps += 1
                    row = {
                        "group": group,
                        "policy_loss": float(policy_loss.detach()),
                        "entropy": float(entropy.detach()),
                        "approx_kl": float(local_kl.detach()),
                        "clip_fraction": float(local_clip.detach()),
                        "joint_kl": float(joint_kl.detach()),
                        "joint_clip": float(joint_clip.detach()),
                        "grad_norm": float(grad),
                        "active_samples": active_slots,
                    }
                    if group == "combat":
                        for slot in range(3):
                            selected = alive[:, slot] > 0.5
                            if selected.any():
                                lr = slot_log_ratio[:, slot][selected]
                                rr = lr.exp()
                                row[f"slot_{slot + 1}_kl"] = float(
                                    (((rr - 1.0) - lr).mean()).detach()
                                )
                                row[f"slot_{slot + 1}_clip"] = float(
                                    (
                                        (rr - 1.0).abs()
                                        > float(t["clip_coef"])
                                    )
                                    .float()
                                    .mean()
                                    .detach()
                                )
                    rows.append(row)

            # The shared Combat local-credit objective above uses slot ratios.
            # As a preceding HAPPO policy group it still contributes its joint
            # product-policy ratio to the following Support group.
            with torch.no_grad():
                new_support, _, new_combat, _ = self._mlp_group_log_probs(
                    obs, actions
                )
                if group == "support":
                    factor = role_group_factor_update(
                        factor,
                        old_lp[:, 0],
                        new_support,
                        masks[:, 0],
                    )
                else:
                    combat_alive = masks[:, 1:4]
                    factor = role_group_factor_update(
                        factor,
                        combat_joint_log_probability(
                            old_lp[:, 1:4], combat_alive
                        ),
                        combat_joint_log_probability(new_combat, combat_alive),
                        (combat_alive.sum(-1) > 0.5).float(),
                    )
            rows.append(
                {
                    "group": group,
                    "active_samples": int(
                        masks[:, 0].sum().item()
                        if group == "support"
                        else masks[:, 1:4].sum().item()
                    ),
                    "optimizer_steps": optimizer_steps,
                    "summary": True,
                }
            )
        return rows, factor

    def update(self) -> dict[str, Any]:
        if self.credit_mode == CREDIT_MODE_TEAM:
            metrics = super().update()
            metrics["credit_mode"] = self.credit_mode
            metrics["team_advantage_mean"] = metrics["advantage_mean"]
            metrics["team_advantage_std"] = metrics["advantage_std"]
            self.last_update_metrics = metrics
            return metrics

        self._schedule()
        t = self.config["training"]
        steps = self.buffer.rollout_steps
        total = steps * self.num_envs
        obs = torch.as_tensor(
            self.buffer.observations.reshape(total, 4, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(
            self.buffer.global_states.reshape(total, self.gs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            self.buffer.actions.reshape(total, 4, 3),
            dtype=torch.float32,
            device=self.device,
        )
        old_lp = torch.as_tensor(
            self.buffer.log_probs.reshape(total, 4),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            self.buffer.agent_alive_masks.reshape(total, 4),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            self.buffer.advantages.reshape(total, 4),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.as_tensor(
            self.buffer.returns.reshape(total, 4),
            dtype=torch.float32,
            device=self.device,
        )
        actor_rows, factor = self._update_credit_actors(
            obs, actions, old_lp, masks, advantages
        )

        with torch.no_grad():
            before = self.role_critics(states, obs).cpu().numpy()
        support_value_losses: list[float] = []
        combat_value_losses: list[float] = []
        critic_grad_norms: list[float] = []
        for _ in range(int(t["ppo_epochs"])):
            order = self.rng.permutation(total)
            for start in range(0, total, int(t["minibatch_size"])):
                idx = torch.as_tensor(
                    order[start : start + int(t["minibatch_size"])],
                    dtype=torch.long,
                    device=self.device,
                )
                predicted = self.role_critics(states[idx], obs[idx])
                support_loss = 0.5 * (
                    returns[idx, 0] - predicted[:, 0]
                ).square().mean()
                combat_loss = 0.5 * (
                    returns[idx, 1:4] - predicted[:, 1:4]
                ).square().mean()
                value_loss = 0.5 * (support_loss + combat_loss)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(t["value_loss_coef"]) * value_loss).backward()
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(t["max_grad_norm"])
                )
                self.critic_optimizer.step()
                support_value_losses.append(float(support_loss.detach()))
                combat_value_losses.append(float(combat_loss.detach()))
                critic_grad_norms.append(float(critic_grad))

        self.update_count += 1
        usable = [row for row in actor_rows if "policy_loss" in row]
        group_rows = {
            group: [row for row in usable if row["group"] == group]
            for group in ("support", "combat")
        }
        summaries = {
            group: next(
                (
                    row
                    for row in reversed(actor_rows)
                    if row.get("group") == group and row.get("summary")
                ),
                {"optimizer_steps": 0, "active_samples": 0},
            )
            for group in ("support", "combat")
        }

        def mean(group: str, key: str) -> float:
            rows = [row for row in group_rows[group] if key in row]
            return (
                float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
            )

        raw_adv = self.buffer.advantages.reshape(total, 4)
        raw_masks = self.buffer.agent_alive_masks.reshape(total, 4) > 0.5

        def masked_stats(column: int) -> tuple[float, float]:
            selected = raw_adv[:, column][raw_masks[:, column]]
            return (
                (float(selected.mean()), float(selected.std()))
                if selected.size
                else (0.0, 0.0)
            )

        support_adv_mean, support_adv_std = masked_stats(0)
        combat_selected = raw_adv[:, 1:4][raw_masks[:, 1:4]]
        combat_adv_mean = (
            float(combat_selected.mean()) if combat_selected.size else 0.0
        )
        combat_adv_std = (
            float(combat_selected.std()) if combat_selected.size else 0.0
        )
        metrics: dict[str, Any] = {
            "credit_mode": self.credit_mode,
            "policy_loss": float(
                np.mean([float(row["policy_loss"]) for row in usable])
            )
            if usable
            else 0.0,
            "actor_loss": float(
                np.mean([float(row["policy_loss"]) for row in usable])
            )
            if usable
            else 0.0,
            "value_loss": float(
                np.mean(support_value_losses + combat_value_losses)
            ),
            "critic_loss": float(
                np.mean(support_value_losses + combat_value_losses)
            ),
            "support_value_loss": float(np.mean(support_value_losses)),
            "combat_value_loss": float(np.mean(combat_value_losses)),
            "critic_grad_norm": float(np.mean(critic_grad_norms)),
            "entropy": float(np.mean([row["entropy"] for row in usable])),
            "approx_kl": float(np.mean([row["approx_kl"] for row in usable])),
            "clip_fraction": float(
                np.mean([row["clip_fraction"] for row in usable])
            ),
            "actor_grad_norm": float(
                np.mean([row["grad_norm"] for row in usable])
            ),
            "support_policy_loss": mean("support", "policy_loss"),
            "combat_policy_loss": mean("combat", "policy_loss"),
            "support_kl": mean("support", "approx_kl"),
            "combat_local_kl": mean("combat", "approx_kl"),
            "combat_joint_kl": mean("combat", "joint_kl"),
            "support_clip_fraction": mean("support", "clip_fraction"),
            "combat_local_clip_fraction": mean("combat", "clip_fraction"),
            "combat_joint_clip_fraction": mean("combat", "joint_clip"),
            "support_entropy": mean("support", "entropy"),
            "combat_entropy": mean("combat", "entropy"),
            "support_grad_norm": mean("support", "grad_norm"),
            "combat_grad_norm": mean("combat", "grad_norm"),
            "support_optimizer_steps": int(
                summaries["support"]["optimizer_steps"]
            ),
            "combat_optimizer_steps": int(
                summaries["combat"]["optimizer_steps"]
            ),
            "support_active_samples": int(masks[:, 0].sum().item()),
            "combat_active_time_env_samples": int(
                (masks[:, 1:4].sum(-1) > 0).sum().item()
            ),
            "combat_active_slot_count": int(masks[:, 1:4].sum().item()),
            "group_update_order": ">".join(self.last_group_order),
            "factor_mean": float(factor.mean().detach()),
            "factor_min": float(factor.min().detach()),
            "factor_max": float(factor.max().detach()),
            "support_advantage_mean": support_adv_mean,
            "support_advantage_std": support_adv_std,
            "pooled_combat_advantage_mean": combat_adv_mean,
            "pooled_combat_advantage_std": combat_adv_std,
            "env_steps": float(self.env_steps),
            "vector_steps": float(self.vector_steps),
            "update_count": float(self.update_count),
            "effective_rollout_steps": float(steps),
            "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef,
            "support_std": float(
                np.mean(self.actors.support_actor.effective_std_by_dim)
            ),
            "combat_shared_std": float(
                np.mean(self.actors.combat_actor.effective_std_by_dim)
            ),
            "recurrent_hidden_activity": 0.0,
            "hidden_reset_zero_count": int(
                np.count_nonzero(self.hidden_reset_masks <= 0.0)
            ),
            "support_critic_explained_variance": float(
                explained_variance(before[:, 0], self.buffer.returns[..., 0].reshape(total))
            ),
            "pooled_combat_critic_explained_variance": float(
                explained_variance(
                    before[:, 1:4].reshape(-1),
                    self.buffer.returns[..., 1:4].reshape(-1),
                )
            ),
        }
        for slot in range(3):
            adv_mean, adv_std = masked_stats(slot + 1)
            metrics[f"combat_slot_{slot + 1}_raw_advantage_mean"] = adv_mean
            metrics[f"combat_slot_{slot + 1}_raw_advantage_std"] = adv_std
            metrics[f"combat_slot_{slot + 1}_critic_explained_variance"] = float(
                explained_variance(
                    before[:, slot + 1],
                    self.buffer.returns[..., slot + 1].reshape(total),
                )
            )
            metrics[f"combat_slot_{slot + 1}_kl"] = mean(
                "combat", f"slot_{slot + 1}_kl"
            )
            metrics[f"combat_slot_{slot + 1}_clip_fraction"] = mean(
                "combat", f"slot_{slot + 1}_clip"
            )
        numeric = [
            float(value)
            for value in metrics.values()
            if isinstance(value, (int, float, np.number))
        ]
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"non-finite v14B update metrics: {metrics}")
        metrics.update(self.last_rollout_reward_means)
        self.last_update_metrics = metrics
        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        is_best: bool = False,
        scheduled_env_steps: int | None = None,
    ) -> None:
        payload = {
            "checkpoint_family": CHECKPOINT_FAMILY_V14_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_V14_HAPPO_4V3,
            "algorithm_variant": self.experiment_variant,
            "credit_mode": self.credit_mode,
            "role_policy_mapping": ROLE_POLICY_MAPPING,
            "training_signature": self.training_signature(),
            "config": deepcopy(self.config),
            "env_config": self.env_config,
            "reward_contract": deepcopy(self.reward_contract),
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "support_optimizer": self.support_optimizer.state_dict(),
            "combat_optimizer": self.combat_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "scheduled_env_steps": scheduled_env_steps,
            "schedule_env_steps": self.schedule_env_steps,
            "last_group_order": list(self.last_group_order),
            "last_update_metrics": deepcopy(self.last_update_metrics),
            "last_rollout_reward_means": deepcopy(self.last_rollout_reward_means),
            "recent_episodes": deepcopy(self.recent_episodes),
            "evaluation_history": deepcopy(self.evaluation_history),
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "best_evaluation": self.best_evaluation,
            "best_checkpoint_name": self.best_checkpoint_name,
            "best_scheduled_env_steps": self.best_scheduled_env_steps,
            "best_actual_env_steps": self.best_actual_env_steps,
            "next_evaluation_env_steps": self.next_evaluation_env_steps,
            "next_checkpoint_env_steps": self.next_checkpoint_env_steps,
            "seed_manifest": deepcopy(self.seed_manifest),
            "current_episode_seeds": list(self.current_episode_seeds),
            "observations": self.obs,
            "global_states": self.global_states,
            "alive_masks": self.alive_masks,
            "hidden_reset_masks": self.hidden_reset_masks,
            "vector_env_state": self.envs.state_dict(),
            "numpy_rng_state": self.rng.bit_generator.state,
            "episode_seed_rng_state": self.episode_seed_rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "current_actor_lr": self.current_actor_lr,
            "current_critic_lr": self.current_critic_lr,
            "current_entropy_coef": self.current_entropy_coef,
            "is_best": bool(is_best),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_V14_HAPPO_4V3:
            raise ValueError(
                "checkpoint family mismatch: v12/v13 checkpoints cannot load into v14"
            )
        diffs = signature_mismatches(
            ckpt.get("training_signature", {}), self.training_signature()
        )
        if diffs:
            raise ValueError("training signature mismatch:\n" + "\n".join(diffs))
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        self.support_optimizer.load_state_dict(ckpt["support_optimizer"])
        self.combat_optimizer.load_state_dict(ckpt["combat_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"])
        self.vector_steps = int(ckpt["vector_steps"])
        self.update_count = int(ckpt["update_count"])
        self.last_group_order = list(ckpt.get("last_group_order", []))
        self.last_update_metrics = dict(ckpt.get("last_update_metrics", {}))
        self.last_rollout_reward_means = dict(
            ckpt.get("last_rollout_reward_means", {})
        )
        self.recent_episodes = list(ckpt.get("recent_episodes", []))
        self.evaluation_history = list(ckpt.get("evaluation_history", []))
        loaded_best = ckpt.get("best_score")
        self.best_score = tuple(loaded_best) if loaded_best is not None else None
        self.best_score_fields = dict(ckpt.get("best_score_fields", {}))
        self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.best_scheduled_env_steps = ckpt.get("best_scheduled_env_steps")
        self.best_actual_env_steps = ckpt.get("best_actual_env_steps")
        self.next_evaluation_env_steps = ckpt.get("next_evaluation_env_steps")
        self.next_checkpoint_env_steps = ckpt.get("next_checkpoint_env_steps")
        self.seed_manifest = deepcopy(ckpt.get("seed_manifest", {}))
        self.current_episode_seeds = [int(v) for v in ckpt["current_episode_seeds"]]
        self.envs.load_state_dict(ckpt["vector_env_state"])
        self.obs = np.asarray(ckpt["observations"], np.float32)
        self.global_states = np.asarray(ckpt["global_states"], np.float32)
        self.alive_masks = np.asarray(ckpt["alive_masks"], np.float32)
        self.hidden_reset_masks = np.asarray(
            ckpt["hidden_reset_masks"], np.float32
        )
        self.current_actor_lr = float(ckpt["current_actor_lr"])
        self.current_critic_lr = float(ckpt["current_critic_lr"])
        self.current_entropy_coef = float(ckpt["current_entropy_coef"])
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        self.episode_seed_rng.bit_generator.state = ckpt[
            "episode_seed_rng_state"
        ]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in ckpt["torch_cuda_rng_state"]]
            )

    def write_summary(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm_variant": self.experiment_variant,
            "credit_mode": self.credit_mode,
            "role_policy_mapping": ROLE_POLICY_MAPPING,
            "recurrent_actor": False,
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "device": str(self.device),
            "last_update_metrics": self.last_update_metrics,
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "best_checkpoint_name": self.best_checkpoint_name,
            "best_evaluation": self.best_evaluation,
            "final_evaluation": (
                self.evaluation_history[-1]["summary"]
                if self.evaluation_history
                else None
            ),
            "evaluation_history": self.evaluation_history,
            "seed_manifest": self.seed_manifest,
            "reward_contract_version": self.reward_contract_version,
            "reward_contract": self.reward_contract,
            "team_reward_usage": str(
                self.config["training"].get("team_reward_usage", "training")
            ),
        }
        if self.reward_contract_version == REWARD_CONTRACT_VERSION_V16:
            payload["observation_contract"] = self.env_contract_config["combat"][
                "observation_contract"
            ]
        (out / "run_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


__all__ = [
    "CHECKPOINT_FAMILY_V14_HAPPO_4V3",
    "CREDIT_MODE_ROLE_LOCAL",
    "CREDIT_MODE_TEAM",
    "MissionAlignedRoleSharedHAPPO4v3Trainer",
    "combat_local_clipped_policy_loss",
]
