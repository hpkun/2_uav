"""HAPPO trainer for the functional heterogeneous red 4v3 v9 environment."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import load_config
from ..environment_4v3 import GS_DIM_4V3, OBS_DIM_4V3
from ..mappo.trainer_3v3 import linear_schedule, resolve_device
from ..mappo.vector_env_4v3 import RED_TEAM_SIZE_4V3, make_combat_vector_env_4v3
from .buffer_3v3 import HAPPORolloutBuffer3v3
from .metrics import explained_variance
from .networks import CentralizedValueCritic, IndependentHAPPOActors
from .trainer_3v3 import (
    happo_preceding_factor_update,
    normalize_advantages_for_agent,
    ppo_clipped_policy_loss,
    sha256_file,
    signature_mismatches,
)

CHECKPOINT_FAMILY_HAPPO_4V3 = "functional_heterogeneous_4v3_v9_happo"
CHECKPOINT_VERSION_HAPPO_4V3 = 1


def best_score_fields_4v3() -> list[str]:
    return [
        "red_complete_elimination_success_rate",
        "red_at_least_two_attack_kill_rate",
        "red_any_attack_kill_rate",
        "mean_red_attack_kills",
        "support_assisted_kill_rate",
        "mean_red_combat_survivors",
        "negative_timeout_rate",
        "negative_mean_episode_length",
    ]


def compute_best_score_4v3(summary: dict[str, float]) -> tuple[float, dict[str, float]]:
    fields = {
        "red_complete_elimination_success_rate": float(summary.get("red_complete_elimination_success_rate", 0.0)),
        "red_at_least_two_attack_kill_rate": float(summary.get("red_at_least_two_attack_kill_rate", 0.0)),
        "red_any_attack_kill_rate": float(summary.get("red_any_attack_kill_rate", 0.0)),
        "mean_red_attack_kills": float(summary.get("mean_red_attack_kills", 0.0)),
        "support_assisted_kill_rate": float(summary.get("support_assisted_kill_rate", 0.0)),
        "mean_red_combat_survivors": float(summary.get("mean_red_combat_survivors", 0.0)),
        "negative_timeout_rate": -float(summary.get("timeout_rate", 0.0)),
        "negative_mean_episode_length": -float(summary.get("mean_episode_length", 0.0)),
    }
    weights = {
        "red_complete_elimination_success_rate": 100.0,
        "red_at_least_two_attack_kill_rate": 30.0,
        "red_any_attack_kill_rate": 10.0,
        "mean_red_attack_kills": 3.0,
        "support_assisted_kill_rate": 4.0,
        "mean_red_combat_survivors": 1.0,
        "negative_timeout_rate": 2.0,
        "negative_mean_episode_length": 0.002,
    }
    return float(sum(weights[k] * fields[k] for k in weights)), fields


def summarize_4v3_episodes(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"episodes": 0}
    n = len(records)
    return {
        "episodes": float(n),
        "red_complete_elimination_success_rate": float(np.mean([r.get("red_complete_elimination_success", False) for r in records])),
        "red_at_least_two_attack_kill_rate": float(np.mean([int(r.get("red_attack_kills", 0)) >= 2 for r in records])),
        "red_any_attack_kill_rate": float(np.mean([r.get("red_any_attack_kill", False) for r in records])),
        "mean_red_attack_kills": float(np.mean([r.get("red_attack_kills", 0) for r in records])),
        "mean_blue_attack_kills": float(np.mean([r.get("blue_attack_kills", 0) for r in records])),
        "support_assisted_kill_rate": float(np.mean([r.get("support_assisted_kill_rate", 0.0) for r in records])),
        "mean_red_combat_survivors": float(np.mean([r.get("red_combat_survivors", 0) for r in records])),
        "timeout_rate": float(np.mean([r.get("termination_reason") == "timeout" for r in records])),
        "mean_episode_length": float(np.mean([r.get("episode_length", 0) for r in records])),
        "support_unique_detection_step_rate": float(np.mean([r.get("support_unique_detection_step_rate", 0.0) for r in records])),
        "support_shared_target_step_rate": float(np.mean([r.get("support_shared_target_step_rate", 0.0) for r in records])),
        "combat_attack_window_step_rate": float(np.mean([r.get("combat_attack_window_step_rate", 0.0) for r in records])),
    }


class HAPPO4v3Trainer:
    """Four red actors (support + three combat) with the existing HAPPO update."""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.env_contract_config = load_config(self.env_config)
        self.config = deepcopy(config)
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "fixed_rule_blue_heterogeneous_4v3_happo":
            raise ValueError("training_mode must be fixed_rule_blue_heterogeneous_4v3_happo")
        if int(t.get("team_size", -1)) != RED_TEAM_SIZE_4V3:
            raise ValueError("4v3 HAPPO requires training.team_size=4")
        self.device = resolve_device(e["device"])
        torch.manual_seed(int(e["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(e["seed"]))
        self.rng = np.random.default_rng(int(e["seed"]))
        self.num_envs = int(t["num_envs"])
        self.rollout_steps = int(t["rollout_steps"])
        self.envs = make_combat_vector_env_4v3(
            self.env_config,
            self.num_envs,
            int(t.get("num_env_workers", 0)),
            int(e["seed"]),
        )
        self.actors = IndependentHAPPOActors(
            [OBS_DIM_4V3] * RED_TEAM_SIZE_4V3,
            [3] * RED_TEAM_SIZE_4V3,
            hidden_dim=int(n["hidden_dim"]),
            log_std_init=float(n["log_std_init"]),
            log_std_min=float(n["log_std_min"]),
            log_std_max=float(n["log_std_max"]),
        ).to(self.device)
        self.critic = CentralizedValueCritic(GS_DIM_4V3, hidden_dim=int(n["hidden_dim"])).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=float(t["actor_learning_rate"])) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(t["critic_learning_rate"]))
        self.buffer = HAPPORolloutBuffer3v3(self.rollout_steps, self.num_envs, RED_TEAM_SIZE_4V3, OBS_DIM_4V3, 3, GS_DIM_4V3)
        self.obs, self.global_states, self.alive_masks = self.envs.reset()
        self.env_steps = 0
        self.vector_steps = 0
        self.update_count = 0
        self.last_agent_order: list[int] = list(range(RED_TEAM_SIZE_4V3))
        self.best_score = float("-inf")
        self.best_score_fields: dict[str, float] = {}
        self.evaluation_history: list[dict[str, Any]] = []
        self.recent_episodes: list[dict[str, Any]] = []
        self.last_update_metrics: dict[str, float] = {}

    def training_signature(self) -> dict[str, Any]:
        t = self.config["training"]
        return {
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_4V3,
            "env_config_sha256": sha256_file(self.env_config),
            "team_size": RED_TEAM_SIZE_4V3,
            "obs_dim": OBS_DIM_4V3,
            "state_dim": GS_DIM_4V3,
            "num_envs": int(t["num_envs"]),
            "rollout_steps": int(t["rollout_steps"]),
        }

    @torch.no_grad()
    def _select_actions(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        red_obs = torch.as_tensor(obs[:, :RED_TEAM_SIZE_4V3, :], dtype=torch.float32, device=self.device)
        actions, log_probs = self.actors.sample_actions(red_obs)
        values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device))
        return (
            actions.detach().cpu().numpy().astype(np.float32),
            log_probs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )

    def collect_rollout(self) -> list[dict[str, Any]]:
        self.buffer.clear()
        episodes: list[dict[str, Any]] = []
        for _ in range(self.rollout_steps):
            actions, log_probs, values = self._select_actions(self.obs)
            result = self.envs.step(actions)
            self.buffer.add(
                self.obs[:, :RED_TEAM_SIZE_4V3, :],
                self.global_states,
                actions,
                log_probs,
                self.alive_masks[:, :RED_TEAM_SIZE_4V3],
                result.team_rewards,
                values,
                result.terminated | result.truncated,
            )
            self.obs, self.global_states, self.alive_masks = result.observations, result.global_states, result.alive_masks
            for i, summary in enumerate(result.episode_summaries):
                if summary is not None:
                    episodes.append(summary)
                    self.obs[i], self.global_states[i], self.alive_masks[i] = self.envs.reset_at(i)
            self.vector_steps += 1
            self.env_steps += self.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(self.global_states, dtype=torch.float32, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["training"]["gamma"]), float(self.config["training"]["gae_lambda"]))
        self.recent_episodes.extend(episodes)
        self.recent_episodes = self.recent_episodes[-200:]
        return episodes

    def update(self) -> dict[str, float]:
        t = self.config["training"]
        total = self.rollout_steps * self.num_envs
        obs = torch.as_tensor(self.buffer.observations.reshape(total, RED_TEAM_SIZE_4V3, OBS_DIM_4V3), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(total, GS_DIM_4V3), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(total, RED_TEAM_SIZE_4V3, 3), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(total, RED_TEAM_SIZE_4V3), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(self.buffer.agent_alive_masks.reshape(total, RED_TEAM_SIZE_4V3), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(total), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(total), dtype=torch.float32, device=self.device)

        progress = min(1.0, self.env_steps / max(1, int(t["total_env_steps"])))
        for opt in self.actor_optimizers:
            opt.param_groups[0]["lr"] = linear_schedule(float(t["actor_learning_rate"]), float(t["actor_learning_rate_final"]), progress)
        self.critic_optimizer.param_groups[0]["lr"] = linear_schedule(float(t["critic_learning_rate"]), float(t["critic_learning_rate_final"]), progress)
        entropy_coef = linear_schedule(float(t["entropy_coef"]), float(t["entropy_coef_final"]), progress)

        idxs = np.arange(total)
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        for _ in range(int(t["ppo_epochs"])):
            self.rng.shuffle(idxs)
            for start in range(0, total, int(t["minibatch_size"])):
                mb = torch.as_tensor(idxs[start:start + int(t["minibatch_size"])], dtype=torch.long, device=self.device)
                values = self.critic(states[mb])
                critic_loss = 0.5 * (returns[mb] - values).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(t["value_loss_coef"]) * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(t["max_grad_norm"]))
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach().cpu()))

                factor = torch.ones_like(returns[mb])
                order = list(map(int, self.rng.permutation(RED_TEAM_SIZE_4V3)))
                self.last_agent_order = order
                for agent_id in order:
                    new_lp, entropy = self.actors.evaluate_agent_actions(agent_id, obs[mb, agent_id, :], actions[mb, agent_id, :])
                    active = masks[mb, agent_id]
                    adv = normalize_advantages_for_agent(advantages[mb], active)
                    ratio = torch.exp(new_lp - old_log_probs[mb, agent_id])
                    policy_loss = ppo_clipped_policy_loss(ratio, factor * adv, float(t["clip_coef"]))
                    loss = policy_loss - entropy_coef * entropy.mean()
                    opt = self.actor_optimizers[agent_id]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors.actors[agent_id].parameters(), float(t["max_grad_norm"]))
                    opt.step()
                    actor_losses.append(float(policy_loss.detach().cpu()))
                    entropies.append(float(entropy.mean().detach().cpu()))
                    with torch.no_grad():
                        new_lp_after, _ = self.actors.evaluate_agent_actions(agent_id, obs[mb, agent_id, :], actions[mb, agent_id, :])
                        approx_kls.append(float((old_log_probs[mb, agent_id] - new_lp_after).mean().detach().cpu()))
                        factor = happo_preceding_factor_update(factor, old_log_probs[mb, agent_id], new_lp_after, active)
        self.actors.clamp_log_std_()
        with torch.no_grad():
            pred = self.critic(states).detach().cpu().numpy()
        self.update_count += 1
        metrics = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "advantage_mean": float(np.mean(self.buffer.advantages)),
            "advantage_std": float(np.std(self.buffer.advantages)),
            "explained_variance": float(explained_variance(pred, self.buffer.returns.reshape(total))),
            "env_steps": float(self.env_steps),
            "vector_steps": float(self.vector_steps),
            "update_count": float(self.update_count),
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"non-finite HAPPO 4v3 update metrics: {metrics}")
        self.last_update_metrics = metrics
        return metrics

    def save_checkpoint(self, path: str | Path, *, is_best: bool = False) -> None:
        ckpt = {
            "checkpoint_family": CHECKPOINT_FAMILY_HAPPO_4V3,
            "checkpoint_version": CHECKPOINT_VERSION_HAPPO_4V3,
            "training_signature": self.training_signature(),
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizers": [o.state_dict() for o in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "last_agent_order": self.last_agent_order,
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "evaluation_history": self.evaluation_history,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "is_best": bool(is_best),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY_HAPPO_4V3:
            raise ValueError("checkpoint family mismatch")
        diffs = signature_mismatches(ckpt.get("training_signature", {}), self.training_signature())
        if diffs:
            raise ValueError("training signature mismatch:\n" + "\n".join(diffs))
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])
        for opt, state in zip(self.actor_optimizers, ckpt["actor_optimizers"]):
            opt.load_state_dict(state)
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"])
        self.vector_steps = int(ckpt["vector_steps"])
        self.update_count = int(ckpt["update_count"])
        self.last_agent_order = list(ckpt.get("last_agent_order", list(range(RED_TEAM_SIZE_4V3))))
        self.best_score = float(ckpt.get("best_score", float("-inf")))
        self.best_score_fields = dict(ckpt.get("best_score_fields", {}))
        self.evaluation_history = list(ckpt.get("evaluation_history", []))
        self.obs, self.global_states, self.alive_masks = self.envs.reset()
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu() if hasattr(ckpt["torch_cpu_rng_state"], "cpu") else ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])

    def write_summary(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "env_steps": self.env_steps,
            "vector_steps": self.vector_steps,
            "update_count": self.update_count,
            "device": str(self.device),
            "last_update_metrics": self.last_update_metrics,
            "best_score": self.best_score,
            "best_score_fields": self.best_score_fields,
            "policy_modes": self.envs.policy_modes(),
        }
        (out / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self.envs.close()


__all__ = [
    "CHECKPOINT_FAMILY_HAPPO_4V3",
    "HAPPO4v3Trainer",
    "best_score_fields_4v3",
    "compute_best_score_4v3",
    "summarize_4v3_episodes",
]
