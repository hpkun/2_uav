"""FixedBlue3v3MAPPOTrainer with alive-only advantage, episode stats, best score."""
from __future__ import annotations

import time
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..environment_3v3 import GS_DIM, OBS_DIM
from .buffer_3v3 import MAPPOBuffer3v3
from .networks import CentralizedCritic, GaussianActor
from .vector_env_3v3 import (
    VectorStepResult3v3, make_combat_vector_env_3v3,
    decode_3v3_outcome, decode_3v3_termination_reason,
    RED_REWARD_COMPONENT_KEYS_3V3,
)

CHECKPOINT_VERSION_3V3 = 2
CHECKPOINT_FAMILY = "homogeneous_3v3_fixed_blue"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(requested)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return d


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation with progress clipped to [0, 1]."""
    p = float(np.clip(progress, 0.0, 1.0))
    return start + p * (end - start)


def compute_best_score(es: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic checkpoint ranking centered on genuine red attack performance."""
    return (
        float(es.get("red_complete_elimination_success_rate", 0.0)),
        float(es.get("red_any_attack_kill_rate", 1.0 if es.get("mean_red_attack_kills", 0.0) > 0.0 else 0.0)),
        float(es.get("mean_red_attack_kills", 0.0)),
        float(es.get("red_any_attack_window_rate", 0.0)),
        float(es.get("mean_red_attack_window_fraction", 0.0)),
        -float(es.get("mean_blue_survivors", 3.0)),
        float(es.get("mean_red_survivors", 0.0)),
        -float(es.get("mean_red_boundary_deaths", 0.0)),
        -float(es.get("max_steps_rate", 1.0)),
        -float(es.get("mean_episode_length", 600.0)),
    )


def compute_best_score_fields(es: dict[str, Any]) -> dict[str, float]:
    names = (
        "red_complete_elimination_success_rate",
        "red_any_attack_kill_rate",
        "mean_red_attack_kills",
        "red_any_attack_window_rate",
        "mean_red_attack_window_fraction",
        "neg_mean_blue_survivors",
        "mean_red_survivors",
        "neg_mean_red_boundary_deaths",
        "neg_max_steps_rate",
        "neg_mean_episode_length",
    )
    return {name: value for name, value in zip(names, compute_best_score(es))}


def _signature_differences(checkpoint: Any, current: Any, prefix: str = "signature") -> list[str]:
    """Return human-readable differences between checkpoint and current signatures."""
    diffs: list[str] = []
    if isinstance(checkpoint, dict) and isinstance(current, dict):
        keys = sorted(set(checkpoint) | set(current))
        for key in keys:
            child = f"{prefix}.{key}"
            if key not in checkpoint:
                diffs.append(f"- {child}: missing in checkpoint, current={current[key]!r}")
            elif key not in current:
                diffs.append(f"- {child}: checkpoint={checkpoint[key]!r}, missing in current")
            else:
                diffs.extend(_signature_differences(checkpoint[key], current[key], child))
    elif isinstance(checkpoint, list) and isinstance(current, list):
        if len(checkpoint) != len(current):
            diffs.append(f"- {prefix}: checkpoint length={len(checkpoint)} current length={len(current)}")
        else:
            for i, (a, b) in enumerate(zip(checkpoint, current)):
                diffs.extend(_signature_differences(a, b, f"{prefix}[{i}]"))
    elif checkpoint != current:
        diffs.append(f"- {prefix}: checkpoint={checkpoint!r} current={current!r}")
    return diffs


class FixedBlue3v3MAPPOTrainer:
    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.config = deepcopy(config)
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "fixed_rule_blue_3v3":
            raise ValueError("training_mode must be fixed_rule_blue_3v3")
        self.device = resolve_device(e["device"])
        torch.manual_seed(e["seed"])
        self.num_envs = int(t["num_envs"])
        self.num_env_workers = int(t.get("num_env_workers", 4))
        self.rollout_steps = int(t["rollout_steps"])
        self.total_env_steps = int(t["total_env_steps"])
        # Actor with configurable log_std bounds
        log_std_min = float(n.get("log_std_min", -5.0))
        log_std_max = float(n.get("log_std_max", 2.0))
        self.red_actor = GaussianActor(OBS_DIM, 3, n["hidden_dim"], n["log_std_init"],
                                       log_std_min=log_std_min, log_std_max=log_std_max).to(self.device)
        self.team_critic = CentralizedCritic(GS_DIM, n["hidden_dim"]).to(self.device)
        # Learning rate schedule
        self.initial_learning_rate = float(t["learning_rate"])
        self.final_learning_rate = float(t.get("learning_rate_final", self.initial_learning_rate * 0.1))
        self.initial_entropy_coef = float(t.get("entropy_coef", 0.01))
        self.final_entropy_coef = float(t.get("entropy_coef_final", self.initial_entropy_coef * 0.1))
        self.target_kl = float(t.get("target_kl", 0.0))
        self.current_learning_rate = self.initial_learning_rate
        self.current_entropy_coef = self.initial_entropy_coef
        self.actor_optimizer = torch.optim.Adam(self.red_actor.parameters(), lr=self.current_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.team_critic.parameters(), lr=self.current_learning_rate)
        # KL early-stop counter for run_summary
        self.kl_early_stop_count: int = 0
        self.vector_env = make_combat_vector_env_3v3(self.env_config, self.num_envs, self.num_env_workers)
        self.buffer = MAPPOBuffer3v3(self.rollout_steps, self.num_envs)
        self.rng = np.random.default_rng(e["seed"])
        self.env_steps = 0; self.update_count = 0
        self.current_observations = np.empty((self.num_envs, 6, OBS_DIM), np.float32)
        self.current_global_states = np.empty((self.num_envs, GS_DIM), np.float32)
        self.current_alive_masks = np.empty((self.num_envs, 6), np.float32)
        self.episode_returns = np.zeros(self.num_envs, np.float64)
        self.episode_lengths = np.zeros(self.num_envs, np.int32)
        self.best_score: tuple[float, ...] | None = None
        self.best_evaluation: dict[str, Any] | None = None
        self.best_checkpoint_name: str | None = None
        self.evaluation_history: list[dict[str, Any]] = []
        self.total_evaluation_seconds: float = 0.0
        self._timing: dict[str, float] = {"env_step": 0.0, "policy_inference": 0.0, "ppo_update": 0.0, "reset": 0.0}
        self.last_rollout_reward_means: dict[str, float] = {}
        self.reset_environments()

    def reset_environments(self):
        specs = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in range(self.num_envs)]
        obs, gs, am = self.vector_env.reset(specs)
        self.current_observations = obs; self.current_global_states = gs; self.current_alive_masks = am
        self.episode_returns.fill(0); self.episode_lengths.fill(0)

    def collect_rollout(self, remaining: int | None = None) -> list[dict[str, Any]]:
        steps = self.rollout_steps if remaining is None else min(self.rollout_steps, remaining // self.num_envs)
        if steps <= 0: raise ValueError("remaining too small")
        if self.buffer.rollout_steps != steps:
            self.buffer = MAPPOBuffer3v3(steps, self.num_envs)
        self.buffer.clear()
        completed: list[dict[str, Any]] = []
        reward_component_sum = np.zeros(len(RED_REWARD_COMPONENT_KEYS_3V3), dtype=np.float64)
        reward_component_count = 0
        action_sum = np.zeros(3, dtype=np.float64)
        action_sat_sum = np.zeros(3, dtype=np.float64)
        action_count = 0
        N = self.num_envs

        for _ in range(steps):
            t0 = time.perf_counter()
            red_obs = self.current_observations[:, :3, :]
            red_obs_flat = red_obs.reshape(-1, OBS_DIM)
            red_alive_mask = self.current_alive_masks[:, :3]
            with torch.no_grad():
                all_act, all_lp = self.red_actor.sample_action(torch.as_tensor(red_obs_flat, device=self.device))
                team_val = self.team_critic(torch.as_tensor(self.current_global_states, device=self.device))
            red_actions = all_act.cpu().numpy().reshape(N, 3, 3)
            red_log_probs = all_lp.cpu().numpy().reshape(N, 3)
            action_sum += red_actions.reshape(-1, 3).sum(axis=0)
            action_sat_sum += (np.abs(red_actions.reshape(-1, 3)) >= 0.95).sum(axis=0)
            action_count += red_actions.reshape(-1, 3).shape[0]
            t1 = time.perf_counter(); self._timing["policy_inference"] += t1 - t0

            t2 = time.perf_counter()
            r: VectorStepResult3v3 = self.vector_env.step(red_actions)
            t3 = time.perf_counter(); self._timing["env_step"] += t3 - t2

            self.episode_returns += r.team_rewards
            self.episode_lengths += 1
            reward_component_sum += r.red_reward_components.sum(axis=0)
            reward_component_count += N

            done = r.terminated | r.truncated
            done_idx = np.where(done)[0]
            if len(done_idx) > 0:
                t_reset = time.perf_counter()
                sd = np.sort(done_idx)
                for idx in sd:
                    if not r.episode_valid[idx]:
                        self.episode_returns[idx] = 0.0; self.episode_lengths[idx] = 0
                        continue
                    outcome = decode_3v3_outcome(int(r.outcome_codes[idx]))
                    reason = decode_3v3_termination_reason(int(r.termination_reason_codes[idx]))
                    rec = {
                        "episode_return": float(self.episode_returns[idx]),
                        "episode_length": int(self.episode_lengths[idx]),
                        "red_complete_elimination_success": bool(r.red_complete_elimination_success[idx]),
                        "blue_complete_elimination_success": bool(r.blue_complete_elimination_success[idx]),
                        "environment_outcome": outcome,
                        "termination_reason": reason,
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
                        "red_attack_window_agent_steps": int(r.episode_red_attack_window_agent_steps[idx]),
                        "blue_attack_window_agent_steps": int(r.episode_blue_attack_window_agent_steps[idx]),
                        "red_alive_agent_steps": int(r.episode_red_alive_agent_steps[idx]),
                        "blue_alive_agent_steps": int(r.episode_blue_alive_agent_steps[idx]),
                        "red_attack_window_fraction": float(r.episode_red_attack_window_fraction[idx]),
                        "blue_attack_window_fraction": float(r.episode_blue_attack_window_fraction[idx]),
                        "red_any_attack_window": bool(r.episode_red_any_attack_window[idx]),
                        "blue_any_attack_window": bool(r.episode_blue_any_attack_window[idx]),
                        "red_any_attack_kill": bool(r.episode_red_any_attack_kill[idx]),
                        "blue_any_attack_kill": bool(r.episode_blue_any_attack_kill[idx]),
                        "red_target_switch_count": int(r.episode_red_target_switch_count[idx]),
                        "blue_target_switch_count": int(r.episode_blue_target_switch_count[idx]),
                    }
                    # Validate death ledger
                    for team, surv, atk_d, bdy_d, fr_c, cr_c in [
                        ("red", rec["red_survivors"], rec["red_attack_deaths"],
                         rec["red_boundary_deaths"], rec["red_friendly_collision_deaths"],
                         rec["red_cross_collision_deaths"]),
                        ("blue", rec["blue_survivors"], rec["blue_attack_deaths"],
                         rec["blue_boundary_deaths"], rec["blue_friendly_collision_deaths"],
                         rec["blue_cross_collision_deaths"]),
                    ]:
                        total = surv + atk_d + bdy_d + fr_c + cr_c
                        if total != 3:
                            raise RuntimeError(f"Death ledger mismatch for {team}: {total} != 3 in env {idx}")
                        if rec[f"{team}_boundary_deaths"] != (
                            rec[f"{team}_boundary_altitude_deaths"] + rec[f"{team}_boundary_xy_deaths"]
                        ):
                            raise RuntimeError(f"Boundary death mismatch for {team} in env {idx}: {rec}")
                    # Validate attack kill symmetry
                    if rec["red_attack_kills"] != rec["blue_attack_deaths"]:
                        raise RuntimeError(f"red_attack_kills={rec['red_attack_kills']} != blue_attack_deaths={rec['blue_attack_deaths']}")
                    if rec["blue_attack_kills"] != rec["red_attack_deaths"]:
                        raise RuntimeError(f"blue_attack_kills={rec['blue_attack_kills']} != red_attack_deaths={rec['red_attack_deaths']}")
                    completed.append(rec)
                    self.episode_returns[idx] = 0.0; self.episode_lengths[idx] = 0
                specs = [{"seed": int(self.rng.integers(0, 2**31 - 1))} for _ in sd]
                no, ng, na = self.vector_env.reset_at(sd, specs)
                r.observations[sd] = no; r.global_states[sd] = ng; r.alive_masks[sd] = na
                self._timing["reset"] += time.perf_counter() - t_reset

            self.buffer.add(red_obs, self.current_global_states.copy(), red_actions, red_log_probs,
                            red_alive_mask, r.team_rewards, team_val.cpu().numpy(), done)
            self.current_observations = r.observations
            self.current_global_states = r.global_states
            self.current_alive_masks = r.alive_masks
            self.env_steps += N

        with torch.no_grad():
            lv = self.team_critic(torch.as_tensor(self.current_global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(lv, self.config["training"]["gamma"],
                                                    self.config["training"]["gae_lambda"])
        if reward_component_count > 0:
            means = {
                key: float(value)
                for key, value in zip(
                    RED_REWARD_COMPONENT_KEYS_3V3,
                    reward_component_sum / reward_component_count,
                )
            }
            death_components = (
                means["red_attack_death_penalty"],
                means["red_boundary_death_penalty"],
                means["red_collision_death_penalty"],
            )
            event_reward = means.get("red_event_reward", 0.0)
            if event_reward == 0.0:
                if any(value < 0.0 for value in death_components):
                    event_reward = means["red_kill_reward"] + sum(death_components)
                else:
                    event_reward = means["red_kill_reward"] - sum(death_components)
            threat_term = (
                means["red_threat_penalty"]
                if means["red_threat_penalty"] < 0.0
                else -means["red_threat_penalty"]
            )
            self.last_rollout_reward_means = {
                **{f"mean_rollout_{key}": value for key, value in means.items()},
                "mean_rollout_approach_reward": means["red_approach_reward"],
                "mean_rollout_attack_advantage_reward": means["red_attack_advantage_reward"],
                "mean_rollout_threat_penalty": means["red_threat_penalty"],
                "mean_rollout_soft_boundary_penalty": means["red_soft_boundary_penalty"],
                "mean_rollout_altitude_boundary_penalty": means.get("red_altitude_boundary_penalty", 0.0),
                "mean_rollout_xy_boundary_penalty": means.get("red_xy_boundary_penalty", 0.0),
                "mean_rollout_support_information_reward": means.get("red_support_information_reward", 0.0),
                "mean_rollout_friendly_separation_penalty": means["red_friendly_separation_penalty"],
                "mean_rollout_head_on_risk_penalty": means["red_head_on_risk_penalty"],
                "mean_rollout_dense_reward": means["red_dense_reward"],
                "mean_rollout_event_reward": event_reward,
                "mean_rollout_terminal_reward": means["red_terminal_reward"],
                "mean_rollout_total_step_reward": means["red_team_total_reward"],
                "mean_rollout_tactical_reward": (
                    means["red_approach_reward"]
                    + means["red_attack_advantage_reward"]
                    + threat_term
                ),
                "mean_rollout_safety_penalty": (
                    means["red_soft_boundary_penalty"]
                    + means["red_friendly_separation_penalty"]
                    + means["red_head_on_risk_penalty"]
                ),
                "mean_rollout_event_terminal_reward": event_reward + means["red_terminal_reward"],
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

    def update(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        tc = self.config["training"]

        # --- Schedules ---
        progress = float(np.clip(self.env_steps / self.total_env_steps, 0.0, 1.0))
        self.current_learning_rate = linear_schedule(self.initial_learning_rate, self.final_learning_rate, progress)
        self.current_entropy_coef = linear_schedule(self.initial_entropy_coef, self.final_entropy_coef, progress)
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.current_learning_rate
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.current_learning_rate

        target_kl = self.target_kl

        obs_flat = torch.as_tensor(self.buffer.observations.reshape(-1, OBS_DIM), device=self.device)
        act_flat = torch.as_tensor(self.buffer.actions.reshape(-1, 3), device=self.device)
        old_lp_flat = torch.as_tensor(self.buffer.log_probs.reshape(-1), device=self.device)
        adv_team = torch.as_tensor(self.buffer.advantages.reshape(-1, 1), device=self.device)
        adv = adv_team.expand(-1, 3).reshape(-1)
        alive_mask = torch.as_tensor(self.buffer.agent_alive_masks.reshape(-1), device=self.device)
        alive_adv = adv[alive_mask > 0.5]
        if len(alive_adv) > 0:
            adv_norm = (adv - alive_adv.mean()) / (alive_adv.std(unbiased=False) + 1e-8)
        else:
            adv_norm = adv
        total_slots = len(adv); alive_slots = int((alive_mask > 0.5).sum().item())

        # --- Actor update with target-KL early stop ---
        actor_data = []
        actor_epochs_done = 0
        actor_mb_done = 0
        kl_early_stop = False
        max_mb_kl = 0.0

        for epoch_i in range(tc["ppo_epochs"]):
            if kl_early_stop:
                break
            order = self.rng.permutation(len(obs_flat))
            epoch_kls = []
            for start in range(0, len(obs_flat), tc["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + tc["minibatch_size"]], device=self.device)
                idx_alive = idx[alive_mask[idx] > 0.5]
                if len(idx_alive) == 0: continue
                new_lp, ent = self.red_actor.evaluate_actions(obs_flat[idx_alive], act_flat[idx_alive])
                log_ratio = new_lp - old_lp_flat[idx_alive]; ratio = log_ratio.exp()
                a = adv_norm[idx_alive]
                pol_loss = -torch.minimum(ratio * a, ratio.clamp(1 - tc["clip_coef"], 1 + tc["clip_coef"]) * a).mean()
                loss = pol_loss - self.current_entropy_coef * ent.mean()
                self.actor_optimizer.zero_grad(); loss.backward()
                grad = nn.utils.clip_grad_norm_(self.red_actor.parameters(), tc["max_grad_norm"])
                self.actor_optimizer.step()
                self.red_actor.clamp_log_std_()
                kl = ((ratio - 1) - log_ratio).mean().item()
                cf = ((ratio - 1).abs() > tc["clip_coef"]).float().mean().item()
                if kl > max_mb_kl: max_mb_kl = kl
                actor_data.append((pol_loss.item(), ent.mean().item(), float(grad), kl, cf, len(idx_alive)))
                actor_mb_done += 1
                epoch_kls.append(kl)
                # Target-KL early stop: check after each minibatch
                if target_kl > 0 and len(epoch_kls) > 0 and float(np.mean(epoch_kls)) > target_kl:
                    kl_early_stop = True
                    break
            actor_epochs_done += 1

        if kl_early_stop:
            self.kl_early_stop_count += 1

        # --- Critic (always runs full epochs) ---
        st = torch.as_tensor(self.buffer.global_states.reshape(-1, GS_DIM), device=self.device)
        ret = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device)
        cl = []
        for _ in range(tc["ppo_epochs"]):
            order = self.rng.permutation(len(st))
            for start in range(0, len(st), tc["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + tc["minibatch_size"]], device=self.device)
                vl = ((self.team_critic(st[idx]) - ret[idx]) ** 2).mean()
                loss = tc["value_loss_coef"] * vl
                self.critic_optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.team_critic.parameters(), tc["max_grad_norm"])
                self.critic_optimizer.step(); cl.append(vl.item())

        self.update_count += 1
        self._timing["ppo_update"] += time.perf_counter() - t0
        v = np.asarray(actor_data) if actor_data else np.zeros((0, 6))
        log_std_dim = self.red_actor.effective_log_std_by_dim
        std_dim = self.red_actor.effective_std_by_dim
        return {
            "policy_loss": float(v[:, 0].mean()) if len(v) else 0.0,
            "entropy": float(v[:, 1].mean()) if len(v) else 0.0,
            "actor_grad_norm": float(v[:, 2].mean()) if len(v) else 0.0,
            "approx_kl": float(v[:, 3].mean()) if len(v) else 0.0,
            "clip_fraction": float(v[:, 4].mean()) if len(v) else 0.0,
            "alive_actor_samples": int(v[:, 5].sum()) if len(v) else 0,
            "total_actor_slots": total_slots,
            "alive_actor_sample_fraction": alive_slots / total_slots if total_slots > 0 else 0.0,
            "value_loss": float(np.mean(cl)) if cl else 0.0,
            "advantage_mean": float(alive_adv.mean()) if len(alive_adv) > 0 else 0.0,
            "advantage_std": float(alive_adv.std(unbiased=False)) if len(alive_adv) > 0 else 0.0,
            "current_learning_rate": self.current_learning_rate,
            "current_entropy_coef": self.current_entropy_coef,
            "effective_log_std_mean": self.red_actor.effective_log_std_mean,
            "effective_std_mean": self.red_actor.effective_std_mean,
            "effective_log_std_yaw": log_std_dim[0],
            "effective_log_std_pitch": log_std_dim[1],
            "effective_log_std_speed": log_std_dim[2],
            "effective_std_yaw": std_dim[0],
            "effective_std_pitch": std_dim[1],
            "effective_std_speed": std_dim[2],
            "actor_epochs_completed": actor_epochs_done,
            "actor_minibatches_completed": actor_mb_done,
            "kl_early_stop": kl_early_stop,
            "max_minibatch_kl": max_mb_kl,
        }

    def training_signature(self) -> dict[str, Any]:
        tc = self.config["training"]
        nc = self.config["network"]
        sc = deepcopy(self.config)
        sc["training"].pop("total_env_steps", None); sc["training"].pop("num_env_workers", None)
        sc["training"].pop("evaluation_interval_env_steps", None); sc["training"].pop("quick_evaluation_episodes", None)
        sc["experiment"].pop("device", None); sc["experiment"].pop("output_dir", None)
        return {"checkpoint_family": CHECKPOINT_FAMILY, "checkpoint_version": CHECKPOINT_VERSION_3V3,
                "env_config_sha256": sha256_file(self.env_config),
                "observation_dim": OBS_DIM, "global_state_dim": GS_DIM, "team_size": 3,
                "network": deepcopy(nc),
                "ppo": {k: tc[k] for k in ("learning_rate","gamma","gae_lambda","clip_coef",
                                            "value_loss_coef","max_grad_norm","ppo_epochs","minibatch_size")},
                "learning_rate_final": self.final_learning_rate,
                "entropy_coef_final": self.final_entropy_coef,
                "target_kl": self.target_kl,
                "log_std_min": nc.get("log_std_min", -5.0),
                "log_std_max": nc.get("log_std_max", 2.0),
                "num_envs": self.num_envs, "rollout_steps": self.rollout_steps, "config": sc}

    def save_checkpoint(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_family": CHECKPOINT_FAMILY, "checkpoint_version": CHECKPOINT_VERSION_3V3,
                     "config": self.config, "training_signature": self.training_signature(),
                     "env_steps": self.env_steps, "update_count": self.update_count,
                     "shared_red_actor": self.red_actor.state_dict(),
                     "centralized_team_critic": self.team_critic.state_dict(),
                     "actor_optimizer": self.actor_optimizer.state_dict(),
                     "critic_optimizer": self.critic_optimizer.state_dict(),
                     "current_learning_rate": self.current_learning_rate,
                     "current_entropy_coef": self.current_entropy_coef,
                     "kl_early_stop_count": self.kl_early_stop_count,
                     "best_score": self.best_score, "best_evaluation": self.best_evaluation,
                     "best_checkpoint_name": self.best_checkpoint_name,
                     "evaluation_history": self.evaluation_history,
                     "total_evaluation_seconds": self.total_evaluation_seconds,
                     "num_env_workers": self.num_env_workers,
                     "numpy_rng_state": self.rng.bit_generator.state,
                     "torch_cpu_rng_state": torch.get_rng_state(),
                     "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY:
            raise RuntimeError(f"Expected {CHECKPOINT_FAMILY}, got {ckpt.get('checkpoint_family')}")
        if int(ckpt.get("checkpoint_version", -1)) != CHECKPOINT_VERSION_3V3:
            raise RuntimeError(
                f"Expected checkpoint_version={CHECKPOINT_VERSION_3V3}, "
                f"got {ckpt.get('checkpoint_version')}"
            )
        if "training_signature" not in ckpt:
            raise RuntimeError("checkpoint missing training_signature")
        diffs = _signature_differences(ckpt["training_signature"], self.training_signature())
        if diffs:
            preview = "\n".join(diffs[:20])
            extra = "" if len(diffs) <= 20 else f"\n... {len(diffs) - 20} more differences"
            raise RuntimeError(f"checkpoint signature mismatch:\n{preview}{extra}")
        self.red_actor.load_state_dict(ckpt["shared_red_actor"])
        self.team_critic.load_state_dict(ckpt["centralized_team_critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"]); self.update_count = int(ckpt["update_count"])
        self.kl_early_stop_count = int(ckpt.get("kl_early_stop_count", 0))
        self.best_score = ckpt.get("best_score"); self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.evaluation_history = ckpt.get("evaluation_history", [])
        self.total_evaluation_seconds = float(ckpt.get("total_evaluation_seconds", 0.0))
        # Restore LR/entropy schedule state (will be recalculated on next update)
        self.current_learning_rate = float(ckpt.get("current_learning_rate", self.initial_learning_rate))
        self.current_entropy_coef = float(ckpt.get("current_entropy_coef", self.initial_entropy_coef))
        # The project checkpoint does not persist the physical VectorEnv state.
        # Recreate fresh episodes first, then restore algorithm RNG states so
        # reset seed generation cannot consume the checkpointed RNG position.
        self.reset_environments()
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])

    def close(self): self.vector_env.close()
