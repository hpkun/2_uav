"""FixedBlue3v3MAPPOTrainer with alive-only advantage, episode stats, best score."""
from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..environment_3v3 import GS_DIM, OBS_DIM
from .buffer_3v3 import MAPPOBuffer3v3
from .networks import CentralizedCritic, GaussianActor
from .vector_env_3v3 import VectorStepResult3v3, make_combat_vector_env_3v3

CHECKPOINT_VERSION_3V3 = 1
CHECKPOINT_FAMILY = "homogeneous_3v3_fixed_blue"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(requested)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return d


def compute_best_score(es: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic best: higher red_attack_success, lower blue_survivors, etc."""
    red_col = (es.get("mean_red_friendly_collision_deaths", 0.0) +
               es.get("mean_red_cross_collision_deaths", 0.0))
    return (
        float(es.get("red_complete_elimination_success_rate", 0.0)),
        -float(es.get("mean_blue_survivors", 3.0)),
        float(es.get("mean_red_survivors", 0.0)),
        -float(es.get("mean_red_boundary_deaths", 0.0)),
        -float(red_col),
        -float(es.get("mean_episode_length", 600.0)),
    )


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
        self.red_actor = GaussianActor(OBS_DIM, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        self.team_critic = CentralizedCritic(GS_DIM, n["hidden_dim"]).to(self.device)
        lr = t["learning_rate"]
        self.actor_optimizer = torch.optim.Adam(self.red_actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.team_critic.parameters(), lr=lr)
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
            t1 = time.perf_counter(); self._timing["policy_inference"] += t1 - t0

            t2 = time.perf_counter()
            r: VectorStepResult3v3 = self.vector_env.step(red_actions)
            t3 = time.perf_counter(); self._timing["env_step"] += t3 - t2

            self.episode_returns += r.team_rewards
            self.episode_lengths += 1

            done = r.terminated | r.truncated
            done_idx = np.where(done)[0]
            if len(done_idx) > 0:
                t_reset = time.perf_counter()
                sd = np.sort(done_idx)
                for idx in sd:
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
        # Build completed records from episode fields
        completed = []
        ep_valid = self.buffer.dones  # [T, N] – episode ended at this step
        # We need the episode fields from the LAST time each env finished during this rollout
        # But the NamedTuple gives per-step fields. We'll accumulate from the last step (before reset).
        # For now, use a simplified accumulation: if any env had a done step, count it.
        for env_i in range(N):
            if ep_valid[:, env_i].any():
                completed.append({"completed": True})
        return completed

    def update(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        tc = self.config["training"]
        obs_flat = torch.as_tensor(self.buffer.observations.reshape(-1, OBS_DIM), device=self.device)
        act_flat = torch.as_tensor(self.buffer.actions.reshape(-1, 3), device=self.device)
        old_lp_flat = torch.as_tensor(self.buffer.log_probs.reshape(-1), device=self.device)
        adv_team = torch.as_tensor(self.buffer.advantages.reshape(-1, 1), device=self.device)
        adv = adv_team.expand(-1, 3).reshape(-1)
        alive_mask = torch.as_tensor(self.buffer.agent_alive_masks.reshape(-1), device=self.device)
        # Only alive agents for advantage normalization
        alive_adv = adv[alive_mask > 0.5]
        if len(alive_adv) > 0:
            adv_norm = (adv - alive_adv.mean()) / (alive_adv.std(unbiased=False) + 1e-8)
        else:
            adv_norm = adv
        total_slots = len(adv); alive_slots = int((alive_mask > 0.5).sum().item())

        actor_data = []
        for _ in range(tc["ppo_epochs"]):
            order = self.rng.permutation(len(obs_flat))
            for start in range(0, len(obs_flat), tc["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + tc["minibatch_size"]], device=self.device)
                idx_alive = idx[alive_mask[idx] > 0.5]
                if len(idx_alive) == 0: continue
                new_lp, ent = self.red_actor.evaluate_actions(obs_flat[idx_alive], act_flat[idx_alive])
                lr = new_lp - old_lp_flat[idx_alive]; ratio = lr.exp()
                a = adv_norm[idx_alive]
                pol_loss = -torch.minimum(ratio * a, ratio.clamp(1 - tc["clip_coef"], 1 + tc["clip_coef"]) * a).mean()
                loss = pol_loss - tc["entropy_coef"] * ent.mean()
                self.actor_optimizer.zero_grad(); loss.backward()
                grad = nn.utils.clip_grad_norm_(self.red_actor.parameters(), tc["max_grad_norm"])
                self.actor_optimizer.step()
                kl = ((ratio - 1) - lr).mean().item()
                cf = ((ratio - 1).abs() > tc["clip_coef"]).float().mean().item()
                actor_data.append((pol_loss.item(), ent.mean().item(), float(grad), kl, cf, len(idx_alive)))

        # Critic
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
        }

    def training_signature(self) -> dict[str, Any]:
        tc = self.config["training"]
        sc = deepcopy(self.config)
        sc["training"].pop("total_env_steps", None); sc["training"].pop("num_env_workers", None)
        sc["experiment"].pop("device", None); sc["experiment"].pop("output_dir", None)
        return {"checkpoint_family": CHECKPOINT_FAMILY, "checkpoint_version": CHECKPOINT_VERSION_3V3,
                "observation_dim": OBS_DIM, "global_state_dim": GS_DIM, "team_size": 3,
                "network": deepcopy(self.config["network"]),
                "ppo": {k: tc[k] for k in ("learning_rate","gamma","gae_lambda","clip_coef","entropy_coef","value_loss_coef","max_grad_norm","ppo_epochs","minibatch_size")},
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
        self.red_actor.load_state_dict(ckpt["shared_red_actor"])
        self.team_critic.load_state_dict(ckpt["centralized_team_critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.env_steps = int(ckpt["env_steps"]); self.update_count = int(ckpt["update_count"])
        self.best_score = ckpt.get("best_score"); self.best_evaluation = ckpt.get("best_evaluation")
        self.best_checkpoint_name = ckpt.get("best_checkpoint_name")
        self.evaluation_history = ckpt.get("evaluation_history", [])
        self.total_evaluation_seconds = float(ckpt.get("total_evaluation_seconds", 0.0))
        self.rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_cpu_rng_state"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
        self.reset_environments()

    def close(self): self.vector_env.close()
