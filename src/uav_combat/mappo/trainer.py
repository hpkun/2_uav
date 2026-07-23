"""参数共享 MAPPO 的同步收集、PPO 更新、评估与检查点。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..environment import HomogeneousAirCombatEnv
from ..rule_policy import PurePursuitPolicy
from .buffer import MAPPOBuffer
from .networks import CentralizedCritic, SharedActor

AGENT_IDS = ("red_0", "blue_0")


def resolve_device(requested: str) -> torch.device:
    """解析 auto/cpu/cuda 设备设置。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class MAPPOTrainer:
    """双方共享 Actor、集中式双价值 Critic 的简化 MAPPO 基线。"""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config = str(env_config)
        self.config = config
        training, network, experiment = config["training"], config["network"], config["experiment"]
        self.device = resolve_device(experiment["device"])
        self.num_envs = int(training["num_envs"])
        self.rollout_steps = int(training["rollout_steps"])
        actor_samples = self.num_envs * self.rollout_steps * 2
        if training["minibatch_size"] > actor_samples:
            raise ValueError("minibatch_size cannot exceed rollout_steps * num_envs * 2")
        self.actor = SharedActor(13, 3, network["hidden_dim"], network["log_std_init"]).to(self.device)
        self.critic = CentralizedCritic(26, network["hidden_dim"]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=training["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=training["learning_rate"])
        self.envs = [HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]
        self.buffer = MAPPOBuffer(self.rollout_steps, self.num_envs)
        self.rng = np.random.default_rng(experiment["seed"])
        self.current_observations: list[dict[str, np.ndarray]] = []
        for index, env in enumerate(self.envs):
            observation, _ = env.reset(seed=experiment["seed"] + index)
            self.current_observations.append(observation)
        self.episode_returns = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.update_count = 0
        self.env_steps = 0

    def collect_rollout(self) -> list[dict[str, Any]]:
        """同步收集固定长度 rollout，结束的环境立即随机重置。"""
        self.buffer.clear()
        completed: list[dict[str, Any]] = []
        for _ in range(self.rollout_steps):
            observations = np.asarray([
                [item["red_0"], item["blue_0"]] for item in self.current_observations
            ], dtype=np.float32)
            globals_ = observations.reshape(self.num_envs, 26)
            with torch.no_grad():
                observation_tensor = torch.as_tensor(observations.reshape(-1, 13), device=self.device)
                actions_tensor, log_probs_tensor = self.actor.sample_action(observation_tensor)
                values_tensor = self.critic(torch.as_tensor(globals_, device=self.device))
            actions = actions_tensor.cpu().numpy().reshape(self.num_envs, 2, 3)
            log_probs = log_probs_tensor.cpu().numpy().reshape(self.num_envs, 2)
            values = values_tensor.cpu().numpy()
            rewards = np.zeros((self.num_envs, 2), dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            next_observations: list[dict[str, np.ndarray]] = []
            for env_index, env in enumerate(self.envs):
                observation, reward, terminated, truncated, info = env.step({
                    "red_0": actions[env_index, 0], "blue_0": actions[env_index, 1]
                })
                rewards[env_index] = [reward["red_0"], reward["blue_0"]]
                self.episode_returns[env_index] += rewards[env_index]
                self.episode_lengths[env_index] += 1
                done = terminated or truncated
                dones[env_index] = done
                if done:
                    completed.append({
                        "returns": self.episode_returns[env_index].copy(),
                        "length": int(self.episode_lengths[env_index]),
                        "outcome": info["outcome"],
                        "scenario_name": info["scenario_name"],
                    })
                    self.episode_returns[env_index] = 0.0
                    self.episode_lengths[env_index] = 0
                    observation, _ = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                next_observations.append(observation)
            self.buffer.add(observations, globals_, actions, log_probs, rewards, values, dones)
            self.current_observations = next_observations
            self.env_steps += self.num_envs
        final_observations = np.asarray([
            [item["red_0"], item["blue_0"]] for item in self.current_observations
        ], dtype=np.float32)
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(final_observations.reshape(self.num_envs, 26), device=self.device)).cpu().numpy()
        training = self.config["training"]
        self.buffer.compute_returns_and_advantages(last_values, training["gamma"], training["gae_lambda"])
        return completed

    def update(self) -> dict[str, float]:
        """在红蓝联合 Actor 样本和集中式 Critic 样本上执行多轮 PPO 更新。"""
        training = self.config["training"]
        actor_observations = torch.as_tensor(self.buffer.observations.reshape(-1, 13), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        global_observations = torch.as_tensor(self.buffer.global_observations.reshape(-1, 26), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1, 2), device=self.device)
        old_values = self.buffer.values.reshape(-1, 2)
        minibatch = int(training["minibatch_size"])
        actor_metrics: list[tuple[float, float, float, float, float]] = []
        critic_metrics: list[float] = []
        for _ in range(int(training["ppo_epochs"])):
            actor_indices = self.rng.permutation(len(actor_observations))
            for start in range(0, len(actor_indices), minibatch):
                index = torch.as_tensor(actor_indices[start:start + minibatch], device=self.device)
                new_log_prob, entropy = self.actor.evaluate_actions(actor_observations[index], actions[index])
                log_ratio = new_log_prob - old_log_probs[index]
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - training["clip_coef"], 1.0 + training["clip_coef"])
                policy_loss = -torch.minimum(ratio * advantages[index], clipped_ratio * advantages[index]).mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - training["entropy_coef"] * entropy_mean
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), training["max_grad_norm"])
                self._require_finite("actor", actor_loss, grad_norm)
                self.actor_optimizer.step()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > training["clip_coef"]).float().mean()
                actor_metrics.append((policy_loss.item(), entropy_mean.item(), approx_kl.item(), clip_fraction.item(), float(grad_norm)))

            critic_indices = self.rng.permutation(len(global_observations))
            critic_minibatch = min(minibatch, len(critic_indices))
            for start in range(0, len(critic_indices), critic_minibatch):
                index = torch.as_tensor(critic_indices[start:start + critic_minibatch], device=self.device)
                predicted = self.critic(global_observations[index])
                value_loss = torch.mean((predicted - returns[index]) ** 2)
                self.critic_optimizer.zero_grad()
                (training["value_loss_coef"] * value_loss).backward()
                grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), training["max_grad_norm"])
                self._require_finite("critic", value_loss, grad_norm)
                self.critic_optimizer.step()
                critic_metrics.append(value_loss.item())
        self._check_parameters()
        returns_array = self.buffer.returns.reshape(-1)
        values_array = old_values.reshape(-1)
        variance = np.var(returns_array)
        explained_variance = float(1.0 - np.var(returns_array - values_array) / variance) if variance > 1e-12 else 0.0
        self.update_count += 1
        actor_array = np.asarray(actor_metrics)
        metrics = {
            "policy_loss": float(actor_array[:, 0].mean()),
            "entropy": float(actor_array[:, 1].mean()),
            "approx_kl": float(actor_array[:, 2].mean()),
            "clip_fraction": float(actor_array[:, 3].mean()),
            "actor_grad_norm": float(actor_array[:, 4].mean()),
            "value_loss": float(np.mean(critic_metrics)),
            "explained_variance": explained_variance,
        }
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError("non-finite PPO metrics")
        return metrics

    @staticmethod
    def _require_finite(label: str, *values: torch.Tensor) -> None:
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError(f"non-finite {label} loss or gradient")

    def _check_parameters(self) -> None:
        if not all(torch.isfinite(parameter).all() for module in (self.actor, self.critic) for parameter in module.parameters()):
            raise FloatingPointError("non-finite network parameter")

    def save_checkpoint(self, path: str | Path) -> None:
        """保存网络、优化器、训练进度和完整配置。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "update": self.update_count, "env_steps": self.env_steps,
            "config": self.config, "seed": self.config["experiment"]["seed"],
        }, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> None:
        """恢复网络和可选优化器状态。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint["update"])
        self.env_steps = int(checkpoint["env_steps"])


def evaluate_policy(actor: SharedActor, env_config: str | Path, episodes: int, device: torch.device, opponent: str = "zero", side: str = "both", scenario: str = "all", seed: int = 10000) -> dict[str, float | int]:
    """以分散确定性 Actor 对 zero 或 pursuit 对手进行固定侧/双侧评估。"""
    if opponent not in {"zero", "pursuit"} or side not in {"red", "blue", "both"}:
        raise ValueError("invalid opponent or side")
    templates = ("tail_chase", "offset_head_on", "crossing")
    outcomes = {"wins": 0, "losses": 0, "draws": 0}
    returns, lengths, distances, atas, aas = [], [], [], [], []
    actor.eval()
    for episode in range(episodes):
        learned_side = side if side != "both" else ("red" if episode % 2 == 0 else "blue")
        scenario_name = templates[episode % len(templates)] if scenario == "all" else scenario
        env = HomogeneousAirCombatEnv(env_config)
        observation, _ = env.reset(seed=seed + episode, scenario_name=scenario_name)
        action_config = env.config["action"]
        pursuit = PurePursuitPolicy(action_config["delta_yaw_max"], action_config["delta_pitch_max"], action_config["delta_speed_max"])
        episode_return = 0.0
        info: dict[str, Any] = {}
        for _ in range(env.config["simulation"]["max_steps"]):
            actions: dict[str, np.ndarray] = {}
            for agent_index, agent_id in enumerate(AGENT_IDS):
                team = agent_id.split("_")[0]
                if team == learned_side:
                    with torch.no_grad():
                        tensor = torch.as_tensor(observation[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                        actions[agent_id] = actor.deterministic_action(tensor).squeeze(0).cpu().numpy()
                elif opponent == "zero":
                    actions[agent_id] = np.zeros(3, dtype=float)
                else:
                    own = next(aircraft for aircraft in env.aircraft if aircraft.aircraft_id == agent_id)
                    target = next(aircraft for aircraft in env.aircraft if aircraft.team != own.team)
                    actions[agent_id] = pursuit.action(own, target)
            observation, rewards, terminated, truncated, info = env.step(actions)
            episode_return += rewards[f"{learned_side}_0"]
            if terminated or truncated:
                break
        if info["outcome"] == learned_side:
            outcomes["wins"] += 1
        elif info["outcome"] == "draw":
            outcomes["draws"] += 1
        else:
            outcomes["losses"] += 1
        geometry = info["geometries"][f"{learned_side}_0"]
        returns.append(episode_return); lengths.append(info["step_count"])
        distances.append(geometry.distance); atas.append(geometry.ata); aas.append(geometry.aa)
    actor.train()
    count = max(episodes, 1)
    return {
        "episodes": episodes, **outcomes,
        "win_rate": outcomes["wins"] / count, "loss_rate": outcomes["losses"] / count, "draw_rate": outcomes["draws"] / count,
        "mean_return": float(np.mean(returns)), "mean_episode_length": float(np.mean(lengths)),
        "mean_terminal_distance": float(np.mean(distances)), "mean_ATA": float(np.mean(atas)), "mean_AA": float(np.mean(aas)),
    }
