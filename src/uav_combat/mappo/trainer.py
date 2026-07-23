"""双独立 Actor、集中式双价值 Critic 的竞争式 MAPPO。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from ..environment import HomogeneousAirCombatEnv
from ..rule_policy import PurePursuitPolicy
from .buffer import MAPPOBuffer
from .networks import CentralizedCritic, GaussianActor

AGENT_IDS = ("red_0", "blue_0")


def resolve_device(requested: str) -> torch.device:
    """解析 auto/cpu/cuda，显式 cuda 不可用时拒绝回退。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class MAPPOTrainer:
    """红蓝策略独立更新、Critic 集中训练的竞争式 MAPPO。"""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config, self.config = str(env_config), config
        training, network, experiment = config["training"], config["network"], config["experiment"]
        self.device = resolve_device(experiment["device"]); self.num_envs = int(training["num_envs"]); self.rollout_steps = int(training["rollout_steps"])
        if training["minibatch_size"] > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed rollout_steps * num_envs per actor")
        self.red_actor = GaussianActor(14, 3, network["hidden_dim"], network["log_std_init"]).to(self.device)
        self.blue_actor = GaussianActor(14, 3, network["hidden_dim"], network["log_std_init"]).to(self.device)
        self.critic = CentralizedCritic(14, network["hidden_dim"]).to(self.device)
        lr = training["learning_rate"]
        self.red_actor_optimizer = torch.optim.Adam(self.red_actor.parameters(), lr=lr)
        self.blue_actor_optimizer = torch.optim.Adam(self.blue_actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.envs = [HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]
        self.buffer = MAPPOBuffer(self.rollout_steps, self.num_envs)
        self.rng = np.random.default_rng(experiment["seed"]); self.env_steps = 0; self.update_count = 0
        self.current_observations = [env.reset(experiment["seed"] + i, "tail_chase")[0] for i, env in enumerate(self.envs)]
        self.episode_returns = np.zeros((self.num_envs, 2)); self.episode_lengths = np.zeros(self.num_envs, dtype=int)

    def phase(self) -> str:
        """返回当前课程阶段。"""
        return "tail_chase_curriculum" if self.env_steps < self.config["training"]["curriculum_tail_chase_env_steps"] else "all_scenarios"

    def _reset_scenario(self) -> str | None:
        return "tail_chase" if self.phase() == "tail_chase_curriculum" else None

    def collect_rollout(self) -> list[dict[str, Any]]:
        """分别用红蓝 Actor 采样，并从环境绝对全局状态估值。"""
        self.buffer.clear(); completed: list[dict[str, Any]] = []
        for _ in range(self.rollout_steps):
            observations = np.asarray([[item["red_0"], item["blue_0"]] for item in self.current_observations], dtype=np.float32)
            global_states = np.asarray([env.global_state() for env in self.envs], dtype=np.float32)
            with torch.no_grad():
                red_actions, red_logs = self.red_actor.sample_action(torch.as_tensor(observations[:, 0], device=self.device))
                blue_actions, blue_logs = self.blue_actor.sample_action(torch.as_tensor(observations[:, 1], device=self.device))
                values = self.critic(torch.as_tensor(global_states, device=self.device))
            actions = np.stack((red_actions.cpu().numpy(), blue_actions.cpu().numpy()), axis=1)
            log_probs = np.stack((red_logs.cpu().numpy(), blue_logs.cpu().numpy()), axis=1)
            rewards = np.zeros((self.num_envs, 2), np.float32); dones = np.zeros(self.num_envs, bool); next_observations = []
            for index, env in enumerate(self.envs):
                observation, reward, terminated, truncated, info = env.step({"red_0": actions[index, 0], "blue_0": actions[index, 1]})
                rewards[index] = [reward["red_0"], reward["blue_0"]]; self.episode_returns[index] += rewards[index]; self.episode_lengths[index] += 1
                done = terminated or truncated; dones[index] = done
                if done:
                    completed.append({"returns": self.episode_returns[index].copy(), "length": int(self.episode_lengths[index]), "outcome": info["outcome"], "reason": info["termination_reason"], "scenario_name": info["scenario_name"]})
                    self.episode_returns[index] = 0; self.episode_lengths[index] = 0
                    observation, _ = env.reset(int(self.rng.integers(2**31 - 1)), self._reset_scenario())
                next_observations.append(observation)
            self.buffer.add(observations, global_states, actions, log_probs, rewards, values.cpu().numpy(), dones)
            self.current_observations = next_observations; self.env_steps += self.num_envs
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(np.asarray([env.global_state() for env in self.envs], dtype=np.float32), device=self.device)).cpu().numpy()
        training = self.config["training"]; self.buffer.compute_returns_and_advantages(last_values, training["gamma"], training["gae_lambda"])
        return completed

    def _update_actor(self, actor: GaussianActor, optimizer: torch.optim.Optimizer, side: int, label: str) -> dict[str, float]:
        training = self.config["training"]
        observations = torch.as_tensor(self.buffer.observations[:, :, side].reshape(-1, 14), device=self.device)
        actions = torch.as_tensor(self.buffer.actions[:, :, side].reshape(-1, 3), device=self.device)
        old_logs = torch.as_tensor(self.buffer.log_probs[:, :, side].reshape(-1), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages[:, :, side].reshape(-1), device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        metrics = []
        for _ in range(int(training["ppo_epochs"])):
            indices = self.rng.permutation(len(observations))
            for start in range(0, len(indices), int(training["minibatch_size"])):
                index = torch.as_tensor(indices[start:start + int(training["minibatch_size"])], device=self.device)
                new_logs, entropy = actor.evaluate_actions(observations[index], actions[index]); log_ratio = new_logs - old_logs[index]; ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - training["clip_coef"], 1 + training["clip_coef"])
                policy_loss = -torch.minimum(ratio * advantages[index], clipped * advantages[index]).mean(); entropy_mean = entropy.mean()
                loss = policy_loss - training["entropy_coef"] * entropy_mean
                optimizer.zero_grad(); loss.backward(); grad = nn.utils.clip_grad_norm_(actor.parameters(), training["max_grad_norm"])
                self._require_finite(label, loss, grad); optimizer.step()
                kl = ((ratio - 1) - log_ratio).mean(); clip = ((ratio - 1).abs() > training["clip_coef"]).float().mean()
                metrics.append((policy_loss.item(), entropy_mean.item(), kl.item(), clip.item(), float(grad)))
        array = np.asarray(metrics)
        return {"policy_loss": float(array[:, 0].mean()), "entropy": float(array[:, 1].mean()), "approx_kl": float(array[:, 2].mean()), "clip_fraction": float(array[:, 3].mean()), "grad_norm": float(array[:, 4].mean())}

    def update(self) -> dict[str, float]:
        """红蓝 advantage 分离标准化并分别更新两个 Actor。"""
        red = self._update_actor(self.red_actor, self.red_actor_optimizer, 0, "red_actor")
        blue = self._update_actor(self.blue_actor, self.blue_actor_optimizer, 1, "blue_actor")
        training = self.config["training"]; states = torch.as_tensor(self.buffer.global_states.reshape(-1, 14), device=self.device); returns = torch.as_tensor(self.buffer.returns.reshape(-1, 2), device=self.device)
        losses = []
        for _ in range(int(training["ppo_epochs"])):
            indices = self.rng.permutation(len(states))
            for start in range(0, len(indices), int(training["minibatch_size"])):
                index = torch.as_tensor(indices[start:start + int(training["minibatch_size"])], device=self.device)
                loss = ((self.critic(states[index]) - returns[index]) ** 2).mean(); self.critic_optimizer.zero_grad(); (training["value_loss_coef"] * loss).backward()
                grad = nn.utils.clip_grad_norm_(self.critic.parameters(), training["max_grad_norm"]); self._require_finite("critic", loss, grad); self.critic_optimizer.step(); losses.append(loss.item())
        metrics: dict[str, float] = {f"red_{key}": value for key, value in red.items()} | {f"blue_{key}": value for key, value in blue.items()}
        metrics["value_loss"] = float(np.mean(losses))
        for side, name in enumerate(("red", "blue")):
            actual, predicted = self.buffer.returns[:, :, side].reshape(-1), self.buffer.values[:, :, side].reshape(-1); variance = np.var(actual)
            metrics[f"{name}_explained_variance"] = float(1 - np.var(actual - predicted) / variance) if variance > 1e-12 else 0.0
        self.update_count += 1; self._check_parameters()
        if not np.all(np.isfinite(list(metrics.values()))): raise FloatingPointError("non-finite PPO metrics")
        return metrics

    @staticmethod
    def _require_finite(label: str, *values: torch.Tensor) -> None:
        if not all(torch.isfinite(value).all() for value in values): raise FloatingPointError(f"non-finite {label} loss or gradient")

    def _check_parameters(self) -> None:
        if not all(torch.isfinite(p).all() for m in (self.red_actor, self.blue_actor, self.critic) for p in m.parameters()): raise FloatingPointError("non-finite network parameter")

    def save_checkpoint(self, path: str | Path) -> None:
        """保存 v2 双 Actor 检查点。"""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_version": 2, "red_actor": self.red_actor.state_dict(), "blue_actor": self.blue_actor.state_dict(), "critic": self.critic.state_dict(), "red_actor_optimizer": self.red_actor_optimizer.state_dict(), "blue_actor_optimizer": self.blue_actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(), "update": self.update_count, "env_steps": self.env_steps, "config": self.config, "seed": self.config["experiment"]["seed"]}, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> None:
        """加载 v2；明确拒绝旧共享 Actor 检查点。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version", 0) < 2: raise RuntimeError("旧共享Actor检查点不能恢复到竞争式双Actor训练；需要 checkpoint_version >= 2")
        self.red_actor.load_state_dict(checkpoint["red_actor"]); self.blue_actor.load_state_dict(checkpoint["blue_actor"]); self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.red_actor_optimizer.load_state_dict(checkpoint["red_actor_optimizer"]); self.blue_actor_optimizer.load_state_dict(checkpoint["blue_actor_optimizer"]); self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count, self.env_steps = int(checkpoint["update"]), int(checkpoint["env_steps"])


def evaluate_actor(actor: GaussianActor, env_config: str | Path, episodes: int, device: torch.device, opponent: str = "zero", side: str = "both", scenario: str = "all", seed: int = 10000) -> dict[str, Any]:
    """按整体和模板评估单个 Actor。"""
    templates = ("tail_chase", "offset_head_on", "crossing"); records = []
    actor.eval()
    for episode in range(episodes):
        learned = side if side != "both" else ("red" if episode % 2 == 0 else "blue"); name = templates[episode % 3] if scenario == "all" else scenario
        env = HomogeneousAirCombatEnv(env_config); observations, _ = env.reset(seed + episode, name); action_config = env.config["action"]
        pursuit = PurePursuitPolicy(action_config["delta_yaw_max"], action_config["delta_pitch_max"], action_config["delta_speed_max"]); total = 0.0
        for _ in range(env.config["simulation"]["max_steps"]):
            actions = {}
            for agent_id in AGENT_IDS:
                team = agent_id.split("_")[0]
                if team == learned:
                    with torch.no_grad(): actions[agent_id] = actor.deterministic_action(torch.as_tensor(observations[agent_id], dtype=torch.float32, device=device)[None]).squeeze(0).cpu().numpy()
                elif opponent == "zero": actions[agent_id] = np.zeros(3)
                else:
                    own = next(a for a in env.aircraft if a.aircraft_id == agent_id); target = next(a for a in env.aircraft if a.team != own.team); actions[agent_id] = pursuit.action(own, target)
            observations, rewards, terminated, truncated, info = env.step(actions); total += rewards[f"{learned}_0"]
            if terminated or truncated: break
        result = "win" if info["outcome"] == learned else ("draw" if info["outcome"] == "draw" else "loss")
        records.append({"scenario": name, "result": result, "return": total, "length": info["step_count"]})
    actor.train()
    def summarize(items: list[dict[str, Any]]) -> dict[str, float | int]:
        count = len(items); return {"episodes": count, "wins": sum(x["result"] == "win" for x in items), "losses": sum(x["result"] == "loss" for x in items), "draws": sum(x["result"] == "draw" for x in items), "win_rate": sum(x["result"] == "win" for x in items) / count, "loss_rate": sum(x["result"] == "loss" for x in items) / count, "draw_rate": sum(x["result"] == "draw" for x in items) / count, "mean_return": float(np.mean([x["return"] for x in items])), "mean_episode_length": float(np.mean([x["length"] for x in items]))}
    return {"overall": summarize(records), "by_scenario": {name: summarize([x for x in records if x["scenario"] == name]) for name in templates if any(x["scenario"] == name for x in records)}}
