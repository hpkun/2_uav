"""固定长度 MAPPO rollout buffer 与 GAE。"""
import numpy as np


class MAPPOBuffer:
    """保存 [T,N,2] 红蓝固定顺序的 on-policy rollout。"""

    def __init__(self, rollout_steps: int, num_envs: int, observation_dim: int = 13, action_dim: int = 3) -> None:
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must be positive")
        self.rollout_steps, self.num_envs = rollout_steps, num_envs
        self.observations = np.zeros((rollout_steps, num_envs, 2, observation_dim), dtype=np.float32)
        self.global_observations = np.zeros((rollout_steps, num_envs, observation_dim * 2), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, 2, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=bool)
        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)
        self.position = 0

    def add(self, observations: np.ndarray, global_observations: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> None:
        """加入一个并行环境时间步，并严格检查维数。"""
        if self.position >= self.rollout_steps:
            raise RuntimeError("buffer is full")
        expected = [
            (observations, self.observations.shape[1:]), (global_observations, self.global_observations.shape[1:]),
            (actions, self.actions.shape[1:]), (log_probs, self.log_probs.shape[1:]),
            (rewards, self.rewards.shape[1:]), (values, self.values.shape[1:]), (dones, self.dones.shape[1:]),
        ]
        for value, shape in expected:
            if np.asarray(value).shape != shape:
                raise ValueError(f"expected shape {shape}, got {np.asarray(value).shape}")
        index = self.position
        self.observations[index] = observations
        self.global_observations[index] = global_observations
        self.actions[index] = actions
        self.log_probs[index] = log_probs
        self.rewards[index] = rewards
        self.values[index] = values
        self.dones[index] = dones
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """分别为两个智能体反向计算 GAE，done 后不跨 episode 传播。"""
        if self.position != self.rollout_steps or np.asarray(last_values).shape != (self.num_envs, 2):
            raise ValueError("buffer must be full and last_values must have shape (N,2)")
        next_advantage = np.zeros((self.num_envs, 2), dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_value = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
            nonterminal = (~self.dones[step]).astype(np.float32)[:, None]
            delta = self.rewards[step] + gamma * next_value * nonterminal - self.values[step]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            self.advantages[step] = next_advantage
        self.returns[:] = self.advantages + self.values
        if not np.all(np.isfinite(self.advantages)) or not np.all(np.isfinite(self.returns)):
            raise FloatingPointError("non-finite GAE result")

    def clear(self) -> None:
        """清零所有数组并重置写入位置。"""
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.fill(0)
        self.position = 0
