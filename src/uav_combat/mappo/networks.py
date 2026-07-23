"""共享 Actor 与集中式 Critic。"""
import torch
from torch import nn
from torch.distributions import Normal


class SharedActor(nn.Module):
    """供红蓝双方共享的 tanh-squashed Gaussian Actor。"""

    def __init__(self, observation_dim: int = 13, action_dim: int = 3, hidden_dim: int = 128, log_std_init: float = -0.5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.epsilon = 1e-6

    def _distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.network(observation)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def sample_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """采样严格有界动作并返回带 Jacobian 修正的联合 log probability。"""
        distribution = self._distribution(observation)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = (distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + self.epsilon)).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """以稳定 atanh 反变换计算已有动作的概率与高斯熵。"""
        bounded = action.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        raw_action = torch.atanh(bounded)
        distribution = self._distribution(observation)
        log_prob = (distribution.log_prob(raw_action) - torch.log(1.0 - bounded.square() + self.epsilon)).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """返回均值经 tanh 映射后的分散执行动作。"""
        return torch.tanh(self.network(observation))


class CentralizedCritic(nn.Module):
    """从固定 red、blue 拼接的 26 维观测输出两个价值。"""

    def __init__(self, observation_dim: int = 26, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, global_observation: torch.Tensor) -> torch.Tensor:
        """计算红蓝双方集中式状态价值。"""
        return self.network(global_observation)
