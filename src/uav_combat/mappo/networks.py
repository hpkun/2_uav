"""竞争式 MAPPO 的独立 Gaussian Actor 与集中式 Critic。"""
import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """使用 tanh-squashed Gaussian 的单方分散执行 Actor。"""

    def __init__(self, observation_dim: int = 14, action_dim: int = 3, hidden_dim: int = 128,
                 log_std_init: float = -0.5, log_std_min: float = -5.0, log_std_max: float = 2.0) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.network = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.epsilon = 1e-6

    @property
    def effective_log_std_mean(self) -> float:
        with torch.no_grad():
            return float(self.log_std.clamp(self.log_std_min, self.log_std_max).mean().item())

    @property
    def effective_std_mean(self) -> float:
        with torch.no_grad():
            return float(self.log_std.clamp(self.log_std_min, self.log_std_max).exp().mean().item())

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.log_std.clamp_(self.log_std_min, self.log_std_max)

    def _distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.network(observation)
        effective_log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = effective_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def sample_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """采样有界动作及 Jacobian 修正后的 log probability。"""
        distribution = self._distribution(observation)
        raw_action = distribution.rsample(); action = torch.tanh(raw_action)
        return action, (distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + self.epsilon)).sum(-1)

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """重新计算已有动作的概率与高斯熵。"""
        bounded = action.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon); raw = torch.atanh(bounded)
        distribution = self._distribution(observation)
        log_prob = (distribution.log_prob(raw) - torch.log(1.0 - bounded.square() + self.epsilon)).sum(-1)
        return log_prob, distribution.entropy().sum(-1)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """返回 tanh(mean)。"""
        return torch.tanh(self.network(observation))


SharedActor = GaussianActor


class CentralizedCritic(nn.Module):
    """从 14 维绝对全局状态输出红蓝两个价值。"""

    def __init__(self, observation_dim: int = 14, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """计算双价值。"""
        return self.network(global_state).squeeze(-1)
