from __future__ import annotations

import numpy as np
import torch

from algorithm.common.buffer import RolloutBuffer
from algorithm.common.networks import GaussianActor
from algorithm.happo.trainer import preceding_factor_update
from env.mavuav import RED_IDS
from tools.audit_happo_health import summarize_window, training_stability


def _row(step: int, episodes: int, value: float) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "sampled_steps": step, "completed_episodes": episodes,
        "episode_weight": episodes,
        "curriculum_alpha": min(step / 400_000, 1.0),
    }
    from tools.audit_happo_health import EPISODE_METRICS, UPDATE_METRICS
    for field in EPISODE_METRICS:
        row[field] = value
    for field in UPDATE_METRICS:
        row[field] = value
    return row


def test_preceding_factor_is_active_only_ratio_and_detached():
    factor = torch.tensor([2.0, 3.0], requires_grad=True)
    old = torch.tensor([0.1, -0.2], requires_grad=True)
    new = torch.tensor([0.4, 0.7], requires_grad=True)
    result = preceding_factor_update(factor, old, new, torch.tensor([1.0, 0.0]))
    assert torch.allclose(result, result.new_tensor([2.0 * np.exp(0.3), 3.0]))
    assert not result.requires_grad


def test_team_gae_stops_on_terminated_and_truncated_boundaries():
    buffer = RolloutBuffer(horizon=2, num_envs=2)
    buffer.position = 2
    buffer.rewards[:] = 1.0
    buffer.values[:] = 0.0
    buffer.terminated[0, 0] = True
    buffer.truncated[0, 1] = True
    buffer.compute_returns_and_advantages(np.asarray([10.0, 10.0], np.float32), 1.0, 1.0)
    assert np.allclose(buffer.advantages[0], [1.0, 1.0])
    assert np.allclose(buffer.advantages[1], [11.0, 11.0])


def test_squashed_gaussian_rollout_and_training_log_prob_match():
    torch.manual_seed(19)
    actor = GaussianActor(observation_dim=5, action_dim=3, hidden_dim=8)
    observations = torch.randn(32, 5)
    actions, rollout_log_prob = actor.sample(observations)
    training_log_prob, entropy = actor.evaluate_actions(observations, actions)
    assert torch.allclose(rollout_log_prob, training_log_prob, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(entropy).all()


def test_training_windows_use_completed_episode_weights():
    first = _row(50_000, 1, 0.0)
    second = _row(100_000, 9, 10.0)
    second["episode_weight"] = 8
    result = summarize_window([first, second], seed=5, lower=0, upper=100_000)
    assert result["completed_episodes"] == 9
    assert np.isclose(result["mean_episode_return"], 80.0 / 9.0)
    # Optimizer metrics are per update, not per completed episode.
    assert np.isclose(result["critic_loss"], 5.0)


def test_late_stability_reports_slope_range_and_drawdown():
    rows = []
    completed = 0
    for index in range(10):
        completed += 10
        row = _row((index + 1) * 100_000, completed, float(index))
        row["episode_weight"] = 10
        rows.append(row)
    windows, late = training_stability(rows, seed=9)
    assert len(windows) == 10
    assert late["return"]["linear_slope_per_1M_steps"] > 0.0
    assert late["return"]["maximum_single_window_drawdown"] == 0.0


def test_rollout_buffer_team_reward_is_agent_mean_in_red_id_order():
    buffer = RolloutBuffer(horizon=1, num_envs=1)
    rewards = np.asarray([[1.0, 2.0, 3.0, 4.0]], np.float32)
    buffer.insert(
        np.zeros((1, len(RED_IDS), 100), np.float32), np.zeros((1, 117), np.float32),
        np.zeros((1, len(RED_IDS), 3), np.float32), np.zeros((1, len(RED_IDS)), np.float32),
        rewards, np.zeros(1, np.float32), np.zeros(1, bool), np.zeros(1, bool),
        np.ones((1, len(RED_IDS)), np.float32),
    )
    assert buffer.rewards[0, 0] == 2.5
