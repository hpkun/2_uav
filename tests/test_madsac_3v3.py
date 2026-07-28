"""Tests for the homogeneous 3v3 MADSAC baseline."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from uav_combat.madsac.evaluation_3v3 import evaluate_madsac_fixed_blue_3v3
from uav_combat.madsac.metrics import MADSACMetricAccumulator
from uav_combat.madsac.networks import AttentionCritic, SharedSquashedGaussianActor, TwinAttentionCritic
from uav_combat.madsac.replay_buffer import MADSACReplayBuffer
from uav_combat.madsac.trainer_3v3 import (
    CHECKPOINT_FAMILY_MADSAC_3V3,
    MADSAC3v3Trainer,
    masked_mean,
    soft_update_,
)
from scripts.train_madsac_3v3 import load_config, next_strict_milestone


ROOT = Path(__file__).parents[1]
ENV_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
MADSAC_PAPER_CONFIG = ROOT / "configs" / "madsac_3v3_paper.yaml"


def _script_args(**overrides):
    values = {
        "train_config": str(MADSAC_PAPER_CONFIG),
        "smoke": False,
        "total_env_steps": None,
        "num_envs": None,
        "env_workers": None,
        "gradient_steps": None,
        "seed": None,
        "device": None,
        "output_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tiny_config(tmp_path, workers=1, policy_delay=2):
    return {
        "experiment": {"seed": 123, "device": "cpu", "output_dir": str(tmp_path)},
        "network": {
            "actor_hidden_dim": 32,
            "critic_hidden_dim": 32,
            "attention_heads": 2,
            "log_std_bias_init": -0.5,
            "log_std_min": -2.0,
            "log_std_max": 0.0,
        },
        "training": {
            "training_mode": "fixed_rule_blue_3v3_madsac",
            "total_env_steps": 16,
            "num_envs": 2,
            "num_env_workers": workers,
            "replay_capacity": 128,
            "batch_size": 8,
            "learning_starts": 8,
            "gradient_steps": 1,
            "gamma": 0.99,
            "tau": 0.1,
            "alpha": 0.1,
            "policy_delay": policy_delay,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "max_actor_grad_norm": 10.0,
            "max_critic_grad_norm": 10.0,
            "evaluation_interval_env_steps": 100,
            "quick_evaluation_episodes": 2,
            "checkpoint_interval_env_steps": 100,
        },
        "evaluation": {"episodes": 2, "deterministic": True},
    }


def test_madsac_paper_config_defaults_to_16_envs_and_4_workers():
    with MADSAC_PAPER_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["num_envs"] == 16
    assert cfg["training"]["num_env_workers"] == 4
    assert cfg["training"]["gradient_steps"] == 2


def test_madsac_load_config_preserves_16_env_and_two_gradient_steps_without_cli_override():
    cfg = load_config(_script_args())
    assert cfg["training"]["num_envs"] == 16
    assert cfg["training"]["num_env_workers"] == 4
    assert cfg["training"]["gradient_steps"] == 2


def test_madsac_load_config_allows_cli_num_envs_and_gradient_steps_override_to_8_and_1():
    cfg = load_config(_script_args(num_envs=8, gradient_steps=1))
    assert cfg["training"]["num_envs"] == 8
    assert cfg["training"]["num_env_workers"] == 4
    assert cfg["training"]["gradient_steps"] == 1


def test_madsac_load_config_rejects_non_positive_gradient_steps():
    try:
        load_config(_script_args(gradient_steps=0))
    except ValueError as exc:
        assert "--gradient-steps must be positive" in str(exc)
    else:
        raise AssertionError("expected non-positive gradient steps to be rejected")


def test_madsac_evaluation_helper_default_num_envs_is_16():
    sig = inspect.signature(evaluate_madsac_fixed_blue_3v3)
    assert sig.parameters["num_envs"].default == 16
    assert sig.parameters["num_env_workers"].default == 4


def test_actor_shapes_bounds_finiteness_and_single_input():
    actor = SharedSquashedGaussianActor(hidden_dim=32)
    obs = torch.zeros(5, 3, 68)
    actions, log_probs = actor.sample(obs)
    single_actions, single_log_probs = actor.sample(torch.zeros(5, 68))
    assert actions.shape == (5, 3, 3)
    assert log_probs.shape == (5, 3)
    assert single_actions.shape == (5, 3)
    assert single_log_probs.shape == (5,)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)


def test_actor_tanh_jacobian_matches_manual_formula():
    torch.manual_seed(7)
    actor = SharedSquashedGaussianActor(hidden_dim=32)
    obs = torch.randn(4, 3, 68)
    torch.manual_seed(11)
    public_action, public_log_prob = actor.sample(obs)
    torch.manual_seed(11)
    mean, log_std = actor._mean_log_std(obs)
    dist = torch.distributions.Normal(mean, log_std.exp())
    raw = dist.rsample()
    action = torch.tanh(raw)
    manual = (dist.log_prob(raw) - torch.log(1.0 - action.square() + actor.epsilon)).sum(-1)
    assert torch.allclose(public_action, action, atol=1e-6)
    assert torch.allclose(manual, public_log_prob, atol=1e-6)


def test_deterministic_action_equals_tanh_mean():
    actor = SharedSquashedGaussianActor(hidden_dim=32)
    obs = torch.randn(3, 3, 68)
    mean, _ = actor._mean_log_std(obs)
    assert torch.allclose(actor.deterministic(obs), torch.tanh(mean), atol=1e-7)


def test_attention_critic_masks_dead_agents_and_all_dead_is_finite():
    critic = AttentionCritic(hidden_dim=32)
    obs = torch.randn(4, 3, 68)
    act = torch.randn(4, 3, 3).clamp(-1, 1)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=torch.float32)
    q = critic(obs, act, mask)
    assert q.shape == (4, 3)
    assert torch.isfinite(q).all()
    assert torch.all(q[mask == 0] == 0)


def test_dead_agent_observation_action_does_not_affect_alive_q():
    critic = AttentionCritic(hidden_dim=32)
    obs = torch.randn(1, 3, 68)
    act = torch.randn(1, 3, 3).clamp(-1, 1)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    q1 = critic(obs, act, mask)
    obs2, act2 = obs.clone(), act.clone()
    obs2[:, 1, :] = 999.0
    act2[:, 1, :] = -999.0
    q2 = critic(obs2, act2, mask)
    assert torch.allclose(q1[:, [0, 2]], q2[:, [0, 2]], atol=1e-6)


def test_attention_critic_excludes_self_token_from_other_agent_attention():
    critic = AttentionCritic(hidden_dim=32, attention_heads=2)
    obs = torch.randn(1, 3, 68)
    act = torch.randn(1, 3, 3).clamp(-1, 1)
    single_alive = torch.tensor([[1.0, 0.0, 0.0]])
    all_dead = torch.zeros(1, 3)
    q_single = critic(obs, act, single_alive)
    q_all_dead = critic(obs, act, all_dead)
    obs_dead_changed = obs.clone()
    act_dead_changed = act.clone()
    obs_dead_changed[:, 1:, :] = 999.0
    act_dead_changed[:, 1:, :] = -999.0
    q_changed = critic(obs_dead_changed, act_dead_changed, single_alive)
    assert torch.isfinite(q_single).all()
    assert torch.isfinite(q_all_dead).all()
    assert torch.all(q_all_dead == 0.0)
    assert torch.allclose(q_single[:, 0], q_changed[:, 0], atol=1e-6)


def test_attention_critic_alive_teammate_can_affect_other_alive_agent_q():
    critic = AttentionCritic(hidden_dim=32, attention_heads=2)
    obs = torch.randn(1, 3, 68)
    act = torch.randn(1, 3, 3).clamp(-1, 1)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    q1 = critic(obs, act, mask)
    obs2, act2 = obs.clone(), act.clone()
    obs2[:, 1, :] += 10.0
    act2[:, 1, :] = (act2[:, 1, :] + 0.5).clamp(-1, 1)
    q2 = critic(obs2, act2, mask)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert not torch.allclose(q1[:, 0], q2[:, 0])


def test_q1_q2_parameters_are_independent():
    twin = TwinAttentionCritic(hidden_dim=32)
    assert next(twin.q1.parameters()) is not next(twin.q2.parameters())
    q1, q2 = twin(torch.zeros(2, 3, 68), torch.zeros(2, 3, 3), torch.ones(2, 3))
    assert q1.shape == q2.shape == (2, 3)


def test_target_network_initialization_requires_grad_and_soft_update(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        for p, tp in zip(trainer.actor.parameters(), trainer.target_actor.parameters()):
            assert torch.allclose(p, tp)
            assert not tp.requires_grad
        old = [p.detach().clone() for p in trainer.target_actor.parameters()]
        with torch.no_grad():
            for p in trainer.actor.parameters():
                p.add_(1.0)
        soft_update_(trainer.target_actor, trainer.actor, 0.25)
        for before, online, target in zip(old, trainer.actor.parameters(), trainer.target_actor.parameters()):
            assert torch.allclose(target, 0.25 * online + 0.75 * before)
        for p, tp in zip(trainer.critic.q1.parameters(), trainer.target_critic.q1.parameters()):
            assert torch.allclose(p, tp)
            assert not tp.requires_grad
        for p, tp in zip(trainer.critic.q2.parameters(), trainer.target_critic.q2.parameters()):
            assert torch.allclose(p, tp)
            assert not tp.requires_grad
        old_q1 = [p.detach().clone() for p in trainer.target_critic.q1.parameters()]
        old_q2 = [p.detach().clone() for p in trainer.target_critic.q2.parameters()]
        with torch.no_grad():
            for p in trainer.critic.q1.parameters():
                p.add_(2.0)
            for p in trainer.critic.q2.parameters():
                p.sub_(2.0)
        soft_update_(trainer.target_critic, trainer.critic, 0.5)
        for before, online, target in zip(old_q1, trainer.critic.q1.parameters(), trainer.target_critic.q1.parameters()):
            assert torch.allclose(target, 0.5 * online + 0.5 * before)
        for before, online, target in zip(old_q2, trainer.critic.q2.parameters(), trainer.target_critic.q2.parameters()):
            assert torch.allclose(target, 0.5 * online + 0.5 * before)
    finally:
        trainer.close()


def test_replay_batch_add_wrap_sample_dtype_and_device():
    buf = MADSACReplayBuffer(capacity=10)
    rng = np.random.default_rng(3)
    for _ in range(2):
        n = 8
        buf.add_batch(
            np.ones((n, 3, 68), np.float32),
            np.zeros((n, 3, 3), np.float32),
            np.ones(n, np.float32),
            np.ones((n, 3, 68), np.float32) * 2,
            np.ones((n, 3), np.float32),
            np.ones((n, 3), np.float32),
            np.zeros(n, bool),
            np.ones(n, bool),
        )
    assert buf.size == 10
    assert buf.position == 6
    sample = buf.sample(4, rng, torch.device("cpu"))
    assert sample["observations"].shape == (4, 3, 68)
    assert sample["terminated"].dtype == torch.bool
    assert sample["truncated"].dtype == torch.bool
    assert sample["done_for_bootstrap"].dtype == torch.bool
    assert sample["observations"].dtype == torch.float32


def test_replay_done_for_bootstrap_is_terminated_or_truncated():
    buf = MADSACReplayBuffer(capacity=2)
    buf.add_batch(
        np.zeros((2, 3, 68), np.float32),
        np.zeros((2, 3, 3), np.float32),
        np.array([4.0, -2.0], np.float32),
        np.zeros((2, 3, 68), np.float32),
        np.ones((2, 3), np.float32),
        np.ones((2, 3), np.float32),
        np.array([True, False]),
        np.array([False, True]),
    )
    assert np.array_equal(np.logical_or(buf.terminated[:2], buf.truncated[:2]), np.array([True, True]))


def test_td_target_masks_done_and_next_dead_agents(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        batch = {
            "next_observations": torch.zeros(2, 3, 68),
            "next_alive_masks": torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.float32),
            "team_rewards": torch.tensor([2.0, 3.0]),
            "done_for_bootstrap": torch.tensor([False, True]),
        }
        y = trainer.compute_td_target(batch)
        assert y.shape == (2, 3)
        assert torch.allclose(y[0, 1], torch.tensor(2.0))
        assert torch.allclose(y[1], torch.full((3,), 3.0))
    finally:
        trainer.close()


def test_td_target_uses_minimum_of_twin_target_q_values(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        class FakeTargetActor:
            def sample(self, observations):
                return torch.zeros(observations.shape[:-1] + (3,)), torch.zeros(observations.shape[:-1])

        class FakeTargetCritic:
            def __call__(self, observations, actions, alive_masks):
                q1 = torch.tensor([[10.0, -5.0, 7.0]], dtype=torch.float32)
                q2 = torch.tensor([[1.0, 6.0, -3.0]], dtype=torch.float32)
                return q1, q2

        trainer.target_actor = FakeTargetActor()
        trainer.target_critic = FakeTargetCritic()
        batch = {
            "next_observations": torch.zeros(1, 3, 68),
            "next_alive_masks": torch.ones(1, 3),
            "team_rewards": torch.tensor([2.0]),
            "done_for_bootstrap": torch.tensor([False]),
        }
        y = trainer.compute_td_target(batch)
        expected = torch.tensor([[2.0 + 0.99 * 1.0, 2.0 + 0.99 * -5.0, 2.0 + 0.99 * -3.0]])
        assert torch.allclose(y, expected)
    finally:
        trainer.close()


def test_td_target_terminated_and_truncated_each_disable_bootstrap(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        class FakeTargetActor:
            def sample(self, observations):
                return torch.zeros(observations.shape[:-1] + (3,)), torch.zeros(observations.shape[:-1])

        class FakeTargetCritic:
            def __call__(self, observations, actions, alive_masks):
                return torch.full((2, 3), 100.0), torch.full((2, 3), 50.0)

        trainer.target_actor = FakeTargetActor()
        trainer.target_critic = FakeTargetCritic()
        batch = {
            "next_observations": torch.zeros(2, 3, 68),
            "next_alive_masks": torch.ones(2, 3),
            "team_rewards": torch.tensor([4.0, -2.0]),
            "done_for_bootstrap": torch.tensor([True, True]),
        }
        y = trainer.compute_td_target(batch)
        assert torch.allclose(y[0], torch.full((3,), 4.0))
        assert torch.allclose(y[1], torch.full((3,), -2.0))
    finally:
        trainer.close()


def test_dead_current_agents_are_excluded_from_masked_losses():
    values = torch.tensor([[1.0, 999.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    assert torch.allclose(masked_mean(values, mask), torch.tensor(2.0))
    assert masked_mean(values, torch.zeros_like(mask)) is None


def test_policy_delay_and_cpu_update_change_networks(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path, policy_delay=2))
    try:
        actor_before = [p.detach().clone() for p in trainer.actor.parameters()]
        c1_before = [p.detach().clone() for p in trainer.critic.q1.parameters()]
        c2_before = [p.detach().clone() for p in trainer.critic.q2.parameters()]
        target_before = [p.detach().clone() for p in trainer.target_actor.parameters()]
        trainer.train_until(14)
        assert trainer.critic_update_count > 0
        assert trainer.actor_update_count == trainer.critic_update_count // 2
        assert any(not torch.allclose(a, b) for a, b in zip(actor_before, trainer.actor.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(c1_before, trainer.critic.q1.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(c2_before, trainer.critic.q2.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(target_before, trainer.target_actor.parameters()))
        assert trainer.actor_update_count > 0
        for key in ("actor_loss_mean", "critic1_loss_mean", "critic2_loss_mean", "q1_mean", "target_q_mean"):
            assert np.isfinite(float(trainer.last_metrics[key]))
    finally:
        trainer.close()


def test_actor_update_does_not_change_critic_parameters_when_only_actor_optimizer_steps(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path, policy_delay=1))
    try:
        for _ in range(8):
            trainer.step_environment()
        batch = trainer.replay.sample(8, trainer.rng, trainer.device)
        alive = batch["alive_masks"]
        before = [p.detach().clone() for p in trainer.critic.parameters()]
        from uav_combat.madsac.trainer_3v3 import set_requires_grad_
        set_requires_grad_(trainer.critic, False)
        actions_pi, log_probs_pi = trainer.actor.sample(batch["observations"])
        actions_pi = actions_pi * alive.unsqueeze(-1)
        q1_pi, q2_pi = trainer.critic(batch["observations"], actions_pi, alive)
        loss = masked_mean(trainer.alpha * log_probs_pi * alive - torch.minimum(q1_pi, q2_pi), alive)
        trainer.actor_optimizer.zero_grad()
        loss.backward()
        trainer.actor_optimizer.step()
        set_requires_grad_(trainer.critic, True)
        after = list(trainer.critic.parameters())
        assert all(torch.allclose(a, b) for a, b in zip(before, after))
    finally:
        trainer.close()


def test_local_and_worker_trainers_can_be_constructed(tmp_path):
    local = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path / "l", workers=1))
    worker = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path / "w", workers=2))
    try:
        assert local.rule_policy_mapping_modes["blue"] == ["rate_aligned_v1", "rate_aligned_v1"]
        assert worker.rule_policy_mapping_modes["blue"] == ["rate_aligned_v1", "rate_aligned_v1"]
    finally:
        local.close()
        worker.close()


def test_checkpoint_roundtrip_preserves_deterministic_actor_output(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    ckpt = tmp_path / "madsac.pt"
    probe = torch.randn(2, 3, 68)
    try:
        trainer.train_until(16)
        with torch.no_grad():
            before = trainer.actor.deterministic(probe)
        trainer.save_checkpoint(ckpt)
    finally:
        trainer.close()
    restored = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        restored.load_checkpoint(ckpt)
        with torch.no_grad():
            after = restored.actor.deterministic(probe)
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert loaded["checkpoint_family"] == CHECKPOINT_FAMILY_MADSAC_3V3
        assert loaded["replay_metadata"]["full_replay_persisted"] is False
        assert loaded["replay_restored"] is False
        assert torch.allclose(before, after, atol=1e-6)
    finally:
        restored.close()


def test_madsac_metric_accumulator_keeps_actor_updates_separate():
    acc = MADSACMetricAccumulator()
    rows = [
        {"critic_updates_in_call": 1, "actor_updates_in_call": 0, "target_updates_in_call": 0,
         "critic1_loss_mean": 1.0, "critic1_loss_max": 1.0, "critic2_loss_mean": 2.0, "critic2_loss_max": 2.0,
         "q1_mean": 1.0, "q2_mean": 2.0, "target_q_mean": 3.0, "q1_q2_abs_gap_mean": 0.5,
         "q1_q2_abs_gap_max": 0.5, "td_error_abs_mean": 0.7, "td_error_abs_max": 0.7,
         "critic1_grad_norm_pre_clip_mean": 4.0, "critic1_grad_norm_pre_clip_max": 4.0,
         "critic2_grad_norm_pre_clip_mean": 5.0, "critic2_grad_norm_pre_clip_max": 5.0,
         "critic1_grad_clipped_fraction": 0.0, "critic2_grad_clipped_fraction": 1.0,
         "actor_loss_mean": 0.0, "actor_loss_last": None},
        {"critic_updates_in_call": 1, "actor_updates_in_call": 1, "target_updates_in_call": 1,
         "critic1_loss_mean": 3.0, "critic1_loss_max": 3.0, "critic2_loss_mean": 4.0, "critic2_loss_max": 4.0,
         "q1_mean": 3.0, "q2_mean": 4.0, "target_q_mean": 5.0, "q1_q2_abs_gap_mean": 1.5,
         "q1_q2_abs_gap_max": 1.5, "td_error_abs_mean": 1.7, "td_error_abs_max": 1.7,
         "critic1_grad_norm_pre_clip_mean": 6.0, "critic1_grad_norm_pre_clip_max": 6.0,
         "critic2_grad_norm_pre_clip_mean": 7.0, "critic2_grad_norm_pre_clip_max": 7.0,
         "critic1_grad_clipped_fraction": 1.0, "critic2_grad_clipped_fraction": 0.0,
         "actor_loss_mean": 10.0, "actor_loss_last": 10.0, "sampled_log_prob_mean": -2.0,
         "deterministic_action_abs_mean": 0.2, "stochastic_action_abs_mean": 0.3,
         "action_saturation_fraction_mean": 0.1, "action_saturation_fraction_max": 0.1,
         "actor_grad_norm_pre_clip_mean": 8.0, "actor_grad_norm_pre_clip_max": 8.0,
         "actor_grad_clipped_fraction": 0.0},
        {"critic_updates_in_call": 1, "actor_updates_in_call": 0, "target_updates_in_call": 0,
         "critic1_loss_mean": 5.0, "critic1_loss_max": 5.0, "critic2_loss_mean": 6.0, "critic2_loss_max": 6.0,
         "q1_mean": 5.0, "q2_mean": 6.0, "target_q_mean": 7.0, "q1_q2_abs_gap_mean": 2.5,
         "q1_q2_abs_gap_max": 2.5, "td_error_abs_mean": 2.7, "td_error_abs_max": 2.7,
         "critic1_grad_norm_pre_clip_mean": 8.0, "critic1_grad_norm_pre_clip_max": 8.0,
         "critic2_grad_norm_pre_clip_mean": 9.0, "critic2_grad_norm_pre_clip_max": 9.0,
         "critic1_grad_clipped_fraction": 0.0, "critic2_grad_clipped_fraction": 0.0,
         "actor_loss_mean": 0.0, "actor_loss_last": None},
        {"critic_updates_in_call": 1, "actor_updates_in_call": 1, "target_updates_in_call": 1,
         "critic1_loss_mean": 7.0, "critic1_loss_max": 7.0, "critic2_loss_mean": 8.0, "critic2_loss_max": 8.0,
         "q1_mean": 7.0, "q2_mean": 8.0, "target_q_mean": 9.0, "q1_q2_abs_gap_mean": 3.5,
         "q1_q2_abs_gap_max": 3.5, "td_error_abs_mean": 3.7, "td_error_abs_max": 3.7,
         "critic1_grad_norm_pre_clip_mean": 10.0, "critic1_grad_norm_pre_clip_max": 10.0,
         "critic2_grad_norm_pre_clip_mean": 11.0, "critic2_grad_norm_pre_clip_max": 11.0,
         "critic1_grad_clipped_fraction": 1.0, "critic2_grad_clipped_fraction": 1.0,
         "actor_loss_mean": 14.0, "actor_loss_last": 14.0, "sampled_log_prob_mean": -4.0,
         "deterministic_action_abs_mean": 0.4, "stochastic_action_abs_mean": 0.5,
         "action_saturation_fraction_mean": 0.6, "action_saturation_fraction_max": 0.6,
         "actor_grad_norm_pre_clip_mean": 12.0, "actor_grad_norm_pre_clip_max": 12.0,
         "actor_grad_clipped_fraction": 1.0},
    ]
    for row in rows:
        acc.add(row)
    summary = acc.summarize()
    assert summary["update_calls_in_interval"] == 4
    assert summary["critic_updates_in_interval"] == 4
    assert summary["actor_updates_in_interval"] == 2
    assert summary["target_updates_in_interval"] == 2
    assert summary["critic1_loss_mean"] == 4.0
    assert summary["actor_loss_mean"] == 12.0
    assert summary["actor_loss_last"] == 14.0
    assert summary["action_saturation_fraction_max"] == 0.6
    assert summary["actor_grad_norm_pre_clip_max"] == 12.0


def test_resume_next_strict_milestone():
    assert next_strict_milestone(0, 100000) == 100000
    assert next_strict_milestone(100000, 100000) == 200000
    assert next_strict_milestone(550000, 100000) == 600000


def test_checkpoint_signature_mismatch_reports_specific_fields(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    ckpt = tmp_path / "madsac.pt"
    try:
        trainer.save_checkpoint(ckpt)
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        data["training_signature"]["hyperparameters"]["actor_learning_rate"] = 1e-9
        data["training_signature"]["env_config_sha256"] = "bad-sha"
        torch.save(data, ckpt)
        try:
            trainer.load_checkpoint(ckpt)
        except RuntimeError as exc:
            msg = str(exc)
            assert "actor_learning_rate" in msg
            assert "env_config_sha256" in msg
        else:
            raise AssertionError("expected checkpoint signature mismatch")
    finally:
        trainer.close()


def test_best_score_prioritizes_real_red_attack_kills():
    from uav_combat.mappo.trainer_3v3 import compute_best_score
    no_kills = {"red_complete_elimination_success_rate": 0.0, "mean_red_attack_kills": 0.0, "mean_blue_survivors": 0.0}
    real_kills = {"red_complete_elimination_success_rate": 0.0, "mean_red_attack_kills": 1.0, "mean_blue_survivors": 3.0}
    assert compute_best_score(real_kills) > compute_best_score(no_kills)


def test_evaluation_generates_required_fields(tmp_path):
    trainer = MADSAC3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        result = evaluate_madsac_fixed_blue_3v3(trainer.actor, ENV_V4, episodes=2, num_envs=2, num_env_workers=1, device=trainer.device)
        for key in (
            "episodes", "red_complete_elimination_success_rate", "blue_complete_elimination_success_rate",
            "environment_red_outcome_rate", "environment_blue_outcome_rate", "draw_rate",
            "mean_red_attack_kills", "mean_blue_attack_kills", "red_kd_numerator",
            "mean_red_boundary_altitude_deaths", "mean_red_cross_collision_deaths",
            "evaluation_seconds", "environment_steps_per_second",
        ):
            assert key in result
    finally:
        trainer.close()
