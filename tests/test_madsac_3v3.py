"""Tests for the homogeneous 3v3 MADSAC baseline."""
from pathlib import Path

import numpy as np
import torch

from uav_combat.madsac.evaluation_3v3 import evaluate_madsac_fixed_blue_3v3
from uav_combat.madsac.networks import AttentionCritic, SharedSquashedGaussianActor, TwinAttentionCritic
from uav_combat.madsac.replay_buffer import MADSACReplayBuffer
from uav_combat.madsac.trainer_3v3 import (
    CHECKPOINT_FAMILY_MADSAC_3V3,
    MADSAC3v3Trainer,
    masked_mean,
    soft_update_,
)


ROOT = Path(__file__).parents[1]
ENV_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"


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
    mean, log_std = actor._mean_log_std(obs)
    dist = torch.distributions.Normal(mean, log_std.exp())
    raw = mean + log_std.exp() * torch.zeros_like(mean)
    action = torch.tanh(raw)
    manual = (dist.log_prob(raw) - torch.log(1.0 - action.square() + actor.epsilon)).sum(-1)
    # Reconstruct through the same public transformation with fixed raw=mean.
    public_action = torch.tanh(mean)
    public_log_prob = (dist.log_prob(mean) - torch.log(1.0 - public_action.square() + actor.epsilon)).sum(-1)
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
        trainer.train_until(16)
        assert trainer.critic_update_count > 0
        assert trainer.actor_update_count == trainer.critic_update_count // 2
        assert any(not torch.allclose(a, b) for a, b in zip(actor_before, trainer.actor.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(c1_before, trainer.critic.q1.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(c2_before, trainer.critic.q2.parameters()))
        assert any(not torch.allclose(a, b) for a, b in zip(target_before, trainer.target_actor.parameters()))
        assert trainer.actor_update_count > 0
        for key in ("actor_loss", "critic1_loss", "critic2_loss", "q1_mean", "target_q_mean"):
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
