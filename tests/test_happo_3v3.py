"""Tests for the homogeneous 3v3 HAPPO baseline."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from scripts.train_happo_3v3 import load_config
from uav_combat.happo.buffer_3v3 import HAPPORolloutBuffer3v3
from uav_combat.happo.networks import CentralizedValueCritic, IndependentHAPPOActors
from uav_combat.happo.trainer_3v3 import (
    CHECKPOINT_FAMILY_HAPPO_3V3,
    HAPPO3v3Trainer,
    happo_preceding_factor_update,
    normalize_advantages_for_agent,
    ppo_clipped_policy_loss,
    validate_episode_accounting_3v3,
)

ROOT = Path(__file__).parents[1]
ENV_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
HAPPO_CONFIG = ROOT / "configs" / "happo_3v3_fixed_blue.yaml"


def _args(**overrides):
    values = {
        "train_config": str(HAPPO_CONFIG),
        "smoke": False,
        "total_env_steps": None,
        "num_envs": None,
        "env_workers": None,
        "seed": None,
        "device": None,
        "output_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tiny_config(tmp_path, workers=1):
    return {
        "experiment": {"seed": 7, "device": "cpu", "output_dir": str(tmp_path)},
        "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
        "training": {
            "training_mode": "fixed_rule_blue_3v3_happo",
            "total_env_steps": 16,
            "num_envs": 2,
            "num_env_workers": workers,
            "rollout_steps": 4,
            "team_size": 3,
            "observation_dims": [68, 68, 68],
            "action_dims": [3, 3, 3],
            "ppo_epochs": 1,
            "minibatch_size": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "actor_learning_rate": 3e-4,
            "actor_learning_rate_final": 3e-5,
            "critic_learning_rate": 3e-4,
            "critic_learning_rate_final": 3e-5,
            "entropy_coef": 0.01,
            "entropy_coef_final": 0.001,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "evaluation_interval_env_steps": 100,
            "quick_evaluation_episodes": 2,
            "checkpoint_interval_env_steps": 100,
        },
        "evaluation": {"episodes": 2, "deterministic": True},
    }


def test_happo_config_uses_fixed_blue_mode_and_mappo_aligned_defaults():
    with HAPPO_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["training_mode"] == "fixed_rule_blue_3v3_happo"
    assert cfg["training"]["rollout_steps"] == 256
    assert cfg["training"]["gamma"] == 0.99
    assert cfg["training"]["gae_lambda"] == 0.95
    assert cfg["training"]["clip_coef"] == 0.2
    assert cfg["training"]["observation_dims"] == [68, 68, 68]
    assert cfg["training"]["action_dims"] == [3, 3, 3]


def test_happo_load_config_cli_overrides_without_creating_envs():
    cfg = load_config(_args(num_envs=2, env_workers=1, total_env_steps=1024, device="cpu"))
    assert cfg["training"]["num_envs"] == 2
    assert cfg["training"]["num_env_workers"] == 1
    assert cfg["training"]["total_env_steps"] == 1024
    assert cfg["experiment"]["device"] == "cpu"


def test_independent_actors_shapes_bounds_log_probs_and_no_shared_parameters():
    actors = IndependentHAPPOActors([68, 68, 68], [3, 3, 3], hidden_dim=32)
    obs = torch.zeros(5, 3, 68)
    actions, log_probs = actors.sample_actions(obs)
    assert actions.shape == (5, 3, 3)
    assert log_probs.shape == (5, 3)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)
    parameter_ids = [{id(p) for p in actor.parameters()} for actor in actors.actors]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    before = [p.detach().clone() for p in actors.actors[1].parameters()]
    with torch.no_grad():
        for p in actors.actors[0].parameters():
            p.add_(1.0)
    assert all(torch.allclose(a, b) for a, b in zip(before, actors.actors[1].parameters()))


def test_centralized_value_critic_shape_and_finiteness():
    critic = CentralizedValueCritic(state_dim=48, hidden_dim=32)
    value = critic(torch.zeros(6, 48))
    assert value.shape == (6,)
    assert torch.isfinite(value).all()


def test_happo_buffer_gae_and_dead_agent_masks_are_stored():
    buf = HAPPORolloutBuffer3v3(rollout_steps=3, num_envs=2)
    for step in range(3):
        buf.add(
            np.zeros((2, 3, 68), np.float32),
            np.zeros((2, 48), np.float32),
            np.zeros((2, 3, 3), np.float32),
            np.zeros((2, 3), np.float32),
            np.array([[1, 1, 1], [1, 0, 1]], np.float32),
            np.ones(2, np.float32),
            np.zeros(2, np.float32),
            np.array([False, step == 2]),
        )
    buf.compute_returns_and_advantages(np.zeros(2, np.float32), 0.99, 0.95)
    assert buf.agent_alive_masks[0, 1, 1] == 0.0
    assert np.isfinite(buf.advantages).all()
    assert np.isfinite(buf.returns).all()
    assert buf.returns.shape == (3, 2)


def test_happo_factor_update_uses_new_old_ratio_dead_samples_one_and_detaches():
    factor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    old = torch.log(torch.tensor([0.5, 0.5, 0.5]))
    new = torch.log(torch.tensor([1.0, 0.25, 0.75]))
    active = torch.tensor([1.0, 0.0, 1.0])
    updated = happo_preceding_factor_update(factor, old, new, active)
    expected = torch.tensor([2.0, 2.0, 4.5])
    assert torch.allclose(updated, expected)
    assert updated.requires_grad is False


def test_per_agent_advantage_normalization_uses_each_agent_active_mask():
    advantages = torch.tensor([1.0, 2.0, 10.0, 20.0])
    masks = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    norm0 = normalize_advantages_for_agent(advantages, masks[:, 0])
    norm1 = normalize_advantages_for_agent(advantages, masks[:, 1])
    norm2 = normalize_advantages_for_agent(advantages, masks[:, 2])
    expected0 = (advantages - torch.tensor(1.0)) / 1e-8
    expected1 = (advantages - torch.tensor([1.0, 2.0, 20.0]).mean()) / (
        torch.tensor([1.0, 2.0, 20.0]).std(unbiased=False) + 1e-8
    )
    expected2 = (advantages - torch.tensor([1.0, 2.0, 10.0]).mean()) / (
        torch.tensor([1.0, 2.0, 10.0]).std(unbiased=False) + 1e-8
    )
    assert torch.allclose(norm0, expected0)
    assert torch.allclose(norm1, expected1)
    assert torch.allclose(norm2, expected2)
    assert not torch.allclose(norm1, norm2)


def test_per_agent_advantage_normalization_no_active_samples_is_finite_passthrough():
    advantages = torch.tensor([1.0, -2.0, 3.0])
    out = normalize_advantages_for_agent(advantages, torch.zeros(3))
    assert torch.allclose(out, advantages)
    assert torch.isfinite(out).all()


def test_happo_full_sequential_factor_propagation_with_agent_masks():
    advantages = torch.tensor([1.0, -1.0, 2.0])
    active = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    old = torch.zeros(3, 3)
    new = torch.log(torch.tensor([
        [2.0, 3.0, 5.0],
        [4.0, 7.0, 11.0],
        [13.0, 17.0, 19.0],
    ]))
    factor = torch.ones(3)
    used_effective_advantages = []
    for agent_id in [0, 1, 2]:
        norm_i = normalize_advantages_for_agent(advantages, active[:, agent_id])
        used_effective_advantages.append((factor * norm_i).detach())
        factor = happo_preceding_factor_update(factor, old[:, agent_id], new[:, agent_id], active[:, agent_id])
    norm0 = normalize_advantages_for_agent(advantages, active[:, 0])
    norm1 = normalize_advantages_for_agent(advantages, active[:, 1])
    norm2 = normalize_advantages_for_agent(advantages, active[:, 2])
    assert torch.allclose(used_effective_advantages[0], norm0)
    assert torch.allclose(used_effective_advantages[1], torch.tensor([2.0, 4.0, 1.0]) * norm1)
    assert torch.allclose(used_effective_advantages[2], torch.tensor([6.0, 4.0, 17.0]) * norm2)
    assert factor.requires_grad is False
    reverse_factor = torch.ones(3)
    reverse_used = []
    for agent_id in [2, 1, 0]:
        norm_i = normalize_advantages_for_agent(advantages, active[:, agent_id])
        reverse_used.append((reverse_factor * norm_i).detach())
        reverse_factor = happo_preceding_factor_update(reverse_factor, old[:, agent_id], new[:, agent_id], active[:, agent_id])
    assert not torch.allclose(used_effective_advantages[1], reverse_used[1])


def test_ppo_clipped_policy_loss_positive_and_negative_advantages():
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.9])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = ppo_clipped_policy_loss(ratio, adv, 0.2)
    manual = -torch.minimum(
        ratio * adv,
        ratio.clamp(0.8, 1.2) * adv,
    ).mean()
    assert torch.allclose(loss, manual)


def test_actor_update_isolation_for_single_agent_phase(tmp_path):
    trainer = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        obs = torch.randn(4, 68)
        actions, _ = trainer.actors.actors[0].sample_action(obs)
        old_log_prob = torch.zeros(4)
        advantage = torch.ones(4)
        before_actor0 = [p.detach().clone() for p in trainer.actors.actors[0].parameters()]
        before_actor1 = [p.detach().clone() for p in trainer.actors.actors[1].parameters()]
        before_actor2 = [p.detach().clone() for p in trainer.actors.actors[2].parameters()]
        before_critic = [p.detach().clone() for p in trainer.critic.parameters()]
        before_opt1 = trainer.actor_optimizers[1].state_dict()
        before_opt2 = trainer.actor_optimizers[2].state_dict()
        log_prob, entropy = trainer.actors.actors[0].evaluate_actions(obs, actions.detach())
        ratio = torch.exp(log_prob - old_log_prob)
        loss = ppo_clipped_policy_loss(ratio, advantage, 0.2) - 0.01 * entropy.mean()
        trainer.actor_optimizers[0].zero_grad()
        loss.backward()
        trainer.actor_optimizers[0].step()
        assert any(not torch.allclose(a, b) for a, b in zip(before_actor0, trainer.actors.actors[0].parameters()))
        assert all(torch.allclose(a, b) for a, b in zip(before_actor1, trainer.actors.actors[1].parameters()))
        assert all(torch.allclose(a, b) for a, b in zip(before_actor2, trainer.actors.actors[2].parameters()))
        assert all(torch.allclose(a, b) for a, b in zip(before_critic, trainer.critic.parameters()))
        assert before_opt1 == trainer.actor_optimizers[1].state_dict()
        assert before_opt2 == trainer.actor_optimizers[2].state_dict()
    finally:
        trainer.close()


def test_happo_trainer_constructs_independent_optimizers_and_signature(tmp_path):
    trainer = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        assert len(trainer.actor_optimizers) == 3
        assert trainer.actor_optimizers[0] is not trainer.actor_optimizers[1]
        sig = trainer.training_signature()
        assert sig["checkpoint_family"] == CHECKPOINT_FAMILY_HAPPO_3V3
        assert sig["observation_dims"] == [68, 68, 68]
        assert sig["action_dims"] == [3, 3, 3]
        assert "env_config_sha256" in sig
    finally:
        trainer.close()


def test_happo_actor_updates_are_minibatch_steps_and_agents_updated_counts_unique_agents(tmp_path):
    trainer = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    try:
        rows = [
            {"agent_id": 0, "policy_loss": 1.0, "entropy": 0.1, "approx_kl": 0.01,
             "clip_fraction": 0.0, "ratio_mean": 1.0, "ratio_min": 0.9, "ratio_max": 1.1,
             "factor_mean": 1.0, "factor_min": 1.0, "factor_max": 1.0,
             "actor_grad_norm": 0.5, "active_samples": 4},
            {"agent_id": 0, "policy_loss": 2.0, "entropy": 0.2, "approx_kl": 0.02,
             "clip_fraction": 0.1, "ratio_mean": 1.1, "ratio_min": 0.8, "ratio_max": 1.2,
             "factor_mean": 1.0, "factor_min": 1.0, "factor_max": 1.0,
             "actor_grad_norm": 0.6, "active_samples": 4},
            {"agent_id": 1, "active_samples": 0},
            {"agent_id": 2, "policy_loss": 3.0, "entropy": 0.3, "approx_kl": 0.03,
             "clip_fraction": 0.2, "ratio_mean": 1.2, "ratio_min": 0.7, "ratio_max": 1.3,
             "factor_mean": 2.0, "factor_min": 2.0, "factor_max": 2.0,
             "actor_grad_norm": 0.7, "active_samples": 4},
        ]
        metrics = trainer._summarize_update(rows, [0.5], np.zeros(8), np.ones(8))
        assert metrics["actor_updates"] == 3
        assert metrics["agents_updated"] == 2
    finally:
        trainer.close()


def test_happo_checkpoint_signature_mismatch_reports_field(tmp_path):
    trainer = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path))
    ckpt = tmp_path / "happo.pt"
    try:
        trainer.save_checkpoint(ckpt)
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        data["training_signature"]["training"]["num_envs"] = 99
        torch.save(data, ckpt)
        try:
            trainer.load_checkpoint(ckpt)
        except RuntimeError as exc:
            assert "num_envs" in str(exc)
        else:
            raise AssertionError("expected signature mismatch")
    finally:
        trainer.close()


def _nested_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_nested_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_nested_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def _make_optimizer_state_nonempty(trainer):
    for agent_id, actor in enumerate(trainer.actors.actors):
        obs = torch.full((3, 68), float(agent_id + 1))
        action, _ = actor.sample_action(obs)
        log_prob, entropy = actor.evaluate_actions(obs, action.detach())
        loss = -(log_prob + 0.01 * entropy).mean()
        trainer.actor_optimizers[agent_id].zero_grad()
        loss.backward()
        trainer.actor_optimizers[agent_id].step()
    states = torch.arange(96, dtype=torch.float32).reshape(2, 48) / 100.0
    values = trainer.critic(states)
    critic_loss = values.square().mean()
    trainer.critic_optimizer.zero_grad()
    critic_loss.backward()
    trainer.critic_optimizer.step()


def test_happo_checkpoint_full_roundtrip_preserves_actors_critic_optimizers_and_rng(tmp_path):
    trainer = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path / "a"))
    restored = None
    ckpt = tmp_path / "happo_roundtrip.pt"
    probe_obs = torch.randn(4, 3, 68)
    probe_state = torch.randn(4, 48)
    try:
        with torch.no_grad():
            for agent_id, actor in enumerate(trainer.actors.actors):
                for param in actor.parameters():
                    param.add_(0.01 * (agent_id + 1))
            for param in trainer.critic.parameters():
                param.sub_(0.02)
        _make_optimizer_state_nonempty(trainer)
        trainer.env_steps = 128
        trainer.vector_steps = 16
        trainer.update_count = 3
        trainer.last_agent_order = [2, 0, 1]
        trainer.best_score = (1.0, 2.0, -3.0)
        trainer.best_evaluation = {"episodes": 2, "mean_red_attack_kills": 1.5}
        trainer.best_checkpoint_name = "best.pt"
        trainer.evaluation_history = [{"env_steps": 64, "score": [1.0]}]
        with torch.no_grad():
            actor_actions = trainer.actors.deterministic_actions(probe_obs)
            critic_values = trainer.critic(probe_state)
        actor_states = [actor.state_dict() for actor in trainer.actors.actors]
        actor_opt_states = [opt.state_dict() for opt in trainer.actor_optimizers]
        critic_state = trainer.critic.state_dict()
        critic_opt_state = trainer.critic_optimizer.state_dict()
        numpy_state = trainer.rng.bit_generator.state
        torch_state = torch.get_rng_state()
        trainer.save_checkpoint(ckpt)
    finally:
        trainer.close()

    restored = HAPPO3v3Trainer(ENV_V4, _tiny_config(tmp_path / "b"))
    try:
        restored.load_checkpoint(ckpt)
        with torch.no_grad():
            restored_actions = restored.actors.deterministic_actions(probe_obs)
            restored_values = restored.critic(probe_state)
        assert torch.allclose(actor_actions, restored_actions, atol=1e-7)
        assert torch.allclose(critic_values, restored_values, atol=1e-7)
        for expected, actor in zip(actor_states, restored.actors.actors):
            for key, value in expected.items():
                assert torch.equal(value, actor.state_dict()[key])
        for key, value in critic_state.items():
            assert torch.equal(value, restored.critic.state_dict()[key])
        for expected, opt in zip(actor_opt_states, restored.actor_optimizers):
            assert _nested_state_equal(expected, opt.state_dict())
        assert _nested_state_equal(critic_opt_state, restored.critic_optimizer.state_dict())
        assert restored.env_steps == 128
        assert restored.vector_steps == 16
        assert restored.update_count == 3
        assert restored.last_agent_order == [2, 0, 1]
        assert restored.best_score == (1.0, 2.0, -3.0)
        assert restored.best_evaluation == {"episodes": 2, "mean_red_attack_kills": 1.5}
        assert restored.best_checkpoint_name == "best.pt"
        assert restored.evaluation_history == [{"env_steps": 64, "score": [1.0]}]
        assert _nested_state_equal(numpy_state, restored.rng.bit_generator.state)
        assert torch.equal(torch_state, torch.get_rng_state())
    finally:
        restored.close()


def test_happo_episode_accounting_validation_accepts_consistent_record():
    rec = {
        "red_survivors": 1, "red_attack_deaths": 1, "red_boundary_deaths": 1,
        "red_friendly_collision_deaths": 0, "red_cross_collision_deaths": 0,
        "red_boundary_altitude_deaths": 1, "red_boundary_xy_deaths": 0,
        "blue_survivors": 0, "blue_attack_deaths": 2, "blue_boundary_deaths": 0,
        "blue_friendly_collision_deaths": 1, "blue_cross_collision_deaths": 0,
        "blue_boundary_altitude_deaths": 0, "blue_boundary_xy_deaths": 0,
        "red_attack_kills": 2, "blue_attack_kills": 1,
    }
    validate_episode_accounting_3v3(rec, env_index=0)


def test_happo_episode_accounting_validation_rejects_survivor_total_mismatch():
    rec = {
        "red_survivors": 3, "red_attack_deaths": 1, "red_boundary_deaths": 0,
        "red_friendly_collision_deaths": 0, "red_cross_collision_deaths": 0,
        "red_boundary_altitude_deaths": 0, "red_boundary_xy_deaths": 0,
        "blue_survivors": 3, "blue_attack_deaths": 0, "blue_boundary_deaths": 0,
        "blue_friendly_collision_deaths": 0, "blue_cross_collision_deaths": 0,
        "blue_boundary_altitude_deaths": 0, "blue_boundary_xy_deaths": 0,
        "red_attack_kills": 0, "blue_attack_kills": 1,
    }
    try:
        validate_episode_accounting_3v3(rec, env_index=4)
    except RuntimeError as exc:
        assert "Death ledger mismatch" in str(exc)
        assert "env=4" in str(exc)
    else:
        raise AssertionError("expected death ledger mismatch")


def test_happo_episode_accounting_validation_rejects_boundary_mismatch():
    rec = {
        "red_survivors": 2, "red_attack_deaths": 0, "red_boundary_deaths": 1,
        "red_friendly_collision_deaths": 0, "red_cross_collision_deaths": 0,
        "red_boundary_altitude_deaths": 0, "red_boundary_xy_deaths": 0,
        "blue_survivors": 3, "blue_attack_deaths": 0, "blue_boundary_deaths": 0,
        "blue_friendly_collision_deaths": 0, "blue_cross_collision_deaths": 0,
        "blue_boundary_altitude_deaths": 0, "blue_boundary_xy_deaths": 0,
        "red_attack_kills": 0, "blue_attack_kills": 0,
    }
    try:
        validate_episode_accounting_3v3(rec, env_index=1)
    except RuntimeError as exc:
        assert "Boundary death mismatch" in str(exc)
    else:
        raise AssertionError("expected boundary mismatch")


def test_happo_episode_accounting_validation_rejects_attack_mismatch():
    rec = {
        "red_survivors": 3, "red_attack_deaths": 0, "red_boundary_deaths": 0,
        "red_friendly_collision_deaths": 0, "red_cross_collision_deaths": 0,
        "red_boundary_altitude_deaths": 0, "red_boundary_xy_deaths": 0,
        "blue_survivors": 2, "blue_attack_deaths": 1, "blue_boundary_deaths": 0,
        "blue_friendly_collision_deaths": 0, "blue_cross_collision_deaths": 0,
        "blue_boundary_altitude_deaths": 0, "blue_boundary_xy_deaths": 0,
        "red_attack_kills": 0, "blue_attack_kills": 0,
    }
    try:
        validate_episode_accounting_3v3(rec, env_index=2)
    except RuntimeError as exc:
        assert "Attack ledger mismatch" in str(exc)
    else:
        raise AssertionError("expected attack ledger mismatch")
