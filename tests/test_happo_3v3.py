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
    ppo_clipped_policy_loss,
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


def test_ppo_clipped_policy_loss_positive_and_negative_advantages():
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.9])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = ppo_clipped_policy_loss(ratio, adv, 0.2)
    manual = -torch.minimum(
        ratio * adv,
        ratio.clamp(0.8, 1.2) * adv,
    ).mean()
    assert torch.allclose(loss, manual)


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
