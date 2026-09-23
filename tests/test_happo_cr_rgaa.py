from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.cr_rgaa import (
    CR_RGAA_METHOD, RelationalRoleValueNetwork, attention_diagnostics,
    conflict_aware_fusion,
)
from algorithm.happo.rgaa import (
    RGAA_METHOD, ROLE_AUX_REWARD_MODE, RoleAdvantageRolloutBuffer,
    extract_rgaa_auxiliary_rewards,
)
from algorithm.happo.trainer import HAPPOTrainer, preceding_factor_update
from algorithm.train_happo import _algorithm_name
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config


ROOT = Path(__file__).resolve().parents[1]


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(**updates):
    config = {
        "method_variant": CR_RGAA_METHOD, "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 1,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 16, "seed": 23, "role_advantage_coef": 0.5,
        "role_aux_reward_mode": ROLE_AUX_REWARD_MODE,
        "cr_rgaa_relational_dim": 16, "cr_rgaa_attention_heads": 4,
        "cr_rgaa_relational_value_coef": 1.0, "cr_rgaa_gate_beta": 2.0,
        "cr_rgaa_lambda_floor_ratio": 0.2,
    }
    config.update(updates)
    return config


def test_relational_role_critic_shape_sharing_and_zero_initialized_correction():
    critic = RelationalRoleValueNetwork(hidden_dim=16, relational_dim=16, attention_heads=4)
    observations = torch.randn(3, 4, OBS_DIM)
    active = torch.ones(3, 4)
    details = critic(observations, active, return_details=True)
    assert details.values.shape == (3, 4)
    assert details.attention_weights.shape == (3, 4, 4, 4)
    assert torch.count_nonzero(details.residuals) == 0
    assert critic.local_mav is not critic.local_uav
    assert critic.uav_correction is not critic.mav_correction
    # UAV1/UAV2/UAV3 all use these exact two shared module instances.
    assert critic.architecture()["local_sharing"]["UAV1-UAV3"] == "shared"
    assert critic.architecture()["correction_sharing"]["UAV1-UAV3"] == "shared"


def test_attention_mask_blocks_dead_keys_and_all_inactive_is_finite():
    torch.manual_seed(3)
    critic = RelationalRoleValueNetwork(hidden_dim=16, relational_dim=16, attention_heads=4)
    with torch.no_grad():
        critic.mav_correction[-1].weight.fill_(0.1)
        critic.uav_correction[-1].weight.fill_(0.1)
    observations = torch.randn(2, 4, OBS_DIM)
    active = torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.float32)
    first = critic(observations, active, return_details=True)
    changed = observations.clone(); changed[0, 2] += 10000.0
    second = critic(changed, active, return_details=True)
    assert torch.allclose(first.values[0, [0, 1, 3]], second.values[0, [0, 1, 3]], atol=1e-6)
    assert torch.count_nonzero(first.attention_weights[0, :, :, 2]) == 0
    assert torch.count_nonzero(first.values[1]) == 0
    assert torch.isfinite(first.values).all() and torch.isfinite(first.attention_weights).all()
    diagnostics = attention_diagnostics(first.attention_weights, active)
    assert diagnostics and all(np.isfinite(value) for value in diagnostics.values())
    assert 0.0 <= diagnostics["cr_attention_entropy"] <= 1.0


def test_conflict_gate_bounds_monotonicity_and_exact_fusion():
    team = torch.tensor([1.0, 0.0, 1.0])
    role = torch.tensor([1.0, 0.0, -1.0])
    combined, lambdas, consistency = conflict_aware_fusion(
        team, role, role_advantage_coef=0.5, beta=2.0, lambda_floor_ratio=0.2,
    )
    assert torch.equal(consistency, team * role)
    assert torch.allclose(combined, team + lambdas * role)
    assert torch.all((lambdas >= 0.1) & (lambdas <= 0.5))
    assert lambdas[0] > lambdas[1] > lambdas[2]


def test_cr_rgaa_initializes_independently_and_preserves_main_initialization_rng():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    baseline_rng = torch.get_rng_state().clone()
    cr = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert isinstance(cr.buffer, RoleAdvantageRolloutBuffer)
        assert cr.relational_role_critic is not None
        assert torch.equal(torch.get_rng_state(), baseline_rng)
        assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), cr.actors.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), cr.critic.parameters()))
        assert _algorithm_name("vanilla", CR_RGAA_METHOD, "mlp") == "cr_rgaa_happo"
    finally:
        baseline.close(); cr.close()


def _force_role_advantages(trainer: HAPPOTrainer) -> None:
    assert isinstance(trainer.buffer, RoleAdvantageRolloutBuffer)
    count = trainer.buffer.horizon * trainer.buffer.num_envs
    base = np.linspace(-2.0, 2.0, count, dtype=np.float32)
    for agent in range(4):
        trainer.buffer.role_advantages[:, :, agent] = base.reshape(
            trainer.buffer.horizon, trainer.buffer.num_envs,
        ) + np.float32(agent * 0.2)


def test_cr_fusion_is_actual_ppo_input_without_second_normalization_and_factor_is_unchanged():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(num_envs=2, rollout_steps=2, minibatch_size=8),
    )
    try:
        trainer.collect_rollout(); _force_role_advantages(trainer)
        metrics = trainer.update()
        expected = (
            trainer.last_cr_rgaa_team_normalized_advantages
            + trainer.last_cr_rgaa_adaptive_lambdas
            * trainer.last_cr_rgaa_role_normalized_advantages
        )
        assert torch.equal(trainer.last_cr_rgaa_combined_advantages, expected)
        assert torch.equal(trainer.last_cr_rgaa_ppo_advantages, expected)
        assert torch.equal(
            trainer.last_cr_rgaa_consistency,
            trainer.last_cr_rgaa_team_normalized_advantages
            * trainer.last_cr_rgaa_role_normalized_advantages,
        )
        observations = torch.as_tensor(trainer.buffer.observations.reshape(-1, 4, OBS_DIM))
        actions = torch.as_tensor(trainer.buffer.actions.reshape(-1, 4, 3))
        old = torch.as_tensor(trainer.buffer.log_probs.reshape(-1, 4))
        active = torch.as_tensor(trainer.buffer.active_masks.reshape(-1, 4))
        factor = torch.ones_like(old[:, 0])
        assert np.array_equal(trainer.last_cr_rgaa_factor_history[0], factor.numpy())
        for index, agent in enumerate(metrics["agent_update_order"], start=1):
            with torch.no_grad():
                new, _ = trainer.actors.actors[agent].evaluate_actions(
                    observations[:, agent], actions[:, agent],
                )
            factor = preceding_factor_update(factor, old[:, agent], new, active[:, agent])
            np.testing.assert_allclose(
                trainer.last_cr_rgaa_factor_history[index], factor.numpy(), rtol=1e-5, atol=1e-6,
            )
    finally:
        trainer.close()


def test_cr_update_diagnostics_are_finite_and_auxiliary_semantics_are_reused():
    batch = extract_rgaa_auxiliary_rewards(
        [{
            "mav_process_reward": 1.0, "uav1_process_reward": 2.0,
            "uav2_process_reward": 3.0, "uav3_process_reward": 4.0,
            "death_causes": {"UAV2": "boundary"}, "event_reward": 999.0,
        }],
        {"mav_loss": -100.0, "uav_loss": -10.0},
    )
    np.testing.assert_array_equal(batch.auxiliary_rewards, [[1.0, 2.0, -7.0, 4.0]])
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        _, metrics = trainer.train_update()
        required = {
            "cr_role_critic_total_loss", "cr_relational_residual_abs_mean",
            "cr_attention_entropy", "cr_attention_self_mass", "cr_lambda_mean",
            "cr_conflict_rate", "cr_lambda_mean_MAV", "cr_lambda_mean_UAV3",
        }
        assert required <= metrics.keys()
        assert all(np.isfinite(metrics[key]) for key in required)
        assert 0.1 <= metrics["cr_lambda_mean"] <= 0.5
    finally:
        trainer.close()


def _main_parameters_after_updates(method: str, updates: int):
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(method_variant=method, role_advantage_coef=0.0),
    )
    try:
        for _ in range(updates):
            trainer.train_update()
        return (
            [parameter.detach().clone() for parameter in trainer.actors.parameters()],
            [parameter.detach().clone() for parameter in trainer.critic.parameters()],
        )
    finally:
        trainer.close()


def test_cr_coef_zero_preserves_vanilla_main_updates_and_main_rng_sequence():
    vanilla = _main_parameters_after_updates("baseline", 2)
    cr = _main_parameters_after_updates(CR_RGAA_METHOD, 2)
    assert all(torch.equal(a, b) for a, b in zip(vanilla[0], cr[0]))
    assert all(torch.equal(a, b) for a, b in zip(vanilla[1], cr[1]))


def test_cr_checkpoint_exact_resume_rng_and_cross_method_rejection(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "cr.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "cr_rgaa_happo"
    assert payload["method_variant"] == CR_RGAA_METHOD
    assert payload["credit_gate"]["type"] == "analytic_consistency_sigmoid_v1"
    assert payload["relational_role_critic"]["correction_zero_initialized"] is True
    restored = HAPPOTrainer(short_v39(), trainer_config())
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD))
    try:
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        assert restored.cr_rgaa_rng.bit_generator.state == source.cr_rgaa_rng.bit_generator.state
        assert all(
            torch.equal(a, b)
            for a, b in zip(source.relational_role_critic.parameters(), restored.relational_role_critic.parameters())
        )
        observations = torch.zeros(8, 4, OBS_DIM)
        targets = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        active = torch.ones(8, 4)
        source._train_cr_relational_role_critic(observations, targets, active)
        restored._train_cr_relational_role_critic(observations, targets, active)
        assert restored.cr_rgaa_rng.bit_generator.state == source.cr_rgaa_rng.bit_generator.state
        assert all(
            torch.equal(a, b)
            for a, b in zip(source.relational_role_critic.parameters(), restored.relational_role_critic.parameters())
        )
        with pytest.raises(RuntimeError, match="resume method mismatch"):
            rgaa.load_checkpoint(checkpoint)
        rgaa_checkpoint = tmp_path / "rgaa.pt"
        rgaa.save_checkpoint(rgaa_checkpoint)
        with pytest.raises(RuntimeError, match="resume method mismatch"):
            restored.load_checkpoint(rgaa_checkpoint)
    finally:
        source.close(); restored.close(); rgaa.close()


def test_cr_config_only_adds_declared_method_parameters():
    with open("configs/happo_rgaa_v39.yaml", encoding="utf-8") as stream:
        rgaa = yaml.safe_load(stream)["training"]
    with open("configs/happo_cr_rgaa_v39.yaml", encoding="utf-8") as stream:
        cr = yaml.safe_load(stream)["training"]
    assert cr.pop("cr_rgaa_relational_dim") == 64
    assert cr.pop("cr_rgaa_attention_heads") == 4
    assert cr.pop("cr_rgaa_relational_value_coef") == 1.0
    assert cr.pop("cr_rgaa_gate_beta") == 2.0
    assert cr.pop("cr_rgaa_lambda_floor_ratio") == 0.2
    cr["method_variant"] = RGAA_METHOD
    assert cr == rgaa


def test_standalone_evaluator_loads_only_cr_actors(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "cr.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run(
        [
            sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "evaluation_cr_summary.json").read_text())
    assert summary["algorithm"] == "cr_rgaa_happo"
    assert summary["method_variant"] == CR_RGAA_METHOD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cr_rgaa_tiny_cuda_update_is_finite():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(device="cuda", rollout_steps=1, minibatch_size=4),
    )
    try:
        _, metrics = trainer.train_update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
        assert next(trainer.relational_role_critic.parameters()).is_cuda
    finally:
        trainer.close()
