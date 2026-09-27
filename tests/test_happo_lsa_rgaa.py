from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.cr_rgaa import CR_RGAA_METHOD
from algorithm.happo.lp_cr_rgaa import LP_CR_RGAA_METHOD
from algorithm.happo.ls_rgaa import (
    LS_AUXILIARY_SEMANTICS, LS_RGAA_METHOD, LossSeparatedRoleRolloutBuffer,
)
from algorithm.happo.lsa_rgaa import (
    LSA_CREDIT_SEMANTICS, LSA_RGAA_METHOD, compute_loss_advantage,
)
from algorithm.happo.rgaa import RGAA_METHOD
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.train_happo import _algorithm_name
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config
from tools.audit_role_guided_run import analyze_training_rows, validate_checkpoint_contract


ROOT = Path(__file__).resolve().parents[1]


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(**updates):
    config = {
        "method_variant": LSA_RGAA_METHOD,
        "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 2,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 16, "seed": 41, "role_advantage_coef": 0.5,
        "ls_auxiliary_semantics": LS_AUXILIARY_SEMANTICS,
        "lsa_credit_semantics": LSA_CREDIT_SEMANTICS,
    }
    config.update(updates)
    return config


def force_auxiliary_streams(trainer: HAPPOTrainer) -> None:
    assert isinstance(trainer.buffer, LossSeparatedRoleRolloutBuffer)
    process = np.asarray([-2.0, 2.0, -1.0, 1.0], np.float32).reshape(2, 2)
    loss_return = np.asarray([-1.0, 0.0, -0.75, -0.25], np.float32).reshape(2, 2)
    loss_value = np.asarray([-0.9, -0.9, -0.4, -0.6], np.float32).reshape(2, 2)
    for agent in range(4):
        trainer.buffer.process_advantages[:, :, agent] = process + agent * 0.1
        trainer.buffer.process_returns[:, :, agent] = trainer.buffer.process_advantages[:, :, agent]
        trainer.buffer.loss_returns[:, :, agent] = loss_return
        trainer.buffer.loss_values[:, :, agent] = loss_value


def test_compute_loss_advantage_uses_unmodified_rollout_baseline_without_normalization():
    returns = torch.tensor([-1.0, 0.0])
    values = torch.tensor([-0.9, -0.9])
    returns_before = returns.clone()
    values_before = values.clone()
    result = compute_loss_advantage(returns, values)
    torch.testing.assert_close(result, torch.tensor([-0.1, 0.9]))
    assert torch.equal(returns, returns_before)
    assert torch.equal(values, values_before)
    with pytest.raises(ValueError, match="identical shape"):
        compute_loss_advantage(torch.zeros(2), torch.zeros(2, 1))


def test_lsa_exact_fusion_is_actual_ppo_input_and_loss_target_stays_raw(monkeypatch):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    captured: dict[str, torch.Tensor] = {}
    original = trainer._train_ls_auxiliary_critics

    def capture(observations, process_targets, loss_targets, active_masks):
        captured["loss_targets"] = loss_targets.detach().clone()
        return original(observations, process_targets, loss_targets, active_masks)

    monkeypatch.setattr(trainer, "_train_ls_auxiliary_critics", capture)
    try:
        trainer.collect_rollout()
        force_auxiliary_streams(trainer)
        raw_loss_returns = torch.as_tensor(trainer.buffer.loss_returns.reshape(-1, 4)).clone()
        stored_values = torch.as_tensor(trainer.buffer.loss_values.reshape(-1, 4)).clone()
        metrics = trainer.update()
        expected_credit = raw_loss_returns - stored_values
        expected = (
            trainer.last_lsa_rgaa_team_normalized_advantages
            + 0.5 * trainer.last_lsa_rgaa_process_normalized_advantages
            + 0.5 * expected_credit
        )
        assert torch.equal(trainer.last_lsa_rgaa_loss_returns.cpu(), raw_loss_returns)
        assert torch.equal(trainer.last_lsa_rgaa_rollout_loss_values.cpu(), stored_values)
        assert torch.equal(trainer.last_lsa_rgaa_loss_advantages.cpu(), expected_credit)
        assert torch.equal(trainer.last_lsa_rgaa_combined_advantages, expected)
        assert torch.equal(trainer.last_lsa_rgaa_ppo_advantages, expected)
        assert torch.equal(captured["loss_targets"].cpu(), raw_loss_returns)
        assert metrics["loss_advantage_range_violation_count"] == 0.0
        assert metrics["loss_return_positive_violation_count"] == 0.0
        assert "cr_lambda_mean" not in metrics
    finally:
        trainer.close()


def test_lsa_never_recomputes_loss_baseline_after_collection(monkeypatch):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        force_auxiliary_streams(trainer)
        stored = torch.as_tensor(trainer.buffer.loss_values.reshape(-1, 4)).clone()
        expected = torch.as_tensor(trainer.buffer.loss_returns.reshape(-1, 4)) - stored

        def forbidden(*args, **kwargs):
            raise AssertionError("loss value must not be recomputed during PPO update")

        monkeypatch.setattr(trainer, "_ls_values", forbidden)
        trainer.update()
        assert torch.equal(trainer.last_lsa_rgaa_loss_advantages.cpu(), expected)
    finally:
        trainer.close()


def test_ls_v1_and_lsa_credit_paths_are_isolated():
    ls = HAPPOTrainer(short_v39(), trainer_config(method_variant=LS_RGAA_METHOD))
    lsa = HAPPOTrainer(short_v39(), trainer_config())
    try:
        for trainer in (ls, lsa):
            trainer.collect_rollout()
            force_auxiliary_streams(trainer)
            trainer.update()
        assert torch.equal(
            ls.last_ls_rgaa_loss_returns.cpu(),
            torch.as_tensor(ls.buffer.loss_returns.reshape(-1, 4)),
        )
        expected_lsa = (
            torch.as_tensor(lsa.buffer.loss_returns.reshape(-1, 4))
            - torch.as_tensor(lsa.buffer.loss_values.reshape(-1, 4))
        )
        assert torch.equal(lsa.last_lsa_rgaa_loss_advantages.cpu(), expected_lsa)
        assert not torch.equal(
            ls.last_ls_rgaa_combined_advantages.cpu(),
            lsa.last_lsa_rgaa_combined_advantages.cpu(),
        )
    finally:
        ls.close(); lsa.close()


def test_lsa_diagnostics_include_conditional_survival_and_per_agent_fields():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        force_auxiliary_streams(trainer)
        # Attach one boundary and one Blue-attack own-loss event to test conditions.
        trainer.buffer.own_loss_events[:] = 0.0
        trainer.buffer.boundary_loss_events[:] = 0.0
        trainer.buffer.blue_attack_loss_events[:] = 0.0
        trainer.buffer.own_loss_events[0, 0, 1] = 1.0
        trainer.buffer.boundary_loss_events[0, 0, 1] = 1.0
        trainer.buffer.own_loss_events[1, 1, 2] = 1.0
        trainer.buffer.blue_attack_loss_events[1, 1, 2] = 1.0
        # The invariant requires every own-loss transition return to be exactly -1.
        trainer.buffer.loss_returns[0, 0, 1] = -1.0
        trainer.buffer.loss_returns[1, 1, 2] = -1.0
        metrics = trainer.update()
        for name in (
            "loss_value_mean", "loss_value_mean_abs", "loss_advantage_mean",
            "loss_advantage_mean_abs", "loss_advantage_std", "loss_advantage_min",
            "loss_advantage_max", "loss_advantage_positive_rate",
            "loss_advantage_negative_rate", "loss_advantage_range_violation_count",
            "loss_advantage_on_own_loss_mean", "loss_advantage_on_boundary_mean",
            "loss_advantage_on_blue_attack_mean", "survival_loss_advantage_mean",
            "loss_advantage_on_boundary_mean_UAV1",
            "loss_advantage_on_blue_attack_mean_UAV2",
            "survival_loss_advantage_mean_UAV3",
        ):
            assert name in metrics and np.isfinite(metrics[name])
        assert metrics["loss_advantage_range_violation_count"] == 0.0
        assert metrics["loss_advantage_positive_rate"] > 0.0
    finally:
        trainer.close()


def test_lsa_initialization_and_zero_coefficient_preserve_vanilla_update_and_main_rng():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    baseline_global_rng = torch.get_rng_state().clone()
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD))
    lsa = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.0))
    rollout_rng = torch.get_rng_state().clone()
    try:
        assert torch.equal(torch.get_rng_state(), baseline_global_rng)
        for other in (rgaa, lsa):
            assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), other.actors.parameters()))
            assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), other.critic.parameters()))
        assert lsa.process_mav_critic is not lsa.process_uav_critic
        assert lsa.loss_mav_critic is not lsa.loss_uav_critic
        torch.set_rng_state(rollout_rng.clone())
        baseline.train_update()
        torch.set_rng_state(rollout_rng.clone())
        lsa.train_update()
        assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), lsa.actors.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), lsa.critic.parameters()))
        assert baseline.rng.bit_generator.state == lsa.rng.bit_generator.state
        assert _algorithm_name("vanilla", LSA_RGAA_METHOD, "mlp") == "lsa_rgaa_happo"
    finally:
        baseline.close(); rgaa.close(); lsa.close()


def test_lsa_checkpoint_exact_resume_metadata_and_cross_method_rejection(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "lsa.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = HAPPOTrainer(short_v39(), trainer_config())
    others = [
        HAPPOTrainer(short_v39(), trainer_config(method_variant=LS_RGAA_METHOD)),
        HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD)),
        HAPPOTrainer(short_v39(), trainer_config(
            method_variant=CR_RGAA_METHOD, cr_rgaa_relational_dim=16,
            cr_rgaa_attention_heads=4,
        )),
        HAPPOTrainer(short_v39(), trainer_config(
            method_variant=LP_CR_RGAA_METHOD, cr_rgaa_relational_dim=16,
            cr_rgaa_attention_heads=4,
        )),
    ]
    try:
        assert payload["algorithm"] == "lsa_rgaa_happo"
        assert payload["lsa_rgaa_version"] == 1
        assert payload["loss_actor_credit"]["formula"] == "loss_return_minus_rollout_loss_value"
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        assert restored.ls_rgaa_rng.bit_generator.state == source.ls_rgaa_rng.bit_generator.state
        for name in ("process_mav_critic", "process_uav_critic", "loss_mav_critic", "loss_uav_critic"):
            assert all(torch.equal(a, b) for a, b in zip(
                getattr(source, name).parameters(), getattr(restored, name).parameters(),
            ))
        contract = validate_checkpoint_contract(payload, short_v39())
        assert contract["method_variant"] == LSA_RGAA_METHOD
        for other in others:
            with pytest.raises(RuntimeError, match="resume method mismatch"):
                other.load_checkpoint(checkpoint)

        continuation_rng = payload["torch_rng"].clone()
        torch.set_rng_state(continuation_rng.clone())
        source.train_update()
        torch.set_rng_state(continuation_rng.clone())
        restored.train_update()
        assert source.env_steps == restored.env_steps
        assert source.rng.bit_generator.state == restored.rng.bit_generator.state
        assert source.ls_rgaa_rng.bit_generator.state == restored.ls_rgaa_rng.bit_generator.state
        for name in (
            "actors", "critic", "process_mav_critic", "process_uav_critic",
            "loss_mav_critic", "loss_uav_critic",
        ):
            assert all(torch.equal(a, b) for a, b in zip(
                getattr(source, name).parameters(), getattr(restored, name).parameters(),
            ))
    finally:
        source.close(); restored.close()
        for other in others:
            other.close()


def test_lsa_config_differs_from_ls_only_by_method_and_credit_semantics():
    with open("configs/happo_ls_rgaa_v39.yaml", encoding="utf-8") as stream:
        ls = yaml.safe_load(stream)["training"]
    with open("configs/happo_lsa_rgaa_v39.yaml", encoding="utf-8") as stream:
        lsa = yaml.safe_load(stream)["training"]
    assert lsa.pop("lsa_credit_semantics") == LSA_CREDIT_SEMANTICS
    lsa["method_variant"] = LS_RGAA_METHOD
    assert lsa == ls


def test_audit_supports_lsa_metrics_without_changing_ls_support():
    common = {
        "sampled_steps": "100000", "completed_episodes": "10",
        "mean_process_reward": "0.25", "loss_return_mean": "-0.2",
        **{f"own_loss_count_{aid}": "0" for aid in RED_IDS},
        **{f"own_boundary_loss_count_{aid}": "0" for aid in RED_IDS},
        **{f"own_blue_attack_loss_count_{aid}": "0" for aid in RED_IDS},
    }
    _, lsa_phases = analyze_training_rows([
        {**common, "method_variant": LSA_RGAA_METHOD, "loss_advantage_mean": "0.3"},
    ], LSA_RGAA_METHOD)
    _, ls_phases = analyze_training_rows([
        {**common, "method_variant": LS_RGAA_METHOD, "loss_advantage_mean": "0.3"},
    ], LS_RGAA_METHOD)
    assert lsa_phases["phase_0_750k"]["mechanism_metrics"]["loss_advantage_mean"]["mean"] == 0.3
    assert ls_phases["phase_0_750k"]["mechanism_metrics"]["loss_advantage_mean"] is None


def test_standalone_stochastic_evaluator_loads_only_lsa_actors(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "lsa.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run(
        [sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
         "--profile", "learnability", "--episodes", "1", "--device", "cpu",
         "--action-mode", "stochastic", "--action-seed", "2000"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "evaluation_lsa_stochastic_summary.json").read_text())
    assert summary["algorithm"] == "lsa_rgaa_happo"
    assert summary["method_variant"] == LSA_RGAA_METHOD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_lsa_cuda_tiny_update_is_finite_and_credit_range_is_valid():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(device="cuda", num_envs=1, rollout_steps=1, minibatch_size=4),
    )
    try:
        _, metrics = trainer.train_update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
        assert metrics["loss_advantage_range_violation_count"] == 0.0
        assert metrics["loss_return_positive_violation_count"] == 0.0
        for critic in (
            trainer.process_mav_critic, trainer.process_uav_critic,
            trainer.loss_mav_critic, trainer.loss_uav_critic,
        ):
            assert next(critic.parameters()).is_cuda
    finally:
        trainer.close()
