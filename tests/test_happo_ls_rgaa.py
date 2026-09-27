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

from algorithm.happo.ls_rgaa import (
    LS_AUXILIARY_SEMANTICS, LS_RGAA_METHOD, LossSeparatedRoleRolloutBuffer,
    LossValueNetwork, extract_ls_rewards,
)
from algorithm.happo.cr_rgaa import CR_RGAA_METHOD
from algorithm.happo.lp_cr_rgaa import LP_CR_RGAA_METHOD
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
        "method_variant": LS_RGAA_METHOD,
        "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 2,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 16, "seed": 37, "role_advantage_coef": 0.5,
        "ls_auxiliary_semantics": LS_AUXILIARY_SEMANTICS,
    }
    config.update(updates)
    return config


def test_rewards_are_strictly_separated_and_binary_own_loss_only():
    batch = extract_ls_rewards([
        {
            "mav_process_reward": 1.0, "uav1_process_reward": 2.0,
            "uav2_process_reward": 3.0, "uav3_process_reward": 4.0,
            "death_causes": {"UAV1": "boundary", "UAV2": "blue_attack", "Blue1": "red_attack"},
            "event_reward": 999.0,
        },
        {
            "mav_process_reward": -1.0, "uav1_process_reward": -2.0,
            "uav2_process_reward": -3.0, "uav3_process_reward": -4.0,
            "death_causes": {},
        },
    ])
    np.testing.assert_array_equal(batch.process_rewards, [[1, 2, 3, 4], [-1, -2, -3, -4]])
    np.testing.assert_array_equal(batch.loss_rewards, [[0, -1, -1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(batch.boundary_loss_events[0], [0, 1, 0, 0])
    np.testing.assert_array_equal(batch.blue_attack_loss_events[0], [0, 0, 1, 0])


def test_loss_value_network_is_strictly_bounded_and_has_declared_architecture():
    network = LossValueNetwork(hidden_dim=16)
    values = network(torch.randn(128, OBS_DIM) * 100.0)
    assert torch.all(values > -1.0)
    assert torch.all(values < 0.0)
    assert network.architecture()["output_transform"] == "negative_sigmoid"


def _return_buffer(horizon: int = 2) -> LossSeparatedRoleRolloutBuffer:
    buffer = LossSeparatedRoleRolloutBuffer(horizon, 1)
    buffer.position = horizon
    buffer.active_masks[:] = 1.0
    return buffer


def test_loss_return_death_and_safe_terminal_are_exact_and_do_not_bootstrap():
    death = _return_buffer(1)
    death.loss_rewards[0, 0, 1] = -1.0
    death.own_loss_events[0, 0, 1] = 1.0
    last_active = np.ones((1, 4), dtype=np.float32)
    last_active[0, 1] = 0.0
    death.compute_auxiliary_returns(
        np.zeros((1, 4), np.float32), np.full((1, 4), -0.75, np.float32),
        last_active, 0.99, 0.95,
    )
    assert death.loss_returns[0, 0, 1] == -1.0

    safe = _return_buffer(1)
    safe.terminated[0, 0] = True
    safe.compute_auxiliary_returns(
        np.zeros((1, 4), np.float32), np.full((1, 4), -0.75, np.float32),
        np.ones((1, 4), np.float32), 0.99, 0.95,
    )
    np.testing.assert_array_equal(safe.loss_returns[0, 0], np.zeros(4, np.float32))


def test_loss_return_rollout_bootstrap_and_individual_death_continuation():
    buffer = _return_buffer(2)
    buffer.loss_values[:] = -0.2
    # UAV1 dies at t=0 while UAV2 survives and receives future loss at t=1.
    buffer.loss_rewards[0, 0, 1] = -1.0
    buffer.own_loss_events[0, 0, 1] = 1.0
    buffer.active_masks[1, 0, 1] = 0.0
    buffer.loss_rewards[1, 0, 2] = -1.0
    buffer.own_loss_events[1, 0, 2] = 1.0
    last_active = np.ones((1, 4), np.float32)
    last_active[0, 2] = 0.0
    last_loss = np.full((1, 4), -0.4, np.float32)
    buffer.compute_auxiliary_returns(
        np.zeros((1, 4), np.float32), last_loss, last_active, 0.99, 0.95,
    )
    assert buffer.loss_returns[0, 0, 1] == -1.0
    assert buffer.loss_returns[1, 0, 2] == -1.0
    # Surviving UAV3 at rollout boundary bootstraps the negative loss value.
    expected = 0.99 * ((1.0 - 0.95) * -0.4 + 0.95 * -0.4)
    assert np.isclose(buffer.loss_returns[1, 0, 3], expected)
    assert np.all(buffer.loss_returns <= 0.0)


def _force_auxiliary_streams(trainer: HAPPOTrainer) -> None:
    assert isinstance(trainer.buffer, LossSeparatedRoleRolloutBuffer)
    values = np.asarray([-2.0, 2.0, -1.0, 1.0], np.float32).reshape(2, 2)
    loss = np.asarray([-1.0, -0.25, -0.75, 0.0], np.float32).reshape(2, 2)
    for agent in range(4):
        trainer.buffer.process_advantages[:, :, agent] = values + agent * 0.1
        trainer.buffer.process_returns[:, :, agent] = trainer.buffer.process_advantages[:, :, agent]
        trainer.buffer.loss_returns[:, :, agent] = loss
        trainer.buffer.loss_values[:, :, agent] = -0.5


def test_ls_exact_fusion_is_actual_ppo_input_without_loss_normalization_or_gate():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        _force_auxiliary_streams(trainer)
        metrics = trainer.update()
        expected = (
            trainer.last_ls_rgaa_team_normalized_advantages
            + 0.5 * trainer.last_ls_rgaa_process_normalized_advantages
            + 0.5 * trainer.last_ls_rgaa_loss_returns
        )
        assert torch.equal(trainer.last_ls_rgaa_combined_advantages, expected)
        assert torch.equal(trainer.last_ls_rgaa_ppo_advantages, expected)
        assert torch.equal(
            trainer.last_ls_rgaa_loss_returns,
            torch.as_tensor(trainer.buffer.loss_returns.reshape(-1, 4)),
        )
        assert metrics["loss_return_positive_violation_count"] == 0.0
        assert "cr_lambda_mean" not in metrics
    finally:
        trainer.close()


def test_ls_sharing_and_isolated_initialization_preserve_main_networks_and_rng():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    baseline_rng = torch.get_rng_state().clone()
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD))
    ls = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert torch.equal(torch.get_rng_state(), baseline_rng)
        for other in (rgaa, ls):
            assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), other.actors.parameters()))
            assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), other.critic.parameters()))
        assert ls.process_mav_critic is not ls.process_uav_critic
        assert ls.loss_mav_critic is not ls.loss_uav_critic
        assert _algorithm_name("vanilla", LS_RGAA_METHOD, "mlp") == "ls_rgaa_happo"
    finally:
        baseline.close(); rgaa.close(); ls.close()


def test_ls_zero_coefficient_preserves_one_full_vanilla_update():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    ls = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.0))
    rollout_rng = torch.get_rng_state().clone()
    try:
        torch.set_rng_state(rollout_rng.clone())
        baseline.train_update()
        torch.set_rng_state(rollout_rng.clone())
        ls.train_update()
        assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), ls.actors.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), ls.critic.parameters()))
        assert baseline.rng.bit_generator.state == ls.rng.bit_generator.state
    finally:
        baseline.close(); ls.close()


def test_ls_checkpoint_exact_resume_and_cross_method_rejection(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "ls.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = HAPPOTrainer(short_v39(), trainer_config())
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD))
    cr = HAPPOTrainer(short_v39(), trainer_config(
        method_variant=CR_RGAA_METHOD, cr_rgaa_relational_dim=16,
        cr_rgaa_attention_heads=4,
    ))
    lp = HAPPOTrainer(short_v39(), trainer_config(
        method_variant=LP_CR_RGAA_METHOD, cr_rgaa_relational_dim=16,
        cr_rgaa_attention_heads=4,
    ))
    try:
        assert payload["algorithm"] == "ls_rgaa_happo"
        assert payload["ls_rgaa_version"] == 1
        assert payload["loss_credit"]["coefficient_source"] == "role_advantage_coef"
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        assert restored.ls_rgaa_rng.bit_generator.state == source.ls_rgaa_rng.bit_generator.state
        for name in ("process_mav_critic", "process_uav_critic", "loss_mav_critic", "loss_uav_critic"):
            assert all(torch.equal(a, b) for a, b in zip(
                getattr(source, name).parameters(), getattr(restored, name).parameters(),
            ))
        observations = torch.zeros(8, 4, OBS_DIM)
        process_targets = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0
        loss_targets = -torch.linspace(0.0, 1.0, 32).reshape(8, 4)
        active = torch.ones(8, 4)
        source_main_rng = deepcopy(source.rng.bit_generator.state)
        source_torch_rng = torch.get_rng_state().clone()
        source._train_ls_auxiliary_critics(observations, process_targets, loss_targets, active)
        assert source.rng.bit_generator.state == source_main_rng
        assert torch.equal(torch.get_rng_state(), source_torch_rng)
        restored._train_ls_auxiliary_critics(observations, process_targets, loss_targets, active)
        assert restored.ls_rgaa_rng.bit_generator.state == source.ls_rgaa_rng.bit_generator.state
        for name in ("process_mav_critic", "process_uav_critic", "loss_mav_critic", "loss_uav_critic"):
            assert all(torch.equal(a, b) for a, b in zip(
                getattr(source, name).parameters(), getattr(restored, name).parameters(),
            ))
        for other in (rgaa, cr, lp):
            with pytest.raises(RuntimeError, match="resume method mismatch"):
                other.load_checkpoint(checkpoint)
        contract = validate_checkpoint_contract(payload, short_v39())
        assert contract["method_variant"] == LS_RGAA_METHOD
    finally:
        source.close(); restored.close(); rgaa.close(); cr.close(); lp.close()


def test_ls_config_differs_from_rgaa_only_by_declared_semantics_and_method():
    with open("configs/happo_rgaa_v39.yaml", encoding="utf-8") as stream:
        rgaa = yaml.safe_load(stream)["training"]
    with open("configs/happo_ls_rgaa_v39.yaml", encoding="utf-8") as stream:
        ls = yaml.safe_load(stream)["training"]
    assert ls.pop("ls_auxiliary_semantics") == LS_AUXILIARY_SEMANTICS
    ls["method_variant"] = RGAA_METHOD
    assert ls == rgaa


def test_audit_supports_ls_metrics_and_empty_phase_is_na(tmp_path):
    rows = [{
        "sampled_steps": "100000", "completed_episodes": "10",
        "method_variant": LS_RGAA_METHOD, "mean_process_reward": "0.25",
        "loss_return_mean": "-0.2", "loss_event_rate_UAV1": "0.1",
        **{f"own_loss_count_{aid}": "0" for aid in RED_IDS},
        **{f"own_boundary_loss_count_{aid}": "0" for aid in RED_IDS},
        **{f"own_blue_attack_loss_count_{aid}": "0" for aid in RED_IDS},
    }]
    _, phases = analyze_training_rows(rows, LS_RGAA_METHOD)
    assert phases["phase_0_750k"]["mechanism_metrics"]["mean_process_reward"]["mean"] == 0.25
    assert phases["phase_750k_1250k"]["death_events"]["UAV1"]["boundary_events"] is None


def test_standalone_stochastic_evaluator_loads_only_ls_actors(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "ls.pt"
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
    summary = json.loads((tmp_path / "evaluation_ls_stochastic_summary.json").read_text())
    assert summary["algorithm"] == "ls_rgaa_happo"
    assert summary["method_variant"] == LS_RGAA_METHOD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ls_cuda_tiny_update_is_finite_and_loss_returns_are_nonpositive():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(device="cuda", num_envs=1, rollout_steps=1, minibatch_size=4),
    )
    try:
        _, metrics = trainer.train_update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
        assert metrics["loss_return_positive_violation_count"] == 0.0
        assert np.all(trainer.buffer.loss_returns <= 1e-7)
        for critic in (
            trainer.process_mav_critic, trainer.process_uav_critic,
            trainer.loss_mav_critic, trainer.loss_uav_critic,
        ):
            assert next(critic.parameters()).is_cuda
    finally:
        trainer.close()
