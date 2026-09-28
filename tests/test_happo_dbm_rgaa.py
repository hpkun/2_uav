from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.dbm_rgaa import (
    DBMGaussianActor, DBM_INITIALIZATION_SEMANTICS, DBM_RGAA_METHOD,
    RGAA_WIDE_METHOD, build_method_actors, dbm_rollout_diagnostics,
)
from algorithm.happo.rgaa import RoleAdvantageRolloutBuffer
from algorithm.happo.trainer import HAPPOTrainer
import algorithm.train_happo as train_entrypoint
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.train_happo import _algorithm_name
from env.mavuav import RED_IDS, load_environment_config


ROOT = Path(__file__).resolve().parents[1]


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(method=DBM_RGAA_METHOD, **updates):
    config = {
        "method_variant": method, "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 1,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 128, "seed": 23, "role_advantage_coef": 0.5,
        "actor_log_std_init": -0.25,
    }
    config.update(updates)
    return config


def _base_state(actor):
    if isinstance(actor, DBMGaussianActor):
        return {
            "0.weight": actor.network.encoder[0].weight,
            "0.bias": actor.network.encoder[0].bias,
            "2.weight": actor.network.encoder[2].weight,
            "2.bias": actor.network.encoder[2].bias,
            "4.weight": actor.network.base_head.weight,
            "4.bias": actor.network.base_head.bias,
            "log_std": actor.log_std,
        }
    return {**dict(actor.network.named_parameters()), "log_std": actor.log_std}


def _assert_base_equal(left, right):
    ls, rs = _base_state(left), _base_state(right)
    assert ls.keys() == rs.keys()
    for key in ls:
        assert torch.equal(ls[key], rs[key]), key


def test_dbm_architecture_and_exact_parameter_counts():
    actors = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=1, log_std_init=-0.25,
    )
    counts = [sum(p.numel() for p in actor.parameters()) for actor in actors.actors]
    assert counts == [29830, 30862, 30862, 30862]
    assert sum(counts) == 122416
    assert not isinstance(actors.actors[0], DBMGaussianActor)
    assert all(isinstance(actor, DBMGaussianActor) for actor in actors.actors[1:])


def test_dbm_sample_and_evaluate_actions_preserve_gaussian_contract():
    actor = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=2,
    ).actors[1]
    observations = torch.randn(11, 100)
    torch.manual_seed(202)
    actions, sampled_log_prob = actor.sample(observations)
    evaluated_log_prob, entropy = actor.evaluate_actions(observations, actions)
    assert actions.shape == (11, 3)
    assert sampled_log_prob.shape == entropy.shape == (11,)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=2e-5, rtol=2e-5)
    deterministic, _ = actor.sample(observations, deterministic=True)
    assert deterministic.shape == (11, 3)
    assert torch.all(deterministic >= -1.0) and torch.all(deterministic <= 1.0)
    details = actor.mode_diagnostics(observations)
    assert torch.allclose(details["router_probabilities"].sum(-1), torch.ones(11))
    assert torch.all(details["expert_outputs"].abs() <= 1.0)
    assert torch.all(details["scaled_residual"].abs() <= 0.25 + 1e-7)
    assert all(torch.isfinite(value).all() for value in details.values())


def test_wide_exact_parameter_counts():
    actors = build_method_actors(
        method_variant=RGAA_WIDE_METHOD, training_seed=1, log_std_init=-0.25,
    )
    counts = [sum(p.numel() for p in actor.parameters()) for actor in actors.actors]
    assert counts == [29830, 30922, 30922, 30922]
    assert sum(counts) == 122596


def test_reference_rgaa_exact_parameter_count():
    actors = build_method_actors(method_variant="rgaa", training_seed=1)
    assert sum(sum(p.numel() for p in actor.parameters()) for actor in actors.actors) == 119320


@pytest.mark.parametrize("agent", [1, 2, 3])
def test_dbm_initial_router_and_antisymmetric_experts(agent):
    actor = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=7,
    ).actors[agent]
    assert isinstance(actor, DBMGaussianActor)
    assert torch.count_nonzero(actor.network.router.weight) == 0
    assert torch.count_nonzero(actor.network.router.bias) == 0
    assert torch.equal(actor.network.experts[1].weight, -actor.network.experts[0].weight)
    assert torch.equal(actor.network.experts[1].bias, -actor.network.experts[0].bias)
    observations = torch.randn(9, 100)
    details = actor.mode_diagnostics(observations)
    assert torch.equal(details["router_probabilities"], torch.full((9, 2), 0.5))
    assert torch.equal(details["scaled_residual"], torch.zeros(9, 3))
    assert torch.equal(details["final_mean"], details["base_mean"])


def test_dbm_initial_policy_exactly_matches_rgaa_base_for_mean_logprob_and_sampling():
    torch.manual_seed(31)
    baseline = build_method_actors(method_variant="rgaa", training_seed=31)
    torch.manual_seed(31)
    dbm = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=31)
    observations = torch.randn(12, 100)
    for agent in range(4):
        _assert_base_equal(baseline.actors[agent], dbm.actors[agent])
        deterministic_base = baseline.actors[agent].sample(observations, deterministic=True)
        deterministic_dbm = dbm.actors[agent].sample(observations, deterministic=True)
        assert all(torch.equal(a, b) for a, b in zip(deterministic_base, deterministic_dbm))
        torch.manual_seed(919)
        sampled_base = baseline.actors[agent].sample(observations)
        torch.manual_seed(919)
        sampled_dbm = dbm.actors[agent].sample(observations)
        assert all(torch.equal(a, b) for a, b in zip(sampled_base, sampled_dbm))


def test_dbm_router_and_experts_receive_nonzero_gradients_and_break_symmetry():
    actor = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=4,
    ).actors[1]
    observations = torch.randn(32, 100)
    loss = actor._mean(observations).square().mean()
    loss.backward()
    assert actor.network.router.weight.grad.abs().sum() > 0
    assert actor.network.experts[0].weight.grad.abs().sum() > 0
    assert actor.network.experts[1].weight.grad.abs().sum() > 0
    before = actor.network.experts[0].weight.detach().clone()
    torch.optim.Adam(actor.parameters(), lr=1e-3).step()
    assert not torch.equal(actor.network.experts[0].weight, before)
    assert not torch.equal(actor.network.experts[1].weight, -actor.network.experts[0].weight)


def test_dbm_uav_modules_and_optimizers_are_private():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        parameter_ids = [
            {id(parameter) for parameter in trainer.actors.actors[i].parameters()}
            for i in (1, 2, 3)
        ]
        assert not (parameter_ids[0] & parameter_ids[1])
        assert not (parameter_ids[0] & parameter_ids[2])
        assert not (parameter_ids[1] & parameter_ids[2])
        optimizer_ids = [
            {id(parameter) for group in trainer.actor_optimizers[i].param_groups for parameter in group["params"]}
            for i in (1, 2, 3)
        ]
        assert optimizer_ids == parameter_ids
    finally:
        trainer.close()


def test_role_module_disabled_is_exact_rgaa_actor_behavior():
    torch.manual_seed(88)
    rgaa = build_method_actors(method_variant="rgaa", training_seed=88)
    state_after_rgaa = torch.get_rng_state().clone()
    torch.manual_seed(88)
    disabled = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=88, role_module_enabled=False,
    )
    state_after_disabled = torch.get_rng_state().clone()
    assert torch.equal(state_after_rgaa, state_after_disabled)
    assert rgaa.state_dict().keys() == disabled.state_dict().keys()
    for key, value in rgaa.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])


def test_zero_residual_scale_retains_parameters_but_blocks_their_policy_gradient():
    actor = build_method_actors(
        method_variant=DBM_RGAA_METHOD, training_seed=5, dbm_residual_scale=0.0,
    ).actors[1]
    observations = torch.randn(8, 100)
    details = actor.mode_diagnostics(observations)
    assert torch.equal(details["final_mean"], details["base_mean"])
    actor._mean(observations).sum().backward()
    for parameter in [*actor.network.router.parameters(), *actor.network.experts.parameters()]:
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_dbm_and_wide_preserve_main_torch_rng_relative_to_rgaa():
    states = []
    for method in ("rgaa", DBM_RGAA_METHOD, RGAA_WIDE_METHOD):
        torch.manual_seed(333)
        build_method_actors(method_variant=method, training_seed=333)
        states.append(torch.get_rng_state().clone())
    assert torch.equal(states[0], states[1])
    assert torch.equal(states[0], states[2])


def test_complete_trainer_construction_preserves_global_torch_rng_state():
    states = []
    for method in ("rgaa", DBM_RGAA_METHOD, RGAA_WIDE_METHOD):
        trainer = HAPPOTrainer(short_v39(), trainer_config(method=method))
        try:
            states.append(torch.get_rng_state().clone())
        finally:
            trainer.close()
    assert torch.equal(states[0], states[1])
    assert torch.equal(states[0], states[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dbm_and_wide_initialization_do_not_advance_cuda_rng_relative_to_rgaa():
    states = []
    for method in ("rgaa", DBM_RGAA_METHOD, RGAA_WIDE_METHOD):
        torch.manual_seed(444)
        build_method_actors(method_variant=method, training_seed=444)
        states.append(torch.cuda.get_rng_state_all())
    for device_index in range(len(states[0])):
        assert torch.equal(states[0][device_index], states[1][device_index])
        assert torch.equal(states[0][device_index], states[2][device_index])


@pytest.mark.parametrize("method", [DBM_RGAA_METHOD, RGAA_WIDE_METHOD])
def test_method_trainer_preserves_mav_team_and_role_critic_initialization(method):
    baseline = HAPPOTrainer(short_v39(), trainer_config(method="rgaa"))
    candidate = HAPPOTrainer(short_v39(), trainer_config(method=method))
    try:
        _assert_base_equal(baseline.actors.actors[0], candidate.actors.actors[0])
        for key, value in baseline.critic.state_dict().items():
            assert torch.equal(value, candidate.critic.state_dict()[key])
        for left, right in (
            (baseline.mav_role_critic, candidate.mav_role_critic),
            (baseline.uav_role_critic, candidate.uav_role_critic),
        ):
            for key, value in left.state_dict().items():
                assert torch.equal(value, right.state_dict()[key])
    finally:
        baseline.close(); candidate.close()


def test_dbm_update_uses_rgaa_buffer_and_produces_finite_diagnostics():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert isinstance(trainer.buffer, RoleAdvantageRolloutBuffer)
        _, metrics = trainer.train_update()
        assert trainer.env_steps == 2
        assert torch.equal(
            trainer.last_rgaa_combined_advantages,
            trainer.last_rgaa_team_normalized_advantages
            + 0.5 * trainer.last_rgaa_role_normalized_advantages,
        )
        assert torch.equal(
            trainer.last_rgaa_ppo_normalized_advantages,
            trainer.last_rgaa_combined_advantages,
        )
        assert len(trainer.last_rgaa_factor_history) == len(RED_IDS) + 1
        for aid in RED_IDS[1:]:
            assert np.isfinite(metrics[f"dbm_router_entropy_{aid}"])
            assert np.isfinite(metrics[f"dbm_expert_divergence_{aid}"])
            assert metrics[f"dbm_active_sample_count_{aid}"] > 0
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    finally:
        trainer.close()


def test_dbm_ppo_update_changes_dbm_parameters_and_not_unupdated_actor_parameters():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        before = [deepcopy(actor.state_dict()) for actor in trainer.actors.actors]
        trainer.update()
        changed = []
        for agent, actor in enumerate(trainer.actors.actors):
            changed.append(any(
                not torch.equal(value, actor.state_dict()[key])
                for key, value in before[agent].items()
            ))
        assert changed == [True, True, True, True]

        actor_two_before = deepcopy(trainer.actors.actors[2].state_dict())
        actor_one = trainer.actors.actors[1]
        optimizer = trainer.actor_optimizers[1]
        observation = torch.randn(4, 100)
        optimizer.zero_grad()
        actor_one._mean(observation).square().mean().backward()
        optimizer.step()
        for key, value in actor_two_before.items():
            assert torch.equal(value, trainer.actors.actors[2].state_dict()[key])
    finally:
        trainer.close()


def test_role_module_disabled_matches_rgaa_for_two_complete_updates():
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method="rgaa"))
    disabled = HAPPOTrainer(
        short_v39(), trainer_config(role_module_enabled=False),
    )
    try:
        for _ in range(2):
            before = torch.get_rng_state().clone()
            rgaa.train_update()
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            disabled.train_update()
            assert torch.equal(torch.get_rng_state(), after)
        for key, value in rgaa.actors.state_dict().items():
            assert torch.equal(value, disabled.actors.state_dict()[key])
        for key, value in rgaa.critic.state_dict().items():
            assert torch.equal(value, disabled.critic.state_dict()[key])
        for key, value in rgaa.mav_role_critic.state_dict().items():
            assert torch.equal(value, disabled.mav_role_critic.state_dict()[key])
        for key, value in rgaa.uav_role_critic.state_dict().items():
            assert torch.equal(value, disabled.uav_role_critic.state_dict()[key])
    finally:
        rgaa.close(); disabled.close()


def test_dbm_diagnostics_honor_active_masks_and_episode_boundaries():
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=1)
    observations = torch.randn(3, 2, 4, 100)
    active = torch.ones(3, 2, 4)
    active[1:, 0, 1] = 0.0
    terminated = torch.zeros(3, 2, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[0, 1] = True
    metrics = dbm_rollout_diagnostics(actors, observations, active, terminated, truncated)
    assert metrics["dbm_active_sample_count_UAV1"] == 4
    # env0 contributes no pairs after UAV death; env1 transition t0 is an episode boundary.
    assert metrics["dbm_valid_switch_pairs_UAV1"] == 1


def test_dbm_diagnostics_are_parameter_and_rng_read_only_and_zero_pair_safe():
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=12)
    before_parameters = deepcopy(actors.state_dict())
    before_rng = torch.get_rng_state().clone()
    observations = torch.zeros(1, 1, 4, 100)
    active = torch.ones(1, 1, 4)
    boundary = torch.zeros(1, 1, dtype=torch.bool)
    metrics = dbm_rollout_diagnostics(actors, observations, active, boundary, boundary)
    assert torch.equal(before_rng, torch.get_rng_state())
    for key, value in before_parameters.items():
        assert torch.equal(value, actors.state_dict()[key])
    for aid in RED_IDS[1:]:
        assert metrics[f"dbm_valid_switch_pairs_{aid}"] == 0
        assert metrics[f"dbm_mode_switch_rate_{aid}"] == 0.0
        assert metrics[f"dbm_router_l1_movement_{aid}"] == 0.0


def test_dbm_diagnostic_formulas_match_direct_tensor_calculation():
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=14)
    actor = actors.actors[1]
    with torch.no_grad():
        actor.network.router.weight.normal_(0.0, 0.05)
        actor.network.experts[1].weight.add_(0.03)
    observations = torch.randn(3, 1, 4, 100)
    active = torch.ones(3, 1, 4)
    boundary = torch.zeros(3, 1, dtype=torch.bool)
    metrics = dbm_rollout_diagnostics(actors, observations, active, boundary, boundary)
    details = actor.mode_diagnostics(observations[:, :, 1])
    probabilities = details["router_probabilities"].reshape(-1, 2)
    hard = probabilities.argmax(-1)
    residual_norm = details["scaled_residual"].reshape(-1, 3).norm(dim=-1)
    base_norm = details["base_mean"].reshape(-1, 3).norm(dim=-1)
    saturation = (
        torch.tanh(details["final_mean"]).reshape(-1, 3).abs().amax(-1) > 0.95
    ).float().mean()
    assert metrics["dbm_soft_occupancy_mode1_UAV1"] == pytest.approx(probabilities[:, 0].mean().item())
    assert metrics["dbm_hard_occupancy_mode1_UAV1"] == pytest.approx((hard == 0).float().mean().item())
    assert metrics["dbm_router_variance_mode1_UAV1"] == pytest.approx(probabilities[:, 0].var(unbiased=False).item())
    assert metrics["dbm_scaled_residual_magnitude_UAV1"] == pytest.approx(residual_norm.mean().item())
    assert metrics["dbm_residual_base_ratio_UAV1"] == pytest.approx(
        (residual_norm.mean() / (base_norm.mean() + 1e-8)).item()
    )
    assert metrics["dbm_deterministic_action_saturation_UAV1"] == pytest.approx(saturation.item())


def test_dbm_metadata_parameter_counts_and_algorithm_identity():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert _algorithm_name("vanilla", DBM_RGAA_METHOD, "mlp") == "dbm_rgaa_happo"
        assert trainer.dbm_metadata["dbm_initialization_semantics"] == DBM_INITIALIZATION_SEMANTICS
        assert trainer.dbm_metadata["dbm_diagnostics_source"] == (
            "post_update_policy_on_collected_rollout_v1"
        )
        state = trainer.checkpoint_state()
        assert state["algorithm"] == "dbm_rgaa_happo"
        assert state["actor_parameter_counts"]["total"] == 122416
    finally:
        trainer.close()


def test_wide_metadata_parameter_counts_and_algorithm_identity():
    trainer = HAPPOTrainer(short_v39(), trainer_config(method=RGAA_WIDE_METHOD))
    try:
        assert _algorithm_name("vanilla", RGAA_WIDE_METHOD, "mlp") == "rgaa_wide_happo"
        assert trainer.rgaa_wide_metadata["uav_actor_hidden_dim"] == 131
        assert trainer.checkpoint_state()["actor_parameter_counts"]["total"] == 122596
    finally:
        trainer.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("role_module_enabled", False), ("dbm_role_count", 3),
        ("dbm_residual_scale", 0.5), ("dbm_init_scale", 0.02),
        ("dbm_initialization_semantics", "wrong"),
    ],
)
def test_dbm_checkpoint_rejects_contract_mismatch(tmp_path, field, value):
    source = HAPPOTrainer(short_v39(), trainer_config())
    path = tmp_path / "dbm.pt"
    try:
        source.save_checkpoint(path)
    finally:
        source.close()
    target_config = trainer_config(**{field: value})
    if field in ("dbm_role_count", "dbm_initialization_semantics"):
        with pytest.raises(ValueError):
            HAPPOTrainer(short_v39(), target_config)
        return
    target = HAPPOTrainer(short_v39(), target_config)
    try:
        with pytest.raises(RuntimeError):
            target.load_checkpoint(path)
    finally:
        target.close()


@pytest.mark.parametrize("target_method", ["rgaa", RGAA_WIDE_METHOD])
def test_dbm_checkpoint_rejects_cross_method_resume(tmp_path, target_method):
    source = HAPPOTrainer(short_v39(), trainer_config())
    path = tmp_path / "dbm.pt"
    try:
        source.save_checkpoint(path)
    finally:
        source.close()
    target = HAPPOTrainer(short_v39(), trainer_config(method=target_method))
    try:
        with pytest.raises(RuntimeError, match="method mismatch|actor architecture"):
            target.load_checkpoint(path)
    finally:
        target.close()


def test_dbm_checkpoint_exact_state_resume(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    path = tmp_path / "dbm.pt"
    try:
        source.train_update()
        source.save_checkpoint(path)
        expected = source.checkpoint_state()
    finally:
        source.close()
    resumed = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert resumed.load_checkpoint(path) == 2
        actual = resumed.checkpoint_state()
        assert actual["sampled_steps"] == expected["sampled_steps"]
        assert actual["trainer_numpy_rng"] == expected["trainer_numpy_rng"]
        assert actual["rgaa_numpy_rng"] == expected["rgaa_numpy_rng"]
        for key, value in expected["actors"].items():
            assert torch.equal(value, actual["actors"][key])
        for key, value in expected["critic"].items():
            assert torch.equal(value, actual["critic"][key])
    finally:
        resumed.close()


def test_dbm_cpu_exact_continuation_after_resume(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    path = tmp_path / "dbm.pt"
    try:
        source.train_update()
        source.save_checkpoint(path)
        _, reference_metrics = source.train_update()
        reference_actors = deepcopy(source.actors.state_dict())
        reference_critic = deepcopy(source.critic.state_dict())
        reference_mav_role = deepcopy(source.mav_role_critic.state_dict())
        reference_uav_role = deepcopy(source.uav_role_critic.state_dict())
        reference_actor_optimizers = deepcopy([
            optimizer.state_dict() for optimizer in source.actor_optimizers
        ])
    finally:
        source.close()
    resumed = HAPPOTrainer(short_v39(), trainer_config())
    try:
        resumed.load_checkpoint(path)
        _, resumed_metrics = resumed.train_update()
        assert resumed_metrics["agent_update_order"] == reference_metrics["agent_update_order"]
        for key, value in reference_metrics.items():
            if isinstance(value, float):
                assert resumed_metrics[key] == pytest.approx(value, rel=0.0, abs=0.0)
        for key, value in reference_actors.items():
            assert torch.equal(value, resumed.actors.state_dict()[key])
        for key, value in reference_critic.items():
            assert torch.equal(value, resumed.critic.state_dict()[key])
        for key, value in reference_mav_role.items():
            assert torch.equal(value, resumed.mav_role_critic.state_dict()[key])
        for key, value in reference_uav_role.items():
            assert torch.equal(value, resumed.uav_role_critic.state_dict()[key])
        resumed_optimizer_states = [optimizer.state_dict() for optimizer in resumed.actor_optimizers]
        for expected_optimizer, actual_optimizer in zip(
            reference_actor_optimizers, resumed_optimizer_states,
        ):
            assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
            for parameter_id, expected_state in expected_optimizer["state"].items():
                actual_state = actual_optimizer["state"][parameter_id]
                for field, value in expected_state.items():
                    if torch.is_tensor(value):
                        assert torch.equal(value, actual_state[field])
                    else:
                        assert value == actual_state[field]
    finally:
        resumed.close()


def test_wide_checkpoint_full_save_and_resume(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config(method=RGAA_WIDE_METHOD))
    path = tmp_path / "wide.pt"
    try:
        source.train_update()
        source.save_checkpoint(path)
        expected = deepcopy(source.actors.state_dict())
    finally:
        source.close()
    resumed = HAPPOTrainer(short_v39(), trainer_config(method=RGAA_WIDE_METHOD))
    try:
        assert resumed.load_checkpoint(path) == 2
        for key, value in expected.items():
            assert torch.equal(value, resumed.actors.state_dict()[key])
    finally:
        resumed.close()


def test_rgaa_checkpoint_cannot_resume_as_dbm(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config(method="rgaa"))
    path = tmp_path / "rgaa.pt"
    try:
        source.save_checkpoint(path)
    finally:
        source.close()
    target = HAPPOTrainer(short_v39(), trainer_config())
    try:
        with pytest.raises(RuntimeError):
            target.load_checkpoint(path)
    finally:
        target.close()


@pytest.mark.parametrize("method", [DBM_RGAA_METHOD, RGAA_WIDE_METHOD])
def test_standalone_evaluator_reconstructs_actor_from_checkpoint_metadata(tmp_path, method):
    trainer = HAPPOTrainer(short_v39(), trainer_config(method=method))
    checkpoint = tmp_path / f"{method}.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run(
        [
            sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
            "--action-mode", "stochastic", "--action-seed", "2000",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / f"evaluation_{method}_stochastic_summary.json").exists()


@pytest.mark.parametrize("method", [DBM_RGAA_METHOD, RGAA_WIDE_METHOD])
def test_dbm_and_wide_require_identical_resolved_evaluation_environment(method):
    trainer = HAPPOTrainer(short_v39(), trainer_config(method=method))
    try:
        payload = trainer.checkpoint_state()
        validate_checkpoint_contract(payload, deepcopy(trainer.environment_config))
        changed = deepcopy(trainer.environment_config)
        changed["simulation"]["max_decision_steps"] += 1
        with pytest.raises(
            RuntimeError,
            match="evaluation resolved environment config differs from checkpoint",
        ):
            validate_checkpoint_contract(payload, changed)
        missing = deepcopy(payload)
        missing.pop("environment_config")
        with pytest.raises(RuntimeError, match="missing resolved environment_config"):
            validate_checkpoint_contract(missing, deepcopy(trainer.environment_config))
    finally:
        trainer.close()


def test_environment_rejection_precedes_rollout_and_output_creation(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "checkpoint_final.pt"
    try:
        trainer.save_checkpoint(checkpoint)
        changed = deepcopy(trainer.environment_config)
    finally:
        trainer.close()
    changed["simulation"]["max_decision_steps"] += 1
    changed_path = tmp_path / "changed_env.yaml"
    changed_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
            "--env-config", str(changed_path), "--action-mode", "stochastic",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "evaluation resolved environment config differs from checkpoint" in result.stderr
    assert not (tmp_path / "evaluation_final_stochastic.csv").exists()
    assert not (tmp_path / "evaluation_final_stochastic_summary.json").exists()


def test_tiny_training_outputs_complete_stochastic_evaluation_protocol(tmp_path, monkeypatch):
    env = short_v39()
    env["simulation"]["max_decision_steps"] = 1
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(train_entrypoint, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_dbm_rgaa.py", "--steps", "1", "--profile", "learnability",
        "--seed", "3", "--device", "cpu", "--num-envs", "1",
        "--config", str(ROOT / "configs/happo_dbm_rgaa_v39.yaml"),
        "--env-config", str(env_path), "--output-name", "tiny_dbm_protocol",
        "--checkpoint-interval", "0", "--eval-interval", "0",
        "--log-interval", "0", "--final-eval-episodes", "1",
        "--eval-action-mode", "stochastic", "--eval-action-seed", "2000",
    ])
    train_entrypoint.main(
        actor_variant="vanilla", method_variant=DBM_RGAA_METHOD, critic_variant="mlp",
    )
    run_dir = tmp_path / "tiny_dbm_protocol"
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "evaluations.csv").open(encoding="utf-8", newline="") as stream:
        row = list(csv.DictReader(stream))[0]
    assert resolved["evaluation_action_mode"] == "stochastic"
    assert resolved["evaluation_configured_action_seed"] == 2000
    assert resolved["evaluation_effective_action_seed"] == 2000
    assert resolved["evaluation_environment_seed_start"] == 1000
    assert resolved["dbm_diagnostics_source"] == "post_update_policy_on_collected_rollout_v1"
    assert summary["evaluation_action_mode"] == "stochastic"
    assert summary["final_evaluations"][0]["action_mode"] == "stochastic"
    assert summary["final_evaluations"][0]["effective_action_seed"] == 2000
    assert summary["dbm_diagnostics_source"] == "post_update_policy_on_collected_rollout_v1"
    assert row["training_seed"] == "3"
    assert row["evaluation_environment_seed_start"] == "1000"
    assert row["evaluation_episodes"] == "1"
    assert row["action_mode"] == "stochastic"
    assert row["configured_action_seed"] == row["effective_action_seed"] == "2000"
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Evaluation action mode: stochastic" in run_log
    assert "DBM diagnostics source: post_update_policy_on_collected_rollout_v1" in run_log


def test_resume_rejects_legacy_evaluation_csv_schema(tmp_path):
    path = tmp_path / "evaluations.csv"
    path.write_text("sampled_steps,red_win_rate\n1,0.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing evaluation CSV schema"):
        train_entrypoint._validate_existing_evaluation_schema(path)


def test_dbm_and_wide_configs_are_single_variable_extensions_of_rgaa():
    with (ROOT / "configs/happo_rgaa_v39.yaml").open() as stream:
        base = yaml.safe_load(stream)["training"]
    with (ROOT / "configs/happo_dbm_rgaa_v39.yaml").open() as stream:
        dbm = yaml.safe_load(stream)["training"]
    with (ROOT / "configs/happo_rgaa_wide_v39.yaml").open() as stream:
        wide = yaml.safe_load(stream)["training"]
    for key, value in base.items():
        if key == "method_variant":
            continue
        assert dbm.get(key, value) == value
        assert wide.get(key, value) == value
    assert dbm["method_variant"] == DBM_RGAA_METHOD
    assert wide["method_variant"] == RGAA_WIDE_METHOD


@pytest.mark.parametrize("script", ["algorithm/train_dbm_rgaa.py", "algorithm/train_rgaa_wide.py"])
def test_new_entrypoints_expose_training_cli(script):
    result = subprocess.run(
        [sys.executable, script, "--help"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    )
    assert "--steps" in result.stdout
