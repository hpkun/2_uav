from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo import HAPPOTrainer
from algorithm.modules.pcta_v2 import (
    OWN_ATTACK_STREAK_INDEX,
    PCTAv2Actor,
    PCTAv2IndependentActors,
    target_behavior_diagnostics,
)
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENEMY_STARTS = (44, 58, 72, 86)


def observations(*shape: int) -> torch.Tensor:
    torch.manual_seed(9041)
    values = torch.randn(*shape, OBS_DIM) * 0.15
    for start in ENEMY_STARTS:
        values[..., start + 9] = 1.0
        values[..., start + 10] = 1.0
        values[..., start + 11] = 0.0
        values[..., start + OWN_ATTACK_STREAK_INDEX] = 0.0
    return values


def short_env(steps: int = 3):
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = steps
    return config


def trainer_config(**updates):
    return {
        "actor_variant": "pcta_v2", "method_variant": "baseline", "critic_variant": "mlp",
        "num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 2,
        "hidden_dim": 16, "pcta_hidden_dim": 16,
        "pcta_v2_context_dim": 12, "pcta_v2_enemy_dim": 8,
        "pcta_v2_target_dim": 6, "pcta_v2_attention_heads": 4,
        "actor_log_std_init": -0.25, "seed": 37,
        "environment_profile": "learnability", **updates,
    }


def test_actor_shapes_full_observation_fusion_and_alpha_mean():
    actor = PCTAv2Actor()
    obs = observations(7)
    policy_feature, details = actor.encode(obs)
    actions, sampled_log_prob = actor.sample(obs)
    evaluated_log_prob, entropy = actor.evaluate_actions(obs, actions)
    assert actions.shape == (7, 3)
    assert sampled_log_prob.shape == evaluated_log_prob.shape == entropy.shape == (7,)
    assert policy_feature.shape == (7, 136)
    assert details["target_feature"].shape == (7, 32)
    assert details["enemy_attention_heads"].shape == (7, 4, 4)
    torch.testing.assert_close(policy_feature[..., :OBS_DIM], obs, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        details["enemy_attention"], details["enemy_attention_heads"].mean(dim=-2),
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(sampled_log_prob, evaluated_log_prob, rtol=2e-5, atol=2e-5)
    assert torch.isfinite(actions).all() and torch.isfinite(policy_feature).all()


def test_each_additive_head_normalizes_valid_targets_and_masks_invalid_targets():
    actor = PCTAv2Actor()
    obs = observations(5)
    obs[..., ENEMY_STARTS[0] + 9] = 0.0
    obs[..., ENEMY_STARTS[1] + 10] = obs[..., ENEMY_STARTS[1] + 11] = 0.0
    _, details = actor.encode(obs)
    heads = details["enemy_attention_heads"]
    assert torch.equal(heads[..., :2], torch.zeros_like(heads[..., :2]))
    torch.testing.assert_close(heads.sum(dim=-1), torch.ones(5, 4), rtol=0.0, atol=1e-6)


def test_no_valid_enemy_produces_zero_attention_and_finite_policy():
    actor = PCTAv2Actor()
    obs = observations(4)
    for start in ENEMY_STARTS:
        obs[..., start + 9] = 0.0
        obs[..., start + 10] = obs[..., start + 11] = 0.0
    policy_feature, details = actor.encode(obs)
    actions, log_prob = actor.sample(obs)
    assert torch.equal(details["enemy_attention_heads"], torch.zeros(4, 4, 4))
    assert torch.equal(details["enemy_attention"], torch.zeros(4, 4))
    assert torch.isfinite(policy_feature).all()
    assert torch.isfinite(actions).all() and torch.isfinite(log_prob).all()


def test_pursuit_bias_is_nonnegative_and_favors_higher_streak_when_base_scores_are_zero():
    actor = PCTAv2Actor()
    with torch.no_grad():
        for modules in (actor.query_heads, actor.key_heads, actor.score_heads):
            for layer in modules:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
    obs = observations(1)
    for start in ENEMY_STARTS[2:]:
        obs[..., start + 9] = 0.0
        obs[..., start + 10] = obs[..., start + 11] = 0.0
    obs[..., ENEMY_STARTS[0] + OWN_ATTACK_STREAK_INDEX] = 0.0
    obs[..., ENEMY_STARTS[1] + OWN_ATTACK_STREAK_INDEX] = 1.0
    _, details = actor.encode(obs)
    assert torch.all(details["pursuit_bias_gain"] >= 0.0)
    assert torch.all(details["enemy_attention_heads"][..., 1] > details["enemy_attention_heads"][..., 0])


def test_target_diagnostics_are_finite_and_do_not_create_gradients():
    actor = PCTAv2Actor()
    obs = observations(4, 2)
    result = target_behavior_diagnostics(
        actor, obs, torch.zeros(4, 2), torch.zeros(4, 2), torch.ones(4, 2),
    )
    assert result.valid_pairs == 6
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(np.isfinite(value) for value in (
        result.attention_entropy_sum, result.max_attention_weight_sum, result.pursuit_bias_mean,
    ))
    assert 0 <= result.target_switches <= result.valid_pairs


def _synthetic_target_diagnostics(heads, valid_count):
    actor = PCTAv2Actor()
    obs = observations(1, 1)
    for index, start in enumerate(ENEMY_STARTS):
        obs[..., start + 9] = float(index < valid_count)
        obs[..., start + 10] = float(index < valid_count)
        obs[..., start + 11] = 0.0
    head_tensor = torch.tensor(heads, dtype=torch.float32).reshape(1, 1, 4, 4)
    valid = torch.zeros(1, 1, 4, dtype=torch.bool)
    valid[..., :valid_count] = True

    def fake_encode(_observations):
        alpha_mean = head_tensor.mean(dim=-2)
        return torch.zeros(1, 1, 136), {
            "enemy_attention": alpha_mean,
            "enemy_attention_heads": head_tensor,
        }

    actor.encode = fake_encode
    result = target_behavior_diagnostics(
        actor, obs, torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1),
    )
    return result


def test_synthetic_uniform_heads_have_normalized_entropy_one_and_zero_disagreement():
    result = _synthetic_target_diagnostics([[0.25, 0.25, 0.25, 0.25]] * 4, 4)
    assert result.valid_target_states == result.multi_target_states == 1
    assert result.head_normalized_entropy_count == result.head_max_attention_count == 4
    assert result.head_disagreement_count == 1
    assert result.head_normalized_entropy_sum / result.head_normalized_entropy_count == pytest.approx(1.0)
    assert result.head_max_attention_sum / result.head_max_attention_count == pytest.approx(0.25)
    assert result.head_disagreement_sum == pytest.approx(0.0)


def test_synthetic_identical_sharp_heads_have_zero_entropy_and_zero_disagreement():
    result = _synthetic_target_diagnostics([[1.0, 0.0, 0.0, 0.0]] * 4, 4)
    assert result.head_normalized_entropy_sum == pytest.approx(0.0)
    assert result.head_max_attention_sum / result.head_max_attention_count == pytest.approx(1.0)
    assert result.head_disagreement_sum == pytest.approx(0.0)


def test_synthetic_specialized_heads_have_zero_per_head_entropy_but_high_disagreement():
    result = _synthetic_target_diagnostics(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 4,
    )
    assert result.head_normalized_entropy_sum == pytest.approx(0.0)
    assert result.head_max_attention_sum / result.head_max_attention_count == pytest.approx(1.0)
    assert result.head_disagreement_sum == pytest.approx(1.0)


def test_synthetic_two_valid_targets_are_normalized_by_log_two():
    result = _synthetic_target_diagnostics([[0.5, 0.5, 0.0, 0.0]] * 4, 2)
    assert result.valid_target_states == 1 and result.multi_target_states == 1
    assert result.head_normalized_entropy_sum / result.head_normalized_entropy_count == pytest.approx(1.0)


def test_synthetic_one_or_zero_valid_targets_have_safe_counts_and_means():
    one = _synthetic_target_diagnostics([[1.0, 0.0, 0.0, 0.0]] * 4, 1)
    zero = _synthetic_target_diagnostics([[0.0, 0.0, 0.0, 0.0]] * 4, 0)
    assert one.valid_target_states == 1 and one.multi_target_states == 0
    assert one.head_normalized_entropy_count == 0
    assert one.head_max_attention_sum / one.head_max_attention_count == pytest.approx(1.0)
    assert zero.valid_target_states == zero.multi_target_states == 0
    assert zero.head_normalized_entropy_count == zero.head_max_attention_count == zero.head_disagreement_count == 0
    assert all(np.isfinite(value) for value in (zero.head_normalized_entropy_sum, zero.head_disagreement_sum))


def test_target_diagnostics_preserve_parameters_grads_rng_and_training_state():
    torch.manual_seed(812)
    actor = PCTAv2Actor()
    actor.train()
    diagnostic_observations = observations(3, 1)
    parameters = [parameter.detach().clone() for parameter in actor.parameters()]
    gradients = [parameter.grad for parameter in actor.parameters()]
    rng_before = torch.get_rng_state().clone()
    target_behavior_diagnostics(
        actor, diagnostic_observations, torch.zeros(3, 1), torch.zeros(3, 1), torch.ones(3, 1),
    )
    assert actor.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(before, after) for before, after in zip(parameters, actor.parameters()))
    assert all(before is after for before, after in zip(gradients, (parameter.grad for parameter in actor.parameters())))


def test_v2_trainer_has_one_optimizer_step_and_never_calls_legacy_consistency(monkeypatch):
    import algorithm.happo.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module, "pursuit_consistency",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy consistency called")),
    )
    trainer = HAPPOTrainer(short_env(), trainer_config())
    trainer.collect_rollout()
    step_counts = [0 for _ in RED_IDS]
    for index, optimizer in enumerate(trainer.actor_optimizers):
        original_step = optimizer.step

        def counted_step(*args, _index=index, _step=original_step, **kwargs):
            step_counts[_index] += 1
            return _step(*args, **kwargs)

        optimizer.step = counted_step
    metrics = trainer.update()
    assert step_counts == [1] * len(RED_IDS)
    assert "pcta_consistency_loss" not in metrics
    for field in (
        "pcta_v2_attention_entropy", "pcta_v2_target_switch_rate",
        "pcta_v2_pursuit_bias_mean", "pcta_v2_max_attention_weight",
        "pcta_v2_ensemble_attention_entropy", "pcta_v2_ensemble_max_attention_weight",
        "pcta_v2_head_normalized_entropy", "pcta_v2_head_max_attention_weight",
        "pcta_v2_head_disagreement", "pcta_v2_valid_target_states",
        "pcta_v2_multi_target_states",
    ):
        assert np.isfinite(metrics[field])
    assert metrics["pcta_v2_valid_temporal_pairs"] > 0
    assert metrics["pcta_v2_ensemble_attention_entropy"] == metrics["pcta_v2_attention_entropy"]
    assert metrics["pcta_v2_ensemble_max_attention_weight"] == metrics["pcta_v2_max_attention_weight"]
    assert 0.0 <= metrics["pcta_v2_head_normalized_entropy"] <= 1.0
    assert 0.0 <= metrics["pcta_v2_head_disagreement"] <= 1.0
    assert metrics["pcta_v2_valid_target_states"] >= metrics["pcta_v2_multi_target_states"] >= 0
    trainer.close()


def test_legacy_consistency_coefficient_cannot_change_v2_update():
    first = HAPPOTrainer(short_env(), trainer_config(pcta_consistency_coef=0.0))
    first_episodes, first_metrics = first.train_update()
    first_state = first.checkpoint_state()
    first.close()
    second = HAPPOTrainer(short_env(), trainer_config(pcta_consistency_coef=91.0))
    second_episodes, second_metrics = second.train_update()
    second_state = second.checkpoint_state()
    second.close()
    assert first_episodes == second_episodes
    for key in first_metrics:
        if key == "agent_update_order":
            assert first_metrics[key] == second_metrics[key]
        elif isinstance(first_metrics[key], float):
            assert first_metrics[key] == pytest.approx(second_metrics[key], rel=0.0, abs=0.0)
    for section in ("actors", "critic"):
        assert all(torch.equal(value, second_state[section][key]) for key, value in first_state[section].items())


def test_parameter_count_and_log_std_initialization_contract():
    actors = PCTAv2IndependentActors(log_std_init=-0.25)
    assert [sum(parameter.numel() for parameter in actor.parameters()) for actor in actors.actors] == [54602] * 4
    for actor in actors.actors:
        torch.testing.assert_close(actor.log_std, torch.full((3,), -0.25), rtol=0.0, atol=0.0)


def test_checkpoint_round_trip_metadata_and_cross_pcta_rejection(tmp_path):
    source = HAPPOTrainer(short_env(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "pcta_v2.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["actor_variant"] == "pcta_v2"
    assert payload["pcta_v2_auxiliary_consistency"] is False
    assert payload["pcta_v2_attention_heads"] == 4
    assert payload["actor_architecture"] == source.actor_architecture
    restored = HAPPOTrainer(short_env(), trainer_config())
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    legacy = HAPPOTrainer(short_env(), trainer_config(actor_variant="pcta"))
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        legacy.load_checkpoint(checkpoint)
    legacy_checkpoint = tmp_path / "legacy_pcta.pt"
    legacy.save_checkpoint(legacy_checkpoint)
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        restored.load_checkpoint(legacy_checkpoint)
    source.close()
    restored.close()
    legacy.close()


def test_entrypoint_writes_v2_resolved_summary_diagnostics_and_checkpoint():
    output_name = f"pytest_pcta_v2_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        subprocess.run([
            sys.executable, "algorithm/train_happo_pcta_v2.py", "--steps", "2",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--config", "configs/happo_pcta_v2_final.yaml",
            "--output-name", output_name, "--checkpoint-interval", "2",
            "--eval-interval", "0", "--log-interval", "2", "--final-eval-episodes", "1",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        for metadata in (summary, resolved, checkpoint):
            assert metadata["actor_variant"] == "pcta_v2"
            assert metadata["pcta_v2_auxiliary_consistency"] is False
            assert metadata["pcta_v2_attention_heads"] == 4
            assert metadata["pcta_v2_context_dim"] == 64
            assert metadata["pcta_v2_enemy_dim"] == 32
            assert metadata["pcta_v2_target_dim"] == 32
        assert summary["algorithm"] == resolved["algorithm"] == "pcta_v2_happo"
        assert summary["actor_log_std_init"] == resolved["actor_log_std_init"] == -0.25
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {
            "pcta_v2_attention_entropy", "pcta_v2_target_switch_rate",
            "pcta_v2_valid_temporal_pairs", "pcta_v2_pursuit_bias_mean",
            "pcta_v2_max_attention_weight", "pcta_v2_ensemble_attention_entropy",
            "pcta_v2_ensemble_max_attention_weight", "pcta_v2_head_normalized_entropy",
            "pcta_v2_head_max_attention_weight", "pcta_v2_head_disagreement",
            "pcta_v2_valid_target_states", "pcta_v2_multi_target_states",
        }
        assert rows and required <= rows[0].keys()
        assert all(np.isfinite(float(rows[-1][field])) for field in required)
        assert float(rows[-1]["pcta_v2_attention_entropy"]) == float(rows[-1]["pcta_v2_ensemble_attention_entropy"])
        assert float(rows[-1]["pcta_v2_max_attention_weight"]) == float(rows[-1]["pcta_v2_ensemble_max_attention_weight"])
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
