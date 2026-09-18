"""Focused reward decomposition, credit, checkpoint, and evaluation tests for RDC-HAPPO."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys
import uuid
import yaml

import numpy as np
import pytest
import torch

from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.counterfactual_credit import (
    CF_METHOD, RDC_METHOD, RDC_COMPONENT_WEIGHTS,
    combine_rdc_component_credit, component_weights, extract_credit_components,
)
from algorithm.happo.trainer import HAPPOTrainer, preceding_factor_update
from env.mavuav import load_environment_config


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "configs" / "env_v39.yaml"


def short_v39():
    config = deepcopy(load_environment_config(V39))
    config["simulation"]["max_decision_steps"] = 2
    return config


def config(method: str, device: str = "cpu") -> dict:
    return {
        "method_variant": method, "actor_variant": "vanilla", "critic_variant": "mlp",
        "num_envs": 1, "rollout_steps": 1, "ppo_epochs": 1, "minibatch_size": 1,
        "hidden_dim": 8, "seed": 29, "device": device, "environment_profile": "learnability",
    }


def test_rdc_reward_extraction_order_weights_and_exact_identity():
    info = {
        "event_reward": 90.0, "terminal_reward": 100.0, "safety_reward": -2.0,
        "mav_process_reward": .2, "uav1_process_reward": -.4,
        "uav2_process_reward": .6, "uav3_process_reward": 1.0,
    }
    components = np.asarray([[188.0, .2, -.4, .6, 1.0]], np.float32)
    team = components @ RDC_COMPONENT_WEIGHTS
    rewards = np.repeat(team[:, None], 4, axis=1)
    extracted = extract_credit_components([info], rewards, RDC_METHOD)
    assert np.array_equal(extracted, components)
    assert np.array_equal(component_weights(RDC_METHOD), np.asarray([1, .25, .25, .25, .25], np.float32))
    bad = rewards.copy(); bad[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="does not reconstruct"):
        extract_credit_components([info], bad, RDC_METHOD)


def test_required_synthetic_uav3_credit_is_exactly_four():
    actual = torch.tensor([[10.0, 4.0, 8.0, 0.0, 12.0]])
    counterfactual = torch.tensor([[8.0, 4.0, 8.0, 0.0, 4.0]])
    assert combine_rdc_component_credit(actual, counterfactual).item() == 4.0


def test_cross_role_mav_credit_is_included_for_uav1():
    actual = torch.tensor([[10.0, 8.0, 4.0, 4.0, 4.0]])
    counterfactual = torch.tensor([[10.0, 0.0, 4.0, 4.0, 4.0]])
    assert combine_rdc_component_credit(actual, counterfactual).item() == 2.0


def test_agent_specific_advantages_are_distinct_independently_normalized_and_detached():
    trainer = HAPPOTrainer(short_v39(), {**config(RDC_METHOD), "rollout_steps": 3, "minibatch_size": 3})
    try:
        trainer.collect_rollout()
        observations = torch.as_tensor(trainer.buffer.observations.reshape(-1, 4, 100))
        actions = torch.as_tensor(trainer.buffer.actions.reshape(-1, 4, 3))
        states = torch.as_tensor(trainer.buffer.global_states.reshape(-1, 117))
        counterfactual = trainer._sample_counterfactual_joint_actions(observations, actions)
        advantages, credits = trainer._counterfactual_actor_advantages(states, actions, counterfactual)
        assert advantages.shape == (3, 4) and credits.shape == (3, 4, 5)
        assert not advantages.requires_grad and not credits.requires_grad
        assert any(not torch.equal(advantages[:, 0], advantages[:, agent]) for agent in range(1, 4))
        for agent in range(4):
            active = torch.as_tensor(trainer.buffer.active_masks.reshape(-1, 4)[:, agent]) > .5
            normalized = (advantages[:, agent] - advantages[active, agent].mean()) / advantages[active, agent].std(unbiased=False).clamp_min(1e-8)
            assert torch.isfinite(normalized).all()
    finally:
        trainer.close()


def test_preceding_factor_function_is_reused_exactly():
    factor = torch.tensor([1.0, 2.0])
    old = torch.tensor([-.3, .4])
    new = torch.tensor([-.1, .7])
    active = torch.tensor([1.0, 0.0])
    expected = factor * torch.tensor([np.exp(.2), 1.0], dtype=factor.dtype)
    assert torch.allclose(preceding_factor_update(factor, old, new, active), expected)


@pytest.mark.parametrize("method", [CF_METHOD, RDC_METHOD])
def test_credit_checkpoint_metadata_same_method_resume_and_exact_parameters(tmp_path, method):
    env = short_v39(); settings = config(method)
    source = HAPPOTrainer(env, settings)
    restored = HAPPOTrainer(env, settings)
    try:
        source.train_update()
        checkpoint = tmp_path / f"{method}.pt"
        source.save_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_method = "counterfactual_team" if method == CF_METHOD else "role_decomposed_counterfactual"
        assert payload["credit_method"] == expected_method
        assert payload["counterfactual_samples_per_agent"] == 1
        assert "credit_critic" in payload and "credit_critic_optimizer_state" in payload
        validate_checkpoint_contract(payload, source.environment_config)
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        for first, second in zip(source.credit_critic.parameters(), restored.credit_critic.parameters()):
            assert torch.equal(first, second)
        assert restored.credit_critic_optimizer.state_dict()["state"].keys() == source.credit_critic_optimizer.state_dict()["state"].keys()
    finally:
        source.close(); restored.close()


@pytest.mark.parametrize("source_method,target_method", [
    ("baseline", CF_METHOD), ("baseline", RDC_METHOD), (CF_METHOD, RDC_METHOD),
    (RDC_METHOD, CF_METHOD), ("agp", RDC_METHOD),
])
def test_cross_method_resume_is_rejected(tmp_path, source_method, target_method):
    env = short_v39()
    source = HAPPOTrainer(env, config(source_method))
    target = HAPPOTrainer(env, config(target_method))
    try:
        checkpoint = tmp_path / "cross.pt"
        source.save_checkpoint(checkpoint)
        with pytest.raises(RuntimeError, match="resume method mismatch"):
            target.load_checkpoint(checkpoint)
    finally:
        source.close(); target.close()


def test_rdc_metrics_include_all_component_diagnostics_and_random_permutation():
    trainer = HAPPOTrainer(short_v39(), {**config(RDC_METHOD), "rollout_steps": 4, "minibatch_size": 4})
    try:
        _, metrics = trainer.train_update()
        assert sorted(metrics["agent_update_order"]) == [0, 1, 2, 3]
        for name in ("shared", "mav_role", "uav1_role", "uav2_role", "uav3_role"):
            assert np.isfinite(metrics[f"rdc_{name}_credit_mean_abs"])
    finally:
        trainer.close()


@pytest.mark.parametrize("method", [CF_METHOD, RDC_METHOD])
def test_vanilla_evaluator_accepts_credit_checkpoint_actor_only(tmp_path, method):
    trainer = HAPPOTrainer(short_v39(), config(method))
    try:
        checkpoint = tmp_path / f"{method}.pt"
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
    assert (tmp_path / f"evaluation_{method}_summary.json").is_file()


@pytest.mark.parametrize("method,script,config_name", [
    (CF_METHOD, "algorithm/train_happo_cf.py", "configs/happo_cf_v39.yaml"),
    (RDC_METHOD, "algorithm/train_happo_rdc.py", "configs/happo_rdc_v39.yaml"),
])
def test_credit_entrypoint_records_resolved_csv_and_summary_contract(method, script, config_name):
    output_name = f"pytest_{method}_{uuid.uuid4().hex}"
    run_dir = ROOT / "outputs" / output_name
    try:
        result = subprocess.run(
            [
                sys.executable, script, "--steps", "1", "--profile", "learnability",
                "--seed", "3", "--device", "cpu", "--num-envs", "1",
                "--config", config_name, "--env-config", "configs/env_v39.yaml",
                "--output-name", output_name, "--checkpoint-interval", "1",
                "--eval-interval", "0", "--log-interval", "1",
                "--final-eval-episodes", "1",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        with (run_dir / "training.csv").open(newline="") as stream:
            row = list(csv.DictReader(stream))[-1]
        assert resolved["algorithm"] == resolved["method_variant"] == method
        assert summary["algorithm"] == summary["method_variant"] == method
        assert resolved["credit_method"] == summary["credit_method"]
        assert row["method_variant"] == method
        assert row["credit_total_loss"] != ""
        if method == RDC_METHOD:
            assert row["rdc_shared_credit_mean_abs"] != ""
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
