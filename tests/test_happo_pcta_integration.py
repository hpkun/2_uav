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

from algorithm.happo import HAPPOTrainer
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.trainer import preceding_factor_update
from algorithm.modules.pcta import PCTAIndependentActors
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def short_env(steps: int = 3):
    config = deepcopy(load_environment_config(None)); config["simulation"]["max_decision_steps"] = steps
    return config


def config(**updates):
    return {
        "actor_variant": "pcta", "method_variant": "baseline", "critic_variant": "mlp",
        "num_envs": 2, "rollout_steps": 3, "ppo_epochs": 1, "minibatch_size": 6,
        "hidden_dim": 16, "pcta_context_dim": 12, "pcta_enemy_dim": 8,
        "pcta_hidden_dim": 16, "pcta_consistency_coef": 0.05,
        "seed": 31, "environment_profile": "learnability", **updates,
    }


def test_pcta_update_is_finite_and_preceding_factor_uses_post_auxiliary_actor():
    trainer = HAPPOTrainer(short_env(), config())
    trainer.collect_rollout()
    old = torch.as_tensor(trainer.buffer.log_probs.copy(), device=trainer.device)
    obs = torch.as_tensor(trainer.buffer.observations.copy(), device=trainer.device)
    active = torch.as_tensor(trainer.buffer.active_masks.copy(), device=trainer.device)
    metrics = trainer.update()
    factor = torch.ones_like(old[..., 0])
    for agent in metrics["agent_update_order"]:
        with torch.no_grad():
            new, _ = trainer.actors.actors[agent].evaluate_actions(
                obs[:, :, agent].reshape(-1, OBS_DIM),
                torch.as_tensor(trainer.buffer.actions[:, :, agent], device=trainer.device).reshape(-1, 3),
            )
        factor = preceding_factor_update(
            factor.reshape(-1), old[:, :, agent].reshape(-1), new,
            active[:, :, agent].reshape(-1),
        ).reshape_as(factor)
    np.testing.assert_allclose(
        trainer.last_pcta_factor_history[-1], factor.reshape(-1).cpu().numpy(), rtol=1e-5, atol=1e-6,
    )
    assert metrics["pcta_valid_temporal_pairs"] > 0
    for field in (
        "pcta_consistency_loss", "pcta_consistency_weighted_loss", "pcta_attention_entropy",
        "pcta_target_switch_rate", "critic_loss", "entropy",
    ):
        assert np.isfinite(metrics[field])
    trainer.close()


@pytest.mark.parametrize(
    ("variant", "attention_mode", "expected_steps_per_actor"),
    [
        ("pcta", "learned", 2),
        ("pcta_attention_only", "learned", 1),
        ("pcta_uniform", "uniform", 1),
    ],
)
def test_pcta_family_optimizer_step_and_diagnostic_contract(
    variant, attention_mode, expected_steps_per_actor,
):
    trainer = HAPPOTrainer(short_env(), config(actor_variant=variant))
    assert trainer.pcta_attention_mode == attention_mode
    assert trainer.config["pcta_consistency_coef"] == (0.05 if variant == "pcta" else 0.0)
    trainer.collect_rollout()
    step_counts = [0 for _ in RED_IDS]
    for index, optimizer in enumerate(trainer.actor_optimizers):
        original_step = optimizer.step

        def counted_step(*args, _index=index, _step=original_step, **kwargs):
            step_counts[_index] += 1
            return _step(*args, **kwargs)

        optimizer.step = counted_step
    metrics = trainer.update()
    assert step_counts == [expected_steps_per_actor] * len(RED_IDS)
    assert metrics["pcta_valid_temporal_pairs"] > 0
    assert metrics["pcta_consistency_weighted_loss"] == (
        pytest.approx(0.05 * metrics["pcta_consistency_loss"]) if variant == "pcta" else 0.0
    )
    for field in (
        "pcta_consistency_loss", "pcta_consistency_weighted_loss", "pcta_attention_entropy",
        "pcta_target_switch_rate",
    ):
        assert np.isfinite(metrics[field])
    trainer.close()


def test_all_pcta_variants_have_identical_parameter_contract_and_initialization():
    trainers = [
        HAPPOTrainer(short_env(), config(actor_variant=variant))
        for variant in ("pcta", "pcta_attention_only", "pcta_uniform")
    ]
    states = [trainer.actors.state_dict() for trainer in trainers]
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(
        states[0][key].shape == states[index][key].shape
        and torch.equal(states[0][key], states[index][key])
        for index in (1, 2) for key in states[0]
    )
    assert trainers[0].actor_parameter_counts == trainers[1].actor_parameter_counts == trainers[2].actor_parameter_counts
    observations = torch.randn(5, OBS_DIM)
    full, _ = trainers[0].actors.actors[0].sample(observations, deterministic=True)
    attention_only, _ = trainers[1].actors.actors[0].sample(observations, deterministic=True)
    assert torch.equal(full, attention_only)
    for trainer in trainers:
        trainer.close()


def test_pcta_checkpoint_round_trip_and_cross_variant_rejection(tmp_path):
    source = HAPPOTrainer(short_env(), config())
    source.train_update(); checkpoint = tmp_path / "pcta.pt"; source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["actor_variant"] == "pcta" and payload["pcta_consistency_coef"] == 0.05
    assert payload["actor_architecture"] == source.actor_architecture
    restored = HAPPOTrainer(short_env(), config())
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    assert all(torch.equal(restored.actors.state_dict()[key], value) for key, value in source.actors.state_dict().items())
    vanilla = HAPPOTrainer(short_env(), config(actor_variant="vanilla"))
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        vanilla.load_checkpoint(checkpoint)
    vanilla_checkpoint = tmp_path / "vanilla.pt"; vanilla.save_checkpoint(vanilla_checkpoint)
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        restored.load_checkpoint(vanilla_checkpoint)
    source.close(); restored.close(); vanilla.close()


def test_legacy_full_pcta_checkpoint_without_attention_metadata_still_loads(tmp_path):
    source = HAPPOTrainer(short_env(), config())
    source.train_update()
    payload = source.checkpoint_state()
    payload.pop("attention_mode")
    payload.pop("effective_pcta_consistency_coef")
    payload["actor_architecture"].pop("attention_mode")
    checkpoint = tmp_path / "legacy_full_pcta.pt"
    torch.save(payload, checkpoint)
    restored = HAPPOTrainer(short_env(), config())
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    source.close(); restored.close()


@pytest.mark.parametrize(
    ("variant", "attention_mode", "coef"),
    [
        ("pcta", "learned", 0.05),
        ("pcta_attention_only", "learned", 0.0),
        ("pcta_uniform", "uniform", 0.0),
    ],
)
def test_pcta_family_checkpoint_exact_continuation_and_metadata(
    tmp_path, variant, attention_mode, coef,
):
    source = HAPPOTrainer(short_env(), config(actor_variant=variant))
    source.train_update()
    checkpoint = tmp_path / f"{variant}.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["actor_variant"] == variant
    assert payload["attention_mode"] == payload["actor_architecture"]["attention_mode"] == attention_mode
    assert payload["pcta_consistency_coef"] == payload["effective_pcta_consistency_coef"] == coef

    source_episodes, source_metrics = source.train_update()
    source_state = source.checkpoint_state()
    restored = HAPPOTrainer(short_env(), config(actor_variant=variant))
    assert restored.load_checkpoint(checkpoint) == int(payload["sampled_steps"])
    restored_episodes, restored_metrics = restored.train_update()
    restored_state = restored.checkpoint_state()
    assert source_episodes == restored_episodes
    for key in source_metrics:
        if isinstance(source_metrics[key], float):
            assert restored_metrics[key] == pytest.approx(source_metrics[key], rel=0.0, abs=0.0)
        else:
            assert restored_metrics[key] == source_metrics[key]
    for key, value in source_state["actors"].items():
        assert torch.equal(value, restored_state["actors"][key])
    for key, value in source_state["critic"].items():
        assert torch.equal(value, restored_state["critic"][key])
    source.close(); restored.close()


@pytest.mark.parametrize("source_variant", ["pcta", "pcta_attention_only", "pcta_uniform"])
@pytest.mark.parametrize("target_variant", ["pcta", "pcta_attention_only", "pcta_uniform"])
def test_pcta_family_checkpoint_rejects_variant_mismatch(tmp_path, source_variant, target_variant):
    if source_variant == target_variant:
        return
    source = HAPPOTrainer(short_env(), config(actor_variant=source_variant))
    checkpoint = tmp_path / f"{source_variant}.pt"
    source.save_checkpoint(checkpoint)
    target = HAPPOTrainer(short_env(), config(actor_variant=target_variant))
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        target.load_checkpoint(checkpoint)
    source.close(); target.close()


def test_pcta_deterministic_evaluation_runs():
    trainer = HAPPOTrainer(short_env(2), config(num_envs=1, rollout_steps=2))
    records = evaluate_actors(trainer.actors, trainer.environment_config, 1, "learnability", seed=1000, device=trainer.device)
    assert len(records) == 1 and records[0]["episode_length"] <= 2
    trainer.close()


def test_pcta_entrypoint_writes_diagnostics_and_resumes(tmp_path):
    del tmp_path
    output_name = f"pytest_pcta_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        subprocess.run([
            sys.executable, "algorithm/train_happo_pcta.py", "--steps", "4",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--output-name", output_name, "--checkpoint-interval", "4",
            "--eval-interval", "0", "--log-interval", "2", "--final-eval-episodes", "1",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["algorithm"] == "pcta_happo" and summary["actor_variant"] == "pcta"
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {
            "pcta_consistency_loss", "pcta_consistency_weighted_loss",
            "pcta_valid_temporal_pairs", "pcta_attention_entropy", "pcta_target_switch_rate",
        }
        assert rows and required <= rows[0].keys()
        assert int(rows[-1]["pcta_valid_temporal_pairs"]) > 0
        subprocess.run([
            sys.executable, "algorithm/evaluate_happo_pcta.py", str(run_dir / "checkpoint_final.pt"),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        subprocess.run([
            sys.executable, "algorithm/train_happo_pcta.py", "--steps", "6",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "2",
            "--final-eval-episodes", "1", "--resume", str(run_dir / "checkpoint_final.pt"),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            assert int(list(csv.DictReader(stream))[-1]["sampled_steps"]) == 6
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("script", "variant", "algorithm", "attention_mode"),
    [
        ("train_happo_pcta_attention_only.py", "pcta_attention_only", "pcta_attention_only_happo", "learned"),
        ("train_happo_pcta_uniform.py", "pcta_uniform", "pcta_uniform_happo", "uniform"),
    ],
)
def test_pcta_ablation_entrypoint_metadata_and_diagnostics(script, variant, algorithm, attention_mode):
    output_name = f"pytest_{variant}_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        subprocess.run([
            sys.executable, f"algorithm/{script}", "--steps", "2",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--output-name", output_name, "--checkpoint-interval", "2",
            "--eval-interval", "0", "--log-interval", "2", "--final-eval-episodes", "1",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = __import__("yaml").safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        for metadata in (summary, resolved, checkpoint):
            assert metadata["actor_variant"] == variant
            assert metadata["attention_mode"] == attention_mode
            assert metadata["effective_pcta_consistency_coef"] == 0.0
        assert summary["algorithm"] == resolved["algorithm"] == algorithm
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and float(rows[-1]["pcta_consistency_weighted_loss"]) == 0.0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
