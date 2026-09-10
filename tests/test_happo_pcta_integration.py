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
            sys.executable, "algorithm/train_happo_pcta.py", "--steps", "6",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "2",
            "--final-eval-episodes", "1", "--resume", str(run_dir / "checkpoint_final.pt"),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            assert int(list(csv.DictReader(stream))[-1]["sampled_steps"]) == 6
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
