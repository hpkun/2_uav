from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo import HAPPOTrainer
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from env.mavuav import load_environment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config(variant: str, **overrides):
    return {
        "actor_variant": variant, "num_envs": 1, "rollout_steps": 2,
        "ppo_epochs": 1, "minibatch_size": 2, "hidden_dim": 16,
        "hrta_entity_dim": 8, "hrta_role_dim": 4, "hrta_fusion_hidden_dim": 12,
        "seed": 29, "environment_profile": "learnability", **overrides,
    }


def short_env():
    value = deepcopy(load_environment_config(None))
    value["simulation"]["max_decision_steps"] = 3
    return value


def test_structured_uniform_short_update_changes_independent_actors_and_critic():
    trainer = HAPPOTrainer(short_env(), config("structured_uniform", num_envs=2, minibatch_size=4))
    assert isinstance(trainer.actors, StructuredUniformIndependentActors)
    actor_before = [[p.detach().clone() for p in actor.parameters()] for actor in trainer.actors.actors]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    trainer.collect_rollout()
    metrics = trainer.update()
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    assert all(any(not torch.equal(a, b) for a, b in zip(old, actor.parameters())) for old, actor in zip(actor_before, trainer.actors.actors))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic.parameters()))
    assert trainer.env_steps == 4
    trainer.close()


def test_structured_uniform_checkpoint_round_trip_resume_and_cross_variant_rejection(tmp_path):
    env = short_env()
    source = HAPPOTrainer(env, config("structured_uniform"))
    source.train_update()
    checkpoint = tmp_path / "structured_uniform.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format"] == "happo_training_checkpoint_v1"
    assert payload["actor_variant"] == "structured_uniform"
    assert payload["actor_architecture"] == {
        "entity_dim": 8, "role_dim": 4, "fusion_hidden_dim": 12, "action_dim": 3,
    }
    restored = HAPPOTrainer(env, config("structured_uniform"))
    assert restored.load_checkpoint(checkpoint) == source.env_steps == 2
    assert all(torch.equal(restored.actors.state_dict()[k], v) for k, v in source.actors.state_dict().items())
    assert all(torch.equal(restored.critic.state_dict()[k], v) for k, v in source.critic.state_dict().items())
    assert np.array_equal(restored.observations, source.observations)
    assert np.array_equal(restored.global_states, source.global_states)
    assert np.array_equal(restored.active_masks, source.active_masks)
    assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    assert restored.rng.bit_generator.state == source.rng.bit_generator.state
    assert [o.state_dict()["state"].keys() for o in restored.actor_optimizers] == [o.state_dict()["state"].keys() for o in source.actor_optimizers]
    restored.train_update()
    assert restored.env_steps == 4

    hrta = HAPPOTrainer(env, config("hrta"))
    vanilla = HAPPOTrainer(env, config("vanilla"))
    hrta_checkpoint, vanilla_checkpoint = tmp_path / "hrta.pt", tmp_path / "vanilla.pt"
    hrta.save_checkpoint(hrta_checkpoint)
    vanilla.save_checkpoint(vanilla_checkpoint)
    for target, foreign in (
        (restored, hrta_checkpoint), (restored, vanilla_checkpoint),
        (hrta, checkpoint), (vanilla, checkpoint),
    ):
        with pytest.raises(RuntimeError, match="incompatible actor architecture"):
            target.load_checkpoint(foreign)
    source.close(); restored.close(); hrta.close(); vanilla.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_structured_uniform_real_cuda_checkpoint_resume_and_update(tmp_path):
    env = short_env()
    cuda_config = config("structured_uniform", device="cuda", rollout_steps=1, minibatch_size=1)
    source = HAPPOTrainer(env, cuda_config)
    source.train_update()
    checkpoint = tmp_path / "structured_uniform_cuda.pt"
    source.save_checkpoint(checkpoint)
    restored = HAPPOTrainer(env, cuda_config)
    assert restored.load_checkpoint(checkpoint) == 1
    metrics = restored.train_update()[1]
    assert restored.env_steps == 2
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    source.close(); restored.close()


def test_structured_uniform_training_and_evaluation_entrypoints():
    output_name = f"pytest_happo_structured_uniform_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        subprocess.run([
            sys.executable, "algorithm/train_happo_structured_uniform.py", "--steps", "2",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--output-name", output_name, "--checkpoint-interval", "2",
            "--eval-interval", "0", "--log-interval", "1", "--final-eval-episodes", "1",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=240)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        payload = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        for data in (summary, resolved, payload):
            assert data["actor_variant"] == "structured_uniform"
            assert data["actor_architecture"] == {
                "entity_dim": 32, "role_dim": 16, "fusion_hidden_dim": 64, "action_dim": 3,
            }
        assert summary["algorithm"] == resolved["algorithm"] == "happo_structured_uniform"
        assert summary["actor_parameter_count_total"] == sum(summary["actor_parameter_count_per_agent"])

        subprocess.run([
            sys.executable, "algorithm/train_happo_structured_uniform.py", "--steps", "4",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "1",
            "--final-eval-episodes", "1", "--resume", str(run_dir / "checkpoint_final.pt"),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=240)
        assert torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)["sampled_steps"] == 4

        for mode, expected_rows in (("nearest", 1), ("mav_priority", 1), ("both", 2)):
            subprocess.run([
                sys.executable, "algorithm/evaluate_happo_structured_uniform.py",
                str(run_dir / "checkpoint_final.pt"), "--profile", "learnability",
                "--episodes", "1", "--device", "cpu", "--blue-mode", mode,
            ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=240)
            evaluation = json.loads((run_dir / "evaluation_structured_uniform_final_summary.json").read_text(encoding="utf-8"))
            assert evaluation["algorithm"] == "happo_structured_uniform"
            assert evaluation["actor_variant"] == "structured_uniform"
            assert len(evaluation["results"]) == expected_rows
            with (run_dir / "evaluation_structured_uniform_final.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == expected_rows
            assert all(row["algorithm"] == "happo_structured_uniform" and row["actor_variant"] == "structured_uniform" for row in rows)

        for evaluator in ("algorithm/evaluate_happo.py", "algorithm/evaluate_happo_hrta.py"):
            rejected = subprocess.run([
                sys.executable, evaluator, str(run_dir / "checkpoint_final.pt"),
                "--episodes", "1", "--device", "cpu", "--blue-mode", "nearest",
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
            assert rejected.returncode != 0 and "incompatible actor architecture" in rejected.stderr
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_structured_uniform_evaluator_rejects_vanilla_and_hrta(tmp_path):
    env = short_env()
    checkpoints = []
    trainers = []
    for variant in ("vanilla", "hrta"):
        trainer = HAPPOTrainer(env, config(variant))
        path = tmp_path / f"{variant}.pt"
        trainer.save_checkpoint(path)
        trainers.append(trainer); checkpoints.append(path)
    for checkpoint in checkpoints:
        rejected = subprocess.run([
            sys.executable, "algorithm/evaluate_happo_structured_uniform.py", str(checkpoint),
            "--episodes", "1", "--device", "cpu", "--blue-mode", "nearest",
        ], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
        assert rejected.returncode != 0
        assert "structured_uniform evaluator requires a structured_uniform checkpoint" in rejected.stderr
    for trainer in trainers:
        trainer.close()
