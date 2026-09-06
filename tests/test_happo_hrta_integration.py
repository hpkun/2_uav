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

from algorithm.happo import HAPPOTrainer, IndependentActors
from algorithm.modules.hrta import HRTAIndependentActors
from env.mavuav import RED_IDS, load_environment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(variant: str, **overrides):
    return {
        "actor_variant": variant, "num_envs": 1, "rollout_steps": 2,
        "ppo_epochs": 1, "minibatch_size": 2, "hidden_dim": 16,
        "hrta_entity_dim": 8, "hrta_role_dim": 4, "hrta_fusion_hidden_dim": 12,
        "seed": 17, "environment_profile": "learnability", **overrides,
    }


def _short_env():
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = 3
    return config


def test_default_trainer_remains_vanilla_and_hrta_actors_are_independent():
    vanilla = HAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8})
    assert vanilla.config["actor_variant"] == "vanilla"
    assert isinstance(vanilla.actors, IndependentActors)
    vanilla.close()

    hrta = HAPPOTrainer(_short_env(), _config("hrta"))
    assert isinstance(hrta.actors, HRTAIndependentActors)
    parameter_ids = [{id(parameter) for parameter in actor.parameters()} for actor in hrta.actors.actors]
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j]) for i in range(len(RED_IDS)) for j in range(i + 1, len(RED_IDS)))
    hrta.close()


def test_hrta_short_cpu_rollout_and_unchanged_happo_update_are_finite():
    trainer = HAPPOTrainer(_short_env(), _config("hrta", num_envs=2, minibatch_size=4))
    before = [[parameter.detach().clone() for parameter in actor.parameters()] for actor in trainer.actors.actors]
    episodes = trainer.collect_rollout()
    metrics = trainer.update()
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    assert all(any(not torch.equal(a, b) for a, b in zip(snapshot, actor.parameters())) for snapshot, actor in zip(before, trainer.actors.actors))
    assert trainer.env_steps == 4 and trainer.buffer.position == 2
    assert all(record["episode_length"] <= 3 for record in episodes)
    trainer.close()


def test_checkpoint_architecture_round_trip_and_incompatibilities(tmp_path):
    hrta_config = _config("hrta")
    source = HAPPOTrainer(_short_env(), hrta_config)
    source.train_update()
    checkpoint = tmp_path / "hrta.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["actor_variant"] == "hrta"
    assert payload["actor_architecture"] == {
        "entity_dim": 8, "role_dim": 4, "fusion_hidden_dim": 12, "action_dim": 3,
    }
    restored = HAPPOTrainer(_short_env(), hrta_config)
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    assert all(
        torch.equal(restored.actors.state_dict()[key], value)
        for key, value in source.actors.state_dict().items()
    )
    assert all(
        torch.equal(restored.critic.state_dict()[key], value)
        for key, value in source.critic.state_dict().items()
    )
    assert np.array_equal(restored.observations, source.observations)
    assert np.array_equal(restored.global_states, source.global_states)
    assert np.array_equal(restored.active_masks, source.active_masks)
    assert np.array_equal(restored.vector_env.reset_counts, source.vector_env.reset_counts)
    assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    assert restored.rng.bit_generator.state == source.rng.bit_generator.state
    assert torch.equal(torch.get_rng_state(), payload["torch_rng"])
    assert [optimizer.state_dict()["state"].keys() for optimizer in restored.actor_optimizers] == [
        optimizer.state_dict()["state"].keys() for optimizer in source.actor_optimizers
    ]
    assert restored.critic_optimizer.state_dict()["state"].keys() == source.critic_optimizer.state_dict()["state"].keys()

    vanilla = HAPPOTrainer(_short_env(), _config("vanilla"))
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        vanilla.load_checkpoint(checkpoint)
    vanilla_checkpoint = tmp_path / "vanilla.pt"
    vanilla.save_checkpoint(vanilla_checkpoint)
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        restored.load_checkpoint(vanilla_checkpoint)

    changed = HAPPOTrainer(_short_env(), _config("hrta", hrta_entity_dim=9))
    with pytest.raises(RuntimeError, match="incompatible actor architecture"):
        changed.load_checkpoint(checkpoint)
    source.close(); restored.close(); vanilla.close(); changed.close()


def test_legacy_vanilla_checkpoint_without_actor_metadata_is_compatible(tmp_path):
    config = _config("vanilla")
    source = HAPPOTrainer(_short_env(), config)
    checkpoint = tmp_path / "legacy.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("actor_variant"); payload.pop("actor_architecture")
    payload["trainer_config"].pop("actor_variant")
    payload["trainer_config"].pop("hrta_entity_dim")
    payload["trainer_config"].pop("hrta_role_dim")
    payload["trainer_config"].pop("hrta_fusion_hidden_dim")
    torch.save(payload, checkpoint)
    restored = HAPPOTrainer(_short_env(), config)
    assert restored.load_checkpoint(checkpoint) == 0
    source.close(); restored.close()


def test_hrta_direct_training_resume_evaluation_and_attention_output_smoke():
    output_name = f"pytest_happo_hrta_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    attention_path = run_dir / "attention.csv"
    try:
        subprocess.run([
            sys.executable, "algorithm/train_happo_hrta.py", "--steps", "2",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--output-name", output_name, "--checkpoint-interval", "2",
            "--eval-interval", "0", "--log-interval", "1", "--final-eval-episodes", "1",
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        assert summary["algorithm"] == "happo_hrta" and summary["actor_variant"] == "hrta"
        assert resolved["algorithm"] == "happo_hrta" and resolved["actor_variant"] == "hrta"
        assert resolved["actor_parameter_count_total"] == sum(resolved["actor_parameter_count_per_agent"])

        subprocess.run([
            sys.executable, "algorithm/train_happo_hrta.py", "--steps", "4",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "1",
            "--final-eval-episodes", "1", "--resume", str(run_dir / "checkpoint_final.pt"),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)

        subprocess.run([
            sys.executable, "algorithm/evaluate_happo_hrta.py", str(run_dir / "checkpoint_final.pt"),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
            "--blue-mode", "nearest", "--attention-output", str(attention_path),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180)
        evaluation = json.loads((run_dir / "evaluation_hrta_final_summary.json").read_text(encoding="utf-8"))
        assert evaluation["algorithm"] == "happo_hrta"
        with attention_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {
            "episode", "decision_step", "blue_mode", "agent", "outcome",
            "friend_attention_friend1", "friend_attention_friend2", "friend_attention_friend3",
            "enemy_attention_Blue1", "enemy_attention_Blue2", "enemy_attention_Blue3", "enemy_attention_Blue4",
            "Blue1_alive", "Blue2_alive", "Blue3_alive", "Blue4_alive",
            "Blue1_direct_or_datalink_visible", "Blue2_direct_or_datalink_visible",
            "Blue3_direct_or_datalink_visible", "Blue4_direct_or_datalink_visible",
        }
        assert rows and required <= rows[0].keys()
        for row in rows:
            weights = np.asarray([float(row[f"enemy_attention_Blue{i}"]) for i in range(1, 5)])
            eligible = np.asarray([
                int(row[f"Blue{i}_alive"]) and int(row[f"Blue{i}_direct_or_datalink_visible"])
                for i in range(1, 5)
            ], dtype=bool)
            assert np.isfinite(weights).all()
            if eligible.any():
                assert np.isclose(weights.sum(), 1.0, atol=1e-6)
                assert np.equal(weights[~eligible], 0.0).all()
            else:
                assert np.equal(weights, 0.0).all()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
