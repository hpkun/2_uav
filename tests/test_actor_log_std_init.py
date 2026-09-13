import json
import math
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.happo import HAPPOTrainer
from env.mavuav import load_environment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def short_env():
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(actor_variant="vanilla", **updates):
    return {
        "actor_variant": actor_variant,
        "method_variant": "baseline",
        "critic_variant": "mlp",
        "num_envs": 1,
        "rollout_steps": 1,
        "hidden_dim": 16,
        "pcta_context_dim": 12,
        "pcta_enemy_dim": 8,
        "pcta_hidden_dim": 16,
        "pcta_consistency_coef": 0.05,
        "seed": 29,
        "environment_profile": "learnability",
        **updates,
    }


@pytest.mark.parametrize("actor_variant", ["vanilla", "pcta"])
def test_default_actor_log_std_init_preserves_minus_half(actor_variant):
    trainer = HAPPOTrainer(short_env(), trainer_config(actor_variant))
    assert trainer.config["actor_log_std_init"] == -0.5
    for actor in trainer.actors.actors:
        torch.testing.assert_close(actor.log_std, torch.full((3,), -0.5), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actor.log_std.exp(), torch.full((3,), math.exp(-0.5)), rtol=1e-7, atol=0.0,
        )
    trainer.close()


@pytest.mark.parametrize("actor_variant", ["vanilla", "pcta", "pcta_attention_only", "pcta_uniform"])
def test_configured_actor_log_std_init_reaches_vanilla_and_pcta_family(actor_variant):
    trainer = HAPPOTrainer(
        short_env(),
        trainer_config(actor_variant, actor_log_std_init=-0.25),
    )
    for actor in trainer.actors.actors:
        torch.testing.assert_close(actor.log_std, torch.full((3,), -0.25), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actor.log_std.exp(), torch.full((3,), math.exp(-0.25)), rtol=1e-7, atol=0.0,
        )
    trainer.close()


@pytest.mark.parametrize("actor_variant", ["vanilla", "pcta"])
def test_log_std_screen_is_the_only_initial_parameter_difference(actor_variant):
    baseline = HAPPOTrainer(
        short_env(),
        trainer_config(actor_variant, actor_log_std_init=-0.5),
    )
    screened = HAPPOTrainer(
        short_env(),
        trainer_config(actor_variant, actor_log_std_init=-0.25),
    )
    baseline_actors = baseline.actors.state_dict()
    screened_actors = screened.actors.state_dict()
    assert baseline_actors.keys() == screened_actors.keys()
    for name in baseline_actors:
        if name.endswith("log_std"):
            torch.testing.assert_close(
                baseline_actors[name], torch.full((3,), -0.5), rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                screened_actors[name], torch.full((3,), -0.25), rtol=0.0, atol=0.0,
            )
        else:
            assert torch.equal(baseline_actors[name], screened_actors[name]), name
    baseline_critic = baseline.critic.state_dict()
    screened_critic = screened.critic.state_dict()
    assert baseline_critic.keys() == screened_critic.keys()
    assert all(torch.equal(baseline_critic[name], screened_critic[name]) for name in baseline_critic)
    baseline.close()
    screened.close()


def test_log_std_init_checkpoint_metadata_resume_and_legacy_default(tmp_path):
    configured = HAPPOTrainer(
        short_env(),
        trainer_config("pcta", actor_log_std_init=-0.25),
    )
    configured_checkpoint = tmp_path / "configured.pt"
    configured.save_checkpoint(configured_checkpoint)
    configured_payload = torch.load(configured_checkpoint, map_location="cpu", weights_only=False)
    assert configured_payload["trainer_config"]["actor_log_std_init"] == -0.25

    default = HAPPOTrainer(short_env(), trainer_config("pcta"))
    legacy_payload = default.checkpoint_state()
    legacy_payload["trainer_config"].pop("actor_log_std_init")
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(legacy_payload, legacy_checkpoint)

    restored_default = HAPPOTrainer(short_env(), trainer_config("pcta"))
    assert restored_default.load_checkpoint(legacy_checkpoint) == 0
    restored_screen = HAPPOTrainer(
        short_env(),
        trainer_config("pcta", actor_log_std_init=-0.25),
    )
    with pytest.raises(RuntimeError, match="resume config mismatch: actor_log_std_init"):
        restored_screen.load_checkpoint(legacy_checkpoint)

    for trainer in (configured, default, restored_default, restored_screen):
        trainer.close()


def test_log_std_screen_config_diff_and_entrypoint_metadata():
    base_path = PROJECT_ROOT / "configs" / "happo_entropy001_screen.yaml"
    screen_path = PROJECT_ROOT / "configs" / "happo_entropy001_logstd025_screen.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))["training"]
    screen = yaml.safe_load(screen_path.read_text(encoding="utf-8"))["training"]
    assert screen["actor_log_std_init"] == -0.25
    assert {key: value for key, value in screen.items() if key != "actor_log_std_init"} == base

    output_name = f"pytest_logstd025_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        subprocess.run(
            [
                sys.executable,
                "algorithm/train_happo.py",
                "--config",
                str(screen_path),
                "--steps",
                "1",
                "--profile",
                "learnability",
                "--device",
                "cpu",
                "--num-envs",
                "1",
                "--output-name",
                output_name,
                "--checkpoint-interval",
                "0",
                "--eval-interval",
                "0",
                "--log-interval",
                "0",
                "--final-eval-episodes",
                "1",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        assert summary["actor_log_std_init"] == -0.25
        assert resolved["actor_log_std_init"] == -0.25
        assert resolved["happo"]["actor_log_std_init"] == -0.25
        assert checkpoint["trainer_config"]["actor_log_std_init"] == -0.25
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
