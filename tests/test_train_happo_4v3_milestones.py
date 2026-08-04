from __future__ import annotations

from pathlib import Path

import yaml

from scripts.train_happo_4v3 import rollout_lengths_to_milestone, training_throughput
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer


ENV_CONFIG = "configs/heterogeneous_4v3_main_v9.yaml"
TRAIN_CONFIG = "configs/happo_heterogeneous_4v3_main_v9.yaml"


def _tiny_config() -> dict:
    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text(encoding="utf-8"))
    cfg["experiment"].update({"device": "cpu", "seed": 712})
    cfg["training"].update({
        "num_envs": 2,
        "num_env_workers": 0,
        "rollout_steps": 4,
        "total_env_steps": 8,
        "ppo_epochs": 1,
        "minibatch_size": 4,
    })
    return cfg


def test_8192_milestone_uses_four_full_rollouts() -> None:
    assert rollout_lengths_to_milestone(0, 8192, 8, 256) == [256, 256, 256, 256]


def test_100000_milestone_uses_48_full_and_one_212_step_partial_rollout() -> None:
    lengths = rollout_lengths_to_milestone(0, 100000, 8, 256)
    assert len(lengths) == 49
    assert lengths[:48] == [256] * 48
    assert lengths[-1] == 212
    assert sum(lengths) * 8 == 100000


def test_3m_milestone_terminates_exactly_without_overshoot() -> None:
    lengths = rollout_lengths_to_milestone(0, 3_000_000, 8, 256)
    assert sum(lengths) * 8 == 3_000_000
    assert all(1 <= length <= 256 for length in lengths)
    assert len(lengths) == 1465


def test_each_rollout_length_represents_one_update() -> None:
    lengths = rollout_lengths_to_milestone(0, 100000, 8, 256)
    assert len(lengths) == 49
    assert sum(lengths) * 8 == 100000


def test_rollout_milestone_rejects_unreachable_env_step_target() -> None:
    try:
        rollout_lengths_to_milestone(0, 8193, 8, 256)
    except ValueError as exc:
        assert "whole vector steps" in str(exc)
    else:
        raise AssertionError("unreachable target must be rejected")


def test_resume_throughput_uses_only_new_env_steps() -> None:
    assert training_throughput(1200, 1000, 10.0) == 20.0
    assert training_throughput(1000, 1000, 10.0) == 0.0


def test_checkpoint_roundtrip_preserves_current_episode_seeds(tmp_path: Path) -> None:
    cfg = _tiny_config()
    trainer = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    checkpoint = tmp_path / "seed_state.pt"
    try:
        trainer.current_episode_seeds = [901, 902]
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()

    restored = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    try:
        restored.load_checkpoint(checkpoint)
        assert restored.current_episode_seeds == [901, 902]
    finally:
        restored.close()
