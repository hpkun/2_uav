from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.diagnostics import (
    TrainingDiagnostics, draw_return_exceeds_red_win, fixed_evaluation_seeds, load_calibration_checkpoint,
    reward_diagnostic_rows, run_rule_baselines, save_calibration_checkpoint, target_proxies,
    train_to_sampled_steps,
)
from algorithm.mappo import MAPPOTrainer
from algorithm.common.buffer import RolloutBuffer
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from tools.diagnostics import OBSERVATION_GROUPS


def tiny_trainer(seed=1, device="cpu", profile="main"):
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = 3
    return MAPPOTrainer(config, {"num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 2, "hidden_dim": 16, "seed": seed, "device": device, "environment_profile": profile})


def test_fixed_evaluation_seeds_are_identical_across_checkpoints():
    assert fixed_evaluation_seeds(50) == list(range(1000, 1050))
    assert fixed_evaluation_seeds(50) == fixed_evaluation_seeds(50)


def test_action_saturation_statistics_are_agent_and_dimension_specific():
    buffer = RolloutBuffer(1, 1)
    buffer.actions[0, 0] = np.asarray([[1.0, 0.2, -0.96], [0.0, 0.0, 0.0], [-1.0, 0.5, 0.94]])
    buffer.active_masks[0, 0] = 1.0
    diagnostics = TrainingDiagnostics(); diagnostics.observe_rollout(buffer)
    rows = diagnostics.action_rows("mappo", 1, 1)
    mav_ux = next(row for row in rows if row["agent"] == "MAV" and row["action_dimension"] == "ux")
    uav2_uz = next(row for row in rows if row["agent"] == "UAV2" and row["action_dimension"] == "uz")
    assert mav_ux["saturation_rate"] == 1.0 and uav2_uz["saturation_rate"] == 0.0


def test_observation_statistics_are_finite_and_cover_groups():
    buffer = RolloutBuffer(2, 1); buffer.observations[:] = np.linspace(-1, 1, buffer.observations.size).reshape(buffer.observations.shape)
    diagnostics = TrainingDiagnostics(); diagnostics.observe_rollout(buffer)
    rows = diagnostics.observation_rows("mappo", 1, 2)
    from env.mavuav import OBS_DIM
    assert len([row for row in rows if row["row_type"] == "feature"]) == OBS_DIM
    assert any(row["feature"] == "enemy_distance" for row in rows)
    assert all(np.isfinite(row[key]) for row in rows for key in ("mean", "std", "min", "max", "p01", "p99"))


def test_v22_enemy_observation_groups_use_canonical_indices():
    assert OBSERVATION_GROUPS["enemy_relative_position"] == (33, 34, 35, 47, 48, 49)
    assert OBSERVATION_GROUPS["enemy_relative_velocity"] == (37, 38, 39, 51, 52, 53)


def test_compact_diagnostics_preserve_statistics_without_raw_action_history():
    buffer = RolloutBuffer(2, 1)
    buffer.observations[:] = np.linspace(-1, 1, buffer.observations.size).reshape(buffer.observations.shape)
    buffer.actions[:] = np.linspace(-1, 1, buffer.actions.size).reshape(buffer.actions.shape)
    buffer.active_masks[:] = 1.0
    diagnostics = TrainingDiagnostics(max_observation_samples=4)
    diagnostics.observe_rollout(buffer)
    expected_actions = diagnostics.action_rows("mappo", 1, 2)
    expected_observations = diagnostics.observation_rows("mappo", 1, 2)
    state = diagnostics.state_dict()
    assert state["format"] == "training_diagnostics_v2"
    assert "action_batches" not in state and "active_mask_batches" not in state
    assert state["observation_batches"] == []
    restored = TrainingDiagnostics.from_state_dict(state)
    assert restored.action_rows("mappo", 1, 2) == expected_actions
    assert restored.observation_rows("mappo", 1, 2) == expected_observations


def test_legacy_diagnostics_are_compacted_when_loaded():
    actions = np.asarray([[[1.0, 0.0, -0.5], [0.5, 0.25, 0.0], [-1.0, 0.0, 0.5]]], dtype=np.float32)
    legacy = {
        "max_observation_samples": 1,
        "observation_batches": [np.zeros((1, 61), dtype=np.float32)],
        "action_batches": [actions],
        "active_mask_batches": [np.ones((1, 3), dtype=np.float32)],
        "observation_sample_count": 1,
    }
    restored = TrainingDiagnostics.from_state_dict(legacy)
    state = restored.state_dict()
    assert state["observation_batches"] == [] and "action_batches" not in state
    mav_ux = next(row for row in restored.action_rows("mappo", 1, 1) if row["agent"] == "MAV" and row["action_dimension"] == "ux")
    assert mav_ux["mean"] == 1.0 and mav_ux["saturation_rate"] == 1.0


def test_compact_diagnostic_checkpoint_size_does_not_grow_with_action_history():
    buffer = RolloutBuffer(2, 1)
    buffer.active_masks[:] = 1.0
    diagnostics = TrainingDiagnostics(max_observation_samples=4)
    diagnostics.observe_rollout(buffer)
    first = io.BytesIO(); torch.save(diagnostics.state_dict(), first)
    for _ in range(1000):
        diagnostics.observe_rollout(buffer)
    last = io.BytesIO(); torch.save(diagnostics.state_dict(), last)
    assert abs(last.tell() - first.tell()) < 1024


def test_target_proxy_is_read_only_and_does_not_enter_policy_or_reward():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False); observations, _ = env.reset(seed=5)
    before_states = {aid: entity.state.as_array().copy() for aid, entity in env.entities.items()}
    before_reward = env._team_situation_reward(); before_observations = {aid: value.copy() for aid, value in observations.items()}
    proxies = target_proxies(env)
    assert set(proxies) == set(env.red_ids)
    assert np.isclose(before_reward, env._team_situation_reward())
    assert all(np.array_equal(before_states[aid], env.entities[aid].state.as_array()) for aid in env.entities)
    assert all(np.array_equal(before_observations[aid], env._observations()[aid]) for aid in env.red_ids)


def test_reward_diagnostic_decomposition_and_kill_trend():
    records = [
        {"outcome": "red", "situation_reward_sum": 10.0, "event_reward_sum": 100.0, "safety_reward_sum": 0.0, "terminal_reward_sum": 100.0, "episode_return": 210.0, "red_attack_kills": 2},
        {"outcome": "draw", "situation_reward_sum": 21.0, "event_reward_sum": 0.0, "safety_reward_sum": -1.0, "terminal_reward_sum": 0.0, "episode_return": 20.0, "red_attack_kills": 0},
    ]
    rows = reward_diagnostic_rows(records, "mappo", 1, 100, "nearest")
    red = next(row for row in rows if row["outcome"] == "red"); draw = next(row for row in rows if row["outcome"] == "draw")
    assert red["mean_total_return"] == red["mean_situation_reward_sum"] + red["mean_event_reward_sum"] + red["mean_safety_reward_sum"] + red["mean_terminal_reward"]
    assert draw["mean_total_return"] == 20.0 and red["return_red_attack_kills_correlation"] > 0.99
    assert draw_return_exceeds_red_win(rows) is False
    assert draw_return_exceeds_red_win([draw]) is None


def test_zero_and_random_baseline_runner_writes_results(tmp_path: Path):
    config = deepcopy(load_environment_config(None)); config["simulation"]["max_decision_steps"] = 2
    rows = run_rule_baselines(tmp_path, episodes=1, env_config=config, profile="learnability")
    assert {(row["baseline"], row["blue_mode"]) for row in rows} == {("zero", "nearest"), ("zero", "mav_priority"), ("random", "nearest"), ("random", "mav_priority")}
    assert {row["environment_profile"] for row in rows} == {"learnability"}
    assert (tmp_path / "rule_baselines" / "evaluations.csv").exists()
    assert (tmp_path / "rule_baselines" / "summary.json").exists()


def test_calibration_checkpoint_restores_parameters_optimizer_and_rollout_state(tmp_path: Path):
    trainer = tiny_trainer(seed=9); diagnostics = TrainingDiagnostics()
    train_to_sampled_steps(trainer, 2, diagnostics)
    expected = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    expected_observations = trainer.observations.copy()
    checkpoint = tmp_path / "checkpoint.pt"
    save_calibration_checkpoint(checkpoint, trainer, "mappo", 2, diagnostics)
    restored = tiny_trainer(seed=99)
    sampled_steps, restored_diagnostics = load_calibration_checkpoint(checkpoint, restored, "mappo")
    assert sampled_steps == restored.env_steps == 2
    assert restored_diagnostics.observation_sample_count == diagnostics.observation_sample_count
    assert all(torch.equal(a, b) for a, b in zip(expected, restored.actor.parameters()))
    assert np.array_equal(expected_observations, restored.observations)
    trainer.close(); restored.close()


def test_calibration_checkpoint_rejects_environment_profile_mismatch(tmp_path: Path):
    source = tiny_trainer(seed=10, profile="learnability")
    checkpoint = tmp_path / "profile.pt"
    save_calibration_checkpoint(checkpoint, source, "mappo", 0, TrainingDiagnostics())
    source.close()
    target = tiny_trainer(seed=11, profile="main")
    with pytest.raises(RuntimeError, match="environment_profile"):
        load_calibration_checkpoint(checkpoint, target, "mappo")
    target.close()


def test_old_40d_calibration_checkpoint_is_rejected_clearly(tmp_path: Path):
    trainer = tiny_trainer(seed=17); diagnostics = TrainingDiagnostics()
    train_to_sampled_steps(trainer, 2, diagnostics)
    checkpoint = tmp_path / "v2.pt"
    save_calibration_checkpoint(checkpoint, trainer, "mappo", 2, diagnostics)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format"] = "mavuav_learnability_calibration_v1"
    payload["environment_version"] = None
    payload["observation_dim"] = 40
    payload["global_state_dim"] = 40
    old_checkpoint = tmp_path / "old_40d.pt"; torch.save(payload, old_checkpoint)
    restored = tiny_trainer(seed=18)
    with np.testing.assert_raises_regex(RuntimeError, "incompatible calibration checkpoint"):
        load_calibration_checkpoint(old_checkpoint, restored, "mappo")
    trainer.close(); restored.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_calibration_checkpoint_restores_cpu_and_cuda_rng_state(tmp_path: Path):
    trainer = tiny_trainer(seed=21, device="cuda"); diagnostics = TrainingDiagnostics()
    train_to_sampled_steps(trainer, 2, diagnostics)
    checkpoint = tmp_path / "cuda_checkpoint.pt"
    save_calibration_checkpoint(checkpoint, trainer, "mappo", 2, diagnostics)
    restored = tiny_trainer(seed=22, device="cuda")
    sampled_steps, _ = load_calibration_checkpoint(checkpoint, restored, "mappo")
    assert sampled_steps == 2 and restored.observations.shape[-1] == 61 and restored.global_states.shape[-1] == 67
    trainer.close(); restored.close()


def test_sampled_step_counting_uses_num_env_transitions_exactly():
    config = deepcopy(load_environment_config(None)); config["simulation"]["max_decision_steps"] = 2
    trainer = MAPPOTrainer(config, {"num_envs": 2, "rollout_steps": 3, "ppo_epochs": 1, "minibatch_size": 4, "hidden_dim": 16, "seed": 4})
    diagnostics = TrainingDiagnostics(); _, metrics = train_to_sampled_steps(trainer, 10, diagnostics)
    assert trainer.env_steps == 10 and metrics[-1]["sampled_steps"] == 10
    assert diagnostics.observation_sample_count == 30
    trainer.close()
