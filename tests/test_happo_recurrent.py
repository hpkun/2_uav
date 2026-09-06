from __future__ import annotations

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

from algorithm.common.networks import CentralizedCritic
from algorithm.happo import HAPPOTrainer
from algorithm.happo.evaluation import evaluate_recurrent_actors
from algorithm.happo.recurrent import RecurrentGaussianActor
from algorithm.happo.recurrent_buffer import RecurrentRolloutBuffer, sequence_chunks
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, load_environment_config


def _trainer_config(**updates):
    config = {
        "actor_variant": "recurrent", "method_variant": "baseline",
        "num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1,
        "minibatch_size": 4, "hidden_dim": 16, "recurrent_hidden_dim": 16,
        "recurrent_sequence_length": 2, "seed": 17, "device": "cpu",
        "environment_profile": "learnability",
    }
    config.update(updates)
    return config


def _short_env(max_steps: int = 4):
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def _next_policy_values(trainer: HAPPOTrainer):
    actions, logs, hidden = [], [], []
    with torch.no_grad():
        for agent, actor in enumerate(trainer.actors.actors):
            action, log_prob, next_hidden = actor.sample_step(
                torch.as_tensor(trainer.observations[:, agent], device=trainer.device),
                torch.as_tensor(trainer.actor_hidden_states[:, agent], device=trainer.device),
                torch.as_tensor(trainer.actor_recurrent_masks[:, agent], device=trainer.device),
            )
            actions.append(action.cpu())
            logs.append(log_prob.cpu())
            hidden.append(next_hidden.cpu())
    return torch.stack(actions), torch.stack(logs), torch.stack(hidden)


def test_recurrent_actor_step_sequence_shapes_and_parameter_count():
    actor = RecurrentGaussianActor()
    observation = torch.randn(4, OBS_DIM)
    hidden = actor.initial_hidden(4)
    action, log_prob, next_hidden = actor.sample_step(observation, hidden, torch.ones(4))
    assert action.shape == (4, 3) and log_prob.shape == (4,) and next_hidden.shape == (4, 128)
    assert all(torch.isfinite(value).all() for value in (action, log_prob, next_hidden))
    observations = torch.randn(4, 5, OBS_DIM)
    actions = torch.tanh(torch.randn(4, 5, 3))
    logs, entropy, final_hidden = actor.evaluate_actions_sequence(
        observations, actions, hidden, torch.ones(4, 5),
    )
    assert logs.shape == entropy.shape == (4, 5) and final_hidden.shape == (4, 128)
    assert all(torch.isfinite(value).all() for value in (logs, entropy, final_hidden))
    assert sum(parameter.numel() for parameter in actor.parameters()) == 128_902


def test_sequence_evaluation_matches_manual_ordered_steps():
    torch.manual_seed(3)
    actor = RecurrentGaussianActor(hidden_dim=12, recurrent_hidden_dim=9)
    observations = torch.randn(2, 6, OBS_DIM)
    actions = torch.tanh(torch.randn(2, 6, 3))
    masks = torch.tensor([[0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1]], dtype=torch.float32)
    initial = torch.randn(2, 9)
    sequence_logs, sequence_entropy, sequence_hidden = actor.evaluate_actions_sequence(
        observations, actions, initial, masks,
    )
    hidden = initial
    manual_logs, manual_entropy = [], []
    for step in range(6):
        mean, hidden = actor.forward_step(observations[:, step], hidden, masks[:, step])
        clipped = actions[:, step].clamp(-1 + actor.epsilon, 1 - actor.epsilon)
        distribution = actor._distribution(mean)
        raw = torch.atanh(clipped)
        manual_logs.append((distribution.log_prob(raw) - torch.log(1 - clipped.square() + actor.epsilon)).sum(-1))
        manual_entropy.append(distribution.entropy().sum(-1))
    assert torch.allclose(sequence_logs, torch.stack(manual_logs, 1))
    assert torch.allclose(sequence_entropy, torch.stack(manual_entropy, 1))
    assert torch.allclose(sequence_hidden, hidden)


def test_recurrent_mask_resets_history_and_actor_is_history_sensitive():
    torch.manual_seed(5)
    actor = RecurrentGaussianActor(hidden_dim=16, recurrent_hidden_dim=16)
    observation = torch.randn(3, OBS_DIM)
    prior = torch.randn(3, 16)
    reset_mean, reset_hidden = actor.forward_step(observation, prior, torch.zeros(3))
    zero_mean, zero_hidden = actor.forward_step(observation, torch.zeros_like(prior), torch.zeros(3))
    carried_mean, carried_hidden = actor.forward_step(observation, prior, torch.ones(3))
    assert torch.allclose(reset_mean, zero_mean) and torch.allclose(reset_hidden, zero_hidden)
    assert not torch.allclose(carried_mean, reset_mean)
    assert not torch.allclose(carried_hidden, reset_hidden)


def test_recurrent_buffer_stores_pre_action_hidden_and_masks():
    buffer = RecurrentRolloutBuffer(2, 1, recurrent_hidden_dim=4)
    hidden = np.arange(16, dtype=np.float32).reshape(1, len(RED_IDS), 4)
    next_hidden = hidden + 100
    masks = np.asarray([[0, 1, 1, 1]], dtype=np.float32)
    buffer.insert(
        np.zeros((1, len(RED_IDS), OBS_DIM), np.float32), np.zeros((1, GLOBAL_STATE_DIM), np.float32),
        np.zeros((1, len(RED_IDS), 3), np.float32), np.zeros((1, len(RED_IDS)), np.float32),
        np.zeros((1, len(RED_IDS)), np.float32), np.zeros(1, np.float32),
        np.zeros(1, bool), np.zeros(1, bool), np.ones((1, len(RED_IDS)), np.float32),
        hidden, masks, next_hidden,
    )
    assert buffer.actor_hidden_states.shape == (3, 1, len(RED_IDS), 4)
    assert buffer.recurrent_masks.shape == (2, 1, len(RED_IDS))
    assert np.array_equal(buffer.actor_hidden_states[0], hidden)
    assert np.array_equal(buffer.actor_hidden_states[1], next_hidden)
    assert np.array_equal(buffer.recurrent_masks[0], masks)


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (18, [(0, 0, 16), (0, 16, 18), (1, 0, 16), (1, 16, 18)]),
        (72, [(0, 0, 16), (0, 16, 32), (0, 32, 48), (0, 48, 64), (0, 64, 72)]),
    ],
)
def test_sequence_chunks_keep_env_time_order_and_short_tail(horizon, expected):
    chunks = sequence_chunks(horizon, 2 if horizon == 18 else 1, 16)
    assert chunks == expected
    assert sum(end - start for _, start, end in chunks) == horizon * (2 if horizon == 18 else 1)


def test_episode_boundary_inside_sequence_resets_at_next_step():
    actor = RecurrentGaussianActor(hidden_dim=10, recurrent_hidden_dim=10)
    observations = torch.randn(1, 16, OBS_DIM)
    actions = torch.tanh(torch.randn(1, 16, 3))
    masks = torch.ones(1, 16)
    masks[:, 0] = 0
    masks[:, 8] = 0
    _, _, _ = actor.evaluate_actions_sequence(observations, actions, torch.randn(1, 10), masks)
    hidden = torch.randn(1, 10)
    for step in range(8):
        _, hidden = actor.forward_step(observations[:, step], hidden, masks[:, step])
    mean_boundary, hidden_boundary = actor.forward_step(observations[:, 8], hidden, masks[:, 8])
    mean_zero, hidden_zero = actor.forward_step(observations[:, 8], torch.zeros_like(hidden), masks[:, 8])
    assert torch.allclose(mean_boundary, mean_zero) and torch.allclose(hidden_boundary, hidden_zero)


@pytest.mark.parametrize("done,next_masks", [(False, [1, 0, 1, 1]), (True, [1, 1, 1, 1])])
def test_collection_resets_only_required_agent_hidden(monkeypatch, done, next_masks):
    trainer = HAPPOTrainer(_short_env(20), _trainer_config(rollout_steps=1))
    trainer.actor_hidden_states.fill(0.5)
    trainer.actor_recurrent_masks.fill(1.0)
    observations = trainer.observations.copy()
    states = trainer.global_states.copy()

    def controlled_step(actions, reset_nearest_probability=None):
        infos = [{"episode_summary": {"outcome": "draw"}, "auto_reset": True,
                  "reset_info": {"blue_target_mode": "nearest"}}] if done else [{}]
        return (
            observations.copy(), states.copy(), np.zeros((1, len(RED_IDS)), np.float32),
            np.asarray([done]), np.asarray([False]), np.asarray([next_masks], np.float32), infos,
        )

    monkeypatch.setattr(trainer.vector_env, "step", controlled_step)
    trainer.collect_rollout()
    if done:
        assert np.count_nonzero(trainer.actor_hidden_states) == 0
        assert np.count_nonzero(trainer.actor_recurrent_masks) == 0
    else:
        assert np.count_nonzero(trainer.actor_hidden_states[:, 1]) == 0
        assert np.count_nonzero(trainer.actor_hidden_states[:, 0]) > 0
        assert np.count_nonzero(trainer.actor_hidden_states[:, 2]) > 0
        assert np.array_equal(trainer.actor_recurrent_masks, np.asarray([[1, 0, 1, 1]], np.float32))
    trainer.close()


def test_hidden_continues_across_rollout_update_boundary():
    trainer = HAPPOTrainer(_short_env(20), _trainer_config(rollout_steps=1))
    trainer.collect_rollout()
    boundary_hidden = trainer.actor_hidden_states.copy()
    boundary_masks = trainer.actor_recurrent_masks.copy()
    trainer.update()
    assert np.array_equal(trainer.actor_hidden_states, boundary_hidden)
    trainer.buffer = trainer.make_buffer(1)
    trainer.collect_rollout()
    assert np.array_equal(trainer.buffer.actor_hidden_states[0], boundary_hidden)
    assert np.array_equal(trainer.buffer.recurrent_masks[0], boundary_masks)
    trainer.close()


@pytest.mark.parametrize("horizon", [18, 72])
def test_short_final_chunk_recurrent_update_uses_all_steps(horizon):
    trainer = HAPPOTrainer(
        _short_env(200),
        _trainer_config(rollout_steps=horizon, recurrent_sequence_length=16, minibatch_size=32),
    )
    trainer.collect_rollout()
    metrics = trainer.update()
    assert trainer.buffer.position == horizon
    assert sorted(metrics["agent_update_order"]) == list(range(len(RED_IDS)))
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    trainer.close()


def test_recurrent_sequential_factor_matches_sequence_log_probs_and_ignores_inactive():
    trainer = HAPPOTrainer(_short_env(20), _trainer_config(rollout_steps=4, recurrent_sequence_length=3))
    trainer.collect_rollout()
    trainer.buffer.active_masks[1, 0, :] = 0.0
    old = torch.as_tensor(trainer.buffer.log_probs.copy())
    metrics = trainer.update()
    first_agent = metrics["agent_update_order"][0]
    new = trainer._recurrent_log_probs_all(first_agent).cpu()
    active = torch.as_tensor(trainer.buffer.active_masks[:, :, first_agent]) > 0.5
    expected = torch.where(active, torch.exp(new - old[:, :, first_agent]), torch.ones_like(new))
    actual = torch.as_tensor(trainer.last_recurrent_factor_history[1])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual[1, 0] == 1.0
    assert torch.isfinite(actual).all()
    trainer.close()


def test_recurrent_critic_and_architecture_contract_are_unchanged():
    trainer = HAPPOTrainer(config=_trainer_config(hidden_dim=128, recurrent_hidden_dim=128, rollout_steps=1))
    assert isinstance(trainer.critic, CentralizedCritic)
    assert trainer.critic.network[0].in_features == GLOBAL_STATE_DIM == 119
    assert trainer.actor_parameter_counts == {"per_agent": [128_902] * len(RED_IDS), "total": 515_608}
    assert trainer.actor_architecture == {
        "observation_dim": OBS_DIM, "encoder_dim": 128, "recurrent_hidden_dim": 128,
        "head_dim": 128, "action_dim": 3,
    }
    trainer.close()


def test_recurrent_checkpoint_exactly_restores_hidden_rng_and_next_policy_values(tmp_path):
    config = _trainer_config(rollout_steps=2)
    env_config = _short_env(20)
    source = HAPPOTrainer(env_config, config)
    source.train_update()
    checkpoint = tmp_path / "recurrent.pt"
    source.save_checkpoint(checkpoint)
    expected_hidden = source.actor_hidden_states.copy()
    expected_masks = source.actor_recurrent_masks.copy()
    expected = _next_policy_values(source)
    restored = HAPPOTrainer(env_config, config)
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    actual = _next_policy_values(restored)
    assert np.array_equal(restored.actor_hidden_states, expected_hidden)
    assert np.array_equal(restored.actor_recurrent_masks, expected_masks)
    assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    for expected_value, actual_value in zip(expected, actual):
        assert torch.equal(expected_value, actual_value)
    source.close()
    restored.close()


@pytest.mark.parametrize(("field", "value"), [("recurrent_hidden_dim", 8), ("recurrent_sequence_length", 1)])
def test_recurrent_resume_rejects_recurrent_config_mismatch(tmp_path, field, value):
    env_config = _short_env()
    config = _trainer_config()
    source = HAPPOTrainer(env_config, config)
    checkpoint = tmp_path / f"{field}.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    target = HAPPOTrainer(env_config, {**config, field: value})
    with pytest.raises(RuntimeError, match=field):
        target.load_checkpoint(checkpoint)
    target.close()


@pytest.mark.parametrize("missing", ["actor_hidden_states", "actor_recurrent_masks"])
def test_recurrent_resume_rejects_missing_continuation_state(tmp_path, missing):
    env_config = _short_env()
    config = _trainer_config()
    source = HAPPOTrainer(env_config, config)
    checkpoint = tmp_path / "missing.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rollout_state"].pop(missing)
    torch.save(payload, checkpoint)
    target = HAPPOTrainer(env_config, config)
    with pytest.raises(RuntimeError, match="missing hidden-state"):
        target.load_checkpoint(checkpoint)
    target.close()


def test_recurrent_evaluation_is_deterministic_and_episode_local():
    trainer = HAPPOTrainer(_short_env(2), _trainer_config(rollout_steps=1))
    first = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "nearest", "learnability", seed=1000,
    )
    second = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "nearest", "learnability", seed=1000,
    )
    assert first == second and len(first) == 2
    trainer.close()


def test_recurrent_entrypoints_expose_cli_help():
    root = Path(__file__).resolve().parents[1]
    for entrypoint in ("algorithm/train_happo_recurrent.py", "algorithm/evaluate_happo_recurrent.py"):
        result = subprocess.run(
            [sys.executable, entrypoint, "--help"], cwd=root, text=True, capture_output=True, check=True,
        )
        assert "--device" in result.stdout


def test_recurrent_training_and_evaluation_entrypoints_cpu_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_name = f"pytest_happo_recurrent_{uuid.uuid4().hex}"
    run_dir = root / "outputs" / output_name
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(_short_env(2), sort_keys=False), encoding="utf-8")
    try:
        subprocess.run([
            sys.executable, "algorithm/train_happo_recurrent.py", "--steps", "2",
            "--profile", "learnability", "--device", "cpu", "--num-envs", "1",
            "--env-config", str(env_path), "--output-name", output_name,
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "0",
            "--final-eval-episodes", "1",
        ], cwd=root, check=True, capture_output=True, text=True, timeout=180)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        for data in (summary, resolved, checkpoint):
            assert data["actor_variant"] == "recurrent"
            assert data["actor_architecture"] == {
                "observation_dim": OBS_DIM, "encoder_dim": 128, "recurrent_hidden_dim": 128,
                "head_dim": 128, "action_dim": 3,
            }
        assert summary["algorithm"] == resolved["algorithm"] == "happo_recurrent"
        assert checkpoint["format"] == "happo_training_checkpoint_v1"
        assert "actor_hidden_states" in checkpoint["rollout_state"]
        assert "actor_recurrent_masks" in checkpoint["rollout_state"]
        subprocess.run([
            sys.executable, "algorithm/evaluate_happo_recurrent.py",
            str(run_dir / "checkpoint_final.pt"), "--profile", "learnability",
            "--episodes", "1", "--device", "cpu", "--blue-mode", "nearest",
            "--env-config", str(env_path),
        ], cwd=root, check=True, capture_output=True, text=True, timeout=180)
        evaluation = json.loads(
            (run_dir / "evaluation_recurrent_final_summary.json").read_text(encoding="utf-8"),
        )
        assert evaluation["algorithm"] == "happo_recurrent"
        assert evaluation["actor_variant"] == "recurrent"
        assert len(evaluation["results"]) == 1
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_recurrent_cuda_one_update_checkpoint_and_eval_smoke(tmp_path):
    config = _trainer_config(device="cuda", rollout_steps=2)
    env_config = _short_env(2)
    trainer = HAPPOTrainer(env_config, config)
    _, metrics = trainer.train_update()
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    checkpoint = tmp_path / "cuda.pt"
    trainer.save_checkpoint(checkpoint)
    restored = HAPPOTrainer(env_config, config)
    restored.load_checkpoint(checkpoint)
    records = evaluate_recurrent_actors(
        restored.actors, env_config, 1, "nearest", "learnability", device="cuda",
    )
    assert len(records) == 1
    trainer.close()
    restored.close()
