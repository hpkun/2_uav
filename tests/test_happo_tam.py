from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.happo import HAPPOTrainer, preceding_factor_update
from algorithm.happo.evaluation import evaluate_recurrent_actors
from algorithm.happo.tam import TAMAttentionCritic, TAMGaussianActor, TAMIndependentActors
from algorithm.happo.tam_buffer import TAMRolloutBuffer
import algorithm.happo.trainer as trainer_module
from algorithm.train_happo import _algorithm_name
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, load_environment_config


ROOT = Path(__file__).resolve().parents[1]


def _env(max_steps: int = 4):
    config = deepcopy(load_environment_config(ROOT / "configs" / "env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def _config(**updates):
    config = {
        "actor_variant": "tam", "critic_variant": "tam_attention", "method_variant": "baseline",
        "environment_profile": "learnability", "seed": 11, "device": "cpu", "num_envs": 1,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 4,
        "tam_actor_gru_hidden_dim": 16, "tam_critic_gru_hidden_dim": 16,
        "tam_token_dim": 16, "tam_attention_heads": 4,
        "tam_actor_hidden_layers": [24, 12], "tam_critic_hidden_layers": [24, 12],
        "tam_recurrent_sequence_length": 2, "actor_log_std_init": -0.25,
    }
    config.update(updates)
    return config


def test_tam_actors_are_independent_and_state_memory_first_topology():
    assert _algorithm_name("tam", "baseline", "tam_attention") == "tam_happo"
    actors = TAMIndependentActors()
    assert len(actors.actors) == len(RED_IDS) == 4
    parameter_ids = [{id(parameter) for parameter in actor.parameters()} for actor in actors.actors]
    storage = [{parameter.untyped_storage().data_ptr() for parameter in actor.parameters()} for actor in actors.actors]
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j]) for i in range(4) for j in range(i + 1, 4))
    assert all(storage[i].isdisjoint(storage[j]) for i in range(4) for j in range(i + 1, 4))
    actor = actors.actors[0]
    assert actor.gru.input_size == OBS_DIM and actor.gru.hidden_size == 128
    assert actor.temporal_projection.in_features == 128 and actor.temporal_projection.out_features == OBS_DIM
    assert actor.policy_fc1.in_features == 2 * OBS_DIM and actor.policy_fc1.out_features == 256
    assert actor.policy_fc2.in_features == 256 and actor.policy_fc2.out_features == 128


def test_tam_actor_sequence_preserves_history_and_exact_gaussian_shapes():
    actor = TAMGaussianActor(recurrent_hidden_dim=12, hidden_layers=(20, 10))
    observation = torch.randn(2, OBS_DIM)
    prior = torch.randn(2, 12)
    reset_mean, reset_hidden = actor.forward_step(observation, prior, torch.zeros(2))
    zero_mean, _ = actor.forward_step(observation, torch.zeros_like(prior), torch.zeros(2))
    carried_mean, carried_hidden = actor.forward_step(observation, prior, torch.ones(2))
    assert torch.allclose(reset_mean, zero_mean)
    assert not torch.allclose(carried_mean, reset_mean)
    assert not torch.allclose(carried_hidden, reset_hidden)
    observations = torch.randn(2, 3, OBS_DIM)
    actions = torch.tanh(torch.randn(2, 3, 3))
    logs, entropy, final_hidden = actor.evaluate_actions_sequence(
        observations, actions, torch.zeros(2, 12), torch.ones(2, 3),
    )
    assert logs.shape == entropy.shape == (2, 3) and final_hidden.shape == (2, 12)
    assert all(torch.isfinite(value).all() for value in (logs, entropy, final_hidden))


def test_tam_critic_parses_exact_state_and_masks_dead_entities():
    critic = TAMAttentionCritic(recurrent_hidden_dim=16, token_dim=16, attention_heads=4, hidden_layers=(24, 12))
    states = torch.randn(3, GLOBAL_STATE_DIM)
    states[:, 6] = 0.0
    states[:, 16] = 1.0
    value, hidden = critic.forward_step(states, torch.zeros(3, 16), torch.zeros(3))
    assert value.shape == (3,) and hidden.shape == (3, 16)
    assert torch.isfinite(value).all() and torch.isfinite(hidden).all()
    assert critic.last_token_count == 9
    assert critic.last_attention_key_padding_mask.shape == (3, 9)
    assert critic.last_attention_key_padding_mask[:, 0].all()
    assert not critic.last_attention_key_padding_mask[:, 1].any()
    assert not critic.last_attention_key_padding_mask[:, -1].any()
    assert isinstance(critic.value_head[1], torch.nn.LayerNorm)
    assert tuple(critic.value_head[1].normalized_shape) == (24,)
    assert isinstance(critic.value_head[4], torch.nn.LayerNorm)
    assert tuple(critic.value_head[4].normalized_shape) == (12,)
    architecture = critic.architecture()
    assert architecture["post_attention_layer_norm_dims"] == [24, 12]
    assert architecture["attention_residual_layer_norm"] is True


def test_tam_environment_contract_accepts_only_v39_and_expected_reward(monkeypatch):
    valid = HAPPOTrainer(_env(2), _config(rollout_steps=1))
    valid.close()

    non_v39 = deepcopy(load_environment_config(ROOT / "configs" / "env_v38.yaml"))
    non_v39["simulation"]["max_decision_steps"] = 2
    with pytest.raises(ValueError, match="TAM-HAPPO requires"):
        HAPPOTrainer(non_v39, _config(rollout_steps=1))

    monkeypatch.setattr(trainer_module, "_resolved_reward_mode", lambda config: "absolute")
    with pytest.raises(ValueError, match="heterogeneous_role_coupled_gate_v1"):
        HAPPOTrainer(_env(2), _config(rollout_steps=1))


def test_tam_buffer_stores_actor_and_critic_pre_step_hidden():
    buffer = TAMRolloutBuffer(1, 1, actor_hidden_dim=3, critic_hidden_dim=5)
    actor_hidden = np.ones((1, 4, 3), np.float32)
    critic_hidden = np.ones((1, 5), np.float32) * 2
    buffer.insert(
        np.zeros((1, 4, OBS_DIM), np.float32), np.zeros((1, GLOBAL_STATE_DIM), np.float32),
        np.zeros((1, 4, 3), np.float32), np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32),
        np.zeros(1, np.float32), np.zeros(1, bool), np.zeros(1, bool), np.ones((1, 4), np.float32),
        actor_hidden, np.ones((1, 4), np.float32), actor_hidden + 1,
        critic_hidden, np.ones(1, np.float32), critic_hidden + 1,
    )
    assert np.array_equal(buffer.actor_hidden_states[0], actor_hidden)
    assert np.array_equal(buffer.critic_hidden_states[0], critic_hidden)
    assert np.array_equal(buffer.critic_hidden_states[1], critic_hidden + 1)


def test_tam_collection_masks_dead_action_and_resets_hidden(monkeypatch):
    trainer = HAPPOTrainer(_env(20), _config(rollout_steps=1))
    trainer.actor_hidden_states.fill(0.5)
    trainer.actor_recurrent_masks.fill(1.0)
    captured = {}

    def controlled_step(actions):
        captured["actions"] = actions.copy()
        return (
            trainer.observations.copy(), trainer.global_states.copy(), np.zeros((1, 4), np.float32),
            np.asarray([False]), np.asarray([False]), np.asarray([[1, 0, 1, 1]], np.float32), [{}],
        )

    trainer.active_masks[0, 1] = 0.0
    monkeypatch.setattr(trainer.vector_env, "step", controlled_step)
    trainer.collect_rollout()
    assert np.array_equal(captured["actions"][0, 1], np.zeros(3))
    assert np.count_nonzero(trainer.actor_hidden_states[:, 1]) == 0
    assert np.count_nonzero(trainer.actor_hidden_states[:, 0]) > 0
    trainer.close()


def test_tam_episode_end_resets_all_actor_and_critic_hidden(monkeypatch):
    trainer = HAPPOTrainer(_env(20), _config(rollout_steps=1))
    trainer.actor_hidden_states.fill(0.5)
    trainer.actor_recurrent_masks.fill(1.0)
    trainer.critic_hidden_states.fill(0.5)
    trainer.critic_recurrent_masks.fill(1.0)

    def terminal_step(actions):
        return (
            trainer.observations.copy(), trainer.global_states.copy(), np.zeros((1, 4), np.float32),
            np.asarray([True]), np.asarray([False]), np.ones((1, 4), np.float32),
            [{"episode_summary": {"outcome": "draw"}}],
        )

    monkeypatch.setattr(trainer.vector_env, "step", terminal_step)
    trainer.collect_rollout()
    assert np.count_nonzero(trainer.actor_hidden_states) == 0
    assert np.count_nonzero(trainer.critic_hidden_states) == 0
    assert np.count_nonzero(trainer.actor_recurrent_masks) == 0
    assert np.count_nonzero(trainer.critic_recurrent_masks) == 0
    trainer.close()


def test_tam_sequential_update_uses_ordered_replay_and_inactive_factor_one():
    trainer = HAPPOTrainer(_env(20), _config(rollout_steps=3, tam_recurrent_sequence_length=2))
    trainer.collect_rollout()
    trainer.buffer.active_masks[1, 0] = 0.0
    old = torch.as_tensor(trainer.buffer.log_probs.copy())
    metrics = trainer.update()
    first = metrics["agent_update_order"][0]
    new = trainer._recurrent_log_probs_all(first).cpu()
    active = torch.as_tensor(trainer.buffer.active_masks[:, :, first])
    expected = preceding_factor_update(torch.ones_like(new), old[:, :, first], new, active)
    assert torch.allclose(torch.as_tensor(trainer.last_recurrent_factor_history[1]), expected, atol=1e-6)
    assert trainer.last_recurrent_factor_history[1][1, 0] == 1.0
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    trainer.close()


def test_tam_huber_update_checkpoint_round_trip_and_metadata(tmp_path):
    config = _config()
    source = HAPPOTrainer(_env(20), config)
    _, metrics = source.train_update()
    assert np.isfinite(metrics["critic_loss"])
    checkpoint = tmp_path / "tam.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["actor_variant"] == "tam" and payload["critic_variant"] == "tam_attention"
    assert payload["algorithm"] == "tam_happo" and payload["base_algorithm"] == "happo"
    assert payload["tam_state_memory"] is True
    assert payload["independent_actor_count"] == 4 and payload["sequential_happo_update"] is True
    weights = tmp_path / "tam_weights.pt"
    source.save(weights)
    weights_payload = torch.load(weights, map_location="cpu", weights_only=False)
    assert weights_payload["algorithm"] == "tam_happo"
    assert weights_payload["base_algorithm"] == "happo"
    restored = HAPPOTrainer(_env(20), config)
    assert restored.load_checkpoint(checkpoint) == source.env_steps
    assert np.array_equal(restored.actor_hidden_states, source.actor_hidden_states)
    assert np.array_equal(restored.critic_hidden_states, source.critic_hidden_states)
    for left, right in zip(source.actors.parameters(), restored.actors.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(source.critic.parameters(), restored.critic.parameters()):
        assert torch.equal(left, right)
    source.close(); restored.close()


def test_tam_recurrent_evaluation_deterministic_and_stochastic_seed_protocol():
    trainer = HAPPOTrainer(_env(2), _config(rollout_steps=1))
    deterministic_a = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "learnability", seed=1000, deterministic=True,
        inactive_mask=True,
    )
    deterministic_b = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "learnability", seed=1000, deterministic=True,
        action_seed=9999, inactive_mask=True,
    )
    stochastic_a = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "learnability", seed=1000, deterministic=False,
        action_seed=2000, inactive_mask=True,
    )
    stochastic_b = evaluate_recurrent_actors(
        trainer.actors, trainer.environment_config, 2, "learnability", seed=1000, deterministic=False,
        action_seed=2000, inactive_mask=True,
    )
    assert deterministic_a == deterministic_b and stochastic_a == stochastic_b
    torch.manual_seed(2000)
    first, _, _ = trainer.actors.actors[0].sample_step(
        torch.as_tensor(trainer.observations[:, 0]), torch.zeros(1, 16), torch.zeros(1), False,
    )
    torch.manual_seed(2001)
    second, _, _ = trainer.actors.actors[0].sample_step(
        torch.as_tensor(trainer.observations[:, 0]), torch.zeros(1, 16), torch.zeros(1), False,
    )
    assert not torch.equal(first, second)
    trainer.close()


@pytest.mark.parametrize("toggle", ["tam_state_memory", "tam_attention", "tam_inactive_mask"])
def test_tam_ablation_toggles_construct(toggle):
    trainer = HAPPOTrainer(_env(2), _config(**{toggle: False}, rollout_steps=1))
    assert trainer.config[toggle] is False
    trainer.close()
