import numpy as np
import pytest
import torch
from algorithm.happo import HAPPOTrainer
from copy import deepcopy
from env.mavuav import load_environment_config


@pytest.mark.parametrize("profile", ["learnability", "main"])
def test_happo_trainer_propagates_environment_profile(profile):
    trainer = HAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": profile})
    assert trainer.config["environment_profile"] == profile
    assert trainer.vector_env.get_env_states()[0]["profile"] == profile
    trainer.close()


def test_happo_checkpoint_rejects_environment_profile_mismatch(tmp_path):
    checkpoint = tmp_path / "happo.pt"
    source = HAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": "learnability"})
    source.save(checkpoint); source.close()
    target = HAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": "main"})
    with pytest.raises(RuntimeError, match="environment profile"):
        target.load(checkpoint)
    target.close()


def test_happo_rollout_gae_sequential_updates_are_finite_and_change_parameters():
    env_config = deepcopy(load_environment_config(None)); env_config["simulation"]["max_decision_steps"] = 3
    trainer = HAPPOTrainer(env_config, {"num_envs": 2, "rollout_steps": 8, "ppo_epochs": 2, "minibatch_size": 8, "hidden_dim": 16, "seed": 13})
    before = [[p.detach().clone() for p in actor.parameters()] for actor in trainer.actors.actors]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    assert len(trainer.actors.actors) == 3 and len({id(actor) for actor in trainer.actors.actors}) == 3
    assert all(actor.network[0].in_features == 55 for actor in trainer.actors.actors)
    assert trainer.critic.network[0].in_features == 67
    completed = trainer.collect_rollout()
    assert completed and all(record["episode_length"] <= 3 for record in completed)
    assert np.all(np.isfinite(trainer.buffer.advantages)) and trainer.buffer.position == 8
    metrics = trainer.update()
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float)) and sorted(metrics["agent_update_order"]) == [0, 1, 2]
    assert all(any(not torch.equal(a, b) for a, b in zip(snapshot, actor.parameters())) for snapshot, actor in zip(before, trainer.actors.actors))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic.parameters()))
    assert np.all(np.isfinite(trainer.buffer.returns))
    trainer.collect_rollout(); trainer.update(); trainer.close()


def test_happo_full_checkpoint_restores_optimizer_rng_and_vector_state(tmp_path):
    env_config = deepcopy(load_environment_config(None))
    env_config["simulation"]["max_decision_steps"] = 3
    config = {
        "num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 2,
        "hidden_dim": 16, "seed": 31, "environment_profile": "learnability",
    }
    source = HAPPOTrainer(env_config, config)
    source.train_update()
    checkpoint = tmp_path / "checkpoint_2.pt"
    source.save_checkpoint(checkpoint)
    expected_observations = source.observations.copy()
    expected_states = source.global_states.copy()
    expected_masks = source.active_masks.copy()
    expected_reset_counts = source.vector_env.reset_counts.copy()
    expected_actor_optimizer = source.actor_optimizers[0].state_dict()

    restored = HAPPOTrainer(env_config, config)
    assert restored.load_checkpoint(checkpoint) == source.env_steps == 2
    assert np.array_equal(restored.observations, expected_observations)
    assert np.array_equal(restored.global_states, expected_states)
    assert np.array_equal(restored.active_masks, expected_masks)
    assert np.array_equal(restored.vector_env.reset_counts, expected_reset_counts)
    assert restored.actor_optimizers[0].state_dict()["state"].keys() == expected_actor_optimizer["state"].keys()
    assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    source.close(); restored.close()


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [("seed", 2), ("clip_coef", 0.1), ("rollout_steps", 3), ("num_envs", 2), ("environment_profile", "main")],
)
def test_happo_resume_rejects_training_config_mismatch(tmp_path, field, changed_value):
    env_config = deepcopy(load_environment_config(None))
    base = {
        "num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 2,
        "hidden_dim": 16, "seed": 1, "environment_profile": "learnability",
    }
    source = HAPPOTrainer(env_config, base)
    checkpoint = tmp_path / f"{field}.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    current = {**base, field: changed_value}
    target = HAPPOTrainer(env_config, current)
    with pytest.raises(RuntimeError, match=rf"resume config mismatch: {field}"):
        target.load_checkpoint(checkpoint)
    target.close()


def test_happo_resume_allows_device_metadata_to_differ(tmp_path):
    env_config = deepcopy(load_environment_config(None))
    config = {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "seed": 1}
    source = HAPPOTrainer(env_config, config)
    checkpoint = tmp_path / "device.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["trainer_config"]["device"] = "cuda:0"
    torch.save(payload, checkpoint)
    target = HAPPOTrainer(env_config, config)
    assert target.load_checkpoint(checkpoint) == 0
    target.close()


def test_happo_full_checkpoint_rejects_environment_configuration_change(tmp_path):
    env_config = deepcopy(load_environment_config(None))
    config = {"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": "main"}
    source = HAPPOTrainer(env_config, config)
    checkpoint = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    changed = deepcopy(env_config)
    changed["simulation"]["max_decision_steps"] -= 1
    target = HAPPOTrainer(changed, config)
    with pytest.raises(RuntimeError, match="environment config mismatch"):
        target.load_checkpoint(checkpoint)
    target.close()
