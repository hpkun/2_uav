import numpy as np
import pytest
import torch
from algorithm.mappo import MAPPOTrainer
from copy import deepcopy
from env.mavuav import load_environment_config


@pytest.mark.parametrize("profile", ["learnability", "main"])
def test_mappo_trainer_propagates_environment_profile(profile):
    trainer = MAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": profile})
    assert trainer.config["environment_profile"] == profile
    assert trainer.vector_env.get_env_states()[0]["profile"] == profile
    trainer.close()


def test_mappo_checkpoint_rejects_environment_profile_mismatch(tmp_path):
    checkpoint = tmp_path / "mappo.pt"
    source = MAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": "learnability"})
    source.save(checkpoint); source.close()
    target = MAPPOTrainer(config={"num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "environment_profile": "main"})
    with pytest.raises(RuntimeError, match="environment profile"):
        target.load(checkpoint)
    target.close()


def test_mappo_rollout_gae_updates_are_finite_and_change_parameters():
    env_config = deepcopy(load_environment_config(None)); env_config["simulation"]["max_decision_steps"] = 3
    trainer = MAPPOTrainer(env_config, {"num_envs": 2, "rollout_steps": 8, "ppo_epochs": 2, "minibatch_size": 8, "hidden_dim": 16, "seed": 12})
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
    assert trainer.actor.network[0].in_features == 55
    assert trainer.critic.network[0].in_features == 67
    completed = trainer.collect_rollout()
    assert completed and all(record["episode_length"] <= 3 for record in completed)
    assert np.all(np.isfinite(trainer.buffer.advantages)) and trainer.buffer.position == 8
    metrics = trainer.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic.parameters()))
    assert np.all(np.isfinite(trainer.buffer.returns))
    trainer.collect_rollout(); trainer.update(); trainer.close()
