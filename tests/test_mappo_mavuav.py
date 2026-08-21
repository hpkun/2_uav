import numpy as np
import torch
from uav_combat.mappo import MAPPOTrainer
from copy import deepcopy
from uav_combat.mavuav import load_environment_config


def test_mappo_rollout_gae_updates_are_finite_and_change_parameters():
    env_config = deepcopy(load_environment_config(None)); env_config["simulation"]["max_decision_steps"] = 3
    trainer = MAPPOTrainer(env_config, {"num_envs": 2, "rollout_steps": 8, "ppo_epochs": 2, "minibatch_size": 8, "hidden_dim": 16, "seed": 12})
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    completed = trainer.collect_rollout()
    assert completed and all(record["episode_length"] <= 3 for record in completed)
    assert np.all(np.isfinite(trainer.buffer.advantages)) and trainer.buffer.position == 8
    metrics = trainer.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters()))
    trainer.collect_rollout(); trainer.update(); trainer.close()
