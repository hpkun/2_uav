import numpy as np
import torch
from uav_combat.happo import HAPPOTrainer
from copy import deepcopy
from uav_combat.mavuav import load_environment_config


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
