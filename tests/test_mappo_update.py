from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from uav_combat.mappo.trainer import MAPPOTrainer

CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {
        "experiment": {"seed": 3, "device": "cpu", "output_dir": str(tmp_path)},
        "network": {"hidden_dim": 32, "log_std_init": -0.5},
        "training": {
            "total_env_steps": 128, "num_envs": 2, "rollout_steps": 64, "ppo_epochs": 1,
            "minibatch_size": 128, "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
            "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01,
            "max_grad_norm": 0.5, "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
        },
        "evaluation": {"episodes": 2, "deterministic": True},
    }


def changed(before, module):
    return any(not torch.equal(value, module.state_dict()[key]) for key, value in before.items())


def test_collect_update_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3); np.random.seed(3)
    trainer = MAPPOTrainer(CONFIG, tiny_config(tmp_path))
    actor_before, critic_before = deepcopy(trainer.actor.state_dict()), deepcopy(trainer.critic.state_dict())
    trainer.collect_rollout(); metrics = trainer.update()
    assert all(np.isfinite(value) for value in metrics.values())
    assert changed(actor_before, trainer.actor) and changed(critic_before, trainer.critic)
    assert np.isfinite(metrics["actor_grad_norm"])
    observation = torch.zeros(1, 13)
    expected = trainer.actor.deterministic_action(observation).detach().clone()
    checkpoint = tmp_path / "roundtrip.pt"; trainer.save_checkpoint(checkpoint)
    restored = MAPPOTrainer(CONFIG, tiny_config(tmp_path)); restored.load_checkpoint(checkpoint)
    actual = restored.actor.deterministic_action(observation).detach()
    assert torch.equal(expected, actual)
