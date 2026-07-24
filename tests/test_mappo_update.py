from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_combat.mappo.trainer import MAPPOTrainer, SCENARIOS

ENV_CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {"experiment": {"seed": 3, "device": "cpu", "output_dir": str(tmp_path)}, "network": {"hidden_dim": 32, "log_std_init": -0.5}, "training": {"training_mode": "alternating_self_play", "total_env_steps": 256, "num_envs": 2, "rollout_steps": 16, "alternating_block_env_steps": 64, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5, "eval_interval_updates": 10, "checkpoint_interval_updates": 10}, "evaluation": {"episodes": 2, "deterministic": True}}


def changed(before, module):
    return any(not torch.equal(value, module.state_dict()[key]) for key, value in before.items())


def same_state(left, right):
    return all(torch.equal(value, right[key]) for key, value in left.items())


def test_actors_start_equal_but_have_independent_parameters(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    assert same_state(trainer.red_actor.state_dict(), trainer.blue_actor.state_dict())
    assert all(red is not blue and red.data_ptr() != blue.data_ptr() for red, blue in zip(trainer.red_actor.parameters(), trainer.blue_actor.parameters()))


def test_two_scalar_critics_and_four_distinct_optimizers(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); state = torch.zeros(4, 14)
    assert trainer.red_critic(state).shape == trainer.blue_critic(state).shape == (4,)
    assert trainer.red_critic is not trainer.blue_critic
    assert len({id(trainer.red_actor_optimizer), id(trainer.blue_actor_optimizer), id(trainer.red_critic_optimizer), id(trainer.blue_critic_optimizer)}) == 4


def test_red_block_updates_only_red_actor_critic_and_optimizers(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    before = {name: deepcopy(getattr(trainer, name).state_dict()) for name in ("red_actor", "blue_actor", "red_critic", "blue_critic")}
    blue_actor_opt = deepcopy(trainer.blue_actor_optimizer.state_dict()); blue_critic_opt = deepcopy(trainer.blue_critic_optimizer.state_dict())
    trainer.collect_rollout(); metrics = trainer.update("red")
    assert changed(before["red_actor"], trainer.red_actor) and changed(before["red_critic"], trainer.red_critic)
    assert same_state(before["blue_actor"], trainer.blue_actor.state_dict()) and same_state(before["blue_critic"], trainer.blue_critic.state_dict())
    assert blue_actor_opt == trainer.blue_actor_optimizer.state_dict() and blue_critic_opt == trainer.blue_critic_optimizer.state_dict()
    assert np.isfinite(metrics["red_policy_loss"]) and np.isnan(metrics["blue_policy_loss"])


def test_blue_block_updates_only_blue_actor_and_critic(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); trainer.env_steps = 64
    red_actor = deepcopy(trainer.red_actor.state_dict()); red_critic = deepcopy(trainer.red_critic.state_dict())
    blue_actor = deepcopy(trainer.blue_actor.state_dict()); blue_critic = deepcopy(trainer.blue_critic.state_dict())
    trainer.collect_rollout(); trainer.update("blue")
    assert same_state(red_actor, trainer.red_actor.state_dict()) and same_state(red_critic, trainer.red_critic.state_dict())
    assert changed(blue_actor, trainer.blue_actor) and changed(blue_critic, trainer.blue_critic)


def test_active_side_alternates_at_exact_block_boundaries(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    for steps, side, index in ((0, "red", 0), (63, "red", 0), (64, "blue", 1), (128, "red", 2)):
        trainer.env_steps = steps; assert trainer.active_side() == side and trainer.block_index() == index


def test_both_actors_sample_actions_and_values_keep_t_n_2(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); trainer.collect_rollout()
    assert trainer.buffer.actions.shape == (16, 2, 2, 3) and trainer.buffer.values.shape == (16, 2, 2)
    assert np.any(trainer.buffer.actions[:, :, 0] != 0) and np.any(trainer.buffer.actions[:, :, 1] != 0)


def test_exact_partial_rollout_stops_on_boundary(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); trainer.collect_rollout(32)
    assert trainer.env_steps == 32 and trainer.buffer.rollout_steps == 16
    trainer.collect_rollout(32); assert trainer.env_steps == 64


def test_scenarios_use_deterministic_balanced_cycle(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    assert trainer.current_scenarios == list(SCENARIOS[:2])
    observed = []
    for _ in range(6):
        _, scenario, _ = trainer._next_reset(trainer.envs[0]); observed.append(scenario)
    assert observed == ["crossing", "tail_chase", "offset_head_on", "crossing", "tail_chase", "offset_head_on"]


def test_tail_chase_rear_team_alternates_and_is_counted(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); rears = []
    for _ in range(10):
        _, scenario, rear = trainer._next_reset(trainer.envs[0])
        if scenario == "tail_chase": rears.append(rear)
    assert rears == ["blue", "red", "blue"]
    assert abs(trainer.tail_rear_counts["red"] - trainer.tail_rear_counts["blue"]) <= 1


def test_v4_checkpoint_roundtrip_restores_all_models_and_counters(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); trainer.collect_rollout(); trainer.update("red")
    path = tmp_path / "v4.pt"; trainer.save_checkpoint(path); raw = torch.load(path, weights_only=False)
    required = {"red_actor", "blue_actor", "red_critic", "blue_critic", "red_actor_optimizer", "blue_actor_optimizer", "red_critic_optimizer", "blue_critic_optimizer", "environment_steps", "update", "active_side", "alternating_block_index", "config", "python_random_state", "numpy_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state"}
    assert raw["checkpoint_version"] == 4 and required <= raw.keys()
    restored = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); restored.load_checkpoint(path)
    for name in ("red_actor", "blue_actor", "red_critic", "blue_critic"): assert same_state(getattr(trainer, name).state_dict(), getattr(restored, name).state_dict())
    assert restored.env_steps == trainer.env_steps and restored.update_count == trainer.update_count


def test_v3_checkpoint_is_explicitly_rejected(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); old = tmp_path / "v3.pt"; torch.save({"checkpoint_version": 3}, old)
    with pytest.raises(RuntimeError, match="v3"): trainer.load_checkpoint(old)


def test_invalid_old_training_mode_is_rejected(tmp_path):
    config = tiny_config(tmp_path); config["training"]["training_mode"] = "paper_staged"
    with pytest.raises(ValueError, match="alternating_self_play"): MAPPOTrainer(ENV_CONFIG, config)
