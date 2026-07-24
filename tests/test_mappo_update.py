from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_combat.mappo.trainer import MAPPOTrainer, SCENARIOS

ENV_CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {"experiment": {"seed": 3, "device": "cpu", "output_dir": str(tmp_path)}, "network": {"hidden_dim": 32, "log_std_init": -0.5}, "training": {"training_mode": "alternating_self_play", "total_env_steps": 256, "num_envs": 2, "rollout_steps": 16, "alternating_block_env_steps": 64, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5, "eval_interval_updates": 10, "checkpoint_interval_updates": 10, "opponent_history_latest_probability": 0.7}, "evaluation": {"episodes": 2, "deterministic": True}}


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


def test_v5_checkpoint_roundtrip_restores_models_history_and_frozen_opponent(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); trainer.collect_rollout(); trainer.update("red")
    path = tmp_path / "v5.pt"; trainer.save_checkpoint(path); raw = torch.load(path, weights_only=False)
    required = {"red_actor", "blue_actor", "red_critic", "blue_critic", "red_actor_optimizer", "blue_actor_optimizer", "red_critic_optimizer", "blue_critic_optimizer", "red_actor_history", "blue_actor_history", "red_generation_metadata", "blue_generation_metadata", "current_opponent_side", "current_opponent_generation", "current_behavior_actor_state_dict", "current_block_index", "opponent_history_latest_probability", "history_selection_counts", "environment_steps", "update", "config", "python_random_state", "numpy_rng_state", "opponent_numpy_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state"}
    assert raw["checkpoint_version"] == 5 and required <= raw.keys()
    restored = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); restored.load_checkpoint(path)
    for name in ("red_actor", "blue_actor", "red_critic", "blue_critic"): assert same_state(getattr(trainer, name).state_dict(), getattr(restored, name).state_dict())
    assert restored.env_steps == trainer.env_steps and restored.update_count == trainer.update_count
    assert restored.current_opponent_generation == trainer.current_opponent_generation
    assert restored.current_opponent_side == trainer.current_opponent_side
    assert len(restored.red_actor_history) == len(trainer.red_actor_history)
    assert all(value.device.type == "cpu" for state in restored.red_actor_history for value in state.values())


@pytest.mark.parametrize("version", [3, 4])
def test_v4_and_earlier_checkpoints_are_explicitly_rejected(tmp_path, version):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path)); old = tmp_path / f"v{version}.pt"; torch.save({"checkpoint_version": version}, old)
    with pytest.raises(RuntimeError, match="v4 and earlier"): trainer.load_checkpoint(old)


def test_invalid_old_training_mode_is_rejected(tmp_path):
    config = tiny_config(tmp_path); config["training"]["training_mode"] = "paper_staged"
    with pytest.raises(ValueError, match="alternating_self_play"): MAPPOTrainer(ENV_CONFIG, config)


class ControlledRng:
    def __init__(self, base, random_values):
        self.base = base
        self.random_values = iter(random_values)
    def random(self):
        return next(self.random_values)
    def integers(self, *args, **kwargs):
        return self.base.integers(*args, **kwargs)
    def permutation(self, *args, **kwargs):
        return self.base.permutation(*args, **kwargs)
    @property
    def bit_generator(self):
        return self.base.bit_generator


def test_generation_zero_is_cpu_deep_copy_and_stays_unchanged(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    initial = deepcopy(trainer.red_actor_history[0])
    assert all(value.device.type == "cpu" for value in initial.values())
    trainer.collect_rollout(); trainer.update("red")
    assert same_state(initial, trainer.red_actor_history[0])
    assert changed(initial, trainer.red_actor)


def test_single_generation_selects_zero_and_same_block_keeps_it(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    first = trainer.configure_block_opponent(0, "red")
    trainer.collect_rollout(32)
    second = trainer.configure_block_opponent(0, "red")
    assert first["opponent_generation"] == second["opponent_generation"] == 0
    assert first["opponent_is_latest"] and second["opponent_is_latest"]
    assert len(trainer.block_history) == 1


def test_controlled_latest_and_old_generation_selection(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    trainer.blue_actor_history.extend([deepcopy(trainer.blue_actor_history[0]), deepcopy(trainer.blue_actor_history[0])])
    trainer.opponent_rng = ControlledRng(np.random.default_rng(4), [.69, .70])
    generation, latest = trainer._select_opponent_generation("blue")
    assert generation == 2 and latest
    generation, latest = trainer._select_opponent_generation("blue")
    assert generation < 2 and not latest


def test_behavior_actor_and_frozen_training_state_do_not_update(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    trainer.configure_block_opponent(0, "red")
    behavior = deepcopy(trainer.blue_behavior_actor.state_dict())
    actor = deepcopy(trainer.blue_actor.state_dict()); critic = deepcopy(trainer.blue_critic.state_dict())
    actor_opt = deepcopy(trainer.blue_actor_optimizer.state_dict()); critic_opt = deepcopy(trainer.blue_critic_optimizer.state_dict())
    trainer.collect_rollout(); trainer.update("red")
    assert same_state(behavior, trainer.blue_behavior_actor.state_dict())
    assert same_state(actor, trainer.blue_actor.state_dict()) and same_state(critic, trainer.blue_critic.state_dict())
    assert actor_opt == trainer.blue_actor_optimizer.state_dict() and critic_opt == trainer.blue_critic_optimizer.state_dict()
    assert not trainer.blue_behavior_actor.training
    assert all(not parameter.requires_grad for parameter in trainer.blue_behavior_actor.parameters())


def test_block_finish_appends_only_active_generation(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    trainer.configure_block_opponent(0, "red"); trainer.collect_rollout(64); trainer.update("red")
    trainer.finish_block("red", 0)
    assert len(trainer.red_actor_history) == 2 and len(trainer.blue_actor_history) == 1
    assert trainer.red_generation_metadata[-1]["block_index"] == 0
    trainer.finish_block("red", 0)
    assert len(trainer.red_actor_history) == 2


def test_tiny_four_block_training_exercises_latest_and_old_paths(tmp_path):
    trainer = MAPPOTrainer(ENV_CONFIG, tiny_config(tmp_path))
    trainer.opponent_rng = ControlledRng(np.random.default_rng(3), [.1, .9, .1])
    total = trainer.config["training"]["total_env_steps"]
    while trainer.env_steps < total:
        block = trainer.block_index(); active = trainer.active_side(); end = (block + 1) * trainer.block_env_steps
        trainer.configure_block_opponent(block, active)
        while trainer.env_steps < end:
            trainer.collect_rollout(end - trainer.env_steps); trainer.update(active)
        trainer.finish_block(active, block)
        if trainer.env_steps < total: trainer.reset_environments()
    assert trainer.env_steps == 256
    assert [row["active_side"] for row in trainer.block_history] == ["red", "blue", "red", "blue"]
    assert trainer.history_selection_counts["latest"] >= 1 and trainer.history_selection_counts["old"] >= 1
    assert len(trainer.red_actor_history) == len(trainer.blue_actor_history) == 3
