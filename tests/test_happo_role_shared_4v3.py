from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.environment_4v3_v12 import FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv
from uav_combat.happo.evaluation_role_shared_4v3 import evaluate_role_shared_happo_fixed_blue_4v3
from uav_combat.happo.role_shared_buffer import RoleSharedRolloutBuffer4v3
from uav_combat.happo.role_shared_networks import RoleHiddenState, RoleSharedHAPPOActors
from uav_combat.happo.trainer_role_shared_4v3 import (
    RoleSharedHAPPO4v3Trainer,
    combat_alive_mean_entropy,
    combat_joint_log_probability,
    role_group_factor_update,
)

ENV = Path("configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")
V13A = Path("configs/happo_heterogeneous_4v3_main_v13a_role_shared_combat_mlp.yaml")
V13B = Path("configs/happo_heterogeneous_4v3_main_v13b_role_shared_combat_gru_mask.yaml")
V12 = Path("configs/happo_heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")


def _tiny(path: Path, *, recurrent: bool | None = None, num_envs: int = 2) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["experiment"].update({"device": "cpu", "seed": 941})
    cfg["training"].update({
        "num_envs": num_envs, "num_env_workers": 0, "rollout_steps": 4,
        "total_env_steps": 8, "schedule_env_steps": 8, "ppo_epochs": 1,
        "minibatch_size": 4, "evaluation_interval_env_steps": 8,
        "checkpoint_interval_env_steps": 8,
    })
    if recurrent is not None:
        cfg["training"]["recurrent_actor"] = recurrent
    cfg["evaluation"].update({"selection_episodes": 2, "test_episodes": 2})
    return cfg


def test_role_mapping_uses_distinct_support_and_one_shared_combat_actor_and_optimizer() -> None:
    trainer = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13A))
    try:
        assert trainer.actors.actor_for_slot(0) is trainer.actors.support_actor
        assert trainer.actors.actor_for_slot(0) is not trainer.actors.combat_actor
        assert trainer.actors.actor_for_slot(1) is trainer.actors.actor_for_slot(2)
        assert trainer.actors.actor_for_slot(2) is trainer.actors.actor_for_slot(3)
        assert set(trainer.actor_optimizers) == {"support", "combat"}
        assert trainer.actor_optimizers["combat"] is trainer.combat_optimizer
    finally:
        trainer.close()


def test_mlp_deterministic_actions_have_shape_and_shared_policy_semantics() -> None:
    actors = RoleSharedHAPPOActors(118, recurrent=False)
    obs = torch.randn(2, 4, 118)
    obs[:, 2] = obs[:, 1]
    alive = torch.ones(2, 4)
    actions, hidden = actors.deterministic_actions(obs, alive)
    assert hidden is None
    assert actions.shape == (2, 4, 3)
    assert torch.allclose(actions[:, 1], actions[:, 2], rtol=0.0, atol=1e-7)
    obs[:, 3] = obs[:, 1] + 1.0
    changed, _ = actors.deterministic_actions(obs, alive)
    assert not torch.equal(changed[:, 1], changed[:, 3])


def test_dead_slots_output_zero_and_do_not_advance_hidden() -> None:
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=16)
    obs = torch.randn(2, 4, 118)
    alive = torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.float32)
    hidden = actors.initial_hidden(2, "cpu")
    actions, log_probs, next_hidden = actors.sample_actions(obs, alive, hidden, alive)
    assert torch.equal(actions[0, 2], torch.zeros(3))
    assert torch.equal(actions[1], torch.zeros(4, 3))
    assert torch.equal(log_probs[1], torch.zeros(4))
    assert torch.equal(next_hidden.combat[0, 1], torch.zeros(16))
    assert torch.equal(next_hidden.support[1], torch.zeros(16))


def test_combat_joint_probability_ratio_and_entropy_masks_dead_slots() -> None:
    old = torch.tensor([[1.0, 2.0, 3.0], [0.2, 0.3, 0.4]])
    new = old + torch.tensor([[0.1, 9.0, -0.2], [0.5, 0.5, 0.5]])
    alive = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    old_joint = combat_joint_log_probability(old, alive)
    new_joint = combat_joint_log_probability(new, alive)
    assert torch.allclose(old_joint, torch.tensor([4.0, 0.3]))
    assert torch.allclose(torch.exp(new_joint - old_joint), torch.exp(torch.tensor([-0.1, 0.5])))
    entropy = combat_alive_mean_entropy(torch.tensor([[3.0, 99.0, 1.0], [8.0, 2.0, 4.0]]), alive)
    assert torch.allclose(entropy, torch.tensor([2.0, 2.0]))


def test_preceding_factor_uses_one_group_ratio_and_masks_inactive_samples() -> None:
    factor = torch.ones(3)
    old = torch.zeros(3)
    new = torch.log(torch.tensor([2.0, 3.0, 4.0]))
    updated = role_group_factor_update(factor, old, new, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(updated, torch.tensor([2.0, 1.0, 4.0]))
    assert not updated.requires_grad


def test_single_alive_combat_reduces_joint_log_probability_to_that_slot() -> None:
    values = torch.tensor([[4.0, -2.0, 8.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])
    assert combat_joint_log_probability(values, mask).item() == -2.0


def test_recurrent_hidden_shapes_and_slot_independence() -> None:
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=32)
    hidden = actors.initial_hidden(3, "cpu")
    assert hidden.support.shape == (3, 32)
    assert hidden.combat.shape == (3, 3, 32)
    hidden.combat[0, 0, 0] = 7.0
    assert hidden.combat[0, 1, 0].item() == 0.0


def test_recurrent_identical_observation_and_hidden_produce_identical_action() -> None:
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=16)
    obs = torch.randn(1, 4, 118)
    obs[:, 2] = obs[:, 1]
    hidden = actors.initial_hidden(1, "cpu")
    alive = torch.ones(1, 4)
    actions, _ = actors.deterministic_actions(obs, alive, hidden, torch.zeros_like(alive))
    assert torch.allclose(actions[:, 1], actions[:, 2], rtol=0.0, atol=1e-7)


def test_sequence_chunks_split_at_done_and_padding_is_invalid() -> None:
    buffer = RoleSharedRolloutBuffer4v3(7, 1, 5, 6, recurrent=True, recurrent_hidden_dim=3)
    buffer.dones[:, 0] = [False, True, False, False, True, False, False]
    chunks = buffer.sequence_chunks(3)
    assert [(c.start, c.stop) for c in chunks] == [(0, 2), (2, 5), (5, 7)]
    batch = buffer.padded_chunk_batch(chunks, 3)
    assert batch["valid_mask"].sum() == 7
    assert np.all(batch["factor_indices"][batch["valid_mask"] == 0] == -1)
    for chunk in chunks:
        assert not any(buffer.dones[chunk.start:chunk.stop - 1, chunk.env_index])


def test_gae_stops_at_episode_done() -> None:
    buffer = RoleSharedRolloutBuffer4v3(3, 1, 2, 2)
    buffer.position = 3
    buffer.team_rewards[:, 0] = [1.0, 10.0, 100.0]
    buffer.team_values.fill(0)
    buffer.dones[:, 0] = [True, True, False]
    buffer.compute_returns_and_advantages(np.array([0], np.float32), 1.0, 1.0)
    assert buffer.advantages[:, 0].tolist() == [1.0, 10.0, 100.0]


@pytest.mark.parametrize("config_path", [V13A, V13B])
def test_one_update_is_finite_changes_both_policy_groups_and_steps_each_optimizer(config_path: Path) -> None:
    trainer = RoleSharedHAPPO4v3Trainer(ENV, _tiny(config_path))
    support_before = [p.detach().clone() for p in trainer.actors.support_actor.parameters()]
    combat_before = [p.detach().clone() for p in trainer.actors.combat_actor.parameters()]
    try:
        trainer.collect_rollout()
        metrics = trainer.update()
        assert metrics["support_optimizer_steps"] > 0
        assert metrics["combat_optimizer_steps"] > 0
        assert metrics["group_update_order"] in {"support>combat", "combat>support"}
        assert all(np.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
        assert any(not torch.equal(a, b) for a, b in zip(support_before, trainer.actors.support_actor.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(combat_before, trainer.actors.combat_actor.parameters()))
        assert len(trainer.actor_optimizers) == 2
        if trainer.recurrent:
            assert metrics["recurrent_hidden_activity"] > 0.0
    finally:
        trainer.close()


def test_all_combat_dead_safely_skips_combat_update() -> None:
    trainer = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13A))
    try:
        trainer.collect_rollout()
        trainer.buffer.agent_alive_masks[:, :, 1:4] = 0.0
        metrics = trainer.update()
        assert metrics["combat_optimizer_steps"] == 0
        assert metrics["combat_active_time_env_samples"] == 0
        assert np.isfinite(metrics["policy_loss"])
    finally:
        trainer.close()


def test_reset_hidden_at_only_clears_selected_environment() -> None:
    trainer = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13B, num_envs=2))
    try:
        trainer.hidden.support.fill_(1.0); trainer.hidden.combat.fill_(2.0); trainer.hidden_reset_masks.fill(1.0)
        trainer.reset_hidden_at(1)
        assert torch.all(trainer.hidden.support[0] == 1.0)
        assert torch.all(trainer.hidden.combat[0] == 2.0)
        assert torch.all(trainer.hidden.support[1] == 0.0)
        assert torch.all(trainer.hidden.combat[1] == 0.0)
        assert np.all(trainer.hidden_reset_masks[1] == 0.0)
        assert np.all(trainer.hidden_reset_masks[0] == 1.0)
    finally:
        trainer.close()


@pytest.mark.parametrize("config_path", [V13A, V13B])
def test_checkpoint_roundtrip_preserves_models_optimizers_state_and_next_action(tmp_path: Path, config_path: Path) -> None:
    cfg = _tiny(config_path)
    trainer = RoleSharedHAPPO4v3Trainer(ENV, cfg)
    checkpoint = tmp_path / "v13.pt"
    try:
        trainer.collect_rollout(); trainer.update(); trainer.save_checkpoint(checkpoint)
        expected = trainer._select_actions()[:3]
    finally:
        trainer.close()
    restored = RoleSharedHAPPO4v3Trainer(ENV, cfg)
    try:
        restored.load_checkpoint(checkpoint)
        actual = restored._select_actions()[:3]
        for left, right in zip(expected, actual):
            assert np.array_equal(left, right)
        assert restored.env_steps == 8 and restored.update_count == 1
        assert len(restored.actor_optimizers) == 2
    finally:
        restored.close()


def test_v12_checkpoint_and_v13_architectures_cannot_be_mixed(tmp_path: Path) -> None:
    trainer_a = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13A))
    checkpoint = tmp_path / "v13a.pt"
    try:
        trainer_a.save_checkpoint(checkpoint)
    finally:
        trainer_a.close()
    trainer_b = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13B))
    try:
        with pytest.raises(ValueError, match="signature mismatch"):
            trainer_b.load_checkpoint(checkpoint)
        fake_v12 = tmp_path / "v12.pt"
        torch.save({"checkpoint_family": "functional_heterogeneous_4v3_v9_happo"}, fake_v12)
        with pytest.raises(ValueError, match="checkpoint family mismatch"):
            trainer_b.load_checkpoint(fake_v12)
    finally:
        trainer_b.close()


def test_deterministic_evaluation_repeats_and_reports_slot_dispersion() -> None:
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=16)
    seeds = [150042]
    first = evaluate_role_shared_happo_fixed_blue_4v3(actors, ENV, seeds=seeds, device="cpu")
    second = evaluate_role_shared_happo_fixed_blue_4v3(actors, ENV, seeds=seeds, device="cpu")
    assert first["task_win_rate"] == second["task_win_rate"]
    assert first["mean_return"] == second["mean_return"]
    for slot in (1, 2, 3):
        for metric in ("kills", "max_lock", "half_lock", "survival"):
            assert f"red_{slot}_{metric}" in first
    assert "combat_slot_kills_std" in first and "combat_slot_max_lock_range" in first


def test_v13_configs_freeze_v12_training_contract_except_algorithm_structure() -> None:
    baseline = yaml.safe_load(Path("outputs/happo_heterogeneous_4v3_main_v12_soft_boundary_combat_aligned_3m_seed42/resolved_training_config.yaml").read_text(encoding="utf-8"))
    keys = [
        "total_env_steps", "schedule_env_steps", "num_envs", "num_env_workers", "rollout_steps",
        "ppo_epochs", "minibatch_size", "gamma", "gae_lambda", "clip_coef", "actor_lr",
        "actor_lr_final", "critic_lr", "critic_lr_final", "entropy_coef", "entropy_coef_final",
        "evaluation_interval_env_steps", "checkpoint_interval_env_steps",
    ]
    for path in (V13A, V13B):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert {key: cfg["training"][key] for key in keys} == {key: baseline["training"][key] for key in keys}
        assert cfg["evaluation"] == baseline["evaluation"]


def test_frozen_v12_environment_reward_is_identical_for_identical_actions() -> None:
    left = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(ENV)
    right = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(ENV)
    left.reset(77); right.reset(77)
    actions = {f"red_{i}": np.array([0.2, -0.1, 0.3], np.float32) for i in range(4)}
    for _ in range(5):
        l = left.step(actions); r = right.step(actions)
        assert l[3] == r[3]
        assert l[6]["reward_components"] == r[6]["reward_components"]


def test_mlp_configuration_does_not_create_online_hidden() -> None:
    trainer = RoleSharedHAPPO4v3Trainer(ENV, _tiny(V13A))
    try:
        assert trainer.hidden is None
        assert trainer.buffer.support_hidden_before is None
        assert trainer.buffer.combat_hidden_before is None
    finally:
        trainer.close()
