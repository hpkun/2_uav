from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.environment_4v3_v11 import lock_quality_v11
from uav_combat.environment_4v3_v17 import (
    FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv,
)
from uav_combat.environment_4v3_v18 import (
    FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv,
    fire_quality_v18,
)
from uav_combat.happo.evaluation_v14_4v3 import (
    evaluate_v14_happo_fixed_blue_4v3,
)
from uav_combat.happo.recurrent_role_credit_buffer import (
    RecurrentAgentCreditRolloutBuffer4v3,
)
from uav_combat.happo.role_shared_networks import (
    RecurrentHAPPOGaussianActor,
    RoleSharedHAPPOActors,
)
from uav_combat.happo.trainer_v14_4v3 import (
    MissionAlignedRoleSharedHAPPO4v3Trainer,
)
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]
ENV_V17 = ROOT / "configs/heterogeneous_4v3_main_v17_role_situation_event_mission_reward.yaml"
ENV_V18 = ROOT / "configs/heterogeneous_4v3_main_v18_recurrent_fire_geometry.yaml"
TRAIN_V17 = ROOT / "configs/happo_heterogeneous_4v3_main_v17_role_shared_combat_mlp_role_situation_event_mission_reward.yaml"
TRAIN_V18 = ROOT / "configs/happo_heterogeneous_4v3_main_v18_role_shared_combat_gru_role_situation_event_mission_reward.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny_v18(*, num_envs: int = 2, rollout_steps: int = 4) -> dict:
    config = _config(TRAIN_V18)
    config["experiment"]["device"] = "cpu"
    config["training"].update(
        {
            "num_envs": num_envs,
            "num_env_workers": 0,
            "rollout_steps": rollout_steps,
            "total_env_steps": num_envs * rollout_steps,
            "schedule_env_steps": num_envs * rollout_steps,
            "ppo_epochs": 1,
            "minibatch_size": num_envs * rollout_steps,
            "sequence_chunk_length": min(2, rollout_steps),
        }
    )
    config["evaluation"].update(selection_episodes=1, test_episodes=1)
    return config


def _state(*, bearing: float = 0.0, target_heading: float = 0.0, distance: float = 1000.0):
    attacker = AircraftState(0.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    target = AircraftState(
        distance * math.cos(bearing),
        distance * math.sin(bearing),
        -3000.0,
        150.0,
        0.0,
        target_heading,
    )
    return attacker, target


def _parameter_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [value.detach().clone() for value in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


@pytest.mark.parametrize(
    ("ata", "expected"),
    [(0.0, 1.0), (math.pi / 4.0, 0.5), (math.pi / 2.0, 0.0)],
)
def test_v18_fire_quality_ata_endpoints(ata, expected):
    profile = _config(ENV_V18)["combat_profile"]
    attacker, target = _state(bearing=ata)
    assert fire_quality_v18(attacker, target, profile) == pytest.approx(expected, abs=2e-10)


def test_v18_fire_quality_directly_calls_v11_distance(monkeypatch):
    calls = []

    def fake(distance, profile):
        calls.append((distance, profile))
        return 0.37

    monkeypatch.setattr("uav_combat.environment_4v3_v18.distance_score_v11", fake)
    profile = _config(ENV_V18)["combat_profile"]
    attacker, target = _state()
    assert fire_quality_v18(attacker, target, profile) == pytest.approx(0.37)
    assert calls == [(pytest.approx(1000.0), profile)]


def test_v18_fire_quality_is_aa_independent_but_v17_remains_aa_dependent():
    profile = _config(ENV_V18)["combat_profile"]
    attacker, tail_chase = _state(target_heading=0.0)
    _, head_on = _state(target_heading=math.pi)
    q_tail = fire_quality_v18(attacker, tail_chase, profile)
    q_head = fire_quality_v18(attacker, head_on, profile)
    assert q_tail == pytest.approx(q_head) == pytest.approx(1.0)
    assert q_head > 0.0
    assert lock_quality_v11(attacker, tail_chase, profile) == pytest.approx(1.0)
    assert lock_quality_v11(attacker, head_on, profile) == pytest.approx(0.0)


def test_v18_lock_increment_decay_kill_and_red_blue_symmetry():
    env = FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv(ENV_V18)
    env.reset(1801)
    red = env._by_id("red_1")
    blue = env._by_id("blue_0")
    red.state, blue.state = _state(target_heading=math.pi)
    assert env._attack_quality(red.state, blue.state) == pytest.approx(1.0)
    assert env._attack_quality(blue.state, red.state) == pytest.approx(1.0)

    env.targets.update({key: None for key in env.targets})
    env.targets["red_1"] = "blue_0"
    env.lock_progress["red_1"] = 0.0
    direct = {aircraft.aircraft_id: set() for aircraft in env.aircraft}
    direct["red_1"].add("blue_0")
    env._update_locks(direct)
    assert env.lock_progress["red_1"] == pytest.approx(0.17)
    env.lock_progress["red_1"] = 0.90
    _, killers = env._update_locks(direct)
    assert env.lock_progress["red_1"] == pytest.approx(1.0)
    assert killers == {"blue_0": "red_1"}
    env.lock_progress["red_1"] = 0.50
    env._update_locks({key: set() for key in direct})
    assert env.lock_progress["red_1"] == pytest.approx(0.47)


def test_v18_freezes_v17_environment_reward_observation_target_and_cue_contracts():
    baseline = _config(ENV_V17)
    candidate = _config(ENV_V18)
    frozen = deepcopy(candidate)
    frozen["combat"] = deepcopy(baseline["combat"])
    assert frozen == baseline
    assert candidate["combat"]["observation_contract"] == "legacy_fixed_order"
    assert candidate["combat_profile"]["lock_increment_scale"] == 0.17
    assert candidate["combat_profile"]["lock_decay_per_step"] == 0.03
    assert candidate["combat_profile"]["lock_kill_threshold"] == 1.0
    assert candidate["combat_profile"]["target_min_hold_steps"] == 30
    assert candidate["combat_profile"]["target_lost_release_steps"] == 10
    assert candidate["combat_profile"]["target_switch_distance_ratio"] == 0.70
    old = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    new = FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv(ENV_V18)
    old_values = old.reset(1818)
    new_values = new.reset(1818)
    assert all(np.array_equal(left, right) for left, right in zip(old_values, new_values))
    assert new_values[0].shape == (7, 118)
    assert new_values[1].shape == (70,)


def test_v18_production_config_enables_only_actor_recurrence():
    config = _config(TRAIN_V18)
    training = config["training"]
    assert training["recurrent_actor"] is True
    assert training["recurrent_hidden_dim"] == 128
    assert training["recurrent_num_layers"] == 1
    assert training["sequence_chunk_length"] == 32
    assert training["mask_inactive_hidden"] is True
    assert training["num_envs"] == 16
    assert training["num_env_workers"] == 4
    assert training["rollout_steps"] == 256
    assert training["minibatch_size"] == 1024
    assert training["ppo_epochs"] == 5


def test_v18_support_and_shared_combat_are_recurrent_with_independent_hidden():
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=128)
    assert isinstance(actors.support_actor, RecurrentHAPPOGaussianActor)
    assert isinstance(actors.combat_actor, RecurrentHAPPOGaussianActor)
    assert actors.actor_for_slot(1) is actors.actor_for_slot(2) is actors.actor_for_slot(3)
    hidden = actors.initial_hidden(2, "cpu")
    assert hidden.support.shape == (2, 128)
    assert hidden.combat.shape == (2, 3, 128)
    hidden.combat[0, 0, 0] = 1.0
    assert hidden.combat[0, 1, 0] == 0.0


def test_recurrent_output_depends_on_hidden_and_sequence_history():
    torch.manual_seed(1802)
    actor = RecurrentHAPPOGaussianActor(118, 3, hidden_dim=128)
    observation = torch.zeros(1, 118)
    zero = torch.zeros(1, 128)
    nonzero = torch.ones(1, 128)
    action_zero, _ = actor.deterministic_step(observation, zero, torch.ones(1))
    action_nonzero, _ = actor.deterministic_step(observation, nonzero, torch.ones(1))
    assert not torch.equal(action_zero, action_nonzero)
    history_a = torch.ones(1, 118)
    history_b = -torch.ones(1, 118)
    _, hidden_a = actor.deterministic_step(history_a, zero, torch.zeros(1))
    _, hidden_b = actor.deterministic_step(history_b, zero, torch.zeros(1))
    current_a, _ = actor.deterministic_step(observation, hidden_a, torch.ones(1))
    current_b, _ = actor.deterministic_step(observation, hidden_b, torch.ones(1))
    assert not torch.equal(current_a, current_b)


def test_agent_death_resets_only_its_hidden_and_episode_reset_clears_all():
    torch.manual_seed(1803)
    actors = RoleSharedHAPPOActors(118, recurrent=True, recurrent_hidden_dim=128)
    hidden = actors.initial_hidden(1, "cpu")
    hidden.support.fill_(1.0)
    hidden.combat.fill_(1.0)
    observations = torch.randn(1, 4, 118)
    alive = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    _, next_hidden = actors.deterministic_actions(
        observations, alive, hidden, torch.ones_like(alive)
    )
    assert torch.count_nonzero(next_hidden.combat[:, 0]) == 0
    assert torch.count_nonzero(next_hidden.combat[:, 1:]) > 0

    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V18, _tiny_v18())
    try:
        trainer.hidden.support.fill_(1.0)
        trainer.hidden.combat.fill_(1.0)
        trainer.hidden_reset_masks.fill(1.0)
        trainer.reset_hidden_at(0)
        assert torch.count_nonzero(trainer.hidden.support[0]) == 0
        assert torch.count_nonzero(trainer.hidden.combat[0]) == 0
        assert np.count_nonzero(trainer.hidden_reset_masks[0]) == 0
        assert torch.count_nonzero(trainer.hidden.support[1]) > 0
        assert torch.count_nonzero(trainer.hidden.combat[1]) > 0
    finally:
        trainer.close()


def _filled_buffer() -> RecurrentAgentCreditRolloutBuffer4v3:
    buffer = RecurrentAgentCreditRolloutBuffer4v3(5, 1, 2, 3, recurrent_hidden_dim=4)
    for step in range(5):
        buffer.add(
            np.full((1, 4, 2), step, np.float32),
            np.zeros((1, 3), np.float32),
            np.zeros((1, 4, 3), np.float32),
            np.zeros((1, 4), np.float32),
            np.ones((1, 4), np.float32),
            np.ones((1, 4), np.float32),
            np.zeros(1, np.float32),
            np.zeros((1, 4), np.float32),
            np.zeros((1, 4), np.float32),
            np.array([step == 1]),
            support_hidden_before=np.full((1, 4), step, np.float32),
            combat_hidden_before=np.full((1, 3, 4), step, np.float32),
        )
    return buffer


def test_recurrent_role_credit_chunks_stop_at_done_and_padding_is_invalid():
    buffer = _filled_buffer()
    chunks = buffer.sequence_chunks(3)
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 2), (2, 5)]
    batch = buffer.padded_chunk_batch(chunks, 3)
    assert batch["valid_mask"].tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    assert batch["factor_indices"][0, 2] == -1
    assert np.array_equal(batch["support_initial_hidden"][:, 0], [0.0, 2.0])
    assert batch["advantages"].shape == (2, 3, 4)


def test_recurrent_role_credit_gae_ignores_individual_death_and_stops_at_done():
    buffer = RecurrentAgentCreditRolloutBuffer4v3(3, 1, 1, 1, recurrent_hidden_dim=2)
    alive = [np.ones((1, 4), np.float32), np.array([[1, 0, 1, 1]], np.float32), np.ones((1, 4), np.float32)]
    rewards = [np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32), np.array([[0, 5, 0, 0]], np.float32)]
    for step in range(3):
        buffer.add(
            np.zeros((1, 4, 1), np.float32), np.zeros((1, 1), np.float32),
            np.zeros((1, 4, 3), np.float32), np.zeros((1, 4), np.float32), alive[step],
            alive[step], np.zeros(1, np.float32), rewards[step],
            np.zeros((1, 4), np.float32), np.array([step == 2]),
            support_hidden_before=np.zeros((1, 2), np.float32),
            combat_hidden_before=np.zeros((1, 3, 2), np.float32),
        )
    buffer.compute_returns_and_advantages(np.zeros((1, 4), np.float32), 1.0, 1.0)
    assert buffer.advantages[:, 0, 1].tolist() == pytest.approx([5.0, 5.0, 5.0])


def test_v18_real_recurrent_role_local_ppo_updates_all_trainable_groups(monkeypatch):
    config = _tiny_v18(num_envs=2, rollout_steps=4)
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V18, config)
    support_before = _parameter_snapshot(trainer.actors.support_actor)
    combat_before = _parameter_snapshot(trainer.actors.combat_actor)
    support_critic_before = _parameter_snapshot(trainer.role_critics.support_critic)
    combat_critic_before = _parameter_snapshot(trainer.role_critics.combat_critic)
    joint_calls = []
    import uav_combat.happo.trainer_v14_4v3 as trainer_module
    original_joint = trainer_module.combat_joint_log_probability

    def tracked_joint(log_probs, alive_masks):
        joint_calls.append((tuple(log_probs.shape), tuple(alive_masks.shape)))
        return original_joint(log_probs, alive_masks)

    monkeypatch.setattr(trainer_module, "combat_joint_log_probability", tracked_joint)
    try:
        trainer.collect_rollout()
        metrics = trainer.update()
        assert _changed(support_before, trainer.actors.support_actor)
        assert _changed(combat_before, trainer.actors.combat_actor)
        assert _changed(support_critic_before, trainer.role_critics.support_critic)
        assert _changed(combat_critic_before, trainer.role_critics.combat_critic)
        assert joint_calls
        assert metrics["support_optimizer_steps"] > 0
        assert metrics["combat_optimizer_steps"] > 0
        assert metrics["recurrent_hidden_activity"] > 0.0
        assert np.isfinite([value for value in metrics.values() if isinstance(value, (int, float))]).all()
    finally:
        trainer.close()


def test_v18_checkpoint_resume_next_step_and_hidden_are_exact(tmp_path: Path):
    config = _tiny_v18()
    original = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V18, config)
    restored = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V18, config)
    try:
        original.collect_rollout()
        original.update()
        checkpoint = tmp_path / "v18.pt"
        original.save_checkpoint(checkpoint, scheduled_env_steps=original.env_steps)
        expected = original._select_actions()
        restored.load_checkpoint(checkpoint)
        actual = restored._select_actions()
        for left, right in zip(expected[:3], actual[:3]):
            assert np.array_equal(left, right)
        assert torch.equal(expected[3].support, actual[3].support)
        assert torch.equal(expected[3].combat, actual[3].combat)
        assert torch.equal(original.hidden.support, restored.hidden.support)
        assert torch.equal(original.hidden.combat, restored.hidden.combat)
        assert np.array_equal(original.hidden_reset_masks, restored.hidden_reset_masks)
    finally:
        original.close()
        restored.close()


def test_v18_deterministic_recurrent_evaluation_is_exact():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(
        ENV_V18, _tiny_v18(num_envs=1, rollout_steps=2)
    )
    try:
        first = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V18, seeds=[91801], num_envs=1, device="cpu"
        )
        second = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V18, seeds=[91801], num_envs=1, device="cpu"
        )
        assert first["episode_records"] == second["episode_records"]
        assert first["recurrent_actor"] is True
    finally:
        trainer.close()


def test_v17_mlp_and_checkpoint_signature_remain_isolated_from_v18(tmp_path: Path):
    old_config = _config(TRAIN_V17)
    old_config["experiment"]["device"] = "cpu"
    old_config["training"].update(
        num_envs=1, num_env_workers=0, rollout_steps=2, total_env_steps=2,
        schedule_env_steps=2, ppo_epochs=1, minibatch_size=2,
    )
    old = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V17, old_config)
    new = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V18, _tiny_v18(num_envs=1, rollout_steps=2))
    try:
        assert old.recurrent is False
        assert new.recurrent is True
        path = tmp_path / "v18-signature.pt"
        torch.save(
            {
                "checkpoint_family": new.training_signature()["checkpoint_family"],
                "training_signature": new.training_signature(),
            },
            path,
        )
        with pytest.raises(ValueError, match="training signature mismatch"):
            old.load_checkpoint(path)
    finally:
        old.close()
        new.close()
