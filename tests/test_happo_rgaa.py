from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.common.buffer import RolloutBuffer
from algorithm.happo.rgaa import (
    RGAA_METHOD, ROLE_AUX_REWARD_MODE, RoleAdvantageRolloutBuffer,
    extract_rgaa_auxiliary_rewards,
)
from algorithm.happo.trainer import HAPPOTrainer, preceding_factor_update
from algorithm.train_happo import _algorithm_name
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config


ROOT = Path(__file__).resolve().parents[1]


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(**updates):
    config = {
        "method_variant": RGAA_METHOD, "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 1,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 16, "seed": 17, "role_advantage_coef": 0.5,
    }
    config.update(updates)
    return config


def test_vanilla_buffer_is_unchanged_and_rgaa_uses_role_buffer():
    vanilla = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    rgaa = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert type(vanilla.buffer) is RolloutBuffer
        assert isinstance(rgaa.buffer, RoleAdvantageRolloutBuffer)
    finally:
        vanilla.close(); rgaa.close()


def _aux_info(death_causes=None, **updates):
    info = {
        "mav_process_reward": 1.25, "uav1_process_reward": -2.0,
        "uav2_process_reward": 3.5, "uav3_process_reward": 4.75,
        "death_causes": death_causes or {},
    }
    info.update(updates)
    return info


def _reward_config():
    return {"mav_loss": -73.0, "uav_loss": -11.0}


def test_no_death_auxiliary_reward_equals_exact_process_reward():
    infos = [{
        "mav_process_reward": 1.25, "uav1_process_reward": -2.0,
        "uav2_process_reward": 3.5, "uav3_process_reward": 4.75,
        "death_causes": {}, "event_reward": 999.0,
    }]
    batch = extract_rgaa_auxiliary_rewards(infos, _reward_config())
    np.testing.assert_array_equal(
        batch.process_rewards, np.asarray([[1.25, -2.0, 3.5, 4.75]], np.float32),
    )
    np.testing.assert_array_equal(batch.auxiliary_rewards, batch.process_rewards)
    assert not batch.own_loss_events.any()


def test_uav_boundary_and_blue_attack_losses_are_agent_local_and_config_driven():
    boundary = extract_rgaa_auxiliary_rewards(
        [_aux_info({"UAV1": "boundary"})], _reward_config(),
    )
    np.testing.assert_array_equal(
        boundary.auxiliary_rewards,
        np.asarray([[1.25, -13.0, 3.5, 4.75]], np.float32),
    )
    np.testing.assert_array_equal(boundary.own_loss_events, [[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(boundary.boundary_loss_events, boundary.own_loss_events)

    attacked = extract_rgaa_auxiliary_rewards(
        [_aux_info({"UAV2": "blue_attack"})], _reward_config(),
    )
    np.testing.assert_array_equal(
        attacked.auxiliary_rewards,
        np.asarray([[1.25, -2.0, -7.5, 4.75]], np.float32),
    )
    np.testing.assert_array_equal(attacked.own_loss_events, [[0.0, 0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(attacked.blue_attack_loss_events, attacked.own_loss_events)


@pytest.mark.parametrize("cause", ["boundary", "blue_attack"])
def test_mav_own_loss_uses_configured_mav_penalty(cause):
    batch = extract_rgaa_auxiliary_rewards(
        [_aux_info({"MAV": cause})], _reward_config(),
    )
    np.testing.assert_array_equal(
        batch.auxiliary_rewards,
        np.asarray([[-71.75, -2.0, 3.5, 4.75]], np.float32),
    )
    np.testing.assert_array_equal(batch.own_loss_events, [[1.0, 0.0, 0.0, 0.0]])


def test_blue_kill_shared_event_and_other_uav_loss_do_not_pollute_auxiliary_rewards():
    batch = extract_rgaa_auxiliary_rewards(
        [_aux_info(
            {"Blue1": "red_attack", "UAV3": "blue_attack"},
            event_reward=9999.0, terminal_reward=8888.0, safety_reward=7777.0,
        )],
        _reward_config(),
    )
    np.testing.assert_array_equal(
        batch.auxiliary_rewards,
        np.asarray([[1.25, -2.0, 3.5, -6.25]], np.float32),
    )
    assert batch.auxiliary_rewards[0, 0] == batch.process_rewards[0, 0]
    assert batch.auxiliary_rewards[0, 1] == batch.process_rewards[0, 1]
    assert batch.auxiliary_rewards[0, 2] == batch.process_rewards[0, 2]


def test_role_gae_matches_team_boundary_semantics_for_terminated_and_truncated():
    buffer = RoleAdvantageRolloutBuffer(3, 2)
    buffer.position = 3
    buffer.role_rewards[:] = 1.0
    buffer.role_values[:] = 0.0
    buffer.active_masks[:] = 1.0
    buffer.terminated[1, 0] = True
    buffer.truncated[1, 1] = True
    buffer.compute_role_returns_and_advantages(
        np.full((2, 4), 9.0, np.float32), np.ones((2, 4), np.float32), 1.0, 1.0,
    )
    np.testing.assert_array_equal(buffer.role_advantages[:, 0, 0], [2.0, 1.0, 10.0])
    np.testing.assert_array_equal(buffer.role_advantages[:, 1, 0], [2.0, 1.0, 10.0])


def test_role_gae_stops_at_individual_uav_death_but_alive_uav_continues():
    buffer = RoleAdvantageRolloutBuffer(3, 1)
    buffer.position = 3
    buffer.role_rewards[:] = 1.0
    buffer.role_values[:] = 0.0
    buffer.active_masks[:] = 1.0
    # UAV1 is alive before action t=0, dies during that transition, while the episode continues.
    buffer.active_masks[1:, 0, 1] = 0.0
    buffer.role_rewards[0, 0, 1] = -10.0
    buffer.role_rewards[1:, 0, 1] = 100.0
    buffer.compute_role_returns_and_advantages(
        np.zeros((1, 4), np.float32),
        np.asarray([[1.0, 0.0, 1.0, 1.0]], np.float32),
        1.0, 1.0,
    )
    assert buffer.role_advantages[0, 0, 1] == -10.0
    np.testing.assert_array_equal(buffer.role_advantages[:, 0, 2], [3.0, 2.0, 1.0])


def test_role_gae_last_step_uses_trainer_bootstrap_active_mask():
    buffer = RoleAdvantageRolloutBuffer(1, 1)
    buffer.position = 1
    buffer.role_rewards[:] = 1.0
    buffer.role_values[:] = 2.0
    buffer.active_masks[:] = 1.0
    buffer.compute_role_returns_and_advantages(
        np.full((1, 4), 100.0, np.float32),
        np.asarray([[1.0, 0.0, 1.0, 1.0]], np.float32),
        1.0, 1.0,
    )
    assert buffer.role_advantages[0, 0, 1] == -1.0
    assert buffer.role_advantages[0, 0, 2] == 99.0


def test_coef_zero_combined_advantage_is_exact_team_normalized_advantage():
    trainer = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.0))
    try:
        trainer.collect_rollout(); trainer.update()
        assert torch.equal(
            trainer.last_rgaa_combined_advantages,
            trainer.last_rgaa_team_normalized_advantages,
        )
    finally:
        trainer.close()


def _force_nonzero_role_advantages(trainer: HAPPOTrainer) -> None:
    assert isinstance(trainer.buffer, RoleAdvantageRolloutBuffer)
    sample_count = trainer.buffer.horizon * trainer.buffer.num_envs
    pattern = np.linspace(-2.0, 2.0, sample_count, dtype=np.float32)
    for agent in range(len(RED_IDS)):
        trainer.buffer.role_advantages[:, :, agent] = pattern.reshape(
            trainer.buffer.horizon, trainer.buffer.num_envs,
        ) + np.float32(agent * 0.25)


def test_rgaa_combined_advantage_is_exact_formula_and_actual_ppo_input():
    trainer = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.5))
    try:
        trainer.collect_rollout()
        _force_nonzero_role_advantages(trainer)
        trainer.update()
        expected = (
            trainer.last_rgaa_team_normalized_advantages
            + 0.5 * trainer.last_rgaa_role_normalized_advantages
        )
        assert torch.equal(trainer.last_rgaa_combined_advantages, expected)
        assert torch.equal(
            trainer.last_rgaa_ppo_normalized_advantages,
            trainer.last_rgaa_combined_advantages,
        )
        active = torch.as_tensor(trainer.buffer.active_masks.reshape(-1, len(RED_IDS))) > 0.5
        assert torch.any(
            trainer.last_rgaa_combined_advantages[active]
            != trainer.last_rgaa_team_normalized_advantages[active]
        )
    finally:
        trainer.close()


def test_own_loss_auxiliary_signal_reaches_actor_ppo_input():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(
            role_advantage_coef=0.5, num_envs=2, rollout_steps=1, minibatch_size=8,
        ),
    )
    try:
        trainer.collect_rollout()
        batch = extract_rgaa_auxiliary_rewards(
            [
                _aux_info({"UAV1": "boundary"}, mav_process_reward=0.0,
                          uav1_process_reward=0.0, uav2_process_reward=0.0,
                          uav3_process_reward=0.0),
                _aux_info({}, mav_process_reward=0.0, uav1_process_reward=0.0,
                          uav2_process_reward=0.0, uav3_process_reward=0.0),
            ],
            trainer.environment_config["reward"],
        )
        trainer.buffer.role_rewards[0] = batch.auxiliary_rewards
        trainer.buffer.role_values[0] = 0.0
        last_active = np.ones((2, len(RED_IDS)), np.float32)
        last_active[0, 1] = 0.0
        trainer.buffer.compute_role_returns_and_advantages(
            np.zeros((2, len(RED_IDS)), np.float32), last_active, 1.0, 1.0,
        )
        trainer.update()
        active = torch.as_tensor(trainer.buffer.active_masks.reshape(-1, len(RED_IDS)))[:, 1] > 0.5
        assert torch.any(trainer.last_rgaa_role_normalized_advantages[active, 1] != 0.0)
        assert torch.any(
            trainer.last_rgaa_combined_advantages[active, 1]
            != trainer.last_rgaa_team_normalized_advantages[active, 1]
        )
        assert torch.equal(
            trainer.last_rgaa_ppo_normalized_advantages[:, 1],
            trainer.last_rgaa_combined_advantages[:, 1],
        )
    finally:
        trainer.close()


def _controlled_main_update(method: str, coefficient: float):
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(method_variant=method, role_advantage_coef=coefficient),
    )
    try:
        trainer.collect_rollout()
        rollout = {
            "observations": trainer.buffer.observations.copy(),
            "actions": trainer.buffer.actions.copy(),
            "advantages": trainer.buffer.advantages.copy(),
        }
        if method == RGAA_METHOD:
            _force_nonzero_role_advantages(trainer)
        trainer.update()
        return {
            "rollout": rollout,
            "actors": [parameter.detach().clone() for parameter in trainer.actors.parameters()],
            "critic": [parameter.detach().clone() for parameter in trainer.critic.parameters()],
        }
    finally:
        trainer.close()


def test_controlled_role_signal_changes_actor_update_but_not_team_critic():
    vanilla = _controlled_main_update("baseline", 0.0)
    zero = _controlled_main_update(RGAA_METHOD, 0.0)
    guided = _controlled_main_update(RGAA_METHOD, 0.5)
    for key in vanilla["rollout"]:
        assert np.array_equal(vanilla["rollout"][key], zero["rollout"][key])
        assert np.array_equal(vanilla["rollout"][key], guided["rollout"][key])
    assert all(torch.equal(left, right) for left, right in zip(vanilla["actors"], zero["actors"]))
    assert any(not torch.equal(left, right) for left, right in zip(vanilla["actors"], guided["actors"]))
    assert all(torch.equal(left, right) for left, right in zip(vanilla["critic"], zero["critic"]))
    assert all(torch.equal(left, right) for left, right in zip(vanilla["critic"], guided["critic"]))


def test_uavs_share_one_role_critic_while_all_actors_remain_independent():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert trainer.mav_role_critic is not trainer.uav_role_critic
        assert len({id(actor) for actor in trainer.actors.actors}) == len(RED_IDS)
        assert trainer.rgaa_metadata["role_critic_sharing"]["UAV1-UAV3"] == "shared"
    finally:
        trainer.close()


def test_shared_uav_role_critic_receives_all_active_uav_samples():
    trainer = HAPPOTrainer(short_v39(), trainer_config(minibatch_size=100))
    seen = []
    hook = trainer.uav_role_critic.register_forward_hook(
        lambda _module, inputs, _output: seen.append(int(inputs[0].shape[0])),
    )
    try:
        observations = torch.randn(5, 4, OBS_DIM)
        targets = torch.randn(5, 4)
        active = torch.ones(5, 4)
        trainer._train_role_critics(observations, targets, active)
        assert seen == [15]
    finally:
        hook.remove(); trainer.close()


def test_dead_uav_samples_have_zero_role_correction():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        trainer.buffer.active_masks[:, :, 2] = 0.0
        trainer.buffer.role_advantages[:, :, 2] = 123.0
        trainer.update()
        assert torch.count_nonzero(trainer.last_rgaa_role_normalized_advantages[:, 2]) == 0
    finally:
        trainer.close()


def test_main_critic_and_role_critics_are_parameter_independent():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        groups = [
            {id(p) for p in trainer.critic.parameters()},
            {id(p) for p in trainer.mav_role_critic.parameters()},
            {id(p) for p in trainer.uav_role_critic.parameters()},
        ]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    finally:
        trainer.close()


def test_rgaa_factor_history_uses_unchanged_preceding_factor_formula():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout(); metrics = trainer.update()
        observations = torch.as_tensor(trainer.buffer.observations.reshape(-1, 4, OBS_DIM))
        actions = torch.as_tensor(trainer.buffer.actions.reshape(-1, 4, 3))
        old = torch.as_tensor(trainer.buffer.log_probs.reshape(-1, 4))
        active = torch.as_tensor(trainer.buffer.active_masks.reshape(-1, 4))
        expected = torch.ones_like(old[:, 0])
        assert np.array_equal(trainer.last_rgaa_factor_history[0], expected.numpy())
        for history_index, agent in enumerate(metrics["agent_update_order"], start=1):
            with torch.no_grad():
                new, _ = trainer.actors.actors[agent].evaluate_actions(
                    observations[:, agent], actions[:, agent],
                )
            expected = preceding_factor_update(expected, old[:, agent], new, active[:, agent])
            np.testing.assert_allclose(
                trainer.last_rgaa_factor_history[history_index], expected.numpy(), rtol=1e-5, atol=1e-6,
            )
    finally:
        trainer.close()


def test_rgaa_checkpoint_resume_restores_role_critics_and_optimizers(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "rgaa.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "rgaa_happo"
    assert payload["base_algorithm"] == "happo"
    assert payload["method_variant"] == RGAA_METHOD
    assert payload["role_advantage_coef"] == 0.5
    assert payload["role_aux_reward_mode"] == ROLE_AUX_REWARD_MODE
    assert payload["role_critic_sharing"] == {"MAV": "independent", "UAV1-UAV3": "shared"}
    assert "role_critic_mav_optimizer_state" in payload
    assert "role_critic_uav_optimizer_state" in payload
    assert "rgaa_numpy_rng" in payload
    restored = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        for left, right in zip(source.mav_role_critic.parameters(), restored.mav_role_critic.parameters()):
            assert torch.equal(left, right)
        for left, right in zip(source.uav_role_critic.parameters(), restored.uav_role_critic.parameters()):
            assert torch.equal(left, right)
        assert restored.mav_role_critic_optimizer.state_dict()["state"]
        assert restored.uav_role_critic_optimizer.state_dict()["state"]
    finally:
        source.close(); restored.close()


def test_rgaa_resume_rejects_method_and_coefficient_mismatch(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "rgaa.pt"
    source.save_checkpoint(checkpoint); source.close()
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    changed = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.25))
    try:
        with pytest.raises(RuntimeError, match="resume method mismatch"):
            baseline.load_checkpoint(checkpoint)
        with pytest.raises(RuntimeError, match="role_advantage_coef"):
            changed.load_checkpoint(checkpoint)
    finally:
        baseline.close(); changed.close()


def test_rgaa_resume_rejects_legacy_process_only_auxiliary_semantics(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    payload = source.checkpoint_state()
    source.close()
    payload.pop("role_aux_reward_mode")
    checkpoint = tmp_path / "legacy_process_only_rgaa.pt"
    torch.save(payload, checkpoint)
    target = HAPPOTrainer(short_v39(), trainer_config())
    try:
        with pytest.raises(RuntimeError, match="role_aux_reward_mode"):
            target.load_checkpoint(checkpoint)
    finally:
        target.close()


def test_rgaa_config_is_a_strict_single_method_extension_of_entropy001_screen():
    with open("configs/happo_entropy001_logstd025_screen.yaml", encoding="utf-8") as stream:
        baseline = yaml.safe_load(stream)["training"]
    with open("configs/happo_rgaa_v39.yaml", encoding="utf-8") as stream:
        rgaa = yaml.safe_load(stream)["training"]
    assert _algorithm_name("vanilla", RGAA_METHOD, "mlp") == "rgaa_happo"
    assert rgaa.pop("role_advantage_coef") == 0.5
    assert rgaa.pop("role_aux_reward_mode") == ROLE_AUX_REWARD_MODE
    rgaa["method_variant"] = "baseline"
    assert rgaa == baseline


def test_vanilla_evaluator_loads_rgaa_actor_without_using_role_critics(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "rgaa.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run(
        [
            sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
            "--profile", "learnability", "--episodes", "1", "--device", "cpu",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "evaluation_rgaa_summary.json").read_text())
    assert summary["algorithm"] == "rgaa_happo"
    assert summary["method_variant"] == RGAA_METHOD


def test_rgaa_strict_environment_and_reward_contract():
    old = deepcopy(short_v39()); old["environment_version"] = "heterogeneous_mavuav_4v4_v3_8"
    wrong_reward = deepcopy(short_v39()); wrong_reward["role_reward"]["mode"] = "wrong"
    with pytest.raises(ValueError):
        HAPPOTrainer(old, trainer_config())
    with pytest.raises(ValueError):
        HAPPOTrainer(wrong_reward, trainer_config())


@pytest.mark.parametrize(
    "updates",
    [
        {"actor_variant": "tam", "critic_variant": "tam_attention"},
        {"actor_variant": "pcta_v2"},
        {"critic_variant": "relational"},
    ],
)
def test_rgaa_rejects_other_algorithm_architecture_combinations(updates):
    with pytest.raises(ValueError):
        HAPPOTrainer(short_v39(), trainer_config(**updates))


def test_rgaa_update_exposes_named_finite_diagnostics():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        _, metrics = trainer.train_update()
        expected = {"role_advantage_coef", "mav_role_critic_loss", "uav_role_critic_loss"}
        for aid in RED_IDS:
            expected.update({
                f"role_adv_mean_abs_{aid}", f"role_adv_std_{aid}",
                f"normalized_role_adv_mean_abs_{aid}", f"normalized_role_adv_std_{aid}",
                f"combined_adv_mean_abs_{aid}", f"mean_role_reward_{aid}",
                f"mean_aux_reward_{aid}", f"own_loss_count_{aid}",
                f"own_boundary_loss_count_{aid}", f"own_blue_attack_loss_count_{aid}",
            })
        assert expected <= metrics.keys()
        assert all(np.isfinite(metrics[key]) for key in expected)
    finally:
        trainer.close()


def test_rgaa_preserves_seed_matched_vanilla_actor_and_team_critic_initialization():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    rgaa = HAPPOTrainer(short_v39(), trainer_config())
    try:
        for left, right in zip(baseline.actors.parameters(), rgaa.actors.parameters()):
            assert torch.equal(left, right)
        for left, right in zip(baseline.critic.parameters(), rgaa.critic.parameters()):
            assert torch.equal(left, right)
    finally:
        baseline.close(); rgaa.close()


def test_rgaa_role_initialization_preserves_global_torch_rng_state():
    vanilla = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    vanilla_state = torch.get_rng_state().clone()
    vanilla.close()
    rgaa = HAPPOTrainer(short_v39(), trainer_config(role_advantage_coef=0.0))
    try:
        assert torch.equal(torch.get_rng_state(), vanilla_state)
    finally:
        rgaa.close()


def _main_parameter_snapshots(method: str, updates: int):
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(method_variant=method, role_advantage_coef=0.0),
    )
    snapshots = []
    try:
        for _ in range(updates):
            trainer.train_update()
            snapshots.append((
                [parameter.detach().clone() for parameter in trainer.actors.parameters()],
                [parameter.detach().clone() for parameter in trainer.critic.parameters()],
            ))
    finally:
        trainer.close()
    return snapshots


def test_rgaa_coef_zero_matches_vanilla_after_one_complete_update():
    vanilla = _main_parameter_snapshots("baseline", 1)[0]
    rgaa = _main_parameter_snapshots(RGAA_METHOD, 1)[0]
    assert all(torch.equal(left, right) for left, right in zip(vanilla[0], rgaa[0]))
    assert all(torch.equal(left, right) for left, right in zip(vanilla[1], rgaa[1]))


def test_rgaa_coef_zero_matches_vanilla_after_two_complete_updates():
    vanilla = _main_parameter_snapshots("baseline", 2)
    rgaa = _main_parameter_snapshots(RGAA_METHOD, 2)
    for vanilla_step, rgaa_step in zip(vanilla, rgaa):
        assert all(torch.equal(left, right) for left, right in zip(vanilla_step[0], rgaa_step[0]))
        assert all(torch.equal(left, right) for left, right in zip(vanilla_step[1], rgaa_step[1]))


def test_role_critic_training_advances_only_rgaa_numpy_rng():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        observations = torch.zeros(8, 4, OBS_DIM)
        targets = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        active = torch.ones(8, 4)
        main_before = deepcopy(trainer.rng.bit_generator.state)
        auxiliary_before = deepcopy(trainer.rgaa_rng.bit_generator.state)
        trainer._train_role_critics(observations, targets, active)
        assert trainer.rng.bit_generator.state == main_before
        assert trainer.rgaa_rng.bit_generator.state != auxiliary_before
    finally:
        trainer.close()


def test_checkpoint_resume_restores_exact_auxiliary_rng_continuation(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    observations = torch.zeros(8, 4, OBS_DIM)
    targets = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    active = torch.ones(8, 4)
    source._train_role_critics(observations, targets, active)
    checkpoint = tmp_path / "rgaa_rng.pt"
    source.save_checkpoint(checkpoint)
    restored = HAPPOTrainer(short_v39(), trainer_config())
    try:
        restored.load_checkpoint(checkpoint)
        assert restored.rgaa_rng.bit_generator.state == source.rgaa_rng.bit_generator.state
        source._train_role_critics(observations, targets, active)
        restored._train_role_critics(observations, targets, active)
        assert restored.rgaa_rng.bit_generator.state == source.rgaa_rng.bit_generator.state
        for left, right in zip(source.mav_role_critic.parameters(), restored.mav_role_critic.parameters()):
            assert torch.equal(left, right)
        for left, right in zip(source.uav_role_critic.parameters(), restored.uav_role_critic.parameters()):
            assert torch.equal(left, right)
    finally:
        source.close(); restored.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rgaa_tiny_cuda_update_is_finite():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(device="cuda", rollout_steps=1, minibatch_size=4),
    )
    try:
        _, metrics = trainer.train_update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
        assert next(trainer.mav_role_critic.parameters()).is_cuda
        assert next(trainer.uav_role_critic.parameters()).is_cuda
    finally:
        trainer.close()
