from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
import subprocess
import sys

from algorithm.happo.dbm_rgaa import DBM_RGAA_METHOD
from algorithm.happo.tacm_rgaa import (
    TACM_RGAA_METHOD, context_distillation_loss, tactical_teacher,
    tacm_context_coefficient, temporal_router_loss,
)
from algorithm.happo.trainer import HAPPOTrainer
import algorithm.happo.trainer as trainer_module
from algorithm.train_happo import _algorithm_name
from algorithm.evaluate_happo import validate_checkpoint_contract
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.models import AircraftState
from env.vector_env import MAVUAVVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def short_v39(max_steps: int = 4):
    config = deepcopy(load_environment_config(ROOT / "configs" / "env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def tacm_config(**updates):
    with (ROOT / "configs" / "happo_tacm_rgaa_v39.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["training"]
    config.update({"device": "cpu", "num_envs": 1, "rollout_steps": 4,
                   "ppo_epochs": 1, "minibatch_size": 4, "seed": 17})
    config.update(updates)
    return config


def _set_state(env, aid, xyz, psi=0.0, alive=True):
    env.entities[aid].state = AircraftState(*xyz, 220.0, 0.0, psi, alive)


def test_dbm_and_tacm_same_seed_actor_state_is_exactly_equal():
    common = tacm_config(randomization_curriculum_enabled=False)
    dbm = HAPPOTrainer(short_v39(), {**common, "method_variant": DBM_RGAA_METHOD})
    tacm = HAPPOTrainer(short_v39(), common)
    try:
        assert dbm.actors.state_dict().keys() == tacm.actors.state_dict().keys()
        for key, value in dbm.actors.state_dict().items():
            assert torch.equal(value, tacm.actors.state_dict()[key]), key
    finally:
        dbm.close(); tacm.close()


def test_tacm_construction_does_not_advance_main_torch_rng_relative_to_dbm():
    common = tacm_config(randomization_curriculum_enabled=False)
    dbm = HAPPOTrainer(short_v39(), {**common, "method_variant": DBM_RGAA_METHOD})
    dbm_state = torch.get_rng_state().clone()
    dbm.close()
    tacm = HAPPOTrainer(short_v39(), common)
    tacm_state = torch.get_rng_state().clone()
    tacm.close()
    assert torch.equal(dbm_state, tacm_state)


def test_teacher_is_rng_free_and_no_visible_is_uninformative():
    env = HeterogeneousMAVUAVAirCombatEnv(short_v39(), seed=1, randomize=False)
    env.reset()
    for red in RED_IDS:
        _set_state(env, red, (-19000.0, -19000.0, 5000.0))
    for blue in BLUE_IDS:
        _set_state(env, blue, (19000.0, 19000.0, 5000.0))
    np_state = deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    result = tactical_teacher(env.global_state(), env.config)
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(result.probabilities, np.full((1, 3, 2), 0.5, np.float32))
    assert np.count_nonzero(result.confidence) == 0
    assert np.all(result.engagement_target == -1) and np.all(result.threat_target == -1)


def test_teacher_engagement_and_team_aware_cover_allocation():
    env = HeterogeneousMAVUAVAirCombatEnv(short_v39(), seed=2, randomize=False)
    env.reset()
    # UAV1 has an exact engagement gate while Blue aircraft do not threaten the distant MAV.
    _set_state(env, "MAV", (-10000.0, 0.0, 5000.0))
    _set_state(env, "UAV1", (0.0, 0.0, 5000.0))
    _set_state(env, "UAV2", (0.0, 5000.0, 5000.0))
    _set_state(env, "UAV3", (0.0, -5000.0, 5000.0))
    _set_state(env, "Blue1", (2000.0, 0.0, 5000.0))
    for blue in BLUE_IDS[1:]:
        _set_state(env, blue, (19000.0, 19000.0, 5000.0), alive=False)
    engagement = tactical_teacher(env.global_state(), env.config)
    assert engagement.probabilities[0, 0, 0] > engagement.probabilities[0, 0, 1]

    # Blue1 now has an exact attack gate on MAV; UAV1 is the best interceptor.
    _set_state(env, "MAV", (0.0, 0.0, 5000.0))
    _set_state(env, "Blue1", (-2000.0, 0.0, 5000.0), psi=0.0)
    _set_state(env, "UAV1", (-4000.0, 0.0, 5000.0), psi=0.0)
    _set_state(env, "UAV2", (6000.0, 7000.0, 5000.0), psi=np.pi)
    _set_state(env, "UAV3", (6000.0, -7000.0, 5000.0), psi=np.pi)
    cover = tactical_teacher(env.global_state(), env.config)
    assert cover.cover_assignment[0, 0] > cover.cover_assignment[0, 1]
    assert cover.cover_assignment[0, 0] > cover.cover_assignment[0, 2]
    assert np.isclose(cover.cover_assignment[0].sum(), 1.0)
    assert cover.probabilities[0, 0, 1] > engagement.probabilities[0, 0, 1]
    assert not np.all(cover.probabilities[0, :, 1] > cover.probabilities[0, :, 0])


def test_context_loss_direct_gradient_is_router_only():
    trainer = HAPPOTrainer(short_v39(), tacm_config(randomization_curriculum_enabled=False))
    try:
        actor = trainer.actors.actors[1]
        obs = torch.randn(8, 100)
        teacher = torch.tensor([[0.9, 0.1]]).repeat(8, 1)
        confidence = torch.ones(8)
        loss, _ = context_distillation_loss(actor, obs, teacher, confidence)
        loss.backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in actor.network.router.parameters())
        assert all(p.grad is None for p in actor.network.encoder.parameters())
        assert all(p.grad is None for p in actor.network.base_head.parameters())
        assert all(p.grad is None for p in actor.network.experts.parameters())
        assert actor.log_std.grad is None
    finally:
        trainer.close()


def test_temporal_event_alignment_masks_the_preceding_transition():
    trainer = HAPPOTrainer(short_v39(), tacm_config(randomization_curriculum_enabled=False))
    try:
        actor = trainer.actors.actors[1]
        obs = torch.randn(3, 1, 100)
        active = torch.ones(3, 1, dtype=torch.bool)
        boundary = torch.zeros(3, 1, dtype=torch.bool)
        targets = torch.zeros(3, 1, dtype=torch.long)
        teacher = torch.tensor([[[0.9, 0.1]], [[0.9, 0.1]], [[0.9, 0.1]]])
        confidence = torch.ones(3, 1)
        _, count = temporal_router_loss(actor, obs, active, boundary, boundary,
                                        torch.tensor([[True], [False], [False]]), targets,
                                        targets, teacher, confidence)
        assert count == 1  # event[0] masks state[0] -> state[1], not state[1] -> state[2]
        switched = targets.clone(); switched[2] = 1
        _, count = temporal_router_loss(actor, obs, active, boundary, boundary,
                                        torch.zeros_like(boundary), switched, targets,
                                        teacher, confidence)
        assert count == 1
    finally:
        trainer.close()


def test_all_temporal_boundary_and_tactical_switch_masks():
    trainer = HAPPOTrainer(short_v39(), tacm_config(randomization_curriculum_enabled=False))
    try:
        actor = trainer.actors.actors[1]
        obs = torch.randn(4, 1, 100)
        active = torch.ones(4, 1, dtype=torch.bool)
        clear = torch.zeros(4, 1, dtype=torch.bool)
        target = torch.zeros(4, 1, dtype=torch.long)
        teacher = torch.tensor([[[0.9, 0.1]]] * 4)
        confidence = torch.ones(4, 1)

        def count(*, active_mask=active, terminated=clear, truncated=clear,
                  event=clear, engagement=target, threat=target, modes=teacher):
            return temporal_router_loss(actor, obs, active_mask, terminated, truncated,
                                        event, engagement, threat, modes, confidence)[1]

        assert count() == 3
        changed = clear.clone(); changed[0] = True
        assert count(terminated=changed) == 2
        assert count(truncated=changed) == 2
        assert count(event=changed) == 2
        inactive = active.clone(); inactive[1] = False
        assert count(active_mask=inactive) == 1
        switched = target.clone(); switched[1] = 1
        assert count(engagement=switched) == 1
        assert count(threat=switched) == 1
        switched_mode = teacher.clone(); switched_mode[1] = torch.tensor([0.1, 0.9])
        assert count(modes=switched_mode) == 1
    finally:
        trainer.close()


def test_temporal_router_step_precedes_factor_recomputation(monkeypatch):
    pending = []
    original_factor = trainer_module.preceding_factor_update

    def zero_context(actor, observations, teacher, confidence):
        probabilities = torch.softmax(actor.network.router(actor.network.encoder(observations).detach()), -1)
        return actor.network.router.weight.sum() * 0.0, torch.zeros_like(confidence)

    def marked_temporal(actor, *args, **kwargs):
        pending.append((actor, actor.network.router.weight.detach().clone()))
        return actor.network.router.weight.sum(), 1

    def checked_factor(factor, old_log_prob, new_log_prob, active):
        if pending:
            actor, before = pending.pop(0)
            assert not torch.equal(actor.network.router.weight.detach(), before)
        return original_factor(factor, old_log_prob, new_log_prob, active)

    monkeypatch.setattr(trainer_module, "context_distillation_loss", zero_context)
    monkeypatch.setattr(trainer_module, "temporal_router_loss", marked_temporal)
    monkeypatch.setattr(trainer_module, "preceding_factor_update", checked_factor)
    trainer = HAPPOTrainer(short_v39(), tacm_config(randomization_curriculum_enabled=False))
    try:
        trainer.train_update()
        assert pending == []
    finally:
        trainer.close()


@pytest.mark.parametrize("step,alpha,team,slot,altitude,speed,heading", [
    (0, 0.0, 0.0, 200.0, 100.0, 10.0, 3.0),
    (200000, 0.5, 750.0, 250.0, 250.0, 15.0, 6.5),
    (400000, 1.0, 1500.0, 300.0, 400.0, 20.0, 10.0),
    (600000, 1.0, 1500.0, 300.0, 400.0, 20.0, 10.0),
])
def test_curriculum_endpoints(step, alpha, team, slot, altitude, speed, heading):
    trainer = HAPPOTrainer(short_v39(), tacm_config())
    try:
        trainer.env_steps = step
        state = trainer.curriculum_state()
        assert state["alpha"] == alpha
        assert state["team_xy_jitter"] == team
        assert state["slot_xy_jitter"] == slot
        assert state["altitude_jitter"] == altitude
        assert state["speed_jitter"] == speed
        assert state["heading_jitter_deg"] == heading
    finally:
        trainer.close()


@pytest.mark.parametrize("start_step,alpha,team,slot,altitude,speed,heading", [
    (0, 0.0, 0.0, 200.0, 100.0, 10.0, 3.0),
    (200000, 0.5, 750.0, 250.0, 250.0, 15.0, 6.5),
    (400000, 1.0, 1500.0, 300.0, 400.0, 20.0, 10.0),
    (500000, 1.0, 1500.0, 300.0, 400.0, 20.0, 10.0),
])
def test_training_metrics_report_rollout_start_curriculum_contract(
    start_step, alpha, team, slot, altitude, speed, heading,
):
    trainer = HAPPOTrainer(short_v39(), tacm_config())
    try:
        trainer.env_steps = start_step
        _, metrics = trainer.train_update()
        assert trainer.env_steps > start_step
        assert metrics["curriculum_alpha"] == alpha
        assert metrics["curriculum_team_xy_jitter"] == team
        assert metrics["curriculum_slot_xy_jitter"] == slot
        assert metrics["curriculum_altitude_jitter"] == altitude
        assert metrics["curriculum_speed_jitter"] == speed
        assert metrics["curriculum_heading_jitter_deg"] == heading
        assert trainer.vector_env.randomization_override == trainer.applied_curriculum_state["override"]
    finally:
        trainer.close()


def test_context_schedule_metadata_and_checkpoint_exact_resume(tmp_path):
    assert tacm_context_coefficient(0, .05, .01, 500000) == .05
    assert tacm_context_coefficient(500000, .05, .01, 500000) == .01
    config = tacm_config()
    trainer = HAPPOTrainer(short_v39(), config)
    path = tmp_path / "tacm.pt"
    try:
        trainer.train_update()
        trainer.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["algorithm"] == "tacm_rgaa_happo"
        assert payload["tacm_rgaa_version"] == 1
        saved_curriculum = payload["rollout_state"]["randomization_curriculum_state"]
        assert saved_curriculum["sampled_steps"] == trainer.env_steps
        assert saved_curriculum["applied_alpha"] == trainer.applied_curriculum_state["alpha"]
        assert saved_curriculum["applied_override"] == trainer.vector_env.randomization_override
        assert saved_curriculum["applied_values"] == {
            key: trainer.applied_curriculum_state[key] for key in (
                "team_xy_jitter", "slot_xy_jitter", "altitude_jitter",
                "speed_jitter", "heading_jitter_deg",
            )
        }
        saved_steps = trainer.env_steps
    finally:
        trainer.close()
    restored = HAPPOTrainer(short_v39(), config)
    try:
        assert restored.load_checkpoint(path) == saved_steps
        assert restored.vector_env.randomization_override == payload["rollout_state"]["randomization_curriculum_state"]["applied_override"]
        assert restored.applied_curriculum_state == trainer.applied_curriculum_state
    finally:
        restored.close()
    mismatch = HAPPOTrainer(short_v39(), tacm_config(tacm_tau_teacher=0.5))
    try:
        with pytest.raises(RuntimeError, match="TACM|resume config mismatch"):
            mismatch.load_checkpoint(path)
    finally:
        mismatch.close()


def test_tacm_checkpoint_resume_is_exact_continuation(tmp_path):
    config = tacm_config()
    checkpoint = tmp_path / "resume.pt"
    reference = HAPPOTrainer(short_v39(max_steps=8), config)
    try:
        reference.train_update()
        reference.save_checkpoint(checkpoint)
        reference.train_update()
        expected = {
            "actors": {key: value.detach().cpu().clone() for key, value in reference.actors.state_dict().items()},
            "critic": {key: value.detach().cpu().clone() for key, value in reference.critic.state_dict().items()},
            "mav_role": {key: value.detach().cpu().clone() for key, value in reference.mav_role_critic.state_dict().items()},
            "uav_role": {key: value.detach().cpu().clone() for key, value in reference.uav_role_critic.state_dict().items()},
            "observations": reference.observations.copy(),
            "states": reference.global_states.copy(),
            "masks": reference.active_masks.copy(),
            "steps": reference.env_steps,
            "rng": deepcopy(reference.rng.bit_generator.state),
            "rgaa_rng": deepcopy(reference.rgaa_rng.bit_generator.state),
        }
    finally:
        reference.close()
    resumed = HAPPOTrainer(short_v39(max_steps=8), config)
    try:
        resumed.load_checkpoint(checkpoint)
        resumed.train_update()
        for module_name, module in (
            ("actors", resumed.actors), ("critic", resumed.critic),
            ("mav_role", resumed.mav_role_critic), ("uav_role", resumed.uav_role_critic),
        ):
            for key, value in module.state_dict().items():
                assert torch.equal(value.detach().cpu(), expected[module_name][key]), (module_name, key)
        assert np.array_equal(resumed.observations, expected["observations"])
        assert np.array_equal(resumed.global_states, expected["states"])
        assert np.array_equal(resumed.active_masks, expected["masks"])
        assert resumed.env_steps == expected["steps"]
        assert resumed.rng.bit_generator.state == expected["rng"]
        assert resumed.rgaa_rng.bit_generator.state == expected["rgaa_rng"]
    finally:
        resumed.close()


@pytest.mark.parametrize("parallel", [False, True])
def test_vector_reset_preserves_current_randomization_override(parallel):
    config = short_v39(max_steps=1)
    override = {
        "team_xy_jitter": 321.0, "slot_xy_jitter": 222.0,
        "altitude_jitter": 123.0, "speed_jitter": 11.0,
        "heading_jitter_deg": 4.0,
    }
    with MAVUAVVectorEnv(1, config, seed=41, profile="main", parallel=parallel) as vector:
        vector.set_randomization_override(override)
        vector.reset()
        vector.step(np.zeros((1, 4, 3), dtype=np.float32))
        state = vector.get_env_states()[0]
        assert state["randomization_override"] == override
        assert vector.randomization_override == override


def test_curriculum_disabled_preserves_canonical_reset_contract():
    config = tacm_config(randomization_curriculum_enabled=False)
    trainer = HAPPOTrainer(short_v39(), config)
    try:
        assert trainer.curriculum_state()["override"] is None
        assert trainer.vector_env.randomization_override is None
        reference = HeterogeneousMAVUAVAirCombatEnv(short_v39(), seed=17, profile="main")
        reference.reset(seed=17)
        assert np.array_equal(trainer.global_states[0], reference.global_state())
        _, metrics = trainer.train_update()
        assert metrics["curriculum_alpha"] == 1.0
        assert metrics["curriculum_team_xy_jitter"] == 1500.0
        assert metrics["curriculum_slot_xy_jitter"] == 300.0
        assert metrics["curriculum_altitude_jitter"] == 400.0
        assert metrics["curriculum_speed_jitter"] == 20.0
        assert metrics["curriculum_heading_jitter_deg"] == 10.0
        assert trainer.vector_env.randomization_override is None
    finally:
        trainer.close()


def test_tacm_contract_labels_and_short_update_are_finite():
    assert _algorithm_name("vanilla", TACM_RGAA_METHOD, "mlp") == "tacm_rgaa_happo"
    trainer = HAPPOTrainer(short_v39(), tacm_config())
    try:
        _, metrics = trainer.train_update()
        required = ("tacm_context_loss", "tacm_temporal_valid_pairs",
                    "tacm_teacher_engagement_prob_UAV1", "curriculum_alpha")
        assert all(np.isfinite(metrics[key]) for key in required)
        payload = trainer.checkpoint_state()
        validate_checkpoint_contract(payload, trainer.environment_config)
        assert payload["tacm_mode_labels"] == ["engagement", "cover_support"]
    finally:
        trainer.close()


def test_standalone_evaluator_loads_tacm_without_teacher(tmp_path):
    trainer = HAPPOTrainer(short_v39(), tacm_config(randomization_curriculum_enabled=False))
    checkpoint = tmp_path / "checkpoint_final.pt"
    try:
        trainer.save(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run([
        sys.executable, str(ROOT / "algorithm" / "evaluate_happo.py"), str(checkpoint),
        "--profile", "main", "--episodes", "1", "--device", "cpu",
        "--action-mode", "stochastic", "--action-seed", "4000",
        "--env-seed-start", "3000",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    summary = checkpoint.parent / "evaluation_final_stochastic_summary.json"
    data = __import__("json").loads(summary.read_text(encoding="utf-8"))
    assert data["algorithm"] == "tacm_rgaa_happo"
    assert data["evaluation_environment_seed_start"] == 3000
    assert data["effective_action_seed"] == 4000
