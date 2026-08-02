from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import shutil

import numpy as np
import pytest
import torch

from uav_combat.config import aircraft_spec, load_config
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv
from uav_combat.integrator import RK4Integrator
from uav_combat.mappo.trainer_3v3 import CHECKPOINT_VERSION_3V3, FixedBlue3v3MAPPOTrainer, sha256_file
from uav_combat.mappo.vector_env_3v3 import make_combat_vector_env_3v3
from uav_combat.models import AircraftState
from scripts.audit_mappo_v7_10m_failure import (
    _action_stats,
    _load_actor_audit_only,
    _trace_checkpoint,
)


ROOT = Path(__file__).parents[1]
ENV_V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _tiny_mappo_config(tmp_path: Path, *, num_envs: int = 1) -> dict:
    return {
        "experiment": {"seed": 7, "device": "cpu", "output_dir": str(tmp_path)},
        "network": {"hidden_dim": 32, "log_std_init": -0.5},
        "training": {
            "training_mode": "fixed_rule_blue_3v3",
            "total_env_steps": 64,
            "num_envs": num_envs,
            "num_env_workers": 1,
            "rollout_steps": 4,
            "ppo_epochs": 1,
            "minibatch_size": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "learning_rate": 3e-4,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "evaluation_interval_env_steps": 32,
            "quick_evaluation_episodes": 2,
        },
        "evaluation": {"episodes": 2, "deterministic": True},
    }


def _rng_state_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        av, bv = a[key], b[key]
        if isinstance(av, dict):
            if not _rng_state_equal(av, bv):
                return False
        elif isinstance(av, np.ndarray):
            if not np.array_equal(av, bv):
                return False
        else:
            if av != bv:
                return False
    return True


def test_mappo_3v3_checkpoint_roundtrip_restores_algorithm_rng_after_env_reset(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    restored = None
    try:
        trainer.collect_rollout()
        trainer.update()
        obs = torch.as_tensor(trainer.current_observations[:, :3, :].reshape(-1, 68))
        with torch.no_grad():
            expected_action = trainer.red_actor.deterministic_action(obs).detach().clone()
            expected_value = trainer.team_critic(
                torch.as_tensor(trainer.current_global_states)
            ).detach().clone()
        expected_np_state = deepcopy(trainer.rng.bit_generator.state)
        expected_torch_state = torch.get_rng_state().clone()

        ckpt = tmp_path / "mappo_3v3.pt"
        trainer.save_checkpoint(ckpt)

        restored = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
        restored.load_checkpoint(ckpt)
        with torch.no_grad():
            actual_action = restored.red_actor.deterministic_action(obs)
            actual_value = restored.team_critic(torch.as_tensor(trainer.current_global_states))

        assert _rng_state_equal(restored.rng.bit_generator.state, expected_np_state)
        assert torch.equal(torch.get_rng_state(), expected_torch_state)
        assert torch.allclose(actual_action, expected_action, atol=0.0, rtol=0.0)
        assert torch.allclose(actual_value, expected_value, atol=0.0, rtol=0.0)

        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = deepcopy(expected_np_state)
        assert np.array_equal(
            restored.rng.integers(0, 2**31 - 1, size=16),
            reference_rng.integers(0, 2**31 - 1, size=16),
        )
    finally:
        trainer.close()
        if restored is not None:
            restored.close()


def test_mappo_3v3_checkpoint_signature_mismatch_is_rejected(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path, num_envs=1))
    try:
        ckpt = tmp_path / "mappo_3v3.pt"
        trainer.save_checkpoint(ckpt)
    finally:
        trainer.close()

    bad_config = _tiny_mappo_config(tmp_path, num_envs=2)
    restored = FixedBlue3v3MAPPOTrainer(ENV_V7, bad_config)
    try:
        with pytest.raises(RuntimeError, match="checkpoint signature mismatch"):
            restored.load_checkpoint(ckpt)
    finally:
        restored.close()


def test_mappo_3v3_checkpoint_rejects_family_version_missing_signature_and_old_version(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    try:
        ckpt = tmp_path / "mappo_3v3.pt"
        trainer.save_checkpoint(ckpt)
    finally:
        trainer.close()

    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    cases = [
        ("family.pt", {**base, "checkpoint_family": "wrong"}, "Expected homogeneous_3v3_fixed_blue"),
        ("version.pt", {**base, "checkpoint_version": CHECKPOINT_VERSION_3V3 + 1}, "checkpoint_version"),
        ("missing_sig.pt", {k: v for k, v in base.items() if k != "training_signature"}, "missing training_signature"),
        ("old_v1.pt", {**base, "checkpoint_version": 1}, "checkpoint_version"),
    ]
    for filename, payload, match in cases:
        path = tmp_path / filename
        torch.save(payload, path)
        restored = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
        try:
            with pytest.raises(RuntimeError, match=match):
                restored.load_checkpoint(path)
        finally:
            restored.close()


def test_mappo_3v3_training_signature_includes_env_yaml_content_hash(tmp_path):
    cfg_copy = tmp_path / "env_same.yaml"
    shutil.copyfile(ENV_V7, cfg_copy)
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    same_path = None
    changed = None
    try:
        ckpt = tmp_path / "mappo_3v3.pt"
        trainer.save_checkpoint(ckpt)
        sig = trainer.training_signature()
        assert sig["env_config_sha256"] == hashlib.sha256(ENV_V7.read_bytes()).hexdigest()
    finally:
        trainer.close()

    same_path = FixedBlue3v3MAPPOTrainer(cfg_copy, _tiny_mappo_config(tmp_path))
    try:
        same_path.load_checkpoint(ckpt)
    finally:
        same_path.close()

    changed_copy = tmp_path / "env_changed.yaml"
    changed_copy.write_text(ENV_V7.read_text(encoding="utf-8") + "\n# audit hash change\n", encoding="utf-8")
    changed = FixedBlue3v3MAPPOTrainer(changed_copy, _tiny_mappo_config(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="env_config_sha256"):
            changed.load_checkpoint(ckpt)
    finally:
        changed.close()


def test_mappo_3v3_legacy_checkpoint_actor_audit_load_does_not_require_strict_resume(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    try:
        ckpt_path = tmp_path / "legacy_actor.pt"
        trainer.save_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["checkpoint_version"] = 1
        ckpt["training_signature"].pop("env_config_sha256", None)
        torch.save(ckpt, ckpt_path)
    finally:
        trainer.close()

    strict = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="checkpoint_version"):
            strict.load_checkpoint(ckpt_path)
    finally:
        strict.close()
    actor, loaded = _load_actor_audit_only(ckpt_path)
    obs = np.zeros((2, 68), dtype=np.float32)
    stats = _action_stats(actor, obs, mc_samples=2)
    assert loaded["checkpoint_version"] == 1
    assert stats["finite"] is True


def test_v7_observation_self_block_contains_signed_altitude_boundary_information():
    env = Homogeneous3v3AirCombatEnv(ENV_V7)
    try:
        env.reset(123)
        red = env._aircraft_by_id("red_0")
        bf = env.config["battlefield"]
        assert red.state.altitude == pytest.approx(-red.state.z)

        red.state.z = -bf["altitude_min"]
        low_obs = env._agent_observation(red)
        red.state.z = -bf["altitude_max"]
        high_obs = env._agent_observation(red)
        red.state.z = -0.5 * (bf["altitude_min"] + bf["altitude_max"])
        mid_obs = env._agent_observation(red)

        assert low_obs[2] == pytest.approx(-1.0)
        assert high_obs[2] == pytest.approx(1.0)
        assert mid_obs[2] == pytest.approx(0.0)
        assert low_obs[7] == high_obs[7] == mid_obs[7] == pytest.approx(1.0)

        red.state.alive = False
        assert np.all(env._agent_observation(red) == 0.0)
    finally:
        pass


def test_v7_control_rates_and_speed_acceleration_match_configured_limits():
    cfg = load_config(ENV_V7)
    spec = aircraft_spec(cfg)
    controller = TargetStateController(**cfg["action"], gravity=cfg["simulation"]["gravity"])
    dynamics = PointMassDynamics(cfg["simulation"]["gravity"])
    integrator = RK4Integrator(cfg["simulation"]["dt"])
    base = AircraftState(0.0, 0.0, -3000.0, 150.0, 0.0, 0.0)

    state = base.copy()
    psi0 = state.psi
    first_yaw_diag = None
    first_actual_yaw_rate = None
    for _ in range(10):
        target, control = controller.control_from_action(state, np.array([1.0, 0.0, 0.0]), spec)
        if first_yaw_diag is None:
            first_yaw_diag = controller.diagnostics(state, target, control, spec, np.array([1.0, 0.0, 0.0]))
            first_actual_yaw_rate = float(dynamics.derivatives(state, control)[5])
        state = integrator.step(state, control, dynamics, spec)
    avg_yaw_rate = (state.psi - psi0) / (10 * cfg["simulation"]["dt"])
    assert first_yaw_diag["requested_yaw_rate"] == pytest.approx(spec.yaw_rate_max)
    assert 0.0 < first_actual_yaw_rate <= spec.yaw_rate_max
    assert 0.0 < avg_yaw_rate <= spec.yaw_rate_max

    state = base.copy()
    th0 = state.theta
    first_pitch_diag = None
    first_actual_pitch_rate = None
    for _ in range(10):
        target, control = controller.control_from_action(state, np.array([0.0, 1.0, 0.0]), spec)
        if first_pitch_diag is None:
            first_pitch_diag = controller.diagnostics(state, target, control, spec, np.array([0.0, 1.0, 0.0]))
            first_actual_pitch_rate = float(dynamics.derivatives(state, control)[4])
        state = integrator.step(state, control, dynamics, spec)
    assert first_pitch_diag["requested_pitch_rate"] == pytest.approx(spec.pitch_rate_max)
    avg_pitch_rate = (state.theta - th0) / (10 * cfg["simulation"]["dt"])
    assert 0.0 < first_actual_pitch_rate <= spec.pitch_rate_max
    assert 0.0 < avg_pitch_rate <= spec.pitch_rate_max

    state = base.copy()
    v0 = state.v
    for _ in range(5):
        _, control = controller.control_from_action(state, np.array([0.0, 0.0, 1.0]), spec)
        state = integrator.step(state, control, dynamics, spec)
    assert (state.v - v0) / (5 * cfg["simulation"]["dt"]) == pytest.approx(spec.acceleration_max, rel=0.05)
    assert state.v <= spec.v_max


def test_rk4_dt_0p1_close_to_smaller_step_reference_and_high_pitch_finite():
    cfg = load_config(ENV_V7)
    spec = aircraft_spec(cfg)
    dynamics = PointMassDynamics(cfg["simulation"]["gravity"])
    state = AircraftState(0.0, 0.0, -3000.0, 150.0, spec.theta_max * 0.95, 0.2)
    control = TargetStateController(**cfg["action"], gravity=cfg["simulation"]["gravity"]).compute_control(
        state,
        TargetStateController(**cfg["action"], gravity=cfg["simulation"]["gravity"]).action_to_target(
            state, np.array([1.0, 1.0, 1.0]), spec
        ),
        spec,
    )
    coarse = RK4Integrator(0.1).step(state.copy(), control, dynamics, spec)
    ref = state.copy()
    small = RK4Integrator(0.01)
    for _ in range(10):
        ref = small.step(ref, control, dynamics, spec)
    assert np.linalg.norm(coarse.as_array() - ref.as_array()) < 1.0
    assert np.isfinite(coarse.as_array()).all()


def test_v7_boundary_limits_and_one_step_crossing_classification():
    env = Homogeneous3v3AirCombatEnv(ENV_V7)
    try:
        env.reset(321)
        bf = env.config["battlefield"]
        red = env._aircraft_by_id("red_0")
        red.state.z = -bf["altitude_max"]
        assert bf["altitude_min"] <= red.state.altitude <= bf["altitude_max"]
        red.state.z = -(bf["altitude_max"] + 1.0)
        actions = {a.aircraft_id: np.zeros(3, np.float32) for a in env.aircraft if a.state.alive}
        _, _, _, _, info = env.step(actions)
        assert info["death_causes"]["red_0"] == 1
        assert info["boundary_altitude_deaths"]["red"] == 1
    finally:
        pass


def test_v7_local_and_worker_same_seed_zero_action_same_result():
    specs = [{"seed": 1234}, {"seed": 1235}]
    actions = np.zeros((2, 3, 3), dtype=np.float32)
    local = make_combat_vector_env_3v3(ENV_V7, num_envs=2, num_env_workers=1)
    worker = make_combat_vector_env_3v3(ENV_V7, num_envs=2, num_env_workers=2)
    try:
        lo = local.reset(specs)
        wo = worker.reset(specs)
        assert np.allclose(lo[0], wo[0])
        lr = local.step(actions)
        wr = worker.step(actions)
        assert np.allclose(lr.observations, wr.observations)
        assert np.allclose(lr.team_rewards, wr.team_rewards)
        assert np.array_equal(lr.step_death_causes, wr.step_death_causes)
    finally:
        local.close()
        worker.close()


def test_v7_pre_attack_audit_snapshot_aligns_with_reward_components():
    env = Homogeneous3v3AirCombatEnv(ENV_V7)
    try:
        env.audit_trace_enabled = True
        env.reset(2468)
        actions = {a.aircraft_id: np.zeros(3, np.float32) for a in env.aircraft if a.state.alive}
        _, _, _, _, info = env.step(actions)
        audit = info["audit"]["paper_segmented_v4_pre_attack"]
        red_r3 = sum(float(audit[aid]["r3"]) for aid in ("red_0", "red_1", "red_2")) / 3.0
        red_r41 = sum(float(audit[aid]["r41"]) for aid in ("red_0", "red_1", "red_2")) / 3.0
        red_r42 = sum(float(audit[aid]["r42"]) for aid in ("red_0", "red_1", "red_2")) / 3.0
        rc = info["reward_components"]
        assert red_r3 == pytest.approx(rc["red_approach_reward"])
        assert red_r41 == pytest.approx(rc["red_attack_advantage_reward"])
        assert red_r42 == pytest.approx(rc["red_threat_penalty"])
    finally:
        pass


def test_mappo_audit_trace_rows_are_step_aligned_and_predeath_history_not_copied(tmp_path):
    run_ckpt = tmp_path / "ckpt.pt"
    trainer = FixedBlue3v3MAPPOTrainer(ENV_V7, _tiny_mappo_config(tmp_path))
    try:
        trainer.save_checkpoint(run_ckpt)
    finally:
        trainer.close()
    actor, _ = _load_actor_audit_only(run_ckpt)
    episodes, trace_rows, predeath_rows = _trace_checkpoint("tiny", actor, ENV_V7, [240001])
    assert episodes
    assert trace_rows
    for row in trace_rows[:10]:
        assert row["finite"] is True
        if row["reward_r3"] is not None:
            assert float(row["red_r3"]) == pytest.approx(float(row["red_r3"]))
    if predeath_rows:
        triples = {(r["red_r3"], r["red_r41"], r["red_r42"], r["red_team_total_reward"]) for r in predeath_rows}
        assert len(triples) > 1 or len(predeath_rows) <= 1


def test_v7_max_pitch_and_yaw_actions_have_expected_physical_direction():
    cfg = load_config(ENV_V7)
    spec = aircraft_spec(cfg)
    controller = TargetStateController(**cfg["action"], gravity=cfg["simulation"]["gravity"])
    dynamics = PointMassDynamics(cfg["simulation"]["gravity"])
    integrator = RK4Integrator(cfg["simulation"]["dt"])

    base = AircraftState(0.0, 0.0, -3000.0, 150.0, 0.0, 0.0)

    state = base.copy()
    for _ in range(20):
        _, control = controller.control_from_action(state, np.array([0.0, 1.0, 0.0]), spec)
        state = integrator.step(state, control, dynamics, spec)
    assert state.theta > 0.0
    assert state.altitude > base.altitude

    state = base.copy()
    for _ in range(20):
        _, control = controller.control_from_action(state, np.array([0.0, -1.0, 0.0]), spec)
        state = integrator.step(state, control, dynamics, spec)
    assert state.theta < 0.0
    assert state.altitude < base.altitude

    state = base.copy()
    for _ in range(20):
        _, control = controller.control_from_action(state, np.array([1.0, 0.0, 0.0]), spec)
        state = integrator.step(state, control, dynamics, spec)
    assert state.psi > 0.0
    assert state.y > 0.0

    state = base.copy()
    for _ in range(20):
        _, control = controller.control_from_action(state, np.array([-1.0, 0.0, 0.0]), spec)
        state = integrator.step(state, control, dynamics, spec)
    assert state.psi < 0.0
    assert state.y < 0.0
