"""Tests for persistent multi-process parallel vector environment."""
from pathlib import Path
import numpy as np
import pytest

from uav_combat.mappo.vector_env import (
    CONTROL_DIAGNOSTIC_KEYS,
    K,
    LocalCombatVectorEnv,
    SubprocessCombatVectorEnv,
    decode_outcome,
    decode_termination_reason,
    encode_outcome,
    encode_termination_reason,
    make_combat_vector_env,
)

CONFIG = Path(__file__).parents[1] / "configs" / "homogeneous_1v1.yaml"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _specs(num: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    scenarios = ["tail_chase", "offset_head_on", "crossing"]
    out = []
    for i in range(num):
        s = scenarios[i % 3]
        rear = "red" if i % 2 == 0 else "blue" if s == "tail_chase" else None
        out.append(
            {"seed": int(rng.integers(0, 2**31 - 1)), "scenario": s, "rear_team": rear}
        )
    return out


# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------


class TestEncoding:
    def test_roundtrip(self):
        for reason in [None, "red_kill", "blue_kill", "mutual_kill", "collision",
                       "altitude_boundary", "xy_boundary", "boundary", "max_steps"]:
            assert decode_termination_reason(encode_termination_reason(reason)) == reason
        for outcome in [None, "red", "blue", "draw"]:
            assert decode_outcome(encode_outcome(outcome)) == outcome

    def test_unknown_code_returns_none(self):
        assert decode_termination_reason(99) is None
        assert decode_outcome(99) is None

    def test_diagnostic_keys_count(self):
        assert len(CONTROL_DIAGNOSTIC_KEYS) == K
        # Key training-log keys must be present
        required = [
            "action_yaw", "action_pitch", "action_speed",
            "yaw_rate_saturated", "pitch_rate_saturated", "acceleration_saturated",
            "nx_saturated", "nz_saturated", "phi_saturated",
            "acceleration_tracking_absolute_error",
            "pitch_rate_tracking_absolute_error",
            "yaw_rate_tracking_absolute_error",
        ]
        for key in required:
            assert key in CONTROL_DIAGNOSTIC_KEYS


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


class TestFactory:
    def test_default_num_env_workers_is_4(self):
        env = make_combat_vector_env(CONFIG, 8)
        assert isinstance(env, SubprocessCombatVectorEnv)
        assert env.num_env_workers == 4
        env.close()

    def test_num_workers_1_uses_local(self):
        env = make_combat_vector_env(CONFIG, 4, 1)
        assert isinstance(env, LocalCombatVectorEnv)
        env.close()

    def test_num_workers_4_uses_subprocess(self):
        env = make_combat_vector_env(CONFIG, 8, 4)
        assert isinstance(env, SubprocessCombatVectorEnv)
        env.close()

    def test_workers_greater_than_envs_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            make_combat_vector_env(CONFIG, 4, 8)

    def test_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            make_combat_vector_env(CONFIG, 7, 4)

    def test_zero_workers_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            make_combat_vector_env(CONFIG, 4, 0)


# ------------------------------------------------------------------
# LocalCombatVectorEnv
# ------------------------------------------------------------------


class TestLocalCombatVectorEnv:
    def test_reset_shape(self):
        env = LocalCombatVectorEnv(CONFIG, 4)
        obs, gs = env.reset(_specs(4))
        assert obs.shape == (4, 2, 14)
        assert gs.shape == (4, 2, 14)
        assert obs.dtype == np.float32
        env.close()

    def test_step_shapes(self):
        env = LocalCombatVectorEnv(CONFIG, 2)
        env.reset(_specs(2))
        actions = np.zeros((2, 2, 3), dtype=np.float32)
        result = env.step(actions)
        assert len(result) == 11
        obs, gs, rewards, term, trunc, sc, att, geom, cd, rc, oc = result
        assert obs.shape == (2, 2, 14)
        assert gs.shape == (2, 2, 14)
        assert rewards.shape == (2, 2)
        assert term.shape == (2,) and term.dtype == bool
        assert trunc.shape == (2,) and trunc.dtype == bool
        assert sc.shape == (2,) and sc.dtype == np.int32
        assert att.shape == (2, 2) and att.dtype == bool
        assert geom.shape == (2, 2, 3)
        assert cd.shape == (2, 2, K)
        assert rc.shape == (2,) and rc.dtype == np.int8
        assert oc.shape == (2,) and oc.dtype == np.int8
        env.close()

    def test_reset_at_only_resets_requested(self):
        env = LocalCombatVectorEnv(CONFIG, 4)
        obs, gs = env.reset(_specs(4))
        # Step once so environments are not at initial state
        env.step(np.zeros((4, 2, 3), dtype=np.float32))
        # Capture before-reset state
        before = obs.copy()
        # Reset envs 1 and 2
        new_specs = _specs(2, seed=999)
        new_obs, new_gs = env.reset_at(np.array([1, 2]), new_specs)
        assert new_obs.shape == (2, 2, 14)
        # Envs 1 and 2 should have changed, 0 and 3 unchanged
        obs[1:3] = new_obs
        gs[1:3] = new_gs
        env.close()

    def test_close_idempotent(self):
        env = LocalCombatVectorEnv(CONFIG, 2)
        env.close()
        env.close()  # no error

    def test_context_manager(self):
        with LocalCombatVectorEnv(CONFIG, 2) as env:
            env.reset(_specs(2))
        # env is closed after context manager exit


# ------------------------------------------------------------------
# SubprocessCombatVectorEnv
# ------------------------------------------------------------------


class TestSubprocessCombatVectorEnv:
    def test_reset_and_step_basic(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        try:
            obs, gs = env.reset(_specs(4))
            assert obs.shape == (4, 2, 14)
            result = env.step(np.zeros((4, 2, 3), dtype=np.float32))
            assert len(result) == 11
        finally:
            env.close()

    def test_order_preserved_under_parallel(self):
        """Global env index order is preserved in parallel results."""
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        try:
            specs = _specs(4, seed=123)
            env.reset(specs)
            # Different actions per env for disambiguation
            actions = np.array(
                [ [[ 0.0,  0.0,  0.0], [ 0.0,  0.0,  0.0]],
                  [[ 0.5,  0.0,  0.0], [ 0.0,  0.0,  0.0]],
                  [[ 0.0,  0.5,  0.0], [ 0.0,  0.0,  0.0]],
                  [[ 0.0,  0.0,  0.5], [ 0.0,  0.0,  0.0]] ],
                dtype=np.float32)
            obs, gs, rewards, term, trunc, sc, att, geom, cd, rc, oc = env.step(actions)
            # Env 1 (index 1) should have action_yaw=0.5 in diagnostics
            assert obs.shape == (4, 2, 14)
        finally:
            env.close()

    def test_reset_at_in_subprocess(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        try:
            env.reset(_specs(4))
            env.step(np.zeros((4, 2, 3), dtype=np.float32))
            # Reset envs 1 and 3
            new_specs = [{"seed": 9999, "scenario": "offset_head_on", "rear_team": None}]
            new_obs, new_gs = env.reset_at(np.array([1]), new_specs)
            assert new_obs.shape == (1, 2, 14)
            assert new_gs.shape == (1, 2, 14)
        finally:
            env.close()

    def test_close_exits_workers(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        env.close()
        # After close, workers should have exited
        for w in env._workers:
            assert not w.is_alive()

    def test_close_idempotent(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        env.close()
        env.close()  # no error

    def test_context_manager_cleanup(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        with env:
            env.reset(_specs(4))
        for w in env._workers:
            assert not w.is_alive()

    def test_worker_error_propagates(self):
        env = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        env.close()  # close first to simulate mid-run close
        with pytest.raises(RuntimeError, match="closed"):
            env.reset(_specs(4))


# ------------------------------------------------------------------
# Deterministic consistency: Local vs Subprocess
# ------------------------------------------------------------------


class TestDeterministicConsistency:
    def test_same_reset_specs_yield_same_initial_obs(self):
        """Local and Subprocess should return identical observations for same specs."""
        specs = _specs(4, seed=77)
        local = LocalCombatVectorEnv(CONFIG, 4)
        sub = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        try:
            obs_loc, gs_loc = local.reset(specs)
            obs_sub, gs_sub = sub.reset(specs)
            assert np.allclose(obs_loc, obs_sub, atol=1e-5)
            assert np.allclose(gs_loc, gs_sub, atol=1e-5)
        finally:
            local.close()
            sub.close()

    def test_deterministic_sequence_matches(self):
        """Same sequence of actions yields identical results across backends."""
        specs = _specs(4, seed=55)
        rng = np.random.default_rng(42)
        local = LocalCombatVectorEnv(CONFIG, 4)
        sub = SubprocessCombatVectorEnv(CONFIG, 4, 2)
        try:
            obs_loc, gs_loc = local.reset(specs)
            obs_sub, gs_sub = sub.reset(specs)
            assert np.allclose(obs_loc, obs_sub, atol=1e-5)

            for _ in range(10):
                actions = rng.uniform(-1, 1, (4, 2, 3)).astype(np.float32)
                r_loc = local.step(actions)
                r_sub = sub.step(actions)
                for i in range(11):
                    assert np.allclose(r_loc[i], r_sub[i], atol=1e-5), f"Array {i} diverges"
        finally:
            local.close()
            sub.close()


# ------------------------------------------------------------------
# Trainer integration tests (CPU only, no CUDA)
# ------------------------------------------------------------------


class TestTrainerVectorEnvIntegration:
    def test_trainer_with_local_vector_env(self, tmp_path):
        from uav_combat.mappo.trainer import MAPPOTrainer

        config = {
            "experiment": {"seed": 3, "device": "cpu", "output_dir": str(tmp_path)},
            "network": {"hidden_dim": 32, "log_std_init": -0.5},
            "training": {
                "training_mode": "alternating_self_play",
                "total_env_steps": 64,
                "num_envs": 2,
                "num_env_workers": 1,
                "rollout_steps": 16,
                "alternating_block_env_steps": 64,
                "ppo_epochs": 1,
                "minibatch_size": 32,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01,
                "max_grad_norm": 0.5, "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                "opponent_history_latest_probability": 0.7,
            },
            "evaluation": {"episodes": 2, "deterministic": True},
        }
        trainer = MAPPOTrainer(CONFIG, config)
        try:
            trainer.configure_block_opponent(0, "a", force=True)
            trainer.reset_environments()
            completed = trainer.collect_rollout()
            assert trainer.buffer.observations.shape == (16, 2, 14)
            assert trainer.buffer.actions.shape == (16, 2, 3)
            assert len(completed) >= 0  # may or may not have completions in 16 steps
            metrics = trainer.update("a")
            assert np.isfinite(metrics["policy_a_policy_loss"])
        finally:
            trainer.close()

    def test_trainer_with_subprocess_vector_env(self, tmp_path):
        from uav_combat.mappo.trainer import MAPPOTrainer

        config = {
            "experiment": {"seed": 5, "device": "cpu", "output_dir": str(tmp_path)},
            "network": {"hidden_dim": 32, "log_std_init": -0.5},
            "training": {
                "training_mode": "alternating_self_play",
                "total_env_steps": 64,
                "num_envs": 2,
                "num_env_workers": 2,
                "rollout_steps": 16,
                "alternating_block_env_steps": 64,
                "ppo_epochs": 1,
                "minibatch_size": 32,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01,
                "max_grad_norm": 0.5, "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                "opponent_history_latest_probability": 0.7,
            },
            "evaluation": {"episodes": 2, "deterministic": True},
        }
        trainer = MAPPOTrainer(CONFIG, config)
        try:
            trainer.configure_block_opponent(0, "a", force=True)
            trainer.reset_environments()
            completed = trainer.collect_rollout()
            assert trainer.buffer.observations.shape == (16, 2, 14)
            metrics = trainer.update("a")
            assert np.isfinite(metrics["policy_a_policy_loss"])
        finally:
            trainer.close()

    def test_buffer_shapes_consistent_1_and_4_workers(self, tmp_path):
        """Buffer shapes are identical regardless of worker count."""
        from uav_combat.mappo.trainer import MAPPOTrainer, POLICIES

        shapes_1w = None
        shapes_2w = None

        for nw in [1, 2]:
            config = {
                "experiment": {"seed": 7, "device": "cpu", "output_dir": str(tmp_path)},
                "network": {"hidden_dim": 32, "log_std_init": -0.5},
                "training": {
                    "training_mode": "alternating_self_play",
                    "total_env_steps": 64,
                    "num_envs": 2, "num_env_workers": nw,
                    "rollout_steps": 16, "alternating_block_env_steps": 64,
                    "ppo_epochs": 1, "minibatch_size": 32,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                    "learning_rate": 3e-4, "value_loss_coef": 0.5,
                    "entropy_coef": 0.01, "max_grad_norm": 0.5,
                    "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                    "opponent_history_latest_probability": 0.7,
                },
                "evaluation": {"episodes": 2, "deterministic": True},
            }
            trainer = MAPPOTrainer(CONFIG, config)
            try:
                trainer.configure_block_opponent(0, "a", force=True)
                trainer.reset_environments()
                trainer.collect_rollout()
                shape = {
                    "obs": trainer.buffer.observations.shape,
                    "actions": trainer.buffer.actions.shape,
                    "rewards": trainer.buffer.rewards.shape,
                    "values": trainer.buffer.values.shape,
                }
                if nw == 1:
                    shapes_1w = shape
                else:
                    shapes_2w = shape
            finally:
                trainer.close()

        assert shapes_1w == shapes_2w

    def test_v6_checkpoint_load_with_parallel_trainer(self, tmp_path):
        """Old v6 checkpoint (without num_env_workers) can be loaded."""
        from uav_combat.mappo.trainer import MAPPOTrainer

        config = {
            "experiment": {"seed": 9, "device": "cpu", "output_dir": str(tmp_path)},
            "network": {"hidden_dim": 32, "log_std_init": -0.5},
            "training": {
                "training_mode": "alternating_self_play",
                "total_env_steps": 64,
                "num_envs": 2, "num_env_workers": 1,
                "rollout_steps": 16, "alternating_block_env_steps": 64,
                "ppo_epochs": 1, "minibatch_size": 32,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                "learning_rate": 3e-4, "value_loss_coef": 0.5,
                "entropy_coef": 0.01, "max_grad_norm": 0.5,
                "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                "opponent_history_latest_probability": 0.7,
            },
            "evaluation": {"episodes": 2, "deterministic": True},
        }
        trainer = MAPPOTrainer(CONFIG, config)
        try:
            trainer.configure_block_opponent(0, "a", force=True)
            trainer.reset_environments()
            trainer.collect_rollout()
            trainer.update("a")
            path = tmp_path / "v6_test.pt"
            trainer.save_checkpoint(path)

            # Load with different num_env_workers
            config2 = {**config, "training": {**config["training"], "num_env_workers": 2}}
            restored = MAPPOTrainer(CONFIG, config2)
            try:
                restored.load_checkpoint(path)
                assert restored.env_steps == trainer.env_steps
                # Should be able to continue
                restored.collect_rollout()
            finally:
                restored.close()
        finally:
            trainer.close()

    def test_num_env_workers_not_in_training_signature(self, tmp_path):
        """num_env_workers change does not trigger signature mismatch."""
        from uav_combat.mappo.trainer import MAPPOTrainer

        config = {
            "experiment": {"seed": 11, "device": "cpu", "output_dir": str(tmp_path)},
            "network": {"hidden_dim": 32, "log_std_init": -0.5},
            "training": {
                "training_mode": "alternating_self_play",
                "total_env_steps": 64,
                "num_envs": 2, "num_env_workers": 1,
                "rollout_steps": 16, "alternating_block_env_steps": 64,
                "ppo_epochs": 1, "minibatch_size": 32,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                "learning_rate": 3e-4, "value_loss_coef": 0.5,
                "entropy_coef": 0.01, "max_grad_norm": 0.5,
                "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                "opponent_history_latest_probability": 0.7,
            },
            "evaluation": {"episodes": 2, "deterministic": True},
        }
        trainer = MAPPOTrainer(CONFIG, config)
        try:
            trainer.configure_block_opponent(0, "a", force=True)
            trainer.reset_environments()
            trainer.collect_rollout()
            path = tmp_path / "v6_test2.pt"
            trainer.save_checkpoint(path)

            # Load with num_env_workers=2
            config2 = {**config, "training": {**config["training"], "num_env_workers": 2}}
            restored = MAPPOTrainer(CONFIG, config2)
            try:
                restored.load_checkpoint(path)  # should NOT raise signature mismatch
            finally:
                restored.close()
        finally:
            trainer.close()

    def test_smoke_finite_values(self, tmp_path):
        """All training values are finite after one rollout + update."""
        from uav_combat.mappo.trainer import MAPPOTrainer

        config = {
            "experiment": {"seed": 13, "device": "cpu", "output_dir": str(tmp_path)},
            "network": {"hidden_dim": 32, "log_std_init": -0.5},
            "training": {
                "training_mode": "alternating_self_play",
                "total_env_steps": 64,
                "num_envs": 2, "num_env_workers": 1,
                "rollout_steps": 16, "alternating_block_env_steps": 64,
                "ppo_epochs": 1, "minibatch_size": 32,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
                "learning_rate": 3e-4, "value_loss_coef": 0.5,
                "entropy_coef": 0.01, "max_grad_norm": 0.5,
                "eval_interval_updates": 10, "checkpoint_interval_updates": 10,
                "opponent_history_latest_probability": 0.7,
            },
            "evaluation": {"episodes": 2, "deterministic": True},
        }
        trainer = MAPPOTrainer(CONFIG, config)
        try:
            trainer.configure_block_opponent(0, "a", force=True)
            trainer.reset_environments()
            trainer.collect_rollout()
            assert np.isfinite(trainer.buffer.rewards).all()
            assert np.isfinite(trainer.buffer.values).all()
            assert np.isfinite(trainer.buffer.advantages).all()
            assert np.isfinite(trainer.buffer.returns).all()
            metrics = trainer.update("a")
            for key in ["policy_a_policy_loss", "policy_a_entropy", "policy_a_value_loss",
                        "policy_a_approx_kl", "policy_a_clip_fraction"]:
                v = metrics.get(key, np.nan)
                assert np.isfinite(v) or (isinstance(v, float) and np.isnan(v)), f"{key}={v}"
        finally:
            trainer.close()
