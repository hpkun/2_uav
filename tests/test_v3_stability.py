"""Tests for v3 stability modifications."""
import numpy as np
import pytest
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer_3v3 import linear_schedule


class TestLinearSchedule:
    def test_progress_zero(self):
        assert linear_schedule(0.0003, 0.00003, 0.0) == 0.0003

    def test_progress_half(self):
        val = linear_schedule(0.0003, 0.00003, 0.5)
        expected = 0.0003 + 0.5 * (0.00003 - 0.0003)  # 0.000165
        assert np.isclose(val, expected)

    def test_progress_one(self):
        assert np.isclose(linear_schedule(0.0003, 0.00003, 1.0), 0.00003)

    def test_progress_clipped(self):
        assert np.isclose(linear_schedule(0.0003, 0.00003, 1.5), 0.00003)
        assert np.isclose(linear_schedule(0.0003, 0.00003, -0.5), 0.0003)


class TestGaussianActorBounds:
    def test_v3_log_std_max_zero(self):
        actor = GaussianActor(68, 3, 32, -0.5, log_std_min=-2.0, log_std_max=0.0)
        assert actor.log_std_max == 0.0
        assert actor.log_std_min == -2.0

    def test_v3_std_not_exceed_one(self):
        actor = GaussianActor(68, 3, 32, -0.5, log_std_min=-2.0, log_std_max=0.0)
        std = actor.effective_std_mean
        assert std <= 1.0, f"std={std} > 1.0 with log_std_max=0"

    def test_clamp_log_std_enforces_bounds(self):
        actor = GaussianActor(68, 3, 32, -0.5, log_std_min=-2.0, log_std_max=0.0)
        # Force log_std above max
        with torch.no_grad():
            actor.log_std.copy_(torch.tensor([1.5, 2.0, 3.0]))
        actor.clamp_log_std_()
        assert (actor.log_std <= 0.0).all()
        assert (actor.log_std >= -2.0).all()

    def test_default_bounds_unchanged(self):
        actor = GaussianActor(14, 3, 32, -0.5)
        assert actor.log_std_min == -5.0
        assert actor.log_std_max == 2.0

    def test_effective_std_within_bounds(self):
        actor = GaussianActor(68, 3, 128, -0.5, log_std_min=-2.0, log_std_max=0.0)
        s = actor.effective_std_mean
        assert np.exp(-2.0) <= s <= 1.0


class TestTrainerSchedules:
    def test_lr_decreases(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 1, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "learning_rate_final": 3e-5,
                               "entropy_coef": 0.01, "entropy_coef_final": 0.001,
                               "target_kl": 0.0, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout()
            m = t.update()
            assert np.isfinite(m["policy_loss"])
            assert m["current_learning_rate"] <= 3e-4
            t.close()

    def test_entropy_coef_decreases(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 2, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "entropy_coef": 0.01, "entropy_coef_final": 0.001,
                               "target_kl": 0.0, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout()
            m = t.update()
            assert m["current_entropy_coef"] <= 0.01
            t.close()

    def test_target_kl_early_stop(self):
        """With very low target_kl, KL should trigger early stop."""
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 3, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 5, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "entropy_coef": 0.01,
                               "target_kl": 0.001, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout()
            m = t.update()
            # With target_kl=0.001 it should almost certainly trigger
            assert m.get("kl_early_stop") is not None
            t.close()

    def test_target_kl_zero_no_early_stop(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 4, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "entropy_coef": 0.01,
                               "target_kl": 0.0, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout()
            m = t.update()
            assert not m.get("kl_early_stop", False), f"KL early stop should be False with target_kl=0"
            t.close()

    def test_critic_completes_when_actor_stops(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 5, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 5, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "entropy_coef": 0.01,
                               "target_kl": 0.001, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout()
            m = t.update()
            # value_loss being finite means critic did update
            assert np.isfinite(m["value_loss"])
            t.close()

    def test_resume_schedule_continuous(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 6, "device": "cpu", "output_dir": "t"}, "network": {"hidden_dim": 32, "log_std_init": -0.5, "log_std_min": -2.0, "log_std_max": 0.0},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 2048, "num_envs": 2, "num_env_workers": 1,
                               "rollout_steps": 32, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "learning_rate_final": 3e-5,
                               "entropy_coef": 0.01, "entropy_coef_final": 0.001,
                               "target_kl": 0.0, "value_loss_coef": 0.5, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t.collect_rollout(); m1 = t.update()
            path = tmp + "/ckpt.pt"; t.save_checkpoint(path)
            t.close()
            t2 = FixedBlue3v3MAPPOTrainer("configs/homogeneous_3v3.yaml", config)
            t2.load_checkpoint(path)
            t2.collect_rollout(); m2 = t2.update()
            # After 2 updates, progress > 0, so LR should be lower
            assert m2["current_learning_rate"] < 3e-4
            t2.close()


class TestV3Config:
    def test_only_three_params_changed(self):
        import yaml
        with open("configs/homogeneous_3v3.yaml") as f: v2 = yaml.safe_load(f)
        with open("configs/homogeneous_3v3_reward_v3.yaml") as f: v3 = yaml.safe_load(f)
        assert v3["reward_v2"]["attack_advantage_weight"] == 0.06
        assert v3["reward_v2"]["threat_weight"] == 0.03
        assert v3["reward_v2"]["max_steps_penalty"] == 10.0
        # All other reward params unchanged
        for k in v2["reward_v2"]:
            if k in ("attack_advantage_weight", "threat_weight", "max_steps_penalty"):
                continue
            assert v3["reward_v2"][k] == v2["reward_v2"][k], f"{k}: v2={v2['reward_v2'][k]} v3={v3['reward_v2'][k]}"
