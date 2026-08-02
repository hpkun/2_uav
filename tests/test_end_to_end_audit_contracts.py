from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_combat.config import aircraft_spec, load_config
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv
from uav_combat.integrator import RK4Integrator
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
from uav_combat.models import AircraftState


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


def test_v7_observation_self_block_contains_signed_altitude_boundary_information():
    env = Homogeneous3v3AirCombatEnv(ENV_V7)
    try:
        env.reset(123)
        red = env._aircraft_by_id("red_0")
        bf = env.config["battlefield"]
        red.state.altitude == -red.state.z

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

