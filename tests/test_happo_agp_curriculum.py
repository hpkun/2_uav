from __future__ import annotations

from copy import deepcopy
import numpy as np
import pytest
import torch

from algorithm.happo.agp import (
    apply_agp,
    pair_potential,
    smooth_aa_gate,
    smooth_ata_gate,
    smooth_distance_gate,
    team_potential_from_observations,
)
from algorithm.happo.curriculum import nearest_probability
from algorithm.happo import HAPPOTrainer
from env.blue_policy import BluePolicy
from env.mavuav import load_environment_config
from env.vector_env import MAVUAVVectorEnv


DISTANCE_SCALE = 12_000.0


def _observations(*, red_alive: tuple[bool, bool, bool] = (True, True, True)) -> np.ndarray:
    observations = np.zeros((1, 3, 61), dtype=np.float32)
    for agent, alive in enumerate(red_alive):
        observations[0, agent, 6] = float(alive)
    return observations


def _set_blue(
    observations: np.ndarray,
    red: int,
    blue: int,
    *,
    distance: float = 2000.0,
    ata: float = 15.0,
    aa: float = 45.0,
    alive: bool = True,
    direct: bool = True,
    datalink: bool = False,
    killed: bool = False,
) -> None:
    start = (33, 44)[blue]
    observations[0, red, start + 3] = distance / DISTANCE_SCALE
    observations[0, red, start + 7] = ata / 180.0
    observations[0, red, start + 8] = aa / 180.0
    observations[0, red, start + 9] = float(alive)
    observations[0, red, start + 10] = float(direct)
    observations[0, red, start + 11] = float(datalink)
    observations[0, red, start + 13] = float(killed)


def _short_config(max_steps: int = 1) -> dict:
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def _trainer_config(method: str, *, total_steps: int = 8, rollout_steps: int = 1) -> dict:
    return {
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "ppo_epochs": 1,
        "minibatch_size": 1,
        "hidden_dim": 8,
        "seed": 17,
        "environment_profile": "learnability",
        "actor_variant": "vanilla",
        "method_variant": method,
        "agp_lambda": 0.5,
        "curriculum_total_steps": total_steps if method in ("curriculum", "agp_curriculum") else None,
    }


def test_agp_smooth_gates_and_pair_bounds():
    np.testing.assert_allclose(smooth_distance_gate([1000, 2000, 3000]), 1.0)
    assert smooth_distance_gate(500) < 1.0
    assert smooth_distance_gate(4000) < 1.0
    assert smooth_ata_gate(30) == pytest.approx(0.5)
    assert smooth_ata_gate(10) > smooth_ata_gate(30) > smooth_ata_gate(60)
    assert smooth_aa_gate(90) == pytest.approx(0.5)
    values = pair_potential(
        np.linspace(0, 20_000, 100),
        np.linspace(0, 180, 100),
        np.linspace(0, 180, 100),
    )
    assert np.isfinite(values).all() and np.all((0.0 <= values) & (values <= 1.0))


def test_team_potential_observation_contract_visibility_liveness_and_fixed_normalization():
    observations = _observations(red_alive=(True, False, False))
    _set_blue(observations, 0, 0)
    expected_pair = float(pair_potential(2000.0, 15.0, 45.0))
    assert team_potential_from_observations(observations, DISTANCE_SCALE)[0] == pytest.approx(expected_pair / 3.0)

    for change in (
        {"direct": False, "datalink": False},
        {"alive": False},
        {"killed": True},
    ):
        changed = _observations(red_alive=(True, False, False))
        _set_blue(changed, 0, 0, **change)
        assert team_potential_from_observations(changed, DISTANCE_SCALE)[0] == 0.0
    dead_red = _observations(red_alive=(False, False, False))
    _set_blue(dead_red, 0, 0)
    assert team_potential_from_observations(dead_red, DISTANCE_SCALE)[0] == 0.0
    assert team_potential_from_observations(np.zeros((2, 3, 61), dtype=np.float32), DISTANCE_SCALE).shape == (2,)
    with pytest.raises(ValueError, match=r"\[B, 3, 61\]"):
        team_potential_from_observations(np.zeros((3, 61), dtype=np.float32), DISTANCE_SCALE)


def test_agp_terminal_zeroes_auto_reset_successor_and_scale_is_exact():
    current = _observations()
    following = _observations()
    for red in range(3):
        _set_blue(current, red, 0, distance=2500, ata=20, aa=60)
        _set_blue(following, red, 0, distance=2000, ata=0, aa=0)
    base = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    current_phi = team_potential_from_observations(current, DISTANCE_SCALE)[0]
    rewards, raw, shaping = apply_agp(
        base, current, following, np.asarray([True]), DISTANCE_SCALE,
        gamma=0.99, agp_lambda=0.5,
    )
    assert raw[0] == pytest.approx(-current_phi)
    assert shaping[0] == pytest.approx(0.5 * (0.99 * 0.0 - current_phi))
    np.testing.assert_allclose(rewards - base, np.full((1, 3), shaping[0]), rtol=1e-6, atol=1e-6)

    nonterminal_rewards, nonterminal_raw, nonterminal_shaping = apply_agp(
        base, current, following, np.asarray([False]), DISTANCE_SCALE,
        gamma=0.99, agp_lambda=0.5,
    )
    next_phi = team_potential_from_observations(following, DISTANCE_SCALE)[0]
    assert nonterminal_raw[0] == pytest.approx(0.99 * next_phi - current_phi)
    assert nonterminal_shaping[0] == pytest.approx(0.5 * nonterminal_raw[0])
    np.testing.assert_allclose(nonterminal_rewards - base, np.full((1, 3), nonterminal_shaping[0]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("steps", "expected"),
    [(0, 0.0), (249, 0.0), (250, 0.25), (500, 0.75), (750, 1.0), (1000, 1.0)],
)
def test_curriculum_schedule_boundaries(steps, expected):
    assert nearest_probability(steps, 1000) == expected


def test_blue_reset_probability_and_baseline_seeded_sequence_are_exact():
    policy = BluePolicy("mixed_episode", 1.0, 0.1)
    assert {policy.reset(np.random.default_rng(seed), nearest_probability=0.0) for seed in range(8)} == {"mav_priority"}
    assert {policy.reset(np.random.default_rng(seed), nearest_probability=1.0) for seed in range(8)} == {"nearest"}
    with pytest.raises(ValueError, match="mixed_episode"):
        BluePolicy("nearest", 1.0, 0.1).reset(np.random.default_rng(1), nearest_probability=0.5)

    actual_rng = np.random.default_rng(29)
    expected_rng = np.random.default_rng(29)
    actual_policy = BluePolicy("mixed_episode", 1.0, 0.1)
    actual = [actual_policy.reset(actual_rng, nearest_probability=None) for _ in range(20)]
    expected = [str(expected_rng.choice(("nearest", "mav_priority"))) for _ in range(20)]
    assert actual == expected


@pytest.mark.parametrize("parallel", [False, True])
def test_vector_auto_reset_changes_only_the_next_episode_mode(parallel):
    with MAVUAVVectorEnv(
        1, _short_config(1), seed=7, blue_target_mode="mixed_episode", parallel=parallel,
    ) as vector:
        *_, reset_infos = vector.reset(nearest_probability=0.0)
        assert reset_infos[0]["blue_target_mode"] == "mav_priority"
        *_, infos = vector.step(
            np.zeros((1, 3, 3), dtype=np.float32), reset_nearest_probability=1.0,
        )
        assert infos[0]["episode_summary"]["blue_target_mode"] == "mav_priority"
        assert infos[0]["reset_info"]["blue_target_mode"] == "nearest"
        assert vector.get_env_states()[0]["blue_episode_mode"] == "nearest"


def test_baseline_rollout_reward_is_unshaped_and_method_isolation_preserves_actor_size(monkeypatch):
    counts = []
    for method in ("baseline", "agp", "curriculum", "agp_curriculum"):
        trainer = HAPPOTrainer(_short_config(2), _trainer_config(method, total_steps=4))
        counts.append(trainer.actor_parameter_counts["total"])
        assert trainer.agp_enabled == (method in ("agp", "agp_curriculum"))
        assert trainer.curriculum_enabled == (method in ("curriculum", "agp_curriculum"))
        assert trainer.config["actor_variant"] == "vanilla"
        if method == "baseline":
            captured = []
            original_step = trainer.vector_env.step

            def capture(*args, **kwargs):
                result = original_step(*args, **kwargs)
                captured.append(result[2].copy())
                return result

            monkeypatch.setattr(trainer.vector_env, "step", capture)
            trainer.collect_rollout()
            np.testing.assert_array_equal(trainer.buffer.rewards[0], captured[0][:, 0])
            np.testing.assert_array_equal(captured[0], np.repeat(captured[0][:, :1], 3, axis=1))
            assert trainer.last_rollout_metrics["agp_shaping_mean_abs"] == 0.0
        trainer.close()
    assert len(set(counts)) == 1


def test_curriculum_exposure_counts_terminal_transition_before_reset_mode():
    trainer = HAPPOTrainer(_short_config(1), _trainer_config("curriculum", total_steps=4))
    assert trainer.current_blue_modes == ["mav_priority"]
    trainer.collect_rollout()
    assert trainer.mode_transition_counts == {"nearest": 0, "mav_priority": 1}
    assert trainer.mode_episode_counts == {"nearest": 0, "mav_priority": 1}
    assert trainer.last_rollout_metrics["p_nearest"] == 0.0
    trainer.close()


def test_curriculum_checkpoint_exact_resume_restores_modes_rng_and_exposure(tmp_path):
    config = _trainer_config("agp_curriculum", total_steps=8, rollout_steps=2)
    source = HAPPOTrainer(_short_config(1), config)
    checkpoint = tmp_path / "combined.pt"
    source.save_checkpoint(checkpoint)
    source.collect_rollout()
    expected = {
        "observations": source.observations.copy(),
        "states": source.global_states.copy(),
        "modes": list(source.current_blue_modes),
        "transitions": dict(source.mode_transition_counts),
        "episodes": dict(source.mode_episode_counts),
        "env_states": source.vector_env.get_env_states(),
    }

    restored = HAPPOTrainer(_short_config(1), config)
    assert restored.load_checkpoint(checkpoint) == 0
    restored.collect_rollout()
    np.testing.assert_array_equal(restored.observations, expected["observations"])
    np.testing.assert_array_equal(restored.global_states, expected["states"])
    assert restored.current_blue_modes == expected["modes"]
    assert restored.mode_transition_counts == expected["transitions"]
    assert restored.mode_episode_counts == expected["episodes"]
    assert restored.vector_env.get_env_states() == expected["env_states"]
    source.close()
    restored.close()


def test_method_checkpoint_mismatch_and_curriculum_horizon_change_are_rejected(tmp_path):
    source = HAPPOTrainer(_short_config(1), _trainer_config("curriculum", total_steps=8))
    checkpoint = tmp_path / "curriculum.pt"
    source.save_checkpoint(checkpoint)
    source.close()

    baseline = HAPPOTrainer(_short_config(1), _trainer_config("baseline"))
    with pytest.raises(RuntimeError, match="method mismatch"):
        baseline.load_checkpoint(checkpoint)
    baseline.close()

    changed_horizon = HAPPOTrainer(_short_config(1), _trainer_config("curriculum", total_steps=16))
    with pytest.raises(RuntimeError, match="total steps mismatch"):
        changed_horizon.load_checkpoint(checkpoint)
    changed_horizon.close()


def test_old_baseline_checkpoint_without_method_metadata_remains_compatible(tmp_path):
    config = _trainer_config("baseline")
    source = HAPPOTrainer(_short_config(1), config)
    checkpoint = tmp_path / "legacy.pt"
    source.save_checkpoint(checkpoint)
    source.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("method_variant")
    payload["trainer_config"].pop("method_variant")
    rollout = payload["rollout_state"]
    for field in ("current_blue_modes", "mode_transition_counts", "mode_episode_counts"):
        rollout.pop(field)
    torch.save(payload, checkpoint)
    restored = HAPPOTrainer(_short_config(1), config)
    assert restored.load_checkpoint(checkpoint) == 0
    assert len(restored.current_blue_modes) == 1
    restored.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_combined_real_cuda_checkpoint_resume_and_update(tmp_path):
    config = _trainer_config("agp_curriculum", total_steps=4, rollout_steps=1)
    config["device"] = "cuda:0"
    source = HAPPOTrainer(_short_config(1), config)
    _, source_metrics = source.train_update()
    checkpoint = tmp_path / "combined_cuda.pt"
    source.save_checkpoint(checkpoint)

    restored = HAPPOTrainer(_short_config(1), config)
    assert restored.load_checkpoint(checkpoint) == 1
    _, resumed_metrics = restored.train_update()
    assert restored.env_steps == 2
    assert all(np.isfinite(value) for value in source_metrics.values() if isinstance(value, float))
    assert all(np.isfinite(value) for value in resumed_metrics.values() if isinstance(value, float))
    assert restored.current_blue_modes == [restored.vector_env.get_env_states()[0]["blue_episode_mode"]]
    source.close()
    restored.close()
