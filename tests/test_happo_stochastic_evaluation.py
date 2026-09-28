"""Focused stochastic-policy evaluation tests for the standalone HAPPO evaluator."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import algorithm.happo.evaluation as evaluation
import algorithm.train_happo as training_entrypoint
from algorithm.evaluate_happo import parse_args
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.train_happo import _evaluation_row
from env.mavuav import RED_IDS, load_environment_config


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "configs" / "env_v39.yaml"


class RecordingActor:
    def __init__(self) -> None:
        self.deterministic_arguments: list[bool] = []

    def sample(self, observation: torch.Tensor, deterministic: bool = False):
        self.deterministic_arguments.append(bool(deterministic))
        action = torch.zeros((len(observation), 3), device=observation.device)
        if not deterministic:
            action = torch.randn((len(observation), 3), device=observation.device).tanh()
        return action, torch.zeros(len(observation), device=observation.device)


class RecordingActors:
    def __init__(self) -> None:
        self.actors = [RecordingActor() for _ in RED_IDS]


class RecordingEnv:
    instances: list["RecordingEnv"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.red_ids = RED_IDS
        self.reset_seeds: list[int] = []
        self.action_trajectory: list[np.ndarray] = []
        self.step_count = 0
        RecordingEnv.instances.append(self)

    def reset(self, seed: int):
        self.reset_seeds.append(int(seed))
        self.step_count = 0
        return {aid: np.zeros(1, np.float32) for aid in RED_IDS}, {}

    def step(self, actions: np.ndarray):
        self.action_trajectory.append(np.asarray(actions).copy())
        self.step_count += 1
        done = self.step_count == 2
        observations = {aid: np.zeros(1, np.float32) for aid in RED_IDS}
        summary = {
            "outcome": "draw", "episode_return": float(np.asarray(actions).sum()),
            "mav_survived": True, "red_uav_survivors": 3,
            "red_attack_kills": 0, "blue_attack_kills": 0, "episode_length": 2,
        }
        return observations, {}, done, False, {"episode_summary": summary}


def run_recording_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    *, deterministic: bool = True,
    action_seed: int | None = None,
    env_seed: int = 1000,
    episodes: int = 3,
):
    RecordingEnv.instances.clear()
    monkeypatch.setattr(evaluation, "HeterogeneousMAVUAVAirCombatEnv", RecordingEnv)
    actors = RecordingActors()
    records = evaluate_actors(
        actors, None, episodes, "main", seed=env_seed, device="cpu",
        deterministic=deterministic, action_seed=action_seed,
    )
    env = RecordingEnv.instances[-1]
    return records, env, actors


def test_cli_defaults_to_deterministic_action_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_happo.py", "checkpoint.pt"])
    args = parse_args()
    assert args.action_mode == "deterministic"
    assert args.action_seed == 2000


def test_training_cli_defaults_to_deterministic_and_accepts_stochastic(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_happo.py"])
    defaults = training_entrypoint.parse_args()
    assert defaults.eval_action_mode == "deterministic"
    assert defaults.eval_action_seed == 2000
    monkeypatch.setattr(sys, "argv", [
        "train_happo.py", "--eval-action-mode", "stochastic",
        "--eval-action-seed", "3456",
    ])
    configured = training_entrypoint.parse_args()
    assert configured.eval_action_mode == "stochastic"
    assert configured.eval_action_seed == 3456


def test_default_evaluate_actors_behavior_remains_deterministic(monkeypatch):
    first, first_env, first_actors = run_recording_evaluation(monkeypatch)
    second, second_env, _ = run_recording_evaluation(
        monkeypatch, deterministic=True, action_seed=987654,
    )
    assert first == second
    assert first_env.reset_seeds == second_env.reset_seeds == [1000, 1001, 1002]
    assert all(np.array_equal(a, b) for a, b in zip(first_env.action_trajectory, second_env.action_trajectory))
    assert all(actor.deterministic_arguments == [True] * 6 for actor in first_actors.actors)


def test_stochastic_mode_samples_once_and_is_reproducible_by_action_seed(monkeypatch):
    _, first_env, first_actors = run_recording_evaluation(
        monkeypatch, deterministic=False, action_seed=2000,
    )
    _, second_env, _ = run_recording_evaluation(
        monkeypatch, deterministic=False, action_seed=2000,
    )
    assert first_env.reset_seeds == second_env.reset_seeds == [1000, 1001, 1002]
    assert len(first_env.action_trajectory) == 6
    assert all(actor.deterministic_arguments == [False] * 6 for actor in first_actors.actors)
    assert all(np.array_equal(a, b) for a, b in zip(first_env.action_trajectory, second_env.action_trajectory))


def test_different_action_seed_changes_actions_without_changing_environment_seeds(monkeypatch):
    _, first_env, _ = run_recording_evaluation(
        monkeypatch, deterministic=False, action_seed=2000,
    )
    _, second_env, _ = run_recording_evaluation(
        monkeypatch, deterministic=False, action_seed=3000,
    )
    assert first_env.reset_seeds == second_env.reset_seeds == [1000, 1001, 1002]
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(first_env.action_trajectory, second_env.action_trajectory)
    )


def test_deterministic_and_stochastic_share_environment_episode_seeds(monkeypatch):
    _, deterministic_env, _ = run_recording_evaluation(monkeypatch, deterministic=True)
    _, stochastic_env, _ = run_recording_evaluation(
        monkeypatch, deterministic=False, action_seed=2000,
    )
    assert deterministic_env.reset_seeds == stochastic_env.reset_seeds == [1000, 1001, 1002]


def test_cli_stochastic_outputs_are_separate_and_include_action_metadata(tmp_path):
    env = deepcopy(load_environment_config(V39))
    env["simulation"]["max_decision_steps"] = 1
    config = {
        "num_envs": 1, "rollout_steps": 1, "hidden_dim": 8,
        "device": "cpu", "environment_profile": "learnability",
    }
    trainer = HAPPOTrainer(env, config)
    checkpoint = tmp_path / "checkpoint_final.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    base = [
        sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
        "--profile", "learnability", "--episodes", "1", "--device", "cpu",
    ]
    deterministic = subprocess.run(base, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert deterministic.returncode == 0, deterministic.stderr
    deterministic_csv = tmp_path / "evaluation_final.csv"
    deterministic_summary = tmp_path / "evaluation_final_summary.json"
    original_csv = deterministic_csv.read_bytes()
    original_summary = deterministic_summary.read_bytes()
    stochastic = subprocess.run(
        [*base, "--action-mode", "stochastic", "--action-seed", "3456"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert stochastic.returncode == 0, stochastic.stderr
    assert deterministic_csv.read_bytes() == original_csv
    assert deterministic_summary.read_bytes() == original_summary
    stochastic_csv = tmp_path / "evaluation_final_stochastic.csv"
    stochastic_summary = tmp_path / "evaluation_final_stochastic_summary.json"
    assert stochastic_csv.is_file() and stochastic_summary.is_file()
    with deterministic_csv.open(newline="") as stream:
        deterministic_row = list(csv.DictReader(stream))[0]
    with stochastic_csv.open(newline="") as stream:
        stochastic_row = list(csv.DictReader(stream))[0]
    stochastic_payload = json.loads(stochastic_summary.read_text())
    assert deterministic_row["action_mode"] == "deterministic"
    assert deterministic_row["action_seed"] == ""
    assert deterministic_row["configured_action_seed"] == "2000"
    assert deterministic_row["effective_action_seed"] == ""
    assert stochastic_row["action_mode"] == "stochastic"
    assert stochastic_row["action_seed"] == "3456"
    assert stochastic_row["effective_action_seed"] == "3456"
    assert stochastic_payload["action_mode"] == "stochastic"
    assert stochastic_payload["action_seed"] == 3456


class MinimalTrainer:
    is_recurrent = False
    is_tam = False
    actors = RecordingActors()
    environment_config = {"environment_version": "heterogeneous_mavuav_4v4_v3_9"}
    env_steps = 123
    reward_mode = "heterogeneous_role_coupled_gate_v1"
    reward_shaping_mode = "absolute"
    shaping_gamma = 0.0
    config = {
        "actor_variant": "vanilla", "method_variant": "baseline",
        "critic_variant": "mlp", "gamma": 0.99,
        "environment_profile": "learnability",
    }


def _summary_record():
    return {
        "outcome": "draw", "episode_return": 1.0, "mav_survived": True,
        "red_uav_survivors": 3, "red_attack_kills": 0,
        "blue_attack_kills": 0, "episode_length": 2,
    }


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_training_evaluation_passes_stochastic_protocol_and_metadata(monkeypatch):
    captured = {}

    def fake_evaluator(*_args, **kwargs):
        captured.update(kwargs)
        return [_summary_record()]

    monkeypatch.setattr(training_entrypoint, "evaluate_actors", fake_evaluator)
    row = _evaluation_row(
        MinimalTrainer(), 1, "learnability", 7, "cpu", "stochastic", 2000,
    )
    assert captured["deterministic"] is False
    assert captured["seed"] == 1000
    assert captured["action_seed"] == 2000
    assert row["training_seed"] == 7
    assert row["evaluation_environment_seed_start"] == 1000
    assert row["evaluation_episodes"] == 1
    assert row["action_mode"] == "stochastic"
    assert row["configured_action_seed"] == row["effective_action_seed"] == 2000


def test_training_evaluation_default_remains_deterministic(monkeypatch):
    captured = {}

    def fake_evaluator(*_args, **kwargs):
        captured.update(kwargs)
        return [_summary_record()]

    monkeypatch.setattr(training_entrypoint, "evaluate_actors", fake_evaluator)
    row = _evaluation_row(MinimalTrainer(), 1, "learnability", 7, "cpu")
    assert captured["deterministic"] is True
    assert captured["seed"] == 1000
    assert captured["action_seed"] is None
    assert row["action_mode"] == "deterministic"
    assert row["configured_action_seed"] == 2000
    assert row["effective_action_seed"] is None
    assert row["action_seed"] is None


@pytest.mark.parametrize("raises", [False, True])
def test_training_evaluation_restores_cpu_and_numpy_rng(monkeypatch, raises):
    def consuming_evaluator(*_args, **_kwargs):
        torch.rand(17)
        np.random.random(17)
        if raises:
            raise RuntimeError("evaluation failed")
        return [_summary_record()]

    monkeypatch.setattr(training_entrypoint, "evaluate_actors", consuming_evaluator)
    torch.manual_seed(123)
    np.random.seed(456)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    if raises:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            _evaluation_row(MinimalTrainer(), 1, "learnability", 1, "cpu", "stochastic", 2000)
    else:
        _evaluation_row(MinimalTrainer(), 1, "learnability", 1, "cpu", "stochastic", 2000)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert _numpy_state_equal(numpy_before, np.random.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("raises", [False, True])
def test_training_evaluation_restores_cuda_rng(monkeypatch, raises):
    def consuming_evaluator(*_args, **_kwargs):
        torch.rand(17, device="cuda")
        torch.cuda.manual_seed_all(98765)
        if raises:
            raise RuntimeError("evaluation failed")
        return [_summary_record()]

    monkeypatch.setattr(training_entrypoint, "evaluate_actors", consuming_evaluator)
    torch.manual_seed(321)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    if raises:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            _evaluation_row(MinimalTrainer(), 1, "learnability", 1, "cuda", "stochastic", 2000)
    else:
        _evaluation_row(MinimalTrainer(), 1, "learnability", 1, "cuda", "stochastic", 2000)
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def _short_dbm_trainer():
    env = deepcopy(load_environment_config(V39))
    env["simulation"]["max_decision_steps"] = 2
    return HAPPOTrainer(env, {
        "method_variant": "dbm_rgaa", "actor_variant": "vanilla",
        "critic_variant": "mlp", "environment_profile": "learnability",
        "device": "cpu", "num_envs": 1, "rollout_steps": 2,
        "ppo_epochs": 1, "minibatch_size": 8, "hidden_dim": 16,
        "seed": 91, "role_advantage_coef": 0.5,
    })


def _assert_module_equal(left, right):
    for key, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[key]), key


def test_stochastic_mid_training_evaluation_does_not_change_next_update():
    direct = _short_dbm_trainer()
    evaluated = _short_dbm_trainer()
    try:
        before_update_one = torch.get_rng_state().clone()
        direct.train_update()
        after_update_one = torch.get_rng_state().clone()
        torch.set_rng_state(before_update_one)
        evaluated.train_update()
        assert torch.equal(after_update_one, torch.get_rng_state())

        direct_episodes, direct_metrics = direct.train_update()
        after_update_two = torch.get_rng_state().clone()
        torch.set_rng_state(after_update_one)
        _evaluation_row(
            evaluated, 1, "learnability", 91, "cpu", "stochastic", 2000,
        )
        assert torch.equal(after_update_one, torch.get_rng_state())
        evaluated_episodes, evaluated_metrics = evaluated.train_update()
        assert torch.equal(after_update_two, torch.get_rng_state())
        assert direct_episodes == evaluated_episodes
        assert direct_metrics["agent_update_order"] == evaluated_metrics["agent_update_order"]
        _assert_module_equal(direct.actors, evaluated.actors)
        _assert_module_equal(direct.critic, evaluated.critic)
        _assert_module_equal(direct.mav_role_critic, evaluated.mav_role_critic)
        _assert_module_equal(direct.uav_role_critic, evaluated.uav_role_critic)
        assert direct.rng.bit_generator.state == evaluated.rng.bit_generator.state
        assert direct.rgaa_rng.bit_generator.state == evaluated.rgaa_rng.bit_generator.state
        assert direct.env_steps == evaluated.env_steps == 4
        for direct_optimizer, evaluated_optimizer in zip(
            direct.actor_optimizers, evaluated.actor_optimizers,
        ):
            direct_state = direct_optimizer.state_dict()
            evaluated_state = evaluated_optimizer.state_dict()
            assert direct_state["param_groups"] == evaluated_state["param_groups"]
            for parameter_id, state in direct_state["state"].items():
                for field, value in state.items():
                    other = evaluated_state["state"][parameter_id][field]
                    assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    finally:
        direct.close(); evaluated.close()


def test_internal_and_direct_stochastic_evaluation_metrics_are_identical():
    trainer = _short_dbm_trainer()
    try:
        internal = _evaluation_row(
            trainer, 3, "learnability", 91, "cpu", "stochastic", 2000,
        )
        direct = evaluation.summarize_records(evaluate_actors(
            trainer.actors, trainer.environment_config, 3, "learnability",
            seed=1000, device="cpu", deterministic=False, action_seed=2000,
        ))
        for field in (
            "red_win_rate", "mean_episode_return", "mean_red_attack_kills",
            "mean_UAV_survivors", "MAV_survival_rate", "mean_episode_length",
        ):
            assert internal[field] == direct[field]
    finally:
        trainer.close()
