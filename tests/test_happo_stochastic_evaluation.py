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
from algorithm.evaluate_happo import parse_args
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.trainer import HAPPOTrainer
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
    assert deterministic_row["action_seed"] == "2000"
    assert stochastic_row["action_mode"] == "stochastic"
    assert stochastic_row["action_seed"] == "3456"
    assert stochastic_payload["action_mode"] == "stochastic"
    assert stochastic_payload["action_seed"] == 3456
