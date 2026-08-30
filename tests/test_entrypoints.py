from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import yaml

from algorithm.train_happo import MilestoneObserver, planned_rollout_horizon


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, timeout=180,
    )


def test_direct_happo_entrypoints_show_help_without_package_install():
    assert "--steps" in _run("algorithm/train_happo.py", "--help").stdout
    assert "--blue-mode" in _run("algorithm/evaluate_happo.py", "--help").stdout
    assert "--steps" in _run("algorithm/train_happo_hrta.py", "--help").stdout
    assert "--attention-output" in _run("algorithm/evaluate_happo_hrta.py", "--help").stdout


def _simulated_schedule(total: int, interval: int) -> tuple[list[int], list[int]]:
    current = 0
    updates = []
    observed = []
    observer = MilestoneObserver(interval)
    while current < total:
        current += planned_rollout_horizon(current, total, configured_horizon=8, num_envs=1)
        updates.append(current)
        if observer.consume(current):
            observed.append(current)
    return updates, observed


def test_checkpoint_milestone_does_not_truncate_rollouts():
    assert _simulated_schedule(20, 10) == ([8, 16, 20], [16, 20])


def test_evaluation_and_log_milestones_do_not_truncate_rollouts():
    assert _simulated_schedule(20, 6)[0] == [8, 16, 20]
    assert _simulated_schedule(20, 7)[0] == [8, 16, 20]


def test_checkpoint_frequency_does_not_change_update_horizons():
    assert _simulated_schedule(20, 3)[0] == _simulated_schedule(20, 100)[0] == [8, 16, 20]


def test_flat_training_and_cross_profile_evaluation_smoke():
    output_name = f"pytest_happo_smoke_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        training = _run(
            "algorithm/train_happo.py", "--steps", "2", "--profile", "learnability",
            "--device", "cpu", "--num-envs", "1", "--output-name", output_name,
            "--checkpoint-interval", "2", "--eval-interval", "0",
            "--log-interval", "1", "--final-eval-episodes", "1",
        )
        for text in ("[TRAIN]", "steps", "return", "W/B/D", "Red kills", "MAV survival", "critic", "entropy"):
            assert text in training.stdout
        assert run_dir.is_dir()
        assert not (run_dir / "happo_seed1").exists()
        assert not (run_dir / "checkpoints").exists()
        assert not any(path.is_dir() for path in run_dir.iterdir())
        expected = {
            "run.log", "resolved_config.yaml", "training.csv", "evaluations.csv", "summary.json",
            "checkpoint_2.pt", "checkpoint_final.pt",
        }
        assert expected <= {path.name for path in run_dir.iterdir()}
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["sampled_steps"] == 2 and summary["status"] == "complete"
        assert summary["algorithm"] == "happo" and summary["actor_variant"] == "vanilla"
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            assert int(list(csv.DictReader(stream))[-1]["sampled_steps"]) == 2
        with (run_dir / "evaluations.csv").open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 2
        assert training.stdout == (run_dir / "run.log").read_text(encoding="utf-8")

        _run(
            "algorithm/train_happo.py", "--steps", "4", "--profile", "learnability",
            "--seed", "1", "--device", "cpu", "--num-envs", "1",
            "--checkpoint-interval", "2", "--eval-interval", "0", "--log-interval", "1",
            "--final-eval-episodes", "1", "--resume", str(run_dir / "checkpoint_final.pt"),
        )
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        assert resolved["total_steps"] == 2
        assert resolved["resume_history"][-1]["resumed_from_steps"] == 2
        assert resolved["resume_history"][-1]["target_steps"] == 4
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            assert [int(row["sampled_steps"]) for row in csv.DictReader(stream)] == [2, 4]

        _run(
            "algorithm/evaluate_happo.py", str(run_dir / "checkpoint_final.pt"),
            "--profile", "main", "--episodes", "1", "--device", "cpu",
            "--blue-mode", "nearest",
        )
        evaluation = json.loads((run_dir / "evaluation_final_summary.json").read_text(encoding="utf-8"))
        assert evaluation["training_profile"] == "learnability"
        assert evaluation["evaluation_profile"] == "main"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
