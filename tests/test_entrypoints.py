from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, timeout=180,
    )


def test_direct_happo_entrypoints_show_help_without_package_install():
    assert "--steps" in _run("algorithm/train_happo.py", "--help").stdout
    assert "--blue-mode" in _run("algorithm/evaluate_happo.py", "--help").stdout


def test_flat_training_and_cross_profile_evaluation_smoke():
    output_name = f"pytest_happo_smoke_{uuid.uuid4().hex}"
    run_dir = PROJECT_ROOT / "outputs" / output_name
    try:
        _run(
            "algorithm/train_happo.py", "--steps", "2", "--profile", "learnability",
            "--device", "cpu", "--num-envs", "1", "--output-name", output_name,
            "--checkpoint-interval", "2", "--eval-interval", "0",
            "--final-eval-episodes", "1",
        )
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
        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            assert int(list(csv.DictReader(stream))[-1]["sampled_steps"]) == 2

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
