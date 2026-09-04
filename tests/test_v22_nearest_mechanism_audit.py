from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.happo.networks import IndependentActors
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config
from tools.audit_v22_nearest_mechanism import (
    EARLY_FILENAME,
    EPISODE_FILENAME,
    METADATA_FILENAME,
    SUMMARY_FILENAME,
    classify_failure,
    closure_rate,
    gate_flags,
    record_first,
    run_audit,
    select_checkpoint,
    validate_checkpoint_contract,
)


def _payload(seed: int = 7) -> dict:
    actors = IndependentActors(hidden_dim=8)
    return {
        "format": "happo_training_checkpoint_v1",
        "sampled_steps": 2_000_000,
        "environment_version": ENVIRONMENT_VERSION,
        "environment_profile": "main",
        "observation_dim": OBS_DIM,
        "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla",
        "method_variant": "baseline",
        "environment_config": load_environment_config(None),
        "trainer_config": {
            "seed": seed,
            "hidden_dim": 8,
            "actor_variant": "vanilla",
            "method_variant": "baseline",
            "environment_profile": "main",
        },
        "actors": actors.state_dict(),
    }


def _checkpoint(tmp_path: Path, *, final_only: bool = False) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    name = "checkpoint_final.pt" if final_only else "checkpoint_2000000.pt"
    torch.save(_payload(), run_dir / name)
    return run_dir


def test_v22_checkpoint_contract_and_checkpoint_preference(tmp_path):
    payload = _payload()
    validated = validate_checkpoint_contract(payload)
    assert validated == {
        "environment_version": "heterogeneous_mavuav_3v2_v2_2",
        "observation_dim": 61,
        "global_state_dim": 67,
        "actor_variant": "vanilla",
        "method_variant": "baseline",
    }
    for field, bad in (
        ("environment_version", "heterogeneous_mavuav_3v2_v2_1"),
        ("observation_dim", 55),
        ("global_state_dim", 68),
        ("actor_variant", "hrta"),
        ("method_variant", "agp"),
    ):
        changed = _payload()
        changed[field] = bad
        with pytest.raises(RuntimeError, match="incompatible v2.2"):
            validate_checkpoint_contract(changed)

    run_dir = _checkpoint(tmp_path)
    torch.save(_payload(), run_dir / "checkpoint_final.pt")
    assert select_checkpoint(run_dir).name == "checkpoint_2000000.pt"


def test_true_attack_gate_boundaries_are_inclusive_distance_and_strict_angles():
    assert gate_flags(1000.0, math.radians(0), math.radians(0)) == (True, True, True)
    assert gate_flags(3000.0, math.radians(0), math.radians(0)) == (True, True, True)
    assert gate_flags(999.999, math.radians(0), math.radians(0)) == (False, False, False)
    assert gate_flags(3000.001, math.radians(0), math.radians(0)) == (False, False, False)
    assert gate_flags(2000.0, math.radians(30), math.radians(0)) == (True, False, False)
    assert gate_flags(2000.0, math.radians(29.999), math.radians(90)) == (True, True, False)


def test_closure_rate_sign_uses_negative_distance_derivative():
    relative_position = np.asarray([1000.0, 0.0, 0.0])
    assert closure_rate(relative_position, np.asarray([-200.0, 0.0, 0.0])) == pytest.approx(200.0)
    assert closure_rate(relative_position, np.asarray([200.0, 0.0, 0.0])) == pytest.approx(-200.0)
    assert closure_rate(np.zeros(3), np.ones(3)) == 0.0


def test_first_event_step_records_once_and_keeps_missing_as_nan():
    first = {"distance": float("nan"), "kill": float("nan")}
    record_first(first, "distance", False, 2)
    assert math.isnan(first["distance"])
    record_first(first, "distance", True, 3)
    record_first(first, "distance", True, 7)
    assert first["distance"] == 3.0
    assert math.isnan(first["kill"])


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"outcome": "red", "distance_gate": 1, "ata_gate": 1, "full_geometry": 1, "streak2": 1, "first_kill": 1}, "WIN"),
        ({"outcome": "draw", "distance_gate": 0, "ata_gate": 0, "full_geometry": 0, "streak2": 0, "first_kill": 0}, "F1_distance"),
        ({"outcome": "draw", "distance_gate": 1, "ata_gate": 0, "full_geometry": 0, "streak2": 0, "first_kill": 0}, "F2_ATA"),
        ({"outcome": "draw", "distance_gate": 1, "ata_gate": 1, "full_geometry": 0, "streak2": 0, "first_kill": 0}, "F3_AA"),
        ({"outcome": "draw", "distance_gate": 1, "ata_gate": 1, "full_geometry": 1, "streak2": 0, "first_kill": 0}, "F4_hold1"),
        ({"outcome": "draw", "distance_gate": 1, "ata_gate": 1, "full_geometry": 1, "streak2": 1, "first_kill": 0}, "F5_kill"),
        ({"outcome": "draw", "distance_gate": 1, "ata_gate": 1, "full_geometry": 1, "streak2": 1, "first_kill": 1}, "F6_second_kill"),
    ],
)
def test_failure_taxonomy_is_mutually_exclusive(row, expected):
    assert classify_failure(row) == expected


def _assert_one_episode_smoke(tmp_path: Path, device: str) -> None:
    run_dir = _checkpoint(tmp_path, final_only=True)
    output_dir = tmp_path / "audit"
    paths = run_audit([run_dir], episodes=1, device_name=device, output_dir=output_dir)
    assert {path.name for path in paths.values()} == {
        EPISODE_FILENAME, EARLY_FILENAME, SUMMARY_FILENAME, METADATA_FILENAME,
    }
    with paths["episodes"].open(encoding="utf-8", newline="") as stream:
        episode_rows = list(csv.DictReader(stream))
    with paths["early20"].open(encoding="utf-8", newline="") as stream:
        early_rows = list(csv.DictReader(stream))
    with paths["summary"].open(encoding="utf-8", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert len(episode_rows) == len(summary_rows) == 1
    assert episode_rows[0]["evaluation_seed"] == "1000"
    assert 6 <= len(early_rows) <= 120
    assert {row["red_id"] for row in early_rows} == {"MAV", "UAV1", "UAV2"}
    assert {row["blue_id"] for row in early_rows} == {"Blue1", "Blue2"}
    assert max(int(row["step"]) for row in early_rows) <= 19
    assert metadata["profile"] == "main"
    assert metadata["blue_mode"] == "nearest"
    assert metadata["deterministic"] is True
    assert metadata["compatibility_note"] == "v2.2 current code only; no legacy v2.1 checkpoint compatibility"


def test_one_episode_cpu_focused_smoke(tmp_path):
    _assert_one_episode_smoke(tmp_path, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_one_episode_cuda_focused_smoke(tmp_path):
    _assert_one_episode_smoke(tmp_path, "cuda:0")
