from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_policy_trajectories import checkpoint_sha256, validate_run_matrix
from tools import postprocess_policy_trajectory_audit as post


def matrix_records():
    environment = {"environment_version": "heterogeneous_mavuav_4v4_v3_9", "x": [1, 2]}
    return [
        {"method_variant": method, "training_seed": seed, "sampled_steps": 2_000_000,
         "training_profile": "learnability", "evaluation_profile": "learnability",
         "environment_version": "heterogeneous_mavuav_4v4_v3_9", "environment_config": environment}
        for method in post.METHODS for seed in (7, 9, 11)
    ]


def test_exact_three_by_three_matrix_passes():
    validate_run_matrix(matrix_records())


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "incomplete"),
    (lambda rows: rows.append(dict(rows[0])), "duplicate"),
    (lambda rows: rows[0].update(training_seed=8), "unexpected"),
    (lambda rows: rows[0].update(method_variant="baseline"), "unexpected"),
    (lambda rows: rows[0].update(sampled_steps=1_999_999), "exact-2M"),
    (lambda rows: rows[0].update(environment_config={"different": True}), "identical resolved"),
])
def test_invalid_matrix_contract_is_rejected(mutation, match):
    rows = matrix_records(); mutation(rows)
    with pytest.raises(RuntimeError, match=match):
        validate_run_matrix(rows)


def test_checkpoint_sha_is_stable(tmp_path):
    path = tmp_path / "checkpoint.pt"; path.write_bytes(b"read-only checkpoint bytes")
    before = checkpoint_sha256(path); after = checkpoint_sha256(path)
    assert before == after == hashlib.sha256(path.read_bytes()).hexdigest()


def paired_rows():
    rows = []
    lengths = {
        7: {424242: (20, 30, 40), 424243: (30, 30, 30)},
        9: {424242: (40, 30, 20), 424243: (30, 30, 30)},
    }
    for seed, environments in lengths.items():
        for env_seed, values in environments.items():
            for method, length in zip(post.METHODS, values):
                rows.append({"method_variant": method, "training_seed": seed,
                             "environment_seed": env_seed, "action_mode": "deterministic",
                             "outcome": "red", "episode_length": length, "run": f"{method}_{seed}",
                             "episode_dir": f"/{method}/{seed}/{env_seed}", "red_attack_kills": 4,
                             "blue_attack_kills": 0, "mav_survived": 1, "uav_survivors": 3})
    return rows


def test_paired_selection_requires_same_training_and_environment_seed():
    rows = paired_rows()
    rows = [row for row in rows if not (row["method_variant"] == "dbm_rgaa" and row["training_seed"] == 7)]
    choice = post.select_paired_representative(rows)
    chosen = choice["methods"]
    assert {row["run"].rsplit("_", 1)[1] for row in chosen.values()} == {str(choice["training_seed"])}
    assert len({Path(row["episode_dir"]).name for row in chosen.values()}) == 1


def test_paired_selection_requires_all_red_wins():
    rows = paired_rows()
    for row in rows:
        if row["method_variant"] == "rgaa_wide" and row["training_seed"] == 7:
            row["outcome"] = "draw"
    choice = post.select_paired_representative(rows)
    assert choice["training_seed"] == 9


def test_paired_score_and_tie_break_are_deterministic():
    choice = post.select_paired_representative(paired_rows())
    assert choice["score"] == 0
    assert choice["training_seed"] == 7
    assert choice["environment_seed"] == 424243


def metadata(events):
    return {"events": events}


def attack(frame, attacker, target):
    return {"trace_frame": frame, "time_s": float(frame), "type": "attack", "attacker": attacker, "target": target}


def death(frame, target):
    return {"trace_frame": frame, "time_s": float(frame), "type": "death", "entity": target, "cause": "red_attack"}


def test_multi_target_terminal_requires_two_terminal_targets():
    events = [attack(5, "UAV1", "Blue1"), attack(5, "UAV1", "Blue2"), death(5, "Blue1")]
    assert post.multi_target_terminal_events(metadata(events)) == []


def test_two_attackers_one_target_is_not_multi_target():
    events = [attack(5, "UAV1", "Blue1"), attack(5, "UAV2", "Blue1"), death(5, "Blue1")]
    assert post.multi_target_terminal_events(metadata(events)) == []


def test_one_attacker_two_terminal_targets_is_counted():
    events = [attack(5, "UAV3", "Blue1"), attack(5, "UAV3", "Blue2"), death(5, "Blue1"), death(5, "Blue2")]
    result = post.multi_target_terminal_events(metadata(events))
    assert len(result) == 1 and result[0]["distinct_targets"] == 2
    assert result[0]["targets"] == "Blue1|Blue2" and result[0]["same_frame_red_attack_deaths"] == 2


def test_multi_target_death_share_uses_all_red_attack_deaths_denominator(tmp_path):
    episode = tmp_path / "episode"; episode.mkdir()
    events = [attack(5, "UAV1", "Blue1"), attack(5, "UAV1", "Blue2"), death(5, "Blue1"), death(5, "Blue2"),
              attack(7, "UAV2", "Blue3"), death(7, "Blue3")]
    (episode / "metadata.json").write_text(json.dumps(metadata(events)), encoding="utf-8")
    rows = []
    for mode in ("deterministic", "stochastic"):
        for method in post.METHODS:
            rows.append({"episode_dir": str(episode), "method_variant": method, "training_seed": 7,
                         "action_mode": mode, "environment_seed": 1, "action_seed": 2})
    summary, _ = post.summarize_multi_target(rows)
    assert all(row["multi_target_frame_blue_deaths"] == 2 for row in summary)
    assert all(row["all_red_attack_blue_deaths"] == 3 for row in summary)
    assert all(row["blue_death_share"] == pytest.approx(2 / 3) for row in summary)


def test_postprocess_does_not_construct_environment_or_modify_trace(tmp_path, monkeypatch):
    audit = tmp_path / "audit"; audit.mkdir()
    rows = []
    for method in post.METHODS:
        episode = audit / method; episode.mkdir()
        (episode / "metadata.json").write_text(json.dumps({"events": []}), encoding="utf-8")
        (episode / "episode_trace.npz").write_bytes(b"immutable trace")
        rows.append({"run": method, "method_variant": method, "training_seed": 7,
                     "action_mode": "deterministic", "environment_seed": 424242,
                     "action_seed": "", "episode_dir": str(episode), "outcome": "red",
                     "episode_length": 30, "red_attack_kills": 4, "blue_attack_kills": 0,
                     "mav_survived": 1, "uav_survivors": 3})
    with (audit / "episode_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    checkpoint = audit / "checkpoint.pt"; checkpoint.write_bytes(b"x")
    before = {method: hashlib.sha256((audit / method / "episode_trace.npz").read_bytes()).hexdigest() for method in post.METHODS}
    monkeypatch.setattr(post, "load_checkpoint_contracts", lambda _: (matrix_records(), [{
        "run": "x", "method": "rgaa", "training_seed": 7,
        "checkpoint_path": str(checkpoint), "sha256_before": checkpoint_sha256(checkpoint)}]))
    monkeypatch.setattr(post, "build_paired_comparison", lambda paired: {"initial_states_identical": True, "methods": {}})
    monkeypatch.setattr(post, "_paired_text", lambda paired, comparison: "paired")
    post.postprocess(audit)
    after = {method: hashlib.sha256((audit / method / "episode_trace.npz").read_bytes()).hexdigest() for method in post.METHODS}
    assert before == after

