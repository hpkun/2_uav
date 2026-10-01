"""Focused tests for the read-only DBM boundary/mode audit."""
from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithm.happo.dbm_rgaa import DBMGaussianActor, build_method_actors
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import OBS_DIM, RED_IDS, load_environment_config
from tools.audit_dbm_boundary_modes import (
    BOUNDARY_AXES, LoadedAuditRun, _details_row, assert_read_only,
    boundary_axes, categorize_death, ensure_empty_output, execution_summaries, file_sha256,
    episode_output_rows, load_audit_run, pre_boundary_summaries, rollout_episode, run_audit,
    summarize_identity_deaths,
    validate_run_contracts, verify_identity_against_evaluator,
)
from tools.audit_uav_slot_permutation import SOURCE_SLOTS, UAV_IDS, slot_permutations


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "configs" / "env_v39.yaml"
IDENTITY = dict(zip(UAV_IDS, SOURCE_SLOTS))


def tiny_config(steps: int = 2) -> dict:
    config = deepcopy(load_environment_config(V39))
    config["simulation"]["max_decision_steps"] = steps
    return config


def make_run(tmp_path: Path, method: str, *, label: str | None = None, device: str = "cpu") -> Path:
    run = tmp_path / (label or method)
    run.mkdir()
    config = {
        "num_envs": 1, "rollout_steps": 1, "hidden_dim": 8,
        "environment_profile": "learnability", "device": device,
        "method_variant": method, "seed": 13,
    }
    if method == "rgaa_wide":
        config["uav_actor_hidden_dim"] = 11
    trainer = HAPPOTrainer(tiny_config(1), config)
    try:
        trainer.save_checkpoint(run / "checkpoint_final.pt")
    finally:
        trainer.close()
    return run


@pytest.mark.parametrize("method", ["rgaa", "rgaa_wide", "dbm_rgaa"])
def test_supported_checkpoint_loads_strictly(tmp_path, method):
    run_dir = make_run(tmp_path, method)
    loaded = load_audit_run(method, run_dir, "cpu")
    assert loaded.method == method
    assert loaded.sampled_steps == 0
    assert len(loaded.actors.actors) == 4
    if method == "dbm_rgaa":
        assert all(isinstance(loaded.actors.actors[i], DBMGaussianActor) for i in (1, 2, 3))
    else:
        assert all(not isinstance(actor, DBMGaussianActor) for actor in loaded.actors.actors)
    assert_read_only(loaded)


def test_wrong_method_metadata_is_rejected(tmp_path):
    run_dir = make_run(tmp_path, "dbm_rgaa")
    path = run_dir / "checkpoint_final.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["dbm_rgaa_version"] = 999
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="dbm_rgaa_version"):
        load_audit_run("bad", run_dir, "cpu")


def test_observation_or_action_contract_is_rejected(tmp_path):
    run_dir = make_run(tmp_path, "rgaa")
    path = run_dir / "checkpoint_final.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["observation_dim"] = OBS_DIM + 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="observation_dim"):
        load_audit_run("bad", run_dir, "cpu")


def test_cross_run_environment_contract_is_exact(tmp_path):
    first = load_audit_run("a", make_run(tmp_path, "rgaa", label="a"), "cpu")
    second = load_audit_run("b", make_run(tmp_path, "rgaa", label="b"), "cpu")
    validate_run_contracts([first, second])
    second.env_config["simulation"]["max_decision_steps"] += 1
    with pytest.raises(RuntimeError, match="identical resolved environment configs"):
        validate_run_contracts([first, second])


def test_six_permutations_cover_every_actor_slot_twice():
    permutations = slot_permutations()
    assert len(permutations) == 6
    assert all(sum(mapping[aid] == slot for mapping in permutations) == 2
               for aid in UAV_IDS for slot in SOURCE_SLOTS)


@pytest.mark.parametrize("raw,expected", [
    (None, "alive"), ("boundary", "boundary"),
    ("blue_attack", "blue_attack"), ("red_attack", "other"),
])
def test_death_cause_classification(raw, expected):
    assert categorize_death(raw) == expected


@pytest.mark.parametrize(
    "state,expected",
    [
        (SimpleNamespace(x=-2, y=5, h=25), {"x_lower", "y_upper", "altitude_upper"}),
        (SimpleNamespace(x=12, y=-3, h=0), {"x_upper", "y_lower", "altitude_lower"}),
        (SimpleNamespace(x=5, y=2, h=10), set()),
    ],
)
def test_boundary_axes_reports_all_simultaneous_crossings(state, expected):
    battlefield = {"x": [0, 10], "y": [0, 4], "altitude": [1, 20]}
    assert set(boundary_axes(state, battlefield)) == expected


def test_mode_detail_formulas_are_exact_and_probabilities_sum_to_one():
    actors = build_method_actors(method_variant="dbm_rgaa", training_seed=3, hidden_dim=8)
    actor = actors.actors[1]
    observation = torch.linspace(-1, 1, OBS_DIM).unsqueeze(0)
    torch.manual_seed(8)
    with torch.no_grad():
        details = actor.mode_diagnostics(observation)
        before = torch.get_rng_state().clone()
        action, _ = actor.sample(observation)
        after = torch.get_rng_state().clone()
    row = _details_row(details, action)
    assert row["router_p1"] + row["router_p2"] == pytest.approx(1.0)
    experts = details["expert_outputs"][0]
    assert row["expert_divergence"] == pytest.approx(torch.linalg.vector_norm(experts[0] - experts[1]).item())
    expected = details["base_mean"] + details["scaled_residual"]
    np.testing.assert_allclose(
        [row[f"final_mean_{axis}"] for axis in "xyz"], expected[0].numpy(), rtol=0, atol=1e-7,
    )
    assert not torch.equal(before, after)  # the single policy sample consumes RNG


def test_mode_diagnostics_alone_does_not_consume_rng_or_change_sample():
    actors = build_method_actors(method_variant="dbm_rgaa", training_seed=4, hidden_dim=8)
    actor = actors.actors[1]; observation = torch.zeros((1, OBS_DIM))
    torch.manual_seed(991); state = torch.get_rng_state().clone()
    with torch.no_grad():
        expected, _ = actor.sample(observation)
    torch.set_rng_state(state)
    with torch.no_grad():
        actor.mode_diagnostics(observation)
        actual, _ = actor.sample(observation)
    assert torch.equal(actual, expected)


def test_rollout_diagnostics_switch_preserves_actions_and_summary(tmp_path):
    loaded = load_audit_run("dbm", make_run(tmp_path, "dbm_rgaa"), "cpu")
    kwargs = dict(episode=0, env_seed=1000, action_seed=2000,
                  profile="learnability", device="cpu", capture_actions=True)
    on = rollout_episode(loaded, IDENTITY, diagnostics=True, **kwargs)
    off = rollout_episode(loaded, IDENTITY, diagnostics=False, **kwargs)
    assert on[0] == off[0]
    assert on[3] == off[3]
    assert len(on[2]) == 3  # max step one, three active UAVs
    assert off[2] == []


def test_identity_matches_formal_evaluator(tmp_path):
    loaded = load_audit_run("dbm", make_run(tmp_path, "dbm_rgaa"), "cpu")
    stats, records, rows, team_rows, agent_rows = verify_identity_against_evaluator(
        loaded, 2, "learnability", 1000, 2000, "cpu",
    )
    assert stats["completed_episodes"] == len(records) == 2
    assert len(rows) == 6
    assert len(team_rows) == 2 and len(agent_rows) == 6
    assert all({row["actor_id"] for row in agent_rows if row["episode"] == episode} == set(UAV_IDS)
               for episode in range(2))
    assert all(np.isfinite(value) for value in stats.values())


def test_identity_agent_rows_preserve_exact_death_cause_axes_and_summaries():
    run = SimpleNamespace(label="dbm_s5", method="dbm_rgaa", training_seed=5, sampled_steps=2_000_000)
    summary = {
        "episode_return": 12.0, "red_attack_kills": 3, "blue_attack_kills": 1,
        "episode_length": 42, "mav_survived": True, "red_uav_survivors": 1,
        "outcome": "red",
    }
    death = {
        "UAV1": {"cause": "alive", "step": None, "axes": []},
        "UAV2": {"cause": "boundary", "step": 23, "axes": ["altitude_lower", "x_upper"]},
        "UAV3": {"cause": "blue_attack", "step": 31, "axes": []},
    }
    team, agents = episode_output_rows(run, IDENTITY, 0, 1000, 2000, summary, death)
    assert len(agents) == 3 and {row["actor_id"] for row in agents} == set(UAV_IDS)
    by_id = {row["actor_id"]: row for row in agents}
    assert by_id["UAV1"]["final_death_cause"] == "alive" and by_id["UAV1"]["alive"] == 1
    assert by_id["UAV2"]["boundary_crossing_axes"] == "altitude_lower|x_upper"
    assert by_id["UAV2"]["altitude_lower"] == 1 and by_id["UAV2"]["x_upper"] == 1
    assert by_id["UAV3"]["blue_attack"] == 1 and by_id["UAV3"]["boundary"] == 0
    actor_summary, run_summary = summarize_identity_deaths([team], agents)
    assert len(actor_summary) == 3
    aggregate = run_summary[0]
    assert aggregate["uav_alive_count"] + aggregate["uav_boundary_count"] + aggregate["uav_blue_attack_count"] + aggregate["uav_other_death_count"] == 3
    assert aggregate["altitude_lower_count"] == 1
    assert aggregate["boundary_share_of_uav_losses"] == pytest.approx(0.5)
    assert aggregate["altitude_lower_share_of_uav_losses"] == pytest.approx(0.5)


def test_switch_pairs_never_cross_episodes_and_dead_rows_are_absent():
    base = dict(run="r", agent_id="UAV1", router_p1=0.2, router_p2=0.8,
                router_entropy=0.5, expert_divergence=1.0, residual_norm=0.1,
                deterministic_action_delta_norm=0.1, nearest_boundary_margin=2.0,
                speed=200.0, heading=0.0, pitch=0.0)
    rows = [
        {**base, "episode": 0, "decision_step": 0, "hard_mode_proxy": 2, "agent_active": 1},
        {**base, "episode": 1, "decision_step": 0, "hard_mode_proxy": 1, "agent_active": 1},
    ]
    summary = execution_summaries(rows)[0]
    assert summary["valid_temporal_pairs"] == 0
    assert summary["hard_mode_proxy_switch_rate"] == 0.0
    assert all(row["agent_active"] == 1 for row in rows)


def _trajectory_row(step: int, cause: str) -> dict:
    row = {
        "run": "r", "episode": 0, "agent_id": "UAV1", "decision_step": step,
        "final_death_cause": cause, "router_p1": step / 10, "router_entropy": 0.5,
        "residual_norm": step / 100, "deterministic_action_delta_norm": 0.1,
        "nearest_boundary_margin": 100 - step, "speed": 200 + step,
        "heading": step / 20, "pitch": -step / 30,
        "actual_sampled_action_x": 0.1, "actual_sampled_action_y": 0.2,
        "actual_sampled_action_z": 0.3,
    }
    return row


def test_pre_boundary_windows_use_available_steps_and_keep_causes_separate():
    rows = [_trajectory_row(step, "boundary") for step in range(3)]
    summaries = pre_boundary_summaries(rows)
    assert {row["requested_window"] for row in summaries} == {5, 10}
    assert {row["available_step_count"] for row in summaries} == {3}
    assert {row["trajectory_count"] for row in summaries} == {1}
    assert {row["final_death_cause"] for row in summaries} == {"boundary"}


def test_output_directory_must_be_missing_or_empty(tmp_path):
    missing = tmp_path / "new"
    assert ensure_empty_output(missing) == missing.resolve()
    (missing / "sentinel").write_text("x")
    with pytest.raises(FileExistsError, match="missing or empty"):
        ensure_empty_output(missing)


def audit_args(run: Path, output: Path, device: str = "cpu") -> SimpleNamespace:
    return SimpleNamespace(
        run=[("dbm", run)], profile="learnability", device=device,
        env_seed=1000, action_seed=2000, permutation_episodes=1,
        identity_episodes=1, output=output,
    )


def test_tiny_end_to_end_writes_contract_and_is_read_only(tmp_path):
    run = make_run(tmp_path, "dbm_rgaa")
    checkpoint = run / "checkpoint_final.pt"; digest = file_sha256(checkpoint)
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)["actors"]
    torch.manual_seed(1234); rng = torch.get_rng_state().clone()
    output = tmp_path / "audit"
    summary = run_audit(audit_args(run, output))
    assert summary["diagnostics_action_and_result_invariant"] is True
    assert torch.equal(torch.get_rng_state(), rng)
    assert file_sha256(checkpoint) == digest
    after = torch.load(checkpoint, map_location="cpu", weights_only=False)["actors"]
    assert all(torch.equal(before[key], after[key]) for key in before)
    expected = {
        "audit_metadata.json", "permutation_episode_team.csv",
        "permutation_episode_agent.csv", "permutation_summary.csv",
        "actor_slot_boundary_matrix.csv", "actor_boundary_marginals.csv",
        "slot_boundary_marginals.csv", "execution_router_steps.csv",
        "execution_router_summary.csv", "pre_boundary_window_summary.csv",
        "identity_episode_team.csv", "identity_episode_agent.csv",
        "identity_death_summary.csv", "identity_run_death_summary.csv",
        "audit_summary.json", "README.txt",
    }
    assert {item.name for item in output.iterdir()} == expected
    with (output / "execution_router_steps.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(float(row["router_p1"]) + float(row["router_p2"]) == pytest.approx(1.0) for row in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiny_end_to_end(tmp_path):
    run = make_run(tmp_path, "dbm_rgaa")
    torch.cuda.manual_seed_all(4433)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    summary = run_audit(audit_args(run, tmp_path / "cuda_audit", device="cuda"))
    after = torch.cuda.get_rng_state_all()
    assert summary["identity_evaluator_consistency"]["dbm"]["completed_episodes"] == 1
    assert all(torch.equal(left, right) for left, right in zip(before, after))
