from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.networks import IndependentActors
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, load_environment_config
from tools.audit_role_guided_run import (
    CR_CONTINUOUS_FIELDS, EPISODE_FIELDS, analyze_training_rows, audit_episodes,
    death_summary, ensure_output_directory, exploratory_correlations,
    load_actor_only_checkpoint, pearson_or_none, validate_checkpoint_contract, validate_death_accounting,
    write_outputs, _categorize_cause,
)


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def payload(method="rgaa", **updates):
    result = {
        "environment_version": "heterogeneous_mavuav_4v4_v3_9",
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla", "critic_variant": "mlp",
        "method_variant": method, "trainer_config": {
            "method_variant": method, "actor_variant": "vanilla",
            "critic_variant": "mlp", "hidden_dim": 16, "seed": 3,
        },
    }
    result.update(updates)
    return result


@pytest.mark.parametrize("method", ["rgaa", "cr_rgaa"])
def test_checkpoint_contract_accepts_both_role_guided_methods(method):
    contract = validate_checkpoint_contract(payload(method), short_v39())
    assert contract["method_variant"] == method
    assert contract["hidden_dim"] == 16


def test_checkpoint_contract_rejects_nonvanilla_and_environment_mismatch():
    with pytest.raises(RuntimeError, match="actor_variant"):
        validate_checkpoint_contract(payload(actor_variant="tam"), short_v39())
    wrong = payload(); wrong["environment_version"] = "wrong"
    with pytest.raises(RuntimeError, match="environment contract"):
        validate_checkpoint_contract(wrong, short_v39())


def test_actor_only_loader_needs_no_team_or_role_critic_state(tmp_path):
    torch.manual_seed(11)
    source = IndependentActors(hidden_dim=16)
    checkpoint_payload = payload("cr_rgaa")
    checkpoint_payload.update({
        "environment_config": short_v39(),
        "actors": source.state_dict(),
    })
    checkpoint = tmp_path / "actor_only_contract.pt"
    torch.save(checkpoint_payload, checkpoint)
    loaded, loaded_payload, _, contract = load_actor_only_checkpoint(checkpoint, "cpu")
    assert contract["method_variant"] == "cr_rgaa"
    assert "critic" not in loaded_payload and "relational_role_critic_state" not in loaded_payload
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), loaded.parameters()))


def test_stochastic_audit_matches_formal_evaluator_episode_by_episode():
    torch.manual_seed(19)
    actors = IndependentActors(hidden_dim=16)
    config = short_v39()
    formal = evaluate_actors(
        actors, config, 3, "learnability", seed=1000, device="cpu",
        deterministic=False, action_seed=2000,
    )
    audited = audit_episodes(
        actors, config, 3, "learnability", env_seed=1000, device="cpu",
        deterministic=False, action_seed=2000,
    )
    keys = (
        "outcome", "episode_return", "red_attack_kills", "blue_attack_kills",
        "mav_survived", "red_uav_survivors", "episode_length",
    )
    assert [[row[key] for key in keys] for row in audited] == [
        [row[key] for key in keys] for row in formal
    ]
    assert [row["environment_seed"] for row in audited] == [1000, 1001, 1002]
    assert [row["action_seed"] for row in audited] == [2000, 2001, 2002]


def _episode(index, causes, *, mav=True, uavs=3, outcome="draw"):
    return {
        "episode": index, "environment_seed": 1000 + index, "action_seed": 2000 + index,
        "outcome": outcome, "episode_length": 75, "episode_return": float(index),
        "red_attack_kills": index % 4, "blue_attack_kills": 0,
        "mav_survived": mav, "red_uav_survivors": uavs,
        **{f"{aid}_death_cause": causes.get(aid, "alive") for aid in RED_IDS},
    }


def test_death_cause_agent_aggregate_alive_boundary_blue_and_other():
    records = [
        _episode(0, {}),
        _episode(1, {"UAV1": "boundary"}, uavs=2),
        _episode(2, {"MAV": "blue_attack", "UAV2": "other"}, mav=False, uavs=2, outcome="blue"),
    ]
    for record in records:
        validate_death_accounting(record)
    summary = death_summary(records)
    assert summary["by_agent"]["UAV1"] == {
        "alive": 2, "boundary": 1, "blue_attack": 0, "other": 0,
    }
    assert summary["by_agent"]["MAV"]["blue_attack"] == 1
    assert summary["uav_boundary_total"] == 1
    assert summary["uav_other_total"] == 1
    assert summary["uav_boundary_share_of_uav_deaths"] == 0.5
    assert _categorize_cause("unexpected") == "other"


def test_death_accounting_invariant_rejects_missing_cause():
    bad = _episode(0, {}, uavs=2)
    with pytest.raises(AssertionError, match="death accounting"):
        validate_death_accounting(bad)


def _training_row(step, completed, *, metric=None, mav_boundary=0, uav_boundary=0, blue=0):
    row = {"sampled_steps": step, "completed_episodes": completed}
    if metric is not None:
        row.update({field: metric for field in CR_CONTINUOUS_FIELDS})
    for aid in RED_IDS:
        boundary = mav_boundary if aid == "MAV" else uav_boundary
        row[f"own_boundary_loss_count_{aid}"] = boundary
        row[f"own_blue_attack_loss_count_{aid}"] = blue
        row[f"own_loss_count_{aid}"] = boundary + blue
    return row


def test_phase_boundaries_episode_deltas_event_sums_and_normalization():
    rows = [
        _training_row(750_000, 10, metric=1.0, mav_boundary=1, uav_boundary=2),
        _training_row(750_001, 15, metric=3.0, mav_boundary=2, uav_boundary=1, blue=1),
        _training_row(1_250_000, 20, metric=5.0, mav_boundary=0, uav_boundary=1),
        _training_row(1_250_001, 22, metric=7.0),
        _training_row(1_600_000, 25, metric=9.0),
        _training_row(1_600_001, 30, metric=11.0),
        _training_row(2_000_000, 35, metric=13.0),
    ]
    _, phases = analyze_training_rows(rows, "cr_rgaa")
    first = phases["phase_0_750k"]
    second = phases["phase_750k_1250k"]
    assert first["updates"] == 1 and first["episodes"] == 10
    assert second["updates"] == 2 and second["episodes"] == 10
    assert second["mechanism_metrics"]["cr_lambda_mean"]["mean"] == 4.0
    uav = second["death_events"]["UAV_aggregate"]
    assert uav["boundary_events"] == 6.0
    assert uav["blue_attack_events"] == 3.0
    assert uav["boundary_events_per_1000_episodes"] == 600.0
    assert uav["other_or_mismatch_events"] == 0.0
    assert phases["phase_1250k_1600k"]["updates"] == 2
    assert phases["phase_1600k_2000k"]["updates"] == 2


def test_rgaa_missing_cr_fields_is_safe_and_unavailable():
    _, phases = analyze_training_rows([_training_row(500_000, 4)], "rgaa")
    assert phases["phase_0_750k"]["mechanism_metrics"]["cr_lambda_mean"] is None
    assert phases["phase_0_750k"]["death_events"]["MAV"]["boundary_events"] == 0.0


def test_exploratory_correlation_requires_variation_and_twenty_samples():
    assert pearson_or_none([1.0] * 20, list(range(20))) is None
    assert pearson_or_none(list(range(19)), list(range(19))) is None
    rows = [_training_row(800_000 + i, i + 1, metric=1.0, uav_boundary=i % 2) for i in range(20)]
    assert all(value is None for value in exploratory_correlations(rows, "cr_rgaa").values())
    assert all(value is None for value in exploratory_correlations(rows, "rgaa").values())


def test_output_directory_nonempty_rejected_and_json_contains_no_nan(tmp_path):
    occupied = tmp_path / "occupied"; occupied.mkdir(); (occupied / "x").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        ensure_output_directory(occupied)
    output = tmp_path / "audit"
    episode = _episode(0, {})
    phase = {"phase": "p", "value": 1.0}
    write_outputs(output, [episode], [phase], {"finite": 1.0, "not_finite": float("nan")})
    parsed = json.loads((output / "audit_summary.json").read_text())
    assert parsed == {"finite": 1.0, "not_finite": None}
    assert set((output / "death_audit_episodes.csv").read_text().splitlines()[0].split(",")) == set(EPISODE_FIELDS)
