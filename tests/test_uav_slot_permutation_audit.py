"""Focused tests for the read-only realized UAV slot permutation audit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.networks import IndependentActors
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import BLUE_IDS, OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from tools.audit_same_role_policy_divergence import file_sha256
from tools.audit_uav_slot_permutation import (
    SOURCE_SLOTS, UAV_IDS, apply_realized_slot_permutation, build_actor_slot_matrix,
    build_marginals, is_identity_permutation, permutation_label, reset_with_realized_permutation,
    rollout_permuted_episode, run_audit, slot_permutations,
    snapshot_realized_uav_slots, validate_cross_run_environment_contract,
)


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "configs" / "env_v39.yaml"
IDENTITY = dict(zip(UAV_IDS, SOURCE_SLOTS))


def tiny_env_config(max_steps: int = 2) -> dict:
    config = deepcopy(load_environment_config(V39))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def state_tuple(state) -> tuple:
    return tuple(getattr(state, field.name) for field in fields(state))


def test_generates_exactly_six_permutations_with_stable_identity_label():
    generated = slot_permutations()
    assert len(generated) == 6
    assert len({permutation_label(item) for item in generated}) == 6
    assert generated[0] == IDENTITY
    assert is_identity_permutation(generated[0])
    assert permutation_label(generated[0]) == "U1_S1__U2_S2__U3_S3"


def test_snapshot_copies_every_aircraft_state_field_without_aliasing():
    env = HeterogeneousMAVUAVAirCombatEnv(tiny_env_config(), profile="learnability")
    env.reset(seed=1000)
    snapshots = snapshot_realized_uav_slots(env)
    for slot, aid in zip(SOURCE_SLOTS, UAV_IDS):
        assert state_tuple(snapshots[slot]) == state_tuple(env.entities[aid].state)
        assert snapshots[slot] is not env.entities[aid].state
        assert {field.name for field in fields(snapshots[slot])} == {
            "x", "y", "h", "v", "theta", "psi", "alive",
        }


def test_nonidentity_moves_full_realized_states_and_preserves_all_other_aircraft():
    env = HeterogeneousMAVUAVAirCombatEnv(tiny_env_config(), profile="learnability")
    env.reset(seed=1007)
    snapshots = snapshot_realized_uav_slots(env)
    unchanged = {aid: state_tuple(env.entities[aid].state) for aid in ("MAV", *BLUE_IDS)}
    aircraft_identity = {aid: id(env.entities[aid]) for aid in UAV_IDS}
    specs = {aid: env.entities[aid].spec for aid in UAV_IDS}
    mapping = {"UAV1": "S3", "UAV2": "S1", "UAV3": "S2"}
    apply_realized_slot_permutation(env, snapshots, mapping)
    for aid in UAV_IDS:
        assert state_tuple(env.entities[aid].state) == state_tuple(snapshots[mapping[aid]])
        assert env.entities[aid].state is not snapshots[mapping[aid]]
        assert id(env.entities[aid]) == aircraft_identity[aid]
        assert env.entities[aid].spec is specs[aid]
        assert env.entities[aid].aircraft_id == aid
    assert len({id(env.entities[aid].state) for aid in UAV_IDS}) == 3
    assert {aid: state_tuple(env.entities[aid].state) for aid in unchanged} == unchanged


def test_identity_regenerated_observations_equal_reset_observations_exactly():
    env = HeterogeneousMAVUAVAirCombatEnv(tiny_env_config(), profile="learnability")
    expected, _ = env.reset(seed=1011)
    actual, _ = reset_with_realized_permutation(env, 1011, IDENTITY)
    for aid in RED_IDS:
        np.testing.assert_array_equal(actual[aid], expected[aid])


def test_same_environment_seed_reproduces_all_realized_slot_fields():
    first = HeterogeneousMAVUAVAirCombatEnv(tiny_env_config(), profile="learnability")
    second = HeterogeneousMAVUAVAirCombatEnv(tiny_env_config(), profile="learnability")
    first.reset(seed=1020); second.reset(seed=1020)
    one = snapshot_realized_uav_slots(first); two = snapshot_realized_uav_slots(second)
    assert {slot: state_tuple(one[slot]) for slot in SOURCE_SLOTS} == {
        slot: state_tuple(two[slot]) for slot in SOURCE_SLOTS
    }


def synthetic_agent_rows(episodes: int) -> list[dict]:
    rows = []
    for mapping in slot_permutations():
        for episode in range(episodes):
            for actor_index, aid in enumerate(UAV_IDS):
                slot = mapping[aid]
                boundary = int(actor_index == 0 and slot == "S3")
                blue_attack = int(actor_index == 1 and slot == "S2")
                alive = int(not (boundary or blue_attack))
                rows.append({
                    "run": "r", "actor_id": aid, "source_slot": slot,
                    "alive": alive, "boundary": boundary,
                    "blue_attack": blue_attack, "other": 0,
                })
    return rows


@pytest.mark.parametrize("episodes", [1, 4])
def test_actor_slot_coverage_is_exactly_two_n(episodes):
    matrix = build_actor_slot_matrix(synthetic_agent_rows(episodes), episodes)
    assert len(matrix) == 9
    assert {row["sample_count"] for row in matrix} == {2 * episodes}


def test_actor_slot_matrix_and_actor_slot_marginals_are_correct():
    rows = synthetic_agent_rows(2)
    matrix = build_actor_slot_matrix(rows, 2)
    special = next(row for row in matrix if row["actor_id"] == "UAV1" and row["source_slot"] == "S3")
    assert special["boundary_count"] == 4
    assert special["boundary_rate"] == 1.0
    actor = build_marginals(rows, "actor_id", UAV_IDS)
    slot = build_marginals(rows, "source_slot", SOURCE_SLOTS)
    assert next(row for row in actor if row["actor_id"] == "UAV1")["boundary_rate"] == pytest.approx(1 / 3)
    assert next(row for row in slot if row["source_slot"] == "S3")["boundary_rate"] == pytest.approx(1 / 3)


def test_cross_run_contract_accepts_equal_config_and_rejects_mismatch():
    config = tiny_env_config()
    base = dict(observation_dim=OBS_DIM, environment_version=config["environment_version"])
    validate_cross_run_environment_contract([
        SimpleNamespace(**base, env_config=deepcopy(config)),
        SimpleNamespace(**base, env_config=deepcopy(config)),
    ])
    changed = deepcopy(config); changed["simulation"]["max_decision_steps"] += 1
    with pytest.raises(RuntimeError, match="identical resolved environment configs"):
        validate_cross_run_environment_contract([
            SimpleNamespace(**base, env_config=config),
            SimpleNamespace(**base, env_config=changed),
        ])
    with pytest.raises(RuntimeError, match="observation_dim"):
        validate_cross_run_environment_contract([
            SimpleNamespace(observation_dim=OBS_DIM + 1,
                            environment_version=config["environment_version"], env_config=config),
        ])


def seeded_actors() -> IndependentActors:
    state = torch.get_rng_state()
    try:
        torch.manual_seed(77)
        actors = IndependentActors(hidden_dim=8)
        actors.eval()
        return actors
    finally:
        torch.set_rng_state(state)


def test_identity_tiny_rollout_matches_formal_stochastic_evaluator():
    config = tiny_env_config(max_steps=2)
    actors = seeded_actors()
    formal = evaluate_actors(
        actors, config, 1, "learnability", seed=1000, device="cpu",
        deterministic=False, action_seed=2000,
    )[0]
    audited, _ = rollout_permuted_episode(
        actors, config, "learnability", IDENTITY, episode=0,
        env_seed=1000, action_seed=2000, device="cpu",
    )
    assert audited == formal


def test_action_seed_reproducibility_and_fixed_actor_mapping():
    config = tiny_env_config(max_steps=2)
    actors = seeded_actors()
    first = rollout_permuted_episode(
        actors, config, "learnability", {"UAV1": "S3", "UAV2": "S2", "UAV3": "S1"},
        episode=0, env_seed=1000, action_seed=2000, device="cpu",
    )
    second = rollout_permuted_episode(
        actors, config, "learnability", {"UAV1": "S3", "UAV2": "S2", "UAV3": "S1"},
        episode=0, env_seed=1000, action_seed=2000, device="cpu",
    )
    assert first == second
    assert tuple(enumerate(RED_IDS)) == ((0, "MAV"), (1, "UAV1"), (2, "UAV2"), (3, "UAV3"))


def make_tiny_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tiny_env_config(max_steps=1)
    trainer = HAPPOTrainer(config, {
        "num_envs": 1, "rollout_steps": 1, "hidden_dim": 8, "device": "cpu",
        "environment_profile": "learnability", "method_variant": "rgaa",
    })
    try:
        trainer.save_checkpoint(run_dir / "checkpoint_final.pt")
    finally:
        trainer.close()
    return run_dir


def audit_args(run_dir: Path, output: Path, device: str = "cpu") -> SimpleNamespace:
    return SimpleNamespace(
        run=[("tiny", run_dir)], episodes=1, profile="learnability", device=device,
        env_seed=1000, action_seed=2000, output=output,
    )


def test_cpu_tiny_audit_preserves_checkpoint_actors_rng_and_writes_contract_outputs(tmp_path):
    run_dir = make_tiny_run(tmp_path)
    checkpoint = run_dir / "checkpoint_final.pt"
    before_digest = file_sha256(checkpoint)
    before_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_actors = {key: value.clone() for key, value in before_payload["actors"].items()}
    torch.manual_seed(99123)
    before_rng = torch.get_rng_state().clone()
    output = tmp_path / "audit"
    summary = run_audit(audit_args(run_dir, output))
    assert summary["permutation_count"] == 6
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert file_sha256(checkpoint) == before_digest
    after = torch.load(checkpoint, map_location="cpu", weights_only=False)["actors"]
    assert all(torch.equal(after[key], value) for key, value in before_actors.items())
    expected = {
        "episode_agent_records.csv", "episode_team_records.csv", "permutation_summary.csv",
        "actor_slot_matrix.csv", "actor_marginal.csv", "slot_marginal.csv",
        "audit_summary.json", "README.txt",
    }
    assert {path.name for path in output.iterdir()} == expected
    matrix = np.genfromtxt(output / "actor_slot_matrix.csv", delimiter=",", names=True,
                           dtype=None, encoding="utf-8")
    assert len(matrix) == 9
    assert set(matrix["sample_count"].tolist()) == {2}
    performance = np.genfromtxt(
        output / "permutation_summary.csv", delimiter=",", names=True,
        dtype=None, encoding="utf-8",
    )
    assert len(performance) == 6
    for field in ("win_rate", "mean_return", "mean_red_kills", "mean_uav_survivors"):
        assert np.isfinite(performance[field]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiny_rollout_and_rng_restoration(tmp_path):
    run_dir = make_tiny_run(tmp_path)
    torch.cuda.manual_seed_all(7788)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    output = tmp_path / "cuda_audit"
    run_audit(audit_args(run_dir, output, device="cuda"))
    after = torch.cuda.get_rng_state_all()
    assert len(after) == len(before)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    performance = np.genfromtxt(
        output / "permutation_summary.csv", delimiter=",", names=True,
        dtype=None, encoding="utf-8",
    )
    assert len(performance) == 6
    for field in ("win_rate", "mean_return", "mean_red_kills", "mean_uav_survivors"):
        assert np.isfinite(performance[field]).all()
