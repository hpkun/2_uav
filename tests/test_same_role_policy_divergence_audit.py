from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.happo.networks import IndependentActors
from env.mavuav import OBS_DIM, load_environment_config
from tools.audit_same_role_policy_divergence import (
    LoadedRun, balanced_subsample, build_shared_probe_pool, collect_probes,
    compute_actor_outliers, diagonal_gaussian_symmetric_kl,
    evaluate_on_probe_pool, excess_diagnostics, file_sha256, healthy_reference,
    load_run, parse_args, run_audit, validate_cross_run_environment_contract,
)


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def make_checkpoint(
    tmp_path: Path, label: str = "run", *, seed: int = 7,
    env_config: dict | None = None, observation_dim: int = OBS_DIM,
) -> Path:
    torch.manual_seed(seed)
    actors = IndependentActors(hidden_dim=16)
    run_dir = tmp_path / label
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_final.pt"
    torch.save({
        "actors": actors.state_dict(), "environment_config": env_config or short_v39(),
        "environment_version": "heterogeneous_mavuav_4v4_v3_9",
        "observation_dim": observation_dim, "actor_variant": "vanilla",
        "method_variant": "rgaa", "sampled_steps": 2_000_000,
        "trainer_config": {
            "hidden_dim": 16, "seed": seed, "actor_variant": "vanilla",
            "method_variant": "rgaa", "actor_log_std_init": -0.5,
        },
    }, checkpoint)
    return run_dir


def probe(index: int, run: str, aid: str, *, cause: str = "alive") -> dict:
    return {
        "probe_id": index, "source_run": run, "source_method": "rgaa",
        "source_seed": 1, "source_episode": 0, "source_step": index,
        "source_agent": aid, "source_agent_active": True,
        "source_final_death_cause": cause,
        "observation": np.full(OBS_DIM, index / 100.0, dtype=np.float32),
    }


def test_gaussian_skl_identical_is_zero_and_symmetric():
    mean = np.asarray([[0.2, -0.3, 0.4]])
    log_std = np.asarray([[-0.5, -0.4, -0.3]])
    same = diagonal_gaussian_symmetric_kl(mean, log_std, mean, log_std)
    assert all(np.allclose(same[name], 0.0) for name in ("skl_total", "skl_mean", "skl_scale"))
    other_mean = mean + np.asarray([[0.4, -0.1, 0.2]])
    other_std = log_std + 0.25
    forward = diagonal_gaussian_symmetric_kl(mean, log_std, other_mean, other_std)
    backward = diagonal_gaussian_symmetric_kl(other_mean, other_std, mean, log_std)
    for name in ("skl_total", "skl_mean", "skl_scale"):
        np.testing.assert_allclose(forward[name], backward[name])


def test_gaussian_skl_mean_scale_decomposition_cases():
    zero = np.zeros((2, 3))
    shifted = np.ones((2, 3))
    mean_only = diagonal_gaussian_symmetric_kl(zero, zero, shifted, zero)
    assert np.all(mean_only["skl_mean"] > 0.0)
    np.testing.assert_array_equal(mean_only["skl_scale"], np.zeros(2))
    scale_only = diagonal_gaussian_symmetric_kl(zero, zero, zero, np.full((2, 3), 0.5))
    np.testing.assert_array_equal(scale_only["skl_mean"], np.zeros(2))
    assert np.all(scale_only["skl_scale"] > 0.0)
    for values in (mean_only, scale_only):
        np.testing.assert_allclose(values["skl_total"], values["skl_mean"] + values["skl_scale"])


def test_three_actor_outlier_formula_is_exact():
    arrays = {
        "UAV1-UAV2": {"skl_total": np.array([2.0]), "skl_mean": np.array([1.0]), "skl_scale": np.array([1.0])},
        "UAV1-UAV3": {"skl_total": np.array([4.0]), "skl_mean": np.array([3.0]), "skl_scale": np.array([1.0])},
        "UAV2-UAV3": {"skl_total": np.array([8.0]), "skl_mean": np.array([6.0]), "skl_scale": np.array([2.0])},
    }
    result = compute_actor_outliers(arrays)
    assert result["UAV1"]["outlier_total"] == pytest.approx([3.0])
    assert result["UAV2"]["outlier_total"] == pytest.approx([5.0])
    assert result["UAV3"]["outlier_total"] == pytest.approx([6.0])
    for aid in ("UAV1", "UAV2", "UAV3"):
        np.testing.assert_allclose(
            result[aid]["outlier_total"],
            result[aid]["outlier_mean_component"] + result[aid]["outlier_scale_component"],
        )


def test_balanced_subsampling_is_equal_and_probe_seed_reproducible():
    rows = []
    for agent_index, aid in enumerate(("UAV1", "UAV2", "UAV3")):
        rows.extend(probe(agent_index * 100 + i, "A", aid) for i in range(9 - agent_index))
    first = balanced_subsample(rows, 6, 53001, "A")
    second = balanced_subsample(rows, 6, 53001, "A")
    assert [(r["source_agent"], r["source_step"]) for r in first] == [
        (r["source_agent"], r["source_step"]) for r in second
    ]
    assert {aid: sum(r["source_agent"] == aid for r in first) for aid in ("UAV1", "UAV2", "UAV3")} == {
        "UAV1": 6, "UAV2": 6, "UAV3": 6,
    }


def test_shared_pool_is_identical_for_all_checkpoint_evaluations(tmp_path):
    run_a = load_run("A", make_checkpoint(tmp_path, "A", seed=1), "cpu")
    run_b = load_run("B", make_checkpoint(tmp_path, "B", seed=2), "cpu")
    raw = {
        label: [probe(i + offset, label, aid) for aid in ("UAV1", "UAV2", "UAV3") for i in range(3)]
        for label, offset in (("A", 0), ("B", 100))
    }
    pool = build_shared_probe_pool(raw, 2, 53001)
    pairs_a, _ = evaluate_on_probe_pool(run_a, pool, "cpu")
    pairs_b, _ = evaluate_on_probe_pool(run_b, pool, "cpu")
    ids_a = [row["probe_id"] for row in pairs_a if row["pair"] == "UAV1-UAV2"]
    ids_b = [row["probe_id"] for row in pairs_b if row["pair"] == "UAV1-UAV2"]
    assert ids_a == ids_b == list(range(len(pool)))


def test_shared_pool_balances_every_run_agent_group_globally():
    counts = {
        "A": {"UAV1": 10, "UAV2": 9, "UAV3": 8},
        "B": {"UAV1": 6, "UAV2": 7, "UAV3": 9},
    }
    raw = {
        label: [
            probe(run_index * 1000 + agent_index * 100 + index, label, aid)
            for agent_index, aid in enumerate(("UAV1", "UAV2", "UAV3"))
            for index in range(counts[label][aid])
        ]
        for run_index, label in enumerate(("A", "B"))
    }
    pool = build_shared_probe_pool(raw, 10, 53001)
    assert len(pool) == 2 * 3 * 6
    assert {
        (label, aid): sum(row["source_run"] == label and row["source_agent"] == aid for row in pool)
        for label in ("A", "B") for aid in ("UAV1", "UAV2", "UAV3")
    } == {(label, aid): 6 for label in ("A", "B") for aid in ("UAV1", "UAV2", "UAV3")}


def test_cross_run_environment_contract_accepts_identical_configs(tmp_path):
    run_a = load_run("A", make_checkpoint(tmp_path, "A", seed=1), "cpu")
    run_b = load_run("B", make_checkpoint(tmp_path, "B", seed=2), "cpu")
    validate_cross_run_environment_contract([run_a, run_b])


def test_cross_run_environment_contract_rejects_resolved_config_difference(tmp_path):
    config_a = short_v39()
    config_b = deepcopy(config_a)
    config_b["simulation"]["max_decision_steps"] = 3
    run_a = load_run("A", make_checkpoint(tmp_path, "A", seed=1, env_config=config_a), "cpu")
    run_b = load_run("B", make_checkpoint(tmp_path, "B", seed=2, env_config=config_b), "cpu")
    with pytest.raises(RuntimeError, match="identical resolved environment configs"):
        validate_cross_run_environment_contract([run_a, run_b])


def test_checkpoint_observation_dim_mismatch_and_missing_field_are_rejected(tmp_path):
    mismatch = make_checkpoint(tmp_path, "mismatch", observation_dim=OBS_DIM - 1)
    with pytest.raises(RuntimeError, match="observation_dim mismatch"):
        load_run("mismatch", mismatch, "cpu")
    missing = make_checkpoint(tmp_path, "missing")
    checkpoint = missing / "checkpoint_final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("observation_dim")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="missing observation_dim"):
        load_run("missing", missing, "cpu")


def test_collect_probes_annotates_final_death_cause(monkeypatch, tmp_path):
    loaded = load_run("A", make_checkpoint(tmp_path, "A"), "cpu")

    class FakeEnv:
        red_ids = ("MAV", "UAV1", "UAV2", "UAV3")
        active_masks = np.ones(4, np.float32)

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed):
            self.active_masks = np.ones(4, np.float32)
            return {aid: np.zeros(OBS_DIM, np.float32) for aid in self.red_ids}, {}

        def step(self, actions):
            return (
                {aid: np.zeros(OBS_DIM, np.float32) for aid in self.red_ids}, {}, True, False,
                {"death_causes": {"UAV1": "boundary", "UAV2": "blue_attack"}},
            )

    monkeypatch.setattr("tools.audit_same_role_policy_divergence.HeterogeneousMAVUAVAirCombatEnv", FakeEnv)
    rows = collect_probes(loaded, 1, "learnability", env_seed=1000, action_seed=2000, device="cpu")
    assert {row["source_agent"]: row["source_final_death_cause"] for row in rows} == {
        "UAV1": "boundary", "UAV2": "blue_attack", "UAV3": "alive",
    }
    assert all(row["source_agent_active"] is True for row in rows)


def test_stochastic_probe_rollout_is_reproducible(tmp_path):
    loaded = load_run("A", make_checkpoint(tmp_path, "A"), "cpu")
    first = collect_probes(loaded, 2, "learnability", env_seed=1000, action_seed=2000, device="cpu")
    second = collect_probes(loaded, 2, "learnability", env_seed=1000, action_seed=2000, device="cpu")
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert {k: v for k, v in left.items() if k != "observation"} == {
            k: v for k, v in right.items() if k != "observation"
        }
        np.testing.assert_array_equal(left["observation"], right["observation"])


def test_healthy_quantiles_and_excess_rates_are_exact():
    pair_rows = []
    outlier_rows = []
    for run, offset in (("healthy", 0.0), ("candidate", 10.0)):
        for index, pair in enumerate(("UAV1-UAV2", "UAV1-UAV3", "UAV2-UAV3")):
            pair_rows.append({"evaluated_run": run, "pair": pair, "skl_total": offset + index})
        for index, aid in enumerate(("UAV1", "UAV2", "UAV3")):
            outlier_rows.append({"evaluated_run": run, "actor": aid, "outlier_total": offset + index})
    reference = healthy_reference(pair_rows, outlier_rows, ["healthy"])
    assert reference["healthy_pairwise_q95"] == pytest.approx(np.quantile([0, 1, 2], 0.95))
    assert reference["delta_candidate"] == reference["healthy_pairwise_q95"]
    excess = excess_diagnostics(pair_rows, outlier_rows, reference)
    assert all(value == 1.0 for value in excess["candidate"]["pair_excess_rate_q95"].values())
    assert all(value == 1.0 for value in excess["candidate"]["actor_outlier_excess_rate_q95"].values())
    no_reference = healthy_reference(pair_rows, outlier_rows, [])
    no_excess = excess_diagnostics(pair_rows, outlier_rows, no_reference)
    assert no_reference["delta_candidate"] is None
    assert all(value is None for value in no_excess["candidate"]["pair_excess_rate_q95"].values())


def test_cpu_tiny_audit_is_read_only_restores_rng_and_accepts_missing_death_audit(tmp_path):
    run_dir = make_checkpoint(tmp_path, "A")
    checkpoint = run_dir / "checkpoint_final.pt"
    before_hash = file_sha256(checkpoint)
    before_rng = torch.get_rng_state().clone()
    output = tmp_path / "audit"
    args = parse_args([
        "--run", f"A={run_dir}", "--probe-episodes", "1", "--profile", "learnability",
        "--device", "cpu", "--max-probes-per-run-agent", "3", "--output", str(output),
    ])
    summary = run_audit(args)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert file_sha256(checkpoint) == before_hash
    assert summary["runs"]["A"]["death_audit"]["UAV1"]["boundary_rate"] is None
    assert summary["shared_probe_global_per_run_agent"] == 2
    assert summary["shared_probe_count_per_run"] == {"A": 6}
    for name in (
        "probe_manifest.csv", "shared_pairwise_samples.csv", "pairwise_summary.csv",
        "actor_outlier_summary.csv", "conditioned_summary.csv", "healthy_reference.json",
        "audit_summary.json", "README.txt",
    ):
        assert (output / name).is_file()
    manifest = (output / "probe_manifest.csv").read_text(encoding="utf-8")
    assert "observation" in manifest and "source_final_death_cause" in manifest
    assert "causal conclusion" in summary["interpretation_limit"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiny_forward_only_audit(tmp_path):
    run_dir = make_checkpoint(tmp_path, "A")
    output = tmp_path / "cuda_audit"
    args = parse_args([
        "--run", f"A={run_dir}", "--probe-episodes", "1", "--device", "cuda",
        "--max-probes-per-run-agent", "2", "--output", str(output),
    ])
    summary = run_audit(args)
    assert summary["read_only"] is True
    assert summary["shared_probe_count"] == 6
