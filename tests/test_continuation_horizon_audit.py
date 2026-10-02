from __future__ import annotations

from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.dbm_rgaa import DBM_RGAA_METHOD, build_method_actors
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from tools.audit_continuation_horizon import (
    EPISODE_FIELDS, ContinuationAuditEnv, aggregate_checkpoint_summaries,
    load_tacm_checkpoint, project_horizon_records, run_continuation_episode, summarize_checkpoint_rows,
    validate_audit_contract, write_outputs,
)


def config_v39():
    return deepcopy(load_environment_config("configs/env_v39.yaml"))


def tiny_tacm_config():
    with open("configs/happo_tacm_rgaa_v39.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["training"]
    config.update({
        "device": "cpu", "num_envs": 1, "rollout_steps": 2,
        "ppo_epochs": 1, "minibatch_size": 2, "seed": 29,
    })
    return config


def _disable_losses(env):
    env._apply_boundaries = lambda: {}
    env._resolve_attacks = lambda: ([], {})


def test_observation_clock_and_prefix_dynamics_are_invariant_through_step75():
    config = config_v39()
    standard = HeterogeneousMAVUAVAirCombatEnv(config, seed=21, randomize=False, profile="main")
    audit = ContinuationAuditEnv(config, audit_max_decision_steps=100, profile="main")
    standard_obs, _ = standard.reset(seed=21)
    audit_obs, _ = audit.reset(seed=21, options={"randomize": False})
    _disable_losses(standard); _disable_losses(audit)
    actions = np.zeros((4, 3), dtype=np.float32)
    assert all(np.array_equal(standard_obs[aid], audit_obs[aid]) for aid in RED_IDS)
    for step in range(1, 76):
        standard_obs, standard_reward, st, sx, standard_info = standard.step(actions)
        audit_obs, audit_reward, at, ax, audit_info = audit.step(actions)
        for aid in RED_IDS:
            assert np.array_equal(standard_obs[aid], audit_obs[aid])
            assert standard_reward[aid] == audit_reward[aid]
        for aid in (*RED_IDS, *BLUE_IDS):
            assert np.array_equal(standard.entities[aid].state.as_array(), audit.entities[aid].state.as_array())
            assert standard.entities[aid].state.alive == audit.entities[aid].state.alive
        assert standard._attack_streak == audit._attack_streak
        assert standard._red_attack_kills == audit._red_attack_kills
        assert standard._blue_attack_kills == audit._blue_attack_kills
        if step < 75:
            assert not (st or sx or at or ax)
        else:
            assert sx and standard_info["outcome"] == "draw"
            assert not (at or ax) and audit_info["outcome"] is None
            assert standard_obs["MAV"][10] == audit_obs["MAV"][10] == 1.0
    audit_obs, _, terminated, truncated, _ = audit.step(actions)
    assert not (terminated or truncated)
    assert all(observation[10] == 1.0 for observation in audit_obs.values())
    assert audit.max_decision_steps == audit.observation_horizon == 75
    assert audit.config["simulation"]["max_decision_steps"] == 75


def test_audit_terminal_priority_is_unchanged():
    env = ContinuationAuditEnv(config_v39(), audit_max_decision_steps=200)
    env.reset(seed=1)
    env.entities["MAV"].state.alive = False
    assert env._termination() == (True, False, "blue")
    env.reset(seed=1)
    for aid in BLUE_IDS:
        env.entities[aid].state.alive = False
    env._red_attack_kills = set(BLUE_IDS)
    assert env._termination() == (True, False, "red")


@pytest.mark.parametrize("step,outcome,expected", [
    (90, "red", ["draw", "red", "red", "red", "red"]),
    (110, "blue", ["draw", "draw", "blue", "blue", "blue"]),
    (None, None, ["draw", "draw", "draw", "draw", "draw"]),
])
def test_horizon_projection(step, outcome, expected):
    horizons = (75, 100, 125, 150, 200)
    snapshots = {h: {"mav_survived": True, "red_uav_survivors": 2,
                     "blue_survivors": 1, "red_attack_kills": 3,
                     "blue_attack_kills": 0, "cumulative_team_return": float(h)}
                 for h in horizons}
    rows = project_horizon_records(
        snapshots, horizons, actual_terminal_step=step,
        actual_terminal_outcome=outcome,
        metadata={"observation_horizon": 75}, action_trace_sha256="abc",
    )
    assert [row["outcome"] for row in rows] == expected
    assert all(row["outcome_at_75"] == "draw" for row in rows)


def test_stochastic_episode_seed_is_reproducible_and_action_seed_changes_trace():
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=7)
    metadata = {
        "checkpoint": "synthetic", "checkpoint_path": "synthetic.pt",
        "method_variant": "tacm_rgaa", "training_seed": 7, "sampled_steps": 0,
        "action_mode": "stochastic", "environment_profile": "main",
        "environment_version": "heterogeneous_mavuav_4v4_v3_9",
        "observation_horizon": 75, "audit_max_horizon": 75,
    }
    kwargs = dict(actors=actors, env_config=config_v39(), profile="main", episode=0,
                  env_seed=3000, action_mode="stochastic", device="cpu", horizons=(75,),
                  metadata=metadata)
    first = run_continuation_episode(action_seed=4000, **kwargs)
    second = run_continuation_episode(action_seed=4000, **kwargs)
    changed = run_continuation_episode(action_seed=4001, **kwargs)
    assert first == second
    assert first[0]["action_trace_sha256"] != changed[0]["action_trace_sha256"]


def test_checkpoint_environment_contract_rejects_mutation_and_potential_shaping():
    config = config_v39()
    payload = {
        "environment_version": config["environment_version"], "observation_dim": 100,
        "global_state_dim": 117, "reward_mode": "heterogeneous_role_coupled_gate_v1",
        "method_variant": "tacm_rgaa", "algorithm": "tacm_rgaa_happo",
        "trainer_config": {"method_variant": "tacm_rgaa"},
        "environment_config": deepcopy(config),
    }
    before = deepcopy(payload["environment_config"])
    assert validate_audit_contract(payload, config) == 75
    assert payload["environment_config"] == before
    bad = deepcopy(config); bad["reward"]["terminal_draw"] = 1.0
    payload_bad = deepcopy(payload); payload_bad["environment_config"] = deepcopy(bad)
    with pytest.raises(RuntimeError, match="terminal_draw"):
        validate_audit_contract(payload_bad, bad)
    bad = deepcopy(config); bad["shaping"] = {"mode": "potential", "gamma": 0.99}
    payload_potential = deepcopy(payload)
    payload_potential["environment_config"] = deepcopy(bad)
    # The frozen v3.9 loader may reject forbidden shaping before the audit's
    # redundant contract check; either way the incompatible contract is fatal.
    with pytest.raises((RuntimeError, ValueError)):
        validate_audit_contract(payload_potential, bad)


def test_real_tacm_checkpoint_load_is_strict_and_read_only(tmp_path):
    trainer = HAPPOTrainer(config_v39(), tiny_tacm_config())
    checkpoint = tmp_path / "checkpoint_final.pt"
    try:
        trainer.save_checkpoint(checkpoint)
        expected_actors = {
            key: value.detach().clone() for key, value in trainer.actors.state_dict().items()
        }
    finally:
        trainer.close()

    before = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_environment = deepcopy(before["environment_config"])
    loaded = load_tacm_checkpoint(checkpoint, "cpu")
    assert loaded["observation_horizon"] == 75
    assert loaded["environment_config"] == before_environment
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["environment_config"] == before_environment
    for key, expected in expected_actors.items():
        assert torch.equal(loaded["actors"].state_dict()[key], expected), key

    tampered = deepcopy(before)
    tampered["tacm_tau_teacher"] = float(tampered["tacm_tau_teacher"]) + 0.1
    bad_checkpoint = tmp_path / "tampered.pt"
    torch.save(tampered, bad_checkpoint)
    with pytest.raises(RuntimeError, match="tacm_tau_teacher"):
        load_tacm_checkpoint(bad_checkpoint, "cpu")


def test_output_schema_and_checkpoint_labels_do_not_collide(tmp_path):
    horizons = (75, 100)
    rows = []
    for checkpoint, seed in (("run_a/checkpoint.pt", 5), ("run_b/checkpoint.pt", 7)):
        snapshots = {h: {"mav_survived": True, "red_uav_survivors": 2,
                         "blue_survivors": 1, "red_attack_kills": 3,
                         "blue_attack_kills": 0, "cumulative_team_return": 1.0}
                     for h in horizons}
        rows.extend(project_horizon_records(
            snapshots, horizons, actual_terminal_step=90, actual_terminal_outcome="red",
            metadata={
                "checkpoint": checkpoint, "checkpoint_path": checkpoint,
                "method_variant": "tacm_rgaa", "training_seed": seed, "sampled_steps": 2_000_000,
                "episode": 0, "environment_seed": 3000, "action_seed": 4000,
                "action_mode": "stochastic", "environment_profile": "main",
                "environment_version": "heterogeneous_mavuav_4v4_v3_9",
                "observation_horizon": 75, "audit_max_horizon": 100,
            }, action_trace_sha256="abc",
        ))
    summaries = []
    for checkpoint in ("run_a/checkpoint.pt", "run_b/checkpoint.pt"):
        summaries.extend(summarize_checkpoint_rows([r for r in rows if r["checkpoint"] == checkpoint]))
    aggregates = aggregate_checkpoint_summaries(summaries)
    output = tmp_path / "audit"
    write_outputs(output, rows, summaries, aggregates, {"observation_horizon": 75})
    with (output / "episode_horizon_records.csv").open(encoding="utf-8", newline="") as stream:
        loaded = list(csv.DictReader(stream))
    assert len(loaded) == 4
    assert set(EPISODE_FIELDS).issubset(loaded[0])
    assert {row["checkpoint"] for row in loaded} == {"run_a/checkpoint.pt", "run_b/checkpoint.pt"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"]["observation_horizon"] == 75
    assert len(summary["checkpoint_horizon_summary"]) == 4
