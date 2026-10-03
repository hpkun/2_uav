from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from algorithm.happo.dbm_rgaa import DBM_RGAA_METHOD, build_method_actors
from algorithm.happo.evaluation import evaluate_actors
from env.mavuav import RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.vector_env import _environment_state
from tools.audit_combat_skill_causality import (
    CONDITIONS, candidate_attackers, condition_actions, evaluate_streak2_opportunity,
    gate_step_statistics, intervention_action_sets, recompute_reward_targets,
    paired_condition_episode_records, red_attack_event_pairs, reward_target_matches,
    run_condition_episode, write_outputs,
)
from tools.validate_role_guided_curriculum_configs import COMMON_FIELDS, load_training, validate_configs


ROOT = Path(__file__).resolve().parents[1]


class CountingActor:
    def __init__(self, value):
        self.value = torch.as_tensor(value, dtype=torch.float32)
        self.calls = 0

    def sample(self, observation, deterministic=False):
        self.calls += 1
        action = self.value.to(observation.device).expand(observation.shape[0], -1)
        return action, torch.zeros(observation.shape[0], device=observation.device)


class CountingActors:
    def __init__(self):
        self.actors = [CountingActor((index + 1, 0, 0)) for index in range(4)]


def v39():
    return deepcopy(load_environment_config(ROOT / "configs/env_v39.yaml"))


def test_gate_accounting_excludes_unarmed_mav_but_keeps_separate_geometry():
    mav_only = [{
        "attacker": "MAV", "target": "Blue1", "gate": True,
        "previous_streak": 0,
    }]
    v310, next_gate = gate_step_statistics(
        mav_only, mav_can_attack=False, mav_previous_gate={},
    )
    assert (v310["red_exposures"], v310["red_active"], v310["red_entries"]) == (0, 0, 0)
    assert (v310["uav_exposures"], v310["uav_active"], v310["uav_entries"]) == (0, 0, 0)
    assert (
        v310["mav_geometric_exposures"], v310["mav_geometric_active"],
        v310["mav_geometric_entries"],
    ) == (1, 1, 1)
    assert next_gate[("MAV", "Blue1")] is True

    # Remaining inside the geometric gate is active but is not a second entry.
    again, _ = gate_step_statistics(
        mav_only, mav_can_attack=False, mav_previous_gate=next_gate,
    )
    assert again["mav_geometric_active"] == 1
    assert again["mav_geometric_entries"] == 0


def test_gate_accounting_counts_v310_uav_and_preserves_v39_mav_history():
    uav = [{
        "attacker": "UAV1", "target": "Blue2", "gate": True,
        "previous_streak": 0,
    }]
    v310, _ = gate_step_statistics(uav, mav_can_attack=False, mav_previous_gate={})
    assert (v310["red_exposures"], v310["red_active"], v310["red_entries"]) == (1, 1, 1)
    assert (v310["uav_exposures"], v310["uav_active"], v310["uav_entries"]) == (1, 1, 1)

    mav = [{
        "attacker": "MAV", "target": "Blue3", "gate": True,
        "previous_streak": 0,
    }]
    v39_stats, _ = gate_step_statistics(mav, mav_can_attack=True, mav_previous_gate={})
    assert (
        v39_stats["red_exposures"], v39_stats["red_active"], v39_stats["red_entries"],
    ) == (1, 1, 1)
    assert v39_stats["per_agent"]["MAV"]["streak1"] == 1


def test_paired_episode_gate_fraction_uses_corrected_red_contract():
    common = {
        "checkpoint": "run/checkpoint.pt", "training_seed": 5, "episode": 0,
        "environment_seed": 3000, "outcome": "draw", "red_attack_kills": 0,
        "episode_return": 0.0, "attack_event_pair_count": 0, "MAV_survival": True,
    }
    rows = [
        {**common, "condition": "POLICY", "red_gate_active_fraction": 0.25},
        {**common, "condition": "UAV_TRIM", "red_gate_active_fraction": 0.10},
    ]
    paired = paired_condition_episode_records(rows)
    row = next(item for item in paired if item["control_condition"] == "UAV_TRIM")
    assert row["policy_gate_fraction"] == 0.25
    assert row["control_gate_fraction"] == 0.10
    assert np.isclose(row["delta_gate_fraction"], 0.15)


def test_condition_actions_do_not_add_samples_and_cyclic_mapping_is_exact():
    observations = {aid: np.zeros(100, np.float32) for aid in RED_IDS}
    policy = CountingActors()
    normal = condition_actions("POLICY", policy, observations, "cpu", None)
    assert [actor.calls for actor in policy.actors] == [1, 1, 1, 1]
    cyclic = condition_actions("UAV_ACTION_CYCLIC", policy, observations, "cpu", None)
    assert [actor.calls for actor in policy.actors] == [2, 2, 2, 2]
    np.testing.assert_array_equal(cyclic["MAV"], normal["MAV"])
    np.testing.assert_array_equal(cyclic["UAV2"], normal["UAV1"])
    np.testing.assert_array_equal(cyclic["UAV3"], normal["UAV2"])
    np.testing.assert_array_equal(cyclic["UAV1"], normal["UAV3"])
    trim = condition_actions("TRIM", policy, observations, "cpu", None)
    assert all(np.array_equal(value, np.zeros(3)) for value in trim.values())
    assert [actor.calls for actor in policy.actors] == [2, 2, 2, 2]
    uav_trim = condition_actions("UAV_TRIM", policy, observations, "cpu", None)
    assert [actor.calls for actor in policy.actors] == [3, 3, 3, 3]
    np.testing.assert_array_equal(uav_trim["MAV"], normal["MAV"])
    assert all(np.array_equal(uav_trim[aid], np.zeros(3)) for aid in RED_IDS[1:])
    uav_random = condition_actions(
        "UAV_UNIFORM_RANDOM", policy, observations, "cpu", np.random.default_rng(17),
    )
    assert [actor.calls for actor in policy.actors] == [4, 4, 4, 4]
    np.testing.assert_array_equal(uav_random["MAV"], normal["MAV"])
    assert all(np.all((-1 <= uav_random[aid]) & (uav_random[aid] <= 1)) for aid in RED_IDS[1:])


def test_uniform_random_is_bounded_reproducible_and_seed_sensitive():
    observations = {aid: np.zeros(100, np.float32) for aid in RED_IDS}
    first = condition_actions("UNIFORM_RANDOM", None, observations, "cpu", np.random.default_rng(12))
    second = condition_actions("UNIFORM_RANDOM", None, observations, "cpu", np.random.default_rng(12))
    changed = condition_actions("UNIFORM_RANDOM", None, observations, "cpu", np.random.default_rng(13))
    assert all(np.array_equal(first[aid], second[aid]) for aid in RED_IDS)
    assert any(not np.array_equal(first[aid], changed[aid]) for aid in RED_IDS)
    assert all(np.all((-1 <= first[aid]) & (first[aid] <= 1)) for aid in RED_IDS)


def test_reward_target_recomputation_is_read_only_and_match_is_explicit():
    env = HeterogeneousMAVUAVAirCombatEnv(v39(), profile="main")
    env.reset(seed=3000)
    before = pickle.dumps(_environment_state(env))
    targets = recompute_reward_targets(env)
    after = pickle.dumps(_environment_state(env))
    assert before == after
    target = targets["UAV1"]
    assert target is not None and reward_target_matches("UAV1", target, targets)
    mismatch = next(bid for bid in ("Blue1", "Blue2", "Blue3", "Blue4") if bid != target)
    assert not reward_target_matches("UAV1", mismatch, targets)
    assert not reward_target_matches("MAV", target, targets)


def test_simultaneous_candidate_attribution_never_fabricates_unique_killer():
    candidates = candidate_attackers(
        [("MAV", "Blue1"), ("UAV1", "Blue1"), ("UAV1", "Blue2")],
        ["Blue1", "Blue2"],
    )
    assert candidates == {"Blue1": ["MAV", "UAV1"], "Blue2": ["UAV1"]}
    assert red_attack_event_pairs([
        {"attacker": "Blue1", "target": "UAV1"},
        {"attacker": "UAV2", "target": "Blue3"},
    ]) == [("UAV2", "Blue3")]


def test_intervention_changes_only_designated_attacker_action():
    actions = {aid: np.full(3, index + 1, np.float32) for index, aid in enumerate(RED_IDS)}
    branches = dict(intervention_action_sets(actions, "UAV2"))
    for aid in RED_IDS:
        if aid != "UAV2":
            np.testing.assert_array_equal(branches["ATTACKER_TRIM_ONE_STEP"][aid], actions[aid])
            np.testing.assert_array_equal(branches["ATTACKER_PEER_ACTION_ONE_STEP"][aid], actions[aid])
    np.testing.assert_array_equal(branches["ATTACKER_TRIM_ONE_STEP"]["UAV2"], np.zeros(3))
    np.testing.assert_array_equal(branches["ATTACKER_PEER_ACTION_ONE_STEP"]["UAV2"], actions["UAV1"])
    assert len(intervention_action_sets(actions, "MAV")) == 2


def test_streak2_snapshot_and_torch_rng_restore_are_exact():
    env = HeterogeneousMAVUAVAirCombatEnv(v39(), profile="main")
    env.reset(seed=3000)
    env._attack_streak[("UAV1", "Blue1")] = 2
    before_env = pickle.dumps(_environment_state(env))
    torch.manual_seed(778); before_rng = torch.get_rng_state().clone()
    actions = {aid: np.zeros(3, np.float32) for aid in RED_IDS}
    rows = evaluate_streak2_opportunity(env, actions, "UAV1", "Blue1", "group")
    assert {row["branch"] for row in rows} == {
        "NORMAL_POLICY", "ATTACKER_TRIM_ONE_STEP", "ATTACKER_PEER_ACTION_ONE_STEP",
    }
    assert pickle.dumps(_environment_state(env)) == before_env
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(all(np.isfinite(value) for value in row.values() if isinstance(value, float)) for row in rows)


def test_policy_episode_exactly_matches_formal_evaluator():
    config = v39()
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=5)
    loaded = {
        "actors": actors, "environment_config": config, "training_seed": 5,
        "sampled_steps": 0, "checkpoint": ROOT / "synthetic/checkpoint.pt",
    }
    audit_result = run_condition_episode(
        loaded, condition="POLICY", profile="main", episode=0,
        environment_seed=3000, action_seed=4000, control_action_seed=14000,
        device="cpu", max_counterfactual_opportunities=0, collect_counterfactuals=False,
    )
    audited = audit_result["episodes"][0]
    formal = evaluate_actors(
        actors, config, 1, "main", seed=3000, device="cpu",
        deterministic=False, action_seed=4000,
    )[0]
    assert audited["outcome"] == formal["outcome"]
    assert audited["episode_return"] == formal["episode_return"]
    assert audited["red_attack_kills"] == formal["red_attack_kills"]
    assert audited["blue_attack_kills"] == formal["blue_attack_kills"]
    assert audited["MAV_survival"] == formal["mav_survived"]
    assert audited["UAV_survivors"] == formal["red_uav_survivors"]
    assert audited["episode_length"] == formal["episode_length"]
    assert all(row["gate"] and row["attack_streak"] >= 3 for row in audit_result["deaths"])
    assert all(
        row["gate"] and row["attack_streak"] >= 3
        for row in audit_result["geometry"] if row["relative_step"] == 0
    )


def test_output_schema_handles_zero_counterfactuals(tmp_path):
    actor_fields = {}
    for aid in RED_IDS:
        actor_fields.update({
            f"{aid}_gate_active_fraction": 0.0, f"{aid}_streak_ge_1_count": 0,
            f"{aid}_streak_ge_2_count": 0, f"{aid}_attack_event_pair_count": 0,
        })
    rows = []
    for condition in CONDITIONS:
        rows.append({
            "checkpoint": "run/checkpoint.pt", "training_seed": 5, "sampled_steps": 1,
            "condition": condition, "episode": 0, "environment_seed": 3000,
            "action_seed": 4000, "control_action_seed": None, "outcome": "draw",
            "episode_return": 0.0, "episode_length": 75, "MAV_survival": True,
            "UAV_survivors": 3, "red_attack_kills": 0, "blue_attack_kills": 0,
            "red_gate_entry_count": 0, "red_gate_active_pair_steps": 0,
            "red_gate_pair_exposures": 1, "red_gate_active_fraction": 0.0,
            "uav_attack_gate_entry_count": 0, "uav_attack_gate_active_pair_steps": 0,
            "uav_attack_gate_pair_exposures": 1, "uav_attack_gate_active_fraction": 0.0,
            "MAV_geometric_gate_entry_count": 0,
            "MAV_geometric_gate_active_pair_steps": 0,
            "MAV_geometric_gate_pair_exposures": 1,
            "MAV_geometric_gate_fraction": 0.0,
            "streak2_opportunities": 0, "attack_event_pair_count": 0,
            "simultaneous_multi_target_red_attack_steps": 0, "ambiguous_blue_death_count": 0,
            "MAV_attack_event_pair_count": 0, "MAV_death_candidate_count": 0,
            "MAV_only_candidate_death_count": 0, "counterfactual_total_opportunities": 0,
            "counterfactual_audited_opportunities": 0, **actor_fields,
        })
    output = tmp_path / "audit"
    write_outputs(output, {
        "episodes": rows, "attacks": [], "deaths": [], "geometry": [], "counterfactuals": [],
    }, {"profile": "main"})
    expected = {
        "condition_episode_records.csv", "condition_summary.csv", "paired_condition_comparison.csv",
        "paired_condition_episode_records.csv",
        "attack_event_records.csv", "death_candidate_records.csv", "prekill_geometry_records.csv",
        "prekill_geometry_summary.csv", "streak2_counterfactual_records.csv",
        "streak2_counterfactual_summary.csv", "combat_skill_audit_summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "combat_skill_audit_summary.json").read_text(encoding="utf-8"))
    assert set(("FACTS FROM CODE", "DESCRIPTIVE BEHAVIORAL EVIDENCE", "PAIRED INTERVENTION EVIDENCE", "LIMITATIONS")) <= set(summary)


def test_curriculum_configs_are_fair_and_original_configs_remain_noncurriculum():
    for version in ("v3_9", "v3_10"):
        configs = validate_configs(version); reference = configs["tacm_rgaa"]
        for config in configs.values():
            assert all(config[field] == reference[field] for field in COMMON_FIELDS)
    for filename in ("happo_rgaa_v39.yaml", "happo_rgaa_wide_v39.yaml", "happo_dbm_rgaa_v39.yaml"):
        original = load_training(ROOT / "configs" / filename)
        assert original["device"] == "cpu"
        assert "randomization_curriculum_enabled" not in original


def test_launcher_contract_and_manifest_fields_are_explicit():
    for filename in (
        "run_role_guided_v39_main_curriculum_1m.sh",
        "run_role_guided_v310_main_curriculum_1m.sh",
    ):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for token in ("rgaa rgaa_wide dbm_rgaa", "for seed in 5 7 9", "--steps 1000000", "--final-eval-episodes 1", "--eval-interval 0"):
            assert token in text
        for entrypoint in (
            "algorithm/train_happo_rgaa.py", "algorithm/train_rgaa_wide.py",
            "algorithm/train_dbm_rgaa.py",
        ):
            assert entrypoint in text
        assert "method,seed,entrypoint,config,environment_config,environment_version,run_name,output_folder,log_file,checkpoint_final" in text
        assert "PIPESTATUS[0]" in text and "exit \"${status}\"" in text
    tacm = (ROOT / "run_tacm_v310_main_curriculum_1m.sh").read_text(encoding="utf-8")
    assert "algorithm/train_tacm_rgaa.py" in tacm and "for seed in 5 7 9" in tacm
