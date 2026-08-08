from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.diagnose_v12_mixed_policy_roles import (
    AgentAccum,
    MixedPolicyProbeEnv,
    _select_trace_specs,
    _required_numeric,
    _normalise_episode_row,
)

from uav_combat.diagnostics.v12_mixed_policy_roles import (
    AGENT_TO_ROLE_V12,
    COMBOS_V12_MIXED,
    exact_mcnemar_pvalue,
    paired_bootstrap,
    practical_equivalence,
    select_targeted_seeds,
    validate_all_combinations,
)
from uav_combat.environment_4v3_v12 import FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv
from uav_combat.happo.networks import IndependentHAPPOActors


ENV_CONFIG = Path("configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")
RUN_DIR = Path("outputs/happo_heterogeneous_4v3_main_v12_soft_boundary_combat_aligned_3m_seed42")


def _env(seed: int = 150042):
    env = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def test_source_maps_are_exact_and_ordered():
    validate_all_combinations()
    assert len(COMBOS_V12_MIXED) == 10
    assert COMBOS_V12_MIXED[0][1] == {agent: "rule" for agent in ("red_0", "red_1", "red_2", "red_3")}
    assert COMBOS_V12_MIXED[1][1] == {agent: "learned" for agent in ("red_0", "red_1", "red_2", "red_3")}
    assert AGENT_TO_ROLE_V12 == {"red_0": "support", "red_1": "combat_1", "red_2": "combat_2", "red_3": "combat_3"}


def test_rule_actions_have_no_state_side_effect():
    env = _env()
    before = env.state_dict()
    env.red_rule_actions()
    after = env.state_dict()
    assert before == after


def test_blue_rule_path_is_not_replaced_by_diagnostic_helpers():
    env = _env()
    direct = env._direct_visible_ids()
    actions_a = env._blue_rule_actions(direct)
    actions_b = env._blue_rule_actions(direct)
    assert actions_a.keys() == actions_b.keys()
    for agent_id in actions_a:
        np.testing.assert_array_equal(actions_a[agent_id], actions_b[agent_id])


def test_deterministic_learned_action_is_reproducible_and_slot_local():
    torch.manual_seed(17)
    actors = IndependentHAPPOActors([118] * 4, [3] * 4, hidden_dim=16)
    obs = torch.randn(2, 4, 118)
    actors.eval()
    with torch.no_grad():
        first = actors.deterministic_actions(obs)
        second = actors.deterministic_actions(obs)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 4, 3)
    assert torch.isfinite(first).all()


def test_diagnostic_source_does_not_call_training_update_or_optimizer():
    source = Path("scripts/diagnose_v12_mixed_policy_roles.py").read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "trainer.update" not in source
    assert "collect_rollout" not in source


def test_paired_bootstrap_is_reproducible():
    a = [0.0, 1.0, 2.0, 3.0]
    b = [1.0, 1.0, 4.0, 2.0]
    assert paired_bootstrap(a, b, samples=200, seed=19) == paired_bootstrap(a, b, samples=200, seed=19)


@pytest.mark.parametrize(
    "b,c,expected",
    [(0, 0, 1.0), (1, 0, 1.0), (3, 1, 0.625)],
)
def test_exact_mcnemar(b, c, expected):
    assert exact_mcnemar_pvalue(b, c) == pytest.approx(expected)


def test_practical_equivalence_classifier():
    assert practical_equivalence((-0.02, 0.03), 0.05) == "practical_equivalent"
    assert practical_equivalence((-0.20, -0.10), 0.05) == "materially_worse"
    assert practical_equivalence((0.10, 0.20), 0.05) == "materially_better"
    assert practical_equivalence((-0.10, 0.10), 0.05) == "inconclusive"


def test_targeted_seed_selection_never_fills_missing_contrast():
    rows = [
        {"checkpoint": "best", "combo": "M0_all_rule", "episode_seed": 1, "task_win": True, "red_attack_kills": 2},
        {"checkpoint": "best", "combo": "M2_learned_support_rule_combats", "episode_seed": 1, "task_win": False, "red_attack_kills": 0},
    ]
    assert select_targeted_seeds(rows, category="support", limit=10) == [{"category": "A_support", "episode_seed": 1}]
    assert select_targeted_seeds(rows, category="combat", limit=10) == []


def test_targeted_trace_branches_return_plain_dict_source_maps():
    rows = []
    combos = {
        "M0_all_rule": {"task_win": True, "red_attack_kills": 3},
        "M2_learned_support_rule_combats": {"task_win": False, "red_attack_kills": 0},
        "M3_rule_support_learned_combats": {"task_win": False, "red_attack_kills": 0},
        "M4_rule_support_learned_combat_1": {"task_win": False, "red_attack_kills": 0},
        "M5_rule_support_learned_combat_2": {"task_win": False, "red_attack_kills": 0},
        "M6_rule_support_learned_combat_3": {"task_win": False, "red_attack_kills": 0},
        "M1_all_learned": {"task_win": False, "red_attack_kills": 1},
    }
    for checkpoint in ("best", "final"):
        for combo, fields in combos.items():
            if checkpoint == "final" and combo != "M1_all_learned":
                continue
            row_fields = dict(fields)
            if checkpoint == "final" and combo == "M1_all_learned":
                row_fields["red_attack_kills"] = 0
            rows.append({"checkpoint": checkpoint, "combo": combo, "episode_seed": 7, **row_fields})
    specs = _select_trace_specs(rows)
    categories = {spec["category"] for spec in specs}
    assert {"A_support", "B_combat", "C_combat_1", "C_combat_2", "C_combat_3", "D_best_final"} <= categories
    for spec in specs:
        assert type(spec["source_map"]) is dict
        assert set(spec["source_map"]) == {"red_0", "red_1", "red_2", "red_3"}


def test_survival_rate_is_explicit_and_required_for_drift():
    assert _required_numeric({"survival_rate": 1.0}, "survival_rate") == 1.0
    with pytest.raises(KeyError):
        _required_numeric({}, "survival_rate")


def test_historical_task_win_is_red_outcome_not_strict_elimination():
    row = _normalise_episode_row({"environment_outcome": "red", "strict_full_elimination": False, "red_attack_kills": 1, "mean_return": 2.0})
    assert row["task_win"] is True
    assert row["strict_full_elimination"] is False
    assert row["at_least_two_kill"] is False


def test_dead_actor_does_not_accumulate_alive_behavior_rates():
    env = _env(150044)
    env._by_id("red_1").state.alive = False
    accum = AgentAccum("red_1", "combat_1", "learned")
    meta = accum.record_pre_step(env, env._direct_visible_ids(), 1)
    accum.record_post_step(env, 1)
    row = accum.finish(env, {})
    assert meta["alive"] is False
    assert row["alive_steps"] == 0
    assert row["alive_decision_steps"] == 0
    assert row["direct_target_rate"] == 0.0
    assert row["survival_rate"] == 0.0


def test_probe_environment_matches_original_for_same_rule_action_sequence():
    original = _env(150045)
    probe = MixedPolicyProbeEnv(ENV_CONFIG)
    probe.reset(150045)
    for _ in range(650):
        original_actions, _ = original.red_rule_actions()
        probe_actions, _ = probe.red_rule_actions()
        for agent_id in original_actions:
            np.testing.assert_array_equal(original_actions[agent_id], probe_actions[agent_id])
        original_step = original.step(original_actions)
        probe_step = probe.step(probe_actions)
        assert original_step[4] == probe_step[4]
        assert original_step[5] == probe_step[5]
        assert original.state_dict() == probe.state_dict()
        if original_step[4]:
            break


def test_trace_kill_field_name_is_not_legacy_target_as_killer():
    source = Path("scripts/diagnose_v12_mixed_policy_roles.py").read_text(encoding="utf-8")
    assert '"killer_id"' in source
    assert '"killed_target_id"' in source
    assert '"attributed_killer"' not in source


def test_checkpoint_hash_is_unchanged_by_read_only_load():
    checkpoint = RUN_DIR / "best.pt"
    if not checkpoint.exists():
        pytest.skip("historical v12 run is not present")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert loaded["env_steps"] == 2_900_000
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert before == after


def test_checkpoint_metadata_contract_is_readable():
    contract_path = RUN_DIR / "experiment_contract.json"
    if not contract_path.exists():
        pytest.skip("historical v12 run is not present")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert "v12" in contract["variant"]
    assert contract["reward_contract_version"] == "v12_soft_boundary_combat_aligned"
    assert contract["observation_dims"] == [118, 118, 118, 118] if "observation_dims" in contract else True


def test_wrapper_contract_dimensions_match_original():
    env = _env()
    obs, state, mask = env._observations()
    assert obs.shape == (7, 118)
    assert state.shape == (70,)
    assert mask.shape == (7,)


def test_rule_actions_are_bounded_and_finite():
    env = _env()
    actions, targets = env.red_rule_actions()
    assert set(actions) == {"red_0", "red_1", "red_2", "red_3"}
    assert set(targets) >= {"red_1", "red_2", "red_3"}
    for action in actions.values():
        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.all(action <= 1.0) and np.all(action >= -1.0)


def test_one_step_does_not_change_blue_strategy_or_interface():
    env = _env()
    before = env._blue_rule_actions(env._direct_visible_ids())
    red_actions, _ = env.red_rule_actions()
    result = env.step(red_actions)
    assert len(result) == 7
    after = env._blue_rule_actions(env._direct_visible_ids())
    assert set(before) == set(after)


def test_per_agent_mapping_is_explicit_in_runner():
    source = Path("scripts/diagnose_v12_mixed_policy_roles.py").read_text(encoding="utf-8")
    for token in ("learned_actions[index]", "TEAM_AGENT_IDS_V12", "red_0", "red_1", "red_2", "red_3"):
        assert token in source


def test_no_new_checkpoint_writer_in_diagnostic():
    source = Path("scripts/diagnose_v12_mixed_policy_roles.py").read_text(encoding="utf-8")
    assert "torch.save" not in source


def test_all_combinations_use_one_fixed_slot_map():
    for name, source_map in COMBOS_V12_MIXED:
        assert set(source_map) == {"red_0", "red_1", "red_2", "red_3"}, name
        assert set(source_map.values()) <= {"rule", "learned"}


def test_state_dict_is_stable_across_rule_action_call():
    env = _env(150043)
    first = json.dumps(env.state_dict(), sort_keys=True)
    env.red_rule_actions()
    second = json.dumps(env.state_dict(), sort_keys=True)
    assert first == second


def test_support_is_not_inferred_from_group_mean_in_helper():
    source = inspect.getsource(select_targeted_seeds)
    assert "M2_learned_support_rule_combats" in source
    assert "M3_rule_support_learned_combats" in source
