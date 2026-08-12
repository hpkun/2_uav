from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from uav_combat.environment_4v3_v11 import lock_quality_v11
from uav_combat.environment_4v3_v12 import DEATH_LOCK_V12
from uav_combat.environment_4v3_v15 import (
    AGENT_REWARD_COMPONENT_KEYS_V15,
    FunctionalHeterogeneous4v3V15PaperCompactRewardEnv,
    angle_state_score_v15,
    combat_state_reward_v15,
    distance_state_score_v15,
    support_state_reward_v15,
)
from uav_combat.happo.evaluation_v14_4v3 import evaluate_v14_happo_fixed_blue_4v3
from uav_combat.happo.trainer_v14_4v3 import (
    CREDIT_MODE_ROLE_LOCAL,
    MissionAlignedRoleSharedHAPPO4v3Trainer,
)
from uav_combat.happo.trainer_4v3 import (
    V15_BEST_SCORE_FIELDS_4V3,
    V15_REWARD_CONTRACT_VERSION,
    compute_best_score_4v3,
    compute_best_score_v15_4v3,
    compute_experiment_best_score_4v3,
)
from uav_combat.mappo.vector_env_4v3_v14 import make_combat_vector_env_4v3_v14
from uav_combat.models import AircraftState
from scripts.train_happo_role_shared_4v3 import (
    _build_experiment_contract,
    _format_update_line,
    _update_best_from_latest_evaluation,
)
from uav_combat.scenario_4v3_v15 import (
    BLUE_IDS_V15,
    RED_IDS_V15,
    REWARD_COMPONENT_KEYS_V15,
    resolved_reward_contract_v15,
)

ROOT = Path(__file__).resolve().parents[1]
ENV_V14 = ROOT / "configs/heterogeneous_4v3_main_v14_mission_aligned_role_credit.yaml"
ENV_V15 = ROOT / "configs/heterogeneous_4v3_main_v15_paper_compact_attack_reward.yaml"
TRAIN_V15 = ROOT / "configs/happo_heterogeneous_4v3_main_v15_role_shared_combat_mlp_paper_compact_reward.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny(*, num_envs: int = 2, rollout_steps: int = 8) -> dict:
    cfg = _config(TRAIN_V15)
    cfg["experiment"]["device"] = "cpu"
    cfg["training"].update(
        {
            "num_envs": num_envs,
            "num_env_workers": 0,
            "rollout_steps": rollout_steps,
            "total_env_steps": num_envs * rollout_steps,
            "schedule_env_steps": num_envs * rollout_steps,
            "ppo_epochs": 1,
            "minibatch_size": num_envs * rollout_steps,
        }
    )
    cfg["evaluation"]["selection_episodes"] = 2
    cfg["evaluation"]["test_episodes"] = 2
    return cfg


def _reward_call(
    env: FunctionalHeterogeneous4v3V15PaperCompactRewardEnv,
    *,
    deaths: dict[str, int] | None = None,
    killers: dict[str, str] | None = None,
    direct: dict[str, set[str]] | None = None,
):
    return env._compute_reward(
        {}, {}, {}, deaths or {}, set(), killers or {},
        direct or env._direct_visible_ids(),
    )


def test_v15_angle_diagnostic_uses_real_lock_fade_thresholds():
    profile = _config(ENV_V15)["combat_profile"]
    assert angle_state_score_v15(0.0, 0.0, profile) == 1.0
    assert angle_state_score_v15(profile["lock_ata_fade_max"], 0.0, profile) == -1.0
    assert angle_state_score_v15(0.0, profile["lock_aa_fade_max"], profile) == -1.0


def test_v15_freezes_all_v14_nonreward_environment_fields():
    v14 = _config(ENV_V14)
    v15 = _config(ENV_V15)
    assert v15["simulation"] == v14["simulation"]
    assert v15["action"] == v14["action"]
    assert v15["aircraft"] == v14["aircraft"]
    assert v15["battlefield"] == v14["battlefield"]
    assert v15["boundary"] == v14["boundary"]
    assert v15["combat_profile"] == v14["combat_profile"]
    assert v15["scenario"] == v14["scenario"]
    assert v15["heterogeneous"] == v14["heterogeneous"]
    assert v15["blue_rule_policy"] == v14["blue_rule_policy"]
    assert v15["red_rule_policy"] == v14["red_rule_policy"]


def test_v15_distance_state_reuses_v11_curve():
    profile = _config(ENV_V15)["combat_profile"]
    assert distance_state_score_v15(1000.0, profile) == 1.0
    assert distance_state_score_v15(2500.0, profile) == -1.0


def test_v15_combat_state_formula_and_missing_target_penalty():
    assert combat_state_reward_v15(1.0) == pytest.approx(0.02)
    assert combat_state_reward_v15(0.5) == pytest.approx(0.0)
    assert combat_state_reward_v15(0.0) == pytest.approx(-0.02)
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(1)
    env.targets["red_1"] = None
    _, components = _reward_call(env)
    rewards = env._last_agent_rewards
    row = env._last_agent_reward_components["red_1"]
    assert row["angle_state_reward"] == -1.0
    assert row["distance_state_reward"] == -1.0
    assert row["combat_state_reward"] == -0.02
    assert rewards["red_1"] == pytest.approx(-0.02)
    assert components["combat_state_reward"] < 0.0


def _state(x: float, *, psi: float = 0.0) -> AircraftState:
    return AircraftState(x=x, y=0.0, z=-5000.0, v=250.0, theta=0.0, psi=psi)


@pytest.mark.parametrize(
    ("attacker", "target"),
    [
        # ATA reaches its fade maximum at optimal distance.
        (_state(0.0, psi=np.pi / 2.0), _state(1000.0, psi=0.0)),
        # AA reaches pi at optimal distance, beyond its fade maximum.
        (_state(0.0, psi=0.0), _state(1000.0, psi=np.pi)),
        # Perfect angles but distance reaches the fade maximum.
        (_state(0.0, psi=0.0), _state(2500.0, psi=0.0)),
    ],
)
def test_v15_zero_real_lock_quality_always_has_negative_dense_reward(
    attacker: AircraftState, target: AircraftState
):
    profile = _config(ENV_V15)["combat_profile"]
    quality = lock_quality_v11(attacker, target, profile)
    assert quality == pytest.approx(0.0, abs=1e-10)
    assert combat_state_reward_v15(quality) == pytest.approx(-0.02)
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(103)
    env._by_id("red_1").state = attacker
    env._by_id("blue_0").state = target
    env.targets["red_1"] = "blue_0"
    _reward_call(env)
    assert env._last_agent_reward_components["red_1"][
        "combat_state_reward"
    ] == pytest.approx(-0.02)


def test_v15_perfect_geometry_has_full_lock_quality_and_dense_reward():
    profile = _config(ENV_V15)["combat_profile"]
    quality = lock_quality_v11(_state(0.0), _state(1000.0), profile)
    assert quality == pytest.approx(1.0)
    assert combat_state_reward_v15(quality) == pytest.approx(0.02)


def test_v15_environment_calls_lock_quality_source_of_truth(monkeypatch):
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(104)
    env.targets["red_1"] = "blue_0"
    monkeypatch.setattr(
        "uav_combat.environment_4v3_v15.lock_quality_v11",
        lambda attacker, target, profile: 0.25,
    )
    _reward_call(env)
    assert env._last_agent_reward_components["red_1"][
        "combat_state_reward"
    ] == pytest.approx(-0.01)


def test_v15_dead_target_has_missing_target_penalty():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(101)
    target = env.targets["red_1"]
    assert target is not None
    env._by_id(target).state.alive = False
    _reward_call(env)
    row = env._last_agent_reward_components["red_1"]
    assert row["combat_state_reward"] == pytest.approx(-0.02)


def test_v15_agent_total_does_not_add_diagnostics_twice():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(102)
    _reward_call(env)
    row = env._last_agent_reward_components["red_1"]
    expected = sum(
        row[key]
        for key in (
            "combat_state_reward",
            "own_kill_reward",
            "team_kill_reward",
            "death_penalty",
            "boundary_penalty",
            "support_state_reward",
        )
    )
    assert row["agent_total_reward"] == pytest.approx(expected)


def test_v15_dead_combat_has_zero_dense_state_reward():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(2)
    env._by_id("red_1").state.alive = False
    _reward_call(env)
    row = env._last_agent_reward_components["red_1"]
    assert row["combat_state_reward"] == 0.0


def test_v15_killer_and_team_kill_credit_exactly():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(3)
    _reward_call(env, killers={"blue_0": "red_2"})
    rewards = env._last_agent_rewards
    rows = env._last_agent_reward_components
    assert rows["red_2"]["own_kill_reward"] == 8.0
    assert all(rows[agent]["team_kill_reward"] == 1.0 for agent in RED_IDS_V15)
    assert rewards["red_2"] - rows["red_2"]["combat_state_reward"] == 9.0
    assert rows["red_1"]["own_kill_reward"] == 0.0
    assert rows["red_0"]["team_kill_reward"] == 1.0


def test_v15_two_kills_give_two_team_credit():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(4)
    _reward_call(env, killers={"blue_0": "red_1", "blue_1": "red_2"})
    assert all(
        row["team_kill_reward"] == 2.0
        for row in env._last_agent_reward_components.values()
    )


@pytest.mark.parametrize("agent_id", ["red_0", "red_1"])
def test_v15_death_penalty_is_agent_local_and_equal_scale(agent_id: str):
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(5)
    _reward_call(env, deaths={agent_id: DEATH_LOCK_V12})
    rows = env._last_agent_reward_components
    assert rows[agent_id]["death_penalty"] == -4.0
    assert all(
        rows[other]["death_penalty"] == 0.0
        for other in RED_IDS_V15 if other != agent_id
    )


def test_v15_hard_contact_is_agent_local_without_cap():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(6)
    env._episode_metrics["red_3_boundary_hard_contacts_step"] = 2.0
    _reward_call(env)
    rows = env._last_agent_reward_components
    assert rows["red_3"]["boundary_penalty"] == pytest.approx(-0.2)
    assert rows["red_0"]["boundary_penalty"] == 0.0


def test_v15_support_state_position_awareness_formula(monkeypatch):
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(7)
    monkeypatch.setattr(env, "_formation_score", lambda: 0.75)
    direct = {agent: set() for agent in (*RED_IDS_V15, *BLUE_IDS_V15)}
    direct["red_0"] = {"blue_0", "blue_1"}
    _reward_call(env, direct=direct)
    row = env._last_agent_reward_components["red_0"]
    expected_position = 0.5
    expected_awareness = 2.0 * (2.0 / 3.0) - 1.0
    assert row["support_position_state_reward"] == expected_position
    assert row["support_awareness_state_reward"] == pytest.approx(expected_awareness)
    assert row["support_state_reward"] == pytest.approx(
        support_state_reward_v15(expected_position, expected_awareness)
    )


def test_v15_awareness_zero_when_all_blue_dead(monkeypatch):
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(8)
    for blue_id in BLUE_IDS_V15:
        env._by_id(blue_id).state.alive = False
    monkeypatch.setattr(env, "_formation_score", lambda: 0.5)
    _reward_call(env, direct={agent: set() for agent in (*RED_IDS_V15, *BLUE_IDS_V15)})
    row = env._last_agent_reward_components["red_0"]
    assert row["support_awareness_state_reward"] == 0.0


def test_v15_team_reporting_reward_is_agent_arithmetic_mean():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(9)
    _, components = _reward_call(env, killers={"blue_0": "red_2"})
    rewards = env._last_agent_rewards
    assert components["team_total_reward"] == pytest.approx(
        np.mean(list(rewards.values()))
    )


def test_v15_terminal_rewards_are_all_zero_and_inactive():
    contract = resolved_reward_contract_v15(_config(ENV_V15))
    assert set(contract["mission"].values()) == {0.0}
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(10)
    before, _ = _reward_call(env)
    # Terminal outcome is intentionally absent from the _compute_reward inputs.
    after, _ = _reward_call(env)
    assert before == after


def test_v15_active_component_contract_contains_no_legacy_shaping():
    assert REWARD_COMPONENT_KEYS_V15 == (
        "support_state_reward",
        "combat_state_reward",
        "own_kill_reward",
        "team_kill_reward",
        "death_penalty",
        "boundary_penalty",
        "team_total_reward",
    )
    forbidden = {
        "mission_outcome_reward",
        "combat_geometry_progress_reward",
        "combat_lock_progress_reward",
        "combat_half_lock_event_reward",
        "support_unique_detection_reward",
        "support_cue_to_direct_reward",
        "support_cue_to_half_lock_reward",
        "support_assisted_kill_reward",
        "support_formation_progress_reward",
    }
    assert forbidden.isdisjoint(REWARD_COMPONENT_KEYS_V15)


def test_v15_log_only_events_do_not_change_active_reward():
    env = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    env.reset(11)
    baseline, _ = _reward_call(env)
    env._episode_metrics.update(
        {
            "support_unique_detection_events_step": 5.0,
            "support_cue_to_direct_events_step": 4.0,
            "support_cue_to_half_lock_events_step": 3.0,
            "support_assisted_kills": 2.0,
        }
    )
    changed, _ = _reward_call(env)
    assert changed == baseline


@pytest.mark.parametrize("workers", [0, 1])
def test_v15_vector_rewards_and_components_are_finite(workers: int):
    envs = make_combat_vector_env_4v3_v14(ENV_V15, 2, workers, 12)
    try:
        result = envs.step(np.zeros((2, 4, 3), np.float32))
        assert result.agent_rewards.shape == (2, 4)
        assert result.red_reward_components.shape == (
            2, len(REWARD_COMPONENT_KEYS_V15)
        )
        assert result.red_agent_reward_components.shape == (
            2, 4, len(AGENT_REWARD_COMPONENT_KEYS_V15)
        )
        assert np.isfinite(result.team_rewards).all()
        assert np.isfinite(result.agent_rewards).all()
        assert np.isfinite(result.red_agent_reward_components).all()
    finally:
        envs.close()


def test_v15_uses_unchanged_v14b_role_local_trainer():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V15, _tiny())
    try:
        assert trainer.credit_mode == CREDIT_MODE_ROLE_LOCAL
        assert trainer.config["training"]["team_reward_usage"] == "reporting_only"
        assert len(trainer.actor_optimizers) == 2
        trainer.collect_rollout()
        assert trainer.buffer.advantages.shape == (8, 2, 4)
        metrics = trainer.update()
        assert np.isfinite(
            [float(value) for value in metrics.values() if isinstance(value, (int, float))]
        ).all()
        assert "mean_rollout_support_state_reward" in metrics
        assert "mean_rollout_combat_state_reward" in metrics
    finally:
        trainer.close()


def test_v15_checkpoint_resume_roundtrip(tmp_path: Path):
    cfg = _tiny(num_envs=1, rollout_steps=2)
    source = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V15, cfg)
    restored = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V15, cfg)
    path = tmp_path / "v15.pt"
    try:
        source.collect_rollout()
        source.update()
        source.save_checkpoint(path)
        expected = source._select_actions()[:3]
        restored.load_checkpoint(path)
        actual = restored._select_actions()[:3]
        for left, right in zip(expected, actual):
            assert np.array_equal(left, right)
    finally:
        source.close()
        restored.close()


def test_v14_checkpoint_cannot_load_into_v15(tmp_path: Path):
    v15 = MissionAlignedRoleSharedHAPPO4v3Trainer(
        ENV_V15, _tiny(num_envs=1, rollout_steps=2)
    )
    path = tmp_path / "fake_v14.pt"
    payload = {
        "checkpoint_family": "functional_heterogeneous_4v3_mission_aligned_role_shared_happo",
        "training_signature": {
            **v15.training_signature(),
            "reward_contract_version": "v14_mission_aligned_role_credit",
        },
    }
    torch.save(payload, path)
    try:
        with pytest.raises(ValueError, match="training signature mismatch"):
            v15.load_checkpoint(path)
    finally:
        v15.close()


def test_v15_deterministic_evaluation_repeats_and_reports_reward_components():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(
        ENV_V15, _tiny(num_envs=1, rollout_steps=2)
    )
    try:
        one = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V15, seeds=[8001], num_envs=1, device="cpu"
        )
        two = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V15, seeds=[8001], num_envs=1, device="cpu"
        )
        assert one["team_total_reward"] == two["team_total_reward"]
        assert one["mean_support_state_reward"] == two["mean_support_state_reward"]
        assert one["mean_combat_state_reward"] == two["mean_combat_state_reward"]
        assert one["mean_red_0_agent_return"] == two["mean_red_0_agent_return"]
    finally:
        trainer.close()


def _selection_summary(**overrides: float) -> dict[str, float]:
    summary = {key: 0.0 for key in V15_BEST_SCORE_FIELDS_4V3}
    summary.update(overrides)
    return summary


def test_v15_selector_ignores_timeout_task_win_and_prefers_real_attack():
    timeout = _selection_summary(task_win_rate=0.10)
    attack = _selection_summary(any_kill_rate=0.20)
    assert compute_best_score_v15_4v3(attack)[0] > compute_best_score_v15_4v3(timeout)[0]


@pytest.mark.parametrize(
    ("field", "prior"),
    [
        ("strict_full_elimination_rate", []),
        ("at_least_two_kill_rate", ["strict_full_elimination_rate"]),
        ("any_kill_rate", ["strict_full_elimination_rate", "at_least_two_kill_rate"]),
        ("mean_red_kills", ["strict_full_elimination_rate", "at_least_two_kill_rate", "any_kill_rate"]),
        ("half_lock_episode_rate", ["strict_full_elimination_rate", "at_least_two_kill_rate", "any_kill_rate", "mean_red_kills"]),
        ("mean_max_lock_progress", list(V15_BEST_SCORE_FIELDS_4V3[:-1])),
    ],
)
def test_v15_selector_lexicographic_priority(field: str, prior: list[str]):
    lower = _selection_summary(**{name: 0.5 for name in prior})
    higher = dict(lower)
    higher[field] = 0.1
    assert compute_best_score_v15_4v3(higher)[0] > compute_best_score_v15_4v3(lower)[0]


@pytest.mark.parametrize(
    "ignored",
    ["mean_episode_length", "team_total_reward", "mean_return", "mean_red_combat_survivors", "support_survival_rate", "task_win_rate"],
)
def test_v15_selector_ignores_non_attack_fields(ignored: str):
    baseline = _selection_summary()
    changed = {**baseline, ignored: 999.0}
    assert compute_best_score_v15_4v3(changed)[0] == compute_best_score_v15_4v3(baseline)[0]


def test_selector_dispatch_uses_v15_only_for_v15_contract():
    summary = _selection_summary(task_win_rate=0.8, any_kill_rate=0.2)
    v15 = compute_experiment_best_score_4v3(summary, V15_REWARD_CONTRACT_VERSION)
    historical = compute_experiment_best_score_4v3(summary, "v14_mission_aligned_role_credit")
    assert v15 == compute_best_score_v15_4v3(summary)
    assert historical == compute_best_score_4v3(summary)
    assert "task_win_rate" not in v15[1]
    assert "task_win_rate" in historical[1]


def test_v15_initial_and_periodic_best_updates_use_dispatched_score_fields():
    trainer = SimpleNamespace(
        evaluation_history=[], best_score=None, best_score_fields={},
        best_evaluation=None, best_checkpoint_name=None,
        best_scheduled_env_steps=None, best_actual_env_steps=None, env_steps=0,
    )
    initial = _selection_summary(task_win_rate=1.0)
    initial_score, initial_fields = compute_experiment_best_score_4v3(
        initial, V15_REWARD_CONTRACT_VERSION
    )
    trainer.evaluation_history.append(
        {"score": list(initial_score), "score_fields": initial_fields, "summary": initial}
    )
    assert _update_best_from_latest_evaluation(trainer, Path("initial.pt"), 0)
    assert tuple(trainer.best_score_fields) == V15_BEST_SCORE_FIELDS_4V3
    assert "task_win_rate" not in trainer.best_score_fields

    trainer.env_steps = 100_000
    periodic = _selection_summary(any_kill_rate=0.2)
    periodic_score, periodic_fields = compute_experiment_best_score_4v3(
        periodic, V15_REWARD_CONTRACT_VERSION
    )
    trainer.evaluation_history.append(
        {"score": list(periodic_score), "score_fields": periodic_fields, "summary": periodic}
    )
    assert _update_best_from_latest_evaluation(trainer, Path("step_0100000.pt"), 100_000)
    assert trainer.best_checkpoint_name == "step_0100000.pt"
    assert trainer.best_evaluation is periodic


def test_v15_experiment_contract_preregisters_selector_but_v14_does_not():
    cfg = _tiny()
    env_v15 = _config(ENV_V15)
    common = dict(
        checkpoint_family="family", cfg=cfg, env_config=str(ENV_V15),
        train_config=str(TRAIN_V15), manifest_hash="hash",
    )
    v15 = _build_experiment_contract(env_cfg=env_v15, **common)
    assert v15["best_checkpoint_selection"] == list(V15_BEST_SCORE_FIELDS_4V3)
    env_v14 = _config(ENV_V14)
    v14 = _build_experiment_contract(env_cfg=env_v14, **common)
    assert "best_checkpoint_selection" not in v14


def test_v15_terminal_line_uses_existing_team_reward_metric_and_v14_is_safe():
    trainer = SimpleNamespace(env_steps=2048, update_count=1)
    metrics = {
        "group_update_order": "support>combat",
        "support_kl": 0.01,
        "combat_joint_kl": 0.02,
        "entropy": 1.25,
        "mean_rollout_team_total_reward": -0.01234,
        "mean_rollout_combat_state_reward": -0.005,
        "mean_rollout_support_state_reward": 0.004,
    }
    line = _format_update_line(trainer, 8192, metrics, is_v15=True)
    assert "reward=-0.01234" in line
    assert "combat_state_r=-0.00500" in line
    assert "support_state_r=0.00400" in line
    historical = _format_update_line(
        trainer,
        8192,
        {key: value for key, value in metrics.items() if not key.startswith("mean_rollout")},
        is_v15=False,
    )
    assert "reward=" not in historical
