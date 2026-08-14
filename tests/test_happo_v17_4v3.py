from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.train_happo_role_shared_4v3 import (
    _build_experiment_contract,
    _format_update_line,
)
from uav_combat.environment_4v3_v16 import (
    FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv,
)
from uav_combat.environment_4v3_v17 import (
    COMBAT_ACTIVE_REWARD_KEYS_V17,
    SUPPORT_ACTIVE_REWARD_KEYS_V17,
    FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv,
    combat_angle_score_v17,
    combat_distance_score_v17,
    combat_situation_score_v17,
    situation_reward_v17,
    support_situation_score_v17,
)
from uav_combat.happo.evaluation_v14_4v3 import (
    evaluate_v14_happo_fixed_blue_4v3,
)
from uav_combat.happo.trainer_4v3 import (
    V15_BEST_SCORE_FIELDS_4V3,
    V17_REWARD_CONTRACT_VERSION,
    compute_best_score_v15_4v3,
    compute_experiment_best_score_4v3,
)
from uav_combat.happo.trainer_v14_4v3 import (
    MissionAlignedRoleSharedHAPPO4v3Trainer,
)
from uav_combat.mappo.vector_env_4v3_v14 import (
    make_combat_vector_env_4v3_v14,
)

ROOT = Path(__file__).resolve().parents[1]
ENV_V15 = ROOT / "configs/heterogeneous_4v3_main_v15_paper_compact_attack_reward.yaml"
ENV_V16A = ROOT / "configs/heterogeneous_4v3_main_v16a_positive_lock_quality_reward.yaml"
ENV_V16B = ROOT / "configs/heterogeneous_4v3_main_v16b_positive_lock_quality_canonical_obs.yaml"
ENV_V17 = ROOT / "configs/heterogeneous_4v3_main_v17_role_situation_event_mission_reward.yaml"
TRAIN_V15 = ROOT / "configs/happo_heterogeneous_4v3_main_v15_role_shared_combat_mlp_paper_compact_reward.yaml"
TRAIN_V16A = ROOT / "configs/happo_heterogeneous_4v3_main_v16a_role_shared_combat_mlp_positive_lock_quality_reward.yaml"
TRAIN_V16B = ROOT / "configs/happo_heterogeneous_4v3_main_v16b_role_shared_combat_mlp_positive_lock_quality_canonical_obs.yaml"
TRAIN_V17 = ROOT / "configs/happo_heterogeneous_4v3_main_v17_role_shared_combat_mlp_role_situation_event_mission_reward.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny(path: Path) -> dict:
    config = _config(path)
    config["experiment"]["device"] = "cpu"
    config["training"].update(
        {
            "num_envs": 1,
            "num_env_workers": 0,
            "rollout_steps": 2,
            "total_env_steps": 2,
            "schedule_env_steps": 2,
            "ppo_epochs": 1,
            "minibatch_size": 2,
        }
    )
    config["evaluation"]["selection_episodes"] = 1
    config["evaluation"]["test_episodes"] = 1
    return config


def _reward_call(
    env,
    *,
    deaths=None,
    killers=None,
    direct=None,
    pre_locks=None,
    half_events=None,
):
    return env._compute_reward(
        {},
        {},
        pre_locks or {},
        deaths or {},
        half_events or set(),
        killers or {},
        direct if direct is not None else env._direct_visible_ids(),
    )


@pytest.mark.parametrize(
    ("ata", "aa", "expected"),
    [
        (0.0, 0.0, 1.0),
        (math.pi / 2.0, math.pi / 2.0, 0.0),
        (math.pi, math.pi, -1.0),
    ],
)
def test_v17_angle_formula(ata, aa, expected):
    assert combat_angle_score_v17(ata, aa) == pytest.approx(expected)


@pytest.mark.parametrize(("source", "expected"), [(1.0, 1.0), (0.5, 0.0), (0.0, -1.0)])
def test_v17_distance_directly_reuses_v11(monkeypatch, source, expected):
    calls = []

    def fake(distance, profile):
        calls.append((distance, profile))
        return source

    monkeypatch.setattr("uav_combat.environment_4v3_v17.distance_score_v11", fake)
    profile = {"sentinel": 17}
    assert combat_distance_score_v17(123.0, profile) == pytest.approx(expected)
    assert calls == [(123.0, profile)]


@pytest.mark.parametrize(
    ("angle", "distance", "score", "reward"),
    [(1.0, 1.0, 1.0, 0.01), (-1.0, -1.0, -1.0, -0.01), (0.0, 0.0, 0.0, 0.0)],
)
def test_v17_situation_formulas(angle, distance, score, reward):
    actual = combat_situation_score_v17(angle, distance)
    assert actual == pytest.approx(score)
    assert situation_reward_v17(actual) == pytest.approx(reward)
    assert support_situation_score_v17(angle, distance) == pytest.approx(score)


def test_v17_config_freezes_v16a_except_contract_and_reward():
    baseline = _config(ENV_V16A)
    candidate = _config(ENV_V17)
    frozen = deepcopy(candidate)
    frozen["combat"] = deepcopy(baseline["combat"])
    frozen["rewards"] = deepcopy(baseline["rewards"])
    assert frozen == baseline
    assert candidate["combat"]["observation_contract"] == "legacy_fixed_order"
    train = _config(TRAIN_V17)
    assert train["training"]["total_env_steps"] == 3_000_000
    assert train["training"]["schedule_env_steps"] == 3_000_000
    assert train["training"]["num_envs"] == 16
    assert train["training"]["num_env_workers"] == 4


def test_v17_observation_is_bitwise_v16a():
    old = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    new = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    old_values = old.reset(1701)
    new_values = new.reset(1701)
    assert all(np.array_equal(a, b) for a, b in zip(old_values, new_values))
    assert new_values[0].shape == (7, 118)
    assert new_values[1].shape == (70,)


def test_v17_no_target_dead_target_and_dead_combat_have_zero_situation():
    env = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    env.reset(1702)
    env.targets["red_1"] = None
    target = env.targets["red_2"]
    assert target is not None
    env._by_id(target).state.alive = False
    env._by_id("red_3").state.alive = False
    _reward_call(env)
    for agent_id in ("red_1", "red_2", "red_3"):
        assert env._last_agent_reward_components[agent_id][
            "combat_situation_reward"
        ] == 0.0


def test_v17_event_credit_is_strictly_role_local():
    env = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    env.reset(1703)
    _reward_call(
        env,
        deaths={"red_0": 1, "red_2": 1},
        killers={"blue_0": "red_1", "blue_1": "red_2", "blue_2": "red_3"},
    )
    rows = env._last_agent_reward_components
    assert [rows[f"red_{slot}"]["own_kill_reward"] for slot in (1, 2, 3)] == [
        1.0,
        1.0,
        1.0,
    ]
    assert rows["red_0"]["support_team_kill_reward"] == pytest.approx(1.0)
    assert all(
        rows[f"red_{slot}"]["support_team_kill_reward"] == 0.0
        for slot in (1, 2, 3)
    )
    assert rows["red_0"]["death_penalty"] == -1.0
    assert rows["red_2"]["death_penalty"] == -1.0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("red_full_elimination", 3.0),
        ("red_total_loss", -3.0),
        ("mutual_elimination_draw", -3.0),
        ("timeout_red_win", -3.0),
        ("timeout_red_loss", -3.0),
        ("timeout_draw", -3.0),
    ],
)
def test_v17_mission_is_full_amount_in_every_agent_stream(monkeypatch, reason, expected):
    env = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    env.reset(1704)
    monkeypatch.setattr(env, "_terminal_result", lambda: (True, "red", reason))
    _reward_call(env)
    assert all(
        row["mission_reward"] == expected
        for row in env._last_agent_reward_components.values()
    )


def test_v17_active_reward_purity_ignores_lock_cue_assist_switch_and_boundary():
    env = FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(ENV_V17)
    env.reset(1705)
    direct = env._direct_visible_ids()
    _reward_call(env, direct=direct)
    before = deepcopy(env._last_agent_reward_components)
    env.lock_progress = {key: 999.0 for key in env.lock_progress}
    env._episode_metrics.update(
        {
            "red_1_boundary_hard_contacts_step": 999.0,
            "cue_to_direct": 999.0,
            "assisted_kills": 999.0,
            "target_switches": 999.0,
        }
    )
    _reward_call(
        env,
        direct=direct,
        pre_locks={"red_1": 999.0},
        half_events={("red_1", "blue_0")},
    )
    after = env._last_agent_reward_components
    assert {
        key: before[key]["agent_total_reward"] for key in before
    } == {key: after[key]["agent_total_reward"] for key in after}
    source = inspect.getsource(
        FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv._compute_reward
    )
    for forbidden in ("lock_quality_v11", "lock_progress", "boundary_penalty"):
        assert forbidden not in source
    assert COMBAT_ACTIVE_REWARD_KEYS_V17 == (
        "combat_situation_reward",
        "own_kill_reward",
        "death_penalty",
        "mission_reward",
    )
    assert SUPPORT_ACTIVE_REWARD_KEYS_V17 == (
        "support_situation_reward",
        "support_team_kill_reward",
        "death_penalty",
        "mission_reward",
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_v17_local_and_worker_vectors_are_finite(workers):
    envs = make_combat_vector_env_4v3_v14(ENV_V17, 2, workers, seed=1706)
    try:
        obs, states, masks = envs.reset()
        result = envs.step(np.zeros((2, 4, 3), dtype=np.float32))
        assert obs.shape == result.observations.shape == (2, 7, 118)
        assert states.shape == result.global_states.shape == (2, 70)
        assert masks.shape == result.alive_masks.shape == (2, 7)
        for value in (
            result.observations,
            result.global_states,
            result.team_rewards,
            result.agent_rewards,
            result.red_reward_components,
            result.red_agent_reward_components,
        ):
            assert np.isfinite(value).all()
    finally:
        envs.close()


def test_v17_selector_contract_and_log_are_registered():
    summary = {key: 0.0 for key in V15_BEST_SCORE_FIELDS_4V3}
    summary.update(any_kill_rate=0.25, task_win_rate=1.0, team_total_reward=999.0)
    assert compute_experiment_best_score_4v3(
        summary, V17_REWARD_CONTRACT_VERSION
    ) == compute_best_score_v15_4v3(summary)
    contract = _build_experiment_contract(
        checkpoint_family="family",
        cfg=_config(TRAIN_V17),
        env_cfg=_config(ENV_V17),
        env_config=str(ENV_V17),
        train_config=str(TRAIN_V17),
        manifest_hash="hash",
    )
    assert contract["best_checkpoint_selection"] == list(V15_BEST_SCORE_FIELDS_4V3)
    assert contract["observation_contract"] == "legacy_fixed_order"
    line = _format_update_line(
        type("T", (), {"env_steps": 1, "update_count": 1})(),
        2,
        {
            "mean_rollout_team_total_reward": 1.0,
            "mean_rollout_combat_situation_reward": 0.1,
            "mean_rollout_support_situation_reward": 0.2,
            "mean_rollout_mission_reward": 0.3,
            "group_update_order": ["support", "combat"],
            "support_kl": 0.0,
            "combat_joint_kl": 0.0,
            "entropy": 1.0,
        },
        is_v15=False,
        is_v17=True,
    )
    for field in (
        "reward=",
        "combat_sit_r=",
        "support_sit_r=",
        "mission_r=",
        "support_kl=",
        "combat_kl=",
        "entropy=",
    ):
        assert field in line
    assert "combat_state_r=" not in line


def test_v15_v16_v17_checkpoint_signatures_are_isolated(tmp_path: Path):
    trainers = [
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V15, _tiny(TRAIN_V15)),
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V16A, _tiny(TRAIN_V16A)),
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V16B, _tiny(TRAIN_V16B)),
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V17, _tiny(TRAIN_V17)),
    ]
    try:
        signatures = [trainer.training_signature() for trainer in trainers]
        for source_index, source_signature in enumerate(signatures):
            for target_index, target in enumerate(trainers):
                if source_index == target_index:
                    continue
                path = tmp_path / f"signature-{source_index}-{target_index}.pt"
                torch.save(
                    {
                        "checkpoint_family": source_signature["checkpoint_family"],
                        "training_signature": source_signature,
                    },
                    path,
                )
                with pytest.raises(ValueError, match="training signature mismatch"):
                    target.load_checkpoint(path)
    finally:
        for trainer in trainers:
            trainer.close()


def test_v17_checkpoint_roundtrip_and_deterministic_evaluation(tmp_path: Path):
    config = _tiny(TRAIN_V17)
    original = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V17, config)
    restored = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V17, config)
    try:
        original.collect_rollout()
        original.update()
        checkpoint = tmp_path / "v17.pt"
        original.save_checkpoint(checkpoint, scheduled_env_steps=original.env_steps)
        restored.load_checkpoint(checkpoint)
        assert restored.env_steps == original.env_steps
        assert restored.update_count == original.update_count
        for left, right in zip(
            original.actors.parameters(), restored.actors.parameters()
        ):
            assert torch.equal(left, right)
        one = evaluate_v14_happo_fixed_blue_4v3(
            restored.actors, ENV_V17, seeds=[91701], num_envs=1, device="cpu"
        )
        two = evaluate_v14_happo_fixed_blue_4v3(
            restored.actors, ENV_V17, seeds=[91701], num_envs=1, device="cpu"
        )
        assert one["episode_records"] == two["episode_records"]
        for key in (
            "team_total_reward",
            "mean_combat_situation_reward",
            "mean_support_situation_reward",
            "combat_slot_max_lock_mean",
            "mean_red_kills",
            "strict_full_elimination_rate",
        ):
            assert one[key] == two[key]
    finally:
        original.close()
        restored.close()
