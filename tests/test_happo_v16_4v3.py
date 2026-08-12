from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.train_happo_role_shared_4v3 import _build_experiment_contract
from uav_combat.environment_4v3_v12 import DEATH_LOCK_V12
from uav_combat.environment_4v3_v15 import (
    FunctionalHeterogeneous4v3V15PaperCompactRewardEnv,
    combat_state_reward_v15,
)
from uav_combat.environment_4v3_v16 import (
    TEAMMATE_SEGMENT_V16,
    FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv,
    FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv,
    canonicalize_same_team_teammate_blocks_v16,
    combat_state_reward_v16,
)
from uav_combat.happo.evaluation_v14_4v3 import (
    evaluate_v14_happo_fixed_blue_4v3,
)
from uav_combat.happo.trainer_4v3 import (
    V15_BEST_SCORE_FIELDS_4V3,
    V16_REWARD_CONTRACT_VERSION,
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
TRAIN_V16A = ROOT / "configs/happo_heterogeneous_4v3_main_v16a_role_shared_combat_mlp_positive_lock_quality_reward.yaml"
TRAIN_V16B = ROOT / "configs/happo_heterogeneous_4v3_main_v16b_role_shared_combat_mlp_positive_lock_quality_canonical_obs.yaml"
TRAIN_V15 = ROOT / "configs/happo_heterogeneous_4v3_main_v15_role_shared_combat_mlp_paper_compact_reward.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny(path: Path, *, num_envs: int = 1, rollout_steps: int = 2) -> dict:
    config = _config(path)
    config["experiment"]["device"] = "cpu"
    config["training"].update(
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
    config["evaluation"]["selection_episodes"] = 1
    config["evaluation"]["test_episodes"] = 1
    return config


def _reward_call(env, *, deaths=None, killers=None, direct=None):
    return env._compute_reward(
        {},
        {},
        {},
        deaths or {},
        set(),
        killers or {},
        direct or env._direct_visible_ids(),
    )


def _team_observation(env, aircraft_id: str) -> np.ndarray:
    direct = env._direct_visible_ids()
    effective = env._effective_targets(direct)
    return env._obs_for(env._by_id(aircraft_id), direct, effective)


def test_v16_configs_freeze_v15_environment_and_form_independent_experiments():
    v15 = _config(ENV_V15)
    a = _config(ENV_V16A)
    b = _config(ENV_V16B)
    for candidate in (a, b):
        frozen = deepcopy(candidate)
        frozen["combat"] = deepcopy(v15["combat"])
        assert frozen == v15
    assert a["combat"]["observation_contract"] == "legacy_fixed_order"
    assert b["combat"]["observation_contract"] == "canonical_same_team"
    train_a = _config(TRAIN_V16A)
    train_b = _config(TRAIN_V16B)
    for train in (train_a, train_b):
        assert train["training"]["num_envs"] == 16
        assert train["training"]["num_env_workers"] == 4
        assert train["training"]["rollout_steps"] == 256
        assert train["training"]["minibatch_size"] == 1024
        assert train["training"]["ppo_epochs"] == 5
        assert train["training"]["total_env_steps"] == 3_000_000
        assert train["training"]["schedule_env_steps"] == 3_000_000
    assert train_a["experiment"]["variant"] != train_b["experiment"]["variant"]


@pytest.mark.parametrize(
    ("quality", "expected"),
    [(0.0, 0.0), (0.1, 0.002), (0.25, 0.005), (0.5, 0.01), (1.0, 0.02)],
)
def test_v16_positive_lock_quality_formula_has_no_threshold(quality, expected):
    assert combat_state_reward_v16(quality) == pytest.approx(expected)


def test_v15_historical_centered_formula_is_unchanged():
    assert combat_state_reward_v15(0.0) == pytest.approx(-0.02)
    assert combat_state_reward_v15(0.5) == pytest.approx(0.0)
    assert combat_state_reward_v15(1.0) == pytest.approx(0.02)


def test_v16_missing_dead_target_and_dead_combat_have_zero_state_reward():
    env = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    env.reset(40)
    env.targets["red_1"] = None
    _reward_call(env)
    assert env._last_agent_reward_components["red_1"]["combat_state_reward"] == 0.0
    target = env.targets["red_2"]
    assert target is not None
    env._by_id(target).state.alive = False
    _reward_call(env)
    assert env._last_agent_reward_components["red_2"]["combat_state_reward"] == 0.0
    env._by_id("red_3").state.alive = False
    _reward_call(env)
    assert env._last_agent_reward_components["red_3"]["combat_state_reward"] == 0.0


def test_v16_environment_reuses_exact_v11_lock_quality_source(monkeypatch):
    env = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    env.reset(41)
    env.targets["red_1"] = "blue_0"
    monkeypatch.setattr(
        "uav_combat.environment_4v3_v15.lock_quality_v11",
        lambda attacker, target, profile: 0.25,
    )
    _reward_call(env)
    assert env._last_agent_reward_components["red_1"][
        "combat_state_reward"
    ] == pytest.approx(0.005)


def test_v16_frozen_event_support_terminal_and_inactive_reward_terms():
    v15 = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    v16 = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    v15.reset(42)
    v16.reset(42)
    v16._episode_metrics["red_1_boundary_hard_contacts_step"] = 2.0
    _reward_call(
        v16,
        deaths={"red_1": DEATH_LOCK_V12},
        killers={"blue_0": "red_1"},
    )
    rows = v16._last_agent_reward_components
    assert rows["red_1"]["own_kill_reward"] == 8.0
    assert all(rows[agent]["team_kill_reward"] == 1.0 for agent in rows)
    assert rows["red_1"]["death_penalty"] == -4.0
    assert rows["red_1"]["boundary_penalty"] == pytest.approx(-0.2)
    _reward_call(v15)
    clean = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    clean.reset(42)
    _reward_call(clean)
    for key in (
        "support_position_state_reward",
        "support_awareness_state_reward",
        "support_state_reward",
    ):
        assert clean._last_agent_reward_components["red_0"][key] == pytest.approx(
            v15._last_agent_reward_components["red_0"][key]
        )
    row = clean._last_agent_reward_components["red_1"]
    active = (
        "combat_state_reward",
        "own_kill_reward",
        "team_kill_reward",
        "death_penalty",
        "boundary_penalty",
        "support_state_reward",
    )
    assert row["agent_total_reward"] == pytest.approx(sum(row[key] for key in active))
    assert all(value == 0.0 for value in _config(ENV_V16A)["rewards"]["mission"].values())


def test_v16a_observation_is_bitwise_v15_and_v16b_changes_only_teammates():
    v15 = FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(ENV_V15)
    a = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    b = FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv(ENV_V16B)
    obs15, state15, masks15 = v15.reset(43)
    obsa, statea, masksa = a.reset(43)
    obsb, stateb, masksb = b.reset(43)
    assert np.array_equal(obsa, obs15)
    assert np.array_equal(statea, state15)
    assert np.array_equal(masksa, masks15)
    assert obsb.shape == (7, 118)
    assert np.isfinite(obsb).all()
    assert np.array_equal(obsb[:, :12], obsa[:, :12])
    assert np.array_equal(obsb[:, 36:], obsa[:, 36:])
    assert np.array_equal(stateb, statea)
    assert np.array_equal(masksb, masksa)


def test_v16b_canonical_role_and_pure_geometry_ordering():
    env = FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv(ENV_V16B)
    env.reset(44)
    combat = _team_observation(env, "red_2")[TEAMMATE_SEGMENT_V16].reshape(3, 8)
    assert combat[0, 7] == 1.0
    assert np.all(combat[1:, 7] == 0.0)
    assert np.dot(combat[1, :3], combat[1, :3]) <= np.dot(
        combat[2, :3], combat[2, :3]
    )
    support = _team_observation(env, "red_0")[TEAMMATE_SEGMENT_V16].reshape(3, 8)
    physical_relative_position = support[:, :3] * np.asarray(
        [6000.0, 6000.0, 3000.0], dtype=np.float32
    )
    distances = np.sum(np.square(physical_relative_position), axis=1)
    assert np.all(distances[:-1] <= distances[1:] + 1e-8)
    source = inspect.getsource(canonicalize_same_team_teammate_blocks_v16)
    assert "aircraft_id" not in source


def test_v16b_teammate_segment_is_consistent_under_combat_identity_permutation():
    b = FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv(ENV_V16B)
    a = FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(ENV_V16A)
    b.reset(45)
    a.reset(45)
    before_b_combat = _team_observation(b, "red_2")[TEAMMATE_SEGMENT_V16]
    before_b_support = _team_observation(b, "red_0")[TEAMMATE_SEGMENT_V16]
    before_a = _team_observation(a, "red_2")[TEAMMATE_SEGMENT_V16]
    for env in (a, b):
        left = env._by_id("red_1").state.copy()
        right = env._by_id("red_3").state.copy()
        env._by_id("red_1").state = right
        env._by_id("red_3").state = left
    after_b_combat = _team_observation(b, "red_2")[TEAMMATE_SEGMENT_V16]
    after_b_support = _team_observation(b, "red_0")[TEAMMATE_SEGMENT_V16]
    after_a = _team_observation(a, "red_2")[TEAMMATE_SEGMENT_V16]
    assert np.array_equal(before_b_combat, after_b_combat)
    assert np.array_equal(before_b_support, after_b_support)
    assert not np.array_equal(before_a, after_a)


@pytest.mark.parametrize("env_path", [ENV_V16A, ENV_V16B])
@pytest.mark.parametrize("workers", [0, 2])
def test_v16_local_and_worker_vectors_keep_shapes_and_finite_values(env_path, workers):
    envs = make_combat_vector_env_4v3_v14(env_path, 2, workers, seed=46)
    try:
        obs, states, masks = envs.reset()
        result = envs.step(np.zeros((2, 4, 3), dtype=np.float32))
        assert obs.shape == result.observations.shape == (2, 7, 118)
        assert states.shape == result.global_states.shape == (2, 70)
        assert masks.shape == result.alive_masks.shape == (2, 7)
        assert all(
            np.isfinite(value).all()
            for value in (result.observations, result.global_states, result.agent_rewards)
        )
    finally:
        envs.close()


def test_v16_selector_and_experiment_contract_are_attack_only():
    summary = {key: 0.0 for key in V15_BEST_SCORE_FIELDS_4V3}
    summary.update(any_kill_rate=0.2, task_win_rate=1.0, team_total_reward=999.0)
    assert compute_experiment_best_score_4v3(
        summary, V16_REWARD_CONTRACT_VERSION
    ) == compute_best_score_v15_4v3(summary)
    config = _config(TRAIN_V16B)
    contract = _build_experiment_contract(
        checkpoint_family="family",
        cfg=config,
        env_cfg=_config(ENV_V16B),
        env_config=str(ENV_V16B),
        train_config=str(TRAIN_V16B),
        manifest_hash="hash",
    )
    assert contract["best_checkpoint_selection"] == list(V15_BEST_SCORE_FIELDS_4V3)
    assert contract["observation_contract"] == "canonical_same_team"


def test_v15_v16a_v16b_checkpoint_signatures_reject_cross_resume(tmp_path: Path):
    trainers = [
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V15, _tiny(TRAIN_V15)),
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V16A, _tiny(TRAIN_V16A)),
        MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V16B, _tiny(TRAIN_V16B)),
    ]
    try:
        signatures = [trainer.training_signature() for trainer in trainers]
        for source_index, source_signature in enumerate(signatures):
            for target_index, target in enumerate(trainers):
                if source_index == target_index:
                    continue
                path = tmp_path / f"{source_index}-{target_index}.pt"
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


@pytest.mark.parametrize(
    ("env_path", "train_path"),
    [(ENV_V16A, TRAIN_V16A), (ENV_V16B, TRAIN_V16B)],
)
def test_v16_deterministic_evaluation_repeats(env_path, train_path):
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(env_path, _tiny(train_path))
    try:
        one = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, env_path, seeds=[9101], num_envs=1, device="cpu"
        )
        two = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, env_path, seeds=[9101], num_envs=1, device="cpu"
        )
        for key in (
            "team_total_reward",
            "mean_combat_state_reward",
            "mean_max_lock_progress",
            "mean_red_kills",
        ):
            assert one[key] == two[key]
    finally:
        trainer.close()
