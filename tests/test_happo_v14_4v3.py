from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import uav_combat.environment_4v3_v14 as env_v14_module
from uav_combat.environment_4v3_v12 import (
    DEATH_LOCK_V12,
    FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv,
)
from uav_combat.environment_4v3_v14 import (
    AGENT_REWARD_COMPONENT_KEYS_V14,
    FunctionalHeterogeneous4v3V14MissionAlignedEnv,
)
from uav_combat.happo.evaluation_v14_4v3 import evaluate_v14_happo_fixed_blue_4v3
from uav_combat.happo.role_credit_buffer import (
    AgentCreditRolloutBuffer4v3,
    normalize_role_advantages,
)
from uav_combat.happo.role_credit_networks import RoleSharedCentralizedCritics4v3
from uav_combat.happo.role_shared_buffer import RoleSharedRolloutBuffer4v3
from uav_combat.happo.trainer_role_shared_4v3 import (
    combat_joint_log_probability,
    role_group_factor_update,
)
from uav_combat.happo.trainer_v14_4v3 import (
    CREDIT_MODE_ROLE_LOCAL,
    CREDIT_MODE_TEAM,
    MissionAlignedRoleSharedHAPPO4v3Trainer,
    combat_local_clipped_policy_loss,
)
from uav_combat.mappo.vector_env_4v3_v14 import make_combat_vector_env_4v3_v14
from uav_combat.scenario_4v3_v14 import RED_IDS_V14, resolved_reward_contract_v14


ROOT = Path(__file__).resolve().parents[1]
ENV_V12 = ROOT / "configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml"
ENV_V14 = ROOT / "configs/heterogeneous_4v3_main_v14_mission_aligned_role_credit.yaml"
TRAIN_A = ROOT / "configs/happo_heterogeneous_4v3_main_v14a_role_shared_combat_mlp_mission_aligned_team_credit.yaml"
TRAIN_B = ROOT / "configs/happo_heterogeneous_4v3_main_v14b_role_shared_combat_mlp_mission_aligned_role_credit.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny(path: Path, *, num_envs: int = 2, rollout_steps: int = 8) -> dict:
    cfg = _config(path)
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


def _team_components(mission: float = 0.0, formation: float = 0.0) -> dict[str, float]:
    return {
        "mission_outcome_reward": mission,
        "support_formation_progress_reward": formation,
    }


def _agent_reward_call(
    env: FunctionalHeterogeneous4v3V14MissionAlignedEnv,
    *,
    mission: float = 0.0,
    formation: float = 0.0,
    deaths: dict[str, int] | None = None,
    half_events: set[tuple[str, str]] | None = None,
    killers: dict[str, str] | None = None,
    pre_potentials: dict[str, float] | None = None,
    pre_locks: dict[str, float] | None = None,
):
    return env._agent_rewards(
        dict(env.targets),
        pre_potentials or {},
        pre_locks or dict(env.lock_progress),
        deaths or {},
        half_events or set(),
        killers or {},
        _team_components(mission, formation),
    )


def test_v14_mission_contract_exact_values():
    contract = resolved_reward_contract_v14(_config(ENV_V14))
    assert contract["mission"] == {
        "red_full_elimination": 20.0,
        "red_total_loss": -15.0,
        "mutual_elimination_draw": -2.0,
        "timeout_red_win": -15.0,
        "timeout_red_loss": -15.0,
        "timeout_draw": -15.0,
    }


@pytest.mark.parametrize("reason", ["timeout_red_win", "timeout_red_loss", "timeout_draw"])
def test_every_timeout_gives_all_agents_minus_fifteen(reason: str):
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(1)
    mission = env.reward_contract["mission"][reason]
    rewards, components = _agent_reward_call(env, mission=mission)
    assert all(components[agent]["common_mission_reward"] == -15.0 for agent in RED_IDS_V14)
    assert all(reward == -15.0 for reward in rewards.values())


def test_strict_full_gives_all_agents_common_plus_twenty():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(2)
    rewards, components = _agent_reward_call(env, mission=20.0)
    assert all(components[agent]["common_mission_reward"] == 20.0 for agent in RED_IDS_V14)
    assert all(reward == 20.0 for reward in rewards.values())


def test_true_killer_gets_kill_reward_only_and_support_requires_assist():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(3)
    _, components = _agent_reward_call(env, killers={"blue_0": "red_3"})
    assert components["red_3"]["kill_event_reward"] == 8.0
    assert components["red_1"]["kill_event_reward"] == 0.0
    assert components["red_2"]["kill_event_reward"] == 0.0
    assert components["red_0"]["support_assisted_kill_reward"] == 0.0
    env._cue_last_step[("red_3", "blue_0")] = env.step_count
    _, assisted = _agent_reward_call(env, killers={"blue_0": "red_3"})
    assert assisted["red_0"]["support_assisted_kill_reward"] == 1.5


def test_half_lock_is_local_to_the_producing_combat():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(4)
    _, components = _agent_reward_call(
        env, half_events={("red_2", "blue_1")}
    )
    assert components["red_2"]["half_lock_event_reward"] == 0.5
    assert components["red_1"]["half_lock_event_reward"] == 0.0
    assert components["red_3"]["half_lock_event_reward"] == 0.0


def test_combat_geometry_and_lock_progress_do_not_cancel(monkeypatch):
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(5)
    for slot, agent in enumerate(("red_1", "red_2", "red_3")):
        env.targets[agent] = f"blue_{slot}"
        env.lock_progress[agent] = (0.2, 0.0, 0.1)[slot]
    monkeypatch.setattr(env_v14_module, "combat_potential_v11", lambda *args: 0.6)
    _, components = _agent_reward_call(
        env,
        pre_potentials={"red_1": 0.55, "red_2": 0.65, "red_3": 0.6},
        pre_locks={"red_1": 0.1, "red_2": 0.1, "red_3": 0.2},
    )
    assert components["red_1"]["geometry_progress_reward"] == pytest.approx(0.03)
    assert components["red_2"]["geometry_progress_reward"] == pytest.approx(-0.03)
    assert components["red_1"]["lock_progress_reward"] == pytest.approx(0.05)
    assert components["red_3"]["lock_progress_reward"] == pytest.approx(-0.05)


def test_death_penalties_are_agent_local():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(6)
    _, combat = _agent_reward_call(env, deaths={"red_2": DEATH_LOCK_V12})
    assert combat["red_2"]["death_event_penalty"] == -4.0
    assert combat["red_1"]["death_event_penalty"] == 0.0
    _, support = _agent_reward_call(env, deaths={"red_0": DEATH_LOCK_V12})
    assert support["red_0"]["death_event_penalty"] == -1.0
    assert support["red_1"]["death_event_penalty"] == 0.0


def test_hard_contact_is_agent_local():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(7)
    env._episode_metrics["red_3_boundary_hard_contacts_step"] = 1.0
    _, components = _agent_reward_call(env)
    assert components["red_3"]["boundary_event_penalty"] == -0.1
    assert components["red_1"]["boundary_event_penalty"] == 0.0
    assert components["red_0"]["boundary_event_penalty"] == 0.0


def test_support_events_do_not_enter_combat_local_rewards():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(8)
    env._episode_metrics["support_unique_detection_events_step"] = 2.0
    _, components = _agent_reward_call(env)
    assert components["red_0"]["support_unique_detection_reward"] == 0.04
    for agent in ("red_1", "red_2", "red_3"):
        assert components[agent]["support_unique_detection_reward"] == 0.0


def test_agent_total_is_common_plus_local_exactly_once():
    env = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    env.reset(9)
    rewards, components = _agent_reward_call(
        env, mission=20.0, killers={"blue_0": "red_3"}
    )
    assert rewards["red_3"] == 28.0
    assert components["red_3"]["agent_total_reward"] == 28.0
    assert rewards["red_1"] == 20.0


def test_v14_team_nonmission_components_match_v12_on_same_trajectory():
    old = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(ENV_V12)
    new = FunctionalHeterogeneous4v3V14MissionAlignedEnv(ENV_V14)
    old.reset(1234)
    new.reset(1234)
    zero = {agent: np.zeros(3, np.float32) for agent in RED_IDS_V14}
    for _ in range(10):
        old_step = old.step(zero)
        new_step = new.step(zero)
        assert np.array_equal(old_step[0], new_step[0])
        old_components = old_step[-1]["reward_components"]
        new_components = new_step[-1]["reward_components"]
        for key in old_components:
            if key not in {"mission_outcome_reward", "team_total_reward"}:
                assert new_components[key] == pytest.approx(old_components[key], abs=1e-8)


def test_vector_result_carries_finite_team_and_agent_rewards():
    envs = make_combat_vector_env_4v3_v14(ENV_V14, 2, 0, 10)
    try:
        result = envs.step(np.zeros((2, 4, 3), np.float32))
        assert result.agent_rewards.shape == (2, 4)
        assert result.red_agent_reward_components.shape == (
            2,
            4,
            len(AGENT_REWARD_COMPONENT_KEYS_V14),
        )
        assert np.isfinite(result.team_rewards).all()
        assert np.isfinite(result.agent_rewards).all()
    finally:
        envs.close()


def test_multiprocessing_vector_result_carries_agent_rewards():
    envs = make_combat_vector_env_4v3_v14(ENV_V14, 2, 1, 11)
    try:
        result = envs.step(np.zeros((2, 4, 3), np.float32))
        assert result.agent_rewards.shape == (2, 4)
        assert np.isfinite(result.agent_rewards).all()
    finally:
        envs.close()


def test_per_agent_gae_shape_and_episode_done_cut():
    buffer = AgentCreditRolloutBuffer4v3(4, 1, 2, 3)
    for step in range(4):
        reward = np.zeros((1, 4), np.float32)
        if step in (1, 3):
            reward[:] = 1.0
        buffer.add(
            np.zeros((1, 4, 2), np.float32),
            np.zeros((1, 3), np.float32),
            np.zeros((1, 4, 3), np.float32),
            np.zeros((1, 4), np.float32),
            np.ones((1, 4), np.float32),
            np.zeros(1, np.float32),
            reward,
            np.zeros((1, 4), np.float32),
            np.asarray([step in (1, 3)]),
        )
    buffer.compute_returns_and_advantages(np.zeros((1, 4), np.float32), 1.0, 1.0)
    assert buffer.advantages.shape == (4, 1, 4)
    assert np.all(buffer.advantages[0] == 1.0)
    assert np.all(buffer.advantages[2] == 1.0)


def test_agent_death_does_not_cut_later_common_mission_return():
    buffer = AgentCreditRolloutBuffer4v3(9, 1, 2, 3)
    for step in range(9):
        reward = np.zeros((1, 4), np.float32)
        if step == 3:
            reward[0, 1] = -4.0
        if step == 8:
            reward[0] += 20.0
        alive = np.ones((1, 4), np.float32)
        if step > 3:
            alive[0, 1] = 0.0
        buffer.add(
            np.zeros((1, 4, 2), np.float32), np.zeros((1, 3), np.float32),
            np.zeros((1, 4, 3), np.float32), np.zeros((1, 4), np.float32),
            alive, np.zeros(1, np.float32), reward,
            np.zeros((1, 4), np.float32), np.asarray([step == 8]),
        )
    buffer.compute_returns_and_advantages(np.zeros((1, 4), np.float32), 1.0, 1.0)
    assert buffer.advantages[3, 0, 1] == 16.0
    assert buffer.advantages[0, 0, 1] == 16.0
    assert buffer.agent_alive_masks[3, 0, 1] == 1.0
    assert buffer.agent_alive_masks[4, 0, 1] == 0.0


def test_pooled_combat_normalization_uses_only_alive_slots():
    advantages = np.asarray([[[1.0, 1.0, 100.0, 3.0], [2.0, 5.0, 7.0, 9.0]]])
    masks = np.asarray([[[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]])
    support, combat = normalize_role_advantages(advantages, masks)
    active = combat[masks[..., 1:4] > 0.5]
    assert active.mean() == pytest.approx(0.0, abs=1e-6)
    assert active.std() == pytest.approx(1.0, abs=1e-6)
    assert combat[0, 0, 1] == 0.0
    assert support[0, 1] == 0.0


def test_combat_local_surrogate_uses_slot_ratios_and_advantages():
    old = torch.zeros((1, 3))
    new = torch.log(torch.tensor([[1.1, 0.9, 1.2]]))
    advantages = torch.tensor([[1.0, -2.0, 3.0]])
    alive = torch.tensor([[1.0, 1.0, 0.0]])
    loss = combat_local_clipped_policy_loss(new, old, advantages, alive, 0.2)
    expected = -torch.tensor([1.1, -1.8]).mean()
    assert torch.allclose(loss, expected)


def test_combat_preceding_factor_uses_joint_ratio():
    old = torch.tensor([[0.1, 0.2, 0.3]])
    new = torch.tensor([[0.2, 0.4, 0.9]])
    alive = torch.tensor([[1.0, 1.0, 0.0]])
    old_joint = combat_joint_log_probability(old, alive)
    new_joint = combat_joint_log_probability(new, alive)
    factor = role_group_factor_update(torch.ones(1), old_joint, new_joint, torch.ones(1))
    assert torch.allclose(factor, torch.exp(torch.tensor([0.3])))


def test_role_critics_share_one_combat_parameter_object():
    critics = RoleSharedCentralizedCritics4v3(7, 5, 16)
    state = torch.randn(2, 7)
    obs = torch.randn(2, 4, 5)
    values = critics(state, obs)
    assert values.shape == (2, 4)
    assert not hasattr(critics, "combat_1_critic")
    assert len(list(critics.combat_critic.parameters())) > 0


def test_v14a_keeps_scalar_team_buffer_and_advantage():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, _tiny(TRAIN_A))
    try:
        assert trainer.credit_mode == CREDIT_MODE_TEAM
        assert isinstance(trainer.buffer, RoleSharedRolloutBuffer4v3)
        trainer.collect_rollout()
        assert trainer.buffer.advantages.shape == (8, 2)
        metrics = trainer.update()
        assert metrics["credit_mode"] == "team"
        assert np.isfinite(list(v for v in metrics.values() if isinstance(v, (int, float)))).all()
    finally:
        trainer.close()


def test_v14b_uses_per_agent_advantages_and_one_combat_optimizer(monkeypatch):
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, _tiny(TRAIN_B))
    calls = 0
    original = trainer.combat_optimizer.step

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.combat_optimizer, "step", counted)
    try:
        trainer.collect_rollout()
        assert trainer.buffer.advantages.shape == (8, 2, 4)
        metrics = trainer.update()
        assert calls == metrics["combat_optimizer_steps"]
        assert calls == 1
        assert len(trainer.actor_optimizers) == 2
        assert metrics["credit_mode"] == "role_local"
    finally:
        trainer.close()


def test_all_dead_combat_safely_skips_actor_update():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, _tiny(TRAIN_B, num_envs=1, rollout_steps=2))
    try:
        obs = torch.zeros((2, 4, 118))
        actions = torch.zeros((2, 4, 3))
        old = torch.zeros((2, 4))
        masks = torch.zeros((2, 4))
        masks[:, 0] = 1.0
        advantages = torch.ones((2, 4))
        rows, factor = trainer._update_credit_actors(obs, actions, old, masks, advantages)
        summary = next(row for row in rows if row.get("group") == "combat" and row.get("summary"))
        assert summary["optimizer_steps"] == 0
        assert torch.isfinite(factor).all()
    finally:
        trainer.close()


def test_v14_checkpoint_roundtrip_and_credit_mode_mismatch(tmp_path: Path):
    cfg_a = _tiny(TRAIN_A, num_envs=1, rollout_steps=2)
    cfg_b = _tiny(TRAIN_B, num_envs=1, rollout_steps=2)
    source = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, cfg_a)
    restored = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, cfg_a)
    incompatible = MissionAlignedRoleSharedHAPPO4v3Trainer(ENV_V14, cfg_b)
    path = tmp_path / "v14a.pt"
    try:
        source.collect_rollout()
        source.update()
        source.save_checkpoint(path)
        expected = source._select_actions()[:3]
        restored.load_checkpoint(path)
        actual = restored._select_actions()[:3]
        for left, right in zip(expected, actual):
            assert np.array_equal(left, right)
        with pytest.raises(ValueError, match="training signature mismatch"):
            incompatible.load_checkpoint(path)
    finally:
        source.close()
        restored.close()
        incompatible.close()


def test_v13_family_cannot_load_into_v14(tmp_path: Path):
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(
        ENV_V14, _tiny(TRAIN_A, num_envs=1, rollout_steps=2)
    )
    path = tmp_path / "v13.pt"
    torch.save({"checkpoint_family": "functional_heterogeneous_4v3_role_shared_happo"}, path)
    try:
        with pytest.raises(ValueError, match="checkpoint family mismatch"):
            trainer.load_checkpoint(path)
    finally:
        trainer.close()


def test_deterministic_v14_evaluation_repeats_same_agent_returns():
    trainer = MissionAlignedRoleSharedHAPPO4v3Trainer(
        ENV_V14, _tiny(TRAIN_B, num_envs=1, rollout_steps=2)
    )
    try:
        one = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V14, seeds=[7001], num_envs=1, device="cpu"
        )
        two = evaluate_v14_happo_fixed_blue_4v3(
            trainer.actors, ENV_V14, seeds=[7001], num_envs=1, device="cpu"
        )
        assert one["team_total_reward"] == two["team_total_reward"]
        assert one["mean_support_agent_return"] == two["mean_support_agent_return"]
        for key in (
            "task_win_rate",
            "strict_full_elimination_rate",
            "timeout_win_rate",
            "timeout_loss_rate",
            "timeout_draw_rate",
            "any_kill_rate",
            "at_least_two_kill_rate",
        ):
            assert key in one
    finally:
        trainer.close()
