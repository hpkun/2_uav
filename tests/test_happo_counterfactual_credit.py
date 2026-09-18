"""Focused contracts for continuous-action CF-HAPPO credit assignment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.common.buffer import RolloutBuffer
from algorithm.happo.credit_buffer import CreditRolloutBuffer
from algorithm.happo.counterfactual_credit import (
    CF_METHOD, RDC_METHOD, CounterfactualCreditCritic,
    component_credit, compute_component_lambda_returns,
    credit_component_names, extract_credit_components, replace_agent_action,
)
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.train_happo import CREDIT_FIELDS, RDC_FIELDS, _algorithm_name
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, load_environment_config


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "configs" / "env_v39.yaml"


def short_v39():
    config = deepcopy(load_environment_config(V39))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(method: str, device: str = "cpu") -> dict:
    return {
        "method_variant": method, "actor_variant": "vanilla", "critic_variant": "mlp",
        "num_envs": 1, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 2,
        "hidden_dim": 8, "seed": 23, "device": device, "environment_profile": "learnability",
    }


def test_baseline_isolated_from_credit_path_and_agp_still_constructs():
    baseline = HAPPOTrainer(config=trainer_config("baseline"))
    agp = HAPPOTrainer(config=trainer_config("agp"))
    try:
        assert type(baseline.buffer) is RolloutBuffer
        assert baseline.credit_critic is None and baseline.credit_critic_optimizer is None
        assert not baseline.credit_enabled
        assert agp.agp_enabled and not agp.credit_enabled
    finally:
        baseline.close(); agp.close()


@pytest.mark.parametrize("method", [CF_METHOD, RDC_METHOD])
def test_credit_methods_require_v39_vanilla_mlp(method):
    with pytest.raises(ValueError, match="requires v3.9"):
        HAPPOTrainer(config=trainer_config(method))
    for update in ({"actor_variant": "hrta"}, {"critic_variant": "relational"}):
        config = {**trainer_config(method), **update}
        with pytest.raises(ValueError):
            HAPPOTrainer(V39, config)


def test_cf_component_extraction_is_existing_team_reward():
    rewards = np.asarray([[1.0, 2.0, 3.0, 6.0], [-3.0, 2.0, 1.0, 4.0]], dtype=np.float32)
    result = extract_credit_components([{}, {}], rewards, CF_METHOD)
    assert result.shape == (2, 1)
    assert np.array_equal(result[:, 0], rewards.mean(axis=-1))


def test_credit_buffer_shapes_and_component_gae_boundaries_match_team_buffer():
    buffer = CreditRolloutBuffer(3, 1, component_count=1)
    base = RolloutBuffer(3, 1)
    term = (False, True, False)
    trunc = (False, False, True)
    for step in range(3):
        args = (
            np.zeros((1, 4, OBS_DIM), np.float32), np.zeros((1, GLOBAL_STATE_DIM), np.float32),
            np.zeros((1, 4, 3), np.float32), np.zeros((1, 4), np.float32),
            np.full((1, 4), float(step + 1), np.float32), np.asarray([0.25 * step], np.float32),
            np.asarray([term[step]]), np.asarray([trunc[step]]), np.ones((1, 4), np.float32),
        )
        base.insert(*args)
        buffer.insert(*args, credit_rewards=np.asarray([[step + 1]], np.float32))
    base.compute_returns_and_advantages(np.asarray([9.0]), .99, .95)
    component_values = base.values[..., None].copy()
    buffer.compute_credit_returns(component_values, np.asarray([[9.0]]), .99, .95)
    assert buffer.credit_rewards.shape == buffer.credit_values.shape == buffer.credit_returns.shape == (3, 1, 1)
    assert np.array_equal(buffer.credit_returns[..., 0], base.returns)
    direct = compute_component_lambda_returns(
        buffer.credit_rewards, component_values, np.asarray([[9.0]]),
        buffer.terminated, buffer.truncated, .99, .95,
    )
    assert np.array_equal(direct, buffer.credit_returns)


@pytest.mark.parametrize("components", [1, 5])
def test_credit_critic_shapes_and_outputs_are_finite(components):
    critic = CounterfactualCreditCritic(components, hidden_dim=8)
    states = torch.randn(7, GLOBAL_STATE_DIM)
    actions = torch.randn(7, 4, 3).tanh()
    values, q_values = critic.values(states), critic.q_values(states, actions)
    assert values.shape == q_values.shape == (7, components)
    assert torch.isfinite(values).all() and torch.isfinite(q_values).all()


def test_replace_agent_action_changes_only_selected_three_dimensions_bitwise():
    actual = torch.arange(48, dtype=torch.float32).reshape(4, 4, 3)
    replacement = torch.full((4, 3), -7.0, requires_grad=True)
    changed = replace_agent_action(actual, 2, replacement)
    assert torch.equal(changed[:, :2], actual[:, :2])
    assert torch.equal(changed[:, 3:], actual[:, 3:])
    assert torch.equal(changed[:, 2], replacement.detach())
    assert changed[:, 2].grad_fn is None


def test_cf_advantage_is_exact_q_difference():
    actual = torch.tensor([[3.0], [-2.0]])
    counterfactual = torch.tensor([[1.5], [-4.0]])
    assert torch.equal(component_credit(actual, counterfactual).squeeze(-1), torch.tensor([1.5, 2.0]))


@pytest.mark.parametrize("method", [CF_METHOD, RDC_METHOD])
def test_credit_train_update_changes_actors_credit_and_standard_critics(method):
    trainer = HAPPOTrainer(short_v39(), trainer_config(method))
    try:
        actor_before = [[parameter.detach().clone() for parameter in actor.parameters()] for actor in trainer.actors.actors]
        critic_before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]
        credit_before = [parameter.detach().clone() for parameter in trainer.credit_critic.parameters()]
        _, metrics = trainer.train_update()
        assert all(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())) for before, actor in zip(actor_before, trainer.actors.actors))
        assert any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(credit_before, trainer.credit_critic.parameters()))
        assert all(action.requires_grad is False for action in trainer.last_counterfactual_actions)
        for field in ("credit_value_loss", "credit_q_loss", "credit_total_loss"):
            assert np.isfinite(metrics[field])
        for agent in range(4):
            assert np.isfinite(metrics[f"credit_adv_mean_abs_{agent}"])
            assert np.isfinite(metrics[f"credit_adv_std_{agent}"])
    finally:
        trainer.close()


def test_cf_and_rdc_configs_are_exact_single_field_copies():
    with (ROOT / "configs" / "happo_entropy001_logstd025_screen.yaml").open() as stream:
        base = yaml.safe_load(stream)["training"]
    for name, method in (("happo_cf_v39.yaml", CF_METHOD), ("happo_rdc_v39.yaml", RDC_METHOD)):
        with (ROOT / "configs" / name).open() as stream:
            candidate = yaml.safe_load(stream)["training"]
        assert candidate.pop("method_variant") == method
        expected = dict(base); expected.pop("method_variant")
        assert candidate == expected


def test_dimensions_and_component_names_remain_frozen():
    assert OBS_DIM == 100 and GLOBAL_STATE_DIM == 117
    assert credit_component_names(CF_METHOD) == ("team",)
    assert credit_component_names(RDC_METHOD) == (
        "shared", "mav_role", "uav1_role", "uav2_role", "uav3_role",
    )


def test_entrypoint_algorithm_names_and_credit_csv_fields():
    assert _algorithm_name("vanilla", CF_METHOD, "mlp") == CF_METHOD
    assert _algorithm_name("vanilla", RDC_METHOD, "mlp") == RDC_METHOD
    assert {"credit_value_loss", "credit_q_loss", "credit_total_loss"} <= set(CREDIT_FIELDS)
    assert "rdc_shared_credit_mean_abs" in RDC_FIELDS


@pytest.mark.parametrize("method", [CF_METHOD, RDC_METHOD])
def test_counterfactual_sampled_once_per_transition_agent_and_frozen(method, monkeypatch):
    trainer = HAPPOTrainer(short_v39(), trainer_config(method))
    try:
        trainer.collect_rollout()
        counts = [0, 0, 0, 0]
        for index, actor in enumerate(trainer.actors.actors):
            original = actor.sample

            def wrapped(observations, deterministic=False, *, _index=index, _original=original):
                counts[_index] += 1
                return _original(observations, deterministic)

            monkeypatch.setattr(actor, "sample", wrapped)
        trainer.update()
        assert counts == [1, 1, 1, 1]
        assert len(trainer.last_counterfactual_actions) == 4
        assert all(not actions.requires_grad for actions in trainer.last_counterfactual_actions)
    finally:
        trainer.close()
