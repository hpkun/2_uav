from __future__ import annotations

import torch

from algorithm.modules.hrta import HRTAActor, HRTAIndependentActors
from env.mavuav import BLUE_IDS, OBS_DIM, RED_IDS


def observations(batch: int = 4) -> torch.Tensor:
    torch.manual_seed(2026)
    values = torch.randn(batch, OBS_DIM) * 0.2
    values[:, 7:10] = torch.tensor([1.0, 0.0, 0.0])
    for start in (11, 22, 33):
        values[:, start + 7] = 1.0
    for start in (44, 58, 72, 86):
        values[:, start + 9] = 1.0
        values[:, start + 10] = 1.0
        values[:, start + 11] = 0.0
    return values


def test_hrta_actor_action_shape_range_and_finite_distribution_values():
    actor = HRTAActor()
    obs = observations(8)
    actions, sampled_log_prob = actor.sample(obs)
    evaluated_log_prob, entropy = actor.evaluate_actions(obs, actions)
    assert actions.shape == (8, 3)
    assert sampled_log_prob.shape == entropy.shape == (8,)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(sampled_log_prob).all()
    assert torch.isfinite(evaluated_log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=2e-5, rtol=2e-5)


def test_enemy_attention_four_visible_enemies_is_normalized():
    attention = HRTAActor().attention_weights(observations(5))
    assert attention.shape == (5, len(BLUE_IDS))
    assert torch.allclose(attention.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(attention >= 0.0)


def test_invisible_blue1_is_masked():
    obs = observations(3)
    obs[:, 44 + 10] = 0.0
    obs[:, 44 + 11] = 0.0
    attention = HRTAActor().attention_weights(obs)
    assert torch.equal(attention[:, 0], torch.zeros(3))
    assert torch.allclose(attention[:, 1:].sum(dim=-1), torch.ones(3))


def test_dead_blue2_is_masked():
    obs = observations(3)
    obs[:, 58 + 9] = 0.0
    attention = HRTAActor().attention_weights(obs)
    assert torch.equal(attention[:, 1], torch.zeros(3))
    assert torch.allclose(attention.sum(dim=-1), torch.ones(3))


def test_no_visible_blue_returns_zero_attention_and_finite_outputs():
    obs = observations(4)
    for start in (44, 58, 72, 86):
        obs[:, start + 10] = 0.0
        obs[:, start + 11] = 0.0
    actor = HRTAActor()
    features, diagnostics = actor.encode(obs)
    actions, log_prob = actor.sample(obs)
    assert torch.equal(diagnostics["enemy_attention"], torch.zeros(4, len(BLUE_IDS)))
    assert torch.isfinite(features).all()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()


def test_friend_alive_mask_and_all_dead_case_are_numerically_safe():
    obs = observations(2)
    obs[:, 22 + 7] = obs[:, 33 + 7] = 0.0
    actor = HRTAActor()
    features, diagnostics = actor.encode(obs)
    assert torch.equal(diagnostics["friend_attention"][:, 0], torch.ones(2))
    assert torch.equal(diagnostics["friend_attention"][:, 1:], torch.zeros(2, 2))
    obs[:, 11 + 7] = 0.0
    features_all_dead, diagnostics_all_dead = actor.encode(obs)
    assert torch.equal(diagnostics_all_dead["friend_attention"], torch.zeros(2, len(RED_IDS) - 1))
    assert torch.isfinite(features).all() and torch.isfinite(features_all_dead).all()


def test_hrta_backward_produces_finite_gradients():
    actor = HRTAActor()
    obs = observations(6)
    actions, log_prob = actor.sample(obs)
    features, diagnostics = actor.encode(obs)
    loss = actions.square().mean() + log_prob.square().mean() + features.square().mean()
    loss = loss + diagnostics["enemy_attention"][:, 0].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in actor.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hrta_independent_actors_are_distinct_and_do_not_share_parameters():
    actors = HRTAIndependentActors()
    assert len(actors.actors) == len(RED_IDS)
    assert len({id(actor) for actor in actors.actors}) == len(RED_IDS)
    parameter_ids = [{id(parameter) for parameter in actor.parameters()} for actor in actors.actors]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j]) for i in range(len(RED_IDS)) for j in range(i + 1, len(RED_IDS)))
