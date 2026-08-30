from __future__ import annotations

import torch
from torch.distributions import Normal

from algorithm.modules.hrta import HRTAActor
from algorithm.modules.structured_uniform import (
    StructuredUniformActor,
    StructuredUniformIndependentActors,
    masked_uniform_pool,
)
from env.mavuav import OBS_DIM


def observations(batch: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4217)
    values = torch.randn(batch, OBS_DIM, generator=generator) * 0.2
    values[:, 7:10] = torch.tensor([1.0, 0.0, 0.0])
    values[:, (11 + 7, 22 + 7)] = 1.0
    for start in (33, 44):
        values[:, start + 6] = 1.0
        values[:, start + 7] = 1.0
        values[:, start + 8] = 0.0
    return values


def test_masked_uniform_pool_all_eligibility_cases_and_large_masked_value():
    embeddings = torch.tensor([
        [[2.0, 4.0], [6.0, 8.0]],
        [[2.0, 4.0], [1.0e30, -1.0e30]],
        [[1.0e30, -1.0e30], [6.0, 8.0]],
        [[1.0e30, -1.0e30], [-1.0e30, 1.0e30]],
    ])
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    context, weights = masked_uniform_pool(embeddings, mask)
    assert torch.equal(weights, torch.tensor([
        [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
    ]))
    assert torch.equal(context, torch.tensor([
        [4.0, 6.0], [2.0, 4.0], [6.0, 8.0], [0.0, 0.0],
    ]))
    assert torch.isfinite(context).all() and torch.isfinite(weights).all()


def test_masked_uniform_pool_rejects_shape_mismatch():
    try:
        masked_uniform_pool(torch.zeros(2, 2, 4), torch.ones(2, 3, dtype=torch.bool))
    except ValueError as error:
        assert "eligible shape" in str(error)
    else:
        raise AssertionError("shape mismatch must be rejected")


def test_structured_uniform_observation_masks_match_hrta_semantics():
    obs = observations(5)
    # Both eligible; Blue1 alive but invisible; Blue1 dead with stale visibility;
    # Blue1 direct-visible only; Blue1 datalink-visible only. Blue2 is disabled after row 0.
    obs[1:, 44 + 6] = 0.0
    obs[1, 33 + 7] = obs[1, 33 + 8] = 0.0
    obs[2, 33 + 6] = 0.0
    obs[2, 33 + 7] = 1.0
    obs[3, 33 + 7], obs[3, 33 + 8] = 1.0, 0.0
    obs[4, 33 + 7], obs[4, 33 + 8] = 0.0, 1.0
    obs[3, 11 + 7], obs[3, 22 + 7] = 1.0, 0.0
    obs[4, 11 + 7], obs[4, 22 + 7] = 0.0, 0.0
    features, diagnostics = StructuredUniformActor().encode(obs)
    assert torch.equal(diagnostics["enemy_pooling_weights"], torch.tensor([
        [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0],
    ]))
    assert torch.equal(diagnostics["friend_pooling_weights"][0], torch.tensor([0.5, 0.5]))
    assert torch.equal(diagnostics["friend_pooling_weights"][3], torch.tensor([1.0, 0.0]))
    assert torch.equal(diagnostics["friend_pooling_weights"][4], torch.tensor([0.0, 0.0]))
    assert torch.isfinite(features).all()


def test_structured_uniform_actor_distribution_and_independent_actor_semantics():
    actor = StructuredUniformActor()
    obs = observations(8)
    deterministic, _ = actor.sample(obs, deterministic=True)
    sampled, sampled_log_prob = actor.sample(obs)
    evaluated_log_prob, entropy = actor.evaluate_actions(obs, sampled)
    assert deterministic.shape == sampled.shape == (8, 3)
    assert sampled_log_prob.shape == evaluated_log_prob.shape == entropy.shape == (8,)
    assert torch.all(deterministic >= -1.0) and torch.all(deterministic <= 1.0)
    assert all(torch.isfinite(value).all() for value in (deterministic, sampled, sampled_log_prob, evaluated_log_prob, entropy))
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=2e-5, rtol=2e-5)

    actors = StructuredUniformIndependentActors()
    assert len({id(item) for item in actors.actors}) == 3
    parameter_ids = [{id(parameter) for parameter in item.parameters()} for item in actors.actors]
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j]) for i in range(3) for j in range(i + 1, 3))


def _hrta_forced_uniform_distribution(actor: HRTAActor, obs: torch.Tensor) -> Normal:
    self_block, friend_blocks, enemy_blocks = actor._blocks(obs)
    self_embedding = actor.self_encoder(self_block)
    role_embedding = actor.role_embedding(self_block[..., 7:10])
    friend_embeddings = actor.friend_encoder(friend_blocks)
    friend_context, _ = masked_uniform_pool(friend_embeddings, friend_blocks[..., 7] > 0.5)
    enemy_embeddings = actor.enemy_encoder(enemy_blocks)
    enemy_mask = (
        (enemy_blocks[..., 6] > 0.5)
        & ((enemy_blocks[..., 7] > 0.5) | (enemy_blocks[..., 8] > 0.5))
    )
    enemy_context, _ = masked_uniform_pool(enemy_embeddings, enemy_mask)
    features = torch.cat((self_embedding, role_embedding, friend_context, enemy_context), dim=-1)
    return Normal(actor.action_head(features), actor.log_std.clamp(-5.0, 2.0).exp())


def test_structured_uniform_matches_hrta_forced_uniform_intervention():
    torch.manual_seed(90210)
    hrta = HRTAActor(entity_dim=12, role_dim=5, fusion_hidden_dim=17)
    uniform = StructuredUniformActor(entity_dim=12, role_dim=5, fusion_hidden_dim=17)
    for name in ("self_encoder", "friend_encoder", "enemy_encoder", "role_embedding", "action_head"):
        getattr(uniform, name).load_state_dict(getattr(hrta, name).state_dict())
    with torch.no_grad():
        uniform.log_std.copy_(hrta.log_std)

    obs = observations(6)
    # Rows cover two, first-only, second-only, zero, alive-invisible and dead-visible.
    obs[1, 44 + 6] = 0.0
    obs[2, 33 + 6] = 0.0
    obs[3, (33 + 6, 44 + 6)] = 0.0
    obs[4, 33 + 7] = obs[4, 33 + 8] = 0.0
    obs[5, 33 + 6] = 0.0
    obs[5, 33 + 7] = 1.0
    obs[1, 22 + 7] = 0.0
    obs[2, 11 + 7] = 0.0
    obs[3, (11 + 7, 22 + 7)] = 0.0

    expected = _hrta_forced_uniform_distribution(hrta, obs)
    actual = uniform._distribution(obs)
    expected_action = torch.tanh(expected.mean)
    actual_action, _ = uniform.sample(obs, deterministic=True)
    assert torch.allclose(actual.mean, expected.mean, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual.scale, expected.scale, atol=0.0, rtol=0.0)
    assert torch.allclose(actual_action, expected_action, atol=1e-7, rtol=1e-7)
    assert torch.equal(uniform.log_std, hrta.log_std)
