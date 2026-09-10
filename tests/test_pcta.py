from __future__ import annotations

import numpy as np
import torch

from algorithm.modules.pcta import PCTAActor, PCTAIndependentActors, pursuit_consistency
from env.mavuav import OBS_DIM, RED_IDS


def observations(*shape: int) -> torch.Tensor:
    torch.manual_seed(2027)
    values = torch.randn(*shape, OBS_DIM) * 0.15
    for start in (44, 58, 72, 86):
        values[..., start + 9] = 1.0
        values[..., start + 10] = 1.0
        values[..., start + 11] = 0.0
    return values


def test_pcta_actor_shape_range_and_sample_evaluate_consistency():
    actor = PCTAActor(); obs = observations(8)
    actions, sampled = actor.sample(obs)
    evaluated, entropy = actor.evaluate_actions(obs, actions)
    assert actions.shape == (8, 3) and sampled.shape == evaluated.shape == entropy.shape == (8,)
    assert torch.isfinite(actions).all() and torch.all(actions.abs() <= 1.0)
    assert torch.allclose(sampled, evaluated, atol=2e-5, rtol=2e-5)


def test_attention_normalizes_over_valid_targets_and_masks_dead_invisible_slots():
    actor = PCTAActor(); obs = observations(4)
    obs[..., 44 + 9] = 0.0
    obs[..., 58 + 10] = obs[..., 58 + 11] = 0.0
    weights = actor.attention_weights(obs)
    assert torch.equal(weights[:, :2], torch.zeros(4, 2))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_no_valid_blue_has_zero_attention_and_finite_action():
    actor = PCTAActor(); obs = observations(5)
    for start in (44, 58, 72, 86):
        obs[..., start + 10] = obs[..., start + 11] = 0.0
    weights = actor.attention_weights(obs)
    actions, log_prob = actor.sample(obs)
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.isfinite(actions).all() and torch.isfinite(log_prob).all()


def test_enemy_slots_share_one_encoder_and_red_actors_are_parameter_independent():
    actor = PCTAActor()
    assert len([name for name, _ in actor.named_modules() if name == "enemy_encoder"]) == 1
    actors = PCTAIndependentActors()
    parameter_ids = [{id(parameter) for parameter in red_actor.parameters()} for red_actor in actors.actors]
    assert len(actors.actors) == len(RED_IDS)
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j]) for i in range(4) for j in range(i + 1, 4))


def temporal_inputs():
    actor = PCTAActor()
    obs = observations(3, 2)
    terminated = torch.zeros(3, 2, dtype=torch.bool)
    truncated = torch.zeros(3, 2, dtype=torch.bool)
    active = torch.ones(3, 2)
    return actor, obs, terminated, truncated, active


def test_episode_boundary_and_inactive_agent_remove_temporal_pairs():
    actor, obs, terminated, truncated, active = temporal_inputs()
    baseline = pursuit_consistency(actor, obs, terminated, truncated, active)
    assert baseline.valid_pairs == 4
    terminated[0, 0] = True
    truncated[1, 0] = True
    active[2, 1] = 0.0
    filtered = pursuit_consistency(actor, obs, terminated, truncated, active)
    assert filtered.valid_pairs == 1


def test_previous_target_death_or_invisibility_releases_consistency():
    actor, obs, terminated, truncated, active = temporal_inputs()
    with torch.no_grad():
        previous_target = actor.attention_weights(obs)[0].argmax(dim=-1)
    for env_index, slot in enumerate(previous_target.tolist()):
        start = (44, 58, 72, 86)[slot]
        obs[1, env_index, start + 9] = 0.0 if env_index == 0 else 1.0
        obs[1, env_index, start + 10] = 0.0
        obs[1, env_index, start + 11] = 0.0
    result = pursuit_consistency(actor, obs[:2], terminated[:2], truncated[:2], active[:2])
    assert result.valid_pairs == 0


def test_valid_consistency_loss_has_finite_gradient_and_switch_diagnostic():
    actor, obs, terminated, truncated, active = temporal_inputs()
    obs[1:, :, 44:100] += torch.linspace(-0.3, 0.3, 56)
    result = pursuit_consistency(actor, obs, terminated, truncated, active)
    assert result.valid_pairs == 4 and torch.isfinite(result.raw_loss)
    result.raw_loss.backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert 0 <= result.target_switches <= result.valid_pairs
