import torch
from uav_combat.mappo.networks import CentralizedCritic, SharedActor


def test_actor_shapes_probabilities_and_gradients():
    torch.manual_seed(0)
    actor = SharedActor(hidden_dim=32)
    observation = torch.randn(7, 13)
    action, sampled_log_prob = actor.sample_action(observation)
    evaluated_log_prob, entropy = actor.evaluate_actions(observation, action)
    assert action.shape == (7, 3) and sampled_log_prob.shape == (7,)
    assert torch.all(action > -1) and torch.all(action < 1)
    assert torch.isfinite(action).all() and torch.isfinite(sampled_log_prob).all() and torch.isfinite(entropy).all()
    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=2e-4)
    deterministic = actor.deterministic_action(observation)
    assert deterministic.shape == (7, 3) and torch.isfinite(deterministic).all()
    (-evaluated_log_prob.mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_centralized_critic_batch_output():
    value = CentralizedCritic(hidden_dim=32)(torch.randn(5, 26))
    assert value.shape == (5, 2) and torch.isfinite(value).all()

