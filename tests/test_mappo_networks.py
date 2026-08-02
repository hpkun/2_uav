import torch
from uav_combat.mappo.networks import CentralizedCritic, GaussianActor


def test_actor_shapes_probabilities_and_gradients():
    torch.manual_seed(0)
    actor = GaussianActor(hidden_dim=32)
    observation = torch.randn(7, 14)
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
    actor.log_std.data.fill_(100)
    assert torch.allclose(actor._distribution(observation).scale, torch.full((7, 3), torch.exp(torch.tensor(2.0))))


def test_centralized_critic_batch_output():
    value = CentralizedCritic(hidden_dim=32)(torch.randn(5, 14))
    assert value.shape == (5,) and torch.isfinite(value).all()


def test_actor_log_prob_jacobian_consistency_and_squashed_entropy_is_finite():
    torch.manual_seed(11)
    actor = GaussianActor(68, 3, hidden_dim=32, log_std_init=-0.2)
    obs = torch.randn(5, 68)
    action, sampled_logp = actor.sample_action(obs)
    evaluated_logp, raw_entropy = actor.evaluate_actions(obs, action)
    assert torch.allclose(sampled_logp, evaluated_logp, atol=2e-4)
    assert torch.isfinite(raw_entropy).all()

    estimates = []
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        dist = actor._distribution(obs)
        for _ in range(8):
            raw = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=generator)
            squashed = torch.tanh(raw)
            logp = (dist.log_prob(raw) - torch.log(1.0 - squashed.square() + actor.epsilon)).sum(-1)
            estimates.append(-logp)
    squashed_entropy = torch.cat(estimates).mean()
    assert torch.isfinite(squashed_entropy)


def test_raw_entropy_can_coexist_with_saturated_deterministic_actions_and_log_std_by_dim():
    actor = GaussianActor(68, 3, hidden_dim=16, log_std_init=0.5)
    with torch.no_grad():
        for param in actor.network.parameters():
            param.zero_()
        actor.network[-1].bias[:] = torch.tensor([5.0, -5.0, 0.0])
        actor.log_std[:] = torch.tensor([-1.0, 0.0, 0.5])
    obs = torch.zeros(4, 68)
    det = actor.deterministic_action(obs)
    _, raw_entropy = actor.evaluate_actions(obs, det.clamp(-0.999, 0.999))
    assert torch.all(det[:, 0] > 0.99)
    assert torch.all(det[:, 1] < -0.99)
    assert torch.isfinite(raw_entropy).all()
    assert torch.allclose(actor.log_std.clamp(actor.log_std_min, actor.log_std_max), torch.tensor([-1.0, 0.0, 0.5]))

    with torch.no_grad():
        old = det.clone()
        actor.log_std[:] = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(actor.deterministic_action(obs), old)
