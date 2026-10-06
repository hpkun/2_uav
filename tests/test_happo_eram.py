"""ERAM contracts, masking, ordered PPO, and tiny real-environment round trips."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.trainer import HAPPOTrainer
import algorithm.happo.trainer as trainer_module
from algorithm.happo.entity_layout import (
    SELF_FIELDS, FRIEND_FIELDS, ENEMY_FIELDS, AIRCRAFT_FIELDS,
    parse_observation, parse_global_state,
)
from algorithm.happo.eram import (
    EntityRecurrentActor, EntityAttentionRecurrentCritic, ERAMRolloutBuffer,
)
from algorithm.happo.evaluation import evaluate_recurrent_actors, summarize_records
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv, RED_IDS, load_environment_config
from env.vector_env import MAVUAVVectorEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_small_tests(monkeypatch):
    # Tests only; production continues using real subprocess vectorization.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(trainer_module, "MAVUAVVectorEnv",
                        lambda *args, **kwargs: MAVUAVVectorEnv(*args, parallel=False, **kwargs))
    yield
    torch.set_num_threads(old_threads)


def env_config(horizon=None):
    c = load_environment_config(ROOT / "configs/env_v311.yaml")
    if horizon is not None:
        c["simulation"]["max_decision_steps"] = horizon
    return c


def config(**updates):
    c = dict(actor_variant="entity_recurrent", critic_variant="entity_attention_recurrent",
             method_variant="baseline", num_envs=1, rollout_steps=4, ppo_epochs=1,
             minibatch_size=4, recurrent_sequence_length=2, device="cpu", seed=7,
             environment_profile="learnability", actor_log_std_init=-0.25,
             eram_entity_dim=8, eram_actor_attention_heads=2, eram_actor_fusion_dim=12,
             eram_actor_recurrent_hidden_dim=12, eram_critic_token_dim=12,
             eram_critic_attention_heads=2, eram_critic_recurrent_hidden_dim=12)
    c.update(updates)
    return c


def observation():
    x = torch.randn(2, 100)
    p = parse_observation(x)
    p.self_features[..., SELF_FIELDS.index("alive")] = 1
    p.friends[..., FRIEND_FIELDS.index("alive")] = 1
    p.enemies[..., ENEMY_FIELDS.index("alive")] = 1
    p.enemies[..., ENEMY_FIELDS.index("direct")] = 1
    p.enemies[..., ENEMY_FIELDS.index("datalink")] = 0
    return x


def test_named_parser_lossless_matches_environment():
    env = HeterogeneousMAVUAVAirCombatEnv(env_config(), randomize=False)
    obs, _ = env.reset(seed=1)
    x = torch.tensor(np.stack([obs[aid] for aid in RED_IDS]))
    p = parse_observation(x)
    assert p.self_features.shape == (4, 11)
    assert p.friends.shape == (4, 3, 11) and p.enemies.shape == (4, 4, 14)
    assert torch.equal(x, torch.cat((p.self_features, p.friends.flatten(1), p.enemies.flatten(1)), -1))
    states = torch.tensor(env.global_state())[None]
    entities, context, alive = parse_global_state(states)
    assert entities.shape == (1, 8, 10) and context.shape == (1, 37)
    assert alive.all() and torch.equal(states, torch.cat((entities.flatten(1), context), -1))
    # Verify actual builder semantics, rather than dimensions alone.
    assert p.self_features[0, SELF_FIELDS.index("altitude")] == states[0, AIRCRAFT_FIELDS.index("altitude")]
    assert p.friend_valid.all() and p.alive.all()


def test_nominal_datalink_only_is_valid():
    env = HeterogeneousMAVUAVAirCombatEnv(env_config(), randomize=False)
    obs, _ = env.reset(seed=1)
    p = parse_observation(torch.tensor(np.stack([obs[aid] for aid in RED_IDS])))
    assert p.enemy_direct[0].all()
    assert not p.enemy_direct[1:].any()
    assert p.enemy_datalink[1:].all() and p.enemy_valid.all()
    actor = EntityRecurrentActor()
    d = actor.attention_diagnostics(torch.tensor(obs["UAV1"])[None])
    assert d["valid_enemy_count"].item() == 4
    assert d["enemy_datalink_only_mass"].item() == pytest.approx(1.0)
    assert d["enemy_direct_mass"].item() == 0


def test_dead_invisible_entities_are_excluded_not_datalink_only():
    x = observation()
    p = parse_observation(x)
    p.friends[:, 0, FRIEND_FIELDS.index("alive")] = 0
    p.enemies[:, 0, ENEMY_FIELDS.index("alive")] = 0
    p.enemies[:, 1, ENEMY_FIELDS.index("direct")] = 0
    p.enemies[:, 2, ENEMY_FIELDS.index("direct")] = 0
    p.enemies[:, 2, ENEMY_FIELDS.index("datalink")] = 1
    actor = EntityRecurrentActor()
    _, (_, _, _, aw, ew) = actor.encode(x, return_details=True)
    assert not p.enemy_valid[:, :2].any() and p.enemy_valid[:, 2:].all()
    assert (aw[:, 0] == 0).all() and (ew[:, :2] == 0).all()
    # Invalid local entities cannot influence the action or context.
    perturbed = x.clone()
    q = parse_observation(perturbed)
    q.friends[:, 0, :7] += 100
    q.enemies[:, :2, :9] -= 100
    a = actor.forward_step(x, actor.initial_hidden(2), torch.zeros(2))[0]
    b = actor.forward_step(perturbed, actor.initial_hidden(2), torch.zeros(2))[0]
    assert torch.equal(a, b)


@pytest.mark.parametrize("empty", ["allies", "enemies", "both"])
def test_all_masked_zero_context_and_finite_gradients(empty):
    x = observation()
    p = parse_observation(x)
    if empty in ("allies", "both"):
        p.friends[..., FRIEND_FIELDS.index("alive")] = 0
    if empty in ("enemies", "both"):
        p.enemies[..., ENEMY_FIELDS.index("direct")] = 0
    actor = EntityRecurrentActor()
    features, (_, ally, enemy, _, _) = actor.encode(x, True)
    d = actor.attention_diagnostics(x)
    assert all(torch.isfinite(v).all() for v in d.values())
    if empty in ("allies", "both"):
        assert torch.count_nonzero(ally) == 0 and (d["ally_attention_entropy"] == 0).all()
    if empty in ("enemies", "both"):
        assert torch.count_nonzero(enemy) == 0 and (d["enemy_attention_entropy"] == 0).all()
        assert (d["enemy_direct_mass"] == 0).all() and (d["enemy_datalink_only_mass"] == 0).all()
    features.square().mean().backward()
    assert all(torch.isfinite(v.grad).all() for v in actor.parameters() if v.grad is not None)


def test_critic_masks_dead_aircraft_keeps_context_and_no_visibility(monkeypatch):
    env = HeterogeneousMAVUAVAirCombatEnv(env_config(), randomize=False)
    env.reset(seed=1)
    states = torch.tensor(env.global_state())[None]
    entities, _, valid = parse_global_state(states)
    entities[:, 2, AIRCRAFT_FIELDS.index("alive")] = 0
    critic = EntityAttentionRecurrentCritic()
    seen = []
    original = critic.attention.forward
    def spy(*args, **kwargs):
        seen.append(kwargs["key_padding_mask"].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(critic.attention, "forward", spy)
    value, _ = critic.forward_step(states, critic.initial_hidden(1), torch.zeros(1))
    assert seen[0][0, 2] and not seen[0][0, -1]
    assert not seen[0][0, 4:8].any()  # all alive Blues, independent of actor sensing
    entities[..., AIRCRAFT_FIELDS.index("alive")] = 0
    value, _ = critic.forward_step(states, critic.initial_hidden(1), torch.zeros(1))
    assert seen[1][0, :8].all() and not seen[1][0, -1] and torch.isfinite(value).all()


def test_actor_reset_and_death_cut_gradient():
    actor = EntityRecurrentActor(recurrent_hidden_dim=12)
    x = observation()
    hidden = torch.randn(2, 12, requires_grad=True)
    reset, _ = actor.forward_step(x, hidden, torch.zeros(2))
    zero, _ = actor.forward_step(x, torch.zeros_like(hidden), torch.zeros(2))
    assert torch.equal(reset, zero)
    reset.sum().backward()
    assert torch.count_nonzero(hidden.grad) == 0
    parse_observation(x).self_features[..., SELF_FIELDS.index("alive")] = 0
    action, logp, new = actor.sample_step(x, hidden, torch.ones(2))
    assert torch.count_nonzero(action) == torch.count_nonzero(logp) == torch.count_nonzero(new) == 0
    lp, entropy, _ = actor.evaluate_actions_sequence(x[:, None], action[:, None], hidden, torch.ones(2, 1))
    assert torch.count_nonzero(lp) == torch.count_nonzero(entropy) == 0


def test_actor_and_critic_sequence_equals_ordered_unroll_and_reset_gradient():
    actor = EntityRecurrentActor(recurrent_hidden_dim=12)
    x = observation()[:, None].repeat(1, 5, 1)
    actions = torch.tanh(torch.randn(2, 5, 3))
    masks = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 0, 1.]])
    initial = torch.randn(2, 12)
    lp, ent, h = actor.evaluate_actions_sequence(x, actions, initial, masks)
    manual, hidden = [], initial
    for t in range(5):
        l, e, hidden = actor.evaluate_actions_sequence(x[:, t:t+1], actions[:, t:t+1], hidden, masks[:, t:t+1])
        manual.append(l)
    assert torch.equal(lp, torch.cat(manual, 1)) and torch.equal(h, hidden)
    critic = EntityAttentionRecurrentCritic(recurrent_hidden_dim=12)
    states = torch.randn(2, 5, 117)
    values, h = critic.evaluate_values_sequence(states, initial, masks)
    manual, hidden = [], initial
    for t in range(5):
        v, hidden = critic.forward_step(states[:, t], hidden, masks[:, t])
        manual.append(v)
    assert torch.equal(values, torch.stack(manual, 1)) and torch.equal(h, hidden)
    initial = torch.randn(2, 12, requires_grad=True)
    value, _ = critic.forward_step(states[:, 0], initial, torch.zeros(2))
    value.sum().backward()
    assert torch.count_nonzero(initial.grad) == 0


def test_rollout_persistence_buffers_chunks_and_independent_optimizers():
    t = HAPPOTrainer(env_config(), config())
    try:
        t.collect_rollout()
        old_a, old_c = t.actor_hidden_states.copy(), t.critic_hidden_states.copy()
        assert np.any(old_a) and np.any(old_c)
        t.buffer = t.make_buffer(3)
        t.collect_rollout()
        np.testing.assert_array_equal(t.buffer.actor_hidden_states[0], old_a)
        np.testing.assert_array_equal(t.buffer.critic_hidden_states[0], old_c)
        assert isinstance(t.buffer, ERAMRolloutBuffer)
        assert t.buffer.chunks(2) == [(0, 0, 2), (0, 2, 3)]
        assert t.buffer.actor_hidden_states.shape == (4, 1, 4, 12)
        assert t.buffer.critic_hidden_states.shape == (4, 1, 12)
        obs, _, h, mask = t._recurrent_sequence_tensors(1, [(0, 0, 2)])
        np.testing.assert_array_equal(obs.numpy()[0], t.buffer.observations[:2, 0, 1])
        np.testing.assert_array_equal(h.numpy()[0], t.buffer.actor_hidden_states[0, 0, 1])
        ids = [{id(p) for p in actor.parameters()} for actor in t.actors.actors]
        assert all(ids[i].isdisjoint(ids[j]) for i in range(4) for j in range(i+1, 4))
        assert len({id(opt) for opt in t.actor_optimizers}) == 4
    finally:
        t.close()


@pytest.mark.parametrize("boundary", ["terminated", "truncated", "death"])
def test_rollout_resets_actual_masks(boundary, monkeypatch):
    t = HAPPOTrainer(env_config(), config(rollout_steps=1))
    try:
        original = t.vector_env.step
        def step(actions):
            result = list(original(actions))
            if boundary == "death":
                result[5][:, 1] = 0
            else:
                result[3 if boundary == "terminated" else 4][:] = True
                result[5][:] = 1  # auto-reset masks must never leak across episode
            return tuple(result)
        monkeypatch.setattr(t.vector_env, "step", step)
        t.collect_rollout()
        if boundary == "death":
            assert not t.actor_hidden_states[:, 1].any() and t.actor_hidden_states[:, 2].any()
            assert t.critic_hidden_states.any()
        else:
            assert not t.actor_hidden_states.any() and not t.critic_hidden_states.any()
            assert not t.actor_recurrent_masks.any() and not t.critic_recurrent_masks.any()
    finally:
        t.close()


def test_real_sequential_factor_and_inactive_actor_update(monkeypatch):
    t = HAPPOTrainer(env_config(), config())
    try:
        # Kill a UAV before rollout using genuine environment state (test only).
        t.vector_env.envs[0].entities["UAV1"].state.alive = False
        obs = t.vector_env.envs[0]._observations()
        t.observations[0] = np.stack([obs[aid] for aid in RED_IDS])
        t.global_states[0] = t.vector_env.envs[0].global_state()
        t.active_masks[0, 1] = 0
        t.collect_rollout()
        assert not t.buffer.actions[:, :, 1].any()
        before = deepcopy(t.actors.actors[1].state_dict())
        calls = []
        original = trainer_module.preceding_factor_update
        def spy(factor, old, new, active):
            result = original(factor, old, new, active)
            calls.append((factor.clone(), result.clone(), active.clone()))
            return result
        monkeypatch.setattr(trainer_module, "preceding_factor_update", spy)
        metrics = t.update()
        assert metrics["actor_1_loss"] == 0
        assert sorted(metrics["agent_update_order"]) == list(range(4)) and len(calls) == 4
        assert len(t.last_recurrent_factor_history) == 5
        inactive = calls[metrics["agent_update_order"].index(1)]
        assert torch.equal(inactive[0], inactive[1]) and not inactive[2].any()
        assert any(not torch.equal(a, b) for a, b, m in calls if m.any())
        assert all(torch.equal(v, t.actors.actors[1].state_dict()[k]) for k, v in before.items())
    finally:
        t.close()


def test_tiny_cpu_update_checkpoint_exact_resume_and_weights_metadata(tmp_path):
    t = HAPPOTrainer(env_config(8), config())
    other = None
    try:
        t.collect_rollout()
        metrics = t.update()
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))
        path = tmp_path / "checkpoint.pt"
        t.save_checkpoint(path)
        payload = torch.load(path, weights_only=False)
        assert payload["algorithm"] == "eram_happo" and payload["base_algorithm"] == "happo"
        assert payload["independent_actor_count"] == 4 and payload["action_dim"] == 3
        assert payload["sequential_happo_update"] and payload["sampled_steps"] == 4
        assert payload["actor_architecture"] == t.actor_architecture
        assert payload["critic_architecture"] == t.critic_architecture
        assert len(payload["actor_optimizer_states"]) == 4
        # Exact continuation, including both memories, environment states and RNG.
        t.buffer = t.make_buffer(4)
        t.collect_rollout()
        expected_actions = t.buffer.actions.copy()
        t.update()
        expected = deepcopy(t.actors.state_dict()), deepcopy(t.critic.state_dict())
        other = HAPPOTrainer(env_config(8), config())
        assert other.load_checkpoint(path) == 4
        np.testing.assert_array_equal(other.actor_hidden_states, payload["rollout_state"]["actor_hidden_states"])
        np.testing.assert_array_equal(other.critic_hidden_states, payload["rollout_state"]["critic_hidden_states"])
        other.collect_rollout()
        np.testing.assert_array_equal(other.buffer.actions, expected_actions)
        other.update()
        assert all(torch.equal(v, other.actors.state_dict()[k]) for k, v in expected[0].items())
        assert all(torch.equal(v, other.critic.state_dict()[k]) for k, v in expected[1].items())
        t.save(tmp_path / "weights.pt")
        weights = torch.load(tmp_path / "weights.pt", weights_only=False)
        assert weights["algorithm"] == "eram_happo" and weights["actor_active_mask"] == payload["actor_active_mask"]
        records = evaluate_recurrent_actors(other.actors, env_config(2), 1, "learnability")
        assert records[0]["episode_length"] <= 2
        assert "eram_UAV1_enemy_datalink_only_mass" in summarize_records(records)
    finally:
        t.close()
        if other:
            other.close()


@pytest.mark.parametrize("change", ["eram_entity_dim", "eram_critic_token_dim", "recurrent_sequence_length"])
def test_resume_rejects_architecture_or_sequence_drift(change, tmp_path):
    t = HAPPOTrainer(env_config(), config())
    u = None
    try:
        path = tmp_path / "checkpoint.pt"
        t.save_checkpoint(path)
        changed = config()
        changed[change] *= 2
        u = HAPPOTrainer(env_config(), changed)
        with pytest.raises(RuntimeError, match="architecture|ERAM"):
            u.load_checkpoint(path)
    finally:
        t.close()
        if u:
            u.close()


def test_stochastic_repeatability_attention_diagnostics_invariance():
    t = HAPPOTrainer(env_config(3), config())
    try:
        kwargs = dict(episodes=2, profile="learnability", seed=1000,
                      deterministic=False, action_seed=2000)
        a = evaluate_recurrent_actors(t.actors, env_config(3), **kwargs)
        b = evaluate_recurrent_actors(t.actors, env_config(3), **kwargs)
        c = evaluate_recurrent_actors(t.actors, env_config(3), **kwargs, collect_entity_diagnostics=False)
        assert a == b
        assert [{k: v for k, v in row.items() if not k.startswith("eram_")} for row in a] == c
        assert torch.isfinite(torch.tensor([v for row in a for k, v in row.items() if k.startswith("eram_")])).all()
    finally:
        t.close()


def test_eram_config_fairness_and_old_tam_contract():
    eram = yaml.safe_load((ROOT / "configs/happo_eram_v311.yaml").read_text())["training"]
    base = yaml.safe_load((ROOT / "configs/happo_v310_ablation.yaml").read_text())["training"]
    assert {k: eram[k] for k in base if k not in ("actor_variant", "critic_variant")} == {
        k: v for k, v in base.items() if k not in ("actor_variant", "critic_variant")}
    with pytest.raises(ValueError, match="v3.11"):
        HAPPOTrainer(ROOT / "configs/env_v310.yaml", config())
    with pytest.raises(ValueError, match="TAM-HAPPO requires"):
        HAPPOTrainer(env_config(), dict(actor_variant="tam", critic_variant="tam_attention", num_envs=1))


def test_standalone_auto_selects_eram_and_preserves_environment_contract(tmp_path, monkeypatch):
    import algorithm.evaluate_happo as evaluator
    t = HAPPOTrainer(env_config(2), config())
    try:
        checkpoint = tmp_path / "checkpoint_final.pt"
        t.save_checkpoint(checkpoint)
        monkeypatch.setattr(sys, "argv", ["evaluate_happo.py", str(checkpoint), "--device", "cpu",
                                         "--episodes", "1", "--profile", "learnability",
                                         "--action-mode", "stochastic"])
        evaluator.main()
        result = json.loads((tmp_path / "evaluation_final_stochastic_summary.json").read_text())
        assert result["algorithm"] == "eram_happo" and result["action_seed"] == 2000
        assert result["results"][0]["eram_UAV1_valid_enemy_count"] == 4
        assert (tmp_path / "evaluation_final_stochastic.csv").exists()
    finally:
        t.close()


def test_default_network_dimensions_and_no_global_actor_input():
    actor, critic = EntityRecurrentActor(), EntityAttentionRecurrentCritic()
    assert actor.self_encoder[0].in_features == 11 and actor.self_encoder[0].out_features == 64
    assert actor.friend_encoder[0].in_features == 11 and actor.enemy_encoder[0].in_features == 14
    assert actor.ally_attention.num_heads == actor.enemy_attention.num_heads == 4
    assert actor.fusion[0].in_features == 192 and actor.fusion[0].out_features == 128
    assert actor.gru.input_size == actor.gru.hidden_size == 128
    assert critic.gru.input_size == 117 and critic.gru.hidden_size == 128
    assert critic.context_encoder[0].in_features == 165
    assert isinstance(critic.value_head[1], torch.nn.LayerNorm)
    assert isinstance(critic.value_head[4], torch.nn.LayerNorm)
    with pytest.raises(ValueError, match="100"):
        actor.forward_step(torch.zeros(1, 117), actor.initial_hidden(1), torch.zeros(1))


def test_actual_ppo_batches_keep_time_order_and_chunk_initial_hidden(monkeypatch):
    t = HAPPOTrainer(env_config(), config(rollout_steps=5))
    try:
        t.collect_rollout()
        actor_seen, critic_seen = [], []
        for aid, actor in enumerate(t.actors.actors):
            original = actor.evaluate_actions_sequence
            def spy(obs, actions, hidden, masks, aid=aid, original=original):
                for row in range(obs.shape[0]):
                    matches = []
                    for env, start, end in t.buffer.chunks(2):
                        if end-start != obs.shape[1]:
                            continue
                        expected = torch.as_tensor(t.buffer.observations[start:end, env, aid])
                        if torch.equal(obs[row], expected):
                            matches.append((env, start, end))
                    assert len(matches) == 1
                    env, start, end = matches[0]
                    assert torch.equal(hidden[row], torch.as_tensor(t.buffer.actor_hidden_states[start, env, aid]))
                    assert torch.equal(masks[row], torch.as_tensor(t.buffer.recurrent_masks[start:end, env, aid]))
                    actor_seen.append((aid, start, end))
                return original(obs, actions, hidden, masks)
            monkeypatch.setattr(actor, "evaluate_actions_sequence", spy)
        original_critic = t.critic.evaluate_values_sequence
        def critic_spy(states, hidden, masks):
            for row in range(states.shape[0]):
                matches = [spec for spec in t.buffer.chunks(2) if spec[2]-spec[1] == states.shape[1]
                           and torch.equal(states[row], torch.as_tensor(t.buffer.global_states[spec[1]:spec[2], spec[0]]))]
                assert len(matches) == 1
                env, start, end = matches[0]
                assert torch.equal(hidden[row], torch.as_tensor(t.buffer.critic_hidden_states[start, env]))
                assert torch.equal(masks[row], torch.as_tensor(t.buffer.critic_recurrent_masks[start:end, env]))
                critic_seen.append((start, end))
            return original_critic(states, hidden, masks)
        monkeypatch.setattr(t.critic, "evaluate_values_sequence", critic_spy)
        t.update()
        assert set(critic_seen) == {(0, 2), (2, 4), (4, 5)}
        assert len(actor_seen) == 4 * 3 * 2  # optimization + likelihood-ratio re-evaluation
    finally:
        t.close()


def test_entrypoint_two_step_cpu_smoke_records_resolved_summary(tmp_path, monkeypatch):
    import algorithm.train_happo as entry
    from algorithm.train_eram_happo import main
    small_config = tmp_path / "small.yaml"
    small_env = tmp_path / "env.yaml"
    small_config.write_text(yaml.safe_dump({"training": config(rollout_steps=2)}))
    small_env.write_text(yaml.safe_dump(env_config(2)))
    monkeypatch.setattr(entry, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_eram_happo.py", "--steps", "2", "--num-envs", "1",
                                     "--device", "cpu", "--profile", "learnability", "--seed", "7",
                                     "--config", str(small_config), "--env-config", str(small_env),
                                     "--output-name", "tiny", "--checkpoint-interval", "0",
                                     "--eval-interval", "0", "--final-eval-episodes", "1"])
    defaults = entry.DEFAULT_CONFIG, entry.DEFAULT_ENV_CONFIG
    try:
        main()
    finally:
        entry.DEFAULT_CONFIG, entry.DEFAULT_ENV_CONFIG = defaults
    resolved = yaml.safe_load((tmp_path / "tiny/resolved_config.yaml").read_text())
    summary = json.loads((tmp_path / "tiny/summary.json").read_text())
    assert resolved["algorithm"] == summary["algorithm"] == "eram_happo"
    assert resolved["happo"]["actor_variant"] == "entity_recurrent"
    assert summary["sampled_steps"] == 2 and summary["sequential_happo_update"]
    assert summary["actor_recurrent_configuration"]["hidden_dim"] == 12
    assert "eram_UAV3_valid_enemy_count" in summary["final_evaluations"][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tiny_cuda_collect_update_roundtrip_finite(tmp_path):
    t = HAPPOTrainer(env_config(3), config(device="cuda", rollout_steps=2))
    u = None
    try:
        t.collect_rollout()
        metrics = t.update()
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))
        path = tmp_path / "cuda.pt"
        t.save_checkpoint(path)
        u = HAPPOTrainer(env_config(3), config(device="cuda", rollout_steps=2))
        assert u.load_checkpoint(path) == 2
        u.collect_rollout()
        assert all(np.isfinite(v) for v in u.update().values() if isinstance(v, float))
    finally:
        t.close()
        if u:
            u.close()
