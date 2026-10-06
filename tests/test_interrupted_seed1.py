from copy import deepcopy
import json
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

from algorithm.signal_checkpoint import SignalCheckpoint
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import RED_IDS, load_environment_config
from tools.audit_interrupted_seed1 import (
    death_and_cleanup, load_actors, replay, training_analysis, finite_tree,
)
from tools.run_happo_tacm_seed1 import exit_status
from algorithm.happo.evaluation import evaluate_actors, summarize_records

ROOT = Path(__file__).resolve().parents[1]


def test_signal_only_sets_flag_and_normal_handler_preserves_rng():
    controller = SignalCheckpoint()
    rng = torch.get_rng_state().clone()
    before = signal.getsignal(signal.SIGTERM)
    controller.install()
    controller.handle(signal.SIGTERM, None)
    assert controller.request["signal_name"] == "SIGTERM"
    assert torch.equal(rng, torch.get_rng_state())
    controller.restore()
    assert signal.getsignal(signal.SIGTERM) == before
    assert SignalCheckpoint().save_at_boundary(None, None, None) is None
    # SIGKILL is deliberately absent: it cannot be intercepted.
    assert getattr(signal, "SIGKILL", None) not in controller.previous


@pytest.mark.parametrize("code,status", [(0, "COMPLETE"), (1, "FAILED"), (-15, "TERMINATED_BY_SIGNAL"), (143, "TERMINATED_BY_SIGNAL"), (137, "TERMINATED_BY_SIGNAL")])
def test_launcher_status(code, status):
    assert exit_status(code) == status


def row(outcome="draw"):
    return {"outcome": outcome, "red_attack_kills": 3, "blue_survivors": 1,
            "red_uav_survivors": 2, "mav_survived": True,
            **{f"{aid}_death_cause": "alive" for aid in RED_IDS},
            **{f"{aid}_death_step": None for aid in RED_IDS}}


def test_death_categories_and_cleanup():
    first, second = row(), row("red")
    first.update(UAV1_death_cause="boundary", UAV1_death_step=17)
    second.update(UAV2_death_cause="blue_attack", UAV2_death_step=21,
                  UAV3_death_cause="custom_cause", UAV3_death_step=25)
    deaths, cleanup = death_and_cleanup([first, second])
    assert cleanup["three_kills_one_blue_count"] == 1
    assert cleanup["draw_blue_survivors"]["1"] == 1
    assert cleanup["uav_deaths"] == 3
    assert cleanup["boundary_share_of_uav_deaths"] == 1 / 3
    assert cleanup["last_blue_hypothesis"] == "INCONCLUSIVE"
    assert next(r for r in deaths if r["outcome"] == "all" and r["agent"] == "UAV1")["death_step_median"] == 17
    assert "custom_cause" in next(r for r in deaths if r["outcome"] == "all" and r["agent"] == "UAV3")["raw_causes"]


def test_training_counter_analysis_and_episode_weighting():
    rows = [{"sampled_steps": "2048", "completed_episodes": "2", "mean_episode_return": "1",
             "own_loss_count_UAV1": "2", "own_boundary_loss_count_UAV1": "2"},
            {"sampled_steps": "4096", "completed_episodes": "3", "mean_episode_return": "4",
             "own_loss_count_UAV1": "1", "own_boundary_loss_count_UAV1": "1"}]
    result = training_analysis(rows)
    assert result["phases"][0]["mean_episode_return"] == 2
    assert result["early_boundary_hypothesis"]["status"] == "VERIFIED"
    assert result["early_boundary_hypothesis"]["uav_boundary_losses"] == 3
    with pytest.raises(RuntimeError, match="nonmonotonic"):
        training_analysis([rows[0], rows[0]])


@pytest.mark.parametrize("method", ["baseline", "tacm_rgaa"])
def test_cuda_emergency_resume_actor_load_replay_fidelity(tmp_path, method, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tiny smoke")
    import yaml
    config_name = "happo_v310_ablation.yaml" if method == "baseline" else "happo_tacm_rgaa_v310.yaml"
    c = yaml.safe_load((ROOT / "configs" / config_name).read_text())["training"]
    c.update(device="cuda", num_envs=1, rollout_steps=2, ppo_epochs=1, minibatch_size=2)
    environment = load_environment_config(ROOT / "configs/env_v310.yaml")
    trainer = HAPPOTrainer(environment, c)
    controller = SignalCheckpoint()
    checkpoint = tmp_path / "checkpoint_emergency_2.pt"
    saves = []
    original_save = trainer.save_checkpoint
    def save(path):
        assert path.name.endswith(".tmp")
        assert not checkpoint.exists()
        saves.append(path)
        original_save(path)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    try:
        trainer.collect_rollout()
        # A request made during update cannot serialize until our explicit boundary.
        original_step = trainer.critic_optimizer.step
        def optimizer_step(*args, **kwargs):
            controller.handle(signal.SIGTERM, None)
            assert saves == []
            return original_step(*args, **kwargs)
        monkeypatch.setattr(trainer.critic_optimizer, "step", optimizer_step)
        metrics = trainer.update()
        assert finite_tree(metrics) and saves == []
        with pytest.raises(SystemExit) as stopped:
            controller.save_at_boundary(trainer, tmp_path, lambda message: None)
        assert stopped.value.code == 143
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert finite_tree(payload)
        assert all(key in payload for key in ("actors", "critic", "actor_optimizer_states", "critic_optimizer_state",
            "torch_rng", "cuda_rng", "trainer_numpy_rng", "rollout_state", "environment_config", "trainer_config"))
        assert len(saves) == 1 and not saves[0].exists()
        metadata = json.loads((tmp_path / "termination.json").read_text())
        assert metadata["sampled_steps"] == 2
        assert metadata["termination_reason"] == "external_signal"
        expected_torch = torch.rand(3, device="cuda")
        expected_np = trainer.rng.permutation(12)
    finally:
        trainer.close()
    resumed = HAPPOTrainer(environment, c)
    try:
        assert resumed.load_checkpoint(checkpoint) == 2
        assert torch.equal(expected_torch, torch.rand(3, device="cuda"))
        np.testing.assert_array_equal(expected_np, resumed.rng.permutation(12))
        assert all(optimizer.state_dict()["state"] for optimizer in resumed.actor_optimizers)
        assert resumed.critic_optimizer.state_dict()["state"]
        resumed.collect_rollout()
        assert finite_tree(resumed.update())
    finally:
        resumed.close()
    actors, env = load_actors(payload, "cuda")
    before = {k: v.clone() for k, v in actors.state_dict().items()}
    records = replay(actors, env, 1, "cuda")
    formal = evaluate_actors(actors, env, 1, "main", device="cuda", deterministic=False, action_seed=2000)
    assert summarize_records(records) == summarize_records(formal)
    assert all(torch.equal(v, actors.state_dict()[k]) for k, v in before.items())
    bad = deepcopy(payload)
    bad["environment_config"]["environment_version"] = "heterogeneous_mavuav_4v4_v3_11"
    bad["environment_version"] = "heterogeneous_mavuav_4v4_v3_11"
    with pytest.raises(RuntimeError, match="v3.10"):
        load_actors(bad, "cuda")


def test_accounting_rejects_survivor_mismatch():
    from tools.audit_role_guided_run import validate_death_accounting
    good = row()
    good["red_uav_survivors"] = 3
    validate_death_accounting(good)
    good["MAV_death_cause"] = "boundary"
    with pytest.raises(AssertionError, match="MAV"):
        validate_death_accounting(good)


def test_actual_interrupted_tacm_checkpoint_cuda_resume_tiny(monkeypatch):
    checkpoint = ROOT / "outputs/tacm_v310_main_seed1_2m_20261006_155621/checkpoint_501760.pt"
    if not checkpoint.exists() or not torch.cuda.is_available():
        pytest.skip("optional local interrupted-run CUDA smoke")
    from functools import partial
    import algorithm.happo.trainer as module
    from tools.audit_interrupted_seed1 import digest
    sha = digest(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(payload["trainer_config"])
    config["device"] = "cuda"
    # Identical 16 environment states, serial execution only to avoid spawning
    # 16 Python interpreters for this one-step smoke; no science/config change.
    monkeypatch.setattr(module, "MAVUAVVectorEnv", partial(module.MAVUAVVectorEnv, parallel=False))
    trainer = HAPPOTrainer(payload["environment_config"], config)
    try:
        assert trainer.load_checkpoint(checkpoint) == 501760
        assert torch.equal(torch.get_rng_state(), payload["torch_rng"])
        assert trainer.rng.bit_generator.state == payload["trainer_numpy_rng"]
        assert trainer.rgaa_rng.bit_generator.state == payload["rgaa_numpy_rng"]
        assert all(opt.state_dict()["state"] for opt in trainer.actor_optimizers)
        trainer.buffer = trainer.make_buffer(1)
        trainer.collect_rollout()
        assert trainer.env_steps == 501776
        assert finite_tree(trainer.update())
    finally:
        trainer.close()
    assert digest(checkpoint) == sha


def test_atomic_save_failure_does_not_publish_partial_checkpoint(tmp_path):
    class BrokenTrainer:
        env_steps = 2
        def save_checkpoint(self, path):
            path.write_bytes(b"partial")
            raise OSError("simulated write failure")
    checkpoint = tmp_path / "checkpoint_emergency_2.pt"
    checkpoint.write_bytes(b"previous complete checkpoint")
    controller = SignalCheckpoint()
    controller.handle(signal.SIGTERM, None)
    with pytest.raises(OSError, match="simulated"):
        controller.save_at_boundary(BrokenTrainer(), tmp_path, lambda message: None)
    assert checkpoint.read_bytes() == b"previous complete checkpoint"
    assert not checkpoint.with_suffix(".pt.tmp").exists()
    assert not (tmp_path / "termination.json").exists()


def test_full_cleanup_classification():
    records = [row() for _ in range(200)]
    for record in records:
        record["red_uav_survivors"] = 3
    _, result = death_and_cleanup(records)
    assert result["last_blue_hypothesis"] == "VERIFIED"
    assert result["cleanup_draw_MAV_alive_count"] == 200
    for record in records:
        record.update(red_attack_kills=2, blue_survivors=2)
    assert death_and_cleanup(records)[1]["last_blue_hypothesis"] == "REFUTED"


def test_flat_entry_services_signal_after_complete_update(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA tiny entry smoke")
    import argparse
    import yaml
    import algorithm.train_happo as entry
    c = yaml.safe_load((ROOT / "configs/happo_v310_ablation.yaml").read_text())
    c["training"].update(rollout_steps=2, ppo_epochs=1, minibatch_size=2)
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(c))
    args = argparse.Namespace(steps=4, profile="main", seed=1, device="cuda", num_envs=1,
        config=path, env_config=ROOT / "configs/env_v310.yaml", output_name="signal_smoke",
        checkpoint_interval=0, eval_interval=0, log_interval=0, eval_episodes=1,
        final_eval_episodes=1, eval_action_mode="stochastic", eval_action_seed=2000, resume=None)
    monkeypatch.setattr(entry, "parse_args", lambda: args)
    monkeypatch.setattr(entry, "OUTPUT_ROOT", tmp_path)
    original_update = entry.HAPPOTrainer.update
    previous = signal.getsignal(signal.SIGTERM)
    def update(trainer):
        result = original_update(trainer)
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        assert not list((tmp_path / "signal_smoke").glob("checkpoint*.pt"))
        return result
    monkeypatch.setattr(entry.HAPPOTrainer, "update", update)
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == previous
    run = tmp_path / "signal_smoke"
    assert (run / "checkpoint_emergency_2.pt").exists()
    assert not (run / "checkpoint_final.pt").exists()
    assert not (run / "evaluations.csv").exists()
    assert json.loads((run / "termination.json").read_text())["sampled_steps"] == 2
