"""Production preflight checks with bounded CPU fixtures (not production fallback)."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.trainer import HAPPOTrainer
import algorithm.happo.trainer as trainer_module
from env.vector_env import MAVUAVVectorEnv
from tools import preflight_eram_v311 as preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_fixture(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(trainer_module, "MAVUAVVectorEnv",
                        lambda *args, **kw: MAVUAVVectorEnv(*args, parallel=False, **kw))
    yield
    torch.set_num_threads(threads)


def baseline():
    return yaml.safe_load((ROOT / "configs/happo_v311_baseline.yaml").read_text())["training"]


def tiny_config(eram=True):
    filename = "happo_eram_v311.yaml" if eram else "happo_v311_baseline.yaml"
    c = yaml.safe_load((ROOT / "configs" / filename).read_text())["training"]
    c.update(device="cpu", num_envs=1, rollout_steps=2, ppo_epochs=1, minibatch_size=2)
    return c


def test_v311_baseline_exact_public_contract_and_constructible():
    b = baseline()
    e = yaml.safe_load((ROOT / "configs/happo_eram_v311.yaml").read_text())["training"]
    structural = {"actor_variant", "critic_variant"}
    eram_only = {"eram_entity_dim", "eram_actor_attention_heads", "eram_actor_fusion_dim",
                 "eram_actor_recurrent_hidden_dim", "eram_critic_token_dim",
                 "eram_critic_attention_heads", "eram_critic_recurrent_hidden_dim", "recurrent_sequence_length"}
    assert set(e) - set(b) == eram_only
    assert not set(b) - set(e)
    assert {k: v for k, v in b.items() if k not in structural} == {
        k: v for k, v in e.items() if k not in structural | eram_only}
    assert (b["actor_variant"], b["critic_variant"], b["method_variant"]) == ("vanilla", "mlp", "baseline")
    assert b["device"] == "cuda" and b["num_envs"] == 16 and b["rollout_steps"] == 128
    assert b["entropy_coef"] == 0.001 and b["actor_log_std_init"] == -0.25
    t = HAPPOTrainer(ROOT / "configs/env_v311.yaml", tiny_config(False))
    try:
        assert t.environment_config["environment_version"].endswith("v3_11")
        assert t.config["curriculum_steps"] == 400000 and not t.is_recurrent
    finally:
        t.close()


@pytest.mark.parametrize("field", ["algorithm", "base_algorithm", "actor_visibility_mask",
                                   "critic_entity_mask", "recurrent_sequence_length", "critic_architecture",
                                   "mav_direct_attack_capability", "environment_config"])
def test_eram_weights_load_rejects_metadata_before_parameter_mutation(field, tmp_path):
    t = HAPPOTrainer(ROOT / "configs/env_v311.yaml", tiny_config())
    try:
        path = tmp_path / "weights.pt"
        t.save(path)
        payload = torch.load(path, weights_only=False)
        before = deepcopy(t.actors.state_dict())
        payload.pop(field)
        torch.save(payload, path)
        with pytest.raises(RuntimeError, match="ERAM|architecture"):
            t.load(path)
        assert all(torch.equal(v, t.actors.state_dict()[k]) for k, v in before.items())
    finally:
        t.close()


@pytest.mark.parametrize("eram", [True, False])
def test_weights_roundtrip_and_legacy_without_eram_metadata(eram, tmp_path):
    t = HAPPOTrainer(ROOT / "configs/env_v311.yaml", tiny_config(eram))
    try:
        path = tmp_path / "weights.pt"
        t.save(path)
        if not eram:
            data = torch.load(path, weights_only=False)
            assert "entity_layout_version" not in data
            data.pop("algorithm", None)
            data.pop("base_algorithm", None)
            torch.save(data, path)
        before = deepcopy(t.actors.state_dict())
        with torch.no_grad():
            next(t.actors.parameters()).add_(1)
        t.load(path)
        assert all(torch.equal(v, t.actors.state_dict()[k]) for k, v in before.items())
    finally:
        t.close()


def test_preflight_core_cpu_collect_update_save_resume(tmp_path):
    t = HAPPOTrainer(ROOT / "configs/env_v311.yaml", tiny_config())
    other = None
    try:
        row = preflight.timed_update(t)
        assert row["transitions"] == 2 and row["sampled_steps"] == 2
        assert row["rollout_wall_seconds"] > 0 and row["update_wall_seconds"] > 0
        path = tmp_path / "checkpoint.pt"
        t.save_checkpoint(path)
        payload = torch.load(path, weights_only=False)
        preflight.validate_checkpoint(payload, t)
        other = HAPPOTrainer(ROOT / "configs/env_v311.yaml", tiny_config())
        other.load_checkpoint(path)
        preflight.validate_restored_state(other, payload)
        assert preflight.timed_update(other)["sampled_steps"] == 4
        other.buffer.advantages[0, 0] = np.nan
        with pytest.raises(AssertionError, match="advantages"):
            preflight.validate_state(other)
    finally:
        t.close()
        if other is not None:
            other.close()


@pytest.mark.parametrize("parallel,pids", [(False, [2]*16), (True, [2]*16), (True, [os.getpid()]+list(range(2, 17)))])
def test_preflight_rejects_pseudo_parallel(parallel, pids):
    vector = SimpleNamespace(parallel=parallel, num_envs=16, worker_pids=tuple(pids))
    with pytest.raises(AssertionError, match="subprocess"):
        preflight.validate_parallel_env(vector)


def test_preflight_accepts_distinct_workers_and_no_cuda_fails_without_trainer(tmp_path, monkeypatch):
    pids = list(range(200000, 200016))
    assert preflight.validate_parallel_env(SimpleNamespace(parallel=True, num_envs=16,
                                                          worker_pids=tuple(pids))) == pids
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not construct a CPU trainer")
    monkeypatch.setattr(preflight, "HAPPOTrainer", forbidden)
    assert preflight.main() == 1
    summary_path = next((tmp_path / "outputs/preflight_eram_v311").glob("*/summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "FAIL" and "refuses CPU fallback" in summary["error"]
