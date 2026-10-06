"""Bounded CUDA/16-subprocess production-path ERAM preflight, never CPU fallback."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.happo.eram import validate_eram_metadata


def require_finite(value, label):
    finite = torch.isfinite(value).all().item() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not finite:
        raise AssertionError(f"non-finite {label}")


def validate_parallel_env(vector_env, expected_count=16):
    pids = vector_env.worker_pids
    if (not vector_env.parallel or vector_env.num_envs != expected_count
            or len(pids) != expected_count or len(set(pids)) != expected_count
            or os.getpid() in pids):
        raise AssertionError("preflight requires distinct real subprocess environment workers")
    return list(pids)


def validate_state(trainer, metrics=None):
    n = int(trainer.config["num_envs"])
    if trainer.actor_hidden_states.shape != (n, 4, trainer.actor_recurrent_hidden_dim):
        raise AssertionError("actor memory shape mismatch")
    if trainer.critic_hidden_states.shape != (n, trainer.critic_recurrent_hidden_dim):
        raise AssertionError("critic memory shape mismatch")
    for label in ("actor_hidden_states", "critic_hidden_states", "observations", "global_states"):
        require_finite(getattr(trainer, label), label)
    for label in ("actions", "values", "returns", "advantages"):
        require_finite(getattr(trainer.buffer, label), f"buffer.{label}")
    for name, network in (("actors", trainer.actors), ("critic", trainer.critic)):
        for key, parameter in network.named_parameters():
            require_finite(parameter, f"{name}.{key}")
    if metrics is not None:
        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                require_finite(value, key)
        for key in ("actor_loss", "critic_loss", "entropy", *(f"actor_{i}_loss" for i in range(4))):
            require_finite(metrics[key], key)


def validate_checkpoint(payload, trainer):
    expected = {
        "algorithm": "eram_happo", "base_algorithm": "happo",
        "environment_version": "heterogeneous_mavuav_4v4_v3_11",
        "actor_variant": "entity_recurrent", "critic_variant": "entity_attention_recurrent",
        "method_variant": "baseline", "reward_mode": "heterogeneous_role_coupled_gate_v1",
        "sampled_steps": trainer.env_steps,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"checkpoint contract mismatch: {key}")
    validate_eram_metadata(payload, trainer.config)


def timed_update(trainer):
    before = trainer.env_steps
    start = time.perf_counter()
    trainer.collect_rollout()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    rollout_seconds = time.perf_counter() - start
    expected_steps = before + trainer.buffer.horizon * trainer.buffer.num_envs
    if trainer.env_steps != expected_steps:
        raise AssertionError("sampled transition counting mismatch")
    validate_state(trainer)
    start = time.perf_counter()
    metrics = trainer.update()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    update_seconds = time.perf_counter() - start
    validate_state(trainer, metrics)
    return {"sampled_steps": trainer.env_steps, "transitions": trainer.env_steps - before,
            "rollout_wall_seconds": rollout_seconds, "update_wall_seconds": update_seconds,
            "actor_loss": metrics["actor_loss"], "critic_loss": metrics["critic_loss"],
            "entropy": metrics["entropy"]}


def validate_restored_state(trainer, payload):
    """Check actual restored continuation tensors/RNG, beyond a successful load."""
    if trainer.env_steps != int(payload["sampled_steps"]):
        raise AssertionError("resume sampled_steps mismatch")
    for name in ("observations", "global_states", "active_masks", "actor_hidden_states",
                 "actor_recurrent_masks", "critic_hidden_states", "critic_recurrent_masks"):
        np.testing.assert_array_equal(getattr(trainer, name), payload["rollout_state"][name])
    for name in ("actors", "critic"):
        for key, value in getattr(trainer, name).state_dict().items():
            if not torch.equal(value.cpu(), payload[name][key].cpu()):
                raise AssertionError(f"resume parameter mismatch: {name}.{key}")
    if trainer.rng.bit_generator.state != payload["trainer_numpy_rng"]:
        raise AssertionError("resume numpy RNG mismatch")
    def equal_nested(actual, expected):
        if isinstance(expected, torch.Tensor):
            return torch.equal(actual.cpu(), expected.cpu())
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(equal_nested(actual[k], v) for k, v in expected.items())
        if isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(equal_nested(a, b) for a, b in zip(actual, expected))
        return actual == expected
    for actual, expected in zip(trainer.actor_optimizers, payload["actor_optimizer_states"]):
        if not equal_nested(actual.state_dict(), expected):
            raise AssertionError("resume actor optimizer mismatch")
    if not equal_nested(trainer.critic_optimizer.state_dict(), payload["critic_optimizer_state"]):
        raise AssertionError("resume critic optimizer mismatch")
    np.testing.assert_array_equal(trainer.vector_env.reset_counts, payload["rollout_state"]["vector_reset_counts"])
    if trainer.vector_env.base_seed != payload["rollout_state"]["vector_base_seed"]:
        raise AssertionError("resume vector base seed mismatch")
    if not torch.equal(torch.get_rng_state(), payload["torch_rng"].cpu()):
        raise AssertionError("resume Torch RNG mismatch")
    if trainer.device.type == "cuda":
        actual_rng = torch.cuda.get_rng_state_all()
        if len(actual_rng) != len(payload["cuda_rng"]) or any(not torch.equal(a.cpu(), b.cpu()) for a, b in
               zip(actual_rng, payload["cuda_rng"])):
            raise AssertionError("resume CUDA RNG mismatch")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/happo_eram_v311.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env_v311.yaml")
    parser.add_argument("--rollouts", type=int, choices=(2, 3), default=2,
                        help="Pre-resume full rollouts; one more full rollout follows exact resume")
    return parser.parse_args()


def main():
    args = parse_args()
    output = ROOT / "outputs/preflight_eram_v311" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "device": "cuda", "num_envs": 16,
               "sampled_steps": 0, "checkpoint_resume_status": "not_attempted", "output_dir": str(output),
               **{key: None for key in (
                   "rollout_wall_seconds", "update_wall_seconds", "approximate_transitions_per_second",
                   "cuda_allocated_bytes", "cuda_reserved_bytes", "actor_parameter_count",
                   "critic_parameter_count", "actor_loss", "critic_loss", "entropy",
               )}}
    trainer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: preflight refuses CPU fallback")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["training"]
        config = deepcopy(config)
        # Runtime contract, not an edit to the experiment YAML.
        if config["num_envs"] != 16 or not str(config["device"]).startswith("cuda"):
            raise ValueError("production preflight config must specify CUDA and num_envs=16")
        if int(config["rollout_steps"]) != 128:
            raise ValueError("production preflight requires full rollout_steps=128")
        if (config["actor_variant"], config["critic_variant"], config["method_variant"]) != (
                "entity_recurrent", "entity_attention_recurrent", "baseline"):
            raise ValueError("preflight requires baseline ERAM variants")
        trainer = HAPPOTrainer(args.env_config, config)
        summary["device"] = str(trainer.device)
        summary["worker_pids"] = validate_parallel_env(trainer.vector_env)
        summary["actor_parameter_count"] = sum(p.numel() for p in trainer.actors.parameters())
        summary["critic_parameter_count"] = trainer.critic_parameter_count
        updates = []
        summary["updates"] = updates
        for _ in range(args.rollouts):
            updates.append(timed_update(trainer))
            summary["sampled_steps"] = trainer.env_steps
        checkpoint = output / f"checkpoint_{trainer.env_steps}.pt"
        trainer.save_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, trainer)
        summary["checkpoint"] = str(checkpoint)
        summary["checkpoint_resume_status"] = "saved"
        trainer.close()
        trainer = None
        # Release the original CUDA modules before allocating a second trainer.
        torch.cuda.empty_cache()
        trainer = HAPPOTrainer(args.env_config, config)
        summary["resumed_worker_pids"] = validate_parallel_env(trainer.vector_env)
        trainer.load_checkpoint(checkpoint)
        validate_restored_state(trainer, payload)
        updates.append(timed_update(trainer))
        summary["sampled_steps"] = trainer.env_steps
        summary["checkpoint_resume_status"] = "PASS_exact_resume_and_update"
        rollout_time = sum(row["rollout_wall_seconds"] for row in updates)
        update_time = sum(row["update_wall_seconds"] for row in updates)
        transitions = sum(row["transitions"] for row in updates)
        summary.update({"status": "PASS", "rollout_wall_seconds": rollout_time,
                        "update_wall_seconds": update_time,
                        "approximate_transitions_per_second": transitions / (rollout_time + update_time),
                        "rollout_transitions_per_second": transitions / rollout_time,
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(trainer.device),
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(trainer.device),
                        "actor_loss": updates[-1]["actor_loss"], "critic_loss": updates[-1]["critic_loss"],
                        "entropy": updates[-1]["entropy"]})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if trainer is not None:
            try:
                trainer.close()
            except Exception as exc:
                summary["status"] = "FAIL"
                summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
