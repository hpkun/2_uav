"""Fixed-seed evaluation helpers for functional heterogeneous red 4v3 HAPPO."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from ..environment_4v3 import FunctionalHeterogeneous4v3AirCombatEnv
from ..environment_4v3_v11 import FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv
from ..mappo.trainer_3v3 import resolve_device
from ..mappo.vector_env_4v3 import RED_TEAM_SIZE_4V3, make_combat_vector_env_4v3
from .networks import IndependentHAPPOActors
from .trainer_4v3 import summarize_4v3_episodes


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json_4v3(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seed_hash(seeds: Sequence[int]) -> str:
    return sha256_json_4v3([int(seed) for seed in seeds])


def build_evaluation_seed_manifest(
    experiment_seed: int,
    *,
    selection_episodes: int,
    test_episodes: int,
    selection_seed_offset: int,
    test_seed_offset: int,
) -> dict[str, Any]:
    """Build a stable, non-overlapping selection/test seed manifest."""
    if selection_episodes <= 0 or test_episodes <= 0:
        raise ValueError("selection_episodes and test_episodes must be positive")
    selection = [int(experiment_seed) + int(selection_seed_offset) + i for i in range(int(selection_episodes))]
    test = [int(experiment_seed) + int(test_seed_offset) + i for i in range(int(test_episodes))]
    if set(selection) & set(test):
        raise ValueError("selection and test evaluation seeds overlap")
    manifest = {
        "schema_version": 1,
        "experiment_seed": int(experiment_seed),
        "selection": {
            "episodes": len(selection),
            "seed_offset": int(selection_seed_offset),
            "seeds": selection,
            "seed_hash": _seed_hash(selection),
        },
        "test": {
            "episodes": len(test),
            "seed_offset": int(test_seed_offset),
            "seeds": test,
            "seed_hash": _seed_hash(test),
        },
    }
    manifest["selection_seeds"] = selection
    manifest["test_seeds"] = test
    manifest["manifest_hash"] = sha256_json_4v3(manifest)
    return manifest


def validate_evaluation_seed_manifest(manifest: dict[str, Any]) -> None:
    selection = [int(v) for v in manifest.get("selection", {}).get("seeds", manifest.get("selection_seeds", []))]
    test = [int(v) for v in manifest.get("test", {}).get("seeds", manifest.get("test_seeds", []))]
    if len(selection) != len(set(selection)) or len(test) != len(set(test)):
        raise ValueError("evaluation seed lists contain duplicates")
    if set(selection) & set(test):
        raise ValueError("selection and test evaluation seeds overlap")
    if manifest.get("selection", {}).get("seed_hash") and manifest["selection"]["seed_hash"] != _seed_hash(selection):
        raise ValueError("selection seed hash mismatch")
    if manifest.get("test", {}).get("seed_hash") and manifest["test"]["seed_hash"] != _seed_hash(test):
        raise ValueError("test seed hash mismatch")
    if manifest.get("manifest_hash"):
        unsigned = dict(manifest)
        unsigned.pop("manifest_hash", None)
        if manifest["manifest_hash"] != sha256_json_4v3(unsigned):
            raise ValueError("evaluation manifest hash mismatch")


def evaluation_seeds_from_manifest(manifest: dict[str, Any], split: str, episodes: int | None = None) -> list[int]:
    validate_evaluation_seed_manifest(manifest)
    if split not in ("selection", "test"):
        raise ValueError("split must be 'selection' or 'test'")
    values = [int(v) for v in manifest[split]["seeds"]]
    if episodes is not None:
        if int(episodes) <= 0 or int(episodes) > len(values):
            raise ValueError("episodes must be in [1, manifest split length]")
        values = values[: int(episodes)]
    return values


def _resolve_seeds(episodes: int | None, seeds: Sequence[int] | None, fallback_seed: int) -> list[int]:
    if seeds is None:
        if episodes is None or int(episodes) <= 0:
            raise ValueError("episodes must be positive when seeds are not supplied")
        return [int(fallback_seed) + i for i in range(int(episodes))]
    values = [int(seed) for seed in seeds]
    if not values:
        raise ValueError("seeds must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("evaluation seeds must be unique")
    if episodes is not None and int(episodes) != len(values):
        raise ValueError("episodes must equal the explicit seed-list length")
    return values


def _summary_with_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    seeds: list[int],
    elapsed: float,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = summarize_4v3_episodes(records)
    summary.update({
        "split": split,
        "seed_list": list(seeds),
        "seed_hash": _seed_hash(seeds),
        "manifest_hash": manifest.get("manifest_hash") if manifest else None,
        "evaluation_seconds": float(elapsed),
        "episode_records": deepcopy(records),
    })
    return summary


@torch.no_grad()
def evaluate_happo_fixed_blue_4v3(
    actors: IndependentHAPPOActors,
    env_config: str | Path,
    *,
    episodes: int | None = None,
    num_envs: int = 4,
    num_env_workers: int = 0,
    device: str | torch.device = "cpu",
    seed: int = 10000,
    seeds: Sequence[int] | None = None,
    split: str = "selection",
    seed_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly once for every explicit seed, without touching training RNG."""
    dev = resolve_device(str(device))
    seed_list = _resolve_seeds(episodes, seeds, seed)
    if seed_manifest is not None:
        validate_evaluation_seed_manifest(seed_manifest)
    was_training = actors.training
    actors.eval()
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        batch_capacity = max(1, int(num_envs))
        for start in range(0, len(seed_list), batch_capacity):
            batch_seeds = seed_list[start:start + batch_capacity]
            requested_workers = int(num_env_workers)
            workers = requested_workers if requested_workers > 0 and len(batch_seeds) % requested_workers == 0 else 0
            vec = make_combat_vector_env_4v3(env_config, len(batch_seeds), workers, seed=0)
            try:
                indices = np.arange(len(batch_seeds), dtype=np.int32)
                obs, _, _ = vec.reset_at(indices, np.asarray(batch_seeds, dtype=np.int64))
                finished = np.zeros(len(batch_seeds), dtype=bool)
                while not bool(finished.all()):
                    red_obs = torch.as_tensor(obs[:, :RED_TEAM_SIZE_4V3, :], dtype=torch.float32, device=dev)
                    actions = actors.deterministic_actions(red_obs).cpu().numpy().astype(np.float32)
                    result = vec.step(actions)
                    obs = result.observations
                    for i, episode_summary in enumerate(result.episode_summaries):
                        if episode_summary is None or finished[i]:
                            continue
                        record = deepcopy(episode_summary)
                        record["episode_seed"] = int(batch_seeds[i])
                        records.append(record)
                        finished[i] = True
            finally:
                vec.close()
        records.sort(key=lambda record: int(record["episode_seed"]))
        return _summary_with_records(records, split=split, seeds=seed_list, elapsed=time.perf_counter() - t0, manifest=seed_manifest)
    finally:
        if was_training:
            actors.train()


def evaluate_rule_vs_rule_4v3(
    env_config: str | Path,
    *,
    episodes: int = 100,
    seed: int = 20000,
    seeds: Sequence[int] | None = None,
    split: str = "rule",
    seed_manifest: dict[str, Any] | None = None,
    workers: int = 4,
    red_policy: str = "rule",
) -> dict[str, Any]:
    seed_list = _resolve_seeds(episodes, seeds, seed)
    config = yaml.safe_load(Path(env_config).read_text(encoding="utf-8"))
    if config.get("combat", {}).get("reward_contract_version") == "v11_target_lock_support_cue":
        if red_policy not in ("rule", "random"):
            raise ValueError("v11 red_policy must be 'rule' or 'random'")
        runner = _run_v11_baseline_episode
        args = [(str(env_config), int(item_seed), red_policy) for item_seed in seed_list]
        t0 = time.perf_counter()
        if int(workers) <= 1 or len(args) <= 1:
            records = [runner(*item) for item in args]
        else:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(args))) as executor:
                records = list(executor.map(_run_v11_baseline_episode, [item[0] for item in args], [item[1] for item in args], [item[2] for item in args]))
        records.sort(key=lambda record: int(record["episode_seed"]))
        result_split = red_policy if split == "rule" else split
        return _summary_with_records(records, split=result_split, seeds=seed_list, elapsed=time.perf_counter() - t0, manifest=seed_manifest)
    t0 = time.perf_counter()
    if int(workers) <= 1 or len(seed_list) <= 1:
        records = [_run_rule_episode(str(env_config), episode_seed) for episode_seed in seed_list]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(seed_list))) as executor:
            records = list(executor.map(_run_rule_episode, [str(env_config)] * len(seed_list), seed_list))
    records.sort(key=lambda record: int(record["episode_seed"]))
    return _summary_with_records(records, split=split, seeds=seed_list, elapsed=time.perf_counter() - t0, manifest=seed_manifest)


def _run_v11_baseline_episode(env_config: str, episode_seed: int, red_policy: str = "rule") -> dict[str, Any]:
    env = FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv(env_config)
    env.reset(int(episode_seed))
    rng = np.random.default_rng(int(episode_seed) + 99173)
    done = False
    info: dict[str, Any] = {}
    while not done:
        if red_policy == "rule":
            actions, _ = env.red_rule_actions()
        elif red_policy == "random":
            actions = {aid: rng.uniform(-1.0, 1.0, size=3).astype(np.float32) for aid in ("red_0", "red_1", "red_2", "red_3")}
        else:
            raise ValueError(f"unsupported v11 red policy: {red_policy!r}")
        _, _, _, _, done, _, info = env.step(actions)
    record = deepcopy(info["episode_summary"])
    record["episode_seed"] = int(episode_seed)
    record["red_policy"] = red_policy
    return record


def _run_rule_episode(env_config: str, episode_seed: int) -> dict[str, Any]:
    env = FunctionalHeterogeneous4v3AirCombatEnv(env_config)
    env.reset(int(episode_seed))
    done = False
    info: dict[str, Any] = {}
    while not done:
        red_actions, _ = env.red_rule_actions()
        _, _, _, _, done, _, info = env.step(red_actions)
    record = deepcopy(info["episode_summary"])
    record["episode_seed"] = int(episode_seed)
    return record


__all__ = [
    "build_evaluation_seed_manifest",
    "evaluate_happo_fixed_blue_4v3",
    "evaluate_rule_vs_rule_4v3",
    "evaluation_seeds_from_manifest",
    "sha256_json_4v3",
    "validate_evaluation_seed_manifest",
]
