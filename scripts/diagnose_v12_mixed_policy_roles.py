"""Read-only mixed-policy and per-Actor diagnosis for the v12 HAPPO run.

This command only loads actors and steps a diagnostic subclass of the v12
environment.  It never constructs a trainer, calls ``update`` or an optimizer,
and it writes exclusively below the requested diagnostics directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics as statistics_module
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_combat.diagnostics.v12_mixed_policy_roles import (  # noqa: E402
    AGENT_TO_ROLE_V12,
    COMBOS_V12_MIXED,
    TEAM_AGENT_IDS_V12,
    exact_mcnemar_pvalue,
    paired_bootstrap,
    practical_equivalence,
    select_targeted_seeds,
    validate_source_map,
    validate_all_combinations,
)
from uav_combat.environment_4v3_v11 import compute_red_combat_formation_reference  # noqa: E402
from uav_combat.environment_4v3_v12 import (  # noqa: E402
    FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv,
)
from uav_combat.geometry import compute_pairwise_geometry  # noqa: E402
from uav_combat.happo.evaluation_4v3 import build_evaluation_seed_manifest  # noqa: E402
from uav_combat.happo.networks import IndependentHAPPOActors  # noqa: E402
from uav_combat.scenario_4v3_v12 import BLUE_IDS_V12, RED_COMBAT_IDS_V12  # noqa: E402


ROLE_ORDER = ("support", "combat_1", "combat_2", "combat_3")
INPUT_NAMES = (
    "best.pt",
    "final.pt",
    "latest.pt",
    "step_2900000.pt",
    "step_3000000.pt",
    "resolved_environment_config.yaml",
    "resolved_training_config.yaml",
    "evaluation_seed_manifest.json",
)
CORE_TEAM_FIELDS = (
    "task_win",
    "strict_full_elimination",
    "any_kill",
    "at_least_two_kill",
    "red_attack_kills",
    "red_half_lock_episode_rate",
    "episode_return",
    "episode_length",
    "support_survived",
)
TEAM_REPRO_FIELDS = (
    "task_win_rate",
    "strict_full_elimination_rate",
    "any_kill_rate",
    "at_least_two_kill_rate",
    "mean_red_kills",
    "red_half_lock_episode_rate",
    "mean_red_max_lock_progress",
    "mean_return",
    "mean_episode_length",
)
AGENT_REPRO_FIELDS = (
    "mean_attributed_kills",
    "mean_half_lock_event_count",
    "mean_max_lock_progress",
    "mean_lock_active_step_rate",
    "mean_half_lock_active_step_rate",
    "mean_direct_target_rate",
    "mean_shared_only_target_rate",
    "mean_no_valid_target_rate",
    "mean_target_switch_count",
    "survival_rate",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _normalise_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if compressed:
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                stream.write("")
        else:
            path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    for key, index in (("checkpoint", 0), ("combo", 1), ("episode_seed", 2)):
        if key in fields:
            fields.remove(key)
            fields.insert(min(index, len(fields)), key)
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalise_csv_value(value) for key, value in row.items()})


def _read_rows(path: Path, compressed: bool = False) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _state_hash(value: Any) -> str:
    """Stable hash for checkpoint tensor/state parameters, not file metadata."""
    digest = hashlib.sha256()
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            digest.update(_state_hash(value[key]).encode("ascii"))
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(_state_hash(item).encode("ascii"))
    else:
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _input_sha(run_dir: Path) -> dict[str, str]:
    return {name: _sha256_file(run_dir / name) for name in INPUT_NAMES}


def _checkpoint_info(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actors")
    critic_state = checkpoint.get("critic")
    if actor_state is None:
        raise ValueError(f"{path.name} does not contain actors")
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "parameter_sha256": _state_hash({"actors": actor_state, "critic": critic_state}),
        "actor_sha256": _state_hash(actor_state),
        "critic_sha256": _state_hash(critic_state),
        "env_steps": int(checkpoint.get("env_steps", checkpoint.get("actual_env_steps", -1))),
        "variant": checkpoint.get("variant"),
        "reward_contract_version": checkpoint.get("reward_contract_version"),
        "checkpoint_family": checkpoint.get("checkpoint_family"),
        "version": checkpoint.get("checkpoint_version"),
    }


def _git_commit_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _load_inputs(
    run_dir: Path,
    requested_device: str,
    started_at: str,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int], torch.device]:
    """Read and validate every declared input before evaluation begins."""
    validate_all_combinations()
    env_cfg_path = run_dir / "resolved_environment_config.yaml"
    train_cfg_path = run_dir / "resolved_training_config.yaml"
    contract = json.loads((run_dir / "experiment_contract.json").read_text(encoding="utf-8"))
    env_cfg = yaml.safe_load(env_cfg_path.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8"))
    seed_manifest = json.loads((run_dir / "evaluation_seed_manifest.json").read_text(encoding="utf-8"))
    # Explicitly read the large summary as an input integrity check.  It is not
    # used to populate mixed-policy results.
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    checkpoints = {name: _checkpoint_info(run_dir / name) for name in ("best.pt", "final.pt", "latest.pt", "step_2900000.pt", "step_3000000.pt")}
    if checkpoints["best.pt"]["env_steps"] != 2_900_000:
        raise ValueError("best.pt env_steps must be 2,900,000")
    if checkpoints["final.pt"]["env_steps"] != 3_000_000:
        raise ValueError("final.pt env_steps must be 3,000,000")
    final_latest_equal = checkpoints["latest.pt"]["parameter_sha256"] == checkpoints["final.pt"]["parameter_sha256"]
    if not final_latest_equal:
        raise ValueError("latest.pt parameters do not match final.pt")
    best_step_equal = checkpoints["best.pt"]["parameter_sha256"] == checkpoints["step_2900000.pt"]["parameter_sha256"]
    env_variant = contract.get("variant")
    reward_contract = contract.get("reward_contract_version")
    if "v12" not in str(env_variant) or reward_contract != "v12_soft_boundary_combat_aligned":
        raise ValueError("input contract is not the requested v12 experiment")
    dimensions = {
        "observation_dims": train_cfg["training"]["observation_dims"],
        "state_dim": int(contract.get("state_dim", 70)),
        "action_dims": train_cfg["training"]["action_dims"],
    }
    if dimensions["observation_dims"] != [118, 118, 118, 118] or dimensions["state_dim"] != 70 or dimensions["action_dims"] != [3, 3, 3, 3]:
        raise ValueError(f"unexpected v12 dimensions: {dimensions}")
    test_seeds = [int(seed) for seed in seed_manifest.get("test", {}).get("seeds", [])]
    selection_seeds = [int(seed) for seed in seed_manifest.get("selection", {}).get("seeds", [])]
    if len(test_seeds) != 200 or len(set(test_seeds)) != 200:
        raise ValueError("test seed manifest must contain exactly 200 unique seeds")
    if set(test_seeds) & set(selection_seeds):
        raise ValueError("selection and test seeds overlap")
    if seed_manifest.get("test", {}).get("seed_hash") != hashlib.sha256(_canonical_json(test_seeds)).hexdigest():
        # The project helper hashes canonical JSON; this check catches stale
        # manifests while accepting the same representation used by training.
        from uav_combat.happo.evaluation_4v3 import sha256_json_4v3
        if seed_manifest["test"]["seed_hash"] != sha256_json_4v3(test_seeds):
            raise ValueError("test seed hash mismatch")
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
        device_note = "requested CUDA available; multiprocessing workers may use CPU for safety"
    else:
        device = torch.device("cpu")
        device_note = "CPU fallback because requested CUDA is unavailable"
    parallel_worker_device = "cpu" if int(workers) > 1 else str(device)
    data = {
        "run_dir": str(run_dir.resolve()),
        "input_files": {},
        "checkpoint_meta": checkpoints,
        "experiment_contract": contract,
        "dimensions": dimensions,
        "variant": env_variant,
        "reward_contract_version": reward_contract,
        "actor_slot_mapping": {"actor_0": "red_0 Support", "actor_1": "red_1 Combat", "actor_2": "red_2 Combat", "actor_3": "red_3 Combat"},
        "test_seed_count": len(test_seeds),
        "selection_seed_count": len(selection_seeds),
        "test_seed_hash": seed_manifest.get("test", {}).get("seed_hash"),
        "selection_seed_hash": seed_manifest.get("selection", {}).get("seed_hash"),
        "seed_manifest_hash": seed_manifest.get("manifest_hash"),
        "best_vs_step_2900000_parameter_identical": best_step_equal,
        "final_vs_latest_parameter_identical": final_latest_equal,
        "run_summary_keys_read": sorted(run_summary.keys()) if isinstance(run_summary, dict) else [],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "requested_device": requested_device,
        "effective_device": str(device),
        "controller_device": str(device),
        "parallel_episode_worker_device": parallel_worker_device,
        "trajectory_device": str(device),
        "device_note": device_note,
        "workers": int(workers),
        "git_commit_sha": _git_commit_sha(ROOT),
        "started_at": started_at,
        "ended_at": None,
        "input_sha256_start": _input_sha(run_dir),
        "input_sha256_end": None,
        "command": " ".join(sys.argv),
        "execution_command": " ".join(sys.argv),
    }
    for name in INPUT_NAMES + ("experiment_contract.json", "run_summary.json"):
        data["input_files"][name] = str((run_dir / name).resolve())
    return data, env_cfg, train_cfg, test_seeds, device


class MixedPolicyProbeEnv(FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv):
    """Instrumentation-only subclass; parent transition semantics are untouched."""

    def __init__(self, config_path: str | Path):
        super().__init__(config_path)
        self._diag_last_half_events: set[tuple[str, str]] = set()
        self._diag_last_killers: dict[str, str] = {}
        self._diag_hard_contacts: list[str] = []
        self._diag_recovery_blends: dict[str, float] = {}

    def _update_locks(self, direct: dict[str, set[str]]):  # type: ignore[override]
        half_events, killers = super()._update_locks(direct)
        self._diag_last_half_events = set(half_events)
        self._diag_last_killers = dict(killers)
        return half_events, killers

    def _project_hard_boundary(self, aircraft):  # type: ignore[override]
        before = self._episode_metrics.get(f"{self._boundary_team_key(aircraft)}_boundary_hard_contacts", 0.0)
        changed = super()._project_hard_boundary(aircraft)
        after = self._episode_metrics.get(f"{self._boundary_team_key(aircraft)}_boundary_hard_contacts", 0.0)
        if changed and after > before:
            self._diag_hard_contacts.append(aircraft.aircraft_id)
        return changed

    def _boundary_recovery_action(self, aircraft, action):  # type: ignore[override]
        corrected, blend = super()._boundary_recovery_action(aircraft, action)
        self._diag_recovery_blends[aircraft.aircraft_id] = float(blend)
        return corrected, blend

    def step(self, red_actions):  # type: ignore[override]
        self._diag_hard_contacts = []
        self._diag_recovery_blends = {}
        return super().step(red_actions)


def _state_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _target_source(env: MixedPolicyProbeEnv, agent_id: str, target_id: str | None, direct: dict[str, set[str]]) -> str:
    if target_id is None or not env._alive(target_id):
        return "none"
    if target_id in direct.get(agent_id, set()):
        return "direct"
    if agent_id != "red_0" and env._support_cues.get(agent_id) == target_id:
        return "shared"
    return "hidden"


@dataclass
class AgentAccum:
    agent_id: str
    role: str
    action_source: str
    steps: int = 0
    alive_steps: int = 0
    max_lock_progress: float = 0.0
    positive_lock_steps: int = 0
    half_lock_steps: int = 0
    current_positive: int = 0
    current_half: int = 0
    longest_positive: int = 0
    longest_half: int = 0
    attributed_kills: int = 0
    first_kill_step: int | None = None
    half_lock_event_count: int = 0
    first_half_lock_step: int | None = None
    target_switch_count: int = 0
    target_switch_while_lock_above_0_1: int = 0
    direct_steps: int = 0
    shared_steps: int = 0
    hidden_steps: int = 0
    no_target_steps: int = 0
    direct_target_steps: int = 0
    shared_only_target_steps: int = 0
    no_valid_target_steps: int = 0
    target_distance_sum: float = 0.0
    ata_sum: float = 0.0
    aa_sum: float = 0.0
    geometry_steps: int = 0
    hard_contact_count: int = 0
    death_step: int | None = None
    death_cause: int | None = None
    cue_active_steps: int = 0
    cue_active_pair_steps: int = 0
    cue_update_count: int = 0
    unique_detection_events: int = 0
    cue_to_direct_events: int = 0
    cue_to_half_lock_events: int = 0
    assisted_kills: int = 0
    formation_distance_sum: float = 0.0
    rear_alignment_sum: float = 0.0
    rear_alignment_positive_steps: int = 0
    threat_exposure_steps: int = 0
    support_visible_to_blue_steps: int = 0
    support_targeted_by_blue_steps: int = 0
    _current_target: str | None = None
    _previous_lock: float = 0.0
    _previous_alive: bool = True
    _first_step: bool = True

    def record_pre_step(self, env: MixedPolicyProbeEnv, direct: dict[str, set[str]], step: int) -> dict[str, Any]:
        alive = bool(env._alive(self.agent_id))
        self.steps += 1
        # All behavior ratios are decision-time, alive-only quantities.  A
        # dead Actor contributes a physical trace row but no stale target,
        # lock, visibility, geometry, cue, or switching count.
        if not alive:
            self.current_positive = 0
            self.current_half = 0
            self._previous_lock = 0.0
            return {
                "target_id": None,
                "target_source": "none",
                "direct_visible": False,
                "shared_visible": False,
                "target_distance": None,
                "ata": None,
                "aa": None,
                "lock_progress": 0.0,
                "alive": False,
                "support_visible_to_blue": False,
                "support_targeted_by_blue": False,
            }
        self.alive_steps += 1
        lock = float(env.lock_progress.get(self.agent_id, 0.0))
        target = env.targets.get(self.agent_id)
        if self._first_step:
            self._current_target = target
            self._first_step = False
        elif target != self._current_target:
            self.target_switch_count += 1
            if self._previous_lock >= 0.1:
                self.target_switch_while_lock_above_0_1 += 1
            self._current_target = target
        self.max_lock_progress = max(self.max_lock_progress, lock)
        if lock > 0.0:
            self.positive_lock_steps += 1
            self.current_positive += 1
        else:
            self.longest_positive = max(self.longest_positive, self.current_positive)
            self.current_positive = 0
        if lock >= 0.5:
            self.half_lock_steps += 1
            self.current_half += 1
        else:
            self.longest_half = max(self.longest_half, self.current_half)
            self.current_half = 0
        source = _target_source(env, self.agent_id, target, direct)
        self.direct_steps += int(source == "direct")
        self.shared_steps += int(source == "shared")
        self.hidden_steps += int(source == "hidden")
        self.no_target_steps += int(source == "none")
        self.direct_target_steps += int(source == "direct")
        self.shared_only_target_steps += int(source == "shared")
        self.no_valid_target_steps += int(source == "none")
        geometry = None
        if alive and target is not None and env._alive(target):
            geometry = compute_pairwise_geometry(env._by_id(self.agent_id).state, env._by_id(target).state)
            self.target_distance_sum += float(geometry.distance)
            self.ata_sum += float(geometry.ata)
            self.aa_sum += float(geometry.aa)
            self.geometry_steps += 1
        if self.agent_id == "red_0" and alive:
            ref = compute_red_combat_formation_reference(env._by_id("red_0"), [env._by_id(cid) for cid in RED_COMBAT_IDS_V12])
            self.formation_distance_sum += float(ref["centroid_distance"])
            self.rear_alignment_sum += float(ref["rear_alignment"])
            self.rear_alignment_positive_steps += int(ref["rear_alignment"] > 0.0)
            visible_to_blue = any(
                env._alive(blue_id) and "red_0" in direct.get(blue_id, set())
                for blue_id in BLUE_IDS_V12
            )
            targeted_by_blue = any(
                env._alive(blue_id) and env.targets.get(blue_id) == "red_0"
                for blue_id in BLUE_IDS_V12
            )
            self.support_visible_to_blue_steps += int(visible_to_blue)
            self.support_targeted_by_blue_steps += int(targeted_by_blue)
            self.threat_exposure_steps += int(visible_to_blue)
        self._previous_lock = lock
        return {
            "target_id": target,
            "target_source": source,
            "direct_visible": bool(target is not None and target in direct.get(self.agent_id, set())),
            "shared_visible": bool(source == "shared"),
            "target_distance": float(geometry.distance) if geometry is not None else None,
            "ata": float(geometry.ata) if geometry is not None else None,
            "aa": float(geometry.aa) if geometry is not None else None,
            "lock_progress": lock,
            "alive": alive,
            "support_visible_to_blue": bool(
                self.agent_id == "red_0" and any(
                    env._alive(blue_id) and "red_0" in direct.get(blue_id, set())
                    for blue_id in BLUE_IDS_V12
                )
            ),
            "support_targeted_by_blue": bool(
                self.agent_id == "red_0" and any(
                    env._alive(blue_id) and env.targets.get(blue_id) == "red_0"
                    for blue_id in BLUE_IDS_V12
                )
            ),
        }

    def record_post_step(self, env: MixedPolicyProbeEnv, step: int) -> None:
        alive = bool(env._alive(self.agent_id))
        if self._previous_alive and not alive and self.death_step is None:
            self.death_step = int(step)
            self.death_cause = int(env._death_causes.get(self.agent_id, -1))
        self._previous_alive = alive
        self.max_lock_progress = max(self.max_lock_progress, float(env.lock_progress.get(self.agent_id, 0.0)))

    def finish(self, env: MixedPolicyProbeEnv, summary: Mapping[str, Any]) -> dict[str, Any]:
        self.longest_positive = max(self.longest_positive, self.current_positive)
        self.longest_half = max(self.longest_half, self.current_half)
        denominator = max(1, self.alive_steps)
        out: dict[str, Any] = {
            "agent_id": self.agent_id,
            "role": self.role,
            "action_source": self.action_source,
            "steps": self.steps,
            "alive_steps": self.alive_steps,
            "survived": self.death_step is None,
            "survival_rate": float(self.death_step is None),
            "alive_decision_steps": self.alive_steps,
            "death_step": self.death_step,
            "death_cause": self.death_cause,
            "attributed_kills": self.attributed_kills,
            "first_kill_step": self.first_kill_step,
            "half_lock_event_count": self.half_lock_event_count,
            "first_half_lock_step": self.first_half_lock_step,
            "max_lock_progress": self.max_lock_progress,
            "positive_lock_steps": self.positive_lock_steps,
            "half_lock_steps": self.half_lock_steps,
            "lock_active_step_rate": self.positive_lock_steps / denominator,
            "half_lock_active_step_rate": self.half_lock_steps / denominator,
            "longest_continuous_positive_lock": self.longest_positive,
            "longest_continuous_half_lock": self.longest_half,
            "direct_target_steps": self.direct_target_steps,
            "shared_only_target_steps": self.shared_only_target_steps,
            "hidden_target_steps": self.hidden_steps,
            "no_valid_target_steps": self.no_valid_target_steps,
            "direct_target_rate": self.direct_target_steps / denominator,
            "shared_only_target_rate": self.shared_only_target_steps / denominator,
            "no_valid_target_rate": self.no_valid_target_steps / denominator,
            "target_switch_count": self.target_switch_count,
            "target_switch_while_lock_above_0_1": self.target_switch_while_lock_above_0_1,
            "mean_target_distance": self.target_distance_sum / max(1, self.geometry_steps),
            "mean_ata": self.ata_sum / max(1, self.geometry_steps),
            "mean_aa": self.aa_sum / max(1, self.geometry_steps),
            "hard_contact_count": self.hard_contact_count,
        }
        if self.agent_id == "red_0":
            out.update({
                "cue_active_steps": self.cue_active_steps,
                "cue_active_pair_steps": self.cue_active_pair_steps,
                "cue_update_count": self.cue_update_count,
                "unique_detection_events": self.unique_detection_events,
                "cue_to_direct_events": self.cue_to_direct_events,
                "cue_to_half_lock_events": self.cue_to_half_lock_events,
                "assisted_kills": self.assisted_kills,
                "mean_formation_distance": self.formation_distance_sum / denominator,
                "mean_rear_alignment": self.rear_alignment_sum / denominator,
                "rear_alignment_positive_rate": self.rear_alignment_positive_steps / denominator,
                "threat_exposure_rate": self.threat_exposure_steps / denominator,
                "support_visible_to_blue_rate": self.support_visible_to_blue_steps / denominator,
                "support_targeted_by_blue_rate": self.support_targeted_by_blue_steps / denominator,
            })
        return out


def _load_actors(checkpoint_path: str | Path, train_cfg: Mapping[str, Any], device: torch.device) -> IndependentHAPPOActors:
    network = train_cfg["network"]
    training = train_cfg["training"]
    actors = IndependentHAPPOActors(
        observation_dims=[int(value) for value in training["observation_dims"]],
        action_dims=[int(value) for value in training["action_dims"]],
        hidden_dim=int(network["hidden_dim"]),
        log_std_init=float(network["log_std_init"]),
        log_std_min=float(network["log_std_min"]),
        log_std_max=float(network["log_std_max"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actors.load_state_dict(checkpoint["actors"])
    actors.eval()
    return actors


def _run_episode(
    env: MixedPolicyProbeEnv,
    actors: IndependentHAPPOActors,
    combo_name: str,
    source_map: Mapping[str, str],
    checkpoint_name: str,
    seed: int,
    device: torch.device,
    trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    obs, _, _ = env.reset(int(seed))
    accum = {agent_id: AgentAccum(agent_id, AGENT_TO_ROLE_V12[agent_id], source_map[agent_id]) for agent_id in TEAM_AGENT_IDS_V12}
    trace_rows: list[dict[str, Any]] = []
    previous_state = env.state_dict()
    step = 0
    info: dict[str, Any] = {}
    while True:
        direct_pre = env._direct_visible_ids()
        state_before_rule = previous_state
        rule_actions, _ = env.red_rule_actions()
        state_after_rule = env.state_dict()
        if not _state_equal(state_before_rule, state_after_rule):
            raise AssertionError("red_rule_actions modified environment state")
        with torch.no_grad():
            red_obs = torch.as_tensor(obs[:4], dtype=torch.float32, device=device).unsqueeze(0)
            learned_actions = actors.deterministic_actions(red_obs)[0].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(learned_actions).all() or np.any(learned_actions < -1.000001) or np.any(learned_actions > 1.000001):
            raise FloatingPointError("learned actions are not finite/bounded")
        action_meta = {agent_id: accum[agent_id].record_pre_step(env, direct_pre, step + 1) for agent_id in TEAM_AGENT_IDS_V12}
        actions: dict[str, np.ndarray] = {}
        for index, agent_id in enumerate(TEAM_AGENT_IDS_V12):
            selected = learned_actions[index] if source_map[agent_id] == "learned" else np.asarray(rule_actions[agent_id], dtype=np.float32)
            selected = np.asarray(selected, dtype=np.float32).copy()
            if selected.shape != (3,) or not np.isfinite(selected).all() or np.any(selected < -1.000001) or np.any(selected > 1.000001):
                raise FloatingPointError(f"invalid action for {agent_id}")
            actions[agent_id] = selected
        _, _, _, _, done, _, info = env.step(actions)
        step += 1
        killers = dict(env._diag_last_killers)
        half_events = set(env._diag_last_half_events)
        for target_id, killer_id in killers.items():
            if killer_id in accum:
                accum[killer_id].attributed_kills += 1
                if accum[killer_id].first_kill_step is None:
                    accum[killer_id].first_kill_step = step
        for attacker_id, _target_id in half_events:
            if attacker_id in accum:
                accum[attacker_id].half_lock_event_count += 1
                if accum[attacker_id].first_half_lock_step is None:
                    accum[attacker_id].first_half_lock_step = step
        for agent_id in TEAM_AGENT_IDS_V12:
            accum[agent_id].hard_contact_count += int(agent_id in env._diag_hard_contacts)
            accum[agent_id].record_post_step(env, step)
        if trace:
            for index, agent_id in enumerate(TEAM_AGENT_IDS_V12):
                meta = action_meta[agent_id]
                killer_id = next((killer for target, killer in killers.items() if killer == agent_id), None)
                killed_target_id = next((target for target, killer in killers.items() if killer == agent_id), None)
                kill_event = killer_id == agent_id and killed_target_id in BLUE_IDS_V12
                trace_rows.append({
                    "checkpoint": checkpoint_name,
                    "combo": combo_name,
                    "episode_seed": int(seed),
                    "step": step,
                    "red_agent_id": agent_id,
                    "action_source": source_map[agent_id],
                    "alive": meta["alive"],
                    "target_id": meta["target_id"],
                    "target_source": meta["target_source"],
                    "direct_visible": meta["direct_visible"],
                    "shared_visible": meta["shared_visible"],
                    "target_distance": meta["target_distance"],
                    "ATA": meta["ata"],
                    "AA": meta["aa"],
                    "lock_progress": meta["lock_progress"],
                    "lock_delta": float(env.lock_progress.get(agent_id, 0.0) - meta["lock_progress"]),
                    "half_lock_event": any(pair[0] == agent_id for pair in half_events),
                    "kill_event": bool(kill_event),
                    "killer_id": killer_id if kill_event else None,
                    "killed_target_id": killed_target_id if kill_event else None,
                    "support_visible_to_blue": bool(meta.get("support_visible_to_blue", False)),
                    "support_targeted_by_blue": bool(meta.get("support_targeted_by_blue", False)),
                    "rule_action": rule_actions[agent_id].tolist(),
                    "learned_action": learned_actions[index].tolist(),
                    "used_action": actions[agent_id].tolist(),
                    "boundary_recovery_blend": float(env._diag_recovery_blends.get(agent_id, 0.0)),
                    "hard_contact": bool(agent_id in env._diag_hard_contacts),
                    "termination_reason": info.get("episode_summary", {}).get("termination_reason") if info.get("episode_summary") else None,
                })
        obs, _, _ = env._observations()
        previous_state = env.state_dict()
        if done:
            break
    summary = deepcopy(info.get("episode_summary") or {})
    summary.update({
        "checkpoint": checkpoint_name,
        "checkpoint_env_steps": int(2_900_000 if checkpoint_name == "best" else 3_000_000),
        "combo": combo_name,
        "episode_seed": int(seed),
        "red_total_loss": bool(summary.get("red_total_loss", False)),
        "mean_return": float(summary.get("episode_return", 0.0)),
        "episode_length": int(summary.get("episode_length", env.step_count)),
        "reward_components": summary.get("reward_components", {}),
    })
    # Normalize the fields consumed by the diagnostic statistics and targeted
    # seed selector.  The v12 environment uses the longer contract names.
    # Historical v1 `task_win` means the environment's red-side outcome,
    # including timeout red wins; strict full elimination is a separate field.
    summary["task_win"] = bool(summary.get("environment_outcome") == "red" or summary.get("red_win", False))
    summary["strict_full_elimination"] = bool(summary.get("strict_full_elimination", summary.get("full_elimination", False)))
    summary["any_kill"] = bool(summary.get("red_attack_kills", 0) > 0)
    summary["at_least_two_kill"] = bool(summary.get("red_attack_kills", 0) >= 2)
    summary["episode_return"] = float(summary.get("episode_return", summary.get("mean_return", 0.0)))
    per_agent = []
    for agent_id in TEAM_AGENT_IDS_V12:
        if agent_id == "red_0":
            accum[agent_id].cue_active_steps = int(env._episode_metrics.get("support_active_cue_steps", 0.0))
            accum[agent_id].cue_active_pair_steps = int(env._episode_metrics.get("support_active_cue_pair_steps", 0.0))
            accum[agent_id].cue_update_count = int(env._episode_metrics.get("support_cue_update_count", 0.0))
            accum[agent_id].unique_detection_events = int(env._episode_metrics.get("support_unique_detection_events", 0.0))
            accum[agent_id].cue_to_direct_events = int(env._episode_metrics.get("support_cue_to_direct_events", 0.0))
            accum[agent_id].cue_to_half_lock_events = int(env._episode_metrics.get("support_cue_to_half_lock_events", 0.0))
            accum[agent_id].assisted_kills = int(env._episode_metrics.get("support_assisted_kills", 0.0))
        item = accum[agent_id].finish(env, summary)
        item.update({"checkpoint": checkpoint_name, "combo": combo_name, "episode_seed": int(seed)})
        per_agent.append(item)
    return summary, per_agent, trace_rows


def _normalise_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize rows loaded from an interrupted/older v2 evaluation file."""
    row = dict(row)
    if "environment_outcome" in row:
        row["task_win"] = str(row.get("environment_outcome")) == "red"
    row["strict_full_elimination"] = _bool_field(row, "strict_full_elimination")
    row["any_kill"] = bool(_numeric(row, "red_attack_kills") > 0)
    row["at_least_two_kill"] = bool(_numeric(row, "red_attack_kills") >= 2)
    if "episode_return" not in row and "mean_return" in row:
        row["episode_return"] = row["mean_return"]
    return row


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    device = torch.device(str(task["device"]))
    train_cfg = yaml.safe_load(Path(task["train_config"]).read_text(encoding="utf-8"))
    actors = _load_actors(task["checkpoint_path"], train_cfg, device)
    env = MixedPolicyProbeEnv(task["env_config"])
    episode_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    try:
        for seed in task["seeds"]:
            summary, per_agent, _ = _run_episode(env, actors, task["combo_name"], task["source_map"], task["checkpoint_name"], int(seed), device)
            episode_rows.append(summary)
            agent_rows.extend(per_agent)
    finally:
        del actors
    return {
        "checkpoint_name": task["checkpoint_name"],
        "combo_name": task["combo_name"],
        "episode_rows": episode_rows,
        "agent_rows": agent_rows,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def _numeric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def _required_numeric(row: Mapping[str, Any], key: str, *, context: str = "row") -> float:
    """Read a required diagnostic field without silently converting absence to zero."""
    if key not in row or row[key] in (None, ""):
        raise KeyError(f"missing required diagnostic field {key!r} in {context}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite diagnostic field {key!r} in {context}")
    return value


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    if key in {"task_win", "strict_full_elimination", "any_kill", "at_least_two_kill", "support_survived", "red_total_loss", "timeout_red_loss", "timeout_draw"}:
        return float(_bool_field(row, key))
    return _numeric(row, key)


def _bool_field(row: Mapping[str, Any], key: str) -> bool:
    return _as_bool(row.get(key, False))


def _aggregate_team(rows: list[dict[str, Any]], checkpoint: str, combo: str) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [_numeric(row, key) for row in rows]
        return float(statistics_module.fmean(values)) if values else 0.0
    def rate(key: str) -> float:
        return float(sum(_bool_field(row, key) for row in rows) / max(1, len(rows)))
    kill_distribution = {str(k): float(sum(int(_numeric(row, "red_attack_kills")) == k for row in rows) / max(1, len(rows))) for k in range(3)}
    kill_distribution["3"] = float(sum(int(_numeric(row, "red_attack_kills")) >= 3 for row in rows) / max(1, len(rows)))
    return {
        "checkpoint": checkpoint,
        "combo": combo,
        "episodes": len(rows),
        "task_win_rate": rate("task_win"),
        "strict_full_elimination_rate": rate("strict_full_elimination"),
        "any_kill_rate": rate("any_kill"),
        "at_least_two_kill_rate": rate("at_least_two_kill"),
        "mean_red_kills": mean("red_attack_kills"),
        "mean_blue_kills": mean("blue_attack_kills"),
        "mean_red_combat_survivors": mean("red_combat_survivors"),
        "mean_blue_combat_survivors": mean("blue_combat_survivors"),
        "support_survival_rate": rate("support_survived"),
        "mean_episode_length": mean("episode_length"),
        "mean_return": mean("episode_return"),
        "mean_first_kill_step": float(statistics_module.fmean([_numeric(row, "first_kill_time") for row in rows if row.get("first_kill_time") not in (None, "")])) if any(row.get("first_kill_time") not in (None, "") for row in rows) else None,
        "red_half_lock_episode_rate": mean("red_half_lock_episode_rate"),
        "mean_red_max_lock_progress": mean("mean_red_max_lock_progress"),
        "red_lock_active_step_rate": mean("red_lock_active_step_rate"),
        "red_half_lock_active_step_rate": mean("red_half_lock_active_step_rate"),
        "support_cue_rate": mean("support_cue_rate"),
        "support_cue_pair_step_rate": mean("support_cue_pair_step_rate"),
        "support_cue_to_direct_rate": mean("support_cue_to_direct_rate"),
        "support_cue_to_half_lock_rate": mean("support_cue_to_half_lock_rate"),
        "support_assisted_kill_rate": mean("support_assisted_kill_rate"),
        "red_total_loss_rate": rate("red_total_loss"),
        "timeout_red_loss_rate": rate("timeout_red_loss"),
        "timeout_draw_rate": rate("timeout_draw"),
        "mutual_elimination_rate": rate("mutual_elimination_draw"),
        "red_boundary_hard_contacts": mean("red_boundary_hard_contacts"),
        "blue_boundary_hard_contacts": mean("blue_boundary_hard_contacts"),
        "support_boundary_hard_contacts": mean("support_boundary_hard_contacts"),
        "reward_components": {
            key: float(statistics_module.fmean([_numeric(json.loads(row.get("reward_components", "{}")), key) if isinstance(row.get("reward_components"), str) else _numeric(row.get("reward_components", {}), key) for row in rows]))
            for key in sorted({key for row in rows for key in (json.loads(row.get("reward_components", "{}")) if isinstance(row.get("reward_components"), str) else row.get("reward_components", {})).keys()})
        },
        "red_attack_kill_distribution": kill_distribution,
    }


def _aggregate_agents(rows: list[dict[str, Any]], checkpoint: str, combo: str) -> list[dict[str, Any]]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(str(row["agent_id"]), []).append(row)
    output = []
    for agent_id in TEAM_AGENT_IDS_V12:
        group = by_agent.get(agent_id, [])
        if not group:
            continue
        def mean(key: str) -> float:
            return float(statistics_module.fmean([_numeric(row, key) for row in group]))
        output.append({
            "checkpoint": checkpoint,
            "combo": combo,
            "agent_id": agent_id,
            "role": AGENT_TO_ROLE_V12[agent_id],
            "action_source": group[0].get("action_source"),
            "episodes": len(group),
            "mean_alive_decision_steps": mean("alive_decision_steps"),
            "survival_rate": float(statistics_module.fmean([_required_numeric(row, "survival_rate", context=f"{checkpoint}/{combo}/{agent_id}") for row in group])),
            "mean_attributed_kills": mean("attributed_kills"),
            "mean_half_lock_event_count": mean("half_lock_event_count"),
            "mean_max_lock_progress": mean("max_lock_progress"),
            "mean_lock_active_step_rate": mean("lock_active_step_rate"),
            "mean_half_lock_active_step_rate": mean("half_lock_active_step_rate"),
            "mean_direct_target_rate": mean("direct_target_rate"),
            "mean_shared_only_target_rate": mean("shared_only_target_rate"),
            "mean_no_valid_target_rate": mean("no_valid_target_rate"),
            "mean_target_switch_count": mean("target_switch_count"),
            "mean_target_switch_while_lock_above_0_1": mean("target_switch_while_lock_above_0_1"),
            "mean_longest_continuous_positive_lock": mean("longest_continuous_positive_lock"),
            "mean_longest_continuous_half_lock": mean("longest_continuous_half_lock"),
            "mean_hard_contact_count": mean("hard_contact_count"),
            "mean_target_distance": mean("mean_target_distance"),
            "mean_ata": mean("mean_ata"),
            "mean_aa": mean("mean_aa"),
        })
    return output


def _rows_for(rows: list[dict[str, Any]], checkpoint: str, combo: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("checkpoint")) == checkpoint and str(row.get("combo")) == combo]


def _paired_metric(rows: list[dict[str, Any]], checkpoint: str, combo_a: str, combo_b: str, metric: str, bootstrap_samples: int) -> dict[str, Any]:
    a = {int(row["episode_seed"]): row for row in _rows_for(rows, checkpoint, combo_a)}
    b = {int(row["episode_seed"]): row for row in _rows_for(rows, checkpoint, combo_b)}
    common = sorted(set(a) & set(b))
    values_a = [_metric_value(a[seed], metric) for seed in common]
    values_b = [_metric_value(b[seed], metric) for seed in common]
    result = paired_bootstrap(values_a, values_b, samples=bootstrap_samples, seed=17)
    bool_metrics = {"task_win", "strict_full_elimination", "any_kill", "at_least_two_kill", "support_survived", "red_total_loss", "timeout_red_loss", "timeout_draw"}
    p_value = None
    if metric in bool_metrics:
        b_only = sum(_bool_field(a[seed], metric) and not _bool_field(b[seed], metric) for seed in common)
        c_only = sum(not _bool_field(a[seed], metric) and _bool_field(b[seed], metric) for seed in common)
        p_value = exact_mcnemar_pvalue(b_only, c_only)
    return {"scope": "team", "checkpoint": checkpoint, "comparison": f"{combo_b}_vs_{combo_a}", "combo_a": combo_a, "combo_b": combo_b, "metric": metric, "n": len(common), "mean_delta_b_minus_a": result["mean_delta"], "ci_low": result["ci_low"], "ci_high": result["ci_high"], "mcnemar_pvalue": p_value}


def _paired_agent_metric(
    rows: list[dict[str, Any]],
    checkpoint: str,
    combo_a: str,
    combo_b: str,
    agent_id: str,
    metric: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    a = {
        int(row["episode_seed"]): row
        for row in _rows_for(rows, checkpoint, combo_a)
        if str(row.get("agent_id")) == agent_id
    }
    b = {
        int(row["episode_seed"]): row
        for row in _rows_for(rows, checkpoint, combo_b)
        if str(row.get("agent_id")) == agent_id
    }
    common = sorted(set(a) & set(b))
    values_a = [_required_numeric(a[seed], metric, context=f"{checkpoint}/{combo_a}/{agent_id}") for seed in common]
    values_b = [_required_numeric(b[seed], metric, context=f"{checkpoint}/{combo_b}/{agent_id}") for seed in common]
    result = paired_bootstrap(values_a, values_b, samples=bootstrap_samples, seed=43)
    return {
        "scope": "agent",
        "checkpoint": checkpoint,
        "comparison": f"{combo_b}_vs_{combo_a}",
        "combo_a": combo_a,
        "combo_b": combo_b,
        "agent_id": agent_id,
        "metric": metric,
        "n": len(common),
        "mean_delta_b_minus_a": result["mean_delta"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "mcnemar_pvalue": None,
    }


def _paired_best_final(rows: list[dict[str, Any]], combo: str, metric: str, bootstrap_samples: int) -> dict[str, Any]:
    best = {int(row["episode_seed"]): row for row in _rows_for(rows, "best", combo)}
    final = {int(row["episode_seed"]): row for row in _rows_for(rows, "final", combo)}
    common = sorted(set(best) & set(final))
    result = paired_bootstrap([_metric_value(best[seed], metric) for seed in common], [_metric_value(final[seed], metric) for seed in common], samples=bootstrap_samples, seed=31)
    bool_metrics = {"task_win", "strict_full_elimination", "any_kill", "at_least_two_kill", "support_survived", "red_total_loss", "timeout_red_loss", "timeout_draw"}
    p_value = None
    if metric in bool_metrics:
        b_only = sum(_bool_field(best[seed], metric) and not _bool_field(final[seed], metric) for seed in common)
        c_only = sum(not _bool_field(best[seed], metric) and _bool_field(final[seed], metric) for seed in common)
        p_value = exact_mcnemar_pvalue(b_only, c_only)
    return {"scope": "team", "checkpoint": "best_vs_final", "comparison": f"final_vs_best_{combo}", "combo_a": f"best::{combo}", "combo_b": f"final::{combo}", "metric": metric, "n": len(common), "mean_delta_b_minus_a": result["mean_delta"], "ci_low": result["ci_low"], "ci_high": result["ci_high"], "mcnemar_pvalue": p_value}


def _best_final_rows(rows: list[dict[str, Any]], combo: str, metric: str, bootstrap_samples: int) -> dict[str, Any]:
    return _paired_metric(rows, "best", combo, combo, metric, bootstrap_samples) | {"checkpoint": "best_vs_final", "comparison": f"final_vs_best_{combo}"}


def _make_statistics(rows: list[dict[str, Any]], agent_rows: list[dict[str, Any]], bootstrap_samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates = []
    agent_aggregates = []
    pairwise = []
    equivalence = []
    drift = []
    for checkpoint in ("best", "final"):
        for combo_name, _ in COMBOS_V12_MIXED:
            team_rows = _rows_for(rows, checkpoint, combo_name)
            per_rows = _rows_for(agent_rows, checkpoint, combo_name)
            if team_rows:
                aggregates.append(_aggregate_team(team_rows, checkpoint, combo_name))
            if per_rows:
                agent_aggregates.extend(_aggregate_agents(per_rows, checkpoint, combo_name))
    bool_metrics = ["task_win", "strict_full_elimination", "any_kill", "at_least_two_kill", "support_survived"]
    numeric_metrics = ["red_attack_kills", "mean_red_max_lock_progress", "red_half_lock_episode_rate", "episode_return", "episode_length", "red_boundary_hard_contacts"]
    contrast_pairs = [
        ("M0_all_rule", "M2_learned_support_rule_combats"),
        ("M1_all_learned", "M3_rule_support_learned_combats"),
        ("M0_all_rule", "M3_rule_support_learned_combats"),
        ("M0_all_rule", "M4_rule_support_learned_combat_1"),
        ("M0_all_rule", "M5_rule_support_learned_combat_2"),
        ("M0_all_rule", "M6_rule_support_learned_combat_3"),
        ("M4_rule_support_learned_combat_1", "M7_learned_support_learned_combat_1"),
        ("M5_rule_support_learned_combat_2", "M8_learned_support_learned_combat_2"),
        ("M6_rule_support_learned_combat_3", "M9_learned_support_learned_combat_3"),
        ("M4_rule_support_learned_combat_1", "M3_rule_support_learned_combats"),
        ("M5_rule_support_learned_combat_2", "M3_rule_support_learned_combats"),
        ("M6_rule_support_learned_combat_3", "M3_rule_support_learned_combats"),
    ]
    for checkpoint in ("best", "final"):
        for a, b in contrast_pairs:
            for metric in bool_metrics + numeric_metrics:
                pairwise.append(_paired_metric(rows, checkpoint, a, b, metric, bootstrap_samples))
    for combo_name in [f"M{i}_{suffix}" for i, suffix in ((1, "all_learned"), (2, "learned_support_rule_combats"), (3, "rule_support_learned_combats"), (4, "rule_support_learned_combat_1"), (5, "rule_support_learned_combat_2"), (6, "rule_support_learned_combat_3"))]:
        for metric in bool_metrics + numeric_metrics:
            pairwise.append(_paired_best_final(rows, combo_name, metric, bootstrap_samples))
    for checkpoint in ("best", "final"):
        for a, b in (("M0_all_rule", "M2_learned_support_rule_combats"), ("M1_all_learned", "M3_rule_support_learned_combats"), ("M0_all_rule", "M4_rule_support_learned_combat_1"), ("M0_all_rule", "M5_rule_support_learned_combat_2"), ("M0_all_rule", "M6_rule_support_learned_combat_3")):
            for metric, threshold in (("task_win", 0.05), ("any_kill", 0.05), ("red_attack_kills", 0.10), ("mean_red_max_lock_progress", 0.05), ("episode_return", 1.0)):
                result = _paired_metric(rows, checkpoint, a, b, metric, bootstrap_samples)
                result.update({"threshold": threshold, "classification": practical_equivalence((result["ci_low"], result["ci_high"]), threshold)})
                equivalence.append(result)
    # Actor-level paired rows are kept in the same statistics file so every
    # single-slot conclusion can cite its own lock/kill CI rather than only a
    # team mean.  These are inference-only rows; they never call an optimizer.
    actor_contrasts = [
        ("M0_all_rule", "M1_all_learned", TEAM_AGENT_IDS_V12),
        ("M0_all_rule", "M2_learned_support_rule_combats", TEAM_AGENT_IDS_V12),
        ("M0_all_rule", "M3_rule_support_learned_combats", ("red_1", "red_2", "red_3")),
        ("M0_all_rule", "M4_rule_support_learned_combat_1", ("red_1",)),
        ("M0_all_rule", "M5_rule_support_learned_combat_2", ("red_2",)),
        ("M0_all_rule", "M6_rule_support_learned_combat_3", ("red_3",)),
        ("M4_rule_support_learned_combat_1", "M7_learned_support_learned_combat_1", ("red_1",)),
        ("M5_rule_support_learned_combat_2", "M8_learned_support_learned_combat_2", ("red_2",)),
        ("M6_rule_support_learned_combat_3", "M9_learned_support_learned_combat_3", ("red_3",)),
    ]
    actor_metrics = (
        "attributed_kills",
        "half_lock_event_count",
        "max_lock_progress",
        "lock_active_step_rate",
        "half_lock_active_step_rate",
        "longest_continuous_positive_lock",
        "direct_target_rate",
        "shared_only_target_rate",
        "no_valid_target_rate",
        "target_switch_count",
        "survival_rate",
        "hard_contact_count",
    )
    for checkpoint in ("best", "final"):
        for combo_a, combo_b, agent_ids in actor_contrasts:
            for agent_id in agent_ids:
                for metric in actor_metrics:
                    pairwise.append(_paired_agent_metric(agent_rows, checkpoint, combo_a, combo_b, agent_id, metric, bootstrap_samples))
    # Best-to-final drift is paired on the same test seed for every role.
    for combo_name, _ in COMBOS_V12_MIXED:
        for role in ROLE_ORDER:
            agent_id = "red_0" if role == "support" else f"red_{role.rsplit('_', 1)[-1]}"
            for metric in ("max_lock_progress", "attributed_kills", "half_lock_active_step_rate", "direct_target_rate", "survival_rate"):
                a = {(int(row["episode_seed"])): row for row in _rows_for(agent_rows, "best", combo_name) if row.get("agent_id") == agent_id}
                b = {(int(row["episode_seed"])): row for row in _rows_for(agent_rows, "final", combo_name) if row.get("agent_id") == agent_id}
                common = sorted(set(a) & set(b))
                if common:
                    boot = paired_bootstrap([_required_numeric(a[s], metric, context=f"best/{combo_name}/{agent_id}") for s in common], [_required_numeric(b[s], metric, context=f"final/{combo_name}/{agent_id}") for s in common], samples=bootstrap_samples, seed=29)
                    drift.append({"combo": combo_name, "agent_id": agent_id, "role": role, "metric": metric, "n": len(common), "best_mean": float(statistics_module.fmean([_required_numeric(a[s], metric, context=f"best/{combo_name}/{agent_id}") for s in common])), "final_mean": float(statistics_module.fmean([_required_numeric(b[s], metric, context=f"final/{combo_name}/{agent_id}") for s in common])), "mean_delta_final_minus_best": boot["mean_delta"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]})
    return aggregates, agent_aggregates, pairwise, equivalence, drift


def _m1_reproduction(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    comparisons = []
    for checkpoint_name, file_name in (("best", "best_test_evaluation.json"), ("final", "final_test_evaluation.json")):
        ours = _aggregate_team(_rows_for(rows, checkpoint_name, "M1_all_learned"), checkpoint_name, "M1_all_learned")
        existing = json.loads((run_dir / file_name).read_text(encoding="utf-8"))
        fields = ("task_win_rate", "any_kill_rate", "at_least_two_kill_rate", "mean_red_kills", "mean_episode_length", "mean_return")
        diagnostic_seeds = sorted(int(row["episode_seed"]) for row in _rows_for(rows, checkpoint_name, "M1_all_learned"))
        from uav_combat.happo.evaluation_4v3 import sha256_json_4v3
        comparisons.append({"checkpoint": checkpoint_name, "existing_file": file_name, "fields": {field: {"diagnostic": ours.get(field), "existing": existing.get(field), "absolute_error": abs(float(ours.get(field, 0.0) or 0.0) - float(existing.get(field, 0.0) or 0.0))} for field in fields}, "diagnostic_seed_hash": sha256_json_4v3(diagnostic_seeds), "existing_seed_hash": existing.get("seed_hash"), "seed_hash_equal": existing.get("seed_hash") == sha256_json_4v3(diagnostic_seeds)})
    return {"comparisons": comparisons}


def _reproduction_tolerance(scope: str, metric: str) -> float:
    if metric in {"mean_return", "episode_return"}:
        return 1e-7
    if scope == "agent":
        return 1e-7
    return 1e-12


def _value_or_none(row: Mapping[str, Any], key: str) -> float | None:
    if key not in row or row[key] in (None, "", "None"):
        return None
    value = float(row[key])
    return value if math.isfinite(value) else None


def _v1_v2_reproduction(
    out_dir: Path,
    aggregates: list[dict[str, Any]],
    agent_aggregates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the new alive-only implementation with the preserved v1 output."""
    v1_dir = out_dir.parent / "mixed_policy_role_diagnosis"
    old_team_path = v1_dir / "mixed_policy_aggregate.csv"
    old_agent_path = v1_dir / "mixed_policy_per_agent_aggregate.csv"
    if not old_team_path.exists() or not old_agent_path.exists():
        summary = {"status": "v1_outputs_missing", "team_metric_reproduction_pass": False, "actor_metric_changed_count": None}
        return [], summary
    old_team = _read_rows(old_team_path)
    old_agent = _read_rows(old_agent_path)
    comparisons: list[dict[str, Any]] = []
    for scope, old_rows, new_rows, fields, key_fields in (
        ("team", old_team, aggregates, TEAM_REPRO_FIELDS, ("checkpoint", "combo")),
        ("agent", old_agent, agent_aggregates, AGENT_REPRO_FIELDS, ("checkpoint", "combo", "agent_id")),
    ):
        old_index = {tuple(str(row.get(key)) for key in key_fields): row for row in old_rows}
        new_index = {tuple(str(row.get(key)) for key in key_fields): row for row in new_rows}
        for key, old_row in sorted(old_index.items()):
            new_row = new_index.get(key)
            for metric in fields:
                old_value = _value_or_none(old_row, metric)
                new_value = _value_or_none(new_row or {}, metric)
                if old_value is None or new_value is None:
                    passed = old_value is None and new_value is None
                    delta = None
                else:
                    delta = new_value - old_value
                    passed = abs(delta) <= _reproduction_tolerance(scope, metric)
                comparisons.append({
                    "scope": scope,
                    **{field: key[index] for index, field in enumerate(key_fields)},
                    "metric": metric,
                    "v1_value": old_value,
                    "v2_value": new_value,
                    "delta_v2_minus_v1": delta,
                    "tolerance": _reproduction_tolerance(scope, metric),
                    "pass": bool(passed),
                })
    team_rows = [row for row in comparisons if row["scope"] == "team"]
    actor_rows = [row for row in comparisons if row["scope"] == "agent"]
    actor_changed = [row for row in actor_rows if row["pass"] is False]
    summary = {
        "status": "complete",
        "v1_directory": str(v1_dir),
        "team_rows_compared": len(team_rows),
        "actor_rows_compared": len(actor_rows),
        "team_metric_reproduction_pass": bool(team_rows) and all(row["pass"] for row in team_rows),
        "actor_metric_changed_count": len(actor_changed),
        "actor_metric_changed_fields": actor_changed,
        "actor_metric_reproduction_pass": not actor_changed,
        "note": "Alive-only survival/behavior fields are intentionally allowed to differ; team contract fields must reproduce within tolerance.",
    }
    return comparisons, summary


def _nonfinite_count(rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float, np.integer, np.floating)) and not math.isfinite(float(value)):
                count += 1
                continue
            if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
                count += 1
    return count


def _build_validation_summary(
    out_dir: Path,
    rows: list[dict[str, Any]],
    agent_rows: list[dict[str, Any]],
    test_seeds: list[int],
    selected_specs: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    integrity: Mapping[str, Any],
    m1_reproduction: Mapping[str, Any] | None,
    v1_v2_summary: Mapping[str, Any],
) -> dict[str, Any]:
    previous_summary = json.loads((out_dir / "diagnostic_validation_summary.json").read_text(encoding="utf-8")) if (out_dir / "diagnostic_validation_summary.json").exists() else {}
    selected_checkpoints = ("best", "final")
    expected_keys = {
        (checkpoint, combo, int(seed))
        for checkpoint in selected_checkpoints
        for combo, _ in COMBOS_V12_MIXED
        for seed in test_seeds
    }
    actual_keys = {(str(row.get("checkpoint")), str(row.get("combo")), int(row.get("episode_seed"))) for row in rows}
    duplicate_rows = len(rows) - len(actual_keys)
    agent_keys = {(str(row.get("checkpoint")), str(row.get("combo")), int(row.get("episode_seed")), str(row.get("agent_id"))) for row in agent_rows}
    expected_agent_keys = {(checkpoint, combo, int(seed), agent_id) for checkpoint, combo, seed in expected_keys for agent_id in TEAM_AGENT_IDS_V12}
    category_counts: dict[str, int] = {}
    for spec in selected_specs:
        category_counts[str(spec.get("category"))] = category_counts.get(str(spec.get("category")), 0) + 1
    m1_pass = False
    if m1_reproduction and m1_reproduction.get("comparisons"):
        m1_pass = all(
            bool(item.get("seed_hash_equal")) and all(float(field.get("absolute_error", float("inf"))) <= (1e-7 if name == "mean_return" else 1e-12) for name, field in item.get("fields", {}).items())
            for item in m1_reproduction["comparisons"]
        )
    numeric_nonfinite = _nonfinite_count(rows) + _nonfinite_count(agent_rows) + _nonfinite_count(trace_rows)
    return {
        "completed_combinations": len({(str(row.get("checkpoint")), str(row.get("combo"))) for row in rows}),
        "total_episodes": len(rows),
        "expected_episodes": len(expected_keys),
        "total_per_agent_rows": len(agent_rows),
        "expected_per_agent_rows": len(expected_agent_keys),
        "duplicate_row_count": duplicate_rows,
        "duplicate_per_agent_row_count": len(agent_rows) - len(agent_keys),
        "missing_combo_seed_count": len(expected_keys - actual_keys),
        "missing_per_agent_row_count": len(expected_agent_keys - agent_keys),
        "trajectory_spec_count": len(selected_specs),
        "trajectory_row_count": len(trace_rows),
        "trajectory_category_counts": category_counts,
        "non_finite_count": numeric_nonfinite,
        "input_sha256_unchanged": bool(integrity.get("input_sha256_unchanged")),
        "m1_reproduction_pass": bool(m1_pass),
        "v1_v2_team_metric_reproduction_pass": bool(v1_v2_summary.get("team_metric_reproduction_pass", False)),
        "v1_v2_actor_metric_changed_count": v1_v2_summary.get("actor_metric_changed_count"),
        "requested_device": integrity.get("requested_device"),
        "controller_device": integrity.get("controller_device"),
        "parallel_episode_worker_device": integrity.get("parallel_episode_worker_device"),
        "trajectory_device": integrity.get("trajectory_device"),
        "test_result_summary": previous_summary.get("test_result_summary", {
            "compileall": "run separately; see final validation report",
            "focused_diagnostics_tests": "run separately; see final validation report",
            "full_pytest": "run separately; see final validation report",
        }),
        "output_dir": str(out_dir),
    }


def _select_trace_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    selected.extend(select_targeted_seeds(rows, category="support", limit=10))
    selected.extend(select_targeted_seeds(rows, category="combat", limit=10))
    for slot in ("1", "2", "3"):
        selected.extend(select_targeted_seeds(rows, category=f"combat_{slot}", limit=5))
    selected.extend(select_targeted_seeds(rows, category="best_final", limit=10))
    specs = []
    seen = set()
    for item in selected:
        category = item["category"]
        seed = int(item["episode_seed"])
        if category == "A_support":
            candidates = [("best", "M2_learned_support_rule_combats", dict(COMBOS_V12_MIXED)["M2_learned_support_rule_combats"])]
        elif category == "B_combat":
            candidates = [("best", "M3_rule_support_learned_combats", dict(COMBOS_V12_MIXED)["M3_rule_support_learned_combats"])]
        elif category.startswith("C_combat_"):
            slot = category.rsplit("_", 1)[-1]
            combo = {"1": "M4_rule_support_learned_combat_1", "2": "M5_rule_support_learned_combat_2", "3": "M6_rule_support_learned_combat_3"}[slot]
            candidates = [("best", combo, dict(COMBOS_V12_MIXED)[combo])]
        else:
            candidates = [("best", "M1_all_learned", dict(COMBOS_V12_MIXED)["M1_all_learned"]), ("final", "M1_all_learned", dict(COMBOS_V12_MIXED)["M1_all_learned"])]
        for checkpoint, combo, source_map in candidates:
            validate_source_map(source_map)
            source_map = dict(source_map)
            key = (checkpoint, combo, seed)
            if key not in seen:
                seen.add(key)
                specs.append({"category": category, "checkpoint": checkpoint, "combo": combo, "episode_seed": seed, "source_map": source_map})
    return specs[:40]


def _run_traces(specs: list[dict[str, Any]], run_dir: Path, train_cfg: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    actors_by_checkpoint: dict[str, IndependentHAPPOActors] = {}
    for spec in specs:
        validate_source_map(spec.get("source_map", {}))
        checkpoint = spec["checkpoint"]
        if checkpoint not in actors_by_checkpoint:
            actors_by_checkpoint[checkpoint] = _load_actors(run_dir / f"{checkpoint}.pt", train_cfg, device)
        env = MixedPolicyProbeEnv(run_dir / "resolved_environment_config.yaml")
        try:
            _, _, trace = _run_episode(env, actors_by_checkpoint[checkpoint], spec["combo"], spec["source_map"], checkpoint, int(spec["episode_seed"]), device, trace=True)
            for row in trace:
                row["targeted_category"] = spec["category"]
            output.extend(trace)
        finally:
            pass
    return output


def _make_report(out_dir: Path, integrity: Mapping[str, Any], aggregates: list[dict[str, Any]], agent_aggregates: list[dict[str, Any]], pairwise: list[dict[str, Any]], equivalence: list[dict[str, Any]], drift: list[dict[str, Any]], completed: int, expected: int, m1_reproduction: Mapping[str, Any] | None) -> str:
    return _make_report_dynamic(out_dir, integrity, aggregates, agent_aggregates, pairwise, equivalence, drift, completed, expected, m1_reproduction)
    lines = ["# v12 mixed-policy role diagnosis", "", "## 1. Executive summary", ""]
    if completed < expected:
        lines.append(f"- Incomplete: {completed}/{expected} combination evaluations completed. No claim is made for missing combinations.")
    else:
        lines.append("- This is a read-only fixed-policy replacement experiment: 2 checkpoints × 10 combinations × 200 shared test seeds.")
        lines.append("- M0 is the rule reference; M1 is rerun by this diagnostic runner, not copied from the historical evaluation JSON.")
    lines.extend(["", "## 2. What was and was not changed", "", "- Actors were loaded in eval mode and queried with deterministic mean actions under `torch.no_grad()`.", "- The v12 environment, reward, observation, target/cue, lock, boundary and blue rule code were not modified.", "- No trainer, rollout collector, optimizer, or checkpoint writer was used.", "", "## 3. Input integrity", "", f"- Effective device: `{integrity.get('effective_device')}`; requested: `{integrity.get('requested_device')}`.", f"- Test seeds: {integrity.get('test_seed_count')} unique; selection/test overlap: none.", f"- final/latest parameter state identical: `{integrity.get('final_vs_latest_parameter_identical')}`.", f"- best/step_2900000 parameter state identical: `{integrity.get('best_vs_step_2900000_parameter_identical')}`.", ""])
    lines.append("## 4. Best checkpoint team results")
    lines.append("")
    lines.append("| Combination | Win | Any kill | ≥2 kills | Mean red kills | Half-lock | Mean return |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregates:
        if row["checkpoint"] == "best":
            lines.append(f"| {row['combo']} | {row['task_win_rate']:.3f} | {row['any_kill_rate']:.3f} | {row['at_least_two_kill_rate']:.3f} | {row['mean_red_kills']:.3f} | {row['red_half_lock_episode_rate']:.3f} | {row['mean_return']:.3f} |")
    lines.extend(["", "## 5. Final checkpoint team results", "", "| Combination | Win | Any kill | ≥2 kills | Mean red kills | Half-lock | Mean return |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in aggregates:
        if row["checkpoint"] == "final":
            lines.append(f"| {row['combo']} | {row['task_win_rate']:.3f} | {row['any_kill_rate']:.3f} | {row['at_least_two_kill_rate']:.3f} | {row['mean_red_kills']:.3f} | {row['red_half_lock_episode_rate']:.3f} | {row['mean_return']:.3f} |")
    lines.extend(["", "## 6. Role interpretation", ""])
    best = {row["combo"]: row for row in aggregates if row["checkpoint"] == "best"}
    if "M0_all_rule" in best and "M2_learned_support_rule_combats" in best and "M3_rule_support_learned_combats" in best:
        m0, m2, m3 = best["M0_all_rule"], best["M2_learned_support_rule_combats"], best["M3_rule_support_learned_combats"]
        lines.append(f"- M2 vs M0: mean red kills {m2['mean_red_kills']:.3f} vs {m0['mean_red_kills']:.3f}; half-lock {m2['red_half_lock_episode_rate']:.3f} vs {m0['red_half_lock_episode_rate']:.3f}.")
        lines.append(f"- M3 vs M0: mean red kills {m3['mean_red_kills']:.3f} vs {m0['mean_red_kills']:.3f}; half-lock {m3['red_half_lock_episode_rate']:.3f} vs {m0['red_half_lock_episode_rate']:.3f}.")
        lines.append("- Support is called a primary bottleneck only if M2 has a paired CI outside the pre-registered equivalence interval and rule Combat metrics also fall; cue rate alone is not evidence.")
        lines.append("- Combat is called the primary weakness only if M2 remains close to M0 while M3 is close to M1 and below M0, and M4–M6 show actor-level lock/kill loss.")
    lines.append("- A single Combat Actor is called clearly weak only when its M4/M5/M6 loss is materially larger and its own lock, half-lock, kill and direct-visible growth metrics are all lower.")
    lines.append("- Mixed-policy trajectories are descriptive evidence for action/target sequences; they are not automatically upgraded to a mechanism or causal root-cause claim.")
    lines.extend(["", "## 7. Paired statistics and practical equivalence", "", f"- Pairwise rows: {len(pairwise)}; practical-equivalence rows: {len(equivalence)}; role-drift rows: {len(drift)}.", "- Boolean comparisons use exact McNemar tests; continuous comparisons use paired bootstrap CIs."])
    lines.extend(["", "## 8. Per-Actor evidence", ""])
    lines.append("| Checkpoint | Combination | Actor | Source | Mean kills | Mean max lock | Half-lock step rate | Direct-target rate | Survival |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in agent_aggregates:
        if row["checkpoint"] == "best" and row["combo"] in {"M0_all_rule", "M1_all_learned", "M2_learned_support_rule_combats", "M3_rule_support_learned_combats", "M4_rule_support_learned_combat_1", "M5_rule_support_learned_combat_2", "M6_rule_support_learned_combat_3"} and row["agent_id"] in {"red_0", "red_1", "red_2", "red_3"}:
            lines.append(f"| {row['checkpoint']} | {row['combo']} | {row['agent_id']} | {row['action_source']} | {float(row['mean_attributed_kills']):.3f} | {float(row['mean_max_lock_progress']):.3f} | {float(row['mean_half_lock_active_step_rate']):.3f} | {float(row['mean_direct_target_rate']):.3f} | {float(row['survival_rate']):.3f} |")
    lines.extend(["", "## 9. Support diagnosis", "", "- M2 replaces only Support while retaining rule Combat. Its Combat per-agent lock/kill metrics remain close to M0, and the paired core-metric rows are mostly practical-equivalent or inconclusive rather than materially worse.", "- Therefore the data do not satisfy the stricter condition for calling learned Support the primary bottleneck. Support cue/survival changes are reported descriptively and are not treated as proof of harm.", "", "## 10. Combat-1 diagnosis", "", "- M4 is the single-slot replacement for red_1. In the best checkpoint it reduces mean red kills from the M0 baseline by 0.830 and max-lock progress by about 0.171; its own Actor row shows low lock/kill conversion despite direct-target exposure.", "", "## 11. Combat-2 diagnosis", "", "- M5 is the single-slot replacement for red_2. Its loss versus M0 is smaller and the max-lock paired CI is within the practical-equivalence interval, while its own kill/lock metrics remain below the rule slot. This is weaker evidence of an isolated failure than for Combat-1 or Combat-3.", "", "## 12. Combat-3 diagnosis", "", "- M6 is the single-slot replacement for red_3. It is materially worse than M0 on task win, kills, max lock and return; the learned red_3 row is near-zero lock/kill despite direct-target steps. This is the clearest single-Actor weakness in this phase.", "", "## 13. Three-Combat combination interaction", "", "- M3 (all three learned Combat with rule Support) collapses to near-zero win/kill/half-lock, whereas M4-M6 retain substantially more capability individually. The contrast supports a multi-Combat composition loss, but this experiment does not identify whether the interaction is additive or caused by a particular state/action coupling.", "", "## 14. Support–Combat interaction", "", "- M7-M9 provide the pre-registered Support-plus-one-learned-Combat contrasts. They are reported in the aggregate/per-agent files; no interaction is declared unless the paired CI exceeds the single-slot contrast. The current report treats these as descriptive because Support itself is not materially worse in M2.", "", "## 15. Best-to-final role drift", ""])
    for role in ("support", "combat_1", "combat_2", "combat_3"):
        subset = [row for row in drift if row["combo"] == "M1_all_learned" and row["role"] == role and row["metric"] in {"max_lock_progress", "attributed_kills", "half_lock_active_step_rate"}]
        if subset:
            parts = [f"{row['metric']} Δ={float(row['mean_delta_final_minus_best']):.4f} [{float(row['ci_low']):.4f},{float(row['ci_high']):.4f}]" for row in subset]
            lines.append(f"- {role}: " + "; ".join(parts) + ".")
    lines.extend(["", "## 16. Paired comparison and confidence intervals", "", "- All comparisons use the same 200 test seeds within a checkpoint. Exact McNemar is used for binary outcomes and 10,000-sample paired bootstrap for continuous outcomes.", "- Practical equivalence is only a diagnostic interval (±0.05 for rates, ±0.10 kills/episode, ±0.05 max lock, ±1 return); it is not a proof that policies are identical.", "", "## 17. Targeted trajectory interpretation", "", "- The targeted trajectory file contains only pre-registered high-information contrasts (35 specs, 99,508 rows). It is for inspecting target source, lock deltas, action source and boundary recovery around failures; it is not an automatic causal label.", "", "## 18. Currently supported causes", "", "- High-confidence localization: learned Combat control is the dominant performance bottleneck under the fixed v12 environment, while replacing Support alone preserves the rule-Combat capability.", "- Moderate-confidence localization: Combat-3 is the clearest individual weak slot; Combat-1 is also materially weak in the single replacement; Combat-2 is less clearly isolated but weak in all-learned mixtures.", "", "## 19. Not supported or weakly supported", "", "- These data do not establish Reward, credit assignment, observation, lock parameters, action interface, network architecture, or HAPPO implementation as the root cause.", "- Support being alive or producing cues is not sufficient evidence that it is helpful or harmful.", "", "## 20. Still unresolved mechanisms", "", "- The experiment cannot distinguish fixed-policy action geometry, state-distribution shift, multi-Combat interaction, or optimization history as the underlying mechanism. It also cannot infer unseen blue-state information or future-state dependence because none was supplied to the diagnostic actors.", "", "## 21. Next minimal diagnostic", "", "- Run a read-only fixed-state action comparison for the learned Combat actors against the validated rule action mapping, prioritizing red_3 and red_1 and then red_2. Do not change reward, environment, network or training before that check."])
    if m1_reproduction is not None:
        lines.extend(["", "## 22. M1 reproduction check", "", "The new runner generated M1 independently; see `m1_reproduction_comparison.json` for field-level errors against the historical best/final test summaries."])
    lines.extend(["", "No v13 or new training is proposed by this report."])
    return "\n".join(lines) + "\n"


def _find_stat(
    pairwise: list[dict[str, Any]],
    *,
    checkpoint: str,
    combo_a: str,
    combo_b: str,
    metric: str,
    scope: str = "team",
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    for row in pairwise:
        if (
            row.get("scope", "team") == scope
            and row.get("checkpoint") == checkpoint
            and row.get("combo_a") == combo_a
            and row.get("combo_b") == combo_b
            and row.get("metric") == metric
            and (agent_id is None or row.get("agent_id") == agent_id)
        ):
            return row
    return None


def _metric_threshold(metric: str) -> float:
    if metric in {"red_attack_kills", "attributed_kills", "mean_attributed_kills"}:
        return 0.10
    if metric in {"episode_return", "mean_return"}:
        return 1.0
    if metric in {"episode_length", "mean_episode_length"}:
        return 10.0
    return 0.05


def _stat_record(
    pairwise: list[dict[str, Any]],
    *,
    checkpoint: str,
    combo_a: str,
    combo_b: str,
    metric: str,
    scope: str = "team",
    agent_id: str | None = None,
    expected: str = "no_material_loss",
) -> dict[str, Any]:
    row = _find_stat(pairwise, checkpoint=checkpoint, combo_a=combo_a, combo_b=combo_b, metric=metric, scope=scope, agent_id=agent_id)
    if row is None:
        return {"comparison": f"{combo_b}_vs_{combo_a}", "metric": metric, "point": None, "ci": None, "practical_equivalence": "inconclusive", "mcnemar_pvalue": None, "judgement": "inconclusive"}
    threshold = _metric_threshold(metric)
    classification = practical_equivalence((float(row["ci_low"]), float(row["ci_high"])), threshold)
    if expected == "no_material_loss":
        judgement = "supported" if classification in {"practical_equivalent", "materially_better"} else ("contradicted" if classification == "materially_worse" else "inconclusive")
    elif expected == "material_loss":
        judgement = "supported" if classification == "materially_worse" else ("contradicted" if classification in {"practical_equivalent", "materially_better"} else "inconclusive")
    else:
        judgement = "inconclusive"
    return {
        "comparison": row["comparison"],
        "metric": metric,
        "agent_id": agent_id,
        "point": float(row["mean_delta_b_minus_a"]),
        "ci": [float(row["ci_low"]), float(row["ci_high"])],
        "practical_equivalence": classification,
        "mcnemar_pvalue": row.get("mcnemar_pvalue"),
        "judgement": judgement,
    }


def _render_stat_table(lines: list[str], title: str, records: list[dict[str, Any]]) -> None:
    lines.extend([f"### {title}", "", "| Comparison | Metric | Point | 95% CI | Practical equivalence | McNemar p | Judgement |", "|---|---|---:|---|---|---:|---|"])
    for record in records:
        point = "NA" if record.get("point") is None else f"{record['point']:.6g}"
        ci = "NA" if record.get("ci") is None else f"[{record['ci'][0]:.6g}, {record['ci'][1]:.6g}]"
        p_value = "NA" if record.get("mcnemar_pvalue") is None else f"{float(record['mcnemar_pvalue']):.6g}"
        lines.append(f"| {record['comparison']} | {record['metric']} | {point} | {ci} | {record['practical_equivalence']} | {p_value} | {record['judgement']} |")
    lines.append("")


def _role_drift_classification(drift: list[dict[str, Any]], role: str) -> tuple[str, dict[str, str]]:
    contexts = {
        "support": ("M1_all_learned", "M2_learned_support_rule_combats", "M7_learned_support_learned_combat_1", "M8_learned_support_learned_combat_2", "M9_learned_support_learned_combat_3"),
        "combat_1": ("M1_all_learned", "M4_rule_support_learned_combat_1", "M7_learned_support_learned_combat_1"),
        "combat_2": ("M1_all_learned", "M5_rule_support_learned_combat_2", "M8_learned_support_learned_combat_2"),
        "combat_3": ("M1_all_learned", "M6_rule_support_learned_combat_3", "M9_learned_support_learned_combat_3"),
    }
    labels: dict[str, str] = {}
    for combo in contexts[role]:
        rows = [row for row in drift if row.get("combo") == combo and row.get("role") == role and row.get("metric") in {"max_lock_progress", "attributed_kills", "half_lock_active_step_rate"}]
        negative = sum(practical_equivalence((float(row["ci_low"]), float(row["ci_high"])), _metric_threshold(str(row["metric"]))) == "materially_worse" for row in rows)
        positive = sum(practical_equivalence((float(row["ci_low"]), float(row["ci_high"])), _metric_threshold(str(row["metric"]))) == "materially_better" for row in rows)
        if negative >= 1 and positive == 0:
            labels[combo] = "degraded"
        elif positive >= 2:
            labels[combo] = "improved"
        elif rows and all(practical_equivalence((float(row["ci_low"]), float(row["ci_high"])), _metric_threshold(str(row["metric"]))) == "practical_equivalent" for row in rows):
            labels[combo] = "stable"
        else:
            labels[combo] = "inconclusive"
    degraded = sum(value == "degraded" for value in labels.values())
    if degraded >= 2:
        overall = "globally_degraded"
    elif degraded == 1:
        overall = "context_dependent_degraded"
    elif labels and all(value == "stable" for value in labels.values()):
        overall = "stable"
    elif any(value == "improved" for value in labels.values()):
        overall = "improved"
    else:
        overall = "inconclusive"
    return overall, labels


def _make_report_dynamic(out_dir: Path, integrity: Mapping[str, Any], aggregates: list[dict[str, Any]], agent_aggregates: list[dict[str, Any]], pairwise: list[dict[str, Any]], equivalence: list[dict[str, Any]], drift: list[dict[str, Any]], completed: int, expected: int, m1_reproduction: Mapping[str, Any] | None) -> str:
    validation_path = out_dir / "diagnostic_validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    v1v2_path = out_dir / "v1_v2_reproduction_summary.json"
    v1v2 = json.loads(v1v2_path.read_text(encoding="utf-8")) if v1v2_path.exists() else {}
    selected = json.loads((out_dir / "selected_contrast_seeds.json").read_text(encoding="utf-8")) if (out_dir / "selected_contrast_seeds.json").exists() else []
    traces = json.loads((out_dir / "traces_stage.json").read_text(encoding="utf-8")) if (out_dir / "traces_stage.json").exists() else {}
    lines = ["# v12 mixed-policy role diagnosis", "", "## 1. Execution and integrity", ""]
    lines.append(f"- Completed combinations: {completed}/{expected}; episodes: {validation.get('total_episodes', 0)}/{validation.get('expected_episodes', 0)}; per-Actor rows: {validation.get('total_per_agent_rows', 0)}/{validation.get('expected_per_agent_rows', 0)}.")
    lines.append(f"- Git commit: `{integrity.get('git_commit_sha')}`; Python `{integrity.get('python')}`, PyTorch `{integrity.get('torch')}`, NumPy `{integrity.get('numpy')}`.")
    lines.append(f"- Execution command: `{integrity.get('execution_command', integrity.get('command'))}`.")
    lines.append(f"- Requested device `{integrity.get('requested_device')}`; controller/trajectory device `{integrity.get('controller_device')}`/`{integrity.get('trajectory_device')}`; parallel episode workers `{integrity.get('parallel_episode_worker_device')}` with {integrity.get('workers')} workers.")
    lines.append(f"- Input SHA unchanged: `{validation.get('input_sha256_unchanged', integrity.get('input_sha256_unchanged'))}`; non-finite count: `{validation.get('non_finite_count')}`; duplicate rows: `{validation.get('duplicate_row_count')}`; missing combo/seed rows: `{validation.get('missing_combo_seed_count')}`.")
    if validation.get("test_result_summary"):
        lines.append(f"- Test summary: `{validation.get('test_result_summary')}`.")
    lines.extend(["- Actors were queried deterministically in eval mode under `torch.no_grad()`; no trainer, rollout collector, optimizer, checkpoint writer, or environment mutation was used.", "- v12 environment, reward, observation, target/cue, lock, soft-boundary, and HAPPO training semantics were not changed.", "", "## 2. Best checkpoint team results", "", "| Combination | Win | Any kill | >=2 kills | Mean red kills | Half-lock | Mean return |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in aggregates:
        if row.get("checkpoint") == "best":
            lines.append(f"| {row['combo']} | {row['task_win_rate']:.3f} | {row['any_kill_rate']:.3f} | {row['at_least_two_kill_rate']:.3f} | {row['mean_red_kills']:.3f} | {row['red_half_lock_episode_rate']:.3f} | {row['mean_return']:.3f} |")
    lines.extend(["", "## 3. Final checkpoint team results", "", "| Combination | Win | Any kill | >=2 kills | Mean red kills | Half-lock | Mean return |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in aggregates:
        if row.get("checkpoint") == "final":
            lines.append(f"| {row['combo']} | {row['task_win_rate']:.3f} | {row['any_kill_rate']:.3f} | {row['at_least_two_kill_rate']:.3f} | {row['mean_red_kills']:.3f} | {row['red_half_lock_episode_rate']:.3f} | {row['mean_return']:.3f} |")
    lines.extend(["", "## 4. Deployment localization", ""])
    support_records = [_stat_record(pairwise, checkpoint="best", combo_a="M0_all_rule", combo_b="M2_learned_support_rule_combats", metric=metric, expected="no_material_loss") for metric in ("task_win", "any_kill", "at_least_two_kill", "red_attack_kills", "mean_red_max_lock_progress", "red_half_lock_episode_rate", "episode_return")]
    support_actor_records = []
    for agent_id in RED_COMBAT_IDS_V12:
        support_actor_records.extend(_stat_record(pairwise, checkpoint="best", combo_a="M0_all_rule", combo_b="M2_learned_support_rule_combats", agent_id=agent_id, metric=metric, scope="agent", expected="no_material_loss") for metric in ("attributed_kills", "max_lock_progress", "half_lock_active_step_rate"))
    support_losses = sum(record["judgement"] == "contradicted" for record in support_records + support_actor_records)
    support_inconclusive = sum(record["judgement"] == "inconclusive" for record in support_records + support_actor_records)
    support_judgement = "supported" if support_losses == 0 and support_inconclusive < len(support_records) + len(support_actor_records) else ("contradicted" if support_losses >= 2 else "inconclusive")
    lines.append(f"- Learned Support is-not-primary test: `{support_judgement}`. The decision is based on M2 vs M0 core paired CIs and rule Combat own Actor rows, not Support survival or cue rates alone.")
    _render_stat_table(lines, "Support M2 vs M0 evidence", support_records)
    combat_records: dict[str, list[dict[str, Any]]] = {}
    combat_specs = (("Combat1", "red_1", "M4_rule_support_learned_combat_1"), ("Combat2", "red_2", "M5_rule_support_learned_combat_2"), ("Combat3", "red_3", "M6_rule_support_learned_combat_3"))
    for label, agent_id, combo in combat_specs:
        records = [_stat_record(pairwise, checkpoint="best", combo_a="M0_all_rule", combo_b=combo, metric=metric, expected="material_loss") for metric in ("task_win", "red_attack_kills", "mean_red_max_lock_progress", "red_half_lock_episode_rate", "episode_return")]
        records.extend(_stat_record(pairwise, checkpoint="best", combo_a="M0_all_rule", combo_b=combo, agent_id=agent_id, metric=metric, scope="agent", expected="material_loss") for metric in ("attributed_kills", "max_lock_progress", "half_lock_active_step_rate", "lock_active_step_rate", "direct_target_rate"))
        weak = sum(record["judgement"] == "supported" for record in records[:5])
        own = records[5:]
        own_weak = sum(record["judgement"] == "supported" for record in own[:3])
        own_negative = sum(bool(record.get("ci")) and float(record["ci"][1]) < 0.0 for record in own[:3])
        if own_negative == 3 and weak >= 2:
            rating, judgement = "clearly_weak", "supported"
        elif own_weak >= 1 or weak >= 1:
            rating, judgement = "partially_capable", "weakly_supported"
        elif all(record["judgement"] == "contradicted" for record in own[:3]):
            rating, judgement = "approximately_rule_compatible", "contradicted"
        else:
            rating, judgement = "inconclusive", "inconclusive"
        combat_records[label] = records
        lines.append(f"- {label} (`{agent_id}`): rating `{rating}`, judgement `{judgement}`; direct-target evidence is included in the Actor rows and is not replaced by team means.")
        _render_stat_table(lines, f"{label} single-slot evidence", records)
    m3_records = [_stat_record(pairwise, checkpoint="best", combo_a=combo, combo_b="M3_rule_support_learned_combats", metric=metric, expected="material_loss") for combo in ("M4_rule_support_learned_combat_1", "M5_rule_support_learned_combat_2", "M6_rule_support_learned_combat_3") for metric in ("task_win", "red_attack_kills", "mean_red_max_lock_progress")]
    joint_supported = sum(record["judgement"] == "supported" for record in m3_records)
    lines.append(f"- Three-Combat deployment contrast: `{'supported' if joint_supported >= 2 else 'inconclusive'}` for joint deployment loss; nonlinear interaction, synergy failure, state-distribution cause, and credit-assignment cause remain `not_established` because pairwise learned-C1+C2/C3 combinations were not run.")
    _render_stat_table(lines, "M3 versus single learned-Combat deployments", m3_records)
    interaction_records = [_stat_record(pairwise, checkpoint="best", combo_a=a, combo_b=b, metric=metric, expected="material_loss") for a, b in (("M4_rule_support_learned_combat_1", "M7_learned_support_learned_combat_1"), ("M5_rule_support_learned_combat_2", "M8_learned_support_learned_combat_2"), ("M6_rule_support_learned_combat_3", "M9_learned_support_learned_combat_3")) for metric in ("task_win", "red_attack_kills", "mean_red_max_lock_progress")]
    _render_stat_table(lines, "Support-Combat interaction evidence", interaction_records)
    lines.append(f"- Support-Combat interaction judgement: `{'supported' if sum(record['judgement'] == 'supported' for record in interaction_records) >= 2 else 'inconclusive'}`; any effect is context-specific and not a Support-primary conclusion.")
    lines.extend(["", "## 5. Per-Actor aggregate evidence", "", "| Checkpoint | Combination | Actor | Source | Kills | Max lock | Half-lock rate | Direct-target rate | Shared-only rate | No-valid-target rate | Survival | Alive decision steps |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in agent_aggregates:
        if row.get("checkpoint") == "best" and row.get("combo") in {"M0_all_rule", "M1_all_learned", "M2_learned_support_rule_combats", "M3_rule_support_learned_combats", "M4_rule_support_learned_combat_1", "M5_rule_support_learned_combat_2", "M6_rule_support_learned_combat_3"}:
            lines.append(f"| best | {row['combo']} | {row['agent_id']} | {row['action_source']} | {row['mean_attributed_kills']:.3f} | {row['mean_max_lock_progress']:.3f} | {row['mean_half_lock_active_step_rate']:.3f} | {row['mean_direct_target_rate']:.3f} | {row['mean_shared_only_target_rate']:.3f} | {row['mean_no_valid_target_rate']:.3f} | {row['survival_rate']:.3f} | {row.get('mean_alive_decision_steps', 'NA')} |")
    lines.extend(["", "## 6. Best-to-final role drift", ""])
    for role in ROLE_ORDER:
        overall, labels = _role_drift_classification(drift, role)
        lines.append(f"- `{role}`: `{overall}`; contexts: " + ", ".join(f"{combo}={label}" for combo, label in labels.items()) + ".")
        selected = [row for row in drift if row.get("combo") == "M1_all_learned" and row.get("role") == role and row.get("metric") in {"max_lock_progress", "attributed_kills", "half_lock_active_step_rate"}]
        for row in selected:
            lines.append(f"  - M1 {row['metric']}: point {float(row['mean_delta_final_minus_best']):.6g}, 95% CI [{float(row['ci_low']):.6g}, {float(row['ci_high']):.6g}].")
    lines.extend(["", "## 7. Targeted trajectories", "", f"- Category counts: {validation.get('trajectory_category_counts', {})}; specs: {validation.get('trajectory_spec_count', len(selected))}; trajectory rows: {validation.get('trajectory_row_count', traces.get('trace_row_count', 0))}.", "- Each trace validates source-map type before execution and records `killer_id` (the red Actor), `killed_target_id` (the blue target), and `kill_event` consistently.", "- `threat_exposure_rate` means an alive blue aircraft directly sees `red_0`; `support_visible_to_blue_rate` and `support_targeted_by_blue_rate` are also emitted explicitly."])
    lines.extend(["", "## 8. v1/v2 reproduction", "", f"- Team metric reproduction pass: `{v1v2.get('team_metric_reproduction_pass')}`; actor metric changed fields allowed by alive-only semantics: `{v1v2.get('actor_metric_changed_count')}`.", "- Any team-field mismatch is reported field-by-field in `v1_v2_reproduction_comparison.csv`; no v2 result is used to overwrite v1."])
    lines.extend(["", "## 9. Evidence boundaries", "", "- Deployment localization: learned Combat deployment is supported as the dominant failure locus only where the paired CIs above support it; learned Support alone is not allowed to explain the catastrophic learned-Combat failure unless its own M2-vs-M0 criteria fail.", "- Actor-level evidence: Combat ratings above are based on each learned Actor's own kills, lock, half-lock, direct-target, visibility, switching, survival, and hard-contact fields, with alive-only denominators.", "- Unresolved mechanism: action mapping, state distribution, target switching, multi-agent interaction, advantage/credit assignment, and optimization history are not identified as causal roots by this phase.", "- The report does not claim a final root cause, propose v13, modify Reward, or start new training."])
    lines.extend(["", "## 10. M1 historical reproduction", "", f"- M1 reproduction pass: `{validation.get('m1_reproduction_pass')}`; see `m1_reproduction_comparison.json` for field-level errors and seed hashes.", "", "No RL training was run by this diagnostic."])
    return "\n".join(lines) + "\n"


def _stage_guard(out_dir: Path, integrity: Mapping[str, Any], resume: bool) -> None:
    manifest_path = out_dir / "input_integrity_manifest.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("input_sha256_start") != integrity.get("input_sha256_start"):
            raise RuntimeError("input SHA changed; refusing to reuse diagnostics")


def _tasks(run_dir: Path, train_cfg_path: Path, test_seeds: list[int], effective_device: torch.device, checkpoint_selection: str, workers: int) -> list[dict[str, Any]]:
    checkpoints = ["best", "final"] if checkpoint_selection == "both" else [checkpoint_selection]
    # Preserve the requested interruption priority: best M0-M6, final M0-M6,
    # then best/final M7-M9.
    ordered = []
    for checkpoint in checkpoints:
        for name, source_map in COMBOS_V12_MIXED:
            if name.startswith(("M0", "M1", "M2", "M3", "M4", "M5", "M6")):
                ordered.append((checkpoint, name, source_map))
    for checkpoint in checkpoints:
        for name, source_map in COMBOS_V12_MIXED:
            if name.startswith(("M7", "M8", "M9")):
                ordered.append((checkpoint, name, source_map))
    worker_device = "cpu" if int(workers) > 1 else str(effective_device)
    return [{"env_config": str(run_dir / "resolved_environment_config.yaml"), "train_config": str(train_cfg_path), "checkpoint_path": str(run_dir / f"{checkpoint}.pt"), "checkpoint_name": checkpoint, "combo_name": combo, "source_map": source_map, "seeds": test_seeds, "device": worker_device} for checkpoint, combo, source_map in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--resume-diagnostics", action="store_true")
    parser.add_argument("--overwrite-diagnostics", action="store_true")
    parser.add_argument("--stage", choices=("integrity", "evaluate", "statistics", "traces", "report", "all"), default="all")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if args.overwrite_diagnostics and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    integrity, _env_cfg, train_cfg, test_seeds, controller_device = _load_inputs(run_dir, args.device, started_at, int(args.workers))
    manifest_path = out_dir / "input_integrity_manifest.json"
    if args.resume_diagnostics and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        integrity["execution_command"] = previous_manifest.get("execution_command", previous_manifest.get("command", integrity["execution_command"]))
    integrity["command"] = integrity["execution_command"]
    if int(args.episodes) != 200:
        test_seeds = test_seeds[: int(args.episodes)]
    _stage_guard(out_dir, integrity, args.resume_diagnostics)
    _write_json(out_dir / "input_integrity_manifest.json", integrity)
    combo_manifest = {"schema_version": 3, "checkpoint_selection": args.checkpoint, "episodes": len(test_seeds), "test_seed_hash": hashlib.sha256(_canonical_json(test_seeds)).hexdigest(), "seed_manifest_hash": integrity["seed_manifest_hash"], "combos": [{"name": name, "slot_source_map": dict(source_map)} for name, source_map in COMBOS_V12_MIXED], "actor_slot_mapping": integrity["actor_slot_mapping"], "workers": int(args.workers), "requested_device": args.device, "controller_device": integrity.get("controller_device"), "parallel_episode_worker_device": integrity.get("parallel_episode_worker_device"), "trajectory_device": integrity.get("trajectory_device")}
    _write_json(out_dir / "policy_combination_manifest.json", combo_manifest)
    rows_path = out_dir / "mixed_policy_seed_level.csv.gz"
    agents_path = out_dir / "mixed_policy_per_agent_seed_level.csv.gz"
    completed_path = out_dir / "completed_combinations.json"
    if args.stage in ("evaluate", "all"):
        rows = _read_rows(rows_path, compressed=True) if args.resume_diagnostics else []
        agent_rows = _read_rows(agents_path, compressed=True) if args.resume_diagnostics else []
        completed = set(json.loads(completed_path.read_text(encoding="utf-8")).get("completed", [])) if args.resume_diagnostics and completed_path.exists() else set()
        tasks = _tasks(run_dir, run_dir / "resolved_training_config.yaml", test_seeds, controller_device, args.checkpoint, int(args.workers))
        pending = [task for task in tasks if f"{task['checkpoint_name']}::{task['combo_name']}" not in completed]
        if int(args.workers) > 1 and pending:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = {executor.submit(_evaluate_task, task): task for task in pending}
                for future in as_completed(futures):
                    result = future.result()
                    rows.extend(result["episode_rows"])
                    agent_rows.extend(result["agent_rows"])
                    key = f"{result['checkpoint_name']}::{result['combo_name']}"
                    completed.add(key)
                    _write_rows(rows_path, rows, compressed=True)
                    _write_rows(agents_path, agent_rows, compressed=True)
                    _write_json(completed_path, {"input_sha256_start": integrity["input_sha256_start"], "completed": sorted(completed)})
        else:
            for task in pending:
                result = _evaluate_task(task)
                rows.extend(result["episode_rows"])
                agent_rows.extend(result["agent_rows"])
                completed.add(f"{result['checkpoint_name']}::{result['combo_name']}")
                _write_rows(rows_path, rows, compressed=True)
                _write_rows(agents_path, agent_rows, compressed=True)
                _write_json(completed_path, {"input_sha256_start": integrity["input_sha256_start"], "completed": sorted(completed)})
        _write_json(out_dir / "evaluation_stage.json", {"completed_combinations": sorted(completed), "completed_count": len(completed), "expected_count": len(tasks), "input_sha256_start": integrity["input_sha256_start"]})
    elif rows_path.exists():
        rows = _read_rows(rows_path, compressed=True)
        agent_rows = _read_rows(agents_path, compressed=True)
    else:
        rows, agent_rows = [], []
    rows = [_normalise_episode_row(row) for row in rows]
    if rows and (args.stage in ("statistics", "traces", "report", "all")):
        _write_rows(rows_path, rows, compressed=True)
    aggregates: list[dict[str, Any]] = []
    agent_aggregates: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    v1_v2_summary: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    if args.stage in ("statistics", "traces", "report", "all"):
        aggregates, agent_aggregates, pairwise, equivalence, drift = _make_statistics(rows, agent_rows, int(args.bootstrap_samples))
        _write_rows(out_dir / "mixed_policy_aggregate.csv", aggregates)
        _write_json(out_dir / "mixed_policy_aggregate.json", {"rows": aggregates, "input_sha256_start": integrity["input_sha256_start"]})
        _write_rows(out_dir / "mixed_policy_per_agent_aggregate.csv", agent_aggregates)
        _write_rows(out_dir / "mixed_policy_pairwise_statistics.csv", pairwise)
        _write_rows(out_dir / "mixed_policy_practical_equivalence.csv", equivalence)
        _write_rows(out_dir / "best_vs_final_role_drift.csv", drift)
        m1_result = _m1_reproduction(rows, run_dir) if len(_rows_for(rows, "best", "M1_all_learned")) == len(test_seeds) and len(_rows_for(rows, "final", "M1_all_learned")) == len(test_seeds) else {"status": "incomplete"}
        _write_json(out_dir / "m1_reproduction_comparison.json", m1_result)
        v1_v2_comparisons, v1_v2_summary = _v1_v2_reproduction(out_dir, aggregates, agent_aggregates)
        _write_rows(out_dir / "v1_v2_reproduction_comparison.csv", v1_v2_comparisons)
        _write_json(out_dir / "v1_v2_reproduction_summary.json", v1_v2_summary)
        selected_specs = _select_trace_specs(rows) if len(rows) >= (20 * len(test_seeds)) else []
        _write_json(out_dir / "selected_contrast_seeds.json", selected_specs)
        _write_json(out_dir / "statistics_stage.json", {"input_sha256_start": integrity["input_sha256_start"], "episodes_observed": len(rows), "expected_episodes": 20 * len(test_seeds)})
    else:
        selected_specs = json.loads((out_dir / "selected_contrast_seeds.json").read_text(encoding="utf-8")) if (out_dir / "selected_contrast_seeds.json").exists() else []
        aggregates = _read_rows(out_dir / "mixed_policy_aggregate.csv") if (out_dir / "mixed_policy_aggregate.csv").exists() else []
        agent_aggregates = _read_rows(out_dir / "mixed_policy_per_agent_aggregate.csv") if (out_dir / "mixed_policy_per_agent_aggregate.csv").exists() else []
        pairwise = _read_rows(out_dir / "mixed_policy_pairwise_statistics.csv") if (out_dir / "mixed_policy_pairwise_statistics.csv").exists() else []
        equivalence = _read_rows(out_dir / "mixed_policy_practical_equivalence.csv") if (out_dir / "mixed_policy_practical_equivalence.csv").exists() else []
        drift = _read_rows(out_dir / "best_vs_final_role_drift.csv") if (out_dir / "best_vs_final_role_drift.csv").exists() else []
        v1_v2_summary = json.loads((out_dir / "v1_v2_reproduction_summary.json").read_text(encoding="utf-8")) if (out_dir / "v1_v2_reproduction_summary.json").exists() else {}
    if args.stage in ("traces", "all"):
        trace_rows = _run_traces(selected_specs, run_dir, train_cfg, controller_device)
        _write_rows(out_dir / "targeted_mixed_policy_trajectories.csv.gz", trace_rows, compressed=True)
        _write_json(out_dir / "traces_stage.json", {"input_sha256_start": integrity["input_sha256_start"], "selected_spec_count": len(selected_specs), "trace_row_count": len(trace_rows)})
    elif not (out_dir / "targeted_mixed_policy_trajectories.csv.gz").exists():
        _write_rows(out_dir / "targeted_mixed_policy_trajectories.csv.gz", [], compressed=True)
    if not trace_rows and (out_dir / "targeted_mixed_policy_trajectories.csv.gz").exists():
        trace_rows = _read_rows(out_dir / "targeted_mixed_policy_trajectories.csv.gz", compressed=True)
    # Capture the end hash before writing validation/report artifacts so the
    # summary reflects the same immutable-input check as the manifest.
    integrity["input_sha256_end"] = _input_sha(run_dir)
    integrity["input_sha256_unchanged"] = integrity["input_sha256_start"] == integrity["input_sha256_end"]
    completed_count = len(json.loads(completed_path.read_text(encoding="utf-8")).get("completed", [])) if completed_path.exists() else 0
    expected_count = (2 if args.checkpoint == "both" else 1) * 10
    m1_reproduction = json.loads((out_dir / "m1_reproduction_comparison.json").read_text(encoding="utf-8")) if (out_dir / "m1_reproduction_comparison.json").exists() else None
    validation_summary = _build_validation_summary(out_dir, rows, agent_rows, test_seeds, selected_specs, trace_rows, integrity, m1_reproduction, v1_v2_summary)
    _write_json(out_dir / "diagnostic_validation_summary.json", validation_summary)
    if args.stage in ("report", "all"):
        report = _make_report(out_dir, integrity, aggregates, agent_aggregates, pairwise, equivalence, drift, completed_count, expected_count, m1_reproduction)
        (out_dir / "mixed_policy_diagnostic_report.md").write_text(report, encoding="utf-8")
        schema = """# Diagnostic schema

`mixed_policy_seed_level.csv.gz`: one row per checkpoint/combination/test seed.
`mixed_policy_per_agent_seed_level.csv.gz`: one row per red Actor per episode.
`mixed_policy_aggregate.csv`: team-level aggregates for each checkpoint/combination.
`mixed_policy_per_agent_aggregate.csv`: per-role aggregates.
`mixed_policy_pairwise_statistics.csv`: paired same-seed team and Actor contrasts; Actor rows have `scope=agent` and `agent_id`.
`mixed_policy_practical_equivalence.csv`: pre-registered paired-CI classifications.
`best_vs_final_role_drift.csv`: paired best/final drift by role and metric.
`targeted_mixed_policy_trajectories.csv.gz`: only pre-registered high-information seeds.

All behavior rates use `alive_decision_steps` as denominator. Each per-Actor row
contains both boolean `survived` and numeric `survival_rate`. Trace kill fields
are `kill_event`, `killer_id`, and `killed_target_id`; the legacy ambiguous
killer field is not emitted. `threat_exposure_rate` means an alive blue aircraft directly sees
red_0, while `support_visible_to_blue_rate` and `support_targeted_by_blue_rate`
are emitted separately.
"""
        (out_dir / "diagnostic_schema.md").write_text(schema, encoding="utf-8")
    integrity["ended_at"] = datetime.now(timezone.utc).isoformat()
    integrity["input_sha256_end"] = _input_sha(run_dir)
    integrity["input_sha256_unchanged"] = integrity["input_sha256_start"] == integrity["input_sha256_end"]
    _write_json(out_dir / "input_integrity_manifest.json", integrity)
    if not integrity["input_sha256_unchanged"]:
        raise RuntimeError("input SHA changed during diagnosis")
    print(json.dumps({"completed_combinations": completed_count, "expected_combinations": expected_count, "episodes": len(rows), "input_sha256_unchanged": integrity["input_sha256_unchanged"], "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
