"""Evaluate a v9 HAPPO checkpoint on the fixed selection or test seed split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from uav_combat.happo.evaluation_4v3 import (
    build_evaluation_seed_manifest,
    evaluate_happo_fixed_blue_4v3,
    evaluation_seeds_from_manifest,
)
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    fields: set[str] = set()
    for record in records:
        row = {}
        for key, value in record.items():
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            fields.add(key)
        rows.append(row)
    ordered = ["episode_seed"] + sorted(fields - {"episode_seed"})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config")
    parser.add_argument("--train-config")
    parser.add_argument("--seed-manifest")
    parser.add_argument("--split", choices=("selection", "test"), default="selection")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="evaluation_4v3")
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8")) if args.train_config else deepcopy(checkpoint["config"])
    if args.device is not None:
        cfg["experiment"]["device"] = args.device
    env_config = args.env_config or checkpoint.get("env_config", "configs/heterogeneous_4v3_main_v9.yaml")
    if args.seed_manifest:
        manifest = json.loads(Path(args.seed_manifest).read_text(encoding="utf-8"))
    elif checkpoint.get("seed_manifest"):
        manifest = deepcopy(checkpoint["seed_manifest"])
    else:
        evaluation = cfg["evaluation"]
        manifest = build_evaluation_seed_manifest(
            int(cfg["experiment"]["seed"]),
            selection_episodes=int(evaluation["selection_episodes"]),
            test_episodes=int(evaluation["test_episodes"]),
            selection_seed_offset=int(evaluation["selection_seed_offset"]),
            test_seed_offset=int(evaluation["test_seed_offset"]),
        )
    seeds = evaluation_seeds_from_manifest(manifest, args.split, args.episodes)
    trainer = HAPPO4v3Trainer(env_config, cfg)
    try:
        trainer.load_checkpoint(checkpoint_path)
        summary = evaluate_happo_fixed_blue_4v3(
            trainer.actors,
            env_config,
            seeds=seeds,
            num_envs=int(cfg["training"]["num_envs"]),
            num_env_workers=int(cfg["training"].get("num_env_workers", 0)),
            device=trainer.device,
            split=args.split,
            seed_manifest=manifest,
        )
    finally:
        trainer.close()
    records = summary.pop("episode_records", [])
    summary.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_family": checkpoint.get("checkpoint_family"),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "checkpoint_actual_env_steps": checkpoint.get("actual_env_steps", checkpoint.get("env_steps")),
        "seed_list": seeds,
        "episode_record_count": len(records),
    })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_csv(out / "per_episode.csv", records)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
