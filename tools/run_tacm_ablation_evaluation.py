"""Evaluate exact-2M TACM ablation checkpoints under one fixed protocol."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tacm_ablation_protocol import (
    MANIFEST_DIR, METHODS, PAPER_SEEDS, TOTAL_STEPS, evaluation_command,
)


def load_training_manifests(method: str) -> list[dict]:
    methods = tuple(METHODS) if method == "all" else (method,)
    manifests = []
    for name in methods:
        path = MANIFEST_DIR / f"{name}_latest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed training manifest: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "complete" or tuple(data.get("seeds", ())) != PAPER_SEEDS:
            raise RuntimeError(f"training manifest is not a complete 17/23/31 contract: {path}")
        manifests.append(data)
    return manifests


def build_plan(method: str, manifests: list[dict], python: str = sys.executable) -> list[dict]:
    plan = []
    for manifest in manifests:
        for run in manifest["runs"]:
            checkpoint = Path(run["exact_2m_checkpoint"])
            plan.append({
                "method": manifest["method"], "paper_variant": manifest["paper_variant"],
                "training_seed": int(run["seed"]), "checkpoint": str(checkpoint),
                "command": evaluation_command(checkpoint, python=python),
                "evaluation_csv": str(checkpoint.parent / "evaluation_2000000_stochastic.csv"),
                "evaluation_summary": str(checkpoint.parent / "evaluation_2000000_stochastic_summary.json"),
            })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        methods = tuple(METHODS) if args.method == "all" else (args.method,)
        plan = []
        for method in methods:
            for seed in PAPER_SEEDS:
                checkpoint = Path(f"<exact-2m-{method}-seed{seed}>") / "checkpoint_2000000.pt"
                plan.append({
                    "method": method, "paper_variant": METHODS[method]["paper_variant"],
                    "training_seed": seed, "checkpoint": str(checkpoint),
                    "command": evaluation_command(checkpoint, python=sys.executable),
                })
        print(json.dumps({"method": args.method, "runs": [
            {**row, "command": shlex.join(row["command"])} for row in plan
        ]}, indent=2, ensure_ascii=False))
        return
    manifests = load_training_manifests(args.method)
    plan = build_plan(args.method, manifests)
    if not torch.cuda.is_available():
        raise RuntimeError("formal TACM ablation evaluation requires CUDA; CPU fallback is forbidden")
    for row in plan:
        checkpoint = Path(row["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("sampled_steps", -1)) != TOTAL_STEPS:
            raise RuntimeError(f"formal evaluation requires exact-2M checkpoint: {checkpoint}")
        for output in (row["evaluation_csv"], row["evaluation_summary"]):
            if Path(output).exists():
                raise FileExistsError(f"refusing to overwrite formal evaluation: {output}")
        subprocess.run(row["command"], cwd=PROJECT_ROOT, check=True)
        row["status"] = "complete"
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MANIFEST_DIR / f"evaluation_{args.method}_{timestamp}.json"
    path.write_text(json.dumps({"status": "complete", "runs": plan}, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
