"""Run one frozen TACM paper-ablation method for seeds 17/23/31."""
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

from tools.preflight_tacm_ablation import validate_protocol
from tools.tacm_ablation_protocol import (
    MANIFEST_DIR, METHODS, PAPER_SEEDS, TOTAL_STEPS, training_command,
)


def build_plan(method: str, timestamp: str, python: str = sys.executable) -> list[dict[str, object]]:
    plan = []
    for seed in PAPER_SEEDS:
        output_name = f"paper_{method}_seed{seed}_2m_{timestamp}"
        command = training_command(method, seed, output_name, python=python)
        run_dir = PROJECT_ROOT / "outputs" / output_name
        plan.append({
            "method": method, "paper_variant": METHODS[method]["paper_variant"],
            "seed": seed, "entrypoint": str(METHODS[method]["entrypoint"]),
            "config": str(METHODS[method]["config"]),
            "environment_version": "heterogeneous_mavuav_4v4_v3_10",
            "total_steps": TOTAL_STEPS, "output_folder": str(run_dir),
            "log_file": str(run_dir / "run.log"),
            "exact_2m_checkpoint": str(run_dir / "checkpoint_2000000.pt"),
            "command": command,
        })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    preflight = validate_protocol(inspect_structures=not args.dry_run)
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, ensure_ascii=False, default=list))
        raise SystemExit("ablation preflight failed")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(args.method, timestamp)
    manifest = {
        "protocol": "tacm_final_paper_ablation_v1", "status": "dry_run" if args.dry_run else "running",
        "method": args.method, "paper_variant": METHODS[args.method]["paper_variant"],
        "timestamp": timestamp, "seeds": list(PAPER_SEEDS), "runs": plan,
    }
    if args.dry_run:
        printable = dict(manifest)
        printable["runs"] = [{**row, "command": shlex.join(row["command"])} for row in plan]
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("formal TACM ablation training requires CUDA; CPU fallback is forbidden")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{args.method}_{timestamp}.json"
    for row in plan:
        run_dir = Path(str(row["output_folder"]))
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse ablation run directory: {run_dir}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        for row in plan:
            subprocess.run(row["command"], cwd=PROJECT_ROOT, check=True)
            checkpoint = Path(str(row["exact_2m_checkpoint"]))
            if not checkpoint.is_file():
                raise RuntimeError(f"training did not create exact-2M checkpoint: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload.get("sampled_steps", -1)) != TOTAL_STEPS:
                raise RuntimeError(f"checkpoint is not exact 2M: {checkpoint}")
            row["status"] = "complete"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except BaseException:
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest = MANIFEST_DIR / f"{args.method}_latest.json"
    latest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
