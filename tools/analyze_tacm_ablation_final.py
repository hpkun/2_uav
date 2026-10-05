"""Aggregate existing formal TACM ablation evaluations without running an environment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tacm_ablation_protocol import (
    EVALUATION_ACTION_SEED_START, EVALUATION_ENV_SEED_START, EVALUATION_EPISODES,
    MANIFEST_DIR, METHODS, PAPER_SEEDS, TOTAL_STEPS,
)


METRICS = {
    "win_rate": "red_win_rate", "blue_win_rate": "blue_win_rate",
    "draw_rate": "draw_rate", "mean_return": "mean_episode_return",
    "mean_red_kills": "mean_red_attack_kills", "mean_blue_kills": "mean_blue_attack_kills",
    "MAV_survival": "MAV_survival_rate", "mean_UAV_survivors": "mean_UAV_survivors",
    "mean_episode_length": "mean_episode_length",
}
STABILITY_METRICS = {
    "win_rate": "red_win_rate", "return": "mean_episode_return",
    "red_kills": "mean_red_attack_kills",
}
MILESTONES = (1_600_000, 1_700_000, 1_800_000, 1_900_000, 2_000_000)
MAX_MILESTONE_ERROR = 60_000
TRAINING_REQUIRED_FIELDS = (
    "sampled_steps", "red_win_rate", "mean_episode_return", "mean_red_attack_kills",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.mean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data), "min": min(data), "max": max(data),
    }


def validate_and_match_training_milestones(
    rows: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any]]]:
    if not rows:
        raise RuntimeError("training.csv is empty")
    missing = [field for field in TRAINING_REQUIRED_FIELDS if field not in rows[0]]
    if missing:
        raise RuntimeError(f"training.csv missing required fields: {missing}")
    parsed_steps = []
    for index, row in enumerate(rows):
        try:
            step = int(row["sampled_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"training.csv sampled_steps is not an integer at row {index + 2}") from error
        for field in TRAINING_REQUIRED_FIELDS[1:]:
            try:
                float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"training.csv field {field!r} is invalid at row {index + 2}") from error
        parsed_steps.append(step)
    if any(right <= left for left, right in zip(parsed_steps, parsed_steps[1:])):
        raise RuntimeError("training.csv sampled_steps must be strictly increasing without duplicates")
    matches = []
    used_indices: set[int] = set()
    for milestone in MILESTONES:
        index = min(range(len(rows)), key=lambda position: abs(parsed_steps[position] - milestone))
        if index in used_indices:
            raise RuntimeError("late-training milestones do not map to five distinct log records")
        error = parsed_steps[index] - milestone
        if abs(error) > MAX_MILESTONE_ERROR:
            raise RuntimeError(
                f"late-training milestone {milestone} nearest record error {error} exceeds "
                f"{MAX_MILESTONE_ERROR} sampled steps"
            )
        used_indices.add(index)
        matches.append((milestone, error, rows[index]))
    return matches


def load_runs(manifest_dir: Path = MANIFEST_DIR) -> list[dict[str, Any]]:
    runs = []
    for method in METHODS:
        path = manifest_dir / f"{method}_latest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "complete" or tuple(data.get("seeds", ())) != PAPER_SEEDS:
            raise RuntimeError(f"incomplete training manifest: {path}")
        for run in data["runs"]:
            runs.append({**run, "method": method, "paper_variant": METHODS[method]["paper_variant"]})
    return runs


def analyze(manifest_dir: Path, output_dir: Path) -> dict[str, Any]:
    # Shell launchers may pre-create the destination for logging. Accept an
    # empty directory, but never overwrite existing audit artifacts.
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty audit output: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    per_seed: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for run in load_runs(manifest_dir):
        run_dir = Path(run["output_folder"])
        evaluation_path = run_dir / "evaluation_2000000_stochastic_summary.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        result = evaluation["results"][0]
        contract = (
            int(result["sampled_steps"]) == TOTAL_STEPS
            and int(result["evaluation_episodes"]) == EVALUATION_EPISODES
            and int(result["evaluation_environment_seed_start"]) == EVALUATION_ENV_SEED_START
            and int(result["effective_action_seed"]) == EVALUATION_ACTION_SEED_START
            and result["action_mode"] == "stochastic"
            and result["evaluation_profile"] == "main"
        )
        if not contract:
            raise RuntimeError(f"formal evaluation contract mismatch: {evaluation_path}")
        row = {
            "method": run["method"], "paper_variant": run["paper_variant"],
            "training_seed": int(run["seed"]), "exact_steps": int(result["sampled_steps"]),
            **{label: float(result[field]) for label, field in METRICS.items()},
            "evaluation_episodes": int(result["evaluation_episodes"]),
            "environment_seed_start": EVALUATION_ENV_SEED_START,
            "environment_seed_end": EVALUATION_ENV_SEED_START + EVALUATION_EPISODES - 1,
            "action_seed_start": EVALUATION_ACTION_SEED_START,
            "action_seed_end": EVALUATION_ACTION_SEED_START + EVALUATION_EPISODES - 1,
            "action_mode": result["action_mode"], "environment_version": result["environment_version"],
        }
        per_seed.append(row)

        with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
            training = list(csv.DictReader(stream))
        matches = validate_and_match_training_milestones(training)
        chosen = [candidate for _, _, candidate in matches]
        stability_row: dict[str, Any] = {
            "method": run["method"], "paper_variant": run["paper_variant"],
            "training_seed": int(run["seed"]),
        }
        for milestone, error, candidate in matches:
            stability_row[f"milestone_{milestone}_actual_steps"] = int(candidate["sampled_steps"])
            stability_row[f"milestone_{milestone}_step_error"] = error
        for label, field in STABILITY_METRICS.items():
            values = [float(item[field]) for item in chosen]
            stability_row.update({
                f"{label}_late_mean": statistics.mean(values),
                f"{label}_late_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                f"{label}_late_min": min(values), f"{label}_late_max": max(values),
                f"{label}_late_range": max(values) - min(values),
            })
        stability.append(stability_row)

    summary_rows = []
    summary_json: dict[str, Any] = {}
    for method in METHODS:
        current = [row for row in per_seed if row["method"] == method]
        if tuple(sorted(int(row["training_seed"]) for row in current)) != PAPER_SEEDS:
            raise RuntimeError(f"missing fixed training seed for {method}")
        summary_row: dict[str, Any] = {"method": method, "paper_variant": METHODS[method]["paper_variant"], "training_seeds": "17;23;31"}
        method_json = {}
        for metric in ("win_rate", "mean_return", "mean_red_kills", "draw_rate",
                       "MAV_survival", "mean_UAV_survivors", "mean_episode_length"):
            stats = _stats(row[metric] for row in current)
            summary_row[f"{metric}_mean"] = stats["mean"]
            summary_row[f"{metric}_std"] = stats["std"]
            method_json[metric] = stats
        summary_rows.append(summary_row); summary_json[method] = method_json

    _write_csv(output_dir / "final_ablation_per_seed.csv", per_seed)
    _write_csv(output_dir / "final_ablation_summary.csv", summary_rows)
    _write_csv(output_dir / "final_training_stability.csv", stability)
    payload = {
        "protocol": "tacm_final_paper_ablation_v1",
        "training_seed_std_semantics": "sample standard deviation across fixed training seeds",
        "late_training_stability_semantics": "within-seed dispersion at nearest logged 1.6/1.7/1.8/1.9/2.0M records",
        "methods": summary_json,
    }
    (output_dir / "final_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.manifest_dir.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "complete", "methods": list(payload["methods"]),
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
