"""Rule-vs-rule reachability check for functional heterogeneous 4v3 v9."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from uav_combat.happo.evaluation_4v3 import evaluate_rule_vs_rule_4v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    summary = evaluate_rule_vs_rule_4v3(args.env_config, episodes=args.episodes, seed=args.seed, workers=args.workers)
    records = summary.pop("episode_records", [])
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rule_evaluation.json").write_text(text, encoding="utf-8")
        fields = sorted({key for record in records for key in record})
        if "episode_seed" in fields:
            fields.remove("episode_seed")
            fields.insert(0, "episode_seed")
        with (out / "rule_per_episode.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in record.items()
                })


if __name__ == "__main__":
    main()
