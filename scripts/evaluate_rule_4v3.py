"""Rule-vs-rule reachability check for functional heterogeneous 4v3 v9."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_combat.happo.evaluation_4v3 import evaluate_rule_vs_rule_4v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/heterogeneous_4v3_main_v9.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = evaluate_rule_vs_rule_4v3(args.env_config, episodes=args.episodes, seed=args.seed)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
