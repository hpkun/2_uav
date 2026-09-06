"""Deterministically evaluate an RC-HAPPO checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.evaluate_happo import main


if __name__ == "__main__":
    main(expected_critic_variant="relational")
