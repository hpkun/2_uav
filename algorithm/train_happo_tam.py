"""Training entry point for TAM-HAPPO on the frozen v3.9 environment."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.train_happo import main


if __name__ == "__main__":
    main(actor_variant="tam", critic_variant="tam_attention", method_variant="baseline")
