"""Train CF-HAPPO with the shared vanilla HAPPO entry point."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.train_happo import main


if __name__ == "__main__":
    main(actor_variant="vanilla", critic_variant="mlp", method_variant="cf_happo")
