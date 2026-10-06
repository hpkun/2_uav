"""Independent ERAM-HAPPO entry; default v3.11 and baseline screening protocol."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import algorithm.train_happo as training


def main():
    training.DEFAULT_CONFIG = PROJECT_ROOT / "configs/happo_eram_v311.yaml"
    training.DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs/env_v311.yaml"
    training.main(actor_variant="entity_recurrent", critic_variant="entity_attention_recurrent",
                  method_variant="baseline")


if __name__ == "__main__":
    main()
