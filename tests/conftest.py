from pathlib import Path
import pytest
from uav_combat.config import load_config, aircraft_spec


@pytest.fixture
def spec():
    return aircraft_spec(load_config(Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"))

