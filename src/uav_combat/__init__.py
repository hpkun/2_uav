"""MAV/UAV heterogeneous multi-agent air-combat research package."""

from .models import Aircraft, AircraftSpec, AircraftState, OverloadCommand
from .mavuav import (
    BlueSpec, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, HeterogeneousMAVUAVAirCombatEnv,
    MAVSpec, OBS_DIM, UAVSpec,
)
from .vector_env import MAVUAVVectorEnv
from .happo import HAPPOTrainer
from .mappo import MAPPOTrainer

__all__ = [
    "Aircraft", "AircraftSpec", "AircraftState", "OverloadCommand",
    "MAVSpec", "UAVSpec", "BlueSpec", "HeterogeneousMAVUAVAirCombatEnv",
    "ENVIRONMENT_VERSION", "OBS_DIM", "GLOBAL_STATE_DIM",
    "MAVUAVVectorEnv", "HAPPOTrainer", "MAPPOTrainer",
]
