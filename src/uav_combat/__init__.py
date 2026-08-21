"""MAV/UAV heterogeneous multi-agent air-combat research package."""

from .models import Aircraft, AircraftSpec, AircraftState, OverloadCommand
from .mavuav import BlueSpec, HeterogeneousMAVUAVAirCombatEnv, MAVSpec, UAVSpec
from .vector_env import MAVUAVVectorEnv
from .happo import HAPPOTrainer
from .mappo import MAPPOTrainer

__all__ = [
    "Aircraft", "AircraftSpec", "AircraftState", "OverloadCommand",
    "MAVSpec", "UAVSpec", "BlueSpec", "HeterogeneousMAVUAVAirCombatEnv",
    "MAVUAVVectorEnv", "HAPPOTrainer", "MAPPOTrainer",
]
