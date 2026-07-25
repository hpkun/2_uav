"""参数共享 MAPPO 环境可学习性验证基线。"""

from .buffer import MAPPOBuffer
from .networks import CentralizedCritic, GaussianActor, SharedActor
from .trainer import MAPPOTrainer, evaluate_competitive_match, evaluate_competitive_match_parallel, evaluate_matchup
from .vector_env import (
    LocalCombatVectorEnv,
    SubprocessCombatVectorEnv,
    make_combat_vector_env,
)

__all__ = [
    "MAPPOBuffer",
    "GaussianActor",
    "SharedActor",
    "CentralizedCritic",
    "MAPPOTrainer",
    "evaluate_competitive_match",
    "evaluate_competitive_match_parallel",
    "evaluate_matchup",
    "LocalCombatVectorEnv",
    "SubprocessCombatVectorEnv",
    "make_combat_vector_env",
]
