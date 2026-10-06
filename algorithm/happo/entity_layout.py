"""Named layout of v3.11 observations/state, matching env.mavuav builders.

All entity offsets and field indexes are centralized here; no global state is
ever accepted by the actor parser.
"""
from dataclasses import dataclass
import torch
from env.mavuav import OBS_DIM, GLOBAL_STATE_DIM, RED_IDS, BLUE_IDS

SELF_FIELDS = ("x", "y", "altitude", "speed", "theta", "psi", "alive",
               "type_mav", "type_uav", "type_blue", "time")
FRIEND_FIELDS = ("dx", "dy", "dz", "distance", "dvx", "dvy", "dvz", "alive",
                 "type_mav", "type_uav", "type_blue")
ENEMY_FIELDS = ("dx", "dy", "dz", "distance", "dvx", "dvy", "dvz", "ata", "aa",
                "alive", "direct", "datalink", "streak", "killed")
AIRCRAFT_FIELDS = SELF_FIELDS[:-1]
SELF_DIM, FRIEND_DIM, ENEMY_DIM = map(len, (SELF_FIELDS, FRIEND_FIELDS, ENEMY_FIELDS))
AIRCRAFT_DIM = len(AIRCRAFT_FIELDS)
FRIEND_COUNT, ENEMY_COUNT = len(RED_IDS) - 1, len(BLUE_IDS)
AIRCRAFT_COUNT = len(RED_IDS) + len(BLUE_IDS)
CONTEXT_DIM = GLOBAL_STATE_DIM - AIRCRAFT_COUNT * AIRCRAFT_DIM
assert SELF_DIM + FRIEND_COUNT * FRIEND_DIM + ENEMY_COUNT * ENEMY_DIM == OBS_DIM
assert CONTEXT_DIM == 2 * len(RED_IDS) * len(BLUE_IDS) + len(BLUE_IDS) + 1


@dataclass
class ObservationEntities:
    self_features: torch.Tensor
    friends: torch.Tensor
    enemies: torch.Tensor

    @property
    def alive(self):
        return self.self_features[..., SELF_FIELDS.index("alive")] > 0.5

    @property
    def friend_valid(self):
        return self.friends[..., FRIEND_FIELDS.index("alive")] > 0.5

    @property
    def enemy_direct(self):
        return self.enemies[..., ENEMY_FIELDS.index("direct")] > 0.5

    @property
    def enemy_datalink(self):
        return self.enemies[..., ENEMY_FIELDS.index("datalink")] > 0.5

    @property
    def enemy_valid(self):
        return (self.enemies[..., ENEMY_FIELDS.index("alive")] > 0.5) & (self.enemy_direct | self.enemy_datalink)


def parse_observation(observations: torch.Tensor) -> ObservationEntities:
    if observations.shape[-1] != OBS_DIM:
        raise ValueError(f"ERAM observation must have {OBS_DIM} features")
    own, friends, enemies = observations.split(
        (SELF_DIM, FRIEND_COUNT * FRIEND_DIM, ENEMY_COUNT * ENEMY_DIM), dim=-1,
    )
    return ObservationEntities(own, friends.unflatten(-1, (FRIEND_COUNT, FRIEND_DIM)),
                               enemies.unflatten(-1, (ENEMY_COUNT, ENEMY_DIM)))


def parse_global_state(states: torch.Tensor):
    if states.shape[-1] != GLOBAL_STATE_DIM:
        raise ValueError(f"ERAM global state must have {GLOBAL_STATE_DIM} features")
    entities, context = states.split((AIRCRAFT_COUNT * AIRCRAFT_DIM, CONTEXT_DIM), dim=-1)
    entities = entities.unflatten(-1, (AIRCRAFT_COUNT, AIRCRAFT_DIM))
    valid = entities[..., AIRCRAFT_FIELDS.index("alive")] > 0.5
    return entities, context, valid
