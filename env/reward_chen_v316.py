"""Chen-aligned shared-objective adaptation, not a complete paper reproduction.

ENV_MAPPING / SCALE_NORMALIZATION: UAV /35 and fixed process /4.
MAV contribution is REMOVED_FROM_TRAINING_OBJECTIVE (diagnostic only).
"""
from copy import deepcopy
from .reward_chen_v315 import CHEN_CONFIG

VERSION = "heterogeneous_mavuav_4v4_v3_16"
MODE = "chen_shared_event_dominant_v1"
METADATA = {
    "reward_wiring": "shared_event_plus_normalized_role_process",
    "uav_process_normalizer": 35.0,
    "shared_process_denominator": 4,
    "mav_team_contribution_training": False,
    "terminal_reward_enabled": False,
}
CONFIG = deepcopy(CHEN_CONFIG)
CONFIG.update(mode=MODE, **METADATA)
EPISODE_FIELDS = (
    "shared_process_reward_sum", "team_reward_sum", "raw_uav_process_sum",
    "normalized_uav_process_sum", "mav_q_process_sum", "uav1_q_process_sum",
    "uav2_q_process_sum", "uav3_q_process_sum", "uav1_raw_process_sum",
    "uav2_raw_process_sum", "uav3_raw_process_sum", "event_blue_kill_reward_sum",
    "event_uav_combat_loss_reward_sum", "event_uav_boundary_loss_reward_sum",
    "event_mav_death_reward_sum", "mav_team_contribution_diagnostic",
)


def validate_metadata(payload):
    for key, value in METADATA.items():
        if key not in payload or payload[key] != value:
            raise RuntimeError(f"incompatible v3.16 reward contract: {key}")


def shared_reward(dense, death_causes, seen_blue_kills, config):
    """Consume authoritative new death events, never guessed/co-attacker rewards.

    Dense is the existing post-boundary/pre-combat Chen snapshot. Dead/no-target
    UAV components are already zero. No clipping or situation-score addition.
    """
    from .mavuav import BLUE_IDS, RED_IDS
    new_kills = {bid for bid, cause in death_causes.items()
                 if bid in BLUE_IDS and cause == "red_attack"} - seen_blue_kills
    seen_blue_kills.update(new_kills)
    uc, mc = config["uav"], config["mav"]
    info = dict(dense)
    info["mav_q_process"] = dense["mav_r_safety"]
    for aid in RED_IDS[1:]:
        p = aid.lower()
        raw = (uc["speed_weight"] * dense[f"{p}_r_speed"]
               + uc["angle_weight"] * dense[f"{p}_r_angle"]
               + uc["distance_weight"] * dense[f"{p}_r_distance"])
        info[f"{p}_raw_process"] = float(raw)
        info[f"{p}_q_process"] = float(raw / config["uav_process_normalizer"])
    info.update(
        event_blue_kill_reward=float(uc["kill_reward"] * len(new_kills)),
        event_uav_combat_loss_reward=float(uc["combat_loss"] * sum(
            death_causes.get(aid) == "blue_attack" for aid in RED_IDS[1:])),
        event_uav_boundary_loss_reward=float(uc["boundary_loss"] * sum(
            death_causes.get(aid) == "boundary" for aid in RED_IDS[1:])),
        event_mav_death_reward=float(-mc["death_penalty"] * int("MAV" in death_causes)),
    )
    event = sum(info[key] for key in ("event_blue_kill_reward",
                "event_uav_combat_loss_reward", "event_uav_boundary_loss_reward", "event_mav_death_reward"))
    process = sum(info[f"{aid.lower()}_q_process"] for aid in RED_IDS) / config["shared_process_denominator"]
    info.update(shared_event_reward=float(event), shared_process_reward=float(process),
                shared_reward=float(event + process))
    return float(event + process), info, len(new_kills)
