"""Canonical heterogeneous 1 MAV + 2 UAV versus 2 Blue environment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import yaml

from .blue_policy import BluePolicy
from .dynamics import map_normalized_action, rk4_step
from .geometry import compute_pairwise_geometry
from .models import Aircraft, AircraftSpec, AircraftState
from .reward import situation_reward

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "env.yaml"
RED_IDS = ("MAV", "UAV1", "UAV2")
BLUE_IDS = ("Blue1", "Blue2")
ENTITY_IDS = RED_IDS + BLUE_IDS
TYPE_BY_ID = {"MAV": "MAV", "UAV1": "UAV", "UAV2": "UAV", "Blue1": "Blue", "Blue2": "Blue"}
TYPE_ONE_HOT = {
    "MAV": (1.0, 0.0, 0.0),
    "UAV": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
}
ENVIRONMENT_VERSION = "heterogeneous_mavuav_3v2_v2_1"
OBS_DIM = 55
GLOBAL_STATE_DIM = 67
CROSS_TEAM_ATTACK_PAIRS = tuple((red, blue) for red in RED_IDS for blue in BLUE_IDS) + tuple(
    (blue, red) for blue in BLUE_IDS for red in RED_IDS
)


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [lower, upper]")
    pair = (float(value[0]), float(value[1]))
    if not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must have finite lower < upper")
    return pair


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all fields consumed by the environment and reject stale fields."""
    expected = {
        "environment_version", "simulation", "battlefield", "aircraft_specs", "scenario",
        "randomization_profiles", "sensing", "normalization", "safety", "combat", "reward", "blue_policy",
    }
    if set(config) != expected:
        raise ValueError(f"config keys must be exactly {sorted(expected)}, got {sorted(config)}")
    cfg = deepcopy(dict(config))
    if cfg["environment_version"] != ENVIRONMENT_VERSION:
        raise ValueError(f"environment_version must be {ENVIRONMENT_VERSION!r}")
    sim = cfg["simulation"]
    if set(sim) != {"decision_dt", "physics_dt", "max_decision_steps"}:
        raise ValueError("simulation has unknown or missing fields")
    decision_dt, physics_dt = float(sim["decision_dt"]), float(sim["physics_dt"])
    if decision_dt <= 0.0 or physics_dt <= 0.0:
        raise ValueError("decision_dt and physics_dt must be positive")
    ratio = decision_dt / physics_dt
    if not np.isclose(ratio, round(ratio), atol=1e-10):
        raise ValueError("decision_dt / physics_dt must be an integer")
    if int(sim["max_decision_steps"]) <= 0:
        raise ValueError("max_decision_steps must be positive")
    if set(cfg["battlefield"]) != {"x", "y", "altitude"}:
        raise ValueError("battlefield has unknown or missing fields")
    for axis in ("x", "y", "altitude"):
        cfg["battlefield"][axis] = _pair(cfg["battlefield"][axis], f"battlefield.{axis}")
    if set(cfg["aircraft_specs"]) != {"MAV", "UAV", "Blue"}:
        raise ValueError("aircraft_specs must define MAV, UAV and Blue exactly")
    for aircraft_type in ("MAV", "UAV", "Blue"):
        raw = cfg["aircraft_specs"][aircraft_type]
        if set(raw) != {"v_min", "v_max", "nx", "ny", "nz"}:
            raise ValueError(f"aircraft_specs.{aircraft_type} has unknown or missing fields")
        AircraftSpec(aircraft_type, float(raw["v_min"]), float(raw["v_max"]), _pair(raw["nx"], "nx"), _pair(raw["ny"], "ny"), _pair(raw["nz"], "nz"))
    if set(cfg["scenario"]) != {"default_profile", "initial"} or set(cfg["scenario"]["initial"]) != set(ENTITY_IDS):
        raise ValueError("scenario must define default_profile and the five fixed initial entity slots")
    profiles = cfg["randomization_profiles"]
    if set(profiles) != {"learnability", "main"} or cfg["scenario"]["default_profile"] not in profiles:
        raise ValueError("randomization_profiles must define learnability/main and include scenario.default_profile")
    profile_fields = {"team_xy_jitter", "slot_xy_jitter", "altitude_jitter", "speed_jitter", "heading_jitter_deg"}
    for profile_name, profile in profiles.items():
        if set(profile) != profile_fields or any(float(profile[key]) < 0 for key in profile_fields):
            raise ValueError(f"invalid randomization profile: {profile_name}")
    if set(cfg["sensing"]) != {"MAV_range", "UAV_range"} or any(float(value) <= 0 for value in cfg["sensing"].values()):
        raise ValueError("sensing must define positive MAV_range and UAV_range")
    normalization_fields = {"self_xy_scale", "relative_xy_scale", "relative_altitude_scale", "distance_scale", "relative_velocity_scale"}
    if set(cfg["normalization"]) != normalization_fields or any(float(value) <= 0 for value in cfg["normalization"].values()):
        raise ValueError("normalization fields must be complete and positive")
    if set(cfg["safety"]) != {"red_safe_distance", "red_safe_distance_penalty"}:
        raise ValueError("safety has unknown or missing fields")
    if float(cfg["safety"]["red_safe_distance"]) <= 0 or float(cfg["safety"]["red_safe_distance_penalty"]) > 0:
        raise ValueError("safety distance must be positive and penalty non-positive")
    combat = cfg["combat"]
    if set(combat) != {"distance", "ata_deg", "aa_deg", "hold_steps"}:
        raise ValueError("combat has unknown or missing fields")
    combat["distance"] = _pair(combat["distance"], "combat.distance")
    if int(combat["hold_steps"]) <= 0:
        raise ValueError("combat.hold_steps must be positive")
    if not (0.0 < float(combat["ata_deg"]) <= 180.0 and 0.0 < float(combat["aa_deg"]) <= 180.0):
        raise ValueError("combat angles must be in (0, 180] degrees")
    reward_fields = {"blue_kill", "uav_loss", "mav_loss", "terminal_red_win", "terminal_blue_win", "terminal_draw"}
    if set(cfg["reward"]) != reward_fields or not np.all(np.isfinite([float(cfg["reward"][key]) for key in reward_fields])):
        raise ValueError("reward has unknown, missing or non-finite fields")
    if set(cfg["blue_policy"]) != {"target_mode"}:
        raise ValueError("blue_policy has unknown or missing fields")
    mode = cfg["blue_policy"]["target_mode"]
    if mode not in BluePolicy.MODES:
        raise ValueError(f"invalid blue target mode: {mode}")
    for aircraft_id in ENTITY_IDS:
        start = cfg["scenario"]["initial"][aircraft_id]
        if set(start) != {"position", "speed", "heading_deg"} or len(start["position"]) != 3:
            raise ValueError(f"invalid initial state for {aircraft_id}")
        x, y, h = (float(v) for v in start["position"])
        spec = cfg["aircraft_specs"][TYPE_BY_ID[aircraft_id]]
        if not (cfg["battlefield"]["x"][0] <= x <= cfg["battlefield"]["x"][1]):
            raise ValueError(f"{aircraft_id} initial x is outside battlefield")
        if not (cfg["battlefield"]["y"][0] <= y <= cfg["battlefield"]["y"][1]):
            raise ValueError(f"{aircraft_id} initial y is outside battlefield")
        if not (cfg["battlefield"]["altitude"][0] <= h <= cfg["battlefield"]["altitude"][1]):
            raise ValueError(f"{aircraft_id} initial altitude is outside battlefield")
        if not (float(spec["v_min"]) <= float(start["speed"]) <= float(spec["v_max"])):
            raise ValueError(f"{aircraft_id} initial speed is outside its limits")
    return cfg


def load_environment_config(path_or_config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(path_or_config, Mapping):
        return validate_config(path_or_config)
    path = DEFAULT_CONFIG if path_or_config is None else Path(path_or_config)
    with path.open("r", encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))


class HeterogeneousMAVUAVAirCombatEnv:
    """Three trainable Red agents and two fixed-rule homogeneous Blue aircraft."""

    red_ids = RED_IDS
    blue_ids = BLUE_IDS
    observation_dim = OBS_DIM
    global_state_dim = GLOBAL_STATE_DIM
    action_dim = 3

    def __init__(self, config_path: str | Path | Mapping[str, Any] | None = None, *, seed: int | None = None, blue_target_mode: str | None = None, randomize: bool | None = None, profile: str | None = None) -> None:
        self.config = load_environment_config(config_path)
        sim = self.config["simulation"]
        self.decision_dt = float(sim["decision_dt"])
        self.physics_dt = float(sim["physics_dt"])
        self.physics_substeps = int(round(self.decision_dt / self.physics_dt))
        self.max_decision_steps = int(sim["max_decision_steps"])
        self.randomize = True if randomize is None else bool(randomize)
        self.profile = profile or str(self.config["scenario"]["default_profile"])
        if self.profile not in self.config["randomization_profiles"]:
            raise ValueError(f"unknown randomization profile: {self.profile}")
        mode = blue_target_mode or self.config["blue_policy"]["target_mode"]
        self.blue_policy = BluePolicy(mode, self.decision_dt, self.physics_dt)
        self.rng = np.random.default_rng(seed)
        self.entities: dict[str, Aircraft] = {}
        self.step_count = 0
        self.episode_return = 0.0
        self._running = False
        self._attack_streak: dict[tuple[str, str], int] = {}
        self._red_attack_kills: set[str] = set()
        self._blue_attack_kills: set[str] = set()

    @property
    def agents(self) -> list[str]:
        return list(self.red_ids)

    @property
    def active_masks(self) -> np.ndarray:
        return np.asarray([float(self.entities[aid].state.alive) for aid in self.red_ids], dtype=np.float32)

    def _spec(self, aircraft_type: str) -> AircraftSpec:
        raw = self.config["aircraft_specs"][aircraft_type]
        return AircraftSpec(aircraft_type, float(raw["v_min"]), float(raw["v_max"]), tuple(raw["nx"]), tuple(raw["ny"]), tuple(raw["nz"]))

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = {} if options is None else dict(options)
        randomize = bool(options.get("randomize", self.randomize))
        self.profile = str(options.get("profile", self.profile))
        if self.profile not in self.config["randomization_profiles"]:
            raise ValueError(f"unknown randomization profile: {self.profile}")
        random_cfg = self.config["randomization_profiles"][self.profile]
        team_offsets: dict[str, tuple[float, float]] = {}
        if randomize:
            team_jitter = float(random_cfg["team_xy_jitter"])
            for team in ("red", "blue"):
                team_offsets[team] = (
                    float(self.rng.uniform(-team_jitter, team_jitter)),
                    float(self.rng.uniform(-team_jitter, team_jitter)),
                )
        else:
            team_offsets = {"red": (0.0, 0.0), "blue": (0.0, 0.0)}
        self.entities = {}
        for aircraft_id in ENTITY_IDS:
            aircraft_type = TYPE_BY_ID[aircraft_id]
            start = self.config["scenario"]["initial"][aircraft_id]
            x, y, h = (float(v) for v in start["position"])
            speed, heading = float(start["speed"]), np.deg2rad(float(start["heading_deg"]))
            if randomize:
                team = "red" if aircraft_id in RED_IDS else "blue"
                x += team_offsets[team][0]
                y += team_offsets[team][1]
                slot_jitter = float(random_cfg["slot_xy_jitter"])
                x += self.rng.uniform(-slot_jitter, slot_jitter)
                y += self.rng.uniform(-slot_jitter, slot_jitter)
                h += self.rng.uniform(-float(random_cfg["altitude_jitter"]), float(random_cfg["altitude_jitter"]))
                speed += self.rng.uniform(-float(random_cfg["speed_jitter"]), float(random_cfg["speed_jitter"]))
                heading += np.deg2rad(self.rng.uniform(-float(random_cfg["heading_jitter_deg"]), float(random_cfg["heading_jitter_deg"])))
            spec = self._spec(aircraft_type)
            state = AircraftState(x, y, h, float(np.clip(speed, spec.v_min, spec.v_max)), 0.0, float((heading + np.pi) % (2 * np.pi) - np.pi))
            self.entities[aircraft_id] = Aircraft(aircraft_id, "red" if aircraft_id in RED_IDS else "blue", spec, state)
        self.step_count = 0
        self.episode_return = 0.0
        self._attack_streak.clear()
        self._red_attack_kills.clear()
        self._blue_attack_kills.clear()
        self._running = True
        mode = self.blue_policy.reset(self.rng)
        return self._observations(), {
            "outcome": None, "attack_events": [], "killed_ids": [], "death_causes": {},
            "active_masks": self.active_masks.copy(), "blue_target_mode": mode, "profile": self.profile,
        }

    def _action_dict(self, actions: Mapping[str, np.ndarray] | np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(actions, Mapping):
            if set(actions) != set(RED_IDS):
                raise KeyError(f"actions must contain exactly {RED_IDS}")
            result = {aid: np.asarray(actions[aid], dtype=np.float64) for aid in RED_IDS}
        else:
            values = np.asarray(actions, dtype=np.float64)
            if values.shape != (3, 3):
                raise ValueError(f"actions must have shape (3, 3), got {values.shape}")
            result = {aid: values[index] for index, aid in enumerate(RED_IDS)}
        for aid, action in result.items():
            if action.shape != (3,) or not np.all(np.isfinite(action)):
                raise ValueError(f"action for {aid} must be a finite shape-(3,) array")
            if not self.entities[aid].state.alive:
                result[aid] = np.zeros(3, dtype=np.float64)
        return result

    def step(self, actions: Mapping[str, np.ndarray] | np.ndarray | list[np.ndarray]):
        if not self._running:
            raise RuntimeError("reset() must be called before step()")
        red_actions = self._action_dict(actions)
        red_entities = {aid: self.entities[aid] for aid in RED_IDS}
        all_actions = dict(red_actions)
        for aid in BLUE_IDS:
            all_actions[aid] = self.blue_policy.action(self.entities[aid], red_entities)
        commands = {
            aid: map_normalized_action(action, self.entities[aid].state, self.entities[aid].spec)
            for aid, action in all_actions.items() if self.entities[aid].state.alive
        }
        for _ in range(self.physics_substeps):
            for aid in ENTITY_IDS:
                entity = self.entities[aid]
                if entity.state.alive:
                    entity.state = rk4_step(entity.state, commands[aid], self.physics_dt, entity.spec)
        death_causes = self._apply_boundaries()
        attack_events, attack_deaths = self._resolve_attacks()
        death_causes.update(attack_deaths)
        self.step_count += 1
        minimum_friendly_distance = self._minimum_friendly_red_distance()
        safety_cfg = self.config["safety"]
        safety_violation = minimum_friendly_distance < float(safety_cfg["red_safe_distance"])
        safety_reward = float(safety_cfg["red_safe_distance_penalty"]) if safety_violation else 0.0
        terminated, truncated, outcome = self._termination()
        situation = self._team_situation_reward()
        reward_cfg = self.config["reward"]
        event = reward_cfg["blue_kill"] * sum(aid in BLUE_IDS and cause == "red_attack" for aid, cause in death_causes.items())
        event += reward_cfg["uav_loss"] * sum(aid in ("UAV1", "UAV2") for aid in death_causes)
        event += reward_cfg["mav_loss"] * int("MAV" in death_causes)
        terminal = 0.0
        if outcome == "red": terminal = float(reward_cfg["terminal_red_win"])
        elif outcome == "blue": terminal = float(reward_cfg["terminal_blue_win"])
        elif outcome == "draw": terminal = float(reward_cfg["terminal_draw"])
        team_reward = float(situation + event + terminal + safety_reward)
        self.episode_return += team_reward
        rewards = {aid: team_reward for aid in RED_IDS}
        self._running = not (terminated or truncated)
        info: dict[str, Any] = {
            "outcome": outcome, "attack_events": attack_events,
            "killed_ids": sorted(death_causes), "death_causes": dict(sorted(death_causes.items())),
            "active_masks": self.active_masks.copy(), "team_situation": situation,
            "event_reward": float(event), "terminal_reward": float(terminal),
            "minimum_friendly_red_distance": float(minimum_friendly_distance),
            "red_safe_distance_violation": bool(safety_violation), "safety_reward": safety_reward,
        }
        if terminated or truncated:
            info["episode_summary"] = self._episode_summary(outcome)
        observations = self._observations()
        if not np.isfinite(team_reward) or not all(np.all(np.isfinite(v)) for v in observations.values()) or not np.all(np.isfinite(self.global_state())):
            raise FloatingPointError("environment produced non-finite output")
        return observations, rewards, terminated, truncated, info

    def _minimum_friendly_red_distance(self) -> float:
        alive = [self.entities[aid] for aid in RED_IDS if self.entities[aid].state.alive]
        distances = [
            compute_pairwise_geometry(alive[i].state, alive[j].state).distance
            for i in range(len(alive)) for j in range(i + 1, len(alive))
        ]
        return float(min(distances)) if distances else float("inf")

    def _deactivate(self, aid: str, cause: str, deaths: dict[str, str]) -> None:
        entity = self.entities[aid]
        if entity.state.alive:
            entity.state.alive = False
            entity.inactive_cause = cause
            deaths[aid] = cause

    def _apply_boundaries(self) -> dict[str, str]:
        deaths: dict[str, str] = {}
        battlefield = self.config["battlefield"]
        for aid in ENTITY_IDS:
            entity = self.entities[aid]
            if not entity.state.alive:
                continue
            state = entity.state
            outside = not (battlefield["x"][0] <= state.x <= battlefield["x"][1] and battlefield["y"][0] <= state.y <= battlefield["y"][1] and battlefield["altitude"][0] <= state.h <= battlefield["altitude"][1])
            if outside:
                self._deactivate(aid, "blue_escape" if aid in BLUE_IDS else "boundary", deaths)
        return deaths

    def _resolve_attacks(self) -> tuple[list[dict[str, str]], dict[str, str]]:
        combat = self.config["combat"]
        pairs: list[tuple[str, str]] = []
        for attacker_id in ENTITY_IDS:
            attacker = self.entities[attacker_id]
            if not attacker.state.alive:
                continue
            target_ids = BLUE_IDS if attacker.team == "red" else RED_IDS
            for target_id in target_ids:
                target = self.entities[target_id]
                key = (attacker_id, target_id)
                if not target.state.alive:
                    self._attack_streak[key] = 0
                    continue
                geometry = compute_pairwise_geometry(attacker.state, target.state)
                inside = combat["distance"][0] <= geometry.distance <= combat["distance"][1] and geometry.ata < np.deg2rad(combat["ata_deg"]) and geometry.aa < np.deg2rad(combat["aa_deg"])
                self._attack_streak[key] = self._attack_streak.get(key, 0) + 1 if inside else 0
                if self._attack_streak[key] >= int(combat["hold_steps"]):
                    pairs.append(key)
        events = [{"attacker": attacker, "target": target} for attacker, target in sorted(pairs)]
        deaths: dict[str, str] = {}
        for _, target in pairs:
            cause = "red_attack" if target in BLUE_IDS else "blue_attack"
            self._deactivate(target, cause, deaths)
            if cause == "red_attack": self._red_attack_kills.add(target)
            else: self._blue_attack_kills.add(target)
        for key in list(self._attack_streak):
            if key[0] in deaths or key[1] in deaths:
                self._attack_streak[key] = 0
        return events, deaths

    def _termination(self) -> tuple[bool, bool, str | None]:
        mav = self.entities["MAV"]
        if not mav.state.alive:
            return True, False, "blue"
        all_blue_inactive = not any(self.entities[aid].state.alive for aid in BLUE_IDS)
        if all_blue_inactive:
            return True, False, "red" if self._red_attack_kills == set(BLUE_IDS) else "blue"
        if self.step_count >= self.max_decision_steps:
            return False, True, "draw"
        return False, False, None

    def _team_situation_reward(self) -> float:
        visible_alive_blue = [
            self.entities[aid]
            for aid in BLUE_IDS
            if self.entities[aid].state.alive and self.team_visible(aid)
        ]
        total = 0.0
        for aid in RED_IDS:
            own = self.entities[aid]
            if own.state.alive and visible_alive_blue:
                total += max(situation_reward(own.state, target.state) for target in visible_alive_blue)
        return float(total / 3.0)

    def _episode_summary(self, outcome: str | None) -> dict[str, Any]:
        return {
            "outcome": outcome, "episode_length": self.step_count,
            "mav_survived": bool(self.entities["MAV"].state.alive),
            "red_uav_survivors": sum(self.entities[aid].state.alive for aid in ("UAV1", "UAV2")),
            "blue_survivors": sum(self.entities[aid].state.alive for aid in BLUE_IDS),
            "red_attack_kills": len(self._red_attack_kills), "blue_attack_kills": len(self._blue_attack_kills),
            "red_uav_losses": sum(not self.entities[aid].state.alive for aid in ("UAV1", "UAV2")),
            "mav_loss": int(not self.entities["MAV"].state.alive),
            "blue_target_mode": self.blue_policy.episode_mode, "episode_return": float(self.episode_return),
        }

    def _self_xy_norm(self, value: float) -> float:
        return float(np.clip(value / float(self.config["normalization"]["self_xy_scale"]), -1.0, 1.0))

    def _global_xy_norm(self, value: float, axis: str) -> float:
        if axis not in ("x", "y"):
            raise ValueError(f"global XY axis must be 'x' or 'y', got {axis!r}")
        lower, upper = self.config["battlefield"][axis]
        return float(np.clip(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0))

    def _altitude_norm(self, value: float) -> float:
        lower, upper = self.config["battlefield"]["altitude"]
        return float(np.clip(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0))

    def _relative_position_values(self, relative_position: np.ndarray) -> list[float]:
        normalization = self.config["normalization"]
        return [
            float(np.clip(relative_position[0] / float(normalization["relative_xy_scale"]), -1.0, 1.0)),
            float(np.clip(relative_position[1] / float(normalization["relative_xy_scale"]), -1.0, 1.0)),
            float(np.clip(relative_position[2] / float(normalization["relative_altitude_scale"]), -1.0, 1.0)),
        ]

    def _distance_norm(self, distance: float) -> float:
        return float(np.clip(distance / float(self.config["normalization"]["distance_scale"]), 0.0, 1.0))

    def _relative_velocity_values(self, relative_velocity: np.ndarray) -> list[float]:
        scale = float(self.config["normalization"]["relative_velocity_scale"])
        return [float(np.clip(value / scale, -1.0, 1.0)) for value in relative_velocity]

    def direct_visible(self, own_id: str, blue_id: str) -> bool:
        own, blue = self.entities[own_id], self.entities[blue_id]
        if own_id not in RED_IDS or blue_id not in BLUE_IDS or not own.state.alive or not blue.state.alive:
            return False
        sensor_range = float(self.config["sensing"][f"{TYPE_BY_ID[own_id]}_range"])
        return compute_pairwise_geometry(own.state, blue.state).distance <= sensor_range

    def team_visible(self, blue_id: str) -> bool:
        return any(self.direct_visible(red_id, blue_id) for red_id in RED_IDS)

    def datalink_visible(self, own_id: str, blue_id: str) -> bool:
        return self.team_visible(blue_id) and not self.direct_visible(own_id, blue_id)

    def _observations(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for own_id in RED_IDS:
            own = self.entities[own_id]
            state = own.state
            # Stable 55D contract: self 11D, two Red teammates 11D each,
            # then Blue1/Blue2 enemy blocks 11D each.
            values = [
                self._self_xy_norm(state.x), self._self_xy_norm(state.y), self._altitude_norm(state.h),
                float(np.clip(state.v / 400.0, 0.0, 1.0)), state.theta / np.pi, state.psi / np.pi,
                float(state.alive), *TYPE_ONE_HOT[TYPE_BY_ID[own_id]],
                float(np.clip(self.step_count / self.max_decision_steps, 0.0, 1.0)),
            ]
            for friend_id in RED_IDS:
                if friend_id == own_id: continue
                friend = self.entities[friend_id]
                geometry = compute_pairwise_geometry(state, friend.state)
                values.extend([
                    *self._relative_position_values(geometry.relative_position), self._distance_norm(geometry.distance),
                    *self._relative_velocity_values(geometry.relative_velocity), float(friend.state.alive),
                    *TYPE_ONE_HOT[TYPE_BY_ID[friend_id]],
                ])
            for blue_id in BLUE_IDS:
                blue = self.entities[blue_id]
                direct = self.direct_visible(own_id, blue_id)
                datalink = self.datalink_visible(own_id, blue_id)
                if direct or datalink:
                    geometry = compute_pairwise_geometry(state, blue.state)
                    enemy_geometry = [
                        *self._relative_position_values(geometry.relative_position), self._distance_norm(geometry.distance),
                        geometry.ata / np.pi, geometry.aa / np.pi,
                    ]
                else:
                    enemy_geometry = [0.0] * 6
                hold_steps = int(self.config["combat"]["hold_steps"])
                streak = min(self._attack_streak.get((own_id, blue_id), 0), hold_steps) / hold_steps
                values.extend([
                    *enemy_geometry, float(blue.state.alive), float(direct), float(datalink), float(streak),
                    float(blue_id in self._red_attack_kills),
                ])
            observation = np.asarray(values, dtype=np.float32)
            if observation.shape != (OBS_DIM,):
                raise AssertionError(f"observation contract violated: {observation.shape}")
            result[own_id] = observation
        return result

    def global_state(self) -> np.ndarray:
        values: list[float] = []
        for aid in ENTITY_IDS:
            state = self.entities[aid].state
            values.extend([
                self._global_xy_norm(state.x, "x"), self._global_xy_norm(state.y, "y"), self._altitude_norm(state.h),
                float(np.clip(state.v / 400.0, 0.0, 1.0)), state.theta / np.pi, state.psi / np.pi,
                float(state.alive), *TYPE_ONE_HOT[TYPE_BY_ID[aid]],
            ])
        hold_steps = int(self.config["combat"]["hold_steps"])
        values.extend(min(self._attack_streak.get(pair, 0), hold_steps) / hold_steps for pair in CROSS_TEAM_ATTACK_PAIRS)
        values.extend(float(blue_id in self._red_attack_kills) for blue_id in BLUE_IDS)
        values.extend([
            float(self.blue_policy.episode_mode == "nearest"),
            float(self.blue_policy.episode_mode == "mav_priority"),
            float(np.clip(self.step_count / self.max_decision_steps, 0.0, 1.0)),
        ])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (GLOBAL_STATE_DIM,):
            raise AssertionError(f"global state contract violated: {result.shape}")
        return result


MAVSpec = AircraftSpec("MAV", 250.0, 400.0, (-1.0, 5.0), (-1.5, 2.0), (-3.0, 3.0))
UAVSpec = AircraftSpec("UAV", 150.0, 300.0, (-1.0, 5.0), (-1.5, 1.5), (-2.0, 2.0))
BlueSpec = AircraftSpec("Blue", 250.0, 400.0, (-1.0, 5.0), (-1.5, 3.0), (-3.0, 3.0))
