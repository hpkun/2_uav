"""Mission-aligned v14 reward streams over the frozen v12 4v3 mechanics."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .controller import TargetStateController
from .dynamics import PointMassDynamics
from .environment_4v3_v11 import combat_potential_v11
from .environment_4v3_v12 import (
    BLUE_TEAM_SIZE_V12,
    DEATH_LOCK_V12,
    GS_DIM_V12,
    OBS_DIM_V12,
    RED_TEAM_SIZE_V12,
    FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv,
    aggregate_team_reward_v12,
    reward_group_totals_v12,
)
from .integrator import RK4Integrator
from .scenario_4v3_v14 import (
    BLUE_IDS_V14,
    RED_COMBAT_IDS_V14,
    RED_IDS_V14,
    REWARD_COMPONENT_KEYS_V14,
    REWARD_CONTRACT_VERSION_V14,
    FunctionalHeterogeneous4v3V14Scenario,
    resolved_reward_contract_v14,
    validate_heterogeneous_4v3_v14_config,
)

OBS_DIM_V14 = OBS_DIM_V12
GS_DIM_V14 = GS_DIM_V12
RED_TEAM_SIZE_V14 = RED_TEAM_SIZE_V12
BLUE_TEAM_SIZE_V14 = BLUE_TEAM_SIZE_V12

AGENT_REWARD_COMPONENT_KEYS_V14 = (
    "common_mission_reward",
    "kill_event_reward",
    "death_event_penalty",
    "boundary_event_penalty",
    "geometry_progress_reward",
    "lock_progress_reward",
    "half_lock_event_reward",
    "support_unique_detection_reward",
    "support_cue_to_direct_reward",
    "support_cue_to_half_lock_reward",
    "support_assisted_kill_reward",
    "support_formation_progress_reward",
    "local_dense_reward",
    "agent_total_reward",
)


def _empty_agent_components() -> dict[str, dict[str, float]]:
    return {
        agent_id: {key: 0.0 for key in AGENT_REWARD_COMPONENT_KEYS_V14}
        for agent_id in RED_IDS_V14
    }


class FunctionalHeterogeneous4v3V14MissionAlignedEnv(
    FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv
):
    """v12 transition mechanics plus team and per-red-agent v14 rewards."""

    variant = "functional_heterogeneous_4v3_v14_mission_aligned_role_credit"
    reward_contract_version = REWARD_CONTRACT_VERSION_V14

    def __init__(
        self,
        config_path: str | Path = "configs/heterogeneous_4v3_main_v14_mission_aligned_role_credit.yaml",
    ) -> None:
        # This mirrors only v12 construction. All transition, observation,
        # targeting, cue, lock, dynamics, and boundary methods are inherited.
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        validate_heterogeneous_4v3_v14_config(self.config)
        self.reward_contract = resolved_reward_contract_v14(self.config)
        self.profile = self.config["combat_profile"]
        simulation = self.config["simulation"]
        self.scenario = FunctionalHeterogeneous4v3V14Scenario(self.config)
        self.dynamics = PointMassDynamics(float(simulation.get("gravity", 9.81)))
        self.integrator = RK4Integrator(float(simulation["dt"]))
        self.controller = TargetStateController(
            **self.config["action"], gravity=float(simulation.get("gravity", 9.81))
        )
        self.aircraft = []
        self.step_count = 0
        self._running = False
        self._death_causes = {}
        self._attack_kills = {"red": 0, "blue": 0}
        self._kill_steps = {"red": [], "blue": []}
        self.targets = {}
        self.target_hold_steps = {}
        self.target_lost_steps = {}
        self.target_switch_count = 0
        self.lock_progress = {}
        self.max_lock_progress = 0.0
        self._half_lock_pairs = set()
        self._support_seen = set()
        self._cue_pairs = set()
        self._cue_last_step = {}
        self._cue_to_direct_pairs = set()
        self._cue_to_half_pairs = set()
        self._support_cues = {cid: None for cid in RED_COMBAT_IDS_V14}
        self._last_formation_score = 0.0
        self._episode_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V14
        }
        self._last_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V14
        }
        self._last_raw_dense_reward = 0.0
        self._episode_metrics: dict[str, float] = {}
        self._episode_return = 0.0
        self._last_agent_rewards = {agent_id: 0.0 for agent_id in RED_IDS_V14}
        self._last_agent_reward_components = _empty_agent_components()
        self._episode_agent_returns = {agent_id: 0.0 for agent_id in RED_IDS_V14}
        self._episode_agent_reward_components = _empty_agent_components()

    def reset(self, seed: int | None = None):
        values = super().reset(seed)
        self._last_agent_rewards = {agent_id: 0.0 for agent_id in RED_IDS_V14}
        self._last_agent_reward_components = _empty_agent_components()
        self._episode_agent_returns = {agent_id: 0.0 for agent_id in RED_IDS_V14}
        self._episode_agent_reward_components = _empty_agent_components()
        for agent_id in RED_IDS_V14:
            self._episode_metrics[f"{agent_id}_boundary_hard_contacts"] = 0.0
            self._episode_metrics[f"{agent_id}_boundary_hard_contacts_step"] = 0.0
        return values

    def _project_hard_boundary(self, aircraft) -> bool:
        hit = super()._project_hard_boundary(aircraft)
        if hit and aircraft.aircraft_id in RED_IDS_V14:
            agent_id = aircraft.aircraft_id
            self._episode_metrics[f"{agent_id}_boundary_hard_contacts"] += 1.0
            self._episode_metrics[f"{agent_id}_boundary_hard_contacts_step"] += 1.0
        return hit

    def _agent_rewards(
        self,
        pre_targets: dict[str, str | None],
        pre_potentials: dict[str, float],
        pre_locks: dict[str, float],
        deaths: dict[str, int],
        half_events: set[tuple[str, str]],
        killers: dict[str, str],
        team_components: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        values = _empty_agent_components()
        common = float(team_components["mission_outcome_reward"])
        for agent_id in RED_IDS_V14:
            values[agent_id]["common_mission_reward"] = common

        events = self.reward_contract["events"]
        support_events = self.reward_contract["support_events"]
        progress = self.reward_contract["combat_progress"]
        clip_min = float(self.reward_contract["dense_clip"]["min"])
        clip_max = float(self.reward_contract["dense_clip"]["max"])

        # True killer attribution: one blue death rewards only its red killer.
        for target_id, killer_id in killers.items():
            if killer_id in RED_COMBAT_IDS_V14 and target_id in BLUE_IDS_V14:
                values[killer_id]["kill_event_reward"] += float(
                    events["blue_combat_killed"]
                )

        for agent_id, cause in deaths.items():
            if cause != DEATH_LOCK_V12 or agent_id not in RED_IDS_V14:
                continue
            key = "red_support_killed" if agent_id == "red_0" else "red_combat_killed"
            values[agent_id]["death_event_penalty"] += float(events[key])

        for agent_id in RED_IDS_V14:
            contacts = min(
                4.0,
                float(
                    self._episode_metrics.get(
                        f"{agent_id}_boundary_hard_contacts_step", 0.0
                    )
                ),
            )
            values[agent_id]["boundary_event_penalty"] = max(
                -0.4, float(events["red_boundary_hard_contact"]) * contacts
            )

        for combat_id in RED_COMBAT_IDS_V14:
            old_target = pre_targets.get(combat_id)
            new_target = self.targets.get(combat_id)
            if (
                old_target is not None
                and old_target == new_target
                and self._alive(combat_id)
                and new_target is not None
                and self._alive(new_target)
            ):
                current = combat_potential_v11(
                    self._by_id(combat_id).state,
                    self._by_id(new_target).state,
                    self.profile,
                )
                delta = current - pre_potentials.get(combat_id, current)
                values[combat_id]["geometry_progress_reward"] = float(
                    progress["geometry_scale"]
                ) * float(np.clip(delta / 0.05, -1.0, 1.0))
            # The death transition receives its death event but no post-death
            # geometry/lock shaping. Future transitions remain local-zero too.
            if self._alive(combat_id):
                lock_delta = self.lock_progress.get(combat_id, 0.0) - pre_locks.get(
                    combat_id, 0.0
                )
                values[combat_id]["lock_progress_reward"] = float(
                    progress["lock_scale"]
                ) * float(np.clip(lock_delta / 0.10, -1.0, 1.0))
            values[combat_id]["half_lock_event_reward"] = float(
                progress["half_lock_event"]
            ) * sum(
                1
                for attacker_id, target_id in half_events
                if attacker_id == combat_id and target_id in BLUE_IDS_V14
            )
            raw_dense = (
                values[combat_id]["geometry_progress_reward"]
                + values[combat_id]["lock_progress_reward"]
            )
            values[combat_id]["local_dense_reward"] = float(
                np.clip(raw_dense, clip_min, clip_max)
            )

        support = values["red_0"]
        support["support_unique_detection_reward"] = float(
            support_events["unique_detection"]
        ) * float(self._episode_metrics.get("support_unique_detection_events_step", 0.0))
        support["support_cue_to_direct_reward"] = float(
            support_events["cue_to_direct"]
        ) * float(self._episode_metrics.get("support_cue_to_direct_events_step", 0.0))
        support["support_cue_to_half_lock_reward"] = float(
            support_events["cue_to_half_lock"]
        ) * float(self._episode_metrics.get("support_cue_to_half_lock_events_step", 0.0))
        assisted = 0
        for target_id, killer_id in killers.items():
            if killer_id in RED_COMBAT_IDS_V14 and target_id in BLUE_IDS_V14:
                last = self._cue_last_step.get((killer_id, target_id))
                if last is not None and self.step_count - last <= 50:
                    assisted += 1
        support["support_assisted_kill_reward"] = float(
            support_events["assisted_kill"]
        ) * assisted
        support["support_formation_progress_reward"] = float(
            team_components["support_formation_progress_reward"]
        )
        support["local_dense_reward"] = float(
            np.clip(
                support["support_formation_progress_reward"], clip_min, clip_max
            )
        )

        rewards: dict[str, float] = {}
        for agent_id, components in values.items():
            components["agent_total_reward"] = float(
                sum(
                    value
                    for key, value in components.items()
                    if key not in {
                        "geometry_progress_reward",
                        "lock_progress_reward",
                        "support_formation_progress_reward",
                        "agent_total_reward",
                    }
                )
            )
            rewards[agent_id] = components["agent_total_reward"]
        numeric = [value for row in values.values() for value in row.values()]
        if not np.isfinite(numeric).all():
            raise FloatingPointError("v14 agent reward components must be finite")
        return rewards, values

    def _compute_reward(self, *args: Any, **kwargs: Any):
        team_reward, team_components = super()._compute_reward(*args, **kwargs)
        # super uses the v14 contract stored on self, so the aggregation is
        # exactly v12 except for the mission timeout constants.
        team_components["team_total_reward"] = aggregate_team_reward_v12(
            team_components
        )
        team_reward = float(team_components["team_total_reward"])
        agent_rewards, agent_components = self._agent_rewards(
            *args[:6], team_components
        )
        self._last_agent_rewards = agent_rewards
        self._last_agent_reward_components = agent_components
        for agent_id in RED_IDS_V14:
            self._episode_agent_returns[agent_id] += agent_rewards[agent_id]
            for key, value in agent_components[agent_id].items():
                self._episode_agent_reward_components[agent_id][key] += value
        return team_reward, team_components

    def step(self, red_actions):
        for agent_id in RED_IDS_V14:
            self._episode_metrics[f"{agent_id}_boundary_hard_contacts_step"] = 0.0
        obs, state, masks, reward, done, truncated, info = super().step(red_actions)
        info["agent_rewards"] = deepcopy(self._last_agent_rewards)
        info["agent_reward_components"] = deepcopy(
            self._last_agent_reward_components
        )
        info["reward_groups"] = reward_group_totals_v12(info["reward_components"])
        return obs, state, masks, reward, done, truncated, info

    def _episode_summary(self, outcome, reason):
        summary = super()._episode_summary(outcome, reason)
        summary["agent_returns"] = deepcopy(self._episode_agent_returns)
        summary["agent_reward_components"] = deepcopy(
            self._episode_agent_reward_components
        )
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "last_agent_rewards": deepcopy(self._last_agent_rewards),
                "last_agent_reward_components": deepcopy(
                    self._last_agent_reward_components
                ),
                "episode_agent_returns": deepcopy(self._episode_agent_returns),
                "episode_agent_reward_components": deepcopy(
                    self._episode_agent_reward_components
                ),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._last_agent_rewards = deepcopy(state["last_agent_rewards"])
        self._last_agent_reward_components = deepcopy(
            state["last_agent_reward_components"]
        )
        self._episode_agent_returns = deepcopy(state["episode_agent_returns"])
        self._episode_agent_reward_components = deepcopy(
            state["episode_agent_reward_components"]
        )
        for agent_id in RED_IDS_V14:
            self._episode_metrics.setdefault(
                f"{agent_id}_boundary_hard_contacts", 0.0
            )
            self._episode_metrics.setdefault(
                f"{agent_id}_boundary_hard_contacts_step", 0.0
            )


FunctionalHeterogeneous4v3AirCombatEnvV14 = (
    FunctionalHeterogeneous4v3V14MissionAlignedEnv
)

__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V14",
    "BLUE_TEAM_SIZE_V14",
    "GS_DIM_V14",
    "OBS_DIM_V14",
    "RED_TEAM_SIZE_V14",
    "REWARD_COMPONENT_KEYS_V14",
    "FunctionalHeterogeneous4v3AirCombatEnvV14",
    "FunctionalHeterogeneous4v3V14MissionAlignedEnv",
]
