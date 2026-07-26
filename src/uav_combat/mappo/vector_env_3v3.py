"""Vector env for 3v3 with NamedTuple result, per-env blue policies, rule step."""
from __future__ import annotations

import multiprocessing, multiprocessing.connection, traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..environment_3v3 import GS_DIM, OBS_DIM, Homogeneous3v3AirCombatEnv
from ..rule_policy_3v3 import NearestTargetPursuitPolicy3v3
from ..scenario_3v3 import BLUE_IDS, RED_IDS
from .vector_env import CONTROL_DIAGNOSTIC_KEYS, K

# ===================================================================
# 3v3-specific termination / outcome encoding
# ===================================================================

THREE_V_THREE_REASON_CODE_MAP = {
    None: 0,
    "red_elimination": 1,
    "blue_elimination": 2,
    "mutual_elimination": 3,
    "max_steps": 4,
}
THREE_V_THREE_REASON_CODE_INV = {v: k for k, v in THREE_V_THREE_REASON_CODE_MAP.items()}

THREE_V_THREE_OUTCOME_CODE_MAP = {
    None: 0,
    "red": 1,
    "blue": 2,
    "draw": 3,
}
THREE_V_THREE_OUTCOME_CODE_INV = {v: k for k, v in THREE_V_THREE_OUTCOME_CODE_MAP.items()}


def encode_3v3_termination_reason(reason: str | None) -> int:
    return THREE_V_THREE_REASON_CODE_MAP.get(reason, 0)


def decode_3v3_termination_reason(code: int) -> str | None:
    return THREE_V_THREE_REASON_CODE_INV.get(code)


def encode_3v3_outcome(outcome: str | None) -> int:
    return THREE_V_THREE_OUTCOME_CODE_MAP.get(outcome, 0)


def decode_3v3_outcome(code: int) -> str | None:
    return THREE_V_THREE_OUTCOME_CODE_INV.get(code)

AGENT_ORDER = RED_IDS + BLUE_IDS

RED_REWARD_COMPONENT_KEYS_3V3 = (
    "red_approach_reward",
    "red_attack_advantage_reward",
    "red_threat_penalty",
    "red_soft_boundary_penalty",
    "red_friendly_separation_penalty",
    "red_head_on_risk_penalty",
    "red_time_penalty",
    "red_dense_reward",
    "red_kill_reward",
    "red_attack_death_penalty",
    "red_boundary_death_penalty",
    "red_collision_death_penalty",
    "red_terminal_reward",
    "red_team_total_reward",
)


class VectorStepResult3v3(NamedTuple):
    observations: np.ndarray                         # [N, 6, 68]
    global_states: np.ndarray                         # [N, 48]
    team_rewards: np.ndarray                          # [N]
    terminated: np.ndarray                            # [N] bool
    truncated: np.ndarray                             # [N] bool
    alive_masks: np.ndarray                           # [N, 6]
    attack_targets: np.ndarray                        # [N, 6] int8
    step_death_causes: np.ndarray                     # [N, 6] int8
    team_alive_counts: np.ndarray                     # [N, 2] int8
    step_red_attack_kills: np.ndarray                 # [N] int8
    step_blue_attack_kills: np.ndarray                # [N] int8
    step_red_boundary_deaths: np.ndarray              # [N] int8
    step_blue_boundary_deaths: np.ndarray             # [N] int8
    step_red_boundary_altitude_deaths: np.ndarray     # [N] int8
    step_blue_boundary_altitude_deaths: np.ndarray    # [N] int8
    step_red_boundary_xy_deaths: np.ndarray           # [N] int8
    step_blue_boundary_xy_deaths: np.ndarray          # [N] int8
    step_red_collision_deaths: np.ndarray             # [N] int8
    step_blue_collision_deaths: np.ndarray            # [N] int8
    red_reward_components: np.ndarray                 # [N, 14] float32
    red_complete_elimination_success: np.ndarray      # [N] bool
    blue_complete_elimination_success: np.ndarray     # [N] bool
    geometry: np.ndarray                              # [N, 6, 3] (distance, ATA, AA)
    control_diagnostics: np.ndarray                   # [N, 6, K]
    termination_reason_codes: np.ndarray              # [N] int8
    outcome_codes: np.ndarray                         # [N] int8
    episode_valid: np.ndarray                         # [N] bool  – this env-step ended an episode
    episode_red_attack_kills: np.ndarray              # [N] int8
    episode_blue_attack_kills: np.ndarray             # [N] int8
    episode_red_survivors: np.ndarray                 # [N] int8
    episode_blue_survivors: np.ndarray                # [N] int8
    episode_red_attack_deaths: np.ndarray             # [N] int8
    episode_blue_attack_deaths: np.ndarray            # [N] int8
    episode_red_boundary_deaths: np.ndarray           # [N] int8
    episode_blue_boundary_deaths: np.ndarray          # [N] int8
    episode_red_boundary_altitude_deaths: np.ndarray  # [N] int8
    episode_blue_boundary_altitude_deaths: np.ndarray # [N] int8
    episode_red_boundary_xy_deaths: np.ndarray        # [N] int8
    episode_blue_boundary_xy_deaths: np.ndarray       # [N] int8
    episode_red_friendly_collision_deaths: np.ndarray # [N] int8
    episode_blue_friendly_collision_deaths: np.ndarray# [N] int8
    episode_red_cross_collision_deaths: np.ndarray    # [N] int8
    episode_blue_cross_collision_deaths: np.ndarray   # [N] int8
    episode_length: np.ndarray                        # [N] int32


def _extract_diagnostics_3v3(diag):
    vec = np.empty(K, dtype=np.float32)
    for i, key in enumerate(CONTROL_DIAGNOSTIC_KEYS):
        val = diag.get(key, 0.0)
        vec[i] = float(val) if not isinstance(val, (bool, np.bool_)) else float(val)
    return vec


def _fill_geom(info, env, i, geom):
    ng = info.get("nearest_enemy_geometry", {})
    for j, aid in enumerate(AGENT_ORDER):
        g = ng.get(aid, {})
        geom[i, j, 0] = float(g.get("distance", 0.0))
        geom[i, j, 1] = float(g.get("ata", 0.0))
        geom[i, j, 2] = float(g.get("aa", 0.0))


def _fill_red_reward_components(info, i, arr):
    rc = info.get("reward_components", {})
    for j, key in enumerate(RED_REWARD_COMPONENT_KEYS_3V3):
        arr[i, j] = float(rc.get(key, 0.0))


def _episode_fields(info, L, i):
    es = info.get("episode_summary")
    if es is None:
        return  # leave at zero
    arrays, idx = _episode_fields.arrays, i
    arrays["ep_valid"][idx] = True
    arrays["ep_rak"][idx] = es["red_attack_kills"]
    arrays["ep_bak"][idx] = es["blue_attack_kills"]
    arrays["ep_rs"][idx] = es["red_survivors"]
    arrays["ep_bs"][idx] = es["blue_survivors"]
    arrays["ep_rad"][idx] = es["red_death_causes"]["attack_deaths"]
    arrays["ep_bad"][idx] = es["blue_death_causes"]["attack_deaths"]
    arrays["ep_rbd"][idx] = es["red_death_causes"]["boundary_deaths"]
    arrays["ep_bbd"][idx] = es["blue_death_causes"]["boundary_deaths"]
    arrays["ep_rbad"][idx] = es["red_death_causes"]["boundary_altitude_deaths"]
    arrays["ep_bbad"][idx] = es["blue_death_causes"]["boundary_altitude_deaths"]
    arrays["ep_rbxy"][idx] = es["red_death_causes"]["boundary_xy_deaths"]
    arrays["ep_bbxy"][idx] = es["blue_death_causes"]["boundary_xy_deaths"]
    arrays["ep_rfc"][idx] = es["red_death_causes"]["friendly_collision_deaths"]
    arrays["ep_bfc"][idx] = es["blue_death_causes"]["friendly_collision_deaths"]
    arrays["ep_rcc"][idx] = es["red_death_causes"]["cross_team_collision_deaths"]
    arrays["ep_bcc"][idx] = es["blue_death_causes"]["cross_team_collision_deaths"]
    arrays["ep_len"][idx] = es["episode_length"]


def _make_episode_arrays(L):
    return {
        "ep_valid": np.zeros(L, dtype=bool),
        "ep_rak": np.zeros(L, dtype=np.int8), "ep_bak": np.zeros(L, dtype=np.int8),
        "ep_rs": np.zeros(L, dtype=np.int8), "ep_bs": np.zeros(L, dtype=np.int8),
        "ep_rad": np.zeros(L, dtype=np.int8), "ep_bad": np.zeros(L, dtype=np.int8),
        "ep_rbd": np.zeros(L, dtype=np.int8), "ep_bbd": np.zeros(L, dtype=np.int8),
        "ep_rbad": np.zeros(L, dtype=np.int8), "ep_bbad": np.zeros(L, dtype=np.int8),
        "ep_rbxy": np.zeros(L, dtype=np.int8), "ep_bbxy": np.zeros(L, dtype=np.int8),
        "ep_rfc": np.zeros(L, dtype=np.int8), "ep_bfc": np.zeros(L, dtype=np.int8),
        "ep_rcc": np.zeros(L, dtype=np.int8), "ep_bcc": np.zeros(L, dtype=np.int8),
        "ep_len": np.zeros(L, dtype=np.int32),
    }


def _build_result(L, obs, gs, team_rew, term, trunc, am, atg, dc, tac,
                  atk_r, atk_b, bdy_r, bdy_b, bdy_alt_r, bdy_alt_b,
                  bdy_xy_r, bdy_xy_b, col_r, col_b, red_rc,
                  red_succ, blue_succ, geom, cd, rc, oc, ep_arrs):
    return VectorStepResult3v3(
        observations=obs, global_states=gs, team_rewards=team_rew,
        terminated=term, truncated=trunc, alive_masks=am,
        attack_targets=atg, step_death_causes=dc, team_alive_counts=tac,
        step_red_attack_kills=atk_r, step_blue_attack_kills=atk_b,
        step_red_boundary_deaths=bdy_r, step_blue_boundary_deaths=bdy_b,
        step_red_boundary_altitude_deaths=bdy_alt_r,
        step_blue_boundary_altitude_deaths=bdy_alt_b,
        step_red_boundary_xy_deaths=bdy_xy_r,
        step_blue_boundary_xy_deaths=bdy_xy_b,
        step_red_collision_deaths=col_r, step_blue_collision_deaths=col_b,
        red_reward_components=red_rc,
        red_complete_elimination_success=red_succ,
        blue_complete_elimination_success=blue_succ,
        geometry=geom, control_diagnostics=cd,
        termination_reason_codes=rc, outcome_codes=oc,
        episode_valid=ep_arrs["ep_valid"],
        episode_red_attack_kills=ep_arrs["ep_rak"],
        episode_blue_attack_kills=ep_arrs["ep_bak"],
        episode_red_survivors=ep_arrs["ep_rs"],
        episode_blue_survivors=ep_arrs["ep_bs"],
        episode_red_attack_deaths=ep_arrs["ep_rad"],
        episode_blue_attack_deaths=ep_arrs["ep_bad"],
        episode_red_boundary_deaths=ep_arrs["ep_rbd"],
        episode_blue_boundary_deaths=ep_arrs["ep_bbd"],
        episode_red_boundary_altitude_deaths=ep_arrs["ep_rbad"],
        episode_blue_boundary_altitude_deaths=ep_arrs["ep_bbad"],
        episode_red_boundary_xy_deaths=ep_arrs["ep_rbxy"],
        episode_blue_boundary_xy_deaths=ep_arrs["ep_bbxy"],
        episode_red_friendly_collision_deaths=ep_arrs["ep_rfc"],
        episode_blue_friendly_collision_deaths=ep_arrs["ep_bfc"],
        episode_red_cross_collision_deaths=ep_arrs["ep_rcc"],
        episode_blue_cross_collision_deaths=ep_arrs["ep_bcc"],
        episode_length=ep_arrs["ep_len"],
    )


# ===================================================================
# Worker
# ===================================================================

def _worker_main_3v3(conn, env_config, num_local_envs, worker_id):
    envs = [Homogeneous3v3AirCombatEnv(env_config) for _ in range(num_local_envs)]
    act_cfg = envs[0].config["action"]
    blue_policies = [NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
        for _ in range(num_local_envs)]
    red_policies = [NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
        for _ in range(num_local_envs)]
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset_all":
                conn.send(_worker_reset(envs, blue_policies, red_policies, payload))
            elif cmd == "step":
                conn.send(_worker_step(envs, blue_policies, payload))
            elif cmd == "step_rules":
                conn.send(_worker_step_rules(envs, blue_policies, red_policies, payload))
            elif cmd == "reset_at":
                conn.send(_worker_reset_at(envs, blue_policies, red_policies, payload))
            elif cmd == "close":
                break
            else:
                conn.send(("error", f"unknown: {cmd}", ""))
    except Exception:
        conn.send(("error", traceback.format_exc(), ""))
    finally:
        conn.close()


def _worker_reset(envs, blue_policies, red_policies, specs):
    L = len(envs)
    obs, gs, am = np.empty((L, 6, OBS_DIM), np.float32), np.empty((L, GS_DIM), np.float32), np.empty((L, 6), np.float32)
    for i, (env, spec) in enumerate(zip(envs, specs)):
        o, info = env.reset(int(spec["seed"]))
        for j, aid in enumerate(AGENT_ORDER): obs[i, j] = o[aid]
        gs[i] = info["global_state"]
        for j, a in enumerate(env.aircraft): am[i, j] = 1.0 if a.state.alive else 0.0
        blue_policies[i].reset_counters()
        red_policies[i].reset_counters()
    return obs, gs, am


def _worker_step(envs, blue_policies, red_actions):
    L = len(envs)
    obs, gs = np.empty((L, 6, OBS_DIM), np.float32), np.empty((L, GS_DIM), np.float32)
    team_rew = np.empty(L, np.float32); term = np.empty(L, bool); trunc = np.empty(L, bool)
    am = np.empty((L, 6), np.float32); atg = np.zeros((L, 6), np.int8); dc = np.zeros((L, 6), np.int8)
    tac = np.empty((L, 2), np.int8); atk_r = np.zeros(L, np.int8); atk_b = np.zeros(L, np.int8)
    bdy_r = np.zeros(L, np.int8); bdy_b = np.zeros(L, np.int8)
    bdy_alt_r = np.zeros(L, np.int8); bdy_alt_b = np.zeros(L, np.int8)
    bdy_xy_r = np.zeros(L, np.int8); bdy_xy_b = np.zeros(L, np.int8)
    col_r = np.zeros(L, np.int8); col_b = np.zeros(L, np.int8)
    red_rc = np.zeros((L, len(RED_REWARD_COMPONENT_KEYS_3V3)), np.float32)
    red_succ = np.zeros(L, bool); blue_succ = np.zeros(L, bool)
    geom = np.zeros((L, 6, 3), np.float32); cd = np.zeros((L, 6, K), np.float32)
    rc = np.empty(L, np.int8); oc = np.empty(L, np.int8)
    ep_arrs = _make_episode_arrays(L)

    for i, (env, ra) in enumerate(zip(envs, red_actions)):
        reds = [a for a in env.aircraft if a.team == "red"]
        blues = [a for a in env.aircraft if a.team == "blue"]
        ba, _ = blue_policies[i].select_actions(blues, reds)
        actions = {}
        for j, aid in enumerate(RED_IDS): actions[aid] = np.asarray(ra[j], np.float32)
        for aid in BLUE_IDS: actions[aid] = np.asarray(ba.get(aid, np.zeros(3, np.float32)), np.float32)
        o, rewards, t, tr, info = env.step(actions)
        for j, aid in enumerate(AGENT_ORDER):
            obs[i, j] = o[aid]
            a = env._aircraft_by_id(aid); am[i, j] = 1.0 if a.state.alive else 0.0
            tg = info["attacks"].get(aid); atg[i, j] = AGENT_ORDER.index(tg) if tg in AGENT_ORDER else -1
            dc[i, j] = int(info["death_causes"].get(aid, 0))
            cd[i, j] = _extract_diagnostics_3v3(info["control_diagnostics"].get(aid, {}))
        _fill_geom(info, env, i, geom)
        _fill_red_reward_components(info, i, red_rc)
        gs[i] = info["global_state"]; team_rew[i] = rewards["red_0"]
        term[i] = t; trunc[i] = tr
        tac[i, 0] = info["red_alive_count"]; tac[i, 1] = info["blue_alive_count"]
        atk_r[i] = info["attack_kills"]["red"]; atk_b[i] = info["attack_kills"]["blue"]
        bdy_r[i] = info["boundary_deaths"]["red"]; bdy_b[i] = info["boundary_deaths"]["blue"]
        bdy_alt_r[i] = info["boundary_altitude_deaths"]["red"]
        bdy_alt_b[i] = info["boundary_altitude_deaths"]["blue"]
        bdy_xy_r[i] = info["boundary_xy_deaths"]["red"]
        bdy_xy_b[i] = info["boundary_xy_deaths"]["blue"]
        col_r[i] = info["collision_deaths"]["red"]; col_b[i] = info["collision_deaths"]["blue"]
        red_succ[i] = info["red_complete_elimination_success"]
        blue_succ[i] = info["blue_complete_elimination_success"]
        rc[i] = encode_3v3_termination_reason(info["termination_reason"])
        oc[i] = encode_3v3_outcome(info["outcome"])
        _episode_fields.arrays = ep_arrs; _episode_fields(info, L, i)
    return _build_result(L, obs, gs, team_rew, term, trunc, am, atg, dc, tac,
                         atk_r, atk_b, bdy_r, bdy_b, bdy_alt_r, bdy_alt_b,
                         bdy_xy_r, bdy_xy_b, col_r, col_b, red_rc,
                         red_succ, blue_succ, geom, cd, rc, oc, ep_arrs)


def _worker_step_rules(envs, blue_policies, red_policies, modes):
    """modes: [L, 2] int8 – 0=zero, 1=pursuit for [red, blue]"""
    L = len(envs)
    obs, gs = np.empty((L, 6, OBS_DIM), np.float32), np.empty((L, GS_DIM), np.float32)
    team_rew = np.empty(L, np.float32); term = np.empty(L, bool); trunc = np.empty(L, bool)
    am = np.empty((L, 6), np.float32); atg = np.zeros((L, 6), np.int8); dc = np.zeros((L, 6), np.int8)
    tac = np.empty((L, 2), np.int8); atk_r = np.zeros(L, np.int8); atk_b = np.zeros(L, np.int8)
    bdy_r = np.zeros(L, np.int8); bdy_b = np.zeros(L, np.int8)
    bdy_alt_r = np.zeros(L, np.int8); bdy_alt_b = np.zeros(L, np.int8)
    bdy_xy_r = np.zeros(L, np.int8); bdy_xy_b = np.zeros(L, np.int8)
    col_r = np.zeros(L, np.int8); col_b = np.zeros(L, np.int8)
    red_rc = np.zeros((L, len(RED_REWARD_COMPONENT_KEYS_3V3)), np.float32)
    red_succ = np.zeros(L, bool); blue_succ = np.zeros(L, bool)
    geom = np.zeros((L, 6, 3), np.float32); cd = np.zeros((L, 6, K), np.float32)
    rc = np.empty(L, np.int8); oc = np.empty(L, np.int8)
    ep_arrs = _make_episode_arrays(L)

    for i, (env, mode) in enumerate(zip(envs, modes)):
        reds = [a for a in env.aircraft if a.team == "red"]
        blues = [a for a in env.aircraft if a.team == "blue"]
        r_m, b_m = int(mode[0]), int(mode[1])
        # Red actions
        if r_m == 0:
            ra = {a.aircraft_id: np.zeros(3, np.float32) for a in reds if a.state.alive}
        else:
            ra, _ = red_policies[i].select_actions(reds, blues)
        # Blue actions
        if b_m == 0:
            ba = {a.aircraft_id: np.zeros(3, np.float32) for a in blues if a.state.alive}
        else:
            ba, _ = blue_policies[i].select_actions(blues, reds)
        actions = {}
        for aid in RED_IDS: actions[aid] = np.asarray(ra.get(aid, np.zeros(3, np.float32)), np.float32)
        for aid in BLUE_IDS: actions[aid] = np.asarray(ba.get(aid, np.zeros(3, np.float32)), np.float32)
        o, rewards, t, tr, info = env.step(actions)
        for j, aid in enumerate(AGENT_ORDER):
            obs[i, j] = o[aid]
            a = env._aircraft_by_id(aid); am[i, j] = 1.0 if a.state.alive else 0.0
            tg = info["attacks"].get(aid); atg[i, j] = AGENT_ORDER.index(tg) if tg in AGENT_ORDER else -1
            dc[i, j] = int(info["death_causes"].get(aid, 0))
            cd[i, j] = _extract_diagnostics_3v3(info["control_diagnostics"].get(aid, {}))
        _fill_geom(info, env, i, geom)
        _fill_red_reward_components(info, i, red_rc)
        gs[i] = info["global_state"]; team_rew[i] = rewards["red_0"]
        term[i] = t; trunc[i] = tr
        tac[i, 0] = info["red_alive_count"]; tac[i, 1] = info["blue_alive_count"]
        atk_r[i] = info["attack_kills"]["red"]; atk_b[i] = info["attack_kills"]["blue"]
        bdy_r[i] = info["boundary_deaths"]["red"]; bdy_b[i] = info["boundary_deaths"]["blue"]
        bdy_alt_r[i] = info["boundary_altitude_deaths"]["red"]
        bdy_alt_b[i] = info["boundary_altitude_deaths"]["blue"]
        bdy_xy_r[i] = info["boundary_xy_deaths"]["red"]
        bdy_xy_b[i] = info["boundary_xy_deaths"]["blue"]
        col_r[i] = info["collision_deaths"]["red"]; col_b[i] = info["collision_deaths"]["blue"]
        red_succ[i] = info["red_complete_elimination_success"]
        blue_succ[i] = info["blue_complete_elimination_success"]
        rc[i] = encode_3v3_termination_reason(info["termination_reason"])
        oc[i] = encode_3v3_outcome(info["outcome"])
        _episode_fields.arrays = ep_arrs; _episode_fields(info, L, i)
    return _build_result(L, obs, gs, team_rew, term, trunc, am, atg, dc, tac,
                         atk_r, atk_b, bdy_r, bdy_b, bdy_alt_r, bdy_alt_b,
                         bdy_xy_r, bdy_xy_b, col_r, col_b, red_rc,
                         red_succ, blue_succ, geom, cd, rc, oc, ep_arrs)


def _worker_reset_at(envs, blue_policies, red_policies, payload):
    indices, specs = payload
    L = len(indices)
    obs, gs, am = np.empty((L, 6, OBS_DIM), np.float32), np.empty((L, GS_DIM), np.float32), np.empty((L, 6), np.float32)
    for j, idx in enumerate(indices):
        i = int(idx); env = envs[i]
        o, info = env.reset(int(specs[j]["seed"]))
        for k, aid in enumerate(AGENT_ORDER): obs[j, k] = o[aid]
        gs[j] = info["global_state"]
        for k, a in enumerate(env.aircraft): am[j, k] = 1.0 if a.state.alive else 0.0
        blue_policies[i].reset_counters()
        red_policies[i].reset_counters()
    return obs, gs, am


# ===================================================================
# Local backend
# ===================================================================

class LocalCombatVectorEnv3v3:
    def __init__(self, env_config, num_envs):
        self.num_envs = num_envs
        self.envs = [Homogeneous3v3AirCombatEnv(env_config) for _ in range(num_envs)]
        act_cfg = self.envs[0].config["action"]
        self.blue_policies = [NearestTargetPursuitPolicy3v3(
            act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
            for _ in range(num_envs)]
        self.red_policies = [NearestTargetPursuitPolicy3v3(
            act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
            for _ in range(num_envs)]
        self._closed = False

    def reset(self, specs):
        return _worker_reset(self.envs, self.blue_policies, self.red_policies, specs)

    def step(self, red_actions):
        return _worker_step(self.envs, self.blue_policies, red_actions)

    def step_rules(self, modes):
        return _worker_step_rules(self.envs, self.blue_policies, self.red_policies, modes)

    def reset_at(self, indices, specs):
        return _worker_reset_at(self.envs, self.blue_policies, self.red_policies,
                                (np.asarray(indices, dtype=np.int32), specs))

    def close(self): self._closed = True
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ===================================================================
# Subprocess backend
# ===================================================================

class SubprocessCombatVectorEnv3v3:
    def __init__(self, env_config, num_envs, num_env_workers):
        if num_env_workers < 2: raise ValueError("use LocalCombatVectorEnv3v3")
        if num_envs % num_env_workers != 0: raise ValueError("num_envs not divisible")
        self.num_envs = num_envs; self.num_env_workers = num_env_workers
        self.envs_per_worker = num_envs // num_env_workers; self._closed = False
        ctx = multiprocessing.get_context("spawn")
        self._conns, self._workers = [], []
        for w in range(num_env_workers):
            pc, cc = ctx.Pipe()
            p = ctx.Process(target=_worker_main_3v3, args=(cc, str(env_config), self.envs_per_worker, w))
            p.start(); cc.close()
            self._conns.append(pc); self._workers.append(p)

    def _sl(self, w):
        s = w * self.envs_per_worker; return slice(s, s + self.envs_per_worker)
    def _check(self):
        if self._closed: raise RuntimeError("closed")
    def _cerr(self, r, w):
        if isinstance(r, tuple) and len(r) == 3 and isinstance(r[0], str) and r[0] == "error":
            raise RuntimeError(f"Worker {w} error:\n{r[1]}")

    def _fan_out_in(self, cmd, payload):
        self._check()
        for w in range(self.num_env_workers):
            sl = self._sl(w)
            chunk = payload[sl] if isinstance(payload, (np.ndarray, list)) else payload
            self._conns[w].send((cmd, chunk))
        ps = [self._conns[w].recv() for w in range(self.num_env_workers)]
        for w, r in enumerate(ps): self._cerr(r, w)
        return ps

    def reset(self, specs):
        ps = self._fan_out_in("reset_all", list(specs))
        return tuple(np.concatenate([p[i] for p in ps], axis=0) for i in range(3))

    def step(self, red_actions):
        ps = self._fan_out_in("step", red_actions)
        return self._combine_result(ps)

    def step_rules(self, modes):
        ps = self._fan_out_in("step_rules", modes)
        return self._combine_result(ps)

    def _combine_result(self, ps):
        fields = VectorStepResult3v3._fields
        parts = [{f: p[i] for i, f in enumerate(fields)} for p in ps]
        return VectorStepResult3v3(
            **{f: np.concatenate([pp[f] for pp in parts], axis=0) for f in fields})

    def reset_at(self, gi, specs):
        self._check()
        grp = {}
        for j, g in enumerate(gi):
            w = int(g) // self.envs_per_worker; loc = int(g) - w * self.envs_per_worker
            grp.setdefault(w, ([], [])); grp[w][0].append(loc); grp[w][1].append(specs[j])
        rm = {}
        for w, (li, ws) in grp.items():
            self._conns[w].send(("reset_at", (np.asarray(li, dtype=np.int32), ws)))
        for w in grp:
            r = self._conns[w].recv(); self._cerr(r, w)
            for j, g in enumerate([g for g in gi if int(g) // self.envs_per_worker == w]):
                rm[int(g)] = (r[0][j:j+1], r[1][j:j+1], r[2][j:j+1])
        return tuple(np.concatenate([rm[int(g)][i] for g in gi], axis=0) for i in range(3))

    def close(self):
        if self._closed: return
        self._closed = True
        for c in self._conns:
            try: c.send(("close", None))
            except (OSError, BrokenPipeError): pass
        for w in self._workers: w.join(timeout=5.0); (w.is_alive() and w.terminate())
        for c in self._conns:
            try: c.close()
            except OSError: pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()


def make_combat_vector_env_3v3(env_config, num_envs, num_env_workers=4):
    if num_env_workers < 1: raise ValueError(">=1")
    if num_env_workers > num_envs: raise ValueError("workers <= envs")
    if num_env_workers == 1: return LocalCombatVectorEnv3v3(env_config, num_envs)
    if num_envs % num_env_workers: raise ValueError("num_envs divisible")
    return SubprocessCombatVectorEnv3v3(env_config, num_envs, num_env_workers)
