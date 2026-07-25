"""Persistent multi-process vector env for 3v3 – workers auto-generate blue actions."""
from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..environment_3v3 import GS_DIM, OBS_DIM, Homogeneous3v3AirCombatEnv
from ..rule_policy_3v3 import NearestTargetPursuitPolicy3v3
from ..scenario_3v3 import BLUE_IDS, RED_IDS
from .vector_env import (
    CONTROL_DIAGNOSTIC_KEYS,
    K,
    encode_outcome,
    encode_termination_reason,
)

AGENT_ORDER = RED_IDS + BLUE_IDS


def _extract_diagnostics_3v3(diag):
    vec = np.empty(K, dtype=np.float32)
    for i, key in enumerate(CONTROL_DIAGNOSTIC_KEYS):
        val = diag.get(key, 0.0)
        vec[i] = float(val) if not isinstance(val, (bool, np.bool_)) else float(val)
    return vec


# ===================================================================
# Worker
# ===================================================================


def _worker_main_3v3(conn, env_config, num_local_envs, worker_id):
    envs = [Homogeneous3v3AirCombatEnv(env_config) for _ in range(num_local_envs)]
    # Each worker creates its own blue policy
    act_cfg = envs[0].config["action"]
    blue_policy = NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset_all":
                conn.send(_worker_reset(envs, payload))
            elif cmd == "step":
                conn.send(_worker_step(envs, payload, blue_policy))
            elif cmd == "reset_at":
                conn.send(_worker_reset_at(envs, payload))
            elif cmd == "close":
                break
            else:
                conn.send(("error", f"unknown: {cmd}", ""))
    except Exception:
        conn.send(("error", traceback.format_exc(), ""))
    finally:
        conn.close()


def _worker_reset(envs, specs):
    L = len(envs)
    obs = np.empty((L, 6, OBS_DIM), dtype=np.float32)
    gs = np.empty((L, GS_DIM), dtype=np.float32)
    am = np.empty((L, 6), dtype=np.float32)
    for i, (env, spec) in enumerate(zip(envs, specs)):
        o, info = env.reset(int(spec["seed"]))
        for j, aid in enumerate(AGENT_ORDER):
            obs[i, j] = o[aid]
        gs[i] = info["global_state"]
        for j, a in enumerate(env.aircraft):
            am[i, j] = 1.0 if a.state.alive else 0.0
    return obs, gs, am


def _worker_step(envs, red_actions, blue_policy):
    """red_actions: [L, 3, 3] for red_0, red_1, red_2. Blue generated internally."""
    L = len(envs)
    obs = np.empty((L, 6, OBS_DIM), dtype=np.float32)
    gs = np.empty((L, GS_DIM), dtype=np.float32)
    team_rew = np.empty(L, dtype=np.float32)
    term = np.empty(L, dtype=bool)
    trunc = np.empty(L, dtype=bool)
    am = np.empty((L, 6), dtype=np.float32)
    atg = np.zeros((L, 6), dtype=np.int8)
    dc = np.zeros((L, 6), dtype=np.int8)
    tac = np.empty((L, 2), dtype=np.int8)
    atk_red = np.zeros(L, dtype=np.int8)
    atk_blue = np.zeros(L, dtype=np.int8)
    bdy_red = np.zeros(L, dtype=np.int8)
    bdy_blue = np.zeros(L, dtype=np.int8)
    col_red = np.zeros(L, dtype=np.int8)
    col_blue = np.zeros(L, dtype=np.int8)
    red_succ = np.zeros(L, dtype=bool)
    blue_succ = np.zeros(L, dtype=bool)
    geom = np.zeros((L, 6, 3), dtype=np.float32)
    cd = np.zeros((L, 6, K), dtype=np.float32)
    rc = np.empty(L, dtype=np.int8)
    oc = np.empty(L, dtype=np.int8)

    for i, (env, ra) in enumerate(zip(envs, red_actions)):
        reds = [a for a in env.aircraft if a.team == "red"]
        blues = [a for a in env.aircraft if a.team == "blue"]
        blue_acts, _ = blue_policy.select_actions(blues, reds)
        actions = {}
        for j, aid in enumerate(RED_IDS):
            actions[aid] = np.asarray(ra[j], dtype=np.float32)
        for aid in BLUE_IDS:
            actions[aid] = np.asarray(blue_acts.get(aid, np.zeros(3, dtype=np.float32)), dtype=np.float32)

        o, rewards, t, tr, info = env.step(actions)
        for j, aid in enumerate(AGENT_ORDER):
            obs[i, j] = o[aid]
            a = env._aircraft_by_id(aid)
            am[i, j] = 1.0 if a.state.alive else 0.0
            tgt = info["attacks"].get(aid)
            atg[i, j] = AGENT_ORDER.index(tgt) if tgt in AGENT_ORDER else -1
            dc[i, j] = int(info["death_causes"].get(aid, 0))
            cdi = info["control_diagnostics"].get(aid, {})
            cd[i, j] = _extract_diagnostics_3v3(cdi)

        gs[i] = info["global_state"]
        team_rew[i] = rewards["red_0"]
        term[i] = t; trunc[i] = tr
        tac[i, 0] = info["red_alive_count"]; tac[i, 1] = info["blue_alive_count"]
        atk_red[i] = info["attack_kills"]["red"]
        atk_blue[i] = info["attack_kills"]["blue"]
        bdy_red[i] = info["boundary_deaths"]["red"]
        bdy_blue[i] = info["boundary_deaths"]["blue"]
        col_red[i] = info["collision_deaths"]["red"]
        col_blue[i] = info["collision_deaths"]["blue"]
        red_succ[i] = info["red_complete_elimination_success"]
        blue_succ[i] = info["blue_complete_elimination_success"]
        rc[i] = encode_termination_reason(info["termination_reason"])
        oc[i] = encode_outcome(info["outcome"])

    return (obs, gs, team_rew, term, trunc, am, atg, dc, tac,
            atk_red, atk_blue, bdy_red, bdy_blue, col_red, col_blue,
            red_succ, blue_succ, geom, cd, rc, oc)


def _worker_reset_at(envs, payload):
    indices, specs = payload
    L = len(indices)
    obs = np.empty((L, 6, OBS_DIM), dtype=np.float32)
    gs = np.empty((L, GS_DIM), dtype=np.float32)
    am = np.empty((L, 6), dtype=np.float32)
    for j, idx in enumerate(indices):
        i = int(idx); env = envs[i]
        o, info = env.reset(int(specs[j]["seed"]))
        for k, aid in enumerate(AGENT_ORDER):
            obs[j, k] = o[aid]
        gs[j] = info["global_state"]
        for k, a in enumerate(env.aircraft):
            am[j, k] = 1.0 if a.state.alive else 0.0
    return obs, gs, am


# ===================================================================
# Local backend
# ===================================================================


class LocalCombatVectorEnv3v3:
    def __init__(self, env_config, num_envs):
        self.num_envs = num_envs
        self.envs = [Homogeneous3v3AirCombatEnv(env_config) for _ in range(num_envs)]
        act_cfg = self.envs[0].config["action"]
        self.blue_policy = NearestTargetPursuitPolicy3v3(
            act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"])
        self._closed = False

    def reset(self, specs):
        return _worker_reset(self.envs, specs)

    def step(self, red_actions):
        return _worker_step(self.envs, red_actions, self.blue_policy)

    def reset_at(self, indices, specs):
        return _worker_reset_at(self.envs, (np.asarray(indices, dtype=np.int32), specs))

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ===================================================================
# Subprocess backend
# ===================================================================


class SubprocessCombatVectorEnv3v3:
    def __init__(self, env_config, num_envs, num_env_workers):
        if num_env_workers < 2:
            raise ValueError("use LocalCombatVectorEnv3v3")
        if num_envs % num_env_workers != 0:
            raise ValueError(f"num_envs ({num_envs}) not divisible by num_env_workers ({num_env_workers})")
        self.num_envs = num_envs
        self.num_env_workers = num_env_workers
        self.envs_per_worker = num_envs // num_env_workers
        self._closed = False
        ctx = multiprocessing.get_context("spawn")
        self._conns = []
        self._workers = []
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

    def reset(self, specs):
        self._check()
        for w in range(self.num_env_workers):
            self._conns[w].send(("reset_all", specs[self._sl(w)]))
        ps = [self._conns[w].recv() for w in range(self.num_env_workers)]
        for w, r in enumerate(ps): self._cerr(r, w)
        return tuple(np.concatenate([p[i] for p in ps], axis=0) for i in range(3))

    def step(self, red_actions):
        self._check()
        for w in range(self.num_env_workers):
            self._conns[w].send(("step", red_actions[self._sl(w)]))
        ps = [self._conns[w].recv() for w in range(self.num_env_workers)]
        for w, r in enumerate(ps): self._cerr(r, w)
        return tuple(np.concatenate([p[i] for p in ps], axis=0) for i in range(21))

    def reset_at(self, gi, specs):
        self._check()
        grp = {}
        for j, g in enumerate(gi):
            w = int(g) // self.envs_per_worker
            loc = int(g) - w * self.envs_per_worker
            grp.setdefault(w, ([], []))
            grp[w][0].append(loc); grp[w][1].append(specs[j])
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
        for w in self._workers:
            w.join(timeout=5.0)
            if w.is_alive(): w.terminate()
        for c in self._conns:
            try: c.close()
            except OSError: pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()


def make_combat_vector_env_3v3(env_config, num_envs, num_env_workers=4):
    if num_env_workers < 1: raise ValueError("num_env_workers >= 1")
    if num_env_workers > num_envs: raise ValueError("num_env_workers <= num_envs")
    if num_env_workers == 1: return LocalCombatVectorEnv3v3(env_config, num_envs)
    if num_envs % num_env_workers != 0: raise ValueError("num_envs divisible by num_env_workers")
    return SubprocessCombatVectorEnv3v3(env_config, num_envs, num_env_workers)
