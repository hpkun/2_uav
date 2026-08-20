"""Lightweight heterogeneous 1 MAV + 2 UAV versus 2 Blue air-combat env."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import yaml


@dataclass(frozen=True)
class AircraftSpec:
    aircraft_type: str
    v_min: float
    v_max: float
    nx_min: float
    nx_max: float
    ny_min: float
    ny_max: float
    nz_min: float
    nz_max: float
    sensor_range: float = 6000.0


MAVSpec = AircraftSpec("MAV", 120.0, 300.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
UAVSpec = AircraftSpec("UAV", 100.0, 220.0, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0)
BlueSpec = AircraftSpec("Blue", 150.0, 320.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)


@dataclass
class State:
    x: float; y: float; h: float; v: float; theta: float; psi: float; alive: bool = True

    def copy(self) -> "State":
        return State(self.x, self.y, self.h, self.v, self.theta, self.psi, self.alive)


@dataclass
class Entity:
    aircraft_id: str
    team: str
    spec: AircraftSpec
    state: State


def _geometry(a: State, b: State) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    rel = np.array([b.x-a.x, b.y-a.y, b.h-a.h], dtype=float)
    va = np.array([a.v*np.cos(a.theta)*np.cos(a.psi), a.v*np.cos(a.theta)*np.sin(a.psi), a.v*np.sin(a.theta)])
    vb = np.array([b.v*np.cos(b.theta)*np.cos(b.psi), b.v*np.cos(b.theta)*np.sin(b.psi), b.v*np.sin(b.theta)])
    dist = float(np.linalg.norm(rel)); line = rel / max(dist, 1e-9)
    forward_a = np.array([np.cos(a.theta)*np.cos(a.psi), np.cos(a.theta)*np.sin(a.psi), np.sin(a.theta)])
    forward_b = np.array([np.cos(b.theta)*np.cos(b.psi), np.cos(b.theta)*np.sin(b.psi), np.sin(b.theta)])
    ata = float(np.arccos(np.clip(np.dot(forward_a, line), -1.0, 1.0)))
    aa = float(np.arccos(np.clip(np.dot(forward_b, line), -1.0, 1.0)))
    return dist, ata, aa, float(np.linalg.norm(vb-va)), rel, vb-va


class HeterogeneousMAVUAVAirCombatEnv:
    """Three red agents (MAV, UAV, UAV) and a deterministic two-Blue opponent."""
    red_ids = ("MAV", "UAV1", "UAV2")
    blue_ids = ("Blue1", "Blue2")

    def __init__(self, config_path: str | Path = "configs/heterogeneous_mavuav_3v2.yaml", dt: float | None = None, max_steps: int | None = None, seed: int | None = None):
        with Path(config_path).open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        simulation = self.config["simulation"]
        self.dt = float(simulation["dt"] if dt is None else dt)
        self.max_steps = int(simulation["max_steps"] if max_steps is None else max_steps)
        self.rng = np.random.default_rng(seed)
        self.entities: dict[str, Entity] = {}
        self.step_count = 0
        self._attack_streak: dict[tuple[str, str], int] = {}
        self._running = False

    @property
    def agents(self) -> list[str]:
        return list(self.red_ids)

    def reset(self, seed: int | None = None, **_: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None: self.rng = np.random.default_rng(seed)
        spec_cfg = self.config["aircraft_specs"]
        def make_spec(name: str) -> AircraftSpec:
            cfg = spec_cfg[name]
            return AircraftSpec(name, cfg["v_min"], cfg["v_max"], *cfg["nx"], *cfg["ny"], *cfg["nz"], cfg["sensor_range"])
        by_type = {name: make_spec(name) for name in ("MAV", "UAV", "Blue")}
        types = {"MAV": "MAV", "UAV1": "UAV", "UAV2": "UAV", "Blue1": "Blue", "Blue2": "Blue"}
        scenario = self.config["scenario"]
        self.entities = {}
        for aid, aircraft_type in types.items():
            spec = by_type[aircraft_type]
            start = scenario["initial"][aid]
            state = State(*start["position"], start["speed"], 0.0, np.deg2rad(start["heading_deg"]))
            self.entities[aid] = Entity(aid, "red" if aid in self.red_ids else "blue", spec, state)
        self.step_count = 0; self._attack_streak.clear(); self._running = True
        return self._observations(), {"agents": self.agents, "step_count": 0, "attacks": []}

    def step(self, actions: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray):
        if not self._running: raise RuntimeError("reset() must be called before step()")
        if not isinstance(actions, dict): actions = {aid: actions[i] for i, aid in enumerate(self.red_ids)}
        for aid in self.red_ids:
            if aid not in actions: raise KeyError(f"missing action for {aid}")
        all_actions = {aid: np.zeros(3) for aid in self.blue_ids}
        all_actions.update(actions)
        self._blue_actions(all_actions)
        for aid, entity in self.entities.items():
            if entity.state.alive:
                self._integrate(entity, np.asarray(all_actions[aid], dtype=float))
        attacks, killed = self._resolve_attacks()
        self.step_count += 1
        red_alive = all(self.entities[aid].state.alive for aid in ("MAV",))
        blue_alive = any(self.entities[aid].state.alive for aid in self.blue_ids)
        terminated, outcome = False, None
        if not red_alive: terminated, outcome = True, "blue"
        elif not blue_alive: terminated, outcome = True, "red"
        truncated = not terminated and self.step_count >= self.max_steps
        if truncated: outcome = "draw"
        rewards = self._red_rewards(attacks, killed, outcome)
        self._running = not (terminated or truncated)
        info = {"agents": self.agents, "step_count": self.step_count, "attacks": attacks, "killed": killed, "outcome": outcome}
        return self._observations(), rewards, terminated, truncated, info

    def _integrate(self, e: Entity, action: np.ndarray) -> None:
        if action.shape != (3,):
            raise ValueError(f"action for {e.aircraft_id} must have shape (3,), got {action.shape}")
        u = np.clip(action, -1.0, 1.0); s, g, dt = e.state, 9.81, self.dt
        nx = np.interp(u[0], (-1,1), (e.spec.nx_min,e.spec.nx_max)); ny = np.interp(u[1], (-1,1), (e.spec.ny_min,e.spec.ny_max)); nz = np.interp(u[2], (-1,1), (e.spec.nz_min,e.spec.nz_max))
        s.x += dt*s.v*np.cos(s.theta)*np.cos(s.psi); s.y += dt*s.v*np.cos(s.theta)*np.sin(s.psi); s.h += dt*s.v*np.sin(s.theta)
        s.v = float(np.clip(s.v + dt*g*(nx-np.sin(s.theta)), e.spec.v_min, e.spec.v_max))
        s.theta += dt*g/max(s.v, 1e-6)*(ny-np.cos(s.theta)); s.psi += dt*g/(max(s.v,1e-6)*max(abs(np.cos(s.theta)),1e-3))*nz

    def _blue_actions(self, actions: dict[str, np.ndarray]) -> None:
        for aid in self.blue_ids:
            b = self.entities[aid]
            targets = [self.entities[x] for x in self.red_ids if self.entities[x].state.alive]
            if not b.state.alive or not targets: continue
            target = min(targets, key=lambda x: _geometry(b.state, x.state)[0])
            _, _, _, _, rel, _ = _geometry(b.state, target.state)
            candidates = [
                np.array([nx, ny, nz], dtype=float)
                for nx, ny, nz in ((0,0,0), (0,0,-1), (0,0,1), (1,0,0), (0,-1,0), (0,1,0))
            ]
            def score(candidate: np.ndarray) -> float:
                predicted = Entity("predicted", "blue", b.spec, b.state.copy())
                self._integrate(predicted, candidate)
                distance, ata, aa, *_ = _geometry(predicted.state, target.state)
                distance_score = np.exp(-((distance-2000.0)/1500.0)**2)
                return float(distance_score + np.cos(ata) + 0.5*np.cos(aa))
            actions[aid] = max(candidates, key=score)

    def _resolve_attacks(self) -> tuple[list[tuple[str,str]], list[str]]:
        attacks = []
        for attacker in self.entities.values():
            if not attacker.state.alive: continue
            targets = [x for x in self.entities.values() if x.team != attacker.team and x.state.alive]
            for target in targets:
                d, ata, aa, *_ = _geometry(attacker.state, target.state); key=(attacker.aircraft_id,target.aircraft_id)
                combat = self.config["combat"]
                inside = (combat["distance_min"] <= d <= combat["distance_max"] and ata < np.deg2rad(combat["ata_deg"]) and aa < np.deg2rad(combat["aa_deg"]))
                self._attack_streak[key] = self._attack_streak.get(key, 0)+1 if inside else 0
                if self._attack_streak[key] >= combat["hold_steps"]:
                    attacks.append(key); self._attack_streak[key]=0
        killed = sorted({target for _, target in attacks})
        for target in killed:
            self.entities[target].state.alive = False
        return attacks, killed

    def _red_rewards(self, attacks: list[tuple[str,str]], killed: list[str], outcome: str | None) -> dict[str, float]:
        cfg = self.config["reward"]
        situation_terms = []
        enemies = [self.entities[x] for x in self.blue_ids if self.entities[x].state.alive]
        for aid in self.red_ids:
            own = self.entities[aid]
            if not own.state.alive or not enemies:
                continue
            target = min(enemies, key=lambda x: _geometry(own.state, x.state)[0])
            distance, ata, aa, *_ = _geometry(own.state, target.state)
            distance_term = np.exp(-((distance-2000.0)/1500.0)**2)
            speed_term = np.clip((own.state.v-target.state.v)/200.0, -1.0, 1.0)
            situation_terms.append(0.1*(distance_term + np.cos(ata) + np.cos(aa) + speed_term))
        situation = float(np.mean(situation_terms)) if situation_terms else 0.0
        event = cfg["blue_kill"]*sum(target in self.blue_ids for target in killed)
        event += cfg["uav_death"]*sum(aid in killed for aid in ("UAV1", "UAV2"))
        event += cfg["mav_death"]*("MAV" in killed)
        mission = cfg["red_win"] + cfg["blue_eliminated"] if outcome == "red" else (cfg["mav_death_mission"] if outcome == "blue" else 0.0)
        total = float(situation + event + mission)
        return {aid: total for aid in self.red_ids}

    def _observations(self) -> dict[str, np.ndarray]:
        out = {}
        for aid in self.red_ids:
            own = self.entities[aid]; vals = [own.state.x/6000, own.state.y/6000, own.state.h/6000, own.state.v/400, own.state.theta/np.pi, own.state.psi/np.pi, 0.0 if aid=="MAV" else 1.0, float(own.state.alive)]
            friends = [self.entities[x] for x in self.red_ids if x != aid]; enemies = [self.entities[x] for x in self.blue_ids]
            for x in friends:
                d,_,_,_,rel,dv = _geometry(own.state,x.state); vals += [*rel/6000, d/6000, *dv/600, float(x.state.alive), 0.0 if x.aircraft_id=="MAV" else 1.0]
            for x in enemies:
                d,ata,aa,_,rel,_ = _geometry(own.state,x.state); vals += [*rel/6000, d/6000, ata/np.pi, aa/np.pi, float(x.state.alive)]
            out[aid] = np.asarray(vals, dtype=np.float32)
        return out
