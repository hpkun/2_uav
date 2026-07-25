"""Tests for 3v3 episode-level death ledger and attack kill accounting."""
from pathlib import Path
import numpy as np
import pytest
from uav_combat.environment_3v3 import (
    Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS,
    DEATH_NONE, DEATH_ATTACK, DEATH_BOUNDARY_XY, DEATH_BOUNDARY_ALTITUDE,
    DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS,
    _death_summary, _validate_death_ledger,
)
from uav_combat.geometry import compute_pairwise_geometry

CONFIG = Path(__file__).parents[1] / "configs" / "homogeneous_3v3.yaml"


def _set(env, aid, x, y, z=-3000.0, v=150.0, psi=0.0):
    a = env._aircraft_by_id(aid)
    a.state.x, a.state.y, a.state.z = x, y, z
    a.state.v, a.state.psi = v, psi
    a.state.theta = 0.0


def _kill(env, aid):
    env._aircraft_by_id(aid).state.alive = False


def _all_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft}


def _run_to_end(env, actions_fn=None):
    """Step until episode terminates, return episode_summary."""
    actions_fn = actions_fn or _all_actions
    while True:
        acts = actions_fn(env)
        obs, rewards, term, trunc, info = env.step(acts)
        if term or trunc:
            es = info.get("episode_summary")
            assert es is not None, "episode_summary must be present at episode end"
            return es


class TestDeathLedger:
    def test_initial_all_alive(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(42)
        summary = _death_summary({aid: DEATH_NONE for aid in RED_IDS + BLUE_IDS}, "red", RED_IDS)
        assert summary["survivors"] == 3
        assert summary["attack_deaths"] == 0
        assert summary["boundary_deaths"] == 0
        assert summary["collision_deaths"] == 0

    def test_single_attack_kill(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(42)
        _set(env, "red_0", 0, 0, -3000, psi=0)
        _set(env, "blue_0", 500, 0, -3000, psi=0)
        _set(env, "blue_1", 10000, 10000, -3000)
        _set(env, "blue_2", -10000, -10000, -3000)
        _set(env, "red_1", 10000, -10000, -3000)
        _set(env, "red_2", -10000, 10000, -3000)
        es = _run_to_end(env)
        assert es["blue_death_causes"]["attack_deaths"] >= 1, f"blue should have attack deaths: {es}"

    def test_focus_fire_kill_count_one(self):
        """Two reds attacking same blue yields red_attack_kills=1."""
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(44)
        _set(env, "red_0", 0, -40, -3000, psi=0)
        _set(env, "red_1", 0, 40, -3000, psi=0)
        _set(env, "blue_0", 500, 0, -3000, psi=0)
        _set(env, "red_2", -10000, -10000, -3000)
        _set(env, "blue_1", 10000, 10000, -3000)
        _set(env, "blue_2", -10000, 10000, -3000)
        es = _run_to_end(env)
        # blue.attack_deaths must be 1 (focused target dies once)
        assert es["red_attack_kills"] == 1, f"focus fire should yield 1 kill: {es}"
        assert es["blue_death_causes"]["attack_deaths"] == 1

    def test_three_kills_complete_elimination(self):
        """Three reds each kill one blue -> red complete elimination success."""
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(45)
        _set(env, "red_0", 0, -40, -3000, psi=0)
        _set(env, "red_1", 0, 0, -3000, psi=0)
        _set(env, "red_2", 0, 40, -3000, psi=0)
        _set(env, "blue_0", 500, -40, -3000, psi=0)
        _set(env, "blue_1", 500, 0, -3000, psi=0)
        _set(env, "blue_2", 500, 40, -3000, psi=0)
        es = _run_to_end(env)
        assert es["red_attack_kills"] == 3
        assert es["red_complete_elimination_success"], f"should be complete success: {es}"

    def test_boundary_death_not_attack_success(self):
        """Blue boundary death -> env outcome=red but NOT complete elimination success."""
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(46)
        # All blue will fly out of bounds
        for bid in BLUE_IDS:
            _set(env, bid, env.config["battlefield"]["x_limit"] - 10, 0, -3000, v=250, psi=0)
        _set(env, "red_0", 0, 0, -3000)
        _set(env, "red_1", 0, 40, -3000)
        _set(env, "red_2", 0, -40, -3000)

        def boundary_actions(e):
            acts = {}
            for a in e.aircraft:
                if a.state.alive:
                    if a.team == "blue":
                        acts[a.aircraft_id] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # full speed
                    else:
                        acts[a.aircraft_id] = np.zeros(3, dtype=np.float32)
            return acts
        es = _run_to_end(env, boundary_actions)
        assert es["environment_outcome"] == "red" or es["environment_outcome"] == "draw", f"outcome: {es}"
        # If outcome is red from boundary deaths, success must be False
        if es["environment_outcome"] == "red":
            assert not es["red_complete_elimination_success"], f"boundary elimination is not attack success: {es}"
        # Validate death ledger
        for team in ("red", "blue"):
            dc = es[f"{team}_death_causes"]
            total = es[f"{team}_survivors"] + dc["attack_deaths"] + dc["boundary_deaths"] + dc["collision_deaths"]
            assert total == 3, f"{team} ledger: {total} != 3 in {es}"

    def test_sync_mutual_kill_accounting(self):
        """Simultaneous kills: red_0 kills blue_0, blue_1 kills red_1."""
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(47)
        _set(env, "red_0", 0, -50, -3000, psi=0)
        _set(env, "blue_0", 500, -50, -3000, psi=0)
        _set(env, "blue_1", 0, 50, -3000, psi=0)
        _set(env, "red_1", 500, 50, -3000, psi=0)
        _set(env, "red_2", -10000, -10000, -3000)
        _set(env, "blue_2", 10000, 10000, -3000)
        es = _run_to_end(env)
        assert es["red_attack_kills"] >= 1
        assert es["blue_attack_kills"] >= 1
        # both should have at least one attack death
        assert es["red_death_causes"]["attack_deaths"] >= 1
        assert es["blue_death_causes"]["attack_deaths"] >= 1

    def test_death_ledger_always_balances(self):
        """For many random seeds, death ledger always sums to 3 per team."""
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        max_steps = env.config["simulation"]["max_steps"]
        for seed in range(20):
            env.reset(seed + 100)
            for _ in range(max_steps):
                alive = [a for a in env.aircraft if a.state.alive]
                if not alive:
                    break
                acts = {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in alive}
                obs, rewards, term, trunc, info = env.step(acts)
                if term or trunc:
                    es = info["episode_summary"]
                    for team in ("red", "blue"):
                        dc = es[f"{team}_death_causes"]
                        total = es[f"{team}_survivors"] + dc["attack_deaths"] + dc["boundary_deaths"] + dc["collision_deaths"]
                        assert total == 3, f"seed={seed} {team}: {total} != 3; summary={es}"
                    # Attack kill symmetry
                    assert es["red_attack_kills"] == es["blue_death_causes"]["attack_deaths"], \
                        f"seed={seed} red_attack_kills={es['red_attack_kills']} != blue.attack_deaths={es['blue_death_causes']['attack_deaths']}"
                    assert es["blue_attack_kills"] == es["red_death_causes"]["attack_deaths"], \
                        f"seed={seed} blue_attack_kills={es['blue_attack_kills']} != red.attack_deaths={es['red_death_causes']['attack_deaths']}"
                    break
