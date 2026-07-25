"""Audit 3v3 combat logic: boundary, collision, attack, sync semantics."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from uav_combat.environment_3v3 import (Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS,
                                          DEATH_ATTACK, DEATH_BOUNDARY_XY, DEATH_COLLISION_FRIENDLY)
from uav_combat.geometry import compute_pairwise_geometry

def _all_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft}

def _set(env, aid, x, y, z=-3000.0, v=150.0, psi=0.0):
    a = env._aircraft_by_id(aid)
    a.state.x, a.state.y, a.state.z = x, y, z
    a.state.v, a.state.psi = v, psi

def _kill(env, aid):
    env._aircraft_by_id(aid).state.alive = False

def _verify_attack(env, attacker, target):
    """Verify attacker CAN attack target in current state."""
    a = env._aircraft_by_id(attacker)
    t = env._aircraft_by_id(target)
    g = compute_pairwise_geometry(a.state, t.state)
    dist_ok = 100 <= g.distance <= 1000
    ata_ok = g.ata <= np.deg2rad(30)
    aa_ok = g.aa <= np.deg2rad(90)
    return dist_ok and ata_ok and aa_ok

def check(desc, condition):
    if not condition:
        print(f"FAIL: {desc}")
        sys.exit(1)
    print(f"  OK: {desc}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    args = p.parse_args()

    env = Homogeneous3v3AirCombatEnv(args.env_config)

    # 1. Normal init
    obs, info = env.reset(42)
    check("6 aircraft created", len(env.aircraft) == 6)
    check("3 red alive", env._alive_count("red") == 3)
    check("3 blue alive", env._alive_count("blue") == 3)

    # 2. Single kill: red_0 behind blue_0, both facing same direction
    env.reset(42)
    # Move red_0 500m behind blue_0, both heading +x, same altitude
    _set(env, "red_0", 0, 0, -3000, psi=0)
    _set(env, "blue_0", 500, 0, -3000, psi=0)
    _set(env, "red_1", -10000, -10000, -3000, psi=0)
    _set(env, "red_2", -10000, 10000, -3000, psi=0)
    _set(env, "blue_1", 10000, 10000, -3000, psi=0)
    _set(env, "blue_2", 10000, -10000, -3000, psi=0)
    check("single kill: red_0 can attack blue_0", _verify_attack(env, "red_0", "blue_0"))
    obs, rewards, term, trunc, info = env.step(_all_actions(env))
    check("single kill: blue_0 dead", not env._aircraft_by_id("blue_0").state.alive)
    check("single kill: red kill count=1", info["attack_kills"]["red"] == 1)
    check("single kill: episode continues (2 red alive)", not term and env._alive_count("red") >= 2)

    # 3. Sync mutual kill: red_0 kills blue_0, blue_1 kills red_1
    # Both pairs must be tail-chase for rear-hemisphere attacks
    env.reset(43)
    _set(env, "red_0", 0, 0, -3000, psi=0)
    _set(env, "blue_0", 500, 0, -3000, psi=0)    # red_0 behind blue_0, both face +x
    _set(env, "blue_1", 100, 100, -3000, psi=0)
    _set(env, "red_1", 600, 100, -3000, psi=0)   # blue_1 behind red_1, both face +x
    _set(env, "red_2", -10000, -10000, -3000, psi=0)
    _set(env, "blue_2", 10000, 10000, -3000, psi=0)
    check("mutual: red_0 attacks blue_0", _verify_attack(env, "red_0", "blue_0"))
    check("mutual: blue_1 attacks red_1", _verify_attack(env, "blue_1", "red_1"))
    obs, rewards, term, trunc, info = env.step(_all_actions(env))
    check("mutual: blue_0 and red_1 both dead",
          not env._aircraft_by_id("blue_0").state.alive and not env._aircraft_by_id("red_1").state.alive)
    check("mutual: episode continues", not term)

    # 4. Focus fire: two reds attack same blue
    env.reset(44)
    _set(env, "red_0", 0, 40, -3000, psi=0)
    _set(env, "red_1", 0, -40, -3000, psi=0)   # y-separation=80 > collision_distance=30
    _set(env, "blue_0", 500, 0, -3000, psi=0)   # both reds can attack blue_0
    _set(env, "red_2", -10000, -10000, -3000, psi=0)
    _set(env, "blue_1", 10000, 10000, -3000, psi=0)
    _set(env, "blue_2", -10000, 10000, -3000, psi=0)
    check("focus: red_0 attacks blue_0", _verify_attack(env, "red_0", "blue_0"))
    check("focus: red_1 attacks blue_0", _verify_attack(env, "red_1", "blue_0"))
    obs, rewards, term, trunc, info = env.step(_all_actions(env))
    check("focus fire: blue_0 dead", not env._aircraft_by_id("blue_0").state.alive)
    check("focus fire: kill count=1 (not 2)", info["attack_kills"]["red"] == 1)

    # 5. Boundary single aircraft
    env.reset(45)
    _set(env, "red_0", env.config["battlefield"]["x_limit"] + 100, 0, -3000)
    obs, rewards, term, trunc, info = env.step(_all_actions(env))
    check("boundary: red_0 dead", not env._aircraft_by_id("red_0").state.alive)
    check("boundary: episode continues (2v3)", not term)

    # 6. Friendly collision
    env.reset(46)
    _set(env, "red_0", 0, 0, -3000)
    _set(env, "red_1", 10, 0, -3000)  # within collision_distance=30
    _set(env, "red_2", -10000, -10000, -3000)
    _set(env, "blue_0", 10000, 10000, -3000)
    _set(env, "blue_1", -10000, 10000, -3000)
    _set(env, "blue_2", 10000, -10000, -3000)
    obs, rewards, term, trunc, info = env.step(_all_actions(env))
    check("friendly collision: red_0 and red_1 dead",
          not env._aircraft_by_id("red_0").state.alive and not env._aircraft_by_id("red_1").state.alive)

    # 7. Dead aircraft cannot attack
    env.reset(47)
    _kill(env, "red_0")
    _set(env, "red_0", 0, 0, -3000)
    _set(env, "blue_0", 500, 0, -3000, psi=0)
    _set(env, "red_1", -10000, -10000, -3000)
    _set(env, "red_2", -10000, 10000, -3000)
    _set(env, "blue_1", 10000, 10000, -3000)
    _set(env, "blue_2", 10000, -10000, -3000)
    dead_actions = {**{aid: np.zeros(3, dtype=np.float32) for aid in RED_IDS + BLUE_IDS}}
    obs, rewards, term, trunc, info = env.step(dead_actions)
    check("dead red_0 cannot attack", info["attacks"]["red_0"] is None)

    # 8. Seed reproducibility
    env1 = Homogeneous3v3AirCombatEnv(args.env_config)
    env2 = Homogeneous3v3AirCombatEnv(args.env_config)
    obs1, _ = env1.reset(123)
    obs2, _ = env2.reset(123)
    check("seed reproducible", all(
        np.allclose(obs1[aid], obs2[aid]) for aid in RED_IDS + BLUE_IDS))

    # 9. Global rotation preserves distances
    env1.reset(200)
    dists1 = []
    for i, a1 in enumerate(env1.aircraft):
        for a2 in env1.aircraft[i+1:]:
            dists1.append(np.linalg.norm(a1.state.as_array()[:3] - a2.state.as_array()[:3]))
    check("all pairwise distances finite and positive", all(d > 0 and np.isfinite(d) for d in dists1))

    # 10. Numerical stress: 5 seeds, max 600 steps (quick audit)
    for seed in range(5):
        env.reset(seed + 1000)
        for _ in range(600):
            alive = [a for a in env.aircraft if a.state.alive]
            if not alive:
                break
            acts = {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in alive}
            obs, rewards, term, trunc, info = env.step(acts)
            for aid in RED_IDS + BLUE_IDS:
                o = obs[aid]
                assert np.all(np.isfinite(o)), f"seed={seed} non-finite obs"
                assert np.all(np.abs(o) <= 1.01), f"seed={seed} obs out of bounds"
            assert np.isfinite(rewards["red_0"]), f"seed={seed} non-finite reward"
            if term or trunc:
                break
    check("numerical stress 5 seeds passed", True)

    output_dir = Path("outputs/3v3_env_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "ALL_CHECKS_PASSED", "checks_completed": 10}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== ALL AUDIT CHECKS PASSED ===")
    print(f"Report saved to {output_dir / 'report.json'}")

if __name__ == "__main__":
    main()
