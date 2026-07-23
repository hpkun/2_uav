"""运行确定性尾追击杀演示。"""
from pathlib import Path

import numpy as np

from uav_combat.environment import HomogeneousAirCombatEnv
from uav_combat.rule_policy import PurePursuitPolicy


def main() -> None:
    """让红方纯追击静态策略蓝方，并打印战斗结果。"""
    root = Path(__file__).parents[1]
    env = HomogeneousAirCombatEnv(root / "configs/homogeneous_1v1.yaml")
    env.reset(seed=0)
    red, blue = (next(aircraft for aircraft in env.aircraft if aircraft.aircraft_id == aircraft_id) for aircraft_id in ("red_0", "blue_0"))
    red.state.x, red.state.y, red.state.z = 0.0, 0.0, -3000.0
    red.state.v, red.state.theta, red.state.psi = 170.0, 0.0, 0.0
    blue.state.x, blue.state.y, blue.state.z = 1500.0, 0.0, -3000.0
    blue.state.v, blue.state.theta, blue.state.psi = 150.0, 0.0, 0.0
    action = env.config["action"]
    policy = PurePursuitPolicy(action["delta_yaw_max"], action["delta_pitch_max"], action["delta_speed_max"])
    cumulative = {"red_0": 0.0, "blue_0": 0.0}
    info = {}
    for _ in range(300):
        _, rewards, terminated, truncated, info = env.step({"red_0": policy.action(red, blue), "blue_0": np.zeros(3)})
        for aircraft_id, reward in rewards.items():
            cumulative[aircraft_id] += reward
        if terminated or truncated:
            break
    geometry = info["geometries"]["red_0"]
    print(f"steps={info['step_count']}")
    print(f"outcome={info['outcome']}, reason={info['termination_reason']}")
    print(f"distance={geometry.distance:.6f} m, ATA={np.rad2deg(geometry.ata):.6f} deg, AA={np.rad2deg(geometry.aa):.6f} deg")
    print(f"cumulative_rewards={cumulative}")
    print(f"alive={{'red_0': {red.state.alive}, 'blue_0': {blue.state.alive}}}")


if __name__ == "__main__":
    main()
