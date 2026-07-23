"""运行 1v1 环境冒烟演示。"""
from pathlib import Path
import numpy as np
from uav_combat.environment import HomogeneousAirCombatEnv


def main() -> None:
    root = Path(__file__).parents[1]
    env = HomogeneousAirCombatEnv(root / "configs/homogeneous_1v1.yaml")
    observations, _ = env.reset(seed=0)
    info = {}
    for _ in range(200):
        observations, _, terminated, truncated, info = env.step({"red_0": np.array([0.05, 0, 0]), "blue_0": np.zeros(3)})
        if terminated or truncated:
            break
    print(f"steps={info['step_count']}, reason={info['termination_reason']}")
    for aircraft in env.aircraft:
        print(f"{aircraft.aircraft_id}: {aircraft.state}")
    print(f"distance={observations['red_0'][-1]:.2f} m")


if __name__ == "__main__":
    main()

