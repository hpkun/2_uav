"""运行 60 秒确定性目标状态跟踪演示并绘图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from uav_combat.config import aircraft_spec, load_config
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftState, TargetCommand


def main() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/homogeneous_1v1.yaml")
    spec = aircraft_spec(config)
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    target = TargetCommand(np.deg2rad(30), np.deg2rad(10), 170.0)
    controller = TargetStateController(**config["action"], gravity=config["simulation"]["gravity"])
    dynamics = PointMassDynamics(config["simulation"]["gravity"])
    integrator = RK4Integrator(config["simulation"]["dt"])
    history = []
    for index in range(600):
        control = controller.compute_control(state, target, spec)
        state = integrator.step(state, control, dynamics, spec)
        history.append([index * integrator.dt, state.v, state.altitude, np.rad2deg(state.theta), np.rad2deg(state.psi)])
    data = np.asarray(history)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    labels = [("Speed", "m/s"), ("Altitude", "m"), ("Pitch", "deg"), ("Yaw", "deg")]
    for column, (axis, (title, unit)) in enumerate(zip(axes.flat, labels), start=1):
        axis.plot(data[:, 0], data[:, column]); axis.set_title(title); axis.set_ylabel(unit); axis.grid(True)
    axes[1, 0].set_xlabel("Time (s)"); axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    output = root / "outputs/dynamics_demo.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"final: speed={state.v:.2f} m/s, altitude={state.altitude:.2f} m, pitch={np.rad2deg(state.theta):.2f} deg, yaw={np.rad2deg(state.psi):.2f} deg")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

