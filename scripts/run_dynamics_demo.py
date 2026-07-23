"""运行四组独立动力学验证并绘制响应。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_combat.config import aircraft_spec, load_config
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftSpec, AircraftState, TargetCommand


def run_case(
    target: TargetCommand,
    steps: int,
    controller: TargetStateController,
    dynamics: PointMassDynamics,
    integrator: RK4Integrator,
    spec: AircraftSpec,
) -> np.ndarray:
    """从统一水平初态运行一个独立目标并返回时序数据。"""
    state = AircraftState(0, 0, -3000, 150, 0, 0)
    rows = [[0.0, state.v, state.altitude, state.theta, state.psi]]
    for index in range(steps):
        control = controller.compute_control(state, target, spec)
        state = integrator.step(state, control, dynamics, spec)
        rows.append([(index + 1) * integrator.dt, state.v, state.altitude, state.theta, state.psi])
    return np.asarray(rows)


def main() -> None:
    """运行配平、航向、俯仰和速度验证，输出指标并保存图像。"""
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/homogeneous_1v1.yaml")
    spec = aircraft_spec(config)
    controller = TargetStateController(**config["action"], gravity=config["simulation"]["gravity"])
    dynamics = PointMassDynamics(config["simulation"]["gravity"])
    integrator = RK4Integrator(config["simulation"]["dt"])

    trim = run_case(TargetCommand(0, 0, 150), 600, controller, dynamics, integrator, spec)
    yaw = run_case(TargetCommand(np.deg2rad(30), 0, 150), 600, controller, dynamics, integrator, spec)
    pitch = run_case(TargetCommand(0, np.deg2rad(-10), 150), 600, controller, dynamics, integrator, spec)
    speed = run_case(TargetCommand(0, 0, 170), 600, controller, dynamics, integrator, spec)

    trim_drift = np.max(np.abs(trim[:, 1:] - trim[0, 1:]), axis=0)
    print("trim: max_speed_drift={:.12g} m/s, max_altitude_drift={:.12g} m, max_pitch_drift={:.12g} deg, max_yaw_drift={:.12g} deg".format(
        trim_drift[0], trim_drift[1], np.rad2deg(trim_drift[2]), np.rad2deg(trim_drift[3])))
    print("yaw_step: final_yaw={:.9f} deg, final_speed={:.9f} m/s, final_pitch={:.9f} deg, altitude_change={:.9g} m, yaw_error={:.9g} deg".format(
        np.rad2deg(yaw[-1, 4]), yaw[-1, 1], np.rad2deg(yaw[-1, 3]), yaw[-1, 2] - yaw[0, 2], 30 - np.rad2deg(yaw[-1, 4])))
    print("pitch_step: final_pitch={:.9f} deg, final_yaw={:.9f} deg, max_abs_yaw={:.9g} deg, final_speed={:.9f} m/s, pitch_error={:.9g} deg".format(
        np.rad2deg(pitch[-1, 3]), np.rad2deg(pitch[-1, 4]), np.rad2deg(np.max(np.abs(pitch[:, 4]))), pitch[-1, 1], -10 - np.rad2deg(pitch[-1, 3])))
    print("speed_step: final_speed={:.9f} m/s, speed_error={:.9g} m/s, max_altitude_offset={:.9g} m, max_yaw_offset={:.9g} deg".format(
        speed[-1, 1], 170 - speed[-1, 1], np.max(np.abs(speed[:, 2] - speed[0, 2])), np.rad2deg(np.max(np.abs(speed[:, 4])))))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(trim[:, 0], trim[:, 1] - 150, label="speed drift (m/s)")
    axes[0, 0].plot(trim[:, 0], trim[:, 2] - 3000, label="altitude drift (m)")
    axes[0, 0].plot(trim[:, 0], np.rad2deg(trim[:, 3]), label="pitch drift (deg)")
    axes[0, 0].plot(trim[:, 0], np.rad2deg(trim[:, 4]), label="yaw drift (deg)")
    axes[0, 0].set_title("Level-trim drift"); axes[0, 0].legend()
    axes[0, 1].plot(yaw[:, 0], np.rad2deg(yaw[:, 4]), label="yaw (deg)")
    axes[0, 1].axhline(30, color="k", linestyle="--", label="target")
    axes[0, 1].set_title("Yaw step response"); axes[0, 1].legend()
    axes[1, 0].plot(pitch[:, 0], np.rad2deg(pitch[:, 3]), label="pitch (deg)")
    axes[1, 0].plot(pitch[:, 0], np.rad2deg(pitch[:, 4]), label="yaw crosstalk (deg)")
    axes[1, 0].axhline(-10, color="k", linestyle="--", label="target")
    axes[1, 0].set_title("Pitch step and yaw crosstalk"); axes[1, 0].legend()
    axes[1, 1].plot(speed[:, 0], speed[:, 1], label="speed (m/s)")
    axes[1, 1].axhline(170, color="k", linestyle="--", label="target")
    axes[1, 1].set_title("Speed step response"); axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Time (s)"); axis.grid(True)
    fig.tight_layout()
    output = root / "outputs/dynamics_demo.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
