# 同构无人机三维对抗环境

本项目提供面向学术实验的同构无人机三维对抗基础物理层，为后续研究异构能力与零样本迁移保留可配置的飞机性能参数。当前版本仅实现 1v1、三自由度点质量动力学、连续动作、项目自定义低层控制器和同步多机推进；不包含武器、雷达、奖励、训练算法、复杂气动或六自由度模型。

## 坐标与状态

采用 NED 坐标：`x` 指北、`y` 指东、`z` 指下，因此海拔为 `-z`。六维状态为 `[x, y, z, v, theta, psi]`，其中 `theta` 是航迹俯仰角，`psi` 是航迹偏航角。滚转角 `phi` 仅是当前控制区间使用的控制量，不作为独立状态。

动力学方程为：

```text
x_dot = v cos(theta) cos(psi)       y_dot = v cos(theta) sin(psi)
z_dot = -v sin(theta)               v_dot = g (nx - sin(theta))
theta_dot = g/v (nz cos(phi) - cos(theta))
psi_dot = g nz sin(phi) / (v cos(theta))
```

分母带数值保护。控制在每个 `0.1 s` 步长内保持不变，由经典四阶 Runge-Kutta 积分。

## 动作与控制器

策略动作 `[a_yaw, a_pitch, a_speed]` 逐维裁剪到 `[-1, 1]`，分别表示相对当前状态的目标航向、目标俯仰和目标速度增量。默认最大增量为 `pi`、`pi/3` 和 `50 m/s`。

低层控制器是 **project-defined 的公开简化实现，并非对论文未公开控制器的严格复现**。它对最短航向误差、俯仰误差和速度误差进行比例控制及变化率限幅。当前控制范围为 `nx ∈ [-1, 1]`、有符号法向过载 `nz ∈ [-3, 3]`、`phi ∈ [-pi/2, pi/2]`，然后按下式反解：

```text
nx = clip(v_dot_cmd/g + sin(theta))
A = cos(theta) + v/g * theta_dot_cmd
B = v cos(theta)/g * psi_dot_cmd
若 A >= 0：nz_raw = hypot(A, B),  phi_raw = atan2(B, A)
若 A < 0： nz_raw = -hypot(A, B), phi_raw = atan2(-B, -A)
nz = clip(nz_raw), phi = clip(phi_raw)
```

`A < 0` 时使用负 `nz`，使 `phi` 始终处于允许区间，并避免纯下俯动作因滚转角被裁剪而产生非期望偏航。航向动作仍保留论文名义端点 `pi`，控制器内部以 `1e-6 rad` 的 epsilon 将实际端点保护到 `pi - epsilon`，因此最大正、负动作保持各自方向且不会被 `[-pi, pi)` 包装合并。

环境先基于同一批旧状态为所有存活飞机计算控制量，再统一积分，避免实体更新顺序影响结果。当前奖励恒为 `0.0`，仅是基础动力学接口占位值，不能用于强化学习训练。

## 设计边界

论文支持的设计假设包括 NED 坐标、三自由度质点模型、`0.1 s` 时间步，以及目标航向/俯仰/速度增量动作。本项目自行定义目标误差比例控制、变化率限制、`nx/nz/phi` 反解、RK4 数值积分与当前边界终止条件。

当前尚未实现武器、攻击判定、正式奖励、学习算法、ATA/AA、雷达/噪声、生命值、导弹制导、执行器延迟和 2v2/3v3 初始化。

## 安装与运行

```bash
pip install -e .
pytest
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py
```

动力学演示独立运行 60 秒水平配平、30° 航向阶跃、-10° 俯仰阶跃和 170 m/s 速度阶跃验证，输出漂移与最终误差，并将四个响应面板保存为 `outputs/dynamics_demo.png`。
