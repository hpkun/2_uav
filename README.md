# 同构无人机三维对抗环境

本项目提供面向学术实验的简化同构 1v1 空战环境。当前版本在三自由度点质量动力学、连续动作和同步推进之上，加入确定性攻击区域、简化态势奖励、归一化观测与纯追击基线，用于后续强化学习可学习性验证。

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

环境先基于同一批旧状态为所有存活飞机计算控制量，再统一积分，避免实体更新顺序影响结果。

## 空战几何与攻击

相对位置定义为 `D = target_position - own_position`，相对速度为目标速度减自身速度。ATA 是攻击方速度与 `D` 的夹角，AA 是目标速度与 `D` 的夹角；尾追同向时二者均接近 0，迎头时 ATA 接近 0、AA 接近 `pi`。LOS yaw 为 `atan2(D_y, D_x)`；NED 的 z 轴向下，因此 LOS pitch 为 `atan2(-D_z, hypot(D_x, D_y))`。

确定性攻击条件为距离 `100–1000 m`、`ATA ≤ 30°`、`AA ≤ 90°`。条件满足即一次击杀。论文通常采用攻击区域、概率命中、毁伤或生命值；本项目为基础可学习性验证仅采用一次确定性几何击杀，**不是武器系统仿真**，配置数值也不是对特定论文参数的严格复现。

态势评分是受论文角度—距离耦合思想启发的 project-defined 简化评分，并非严格复现论文奖励函数：

```text
ata_score = 0.5 (1 + cos(ATA))
aa_score = 0.5 (1 + cos(AA))
distance_score = exp(-((distance - preferred_distance) / distance_scale)^2)
score = distance_score * ata_score * aa_score
```

非终止态势奖励为双方评分差乘 `0.1`，保持严格零和；击杀或单方越界采用 `±10` 终止奖励，互相击杀和碰撞则双方均为 `-10`，没有额外时间、速度、高度或动作平滑奖励。

## 13 维观测与规则基线

每架飞机的正式观测为 `[归一化自身速度, 归一化俯仰, sin(psi), cos(psi), 归一化相对 x/y/z, 归一化相对 vx/vy/vz, 归一化距离, ATA/pi, AA/pi]`。观测不包含绝对位置、生命值、历史、雷达或友机信息，全部裁剪至 `[-1, 1]`。

`PurePursuitPolicy` 是用于端到端环境验证的单纯视线追击策略：直接跟踪当前 LOS，并尝试比目标快 `20 m/s`。它不预测、不规避、不读取奖励，也没有有限状态切换。

## 设计边界

论文支持的设计假设包括 NED 坐标、三自由度质点模型、`0.1 s` 时间步，以及目标航向/俯仰/速度增量动作。本项目自行定义目标误差比例控制、变化率限制、`nx/nz/phi` 反解、RK4 数值积分与当前边界终止条件。

当前尚未实现雷达、传感器噪声、导弹制导、概率命中、生命值、弹药、多机协同、强化学习算法、执行器延迟和 2v2/3v3。

## 安装与运行

```bash
pip install -e .
pytest
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py
python scripts/run_combat_demo.py
```

动力学演示独立运行 60 秒水平配平、30° 航向阶跃、-10° 俯仰阶跃和 170 m/s 速度阶跃验证，输出漂移与最终误差，并将四个响应面板保存为 `outputs/dynamics_demo.png`。
