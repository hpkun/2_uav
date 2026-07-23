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

非终止态势奖励为双方评分差乘 `0.01`，保持严格零和；600 步内理论最大稠密累计值为 6，低于击杀终止奖励 10，避免策略只保持优势而故意不击杀。击杀或单方越界采用 `±10`，互相击杀和碰撞则双方均为 `-10`，没有额外时间、速度、高度或动作平滑奖励。

## 随机初始场景

训练时从三个可复现模板随机选择：`tail_chase` 为同向尾追并随机交换追击方，`offset_head_on` 为带 400 m 横向偏移的迎头接近，`crossing` 为大致垂直并朝中心交汇。每个模板还施加全局水平旋转、独立高度/速度扰动和小幅航向扰动。给定相同 seed 与模板会产生相同状态；`scenario_name="fixed"` 可继续使用原始确定性场景。

## 13 维观测与规则基线

每架飞机的正式观测为 `[归一化自身速度, 归一化俯仰, sin(psi), cos(psi), 归一化相对 x/y/z, 归一化相对 vx/vy/vz, 归一化距离, ATA/pi, AA/pi]`。相对 x、y 分别除以 `2*x_limit`、`2*y_limit`，相对 z 除以合法高度跨度，距离使用这三个最大相对尺度的欧氏范数；相对速度除以 `2*v_max`。观测不包含绝对位置、生命值、历史、雷达或友机信息，全部裁剪至 `[-1, 1]`。

`PurePursuitPolicy` 是用于端到端环境验证的单纯视线追击策略：直接跟踪当前 LOS，并尝试比目标快 `20 m/s`。它不预测、不规避、不读取奖励，也没有有限状态切换。

## 设计边界

论文支持的设计假设包括 NED 坐标、三自由度质点模型、`0.1 s` 时间步，以及目标航向/俯仰/速度增量动作。本项目自行定义目标误差比例控制、变化率限制、`nx/nz/phi` 反解、RK4 数值积分与当前边界终止条件。

## 参数共享 MAPPO 基线

本项目的 MAPPO 是验证环境可学习性的简化基线，不是研究创新，也不是对某篇论文代码或超参数的严格复现。红蓝双方参与自博弈并共享同一个 Actor；Actor 仅使用各自 13 维 ego-centric 局部观测独立决策，不含 team one-hot 或 agent id。训练时，集中式 Critic 接收固定顺序拼接的 `[red_obs, blue_obs]` 26 维观测，并同时输出两个价值。两方样本共同更新 Actor；执行阶段不需要对方观测，符合集中训练、分散执行。

Actor 是两层 128 单元 Tanh MLP，输出高斯均值并学习三维 `log_std`。它采样 `raw_action ~ Normal(mean,std)`，执行 `action=tanh(raw_action)`，log probability 包含 `-log(1-action²+epsilon)` Jacobian 修正，因此不会用概率不一致的事后 clip。Critic 是同样两层的 MLP，输出 `[V_red,V_blue]`。

训练使用 GAE 与 PPO clipped objective：

```text
delta_t = reward_t + gamma * V_(t+1) * (1-done_t) - V_t
A_t = delta_t + gamma * lambda * (1-done_t) * A_(t+1)
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio*A, clip(ratio,1-epsilon,1+epsilon)*A))
actor_loss = policy_loss - entropy_coef * entropy
```

Actor 与 Critic 使用独立 Adam、多 epoch minibatch、梯度裁剪和有限值检查。

当前尚未实现雷达、传感器噪声、导弹制导、概率命中、生命值、弹药、2v2/3v3、异构飞机和零样本迁移算法。

## 安装与运行

```bash
pip install -e .
pytest
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py
python scripts/run_combat_demo.py

# 完整训练
python scripts/train_mappo.py --env-config configs/homogeneous_1v1.yaml --train-config configs/mappo_1v1.yaml

# 快速管线验证
python scripts/train_mappo.py --env-config configs/homogeneous_1v1.yaml --train-config configs/mappo_1v1.yaml --smoke

# 对零动作或纯追击对手评估
python scripts/evaluate_mappo.py --checkpoint outputs/mappo/checkpoints/best.pt --episodes 100 --opponent zero --side both --scenario all
```

动力学演示独立运行 60 秒水平配平、30° 航向阶跃、-10° 俯仰阶跃和 170 m/s 速度阶跃验证，输出漂移与最终误差，并将四个响应面板保存为 `outputs/dynamics_demo.png`。

MAPPO 输出位于 `outputs/mappo/`：`training_metrics.csv` 保存逐更新指标，`training_curves.png` 保存四面板曲线，`evaluation_*.json` 保存评估结果，`checkpoints/` 保存 `initial.pt`、`latest.pt`、`best.pt` 和 `final.pt`。这些训练产物已由 `.gitignore` 排除。
