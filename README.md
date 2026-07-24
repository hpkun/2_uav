# 交替冻结竞争式MAPPO基线

这是一个面向学术实验的简化同构 1v1 空战环境。当前版本用交替冻结的双方独立 PPO 策略检验竞争对抗的可学习性；它不是武器系统仿真，也不宣称复现 MADSAC、CR-DRL 或其论文最终结果。

## 环境约定

- NED 坐标：`x` 向北、`y` 向东、`z` 向下，高度为 `-z`。
- 状态为 `[x, y, z, v, theta, psi]`；滚转角 `phi` 只是控制量。
- 仿真步长固定为 0.1 s，控制量在一步内保持不变，使用经典 RK4 积分。
- 局部 Actor 观测和集中式全局状态均为 14 维。
- 攻击条件保持为距离 100–1000 m、ATA ≤ 30°、AA ≤ 90°，满足后确定性击杀。
- 最大回合长度保持为 600 步。

三自由度质点方程为：

```text
x_dot = v cos(theta) cos(psi)       y_dot = v cos(theta) sin(psi)
z_dot = -v sin(theta)               v_dot = g (nx - sin(theta))
theta_dot = g/v (nz cos(phi) - cos(theta))
psi_dot = g nz sin(phi) / (v cos(theta))
```

策略输出三维归一化动作，映射为相对目标增量：

```text
delta_yaw   ∈ [-pi, pi]
delta_pitch ∈ [-pi/3, pi/3]
delta_speed ∈ [-50, 50] m/s
```

`TargetStateController` 是本项目定义的公开简化低层控制器。它把目标航向、俯仰和速度误差映射为 `nx/nz/phi`，不是论文未公开控制器的复现。本轮只增加误差、限幅前后控制量及饱和诊断，没有修改其参数或行为。

## 论文依据与项目适配

论文明确支持的内容：

| 内容 | 本项目采用方式 |
|---|---|
| NED 三自由度模型与 0.1 s 时间步 | 保留相同建模层级和时间步 |
| 目标航向、俯仰、速度增量动作及范围 | 严格保留 `pi`、`pi/3`、`50 m/s` |
| 固定策略与交替冻结思想 | 仅作为研究背景；当前训练协议是本项目适配 |
| MADSAC R1–R4 分段奖励数值与角度层级 | 作为 `reward_mode: madsac_segmented` 实现 |

项目自行定义或近似的内容：

| 内容 | 边界说明 |
|---|---|
| `TargetStateController` 与 RK4 | 项目动力学/控制实现，不归因于论文控制器 |
| `PurePursuitPolicy` | 对“飞向最近目标并攻击”的简化近似，不是论文固定策略的完整复现 |
| `abs(pitch_error)` 近似 HA | 三自由度模型没有 HA；因此不能称为论文环境严格复现 |
| 确定性攻击与击杀、碰撞双方 -10 | 项目适配；论文未给出本项目的碰撞处理 |
| 三个 1v1 场景模板 | `tail_chase`、`offset_head_on`、`crossing` 均为项目定义 |
| 100k 环境步交替冻结、100 万总步数 | 项目训练协议，不是论文原始超参数 |
| 当前 PPO/MAPPO 数据结构 | 双方各自 Actor 和标量集中式 Critic；共享 `[T,N,2]` rollout 布局 |
| 竞争胜负、控制跟踪与双边攻击漏斗 | 项目定义的评估与诊断，不是论文指标 |

MADSAC 论文的训练规模和算法均不同于本项目。这里的 100 万环境步只是工程验证预算，不等同于复现其算法或结果。

## 奖励模式

默认 `madsac_segmented` 对红蓝双方分别计算，不强制零和：

- R1：击杀 +10、被击杀 -10、`max_steps` 为 0；碰撞双方 -10 是项目适配。
- R2：自身出界 -10，对手出界 +10；边界终止不会再叠加“被击杀”惩罚。
- R3：距离 ≥ 4000 m、ATA ≤ 30° 且 `abs(pitch_error) ≤ 30°` 时 +0.001。4000 m、30°、0.001 来自论文，`pitch_error` 近似 HA 属于项目适配。
- R4：距离 ≤ 4000 m 且 AA ≤ 30°，按 ATA 与 `abs(pitch_error)` 同时进入 5°/15°/30° 档给予 +0.1/+0.02/+0.01；从对手视角重算几何后，对称档位给予 -0.15/-0.025/-0.015 威胁惩罚。

`info["reward_terms"]` 分别给出 `reward_terminal`、`reward_boundary`、`reward_guide`、`reward_position`、`reward_threat` 和 `reward_total`。原有 `coupled_difference` 模式保留用于消融，不会与分段奖励叠加。

## alternating_self_play 训练协议

红蓝 Actor 数值相同地初始化，但参数对象和 Adam 优化器完全独立；红蓝还分别拥有一个输入 14 维全局状态、输出标量价值的 Critic。双方在每一步都采样动作，rollout 始终保持 `[T,N,2]`。第 `env_steps // alternating_block_env_steps` 个 block 中，偶数 block 只更新红方 Actor/Critic，奇数 block 只更新蓝方 Actor/Critic。冻结方仍参与对抗，但其参数与优化器状态不发生变化。切换时不复制、不重置任何网络。

三个场景按 `tail_chase → offset_head_on → crossing` 确定性均衡循环；尾追场景的后方队伍在红蓝之间交替。block 边界会提前结束 rollout、保存 `block_NNN.pt`、重置并行环境，再切换活动方。默认每个 block 为 100,000 环境步，总预算 1,000,000 环境步。

检查点格式为 v4，保存两套 Actor、两套 Critic、四个优化器、环境步、更新计数、活动方、block、配置以及 Python/NumPy/Torch CPU/CUDA RNG。v3 固定目标课程检查点会被明确拒绝。恢复训练会从新 episode 批次开始，因此不是环境中间状态级的逐步复现。

## 诊断与输出

规则策略基准用 zero 与 `PurePursuitPolicy` 的四种组合判断环境规则可解性，不会根据结果自动修改环境。

训练记录动作均值/标准差/极值、目标增量、控制误差、限幅前后命令率、由 `PointMassDynamics.derivatives` 得到的实际加速度/俯仰率/航向率、跟踪误差、`nx/nz/phi` 和六类饱和率。实际机动率诊断用于判断项目自定义控制器能否跟踪命令，不来自论文。

每个完整回合分别为红蓝记录单独条件和同一步联合条件：distance+ATA、distance+AA、ATA+AA、完整攻击包线，并记录违反裕量。没有完整 episode 的更新把回报、胜负平和 episode 漏斗写为 NaN。确定性攻击模型中完整包线成立便立即击杀，因此“包线到击杀转化率”不能视为独立学习能力指标。

输出位于 `outputs/mappo/`：

- `checkpoints/initial.pt`、`competitive_best.pt`、`latest.pt`、`final.pt` 和约十个 `block_NNN.pt`。
- `evaluation_<checkpoint>_<matchup>_<scenario>_<seedset>.json`，不同检查点和对阵不会覆盖。
- `training_metrics.csv`、`training_curves.png`、`competitive_outcomes.png`、`attack_funnel.png`、`control_saturation.png`、`control_tracking_error.png` 和 `run_summary.json`。
- `outputs/baselines/rule_baselines.json` 保存四组规则策略整体和分场景结果。

竞争最佳模型依次比较决胜率（高）、边界与碰撞率之和（低）、平均回合长度（短），而不是按单方训练回报选择。

## 安装与运行

在本项目约定的 WSL `uav` Conda 环境中：

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py
python scripts/run_combat_demo.py
python scripts/evaluate_rule_baselines.py --episodes-per-scenario 100

# CUDA 冒烟
python scripts/train_mappo.py --smoke --device cuda

# 100 万步交替冻结竞争训练
python scripts/train_mappo.py --device cuda

# 单独评估；blue_vs_* 中检查点蓝 Actor 确实控制蓝机
python scripts/evaluate_mappo.py --checkpoint outputs/mappo/checkpoints/competitive_best.pt --matchup self_play --episodes 300 --scenario all --device cuda
python scripts/evaluate_mappo.py --checkpoint outputs/mappo/checkpoints/competitive_best.pt --matchup blue_vs_pursuit --episodes 300 --scenario all --device cuda
```

项目当前不包含导弹、雷达、噪声、通信、生命值、弹药、2v2、循环/注意力网络、策略池或新的强化学习算法。
