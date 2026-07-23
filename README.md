# 同构无人机三维对抗环境

这是一个面向学术实验的简化同构 1v1 空战环境。当前版本用于检验基础三自由度动力学、连续目标状态控制、确定性几何攻击以及固定对手下的可学习性；它不是武器系统仿真，也不宣称复现 MADSAC、CR-DRL 或其论文最终结果。

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
| 固定蓝方策略训练 | 阶段 A/B 中红方学习，蓝方由固定追击策略控制 |
| MADSAC R1–R4 分段奖励数值与角度层级 | 作为 `reward_mode: madsac_segmented` 实现 |
| 70% 胜率策略保存门槛 | 用作是否进入竞争训练的门槛 |
| alternating freezing 思想 | 门槛通过后交替冻结红蓝 Actor |

项目自行定义或近似的内容：

| 内容 | 边界说明 |
|---|---|
| `TargetStateController` 与 RK4 | 项目动力学/控制实现，不归因于论文控制器 |
| `PurePursuitPolicy` | 对“飞向最近目标并攻击”的简化近似，不是论文固定策略的完整复现 |
| `abs(pitch_error)` 近似 HA | 三自由度模型没有 HA；因此不能称为论文环境严格复现 |
| 确定性攻击与击杀、碰撞双方 -10 | 项目适配；论文未给出本项目的碰撞处理 |
| 三个 1v1 场景模板 | `tail_chase`、`offset_head_on`、`crossing` 均为项目定义 |
| 500k/2M/100k/1M 阶段步数 | 本轮诊断预算，不是论文原始超参数 |
| 300 回合、三场景均分的门槛评估 | 项目评估协议；70% 数值本身有论文依据 |
| 当前 PPO/MAPPO 数据结构 | 集中式价值辅助的单方 PPO；沿用 MAPPO 数据结构以便扩展 |
| 控制饱和与攻击漏斗 | 项目诊断指标，不是论文指标 |

MADSAC 论文总训练量超过 800 万步，并约在 200 万步附近表现出收敛趋势。本项目先把 200 万步作为固定对手可学习性检查点；这不等同于复现 MADSAC 算法或结果。

## 奖励模式

默认 `madsac_segmented` 对红蓝双方分别计算，不强制零和：

- R1：击杀 +10、被击杀 -10、`max_steps` 为 0；碰撞双方 -10 是项目适配。
- R2：自身出界 -10，对手出界 +10；边界终止不会再叠加“被击杀”惩罚。
- R3：距离 ≥ 4000 m、ATA ≤ 30° 且 `abs(pitch_error) ≤ 30°` 时 +0.001。4000 m、30°、0.001 来自论文，`pitch_error` 近似 HA 属于项目适配。
- R4：距离 ≤ 4000 m 且 AA ≤ 30°，按 ATA 与 `abs(pitch_error)` 同时进入 5°/15°/30° 档给予 +0.1/+0.02/+0.01；从对手视角重算几何后，对称档位给予 -0.15/-0.025/-0.015 威胁惩罚。

`info["reward_terms"]` 分别给出 `reward_terminal`、`reward_boundary`、`reward_guide`、`reward_position`、`reward_threat` 和 `reward_total`。原有 `coupled_difference` 模式保留用于消融，不会与分段奖励叠加。

## paper_staged 训练协议

阶段 A（0–500,000 环境步）：只采样红方在后的 `tail_chase`；红方 Actor 更新，蓝方完全由 `PurePursuitPolicy` 控制。蓝方 Actor、其 Adam 状态和蓝方价值输出头不参与有效损失。

阶段 B（500,000–2,000,000 环境步）：红方继续对固定追击蓝方学习，三个场景近似均匀采样。奖励、攻击区和动作尺度不变。

阶段 C：读取 `fixed_best.pt`，使用确定性动作和固定种子集，对三个场景各评估 100 回合（共 300 回合）。draw 按未获胜处理，并报告胜负平、回报、长度、边界、最大步数、攻击包线进入率和击杀率。整体胜率达到 70% 才能进入阶段 D；否则立即停止，不自动改奖励、攻击区、动作或超参数。

阶段 D（仅门槛通过）：先把最佳红方参数复制给蓝方，随后每 100,000 环境步切换活动方，总预算 1,000,000 步。冻结方仍执行自己的策略，但 Actor 参数和 Adam 状态不更新；Critic 只对活动方价值输出计算损失。

恢复训练使用 v3 检查点，保存双 Actor、Critic、各优化器、环境步、更新计数、阶段、活动方、block、门槛结果以及 Python/NumPy/Torch CPU/CUDA RNG。并行环境的完整中间状态不保存，因此恢复从新 episode 批次开始，并非逐状态完全确定性恢复。v2 及更早检查点会给出明确拒绝信息。

## 诊断与输出

训练记录动作均值/标准差/极值、目标增量、控制误差、限幅前后变化率、`nx/nz/phi` 和六类饱和率。饱和使用数值容差判定。

每个完整回合记录最小距离、是否进入 4000 m、攻击距离、ATA 门、AA 门、完整攻击包线、包线步数、击杀、出界和最大步数，并汇总攻击漏斗比例。没有完整 episode 的更新把回报和胜负平写为 NaN，CSV 与绘图保留该缺失语义。

输出位于 `outputs/mappo/`：

- `checkpoints/initial.pt`、`fixed_best.pt`、`fixed_final.pt`；门槛通过时另有 `competitive_initial.pt`、`competitive_best.pt`、`competitive_final.pt`。
- `evaluation_<checkpoint>_<actor>_vs_<opponent>_<scenario>_<seedset>.json`，不同检查点不会覆盖。
- `training_metrics.csv`、`training_curves.png`、`attack_funnel.png`、`control_saturation.png` 和 `run_summary.json`。

固定阶段最佳模型按 pursuit 胜率、攻击包线进入率、平均回报依次比较，而不是按训练 rollout 回报选择。

## 安装与运行

在本项目约定的 WSL `uav` Conda 环境中：

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py

# CUDA 冒烟
python scripts/train_mappo.py --smoke --device cuda

# 200 万步固定对手训练；仅通过门槛后自动继续 100 万步交替冻结
python scripts/train_mappo.py --device cuda

# 单独评估，输出名自动包含检查点、Actor、对手、场景和 seed-set
python scripts/evaluate_mappo.py --checkpoint outputs/mappo/checkpoints/fixed_best.pt --actor red --episodes 300 --opponent pursuit --side red --scenario all --device cuda
```

项目当前不包含导弹、雷达、噪声、通信、生命值、弹药、2v2、循环/注意力网络、策略池或新的强化学习算法。
