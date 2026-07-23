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
| 固定蓝方策略训练 | 阶段 B/C 中红方学习，蓝方由固定追击策略控制 |
| MADSAC R1–R4 分段奖励数值与角度层级 | 作为 `reward_mode: madsac_segmented` 实现 |
| 70% 胜率策略保存门槛 | 用作是否进入竞争训练的门槛 |
| alternating freezing 思想 | 仅保留研究背景；本轮达到门槛也不启动竞争训练 |

项目自行定义或近似的内容：

| 内容 | 边界说明 |
|---|---|
| `TargetStateController` 与 RK4 | 项目动力学/控制实现，不归因于论文控制器 |
| `PurePursuitPolicy` | 对“飞向最近目标并攻击”的简化近似，不是论文固定策略的完整复现 |
| `abs(pitch_error)` 近似 HA | 三自由度模型没有 HA；因此不能称为论文环境严格复现 |
| 确定性攻击与击杀、碰撞双方 -10 | 项目适配；论文未给出本项目的碰撞处理 |
| 三个 1v1 场景模板 | `tail_chase`、`offset_head_on`、`crossing` 均为项目定义 |
| 300k/1M/2M 阶段步数 | 本轮诊断预算，不是论文原始超参数 |
| 300 回合、三场景均分的门槛评估 | 项目评估协议；70% 数值本身有论文依据 |
| 当前 PPO/MAPPO 数据结构 | 集中式价值辅助的单方 PPO；沿用 MAPPO 数据结构以便扩展 |
| zero 对手基础课程、控制跟踪与联合攻击漏斗 | 项目定义的课程和诊断，不是论文指标 |

MADSAC 论文总训练量超过 800 万步，并约在 200 万步附近表现出收敛趋势。本项目先把 200 万步作为固定对手可学习性检查点；这不等同于复现 MADSAC 算法或结果。

## 奖励模式

默认 `madsac_segmented` 对红蓝双方分别计算，不强制零和：

- R1：击杀 +10、被击杀 -10、`max_steps` 为 0；碰撞双方 -10 是项目适配。
- R2：自身出界 -10，对手出界 +10；边界终止不会再叠加“被击杀”惩罚。
- R3：距离 ≥ 4000 m、ATA ≤ 30° 且 `abs(pitch_error) ≤ 30°` 时 +0.001。4000 m、30°、0.001 来自论文，`pitch_error` 近似 HA 属于项目适配。
- R4：距离 ≤ 4000 m 且 AA ≤ 30°，按 ATA 与 `abs(pitch_error)` 同时进入 5°/15°/30° 档给予 +0.1/+0.02/+0.01；从对手视角重算几何后，对称档位给予 -0.15/-0.025/-0.015 威胁惩罚。

`info["reward_terms"]` 分别给出 `reward_terminal`、`reward_boundary`、`reward_guide`、`reward_position`、`reward_threat` 和 `reward_total`。原有 `coupled_difference` 模式保留用于消融，不会与分段奖励叠加。

## paper_staged 训练协议

阶段 A `straight_tail_chase`（0–300,000 环境步）：只采样红方在后的 `tail_chase`，蓝方执行 zero 动作并保持当前目标航向、俯仰和速度。这是项目定义的基础追击技能课程，不来自论文。只更新红方 Actor 和红方价值输出。

阶段 B `pursuit_tail_chase`（300,000–1,000,000 环境步）：仍只使用红方在后的尾追场景，蓝方改用现有 `PurePursuitPolicy`。该策略是对论文固定策略的简化近似，不是完整复现。

阶段 C `pursuit_all_scenarios`（1,000,000–2,000,000 环境步）：蓝方继续纯追击，三个场景近似均匀采样。奖励、攻击区和动作尺度保持不变。

固定训练后读取 `fixed_best.pt`，以确定性动作对三个场景各评估 100 回合。70% 门槛仍只用于记录 `gate_pass`；本轮即使通过也不会自动启动竞争训练。

恢复训练使用 v3 检查点，保存双 Actor、Critic、各优化器、环境步、更新计数、阶段、活动方、block、门槛结果以及 Python/NumPy/Torch CPU/CUDA RNG。并行环境的完整中间状态不保存，因此恢复从新 episode 批次开始，并非逐状态完全确定性恢复。v2 及更早检查点会给出明确拒绝信息。

## 诊断与输出

规则策略基准用 zero 与 `PurePursuitPolicy` 的四种组合判断环境规则可解性，不会根据结果自动修改环境。

训练记录动作均值/标准差/极值、目标增量、控制误差、限幅前后命令率、由 `PointMassDynamics.derivatives` 得到的实际加速度/俯仰率/航向率、跟踪误差、`nx/nz/phi` 和六类饱和率。实际机动率诊断用于判断项目自定义控制器能否跟踪命令，不来自论文。

每个完整回合同时记录单独条件和同一步联合条件：distance+ATA、distance+AA、ATA+AA、完整攻击包线，并记录距离、ATA、AA 与归一化组合违反裕量。没有完整 episode 的更新把回报、胜负平和 episode 漏斗写为 NaN。确定性攻击模型中完整包线成立便立即击杀，因此“包线到击杀转化率”通常为 100%，不能视为独立学习能力指标。

输出位于 `outputs/mappo/`：

- `checkpoints/initial.pt`、`straight_best.pt`、`straight_final.pt`、`pursuit_tail_best.pt`、`pursuit_tail_final.pt`、`fixed_best.pt`、`fixed_final.pt` 和 `latest.pt`。
- `evaluation_<checkpoint>_<actor>_vs_<opponent>_<scenario>_<seedset>.json`，不同检查点不会覆盖。
- `training_metrics.csv`、`training_curves.png`、`attack_funnel.png`、`control_saturation.png`、`control_tracking_error.png` 和 `run_summary.json`。
- `outputs/baselines/rule_baselines.json` 保存四组规则策略整体和分场景结果。

每阶段最佳模型按当前阶段对手胜率、同一步完整攻击包线进入率、平均回报依次比较，而不是按训练 rollout 回报选择。

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

# 200 万步三阶段固定对手训练；只记录门槛结果，不启动竞争训练
python scripts/train_mappo.py --device cuda

# 单独评估，输出名自动包含检查点、Actor、对手、场景和 seed-set
python scripts/evaluate_mappo.py --checkpoint outputs/mappo/checkpoints/fixed_best.pt --actor red --episodes 300 --opponent pursuit --side red --scenario all --device cuda
```

项目当前不包含导弹、雷达、噪声、通信、生命值、弹药、2v2、循环/注意力网络、策略池或新的强化学习算法。
