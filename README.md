# 竞争式 MAPPO v5 基线

这是一个用于学术实验的简化同构 1v1 无人机空战环境。v5 根据论文成功标准修正边界奖励和评估语义，并在既有红蓝交替冻结训练上加入最小历史对手采样。它不是武器系统仿真，也不是 MADSAC 或 SAFAT 的严格复现。

## 环境约定

- NED 坐标：`x` 向北、`y` 向东、`z` 向下，高度为 `-z`。
- 状态为 `[x, y, z, v, theta, psi]`；滚转角 `phi` 只是控制量。
- 决策与仿真步长固定为 0.1 s，使用经典 RK4 积分。
- Actor 局部观测和 Critic 全局状态均为 14 维。
- 攻击条件保持为距离 100-1000 m、ATA <= 30 度、AA <= 90 度，满足后确定性击杀。
- 最大回合长度保持 600 步；场景为 `tail_chase`、`offset_head_on`、`crossing`。

策略输出三维归一化动作，映射为相对目标增量：

```text
delta_yaw   in [-pi, pi]
delta_pitch in [-pi/3, pi/3]
delta_speed in [-50, 50] m/s
```

`TargetStateController` 是项目定义的简化低层控制器，不是论文未公开控制器的复现。v5 没有修改动力学、控制器、动作/观测空间、攻击包线、网络规模、PPO 超参数或三个场景模板方向。

## 论文依据与项目适配

### MADSAC 论文直接支持

Li et al., “Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic”, Aerospace, 2023：

- 使用 0.1 s 决策时间和分段几何奖励。
- 训练与测试成功标准是红方摧毁全部蓝方无人机；平局不是红方胜利。
- 飞出交战区域属于未学会在指定区域作战的局部收敛现象。
- 悬停但不主动攻击属于论文识别到的 MAPPO 过拟合策略。
- 高成功率策略可在后续同时部署给红蓝双方测试。

因此 v5 将论文意义下的 win/success 与实际 kill 对齐。在当前 1v1 中，`red_kill` 才是红方成功，`blue_kill` 才是蓝方成功。boundary、collision、mutual kill 和 timeout 均不计为任何一方的论文意义成功。

论文主要让学习方对固定策略训练，之后才把高成功率策略同时部署给双方。本项目双边交替训练属于项目适配，不能写成论文原训练协议。

### CR-DRL 论文直接支持

Yang et al., “An air combat maneuver decision-making approach using coupled reward in deep reinforcement learning”, Complex & Intelligent Systems, 2025：

- 提出 synchronized alternating freezing adversarial training（SAFAT），两侧模型按代际交替更新。
- 历史超过一代时，以 0.7 概率选择最新一代对手，以 0.3 概率从更早历史代中均匀随机选择。
- 历史对手机制用于减少只针对当前个性化对手训练的问题。
- 原论文的 A、B 可以是不同强化学习算法。

本项目两侧均为同构 PPO/MAPPO Actor，各有独立 Actor、集中式标量 Critic 和优化器。因此当前实现只是 SAFAT 机制在本环境中的最小适配，不是严格复现。

### 项目适配

以下内容是项目定义或近似：

- 1v1、双方同构 PPO/MAPPO、独立红蓝 Actor/Critic、100,000 环境步 block。
- 确定性几何击杀、碰撞双方 -10、mutual kill 双方 -10。
- 对手出界时未出界方 reward 为 0。这是按论文成功标准做出的对称化项目适配，不是 MADSAC 原公式。
- 三个场景模板、项目控制器、`abs(pitch_error)` 对 HA 的近似。
- `competitive_best` 三层排序和 CPU Actor 历史快照。

200 万环境步仅参考 MADSAC 论文约在该尺度附近出现收敛的量级。当前算法和场景不同，这只是环境可学习性验证预算，不能声称复现论文收敛结果。

## 奖励语义

默认 `madsac_segmented`：

- `red_kill`：red +10，blue -10。
- `blue_kill`：red -10，blue +10。
- red 自身出界：red -10，blue 0。
- blue 自身出界：red 0，blue -10。
- collision 与 mutual kill：双方 -10。
- `max_steps`：终局奖励 0。
- R3/R4 数值以及距离、ATA、AA、`pitch_error` 条件保持不变。

单方出界后环境仍结束，并在 `outcome` 中标识未出界方，但未出界方不会获得击杀的 +10。`coupled_difference` 的稠密差保持不变，终局使用相同的 kill/boundary 映射。

## 评估语义

环境裁决单独报告 `red_outcome_wins`、`blue_outcome_wins`、`draws` 及对应 rate、`non_draw_rate`。

论文成功标准单独报告 `red_kills`、`blue_kills`、双方 kill rate，以及：

```text
combat_decisive_rate = (red_kills + blue_kills) / episodes
```

还分别报告红蓝 boundary loss、总体/高度/水平 boundary rate、collision、mutual kill、max steps、回合长度、双边回报和 attack funnel。外部对阵顶层给出 `matchup`、`learned_side`、`learned_*`、`opponent_*`。

outcome win 不等于 kill success。对手出界可以产生 outcome win，但不能计入空战击杀成功率。

## v5 历史对手

generation 0 是数值相同初始化后的红蓝 Actor CPU 深拷贝。每个 block 只选一次冻结对手，所有并行环境使用同一 generation：

- 只有 generation 0 时必选 0。
- 至少两代时，0.7 选最新代，0.3 从旧代均匀选择。
- 活动方用当前 Actor；冻结方用无优化器、`requires_grad=False`、eval 模式的 behavior Actor。
- 历史只加载到 behavior Actor，不覆盖冻结方当前 Actor、Critic 或优化器。
- block 结束只给活动方追加 CPU 深拷贝 generation，并记录 block、环境步和活动方。

训练日志记录 opponent side/generation、latest 标志、历史长度和活动 generation。`opponent_history.png` 展示 block、活动方、generation 及 latest/old 路径。

## v5 检查点

v5 保存两套 Actor、两套 Critic、四个优化器、双方历史及元数据、当前 block 冻结对手与 behavior Actor、0.7/0.3 配置、历史选择计数、block 记录和全部 RNG 状态。

恢复时重建同一 behavior opponent，不重新抽取 generation。环境中间物理状态不保存，恢复后从新 episode 批次开始。v4 及更早检查点会被明确拒绝，因为它们缺少历史对手和新的奖励/评价语义，不能无提示迁移或续训。

## 最佳检查点

`competitive_best.pt` 严格按以下字典序选择：

```text
(
  combat_decisive_rate,
  -(boundary_rate + collision_rate),
  -mean_episode_length
)
```

先比较实际击杀比例，再比较 boundary+collision，最后比较回合长度。`bilateral_kill_rate` 和 `kill_imbalance` 只记录，不参与第一轮选择。outcome win、non-draw 和累计回报不能替代 `combat_decisive_rate`。

## 输出与运行

默认目录为 `outputs/mappo_v5/`，不会覆盖 `outputs/mappo/`、`outputs/mappo_fixed_course_archive_20260723/` 或旧 v3/v4 产物。输出包括四类主检查点、`block_NNN.pt`、训练 CSV、`run_summary.json`、`block_evaluation_summary.csv` 及六张要求的诊断图。

`competitive_outcomes.png` 将 kill success、boundary loss 和 outcome 裁决分成三个坐标组。

在 WSL Ubuntu 的 `uav` Conda 环境中：

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_dynamics_demo.py
python scripts/run_env_smoke.py
python scripts/run_combat_demo.py
python scripts/train_mappo.py --smoke --device cuda
python scripts/train_mappo.py --device cuda --total-env-steps 2000000
```

正式训练后，对 `initial`、`competitive_best`、`final` 分别运行 300 episode self-play；再对 `competitive_best` 运行四组规则对手评估。训练入口会对每个 block 检查点做默认 90 episode self-play，三个场景各 30 回合。

```bash
python scripts/evaluate_mappo.py \
  --checkpoint outputs/mappo_v5/checkpoints/competitive_best.pt \
  --matchup blue_vs_pursuit --episodes 300 --scenario all --device cuda
```

项目不包含导弹、雷达、生命值、弹药、2v2、联赛、Elo、PSRO 或新算法。测试通过只证明实现协议成立；是否可学习必须依据正式训练相对 initial 的 kill 指标判断。
