# v3.16 Chen Event-Dominant Shared HAPPO Baseline

这是 **Chen-aligned shared-objective adaptation for pure HAPPO**，不是 Chen 原论文完全复现。
环境基础完整继承 v3.15；旧 v3.15 reward 分支和配置保留不变。

## 冻结公式

对存活且有 alive + team-visible target 的 UAV：

`raw_i = 10 R_speed + 15 R_angle + 10 R_distance`，`Q_i = raw_i / 35`。
无 target / 死亡 UAV 的 process 为零；不 clip，不改变 situation-assessment target selector。
`R_angle = 1 - (ATA + AA)/π` 保留负值。
speed/distance 分段公式直接复用 env/reward_chen_v315.py。

`Q_M = 0.5 R_dist + 0.2 R_aspect`，沿用 MAV direct-visible alive Blue 及 reverse ATA 映射。
无可见 Blue 时 R_dist=.2、R_aspect=0；死亡 MAV process 为0。

`R_process = (Q_M + Q_U1 + Q_U2 + Q_U3)/4`，固定分母4。

`R_event = 200 N_new_unique_Blue_red_attack - 200 N_UAV_combat - 100 N_UAV_boundary - 200 I_MAV_death`。

`r_shared = R_event + R_process`，四 Red 收到同一个 scalar。
shared GAE / single scalar centralized critic / HAPPO sequential update不变。
episode_return 每 decision step 累加一次 r_shared，不再次求四agent和。

每 unique Blue 仅计 shared +200，与 co-attacker 数无关；同一步多kill/loss相加。
MAV +50、cap200仅为 `mav_team_contribution_diagnostic`，**REMOVED_FROM_TRAINING_OBJECTIVE**。
terminal win/loss/draw、额外 safety 全为0。

## 文献与映射边界

- **PAPER_DIRECT**：Chen speed/angle/distance公式、10/15/10权重、UAV +200/-200/-100事件数值、MAV Safety结构与survival priority原则。
- **ENV_MAPPING / SCALE_NORMALIZATION**：UAV /35、固定process /4、shared event接线、unique Blue一次+200；MAV threat缺项，reverse ATA，危险/安全距离与死亡-200为可实现环境映射，不声称论文规定所有这些数值。
- **REMOVED_FROM_TRAINING_OBJECTIVE**：MAV contribution仅diagnostic，避免共同击杀重复计奖。
- **DISABLED_UNAVAILABLE**：height（公式不足）、dodge/missile threat（无导弹entity）、position/awareness/support（未实现）；无替代shaping。

## 尺度不变量

各 UAV Q_i 理论包络[-1,1]（由原始公式自然给出，不clip）。MAV Q_M最高.1。
`max_process_per_step = (.1+1+1+1)/4 = .775`。
`max_process_150_steps = 116.25 < 200`。
整局理论最大正process小于一次MAV death惩罚的绝对值，也小于一次真实Blue kill。
这保证特定尺度关系，不保证所有trajectory胜负排序或已证明可学习性。

## 诊断与checkpoint

每步 info 保存 shared_reward/event/process、四agent Q、三UAV raw process及四类event贡献。
episode summary、v3.16 training CSV/evaluation保存累计process/event/team及raw/normalized UAV diagnostics。
不保留可被误当训练return的local reward sums。
checkpoint额外保存 reward_wiring、uav_process_normalizer=35、shared_process_denominator=4、
mav_team_contribution_training=false、terminal_reward_enabled=false；load/resume/evaluator严格验证。
environment_version和reward_mode隔离，v3.15/v3.16拒绝交叉加载。
精确resume同时恢复unique-kill去重集合、贡献诊断、累计诊断和全部原有env/RNG/optimizer状态。

## 运行配置

configs/env_v316.yaml与happo_v316_baseline.yaml为独立入口。
优化参数、raw latent Gaussian PPO/preceding factor、ValueNorm、Huber10、value clipping
及collection-time old_normalized_values、orthogonal init均沿用v3.15。
本次实现仅unit/regression/极短CPU与CUDA smoke，不启动正式2M训练。
