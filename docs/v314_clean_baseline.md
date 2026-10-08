# v3.14 Clean HAPPO Baseline

独立版本：`heterogeneous_mavuav_4v4_v3_14`。
环境配置：`configs/env_v314.yaml`；训练配置：`configs/happo_v314_baseline.yaml`。
默认环境及 v3.5–v3.13 配置均不改变。

## 冻结合同

1 MAV + 3 UAV 对 4 Blue；MAV 无武器。保留 3DOF、RK4、CAP-Blue、
100D observation、117D global state、3D action，以及 MAV 12km / UAV 5km
的感知与团队共享机制。CAP-Blue 不因本次修改加入感知限制。

决策间隔 1s，物理步长 0.1s，上限 150 步。三类飞机速度范围均为
[150,300]m/s，初速 250m/s；位置、航向及过载参数保持 v3.13。
main/learnability 初始化扰动全部为零；baseline 关闭初始化 curriculum。
其余 HAPPO 超参数保持 v3.13，仅允许 vanilla / mlp / baseline。

## 武器

完整 gate：0 < d <= 5000m，ATA < 30°，AA < 90°，hold=1。
每名持武器攻击者每个 decision boundary 至多一个 candidate。
多目标优先级 ATA、AA、距离、canonical ID；同步结算后清除失效锁定，
不在同一 boundary 再获取目标。非锁定 pair 不预累积。
武器资格不依赖 reward target 或团队共享视野。

## 共享奖励

event = 10 × Blue击杀数 − 10 × UAV损失数 − 10 × MAV损失数。
terminal：Red 胜 +20，Blue 胜 −20，draw 0。

引导奖励使用物理更新、boundary 处理之后，攻击死亡结算之前的快照：
每架当时存活 UAV 选择存活且 team-visible 的最近 Blue，平局按 canonical ID。
无目标或 ATA >= 30°：0；ATA < 30° 且 d > 5000m：0.01；否则：0.02。
AA 不参与引导，MAV 无 process reward。在此快照中死亡的 UAV 记零；
本步攻击者的引导不会因为目标随后被击杀而消失。

team_guide = (g1 + g2 + g3) / 3，固定分母。
team_reward = event + terminal + team_guide，四名 Red 获得相同奖励。
friendly safety penalty 为零，不调用历史 situation / role / PBRS 公式。

引导 target/reward、事件、终局与团队奖励通过 info 记录；episode summary
保存 team_guide_reward_sum，vector checkpoint 保存该累计值和 weapon locks。
checkpoint、精确恢复和独立 evaluator 均核对版本、奖励及单目标武器合同；
不能跨 v3.13/v3.14 加载。

本次仅执行单元/回归测试及短 CUDA smoke，不代表训练可学习性已经实证完成。
