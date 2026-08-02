# 3v3 端到端审计说明

本文档记录当前项目的端到端审计口径，重点区分论文依据、项目适配和已确认实现问题。

## 真实执行链

当前固定蓝方 3v3 MAPPO 训练入口为 `scripts/train_mappo_3v3.py`。真实路径为：

1. 读取训练配置和环境配置；
2. `FixedBlue3v3MAPPOTrainer` 创建共享红方 Actor、集中式 team critic 和 3v3 VectorEnv；
3. VectorEnv 为每个环境创建 `Homogeneous3v3AirCombatEnv` 和固定规则蓝方；
4. reset 生成 6 架飞机状态、局部 observation `[6, 68]`、global state `[48]` 和 alive mask `[6]`；
5. 红方三个 alive agent 的 observation 展平成 `[num_envs * 3, 68]`；
6. 共享 tanh-squashed Gaussian Actor 采样红方连续动作 `[num_envs, 3, 3]`；
7. 蓝方规则策略生成固定追击动作；
8. 环境执行动作映射、控制器、三自由度动力学导数和 RK4 积分；
9. 先结算边界与碰撞；
10. v7 `paper_segmented_team_v4` 在攻击前捕获 R3/R41/R42 dense 几何快照；
11. 计算可见性、攻击意图、攻击击杀和死亡账本；
12. 根据 post-attack alive 计数产生 terminated/truncated/outcome；
13. 计算 team reward，并广播给对应队伍三架飞机；
14. VectorEnv 返回 observation、global state、team reward、done、alive mask、reward components 和 episode summary；
15. RolloutBuffer 保存 old log-prob、alive mask、team value、team reward 和 done；
16. GAE 使用 `terminated | truncated` 作为 nonterminal mask 的反向；
17. MAPPO/PPO 更新共享 Actor 和 centralized critic；
18. 按固定 evaluation seed 做 deterministic evaluation；
19. 按 `compute_best_score()` 选择 best checkpoint；
20. 写 checkpoint、metrics、evaluation 和 run summary。

## 论文依据与项目适配

- MADSAC 论文式分段奖励用于校对 R1/R2/R3/R4 形式和攻击几何奖励思想。
- 当前 `paper_segmented_team_v4` 是项目适配，不是原论文逐字复刻：项目使用三自由度 NED 点质量模型、确定性击杀、100--1000 m 攻击距离、ATA/AA 几何门限和固定蓝方规则策略。
- 论文未规定的工程细节，例如并行环境、death ledger、checkpoint signature、固定蓝方 greedy policy、timeout 作为红方失败，均属于项目适配。
- 当前 v7 奖励的 team total 语义为：

```text
team_total = dense_reward + event_reward + terminal_reward
dense_reward = R3 + R41 + R42
event_reward = kill - own_loss
```

component 向量同时包含 dense decomposition 和 dense aggregate，调用方不得再次把 decomposition 与 dense aggregate 同时求和。

## 已确认修复

1. v7 dense 时序：R3/R41/R42 使用运动、边界、碰撞之后且攻击结算之前的几何快照。
2. v7 dense double-count：`team_total_reward` 不再等于 `R3 + R41 + R42 + dense_reward + event + terminal`。
3. HAPPO checkpoint RNG：环境 reset 后再恢复算法 RNG。
4. MAPPO 3v3 checkpoint RNG 与签名：load 时检查 family、version、training signature；重新创建 episode 后再恢复 NumPy/Torch RNG。

## 当前 10M 失败的已证伪原因

- 不是 CUDA 崩溃或 Worker 异常；
- 不是核心 loss/KL/advantage 的 NaN 爆炸；
- 不能再归因于旧的 dense double-count，因为 10M 运行目录是 single-dense 修复后的长训练；
- 不是简单的 pitch 符号反向：正 pitch action 使飞机爬升，负 pitch action 使飞机下降；
- 不是 observation 完全缺少自身高度边界信息：self block 第 3 维编码 altitude 在 `[altitude_min, altitude_max]` 内的归一化位置。

## 仍需实验验证的问题

- v7 dense 几何窗口对随机/早期策略是否过于稀疏；
- R3/R41/R42 与最终攻击成功之间的信用链是否太长；
- 10M 后期策略是否因 entropy/log_std 漂移而偏向高度边界；
- best≈900k 与 final 的 deterministic action bias 是否显著不同；
- 当前 `compute_best_score()` 是否偏好“少量击杀 + 存活/拖时”的局部最优，而不是最终可攻击策略。

这些问题不应通过直接调奖励系数或扩大网络来猜测解决，建议先用 `scripts/audit_mappo_v7_10m_failure.py` 做固定 seed 离线诊断。

## 第二轮审计修正说明

- MAPPO 3v3 checkpoint signature 现在包含 `env_config_sha256`，即环境 YAML 文件内容的 SHA-256。严格 resume 同时检查 family、version 和完整 training signature；环境路径改变但内容相同可以恢复，内容改变必须失败。旧 checkpoint 缺少该字段，不能作为严格 resume 使用。
- `scripts/audit_mappo_v7_10m_failure.py` 的 v2 版本只把旧 checkpoint 当作 audit-only Actor 权重读取，不等价于训练恢复。它不会修改 10M 原始输出目录。
- 环境新增默认关闭的 `audit_trace_enabled`。仅当审计脚本显式打开时，`info["audit"]["paper_segmented_v4_pre_attack"]` 会记录与 v7 reward 同一步、同一 pre-attack snapshot 对齐的目标、距离、ATA、AA、可见性、attack-window 和 R3/R41/R42 局部项。
- `altitude_death_pre100.csv` 现在保存历史缓冲中每一步当时已经对齐的动作、控制、几何和 reward，不再把死亡发生那一步的 R3/R41/R42 批量复制到此前 100 步。
- observation bank 来自 initial/best/latest/final checkpoint 的确定性真实访问轨迹，并按类别去重和限额；缺失类别会在 `audit_summary.json` 中报告为 absent，不人工伪造样本。

## MAPPO entropy 字段含义

当前 MAPPO/HAPPO 的训练 loss 使用 `loss = policy_loss - entropy_coef * entropy`，其中 `entropy` 是未经过 tanh 变换前的 raw Gaussian entropy。实际执行动作来自 tanh-squashed Gaussian，因此 raw entropy 与有界动作空间中的 squashed entropy 不是同一个量：当 policy mean 已经很大、`tanh(mean)` 接近 ±1 时，raw entropy 仍可能有限甚至偏高，但 deterministic action 已饱和。第二轮审计新增 `raw_gaussian_entropy`、`estimated_squashed_entropy`、`sampled_action_saturation` 和 `deterministic_action_saturation` 诊断；本轮不改变 entropy bonus 公式。

## HAPPO 论文/官方实现对照结论

对照 HAPPO 论文的 sequential agent update 思路与 HARL 官方实现主干逻辑后，当前项目 HAPPO 的核心顺序更新保持一致：每轮使用一个 agent 顺序，`factor` 初始为 1；每个 agent 用 `factor * advantage` 进入 PPO clipped objective；该 agent 更新完成后，用同一批 old log probability 与更新后的 new log probability 比率更新 preceding-policy ratio factor，并 detach，避免跨 agent 反向传播；critic 在顺序 actor 更新后更新。当前 per-agent active-mask advantage normalization 是项目针对死亡/无效样本的工程适配，已有单元测试覆盖 inactive 样本，不在本轮认定为论文违背。

参考：HAPPO 论文 arXiv 页面 `https://arxiv.org/abs/2109.11251`；HARL 官方实现仓库 `https://github.com/PKU-MARL/HARL`。
