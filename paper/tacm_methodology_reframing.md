# TACM 最终科研逻辑与论文表述备忘

> 整理日期：2026-10-06。用途：长期研究记录、论文摘要/方法/消融解释、答辩汇报的优先依据。
> 本文是对冻结实现与既有实验的概念整理，不是算法修改方案。代码标识、checkpoint 合同、实验目录及原始结果保持不变。
> 证据边界：本地正式消融为 v3.10、main、3 个固定 training seeds（17、23、31）× 2M；不得与历史 v3.9、learnability 或其他 seed 组混用。

## 1. 最终定位：双层异构协同策略学习，而非工程演进史

论文主线应是“面向异构 MAV/UAV 协同空战的双层异构策略学习框架”，不应写成 HAPPO → RGAA → DBM-RGAA → TACM-RGAA 的开发过程。

存在两类不同的异构性：

- **平台级静态异构**：MAV 与 UAV 的任务责任、生存价值、可执行行为和奖励响应不同。在当前 v3.10 中，MAV 无直接攻击能力，UAV 承担攻击任务。共享团队回报表达全局合作，但未显式区分各平台的局部学习责任。
- **UAV 战术级动态异构**：同一 UAV 随战术上下文变化，需要调整交战、支援、目标重获取、威胁响应或普通协同机动倾向。这些是需要表示的行为变化，不是当前网络中五个独立模式，也不意味着每一种行为均已被实验单独验证。

统一逻辑为：**平台级异构学习 → 战术条件化策略表示 → 训练阶段的时间结构约束**。

“共享回报不足”“非结构化策略难以表达动态变化”属于设计动机，不能写成适用于所有任务的数学定理。本文实验支持的是在当前冻结任务与预算下，该结构带来的效果。

## 2. 仅保留两个核心机制

1. **Heterogeneity-Aware Auxiliary Policy Learning：异构平台辅助优势学习。**为已知平台类型补充局部责任相关的 policy-gradient 信号。
2. **Context-Guided Tactical Policy Modulation：上下文引导的战术策略调制。**以基础连续策略加软混合残差表示动态战术倾向；集中式上下文引导和事件感知时间一致性正则属于这一机制内部的训练约束。

不将 RGAA、DBM、Router、Expert、Teacher、Temporal 包装成六项独立创新。Temporal 是第二个机制的内部正则，不是第三个同级核心机制。

## 3. 机制一：异构平台辅助优势学习

### 3.1 已知平台责任，不是角色发现

MAV/UAV 类型与平台角色已知，不使用 role discovery、emergent role 或 role assignment 描述本机制。

共享团队优势回答：“这段轨迹对团队是否有利？”平台特异辅助优势回答：“对当前平台类型，这种局部行为是否具有积极意义？”后者依赖当前已定义的平台过程奖励，而非新发现的角色。

### 3.2 奖励分解、价值网络与优势融合

辅助奖励为：

\[
r^{aux}_{i,t}=r^{process}_{i,t}+r^{own\text{-}loss}_{i,t}.
\]

`extract_rgaa_auxiliary_rewards()` 从 info 读取四个平台过程奖励；若对应 agent 出现在该 transition 的 `death_causes` 中，读取环境配置的 `reward.mav_loss` 或 `reward.uav_loss`，仅给该 agent 加入自己的损失事件。boundary 和 blue_attack 都计入，既不硬编码罚值，也不添加新的环境奖励。

辅助分支**不额外包含** shared kill reward、shared terminal reward、shared safety reward、其他 agent 的 loss 或完整 event reward。这些仍按原有团队奖励合同处理。

训练使用一个集中式团队价值网络 `V_team(s)`、一个 MAV 局部观测辅助价值网络、一个由三架 UAV 共享的局部观测辅助价值网络；四个 actor 始终参数独立。辅助 GAE 在个体死亡及 episode terminated/truncated 时截断：

\[
c_{i,t}=(1-\mathbb{1}_{episode\ boundary,t})\,active_{i,t+1}.
\]

死亡 transition 自身的过程奖励和 own-loss 保留，后续 dead-state value/reward 不再 bootstrap。最后 rollout step 使用真实 bootstrap active mask；不能依据 auto-reset 后 mask 忽略 episode 边界。

最终 actor 优势为：

\[
A_i^*=\operatorname{Norm}_i(A_{team})+
\lambda_r\operatorname{Norm}_i(A_i^{aux}),\qquad \lambda_r=0.5.
\]

两项分别在该 agent 的 active samples 上归一化，组合后**不再二次归一化**。PPO 使用 `F_i * A_i^*`，保留 HAPPO 随机顺序更新和前序修正因子；team critic 仍拟合团队回报，不拟合融合优势。

### 3.3 理论定位及边界

推荐表述：“在保留全局协作目标的同时，为不同类型平台补充局部责任信号。”辅助分支不是第二套完整任务回报，而是平台相关的辅助 policy-gradient direction。

这里的“保留团队目标”指团队奖励、团队 critic 与团队优势继续存在；**不意味着融合梯度严格等价于原始团队目标的无偏梯度，也不构成最优策略不变性证明**。实际 actor 梯度被辅助优势有意调制。

正式 HAPPO 平均胜率为 35.83%，仅保留平台辅助学习基础的 TACM w/o Mode 为 70.17%。这是当前方法的重要基础机制，不应写成附属技巧；但证据是整个辅助学习机制的比较，不能进一步分离出某个罚项或某个 critic 的独立贡献。

## 4. 机制二：上下文引导的战术策略调制

当前单 UAV 结构为：

\[
h_i=f_i(o_i),\qquad p_i=\operatorname{softmax}(g_i(h_i)),
\]
\[
\mu_i=\mu_i^{base}+\rho\sum_{k\in\{eng,sup\}}p_{ik}e_{ik}(h_i),
\qquad \rho=0.25.
\]

`f_i` 是本机分支共享的两层 128 单元 Tanh encoder；基础 mean head 为 Linear(128,3)，router 为 Linear(128,2)+Softmax，两个 expert 各为 Linear(128,3)+Tanh。三个 UAV 架构相同、参数独立，MAV 仍是普通 Gaussian actor。

`mu_base` 表示通用连续机动策略；`e_ik` 表示战术相关残差动作分量；`p_ik` 是连续混合权重。最终语义是**通用机动策略 + 上下文相关战术残差修正**，不是 Engagement Policy 与 Support Policy 二选一。

动作由最终均值与可训练、状态无关的三维 `log_std` 构造高斯分布，经 Tanh 压缩输出。正式配置 `log_std` 初值为 -0.25，分布构造 clamp 为 [-5,2]。残差加在 **pre-tanh mean**，不能把 0.25 描述成物理控制量幅度或最终动作占比。

基础 mean 是可训练分支，并非冻结规则控制器。两项 expert 输出逐维有界于 [-1,1]，缩放后的混合残差逐维不超过 0.25；这体现有限幅度调制，而非全面接管基础控制。

## 5. 两个残差分量与 Expert 的正确语义

推荐术语：

- **Engagement-oriented residual component：交战倾向残差分量。**
- **Support-oriented residual component：支援倾向残差分量。**
- **Tactical residual basis：战术残差基 / 战术动作修正方向。**

`p_eng + p_sup = 1`。例如权重为 0.7/0.3 时，支援分量仍参与加权动作修正；argmax 标签只是分析用 hard proxy，不是执行时的选择动作。

设计动机是不同战术上下文共享大量基础飞行规律，不必为每种上下文重学完整低层策略。代码中的“shared”仅指同一 UAV 的主干及 UAV auxiliary critic，不指三架 UAV 共享 actor 参数。

不能声称“UAV 频繁在两个明确战术模式间切换”。推荐：“UAV 根据局部观测形成交战倾向与支援倾向之间的连续战术混合表示。”

语义名称不自动证明专家功能：teacher 直接监督权重，不直接标定 expert 输出。残差分量是否实际产生某种行为仍需机制证据，不能仅凭分支名断言。

## 6. 集中式战术上下文引导，不是专家动作模仿

论文推荐用 **Privileged Tactical Context Guidance：训练阶段特权战术上下文引导**；Tactical Teacher 仅作为代码映射术语。

教师在 pre-action global state 上解码，但仅使用至少被一架存活 Red 直接探测到的存活 Blue，不是任意全知敌情。它综合交战准备度、Blue 对 MAV 的威胁、UAV 拦截适宜度与软支援责任，产生二维语义分布 `q_i=[q_eng,q_support]` 及置信度。

代码中的交战/威胁质量结合 angle quality、distance quality、normalized attack streak 和 exact attack gate；支援责任由威胁强度与 alive UAV 间的软分配得到。当前两个 temperature 均为 0.25。无可见敌机时 teacher 为 [0.5,0.5]、置信度为 0，不贡献语义监督。

上下文损失是置信度加权 `KL(q_i || p_i)`；代码分母为 `max(sum(confidence),1)`。教师不给出专家动作、连续控制量、高层离散动作或 manager action，也不直接控制 expert。

必须同时说明：

1. 教师仅训练期存在，执行期删除；不作为 actor 的额外输入。
2. Router supervision 使用 detached encoder hidden，直接语义梯度仅进入 router。
3. PPO/RGAA 梯度仍训练完整 actor，低层连续机动仍由强化学习学习。
4. 当前 context coefficient 在前 500k sampled steps 从 0.05 线性降至 0.01。

推荐定位：“利用训练期集中式战术信息，为局部策略内部的低维战术表示提供语义锚点”，而非“告诉 UAV 应该怎样飞”。

## 7. 辅助优势与上下文引导分工不同

| 层次 | 回答的问题 | 直接作用对象 |
|---|---|---|
| 平台特异辅助优势 | 什么局部行为对这一类平台有意义？ | actor 的 policy-gradient advantage `A_i^*` |
| 战术上下文引导 | 当前状态下如何组织连续战术策略表示？ | router probability `p_i` |
| 事件感知时间正则 | 稳定上下文中如何约束相邻软表示？ | 相邻 router distributions |

前者调制学习信号，后两者约束策略内部表示。两层机制并不重复，也不应都被写成角色分配。

## 8. 事件感知时间一致性正则化

正式名称为 **Event-Aware Temporal Consistency Regularization**。不以 Mode Persistence、Tactical Mode Persistence 或“模式持续性”作为主表述。

单对关系为：

\[
\ell_{temp,t}=m_t\min(c_t,c_{t+1})
\left\|p_{t+1}-\operatorname{sg}(p_t)\right\|_2^2.
\]

完整代码对有效加权 pairs 求和，除以 `max(sum(weights),1)`。`sg` 是停止梯度；编码特征也 detached，因此直接正则梯度仅更新 router。

`m_t=1` 必须同时满足：agent 在 t/t+1 active；t 未 terminated/truncated；t 无 kill/death transition event；engagement target 不变；MAV threat target 不变；teacher argmax semantic 不变。结构性事件发生或上下文身份变化时，该 pair 的约束解除。身份相同不等于连续几何完全不变，也不意味着实现保证更长的 hard-mode 持续时间。

Full coefficient 为 0.01；w/o Temporal 为 0，严格跳过额外 temporal optimization step。Temporal 在该 UAV 的 PPO/context 更新后执行，再重新计算 log probabilities 更新 HAPPO preceding factor。

## 9. 三条核心数学关系与真实优化顺序

论文可将机制压缩为三条主关系：

1. `A_i^* = Norm_i(A_team) + lambda_r Norm_i(A_i_aux)`，`lambda_r=0.5`。
2. `mu_i = mu_i_base + rho sum_k p_ik e_ik`，`rho=0.25`。
3. `L = L_HAPPO(A_i^*) + lambda_context L_context + lambda_temporal L_temporal`。

**第三式是目标组成的概括，不是三项在每个 minibatch 一次联合反向传播的实现声明。**代码实际顺序是：按随机 agent 顺序，以融合优势执行 PPO（包含 entropy，UAV minibatch 加 context KL）→该 UAV router-only temporal step→重算该 actor log probability→更新 preceding factor→下一 actor；团队/辅助价值网络按各自目标优化。MAV 不含 context/temporal。

部署只保留 `local observation → actor → continuous action`；删除集中式 critic、平台辅助 critics、上下文教师和 temporal targets，不向 actor 提供 global state。四个 actor 参数独立，保持 decentralized execution。

## 10. 正式消融合同与结果

来源：`outputs/audits/tacm_final_ablation_results_20261005_141445/` 中的 `final_ablation_summary.csv/json`、`final_ablation_per_seed.csv`，协议定义见 `tools/tacm_ablation_protocol.py`。

协议为 v3.10 main，16 envs，每方法固定 training seeds 17/23/31，各实际采样 2,000,000 steps；每 seed 独立 stochastic 200 episodes，environment seeds 12000–12199、action seeds 13000–13199。训练末尾单局 deterministic execution smoke 不是正式性能结果。四方法共用 reset 随机化 curriculum（learnability→main，400k），因此不能把 curriculum 收益归为 TACM 特有机制。

| 论文方法 | 平台辅助优势 | 残差调制与上下文引导 | Temporal | 胜率 mean ± sample SD | Return 均值 | Red kills 均值 | Draw | UAV survivors 均值 |
|---|---|---|---|---|---:|---:|---:|---:|
| HAPPO | 无 | 无 | 无 | 35.83 ± 25.93% | 297.36 | 2.835 | 61.17% | 2.105 |
| TACM w/o Mode | 有 | 无 | 无 | 70.17 ± 10.02% | 420.90 | 3.563 | 29.67% | 2.585 |
| TACM w/o Temporal | 有 | 有 | 无 | 70.50 ± 9.85% | 425.38 | 3.597 | 29.50% | 2.587 |
| Full TACM | 有 | 有 | 有 | 76.17 ± 5.01% | 448.01 | 3.728 | 23.83% | 2.922 |

SD 是三个 training seeds 的样本标准差，不是 episode-level SD 或置信区间。Full 在这组正式结果的 Return、Red kills、Draw（越低越好）、UAV survivors 上也取得最好方法均值。

每个 training seed 的正式胜率原样保留：

| seed | HAPPO | w/o Mode | w/o Temporal | Full |
|---|---:|---:|---:|---:|
| 17 | 29.0% | 80.5% | 78.5% | 81.0% |
| 23 | 14.0% | 60.5% | 59.5% | 76.5% |
| 31 | 64.5% | 69.5% | 73.5% | 71.0% |

不能宣称 Full 在每个 seed 都优于 w/o Temporal；seed31 为 -2.5 pp。三个 seeds 的平均优势和较小离散性是已观察到的事实，不等于对全部 seed 或新任务的稳定性证明。

## 11. 消融解释：完整战术调制体系，不是机械逐模块加分

按未舍入均值计算：Full−w/o Mode = **+6.00 pp**；Full−w/o Temporal = **+5.67 pp**；Full−HAPPO = **+40.33 pp**。w/o Mode−HAPPO 为 +34.33 pp；w/o Temporal−w/o Mode 为 +0.33 pp。用显示到两位的百分数直接相减会产生 0.01 pp 舍入差异，不应当作结果冲突。

核心解释：平台辅助优势首先建立异构学习基础；在此基础上，战术策略调制与事件感知时间正则一起构成动态战术学习机制。Full 相对 w/o Mode 的 +6.00 pp 支持完整战术调制体系在这组实验中的有效性；Full 相对 w/o Temporal 的 +5.67 pp 支持时间结构约束的作用。

但应避免以下误读：

- w/o Mode 实际移除 residual/router/context/temporal 整体，不是只去掉某一层；差值不能独立归因于 router 或 expert。
- +0.33 pp 不支持“单独加入 Mode 显著提升”；w/o Temporal 仍包括 context，因此也不是纯粹无监督 residual 的对照。
- 四组嵌套消融支持“完整调制+正则组合有效”的叙述，**不能严格估计统计交互项或证明两者不可分离的因果协同**。
- 不把平台 +34.33、Mode +0.33、Temporal +5.67 机械解释成可移植、线性独立的贡献。

## 12. Temporal 内部机制证据与统计口径

来源：`outputs/audits/tacm_temporal_mechanism_20261005_151524/` 下六个 `no_temporal/full_seed17/23/31_200ep/semantic_audit_summary.json`。本备忘只读取既有文件，未重放。

以下是每个 seed 内的统计再对三个 seeds 等权平均，不是混池所有 transition：

| 指标 | w/o Temporal | Full | 变化 |
|---|---:|---:|---:|
| Teacher–Router engagement probability correlation | 0.466421 | 0.617511 | 相对约 +32.39% |
| confidence-weighted KL | 0.029118 | 0.022589 | 相对约 -22.42% |
| 高置信交战主导状态后 10 步内 any team kill | 70.4731% | 79.1899% | +8.7169 pp |

概率 correlation 是连续 teacher/router 值的关联，不是 hard label agreement。二分量互补，因此 engagement/support 两个相关性不是两条独立证据。

“高置信”在现有审计中使用 teacher confidence 上四分位口径，状态组还按 router argmax 分类；不是统一绝对置信阈值，也不是离散执行模式。原 `teacher_high_confidence` 字段名及其 grouping definition 必须一起引用。`any_team_kill` 是后续 transition window 中任意团队击杀，不是当前 UAV 必然击杀、不代表专门击杀威胁 MAV 的目标。重叠窗口与同一回合状态彼此相关，不可把 states 数当作独立样本量。

合理的描述性证据链为：**时间正则 → 更好的连续语义组织 → 更好的短期团队战斗转化 → 更高的任务表现**。箭头表达机制假设，不是严格因果证明；方法间访问的状态分布也可能不同。

## 13. Temporal 不应被描述成 hard-mode persistence

既有 `temporal_mechanism_seed_summary.csv` / `temporal_mechanism_comparison.json` 的 mask-valid stable-context 统计：

- hard switch rate：w/o Temporal 约 0.571%，Full 约 1.480%，**并未降低**。
- router engagement probability variance：约 0.01012 → 0.01290，**并未降低**。
- same-mode run length mean：约 5.649 → 5.253，**并未延长**。
- router L1 movement：约 0.02124 → 0.02105，只有很小平均差异，不能包装成显著全面平滑。

late-training 的 1.6–2.0M 窗口统计也不支持“Temporal 降低每个 seed 后期 Win Rate 波动”。例如 seed17：w/o Temporal 的窗口 SD 约 0.0373，Full 约 0.0913。Full 正式跨 training-seed SD 较低，与单 seed 日志窗口更平稳不是同一命题。

因此写作应强调 **soft representation consistency ≠ hard mode persistence**。正则约束的对象是稳定上下文中的 soft tactical representation；观察到的 correlation/KL 改善比 argmax 切换更贴近该定义。

## 14. Support 证据边界

六个正式机制审计中，“teacher-high-confidence + router Support-dominant + 10-step”的样本数均为 **0**。不能把这些状态的行为指标填 0%，它们是无样本、不可估计。

禁止声称“充分验证独立 MAV-Support 战术模式”“存在明确 defense mode / MAV protection mode”或已证明 protect MAV 的因果机制。

推荐：“支援语义主要作为连续战术混合中的偏置分量，而不是频繁出现的独立离散模式。”其中，“非零支援权重能参与动作”是结构事实；“支援分量已经独立造成 MAV 保护效果”并未由当前样本证明。即使缺少 Support-dominant 状态，混合权重非零也不等于分量无效；反之，参与混合也不能自动证明其战术功效。

## 15. 推荐摘要、方法主线与创新点层级

### 推荐主线段落

本文针对异构 MAV/UAV 协同空战中同时存在的平台级静态异构和战术级动态异构问题，提出双层异构协同策略学习框架。首先，通过平台特异辅助优势补充共享团队目标难以显式刻画的局部责任；进一步利用训练阶段集中式战术上下文引导 UAV 的连续残差策略调制，在不直接规定低层动作的前提下形成战术条件化策略；在该调制机制内部，引入事件感知时间一致性正则，对稳定上下文中的软路由表示施加结构化约束。既有固定三-seed实验显示，完整方法提高任务平均表现，并改善软路由与战术上下文的关联，但不将这些机制关联等同于严格因果证明。

### 创新点组织

第一项强调平台辅助梯度与团队合作目标的层次分工；第二项强调上下文引导的有限幅度连续残差调制及其事件条件时间正则。网络层、router、expert、teacher 分别是实现结构或训练手段，不再各自声称独立创新。

## 16. 论文命名候选与工程标识映射

当前工程简称 TACM（Tactical-Context Aware Consistent Mode Learning）容易让 Mode 被理解成离散模式、Consistent 被理解成 hard persistence，且未直接表达平台异构贡献。

候选论文名称仅记录，尚未正式重命名：

1. **首选 HTCM — Heterogeneity-aware Tactical Context Modulation（异构感知战术上下文调制）**。Heterogeneity-aware 对应平台辅助优势，Tactical Context 对应训练期上下文引导，Modulation 对应基础策略加软混合残差。
2. HTM-HAPPO — Heterogeneity-aware Tactical Mixture HAPPO。
3. Tactical-Aware Coordination Modulation（备选展开，非已确认名称）。

论文主体优先使用两个核心机制的概念名称。RGAA、DBM-RGAA、RGAA-Wide、TACM-RGAA-v1 只在开发历史、消融代码映射或复现说明中出现。不批量改动 `tacm_rgaa`、`TACM_RGAA_METHOD`、metadata、configs、checkpoint 或目录名。

## 17. 代码与证据索引、发现的表述边界

| 需核对的语义 | 本次读取的依据 |
|---|---|
| 辅助奖励、individual-death GAE | `algorithm/happo/rgaa.py` |
| 平台优势融合、active normalization、preceding factor | `algorithm/happo/trainer.py` 非 recurrent update |
| 独立 residual actor、soft mixture、Gaussian 输出 | `algorithm/happo/dbm_rgaa.py`、`algorithm/common/networks.py` |
| teacher、confidence KL、event-aware temporal、metadata | `algorithm/happo/tacm_rgaa.py`、trainer rollout/update |
| 正式训练参数及 temporal 消融 | `configs/happo_tacm_rgaa_v310.yaml`、`configs/happo_tacm_rgaa_v310_no_temporal.yaml` |
| 平台基础对照与统一协议 | `configs/happo_rgaa_v310_curriculum.yaml`、`configs/happo_v310_ablation.yaml`、`tools/tacm_ablation_protocol.py` |
| 正式性能与 seed 级明细 | `outputs/audits/tacm_final_ablation_results_20261005_141445/` |
| 语义机制及 hard-proxy 边界 | `outputs/audits/tacm_temporal_mechanism_20261005_151524/`、`tools/audit_tacm_semantic_modes.py` |
| 历史说明 | `docs/tacm_rgaa_v1_spec.md`（历史 v3.9，不用其旧评估协议替代正式 v3.10 协议） |

概念重述与冻结代码未发现必须修改算法才能化解的冲突。必须补充的表述边界已在正文体现：统一 loss 式不是一次联合 step；w/o Mode 是整组调制机制的消融；共享基础主干不是跨 UAV 共享 actor；hard persistence 与机制证据不一致；支援效果证据有限；“保留团队目标”不是原始梯度等价定理。新论文名不等于工程重命名。

本次只新增此备忘，未修改任何算法、环境、reward、配置、checkpoint、训练脚本、evaluator 或实验结果；没有运行训练、评估、replay、测试或新实验。

## 18. Methodology Writing Rules

1. 不按工程开发历史讲最终算法。
2. 不把内部每个零件包装成独立创新。
3. 最终仅两个核心机制：异构平台辅助优势学习、上下文引导的战术策略调制。
4. Temporal 是第二机制内部的训练正则。
5. Mode 是连续战术混合，不是离散执行状态。
6. Teacher 是训练阶段上下文引导，不是动作专家。
7. Expert 是残差动作基，不是完整策略；语义标签不自动保证行为功效。
8. 不宣称 Temporal 降低 hard switch、延长 hard persistence 或普遍降低概率方差。
9. 不宣称独立 Support mode 已充分验证；无样本指标不得写成零效果。
10. 不夸大因果关系，不把 nested ablation 的均值差当独立可加收益或严格交互证明。
11. 不修改冻结算法来迎合论文故事。
12. 正式结果保持原样，不重新选择 seed、不将训练窗口混作 formal evaluation。
13. 说明统计单位、sample SD、评估模式、seed 范围、环境版本和 profile；不跨协议拼接证据。
14. 区分概括目标与实际优化顺序；写明 auxiliary 梯度改变 actor 方向，而不改变 team reward/critic 合同。
15. 论文命名候选只作概念记录，保持工程名称及历史实验兼容性。
