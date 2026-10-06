# TACM 中文网络架构图

## 中文论文图注

**图 X. 战术上下文感知的一致性模式学习（TACM）算法架构。**（a）集中训练与分散执行框架：红方由一架无直接攻击能力的 MAV 和三架 UAV 组成，与四架蓝方飞机对抗。四个参数独立的策略网络根据本机局部观测产生连续动作。集中式团队价值网络、MAV 辅助价值网络和 UAV 共享辅助价值网络分别估计团队优势与包含本机损失事件的角色辅助优势，并通过优势融合参与 HAPPO 顺序更新。仅在训练阶段使用战术上下文教师监督 UAV 路由器，并通过事件感知时序一致性正则化约束相邻路由分布。（b）单个 UAV 的 TACM 策略网络：基础高斯策略均值叠加交战导向与支援导向残差专家的加权软混合。三个 UAV 网络结构相同、参数独立；执行时仅保留策略网络，不使用价值网络、教师或辅助损失。

## 中文术语与图中模块

- **共享编码器**：仅表示同一 UAV 内的基础分支、路由分支和专家分支使用同一个主干；不表示三架 UAV 共享策略参数。
- **软路由器**：输出交战与支援概率 `p_eng`、`p_sup`，用于连续加权残差，不进行离散角色硬切换。
- **基础均值与残差专家**：最终均值为 `mu_base + 0.25 * (p_eng * e_eng + p_sup * e_sup)`，再结合可训练的三维 `log_std` 构造高斯策略，经 Tanh 压缩获得动作。`log_std` 与状态无关，不是编码器输出头。
- **辅助奖励**：本机过程奖励加本机损失事件。损失从 `death_causes` 与既有环境奖励配置读取，不包括其他智能体损失、共享击杀、终局或安全奖励。
- **辅助 GAE**：在个体死亡或回合边界处截断。团队优势与本机辅助优势分别归一化后，以系数 0.5 融合，融合结果不再二次归一化；随后乘 HAPPO 前序修正因子。
- **战术上下文教师**：训练时读取全局状态，仅使用团队可见敌机，结合交战准备度、MAV 威胁和支援责任产生语义分布与置信度；不参与推理动作生成。
- **上下文与时序正则化**：编码特征停止梯度，辅助损失仅直接更新路由器。事件掩码排除死亡、击杀、回合边界及不稳定目标/教师模式的时序对；前一时刻分布停止梯度。
- **图例**：实线表示执行/前向数据流；虚线表示训练期监督或参数更新；点线表示时序一致性正则化；浅色虚线区域内的模块仅用于训练。

## 使用与正文引用建议

建议放在论文“方法：TACM 网络架构”小节开头，先说明整体训练框架，再介绍教师和时序损失。正文可引用：“如图 X 所示，TACM 通过仅训练期使用的战术语义监督塑造软路由器，在不引入执行期特权信息的前提下学习交战与支援导向的残差行为。”

运行 `python tools/render_tacm_architecture.py` 即可重新生成同名 SVG、PDF、PNG。脚本自动选择已安装的中文字体；Windows 优先使用微软雅黑，Linux 可使用 Noto Sans CJK SC，WSL 也可复用 Windows 字体。找不到中文字体时明确报错，不静默生成缺字图。PDF 嵌入字体；SVG 保留可编辑文字，换机器编辑时需安装相同或兼容字体。保留公式与算法缩写，其余图面标注尽量使用中文。以下英文图注及实现记录保留，便于之后恢复英文论文插图。

## Paper caption draft

**Figure X. Architecture of Tactical-Context Aware Consistent Mode Learning (TACM).**
(a) Centralized training and decentralized execution in a heterogeneous team of one unarmed MAV and three UAVs against four Blue aircraft. Four independent policies generate continuous actions from local observations. A centralized team critic and observation-conditioned auxiliary critics provide team and own-loss-aware role advantages for sequential HAPPO optimization. A training-only tactical-context teacher supervises the UAV routers, while event-aware temporal consistency regularizes consecutive router distributions. (b) Each UAV actor augments a base Gaussian-policy mean with a scaled soft mixture of engagement- and support-oriented residual experts. The three UAV actors share their architecture but not their parameters. Only actors are retained at execution; critics, teacher targets, and auxiliary regularization are absent from the inference path.

## Modules and precise implementation semantics

- **Task contract:** the displayed formal configuration is `heterogeneous_mavuav_4v4_v3_10`; MAV is unarmed. Local observations are 100D. Environment/Blue/combat semantics are not altered by TACM.
- **Actors:** MAV remains an ordinary squashed-Gaussian MLP. Each UAV has its own two-layer 128-unit Tanh encoder, base mean head, two-logit softmax router, and two independent Linear(128, 3)-Tanh residual experts. “Shared trunk” refers only to branches inside one UAV actor, never to parameter sharing between UAVs.
- **Mean construction:** `mu_i = mu_base + 0.25 * (p_eng * e_eng + p_sup * e_sup)`. Mode indices 0/1 correspond to `engagement`/`cover_support`. These are continuous residual components, not hard role assignments or rule controllers.
- **Exploration:** `log_std` is a learned state-independent three-vector, not an observation-conditioned head. The displayed configuration initializes it at -0.25; distribution construction clamps it to [-5, 2] before exponentiation. Stochastic execution samples the pre-tanh Normal; deterministic execution uses its mean. Both apply Tanh to obtain 3D actions.
- **Critics:** one global-state-conditioned centralized team critic, one MAV observation-conditioned auxiliary critic, and one UAV observation-conditioned auxiliary critic shared across UAV1/2/3. The team reward remains the existing environment team reward. Auxiliary rewards equal the corresponding process reward plus that same agent's configured loss event from `death_causes`. Other agents' losses, shared kills, terminal rewards, and safety rewards are not added to the auxiliary stream.
- **Role learning:** auxiliary GAE terminates at individual death and episode boundaries. Each agent uses `Norm(A_team) + 0.5 * Norm(A_aux,i)`, normalized on its active samples without a second combined normalization. The HAPPO preceding factor multiplies this fused advantage. Actors update sequentially in randomized order, not simultaneously.
- **Teacher:** deterministic pre-action global-state decoding, restricted to alive Blue aircraft visible to at least one alive Red aircraft. Engagement readiness, MAV threat, and soft UAV support responsibility produce the two-way semantic distribution and normalized-entropy confidence. This privileged teacher is never an inference input.
- **Context supervision:** confidence-weighted `KL(q_teacher || p_router)` is added to the UAV actor optimization. Its coefficient anneals from 0.05 to 0.01 over 500k sampled steps. Encoder features are detached for this auxiliary loss: it directly updates only router parameters; ordinary PPO gradients still train the complete actor.
- **Temporal regularization:** confidence-weighted squared probability difference between consecutive router distributions, with the previous distribution stopped-gradient. Valid pairs require both states active, no episode boundary, no kill/death event, stable engagement and threat target identities, and unchanged teacher mode. The router-only step occurs after that actor's PPO updates and before recomputing probabilities for the next HAPPO preceding factor. Coefficient is 0.01 in Full TACM and 0 in the no-temporal configuration.
- **Figure abstraction:** the temporal box includes rollout router distributions, events, and target/teacher identity records. Input/reward labels summarize rollout records rather than claim rewards are critic-network inputs: critics consume state/observations; rewards feed GAE targets. The optional reset randomization curriculum is experiment infrastructure, not part of the actor network, and is intentionally omitted.
- **Legend:** solid lines denote forward/execution dataflow; dashed lines denote training-only supervision or policy updates; dotted lines denote temporal regularization. Dashed shaded regions contain training-only components. Both context and temporal losses use detached encoder features, not an additional inference encoder.

## Code/config verification

Checked against `algorithm/happo/tacm_rgaa.py`, `algorithm/happo/dbm_rgaa.py`, `algorithm/happo/rgaa.py`, `algorithm/common/networks.py`, and the actor-update/rollout paths in `algorithm/happo/trainer.py`; entrypoints `algorithm/train_tacm_rgaa.py` and `algorithm/train_happo_rgaa.py`; configs `happo_tacm_rgaa_v310.yaml`, `happo_tacm_rgaa_v310_no_temporal.yaml`, and `happo_rgaa_v310_curriculum.yaml`; and `docs/tacm_rgaa_v1_spec.md`. The older v1 spec describes v3.9; the figure explicitly uses the requested current v3.10 contract rather than silently mixing the two versions.

## Suggested main-text reference

Place the figure near the beginning of the **Method / TACM architecture** subsection, before the detailed teacher and temporal-loss definitions. Suggested sentence: “Figure X summarizes the centralized training framework and the decentralized UAV actor; training-only tactical supervision shapes the soft router without introducing privileged information into execution.”

## Regeneration and formats

Run `python tools/render_tacm_architecture.py` from the repository root (or invoke the script by absolute path). Only matplotlib is required. Outputs are `tacm_architecture.svg` (editable vector text), `tacm_architecture.pdf` (vector paths and embedded fonts), and `tacm_architecture.png` (350 dpi). The source canvas is 14.4 inches wide; at a 7.2-inch double-column width, principal 14 pt labels become 7 pt. Prefer the SVG/PDF for typesetting. No training, model loading, or evaluation is performed.
