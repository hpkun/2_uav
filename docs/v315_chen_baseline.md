# v3.15 Chen-aligned Heterogeneous HAPPO Baseline

## 冻结范围与来源

版本：`heterogeneous_mavuav_4v4_v3_15`，reward：`chen_heterogeneous_v1`。
主要依据：Chen et al., 2026, *A deep reinforcement learning cooperative air combat
method with temporal feature and attention enhancement*, Aerospace Science and
Technology 176:112537，PDF Table 1(p8)、权重与训练参数(p9)、situation assessment(p4)。
这是普通3D环境上的明确映射，不是JSBSim/导弹环境的完整复现，也不是TAM-HAPPO。

- **PAPER_DIRECT**：UAV angle/speed/distance分段、10/15/10权重、+200 kill、
  −200 combat loss、−100 boundary；MAV safety/support结构及其原始系数；
  target assessment的0.35/0.25/0.20/0.20；actor/critic LR5e-4、entropy0.01、grad clip10。
- **ENV_MAPPING**：distance以km代入论文分段；target distance indicator使用当前weapon5km，
  altitude/relative velocity用当前10000/800尺度；MAV danger/safe=5000/10000m；
  TA映射为Blue→MAV的ATA；C_d/C_k/C_max=200/50/200；local reward固定平均后作为共享scalar。
  这些具体映射不是论文给出的全部原始参数或advantage实现。
- **DISABLED_UNAVAILABLE**：height（P_V/P_H定义不充分）、dodge和MAV missile threat
  （没有missile entity/状态）、position（center/d_opt/d_max映射不足）、awareness
  （全向sensor没有足够严格AO等价定义）。关闭项诊断为0，不重新归一化其余系数。

动力学、固定初始位置/250m/s、速度150–300、感知12/5km、datalink、CAP-Blue、
150步、MAV unarmed、single-target lock、0<distance<=5km、ATA<30°、AA<90°、
hold1全部继承v3.14；不增加missile/ammo/cooldown，不改变任何历史环境/算法合同。

## Local reward

对每个存活UAV，从alive且team-visible Blue选择最大

`E=.35[1−(ATA+AA)/(2π)]+.25 I(d<=5000)+.20(z_red−z_blue)/10000+.20||v_red−v_blue||/800`。

相同分数按canonical Blue ID；**E只选择目标，不进入reward**。无目标时三个dense项均0。

`r_Ui = 10 R_speed + 15 R_angle + 10 R_distance + R_event,i`

- `R_angle=1−(ATA+AA)/π`，不clip。
- d转km：`R_distance=1`(d<=5)，`exp[−.921(d−5)]`(5<d<10)，`−1`(d>=10)。
- `R_speed=1`(Vb<.5Vr)，`2−2Vb/Vr`(.5Vr<=Vb<=1.5Vr)，`−1`(Vb>1.5Vr)。
- 事件来自真实`attack_events`与`death_causes`：每架新unique Blue毁伤总kill credit+200，
  blue_attack−200，boundary−100。同一transition合法事件相加；相同(UAV,Blue)重复记录只付一次。
  **ENV_MAPPING**：同步有效UAV co-attackers均分该Blue的+200，即每个获得`200 / |A_b|`。
  论文没有规定同步多攻击者的归因方法；这是环境映射，不凭距离推断killer，不修改combat attribution。
  多架Blue分别结算，已结算Blue不重复支付。MAV的team contribution仍只按unique Blue计一次。

`r_M = .5 R_dist + .2 R_aspect + R_event,M`，support=0，threat=0。

- d取MAV自身direct-visible的最近alive Blue，不使用隐藏敌机；无direct-visible Blue时R_dist=.2。
- R_dist：d<5000时`−(1−d/5000)`；5000<=d<10000时
  `−.5[1−(d−5000)/5000]`；d>=10000时+.2。保留原分段边界，不做平滑。
- R_aspect：遍历MAV direct-visible alive Blue，TA=ATA(Blue→MAV)，对TA<π/4累加
  `−[1−TA/(π/4)]`；其他0。
- MAV loss−200（boundary或combat）。新的unique Blue red_attack death给+50，
  每episode累计贡献最多200；重复不计。MAV失活后不累计以后事件。

Dense reward快照继承clean路径的时间点：physics→boundary→dense snapshot→同步combat→reward。
Boundary失活agent没有dense reward，但当前transition仍收到loss event；同步combat失活agent
可保留pre-combat dense和该步合法事件。MAV同一步combat死亡与team kill可同时结算；不累计未来kill。

`r_team=(r_M+r_U1+r_U2+r_U3)/4`；四个Red收到同一个r_team。
一个shared team critic、一个team GAE；没有role critics/role GAE。
原clean guide、±10旧events、±20terminal全部禁用。胜负/终止不变，terminal reward恒0。

Info保留四local reward、所有开启/关闭component、target/score、三类UAV event金额、
MAV cumulative contribution和team reward。Episode summary保留四local累计。
`event_reward`是四local event的固定平均，方便team return会计，绝非再叠加一次shared event。

## v3.15-only HAPPO probability / value contract

`u~Normal(mu,sigma); a=tanh(u)`。环境接收a；`LatentRolloutBuffer`保存a、raw u和
old latent Gaussian log_prob。Actor PPO与preceding factor均直接调用
`evaluate_raw_actions(o,u)`，不atanh、不clip后反推。
确定性tanh的Jacobian在新旧概率比中相消。历史`sample/evaluate_actions`完全保留。
新API`sample_with_raw`没有额外采样；三个接口不改变历史actor state_dict结构。

Entropy是`Normal.entropy().sum(-1)`，metadata为`latent_gaussian`。
它是regularization surrogate，不是tanh后bounded-action entropy。

ValueNorm使用HARL式debiased EMA moments：beta=.99999、epsilon=1e-5、variance floor=.01。
统计用float64 buffer；预测/GAE保留原dtype。每完整rollout在线更新一次return统计，
该rollout的PPO critic epochs使用固定统计，不在各minibatch之间移动尺度。
这项更新频率显式记录为实现选择，并不声称逐行复制HARL。

- Critic输出normalized value；收集时原样保存至v3.15-only `old_normalized_values`用于value clipping。
  同时denormalize后写`values`，bootstrap也denormalize后用于GAE。
- buffer rewards、values、returns、advantages均为environment尺度。
- critic optimization开始时更新统计，仅将raw returns转换为当前normalized targets。
  old clipping baseline直接读取采样时保存的normalized critic prediction，绝不以新统计重新归一化raw old values。
- clipped prediction为old+clip(new−old,−.2,.2)。delta10的Huber采用
  `0.5 e²`(|e|<=10)，否则`10(|e|−5)`；clipped/unclipped逐元素取max再mean。
- value_loss_coef仍.5；Adam不更换，不改HAPPO随机顺序、preceding factor或active mask。

新建v3.15网络使用orthogonal initialization：hidden gain=sqrt(2)，actor mean gain=.01，
critic output gain=1，bias全0；log_std按配置−.25。仍为hidden128四独立MLP actors/一个MLP critic，
不增加GRU或attention。v3.14及历史variant不会执行新初始化。

参考实现（只参考ValueNorm、Huber/clipped loss，不迁移其他算法模块）：
[HARL ValueNorm](https://github.com/PKU-MARL/HARL/blob/main/harl/common/valuenorm.py)、
[HARL value critic](https://github.com/PKU-MARL/HARL/blob/main/harl/algorithms/critics/v_critic.py)。

## Config / checkpoint / evaluator

使用`configs/env_v315.yaml`与`configs/happo_v315_baseline.yaml`，不要覆盖v3.14配置。
公共固定参数：gamma.99、GAE.95、clip.2、16env、rollout128、PPO4、minibatch256、
hidden128、actor_log_std_init−.25、无curriculum。

新checkpoint的顶层及trainer config都保存并严格校验：
`action_probability_protocol=latent_gaussian_raw_v1`、`entropy_semantics=latent_gaussian`、
`use_valuenorm=true`、`use_huber_loss=true`、`huber_delta=10`、
`use_clipped_value_loss=true`、`orthogonal_init=true`。
environment/reward/actor/critic/method/weapon metadata继续验证。

ValueNorm完整moments/debiasing state进入training和weights-only checkpoint；exact resume
还恢复optimizer、RNG、env states/weapon locks/Chen contribution去重集合与累计、reset count、
observations、global state、active masks、sampled steps。保存仍在rollout边界，format不改。
v3.14和v3.15拒绝交叉加载。独立evaluator验证并加载ValueNorm state，执行仍只使用actors。
Resolved config、summary和standalone evaluation输出新协议字段。

## 验证范围

只运行unit/regression和极短CPU/CUDA smoke，不启动正式2M训练。
人工边界覆盖5/10km、速度分段、MAV danger/safe、event/unique contribution、
raw±8/±20 probability identity、scale一致的critic loss、CPU/CUDA exact resume。
CUDA smoke使用真实16 subprocess env，两步rollout/update及1局evaluator读取测试。
历史v3.13/v3.14/CAP/weapon/recurrent/TAM/ERAM/PCTA/RGAA等由原测试回归验证。
