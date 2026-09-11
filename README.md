# MAV/UAV 4v4 Air-Combat Research Environment

本项目包含异构 `1 MAV + 3 UAV vs 4 Blue` 环境、vanilla HAPPO/MAPPO 实现，以及独立的评估和诊断工具。正式研究代码位于 `env/` 与 `algorithm/`，不需要安装当前项目 package。

当前最终冻结的 canonical contract 为 `heterogeneous_mavuav_4v4_v3_5`，actor observation 为 100D，centralized state 为 117D。场景为 1 MAV + 3 Red UAV 对 4 Blue-team UAV；红蓝 UAV 使用完全相同的动力学参数，八架飞机名义初速统一为 275 m/s，MAV 在三架 Red UAV 前线后方 1 km。四架 Blue 各自选择最近存活 Red（包括 MAV），每两个 decision steps 刷新一次目标时刻的 heading/pitch，并在两步之间保持这组 guidance。Blue 使用 privileged true state；这与 Red 的 12/8 km sensing 和理想 datalink 是刻意保留的 benchmark 信息结构非对称。v3.4 及更早 checkpoint 不能续跑、评估、回放或审计。

v3.5 相对 v3.4 唯一的科学行为变化是上述 Blue guidance hold。正式 v3.4 Vanilla HAPPO 2M 结果为 0% Red win、0 Red kills、100% draw，表明每步实时 pure pursuit 与 distance/ATA/AA/hold 攻击门形成了结构性退化几何。两步周期规则用于解除这种规则控制器与攻击判据的结构性耦合，同时保留固定规则对手和 O(1) 高速控制；这不是为了直接调高 Red 胜率。

## 环境准备

在项目根目录安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

无需执行 `pip install -e .`。

## HAPPO 训练

短运行：

```bash
python algorithm/train_happo.py \
    --steps 4096 \
    --profile learnability \
    --device cpu \
    --num-envs 2
```

正式运行：

```bash
python algorithm/train_happo.py \
    --steps 5000000 \
    --profile main \
    --seed 1 \
    --device cuda \
    --num-envs 16 \
    --checkpoint-interval 1000000 \
    --eval-interval 0 \
    --log-interval 100000
```

默认每 1,000,000 sampled environment steps 跨过 checkpoint milestone 后保存 checkpoint，中间 evaluation 默认关闭。checkpoint 和 evaluation milestone 都只在完整 rollout/update 完成后检查，不会截断正常 rollout；只有为了精确到达最终 `--steps` 才允许最后一次 partial rollout。

`--log-interval` 默认每约 100,000 sampled steps 输出一次训练进度。它只汇总训练期间已经完成的 episodes 和最近 HAPPO updates，不运行 evaluation，也不会改变 rollout horizon。stdout 与 `run.log` 内容一致，可实时查看：

```bash
tail -f outputs/<run>/run.log
```

训练完成后只对 canonical `nearest_red_aircraft` 对手做 final deterministic evaluation。checkpoint 保存频率和日志频率均不会触发额外评估。

如需中间评估，显式传入例如：

```bash
python algorithm/train_happo.py --eval-interval 1000000
```

断点续训使用原 run folder，不创建新目录：

```bash
python algorithm/train_happo.py \
    --steps 10000000 \
    --profile main \
    --seed 1 \
    --device cuda \
    --num-envs 16 \
    --resume outputs/<run>/checkpoint_5000000.pt
```

长训练可由用户自行用 `nohup`、systemd、tmux 等系统方式放到后台；训练代码本身不绑定后台管理框架。

## 独立评估

```bash
python algorithm/evaluate_happo.py \
    outputs/<run>/checkpoint_final.pt \
    --profile main \
    --episodes 100
```

评估允许训练 profile 与 evaluation profile 不同，用于跨 profile 泛化检查；环境版本、100D observation 和 117D global state contract 仍会严格校验。

## R-HAPPO 基线

R-HAPPO 使用四个独立 GRU Actor，并保持现有 117D centralized MLP Critic 与环境语义不变。训练和独立评估入口为：

```bash
python algorithm/train_happo_recurrent.py --steps 5000000 --profile main --seed 1 --device cuda --num-envs 16
python algorithm/evaluate_happo_recurrent.py outputs/<run>/checkpoint_final.pt --profile main --episodes 100 --device cuda
```

其 recurrent mask、TBPTT、短尾 chunk 和 checkpoint continuation 语义见 `docs/recurrent_happo_spec.md`。

## PCTA-HAPPO

PCTA-HAPPO 在四个 Blue 固定槽上加入 target-aware attention，并仅在前一主要目标仍存活且 team-visible 时施加 temporal pursuit-consistency regularization；critic、HAPPO sequential update 与环境均不变：

```bash
python algorithm/train_happo_pcta.py --steps 5000000 --profile main --seed 1 --device cuda --num-envs 16
python algorithm/evaluate_happo_pcta.py outputs/<run>/checkpoint_final.pt --profile main --episodes 100 --device cuda
```

结构、损失与诊断字段见 `docs/pcta_happo_spec.md`。

## 输出结构

`outputs/` 下每次训练只对应一个自包含 run folder：

```text
outputs/happo_main_seed1_5m_<timestamp>/
├── run.log
├── resolved_config.yaml
├── training.csv
├── evaluations.csv
├── summary.json
├── checkpoint_1000000.pt
├── ...
└── checkpoint_final.pt
```

不会再创建 `happo_seed1/` 或 `checkpoints/` 子目录。整个 run folder 可以直接复制到其他机器保存或分析。

## 工具

```bash
python tools/audit_env.py --steps 1000 --num-envs 16
python tools/benchmark_env.py --sample-steps 2000 --num-envs 16
python tools/audit_environment_foundations.py --profile main --samples 10000 --seed 1000 \
    --output outputs/foundation_v35_main
python tools/audit_blue_guidance_geometry.py --profile main --base-seed 1000 --seeds 20 \
    --output-dir outputs/blue_guidance_geometry_v35
python tools/plot_trajectory.py outputs/<run>/checkpoint_final.pt \
    --profile main --seed 1000
```

轨迹图片默认直接写入 checkpoint 所在 run folder。诊断辅助函数集中在 `tools/diagnostics.py`，核心环境和 HAPPO trainer 不依赖 `tools/`。

完整 combat replay（建议使用与正式评估 `1000+episode` 隔离的定性种子）：

```bash
python tools/record_combat_episode.py --checkpoint outputs/<run>/checkpoint_final.pt \
    --profile main --seed 424242 --output-dir outputs/visualization/example
python tools/render_combat_episode.py --input-dir outputs/visualization/example
python tools/render_combat_episode_interactive.py --input-dir outputs/visualization/example
```

前者只记录真实 decision-boundary 状态；后两者分别生成固定视角 MP4/preview 和可离线双击打开的交互 3D HTML。原有 `plot_trajectory.py` 静态 PNG 用法保持不变。详见 `docs/combat_episode_visualization.md`。

Combat replay loader 支持 vanilla HAPPO、PCTA-HAPPO、HRTA、Structured Uniform 和 baseline R-HAPPO recurrent checkpoint。R-HAPPO 录像从独立 episode 的 zero hidden/zero mask 开始，使用 deterministic `sample_step`，仅按 Red `active_masks` 做 agent-level hidden reset；它仍是定性可视化，不替代正式 recurrent evaluation。

## 测试

```bash
python -m pytest -q
```

pytest 从项目根目录直接导入 `env.*` 和 `algorithm.*`，同样不要求安装当前项目 package。
