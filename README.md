# MAV/UAV 4v4 Air-Combat Research Environment

本项目包含异构 `1 MAV + 3 UAV vs 4 Blue` 环境、vanilla HAPPO/MAPPO 实现，以及独立的评估和诊断工具。正式研究代码位于 `env/` 与 `algorithm/`，不需要安装当前项目 package。

当前 canonical contract 为 `heterogeneous_mavuav_4v4_v3_0`，actor observation 为 100D，centralized state 为 119D。旧 3v2/v2.2 checkpoint 与结果仅作为历史实验保留，不能续跑到 4v4，也不能与 4v4 baseline 直接合并比较。

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

训练完成后才分别对 `nearest` 和 `mav_priority` 做 final deterministic evaluation。checkpoint 保存频率和日志频率均不会触发额外评估。

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
    --episodes 100 \
    --blue-mode both
```

评估允许训练 profile 与 evaluation profile 不同，用于跨 profile 泛化检查；环境版本、100D observation 和 119D global state contract 仍会严格校验。

## R-HAPPO 基线

R-HAPPO 使用四个独立 GRU Actor，并保持现有 119D centralized MLP Critic 与环境语义不变。训练和独立评估入口为：

```bash
python algorithm/train_happo_recurrent.py --steps 5000000 --profile main --seed 1 --device cuda --num-envs 16
python algorithm/evaluate_happo_recurrent.py outputs/<run>/checkpoint_final.pt --profile main --episodes 100 --blue-mode both --device cuda
```

其 recurrent mask、TBPTT、短尾 chunk 和 checkpoint continuation 语义见 `docs/recurrent_happo_spec.md`。

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
python tools/plot_trajectory.py outputs/<run>/checkpoint_final.pt \
    --profile main --blue-mode nearest --seed 1000
```

轨迹图片默认直接写入 checkpoint 所在 run folder。诊断辅助函数集中在 `tools/diagnostics.py`，核心环境和 HAPPO trainer 不依赖 `tools/`。

完整 combat replay（建议使用与正式评估 `1000+episode` 隔离的定性种子）：

```bash
python tools/record_combat_episode.py --checkpoint outputs/<run>/checkpoint_final.pt \
    --profile main --blue-mode nearest --seed 424242 --output-dir outputs/visualization/example
python tools/render_combat_episode.py --input-dir outputs/visualization/example
python tools/render_combat_episode_interactive.py --input-dir outputs/visualization/example
```

前者只记录真实 decision-boundary 状态；后两者分别生成固定视角 MP4/preview 和可离线双击打开的交互 3D HTML。原有 `plot_trajectory.py` 静态 PNG 用法保持不变。详见 `docs/combat_episode_visualization.md`。

Combat replay loader 支持 vanilla HAPPO、HRTA、Structured Uniform 和 baseline R-HAPPO recurrent checkpoint。R-HAPPO 录像从独立 episode 的 zero hidden/zero mask 开始，使用 deterministic `sample_step`，仅按 Red `active_masks` 做 agent-level hidden reset；它仍是定性可视化，不替代正式 recurrent evaluation。

## 测试

```bash
python -m pytest -q
```

pytest 从项目根目录直接导入 `env.*` 和 `algorithm.*`，同样不要求安装当前项目 package。
