# MAV/UAV 3v2 Air-Combat Research Environment

本项目包含异构 `1 MAV + 2 UAV vs 2 Blue` 环境、vanilla HAPPO/MAPPO 实现，以及独立的评估和诊断工具。正式研究代码位于 `env/` 与 `algorithm/`，不需要安装当前项目 package。

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
    --num-envs 16
```

默认每 1,000,000 sampled environment steps 保存 checkpoint，中间 evaluation 默认关闭。训练完成后才分别对 `nearest` 和 `mav_priority` 做 100 episodes deterministic evaluation，因此 checkpoint 保存频率不会触发额外评估。

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

评估允许训练 profile 与 evaluation profile 不同，用于跨 profile 泛化检查；环境版本、55D observation 和 67D global state contract 仍会严格校验。

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

## 测试

```bash
python -m pytest -q
```

pytest 从项目根目录直接导入 `env.*` 和 `algorithm.*`，同样不要求安装当前项目 package。
