# Combat episode visualization and interactive replay

本工具链将执行和展示分为两层。`record_combat_episode.py` 是唯一会加载 checkpoint、创建环境并运行 policy 的程序；两个 renderer 只读取 `episode_trace.npz` 与 `metadata.json`，不会加载 checkpoint、创建环境或使用 CUDA。这保证重复调整画面不会改变环境、算法、Blue policy 或训练语义。

## 记录真实 episode

```powershell
conda run -n uav python -u tools/record_combat_episode.py `
  --checkpoint outputs/<run>/checkpoint_final.pt `
  --profile main --blue-mode nearest --seed 424242 `
  --device cuda --output-dir outputs/visualization/example
```

`--blue-mode` 只接受 `nearest` 或 `mav_priority`，一条 trace 对应一种环境内部 Blue policy。输出目录默认必须不存在或为空；只有显式 `--overwrite` 才允许复用。种子 `424242` 是独立的 qualitative seed，不属于正式评估使用的 `1000+episode` 序列。元数据固定写入 `qualitative_visualization_only` 和 `used_for_quantitative_metrics: false`。

loader 严格检查 environment version、100D observation 和 119D global state。支持：

- vanilla：从 `trainer_config/config.hidden_dim` 构造 `IndependentActors`，恢复 `payload["actors"]`；baseline、AGP、curriculum 和 AGP-curriculum 只影响方法显示名，不改变推理动作。
- HRTA：必须有精确的 `actor_architecture` 四字段，构造 `HRTAIndependentActors`。
- Structured Uniform：同样严格验证四字段并构造 `StructuredUniformIndependentActors`。
- R-HAPPO recurrent：仅接受 `actor_variant=recurrent` 与 `method_variant=baseline`，严格验证 `observation_dim/encoder_dim/recurrent_hidden_dim/head_dim/action_dim` 五字段并构造 `RecurrentIndependentActors`。

每个 decision 按 `MAV, UAV1, UAV2, UAV3` 顺序读取 observation。前馈 variant 继续调用 `sample(..., deterministic=True)`；R-HAPPO 与正式 recurrent evaluator 使用相同的 `sample_step(..., deterministic=True)` hidden lifecycle：每个独立录像 episode 从 zero hidden 和 zero recurrent mask 开始，Red agent 死亡时只清对应 hidden，Blue 死亡不清理任何 Red hidden，episode 结束时统一清零。checkpoint 中用于训练续跑的 `rollout_state.actor_hidden_states` 不会被录像恢复。Blue 动作仍完全由 `env.blue_policy` 在 `env.step` 内生成。

## Trace 合同

schema version 为 2。frame 0 是 `env.reset` 之后、第一次 actor action 之前的真实状态；每次 `env.step` 完成后追加一帧，所以 `F = episode_steps + 1`。transition `i` 的 attack/death/safety/reward 属于 `trace_frame=i+1`。

`kinematics` 是 `[F,8,6]`，顺序为 `[x,y,h,v,theta,psi]`；单位是 `[m,m,m,m/s,rad,rad]`。`h` 本身就是 positive-up altitude，渲染始终使用 `altitude=h`，绝不转换为 `-h`。`alive` 为 `[F,8]`，`red_actions` 为 `[F-1,4,3]`，并保存各 reward 分量、最小 Red 间距及 separation warning。

NPZ 中没有 pickle object array。可变长 attack、death、boundary、blue escape 与 safety 事件保存在 JSON。若同一步有多个 attacker 指向同一 target，界面逐条显示 `ATTACK` 和单独的 `DESTROYED [cause]`；环境没有提供唯一 killer，因此工具不会臆造唯一击杀者。

## 视觉插值

真实 trace 保持 1.0 s decision boundary，不复制 RK4 循环。renderer 默认临时创建 0.1 s display timeline：`x/y/h/v` 线性插值，`theta/psi` 使用 shortest-angle 插值。`179° -> -179°` 经过约 2° 的短路径。alive 使用最近已发生真实 boundary 的值，因此死亡、death marker、attack line 和 Recent Events 不会提前泄露。元数据明确标记这些 display frames 是 visual interpolation，而不是真实 physics rollout。

## 固定视角 MP4

```powershell
conda run -n uav python tools/render_combat_episode.py `
  --input-dir outputs/visualization/example --visual-dt 0.1 --fps 20 `
  --trail-seconds 10 --elev 27 --azim -55
```

生成 `preview.png` 和 `episode.mp4`。`--preview-only` 不需要 ffmpeg；完整 MP4 需要系统已有 ffmpeg，工具不会自动安装。可用 `--no-heading` 关闭朝向。固定 MP4 camera 不支持鼠标操作。

## Offline interactive HTML

```powershell
conda run -n uav python tools/render_combat_episode_interactive.py `
  --input-dir outputs/visualization/example
```

生成 `episode_interactive.html`，Plotly JS 和 episode payload 均内嵌，不使用 CDN 或服务器。双击即可播放；支持 Play/Pause、前后帧、Restart、display-time slider、0.25–4x speed、Full/5/10/20 s trail、heading/label/death/attack 开关、Episode View、Full Battlefield 和 Reset Camera。Plotly 提供鼠标 rotate/zoom/pan，`uirevision` 使播放和开关操作保持用户 camera。

默认 Episode View 是一次性根据整场 finite raw 轨迹计算的固定立方体。水平范围保留 25% 总余量且至少 10 km；顶部保留至少 1 km 或最高高度 10% 的余量；最终统一 span 向上取整至 0.5 km。X、Y、Altitude 三轴数值 span 完全相等，Z 固定为 `[0, cube_span]`，Plotly 使用 `aspectmode='cube'`，Matplotlib 使用 `set_box_aspect((1,1,1))`。播放、slider、trail、事件和死亡均不会重新计算 world ranges；camera 仍可自由旋转、缩放和平移。

`z=0 km` 的浅灰平面只是 ground reference，用来表达飞机真实离地高度。环境合法最低高度仍是 config 中的 1 km，ground plane 不是环境边界。Full Battlefield 保留 config 的 `X/Y=[-100,100] km` 与 `Altitude=[1,20] km` 作为诊断视图，切换视图和 Reset Camera 互不绑定。

### Live camera interaction

播放时可以持续 rotate、orbit、pan 或滚轮 zoom。Logical replay 使用 `performance.now()`、真实 `time_s` 和 `requestAnimationFrame` 独立推进，不等待 Plotly 绘制完成。camera manipulation 期间不显示额外提示，也不隐藏或重新创建飞机、轨迹、heading、death marker 和 attack line；当前完整 combat snapshot 始终可见，新 combat render 暂停进入 scheduler。松开 pointer 或滚轮停止约 200 ms 后，仅在 logical frame 已推进时重建一次最新状态，并通过 `uirevision` 保留新的 camera 角度；暂停状态下移动视角不会触发 combat refresh。红方 MAV/UAV 统一使用红色，蓝方无人机统一使用蓝色，机型身份继续由 marker、线宽、虚实线和标签区分。这一显示策略不改变 episode 数据、事件或时间语义。

MAV 使用深红菱形和更粗实线，UAV1/UAV2/UAV3 使用同阵营红色圆形，Blue1/Blue2/Blue3/Blue4 使用同阵营蓝色三角与虚线。坐标显示 km，hover 速度保留 m/s，并从 config 元数据读取允许速度和过载范围。顶部显示 evaluation profile 与 Blue policy；右侧显示 Current State、最多五条已发生 Recent Events，Result 只在最后一帧出现。真实 attack line 在事件发生后 0.8 s display time 内可见，拖动 slider 返回该区间时会重现。
