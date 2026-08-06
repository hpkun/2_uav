# v11 Target-Lock Support-Cue Environment Report

## Scope

Implemented the new environment contract:

- Variant: `functional_heterogeneous_4v3_v11_target_lock_support_cue`
- Reward contract: `v11_target_lock_support_cue`
- Config: `configs/heterogeneous_4v3_main_v11_target_lock_support_cue.yaml`
- HAPPO config: `configs/happo_heterogeneous_4v3_main_v11_target_lock_support_cue.yaml`

No formal 3M training, long training, multi-seed RL training, hyperparameter search, JSBSim, missile, probability-hit, noise, delay, GRU, attention, curriculum, imitation learning, self-play, or new action dimension was used.

## Environment Contract

Red and blue Combat aircraft are constructed from the same `aircraft` hardware specification and the same `combat_profile`. The v11 validator checks the six Combat aircraft for the same dynamics, sensor range, lock distances, ATA/AA fades, increment/decay, threshold, target hold, release, and switch ratio. The only asymmetries are policy control, red Support, and Support cue availability.

Final lock parameters after the two allowed calibrations:

- sensor range: `2500m`
- lock distance: `100-1500m` optimal, fades to zero at `2500m`
- ATA fade: `90 degrees`
- AA fade: `150 degrees`
- lock increment: `0.17`
- lock decay: `0.03`
- kill threshold: `1.0`
- minimum target hold: `30` steps
- lost-target release: `10` consecutive steps
- target switch ratio: `0.70`

The final values differ from the initial requested suggestion only because the two permitted reachability calibrations were used. Both sides continue to use exactly the same values.

Lock quality is:

`distance_score * clip(1 - ATA / 90 degrees, 0, 1) * clip(1 - AA / 150 degrees, 0, 1)`

Only direct Combat visibility can increase lock. A Support cue can select a target and expose target geometry, but cannot increase lock or directly kill. Lock is continuous, decays when direct visibility or quality is lost, and produces a deterministic kill at threshold.

Support assigns cues every 20 steps using a greedy nearest pair assignment. A target is assigned to one Combat first; if targets are fewer than Combat aircraft, remaining Combat aircraft may share it. Existing lock >= 0.25 is preserved. Support death stops new cues and leaves current targets until they expire.

Termination records `red_full_elimination`, `red_total_loss`, `timeout_red_win`, `timeout_red_loss`, and `timeout_draw`. Timeout compares only red/blue Combat survivors; Support is excluded. `task_win` is full elimination or timeout red win.

## Reward and Logging

The v11 components are:

`mission_outcome_reward`, `blue_kill_event_reward`, `red_combat_loss_event_penalty`, `support_loss_event_penalty`, `boundary_event_penalty`, `combat_geometry_progress_reward`, `combat_lock_progress_reward`, `combat_half_lock_event_reward`, `support_unique_detection_reward`, `support_cue_to_direct_reward`, `support_cue_to_half_lock_reward`, `support_assisted_kill_reward`, `support_formation_progress_reward`, `total_dense_reward`, and `team_total_reward`.

Geometry and lock rewards use differences only. Static formation, static coverage, static position, and static shared-pair state do not generate continuous reward. Dense components are clipped to `[-0.05, 0.05]`; mission and event rewards are not clipped. All components are included in `info["reward_components"]`, vector results, training CSV, episode summaries, evaluation summaries, and checkpoint state.

The v11 train log is emitted as one ordinary line per update, for example:

`[train] step=2048/8192 upd=1 fps=303.9 r_step=-0.02609 ep_return=-25.18 win=0.000 full=0.000 kills=0.000 timeout=0.000 reward{mission=-0.00732 event=-0.02441 geom=-0.00015 lock=-0.00000 support=+0.00595}`

`r_step` is rollout mean team reward per environment step. `ep_return` is the mean completed-episode return. The CSV includes these values, v11 rolling task metrics, lock/cue metrics, and `mean_rollout_<component>` for every v11 component.

## Baseline Reachability

The same fixed 100 seeds (`20000-20099`) were used for both final baseline runs.

| Red policy | Task win | Full elimination | Any kill | >=2 kills | Mean kills | Lock episode | Support assist | Mean return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule vs rule | 0.52 | 0.25 | 0.67 | 0.45 | 1.37 | 1.00 | 0.6423 | 10.4358 |
| Random vs rule | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.0000 | -27.0746 |

Rule termination split was timeout win `0.27`, timeout loss `0.16`, timeout draw `0.11`; timeout rate was `0.54`. The red kill distribution was `0:33%`, `1:22%`, `2:20%`, `3:25%`. Rule first-kill time averaged `328.64` steps. Support cue-to-direct rate was `0.9913`.

Random produced `2.49` mean blue kills against red and no red kill, full elimination, or task win.

The rule/random gap is clear, but the requested rule any-kill threshold is `>=0.70` and the measured final value is `0.67`. The rule full-elimination, task-win, two-plus-kill, and lock-episode thresholds pass. Since both permitted calibrations were exhausted, this report does not claim that v11 has fully passed the formal reachability gate.

Calibration history:

1. Initial `lock_increment_scale=0.15`, `lock_decay_per_step=0.05`: rule any-kill `0.69`.
2. Calibration 1: increment `0.17`, decay `0.05`: rule any-kill `0.67`.
3. Calibration 2: increment `0.17`, decay `0.03`: rule any-kill `0.67`, lock episode `1.00`.

The before/after files are retained as `outputs/v11_target_lock_support_cue_rule_100.json`, `outputs/v11_target_lock_support_cue_rule_100_calibration1.json`, `outputs/v11_target_lock_support_cue_rule_100_calibration2.json`, and `outputs/v11_target_lock_support_cue_rule_100_final.json`, with corresponding random files.

## CPU Smoke and Tests

The final 4-worker CPU smoke used the v11 HAPPO config with `--smoke`:

- exactly `8192 env steps`
- exactly `4 updates`
- device `cpu`
- all four Actor slots updated
- finite losses, KL, entropy, gradients, and reward components
- actual terminal `r_step`, `ep_return`, mission/event/geom/lock/support log groups
- v11 CSV fields and checkpoint/resume artifacts present
- output: `outputs/happo_v11_cpu_smoke_workers4`
- final smoke throughput: about `311.5 env steps/s`

Validation results:

- full pytest: `512 passed`
- v11 focused tests: `31 passed`
- v9/v10 4v3 and HAPPO milestone regression: `64 passed`
- compileall: passed

Existing v9/v10 environment semantics and configuration files were not changed. No formal v11 RL training was run.

