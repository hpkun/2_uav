# v11 Target-Lock Support-Cue Final Mechanism Closeout

## Scope and restrictions

This closeout keeps the v11 variant and reward contract unchanged:

- variant: `functional_heterogeneous_4v3_v11_target_lock_support_cue`
- reward contract: `v11_target_lock_support_cue`
- no lock, scenario, sensor, reward coefficient, observation dimension, state dimension, HAPPO, PPO, GAE, or learning-rate change
- no formal RL training, 1M training, 3M training, long training, or parameter calibration was run in this closeout

The changes fix implementation errors only: reward double counting, actor-side hidden-state leakage, repeated transition updates, rule-action mutation, mixed lock metrics, invalid cue-rate accounting, blue target priority, mutual-elimination classification, raw dense logging, reward log grouping, and fixed v9/v10 SHA protection.

## Mechanism fixes

### Reward aggregation

The old implementation added `total_dense_reward` and then added geometry, lock, and formation components again. The corrected formula is:

`raw_dense = geometry + lock + formation`

`total_dense = clip(raw_dense, -0.05, 0.05)`

`team_total = mission + combat_events + support_events + half_lock_event + total_dense`

The raw dense subcomponents remain in `reward_components` for diagnosis, but they enter team reward only through `total_dense_reward`. A centralized `aggregate_team_reward_v11()` and `reward_group_totals_v11()` implement the formula.

`info["raw_dense_reward"]` now contains the unclipped value. `reward_components["total_dense_reward"]` contains the clipped value. Episode mean/min/max and saturation rates use the raw value.

### Observation contract

The observation shape remains `(7, 118)` and global state remains `(70,)`.

- The generic other-aircraft block now exposes relative state only for same-team aircraft.
- Enemy aircraft in that block are all zeros.
- The visibility-masked enemy block exposes direct geometry or Support-shared geometry only.
- A retained hidden target keeps only internal lock/hold fields; source flags and target distance are zero.
- Direct and shared target source flags are preserved.
- The centralized critic still receives the unchanged global state.

### Transition timing and rule actions

Reset initializes Support cue and targets once without incrementing hold/lost counters. Each physical transition uses the prepared targets, performs dynamics and lock processing, then updates Support cue once and target state once for the next transition. Hold and lost counters therefore increase at most once per transition.

`red_rule_actions()` now only reads current state and current targets and returns actions plus a target snapshot. It does not update cue, target, lock, counters, or episode metrics.

### Blue policy

Blue Combat still uses direct visibility and the same aircraft, lock, hold, release, and kill mechanics. Its candidate priority is now:

1. directly visible red Combat aircraft;
2. directly visible red Support only when no red Combat is directly visible.

No global state, future trajectory, or Support cue is used by the blue rule.

### Lock and Support metrics

Lock metrics are now tracked independently for red and blue Combat:

- `red_lock_episode_rate`, `red_half_lock_episode_rate`, `mean_red_max_lock_progress`
- `red_lock_active_step_rate`, `red_half_lock_active_step_rate`
- equivalent five blue metrics

The max value is the historical per-episode maximum over the three aircraft, not a per-step red/blue mixture. Legacy aliases `lock_episode_rate`, `half_lock_episode_rate`, and `mean_max_lock_progress` point to red metrics.

Support cue rate now uses valid active cues divided by eligible steps. It tracks active cue steps, active cue-pair steps, eligible steps, unique cue pairs, and cue update calls separately. Cue update calls are diagnostic only.

### Termination

If both Combat sides reach zero before `max_steps`, the reason is `mutual_elimination_draw` and the mission reward is `-2.0`. `timeout_draw` is used only when `max_steps` is reached with equal Combat survivor counts. `timeout_rate` counts only reasons beginning with `timeout_`.

## Fixed baseline results

Both runs use exactly seeds `20000` through `20099`, four workers, and the unchanged v11 parameters.

### Rule red versus rule blue

Output: `outputs/v11_target_lock_support_cue_rule_100_mechanism_fixed.json`

| Metric | Result |
|---|---:|
| task win | 0.56 |
| full elimination | 0.35 |
| any kill | 0.73 |
| >=2 kills | 0.57 |
| mean red kills | 1.65 |
| mean blue kills | 1.83 |
| 0/1/2/3 kills | 0.27 / 0.16 / 0.22 / 0.35 |
| timeout win/loss/draw | 0.21 / 0.02 / 0.10 |
| timeout rate | 0.33 |
| mutual elimination | 0.00 |
| mean red/blue survivors | 1.78 / 1.35 |
| mean first kill time | 304.90 steps |
| red lock episode | 0.99 |
| red half-lock episode | 0.78 |
| mean red max lock | 0.7904 |
| blue lock episode | 1.00 |
| blue half-lock episode | 0.70 |
| mean blue max lock | 0.7211 |
| red/blue lock active step rate | 0.2553 / 0.2542 |
| target switch count | 15.55 |
| active cue rate | 0.9854 |
| cue pair step rate | 0.9457 |
| cue-to-direct rate | 0.9547 |
| assisted kill rate | 0.7818 |
| mean return | 13.6766 |
| dense saturation | 0.00 / 0.00 / 0.00 |

Reward component means were: mission `3.9000`, blue kill `9.9000`, red Combat loss `-4.8800`, Support loss `-0.6100`, boundary `0.0000`, geometry `0.0601`, lock progress `0.2232`, half-lock event `0.5775`, Support detection `0.3000`, cue-to-direct `1.8125`, cue-to-half-lock `1.1050`, assisted kill `1.2900`, formation `-0.0016`, clipped dense `0.2816`, team total `13.6766`.

### Random red versus rule blue

Output: `outputs/v11_target_lock_support_cue_random_100_mechanism_fixed.json`

| Metric | Result |
|---|---:|
| task win | 0.00 |
| full elimination | 0.00 |
| any kill | 0.00 |
| >=2 kills | 0.00 |
| mean red kills | 0.00 |
| mean blue kills | 3.19 |
| 0/1/2/3 kills | 1.00 / 0.00 / 0.00 / 0.00 |
| timeout win/loss/draw | 0.00 / 0.00 / 0.00 |
| timeout rate | 0.00 |
| mutual elimination | 0.00 |
| mean red/blue survivors | 0.00 / 3.00 |
| mean first kill time | unavailable |
| red lock episode | 1.00 |
| red half-lock episode | 0.01 |
| mean red max lock | 0.0362 |
| blue lock episode | 1.00 |
| blue half-lock episode | 1.00 |
| mean blue max lock | 1.0000 |
| red/blue lock active step rate | 0.0244 / 0.5007 |
| target switch count | 9.79 |
| active cue rate | 1.0000 |
| cue pair step rate | 0.9030 |
| cue-to-direct rate | 0.9273 |
| assisted kill rate | 0.00 |
| mean return | -26.1905 |
| dense saturation | 0.00 / 0.00 / 0.00 |

The separated lock metrics show why the old mixed `lock_episode_rate=1.00` was misleading: random red has red max lock `0.0362`, while blue reaches max lock `1.00`.

### Before versus after mechanism fix

| Rule metric | Previous final | Mechanism fixed |
|---|---:|---:|
| task win | 0.52 | 0.56 |
| full elimination | 0.25 | 0.35 |
| any kill | 0.67 | 0.73 |
| >=2 kills | 0.45 | 0.57 |
| mean red kills | 1.37 | 1.65 |
| support assisted kill rate | 0.6423 | 0.7818 |

The random red policy remains at zero task success and zero red kills. No environment parameter was changed to obtain this difference.

## Tests and smoke

- `python -m compileall -q src scripts tests`: passed.
- v11 focused tests: `45 passed`.
- v9/v10 4v3, hardening, HAPPO milestone, resume, and CUDA RNG tests: `64 passed`.
- Combined required mechanism/regression set: `109 passed in 16.69s`.
- A full pytest run was not counted: the previous `512 passed` result is not reused for this closeout, and a bounded full rerun exceeded the tool time limit.

CPU smoke output: `outputs/happo_v11_cpu_smoke_mechanism_fixed`

- exactly `8192 env steps`
- exactly `4 updates`
- `4 workers`, CPU
- all four Actor slots updated; `actor_updates=40`, `agents_updated=4`
- finite loss, KL, entropy, gradient, and reward values
- final throughput about `494.1 env steps/s`
- checkpoint, selection/test smoke artifacts, CSV, and resume state present
- CSV contains mutually exclusive reward groups and red/blue lock and active-cue fields
- independent CSV reconstruction passed: 4 training rows had maximum group error `1.78e-9`; 6 episode rows had maximum component error `3.55e-15`, with zero rows outside the `1e-6` tolerance

Example terminal line:

`[train] step=8192/8192 upd=4 fps=494.1 r_step=-0.07566 ep_return=-24.14 win=0.000 full=0.000 kills=0.000 timeout=0.188 reward{mission=-0.05127 combat_evt=-0.02930 support_evt=+0.00483 half_lock_evt=+0.00000 dense=+0.00007} raw{geom=+0.00008 lock=-0.00000 formation=-0.00000}`

The displayed groups reconstruct `r_step` within floating-point precision.

## v9/v10 SHA protection

Actual pre-change SHA256 values fixed by tests:

- `configs/heterogeneous_4v3_main_v9.yaml`: `A32F261B0A14201F221A0615EBBD23711C6A40AE0F10B3E9F1A690910026B4E5`
- `configs/happo_heterogeneous_4v3_main_v9.yaml`: `1E67A5421BDE43956B6E7A182C9CF2029716EA1093791289D40F9BA4254C124C`
- `configs/heterogeneous_4v3_main_v10_attack_funnel.yaml`: `708C313C6D4A70775E697CDD275D8CB24D85CD31E9DEA5D9EEB3E008CF0BAFF1`
- `configs/happo_heterogeneous_4v3_main_v10_attack_funnel.yaml`: `58D860EC06F194BC861E1DE55C2B98E4417F9D605189EB79360509C1342B3BB6`

The user-provided historical note labels `1E67...` as the v10 environment hash, but it matches the current v9 HAPPO file. The v10 environment's actual hash is `708C...`. No historical config was rewritten.

## Remaining issues and readiness

No known implementation issue from the requested closeout list remains. The fixed rule/random results satisfy the revised reachability gate: task win `0.56`, full elimination `0.35`, >=2 kills `0.57`, any kill `0.73`, mean red kills `1.65`; random remains at zero for all task success metrics and the red lock metrics are clearly separated from blue lock metrics.

This closeout did not run any formal RL long-step training. The v11 environment now meets the requested pre-training engineering gate and can be used by the user to execute the formal 3M training on the server. Formal training results are not claimed here.
