# Main experiment v8

v8 is an isolated main-experiment configuration for the current 3DOF NED
point-mass 3v3 air-combat environment. It does not replace the historical
v3/v4/v6/v7 or functional-heterogeneous v1 configs.

## Environment variants

- `configs/homogeneous_3v3_main_v8.yaml` keeps the validated homogeneous 3v3
  dynamics, controller, attack envelope and 600-step limit, uses
  `rate_aligned_v1`, and uses fixed blue `greedy_team_pursuit_v1`.
- `configs/heterogeneous_3v3_main_v8.yaml` keeps the functional heterogeneous
  structure: one support and two combat aircraft per side, support sensor
  range 6000 m, combat sensor range 3000 m, support-to-combat sharing, and
  support aircraft cannot attack.

## Reward and target consistency

Homogeneous v8 uses `task_aligned_continuous_team_v8`; heterogeneous v8 uses
`task_aligned_heterogeneous_team_v8`.

The team reward is:

```text
R_team = R_progress + R_attack_geometry - R_threat
         - R_boundary - R_time + R_event + R_terminal
```

Dense terms are normalized by the fixed team size of 3, not by alive count.
Progress is continuous inside the attack range and is zero on the first step
after a target switch. Attack geometry and threat use opposite directions of
the same coupled Gaussian geometry score. Altitude and horizontal soft-boundary
risk are recorded separately.

Only v8 modes use persistent engagement targets. At reset, each aircraft chooses
the nearest alive enemy. It keeps that target while the target is alive, and
reselects the nearest alive enemy only after target death. The same engagement
target is used for progress, attack geometry, threat, reward target reporting
and v8 attack selection.

## Training configs

- `configs/mappo_3v3_main_v8.yaml`
- `configs/happo_3v3_main_v8.yaml`
- `configs/happo_heterogeneous_3v3_main_v8.yaml`

These configs align MAPPO and HAPPO rollout/GAE/PPO settings and use the v8
exploration bounds:

```yaml
log_std_init: -1.0
log_std_min: -3.0
log_std_max: -0.3
```

The default `total_env_steps: 1000000` is for later formal runs. Short smoke
checks should continue to use the scripts' `--smoke` mode or an explicit small
step override.

## Main metrics

Episode and evaluation summaries include attack kills, survivors, complete
elimination success, any attack kill, deterministic attack-window agent steps,
alive agent steps, attack-window fraction, any attack-window episode, altitude
and XY boundary deaths, collision deaths, timeouts, target switches and episode
length. Heterogeneous summaries keep support survival, support coverage, and
kills with shared observation.

Best-checkpoint ranking for v8 is lexicographic:

1. red complete-elimination success rate
2. red any-attack-kill rate
3. mean red attack kills
4. red any-attack-window rate
5. mean red attack-window fraction
6. lower mean blue survivors
7. mean red survivors
8. lower mean red boundary deaths
9. lower max-step rate
10. shorter mean episode length

This is an engineering main-experiment target alignment, not a JSBSim, missile,
radar, curriculum, expert-blue or new-algorithm change.
