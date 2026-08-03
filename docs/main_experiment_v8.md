# Main experiment v8

v8 is the simplified main-experiment environment for the current 3DOF NED point-mass 3v3 air-combat project.  This version intentionally removes the engineering-heavy v8 shaping used in the previous iteration and moves the main experiment closer to the simplified multi-UAV setting of Li et al. 2023 MADSAC.

## What v8 keeps

- 3DOF NED point-mass dynamics with RK4 integration.
- Continuous `rate_aligned_v1` action mapping.
- Deterministic attack envelope from the existing combat model.
- Fixed 600-step episode limit.
- MAPPO and HAPPO algorithm structures and stability settings.
- Homogeneous 3v3 and functional heterogeneous 3v3 variants.

## Collision contract

Li et al. 2023 does not define an aircraft-to-aircraft collision model for the simplified multi-UAV air-combat task.  The v8 main experiment therefore disables inter-aircraft collision detection by setting:

```yaml
battlefield:
  collision_distance: 0.0
```

`collision_distance <= 0` means aircraft are allowed to pass through each other in the point-mass model.  Aircraft in v8 can die only by deterministic attack or by leaving the combat area.  The collision detector remains available for historical configurations with positive `collision_distance`, but v8 does not include collision deaths, anti-collision rewards, separation rewards, soft safety rewards, or collision terminal rewards.

This is an intentionally simplified academic learning environment for algorithm comparison, not a high-fidelity flight-safety simulator.

## Observation contract

v8 follows the Equation (24) idea: each agent observes self state, teammate state, and enemy state blocks in its own coordinate frame.  It does not add target IDs, target flags, persistent engagement labels, or target-first ordering.

Enemy observation blocks use deterministic stateless ordering at every step:

1. alive enemies first, sorted by current 3D distance;
2. equal-distance ties broken by `aircraft_id`;
3. dead or invisible enemies are zero blocks;
4. no persistent target state is stored.

The observation dimension remains 68.

## Target and attack causality

v8 uses a stateless nearest-target rule for both reward targeting and attack intent:

- every attack-capable aircraft selects the current nearest alive enemy each step;
- ties are deterministic by `aircraft_id`;
- in the heterogeneous environment, combat aircraft select only currently effective-visible enemies;
- if the nearest selected target is outside the attack envelope, the aircraft does not attack another target in the same step;
- support aircraft never attack.

The homogeneous fixed blue opponent in v8 is `paper_nearest_pursuit_v1`, not the one-to-one greedy team matcher.

## Reward

v8 uses a dedicated Equation (25)-style paper reward helper.  The helper is separate from the older v7 `paper_segmented_local_reward` implementation so that v8 can keep the paper geometry and the project attack envelope distinct.

The reward geometry is horizontal:

- `horizontal_ata`: angle between own horizontal velocity and the horizontal line of sight to the target;
- `horizontal_aa`: angle between target horizontal velocity and the same horizontal line of sight;
- `height_angle`: `abs(atan2(target_altitude - own_altitude, horizontal_distance))`.

The paper reward distance threshold is `paper_segment_distance = 4000 m`.  This is not the same as the deterministic attack envelope, which remains `100-1000 m` with the existing ATA/AA kill checks.  Reward shaping can therefore activate outside the kill envelope, but actual kills are still decided only by the existing combat model.

For each team:

```text
R_team = sum(local paper rewards for the team) / fixed_team_size
```

with `fixed_team_size = 3`.

Local terms:

- `R1`: +10 for an attack kill by this team, -10 when one own aircraft is killed by attack.
- `R2`: -10 when one own aircraft leaves the combat boundary.
- `R3`: +0.001 only when distance is at least 4000 m, `horizontal_ata <= 30°`, and `height_angle <= 30°`.
- `R41`: within 4000 m and `horizontal_aa <= 30°`, then +0.01 / +0.02 / +0.10 for coarse / medium / fine own attack geometry.  The tier requires both `horizontal_ata` and `height_angle` to satisfy the same tier.
- `R42`: reverse geometry is recomputed by swapping own and target.  Within 4000 m and reverse `horizontal_aa <= 30°`, the penalty is -0.015 / -0.025 / -0.150 for coarse / medium / fine reverse threat geometry, again requiring both reverse `horizontal_ata` and reverse `height_angle`.

There is no extra terminal reward, timeout penalty, complete-elimination bonus, time penalty, soft-boundary shaping, continuous distance progress, Gaussian geometry reward, support-information shaping, dense clipping, collision penalty, separation penalty, anti-collision reward, soft safety reward, minimum-separation reward, or terminal collision reward in v8.  Timeout affects outcome/statistics and best-checkpoint selection only.

## Heterogeneous v8

The heterogeneous variant keeps only capability and observation differences:

- one support and two combat aircraft per team;
- support sensor range is 6000 m;
- combat sensor range is 3000 m;
- support cannot attack;
- combat can attack;
- support-to-combat information sharing remains enabled;
- HAPPO uses separate actors for support/combat slots;
- fixed support keeps the simple rear-formation hold rule.
- fixed combat aircraft use `functional_heterogeneous_nearest_pursuit_v8`: each combat aircraft independently selects its nearest alive effective-visible enemy each step and then reuses the same pure-pursuit `rate_aligned_v1` control.  There is no one-to-one target assignment in this v8 heterogeneous rule policy, so two combat aircraft may choose the same enemy.

Support aircraft can receive shared team reward, can be killed, and can incur boundary loss.  They do not produce R3/R41/R42 attack-geometry rewards and do not receive a support-information shaping reward.

Support coverage, support survival, and shared-observation kill metrics may be logged for analysis, but they do not enter the reward.

## Main metrics

The main training/evaluation path keeps:

- episode return;
- red/blue complete-elimination success rates;
- red/blue attack kills and any-attack-kill rates;
- red/blue survivors;
- attack, altitude-boundary, and XY-boundary deaths;
- timeout rate and mean episode length;
- R1 kill/death, R2 boundary, R3, R41, R42, and total team reward;
- actor loss, critic loss, entropy, approximate KL, effective log-std/std;
- compact action mean/saturation diagnostics over alive red slots only.

Heterogeneous evaluation additionally keeps support survival, mean support coverage, kills with shared observation, and shared-observation kill fraction as analysis metrics only.

## Best checkpoint

v8 best checkpoint selection uses the simplified lexicographic score:

1. `red_complete_elimination_success_rate`
2. `red_any_attack_kill_rate`
3. `mean_red_attack_kills`
4. `mean_red_survivors`
5. `-mean_red_boundary_deaths`
6. `-max_steps_rate`
7. `-mean_episode_length`

This preserves the project rule that red only wins after complete blue elimination; timeout/draw is not a red success.

## Rule-vs-rule attackability check

After changing v8 collision semantics, run a deterministic homogeneous v8 rule-vs-rule check with 200 episodes.  The check uses `paper_nearest_pursuit_v1` for both red and blue, `rate_aligned_v1`, the existing deterministic attack model, the `100-1000 m` attack range, the current ATA/AA attack conditions, `collision_distance = 0`, and `max_steps = 600`.

This check is only used to confirm whether the attack envelope is reachable under the fixed pure-pursuit rules.  It should not be used to tune attack distance, angles, rewards, or policy complexity from the same 200 episodes.

## Running long experiments

Short smoke tests are for executable-contract checks only.  Long 1M probe or larger runs should be launched by the user in Ubuntu/WSL with the desired v8 config.
