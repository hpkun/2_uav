# Main experiment v8

v8 is the simplified main-experiment environment for the current 3DOF NED point-mass 3v3 air-combat project.  This version intentionally removes the engineering-heavy v8 shaping used in the previous iteration and moves the main experiment closer to the simplified multi-UAV setting of Li et al. 2023 MADSAC.

## What v8 keeps

- 3DOF NED point-mass dynamics with RK4 integration.
- Continuous `rate_aligned_v1` action mapping.
- Deterministic attack envelope from the existing combat model.
- Fixed 600-step episode limit.
- MAPPO and HAPPO algorithm structures and stability settings.
- Homogeneous 3v3 and functional heterogeneous 3v3 variants.

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

v8 uses the existing tested paper-segmented local helper for Equation (25)-style `R1 + R2 + R3 + R4` terms.

For each team:

```text
R_team = sum(local paper rewards for the team) / fixed_team_size
```

with `fixed_team_size = 3`.

Local terms:

- `R1`: +10 for an attack kill by this team, -10 when one own aircraft is killed by attack.
- `R2`: -10 when one own aircraft leaves the combat boundary.
- `R3`: +0.001 guide reward from the existing paper-segmented helper.
- `R41`: +0.01 / +0.02 / +0.10 for coarse / medium / fine own attack geometry.
- `R42`: -0.015 / -0.025 / -0.150 for coarse / medium / fine reverse threat geometry.

There is no extra terminal reward, timeout penalty, complete-elimination bonus, time penalty, soft-boundary shaping, continuous distance progress, Gaussian geometry reward, or support-information shaping in v8.  Timeout affects outcome/statistics and best-checkpoint selection only.

## Heterogeneous v8

The heterogeneous variant keeps only capability and observation differences:

- one support and two combat aircraft per team;
- support sensor range is 6000 m;
- combat sensor range is 3000 m;
- support cannot attack;
- combat can attack;
- support-to-combat information sharing remains enabled;
- HAPPO uses separate actors for support/combat slots;
- fixed blue support keeps the simple rear-formation hold rule.

Support aircraft can receive shared team reward, can be killed, and can incur boundary loss.  They do not produce R3/R41/R42 attack-geometry rewards and do not receive a support-information shaping reward.

Support coverage, support survival, and shared-observation kill metrics may be logged for analysis, but they do not enter the reward.

## Main metrics

The main training/evaluation path keeps:

- episode return;
- red/blue complete-elimination success rates;
- red/blue attack kills and any-attack-kill rates;
- red/blue survivors;
- attack, altitude-boundary, XY-boundary, and collision deaths;
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
6. `-mean_red_collision_deaths`
7. `-max_steps_rate`
8. `-mean_episode_length`

This preserves the project rule that red only wins after complete blue elimination; timeout/draw is not a red success.

## Running long experiments

Short smoke tests are for executable-contract checks only.  Long 1M probe or larger runs should be launched by the user in Ubuntu/WSL with the desired v8 config.
