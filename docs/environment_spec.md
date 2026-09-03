# Environment specification

The active contract is `heterogeneous_mavuav_3v2_v2_2`.

## Entities and task

Entity order is `MAV, UAV1, UAV2, Blue1, Blue2`. The MAV is armed, high-performance and high-value. Both UAVs are armed but have lower performance and task value. Blue is a fixed-rule homogeneous opponent. MAV death or boundary loss is a Red mission failure; UAV loss does not terminate the episode.

## Dynamics and timing

The state is `[x, y, h, v, theta, psi]`, with altitude positive upward. Controls are physical overloads:

```text
dx/dt     = v cos(theta) cos(psi)
dy/dt     = v cos(theta) sin(psi)
dh/dt     = v sin(theta)
dv/dt     = g (nx - sin(theta))
dtheta/dt = g/v (ny - cos(theta))
dpsi/dt   = g/(v cos(theta)) nz
```

For normalized action `u`, each overload dimension is mapped piecewise around local trim `(sin(theta), cos(theta), 0)`. Positive `u` interpolates from trim to the upper limit; negative `u` interpolates from trim to the lower limit. Thus zero action preserves speed, pitch and heading away from boundaries.

One decision is 1 s. Red and Blue actions remain fixed for ten RK4 substeps of 0.1 s. Pitch is clipped to +/-60 degrees, heading is wrapped to `[-pi, pi)`, and speed is clipped by aircraft type.

| Type | Speed (m/s) | nx | ny | nz |
|---|---:|---:|---:|---:|
| MAV | 250-400 | -1 to 5 | -1.5 to 2 | -3 to 3 |
| UAV | 150-300 | -1 to 5 | -1.5 to 1.5 | -2 to 2 |
| Blue | 250-400 | -1 to 5 | -1.5 to 3 | -3 to 3 |

## Scenario and boundaries

Nominal starts are MAV `(-4500,0,5000)`, UAV1 `(-4000,-800,5000)`, UAV2 `(-4000,800,5000)`, Blue1 `(4000,-600,5000)`, and Blue2 `(4000,600,5000)` metres. Red heads 0 degrees and Blue 180 degrees. Speeds are interval midpoints: 325, 225 and 325 m/s for MAV, UAV and Blue.

Reset randomization is seeded and profile-based. `learnability` uses zero team translation, +/-200 m independent slot x/y jitter, +/-100 m altitude, +/-10 m/s speed and +/-3 degrees heading. Default `main` independently translates the whole Red and Blue formations by +/-1500 m in x/y, then adds +/-300 m slot x/y, +/-400 m altitude, +/-20 m/s speed and +/-10 degrees heading. The common team offset preserves formation structure; `reset(options={"profile": ...})` selects a profile. Randomization can be disabled for nominal tests. Formal training, benchmark and evaluation entry points explicitly select `main` or `learnability`; outputs and checkpoints record `environment_profile`, and checkpoint loading rejects a mismatch.

Valid altitude is 1-20 km; x and y are each +/-100 km. A Red UAV leaving the volume becomes inactive. A Blue leaving becomes escaped and is not a Red kill. MAV leaving is mission failure.

## Sensor and Red datalink

MAV direct sensing range is 12,000 m and UAV direct sensing range is 8,000 m. For an alive Red `i` and alive Blue `j`, `direct_visible(i,j)` is exactly `distance(i,j) <= sensor_range(type_i)`. There is no FOV, noise, delay, probability or memory.

`team_visible(j)` is true when any alive Red directly sees Blue `j`. For agent `i`, `datalink_visible(i,j) = team_visible(j) and not direct_visible(i,j)`. Direct or datalink visibility exposes the Blue geometry and relative motion; otherwise its relative position, distance, relative velocity, ATA and AA are zero. The real alive bit remains present. Red teammate state is always available. Blue's fixed policy continues to use true Red state.

## Observation and centralized state

Each Red agent has a stable 61D observation. Self 11D is `[x,y,h,v,theta,psi,alive,type_MAV,type_UAV,type_Blue,time_fraction]`. Two teammate blocks follow stable `RED_IDS` order while skipping self; each 11D block is `[dx,dy,dh,distance,dvx,dvy,dvh,alive,type_MAV,type_UAV,type_Blue]`. Blue1 and Blue2 each contribute `[dx,dy,dh,distance,dvx,dvy,dvh,ATA,AA,alive,direct_visible,datalink_visible,own_attack_streak,killed_by_red]`. Attack streak is divided by `hold_steps`; time is `step_count/75`.

The actor layout is `self[0:11]`, `friend1[11:22]`, `friend2[22:33]`, `Blue1[33:47]`, and `Blue2[47:61]`. Enemy relative velocity is `Blue_velocity - own_Red_velocity`, normalized by the existing `relative_velocity_scale`. For an invisible or dead Blue, the nine geometry/motion values through ATA/AA are zero while status flags retain their existing semantics. This v2.2 change exposes enemy relative velocity to test whether explicit relative-motion information improves learnability and seed robustness; it does not claim to solve the nearest task before retraining.

The critic receives a 67D state. Five entity blocks are `[x,y,h,v,theta,psi,alive,type_MAV,type_UAV,type_Blue]` (50D). They are followed by the normalized streaks for six Red-to-Blue pairs and six Blue-to-Red pairs in entity slot order (12D), Blue1/Blue2 `killed_by_red` flags (2D), actual episode-mode one-hot `[nearest,mav_priority]` (2D), and time fraction (1D). This includes the nonphysical state needed to determine transitions and termination. The active mask remains `[MAV,UAV1,UAV2]`; dead UAV actions are ignored and their actor samples are masked.

Normalization is explicit: Actor self x/y divide by 30,000 and clip to `[-1,1]`. Centralized-state x/y use `2 * (value - lower) / (upper - lower) - 1` with the corresponding battlefield bounds, so `-100 km`, `0`, and `100 km` map to `-1`, `0`, and `1` without earlier saturation. Altitude maps battlefield `[1000,20000]` to `[-1,1]`; speed divides by 400; theta/psi divide by pi. Relative x/y divide by 12,000, relative altitude by 10,000, relative velocity by 800, all clipped to `[-1,1]`. Distance divides by 12,000 and clips to `[0,1]`; ATA/AA divide by pi. Thus 1 km and 3 km are about 0.0833 and 0.25.

## Attack and reward

Every alive cross-team attacker-target pair is checked once per decision boundary. A kill requires distance 1000-3000 m, attacker target angle below 30 degrees and entering angle below 90 degrees for three consecutive decisions. Streaks are pair-specific and reset when geometry breaks. All kills reached on one boundary are collected before any death is applied.

The situation reward uses the specified segmented bearing, entering-angle, distance, speed and height functions with weights `0.32, 0.43, 0.10, 0.10, 0.05`. Let `V` be the alive Blue aircraft for which `team_visible` is true. Each alive Red contributes its maximum situation score over `V`, or zero when `V` is empty. Team situation is the sum over the three fixed MAV/UAV1/UAV2 slots divided by three; dead Red slots contribute zero.

Events are +50 per Blue attack kill, -10 per UAV loss, and -100 per MAV loss. Terminal reward is +100 for Red win, -100 for Blue win/Red failure and zero for timeout draw. These event and terminal terms remain global mission outcomes, independent of current Red visibility. If any alive Red pair is closer than 100 m at a decision boundary, one team safety reward of -1 is added, at most once per step. It does not cause damage or death. Total team reward is situation + event + terminal + safety, shared by all Red agents.

## Blue policy and termination

Blue evaluates exactly 27 normalized overload actions from `{-1,0,1}^3`. Each candidate uses the same trim mapping, dynamics, 1 s lookahead and situation reward as Red. `nearest` targets the nearest living Red; `mav_priority` targets a living MAV first; `mixed_episode` samples one of those modes with equal probability at reset and holds it for the episode.

Red wins only if both Blue aircraft were killed by Red attacks and the MAV survives. Red fails if the MAV is inactive, or if all Blue are inactive without both being Red attack kills. Timeout after 75 decisions is a draw.
