# Environment specification

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

Training randomization independently adds +/-200 m in x/y, +/-100 m altitude, +/-10 m/s speed and +/-3 degrees heading, using the reset seed. It can be disabled for nominal tests. Valid altitude is 1-20 km; x and y are each +/-100 km. A Red UAV leaving the volume becomes inactive. A Blue leaving becomes escaped and is not a Red kill. MAV leaving is mission failure.

## Observation and centralized state

Each Red agent has a stable 40D observation: 8D self, two 9D friend blocks and two 7D Blue blocks. Blocks retain dead entities and use an alive bit. Position, altitude, speed, angle, distance and relative velocity have centralized normalization constants in `mavuav.py`; type is MAV=0, UAV=1 and Blue=2 where applicable.

The critic receives a 40D state made of five 8D blocks `[x,y,h,v,theta,psi,alive,type]`. `global_state()` is the only trainer source for this state. The active mask is `[MAV,UAV1,UAV2]`; dead UAV actions are ignored and their samples are excluded from actor losses.

## Attack and reward

Every alive cross-team attacker-target pair is checked once per decision boundary. A kill requires distance 1000-3000 m, attacker target angle below 30 degrees and entering angle below 90 degrees for three consecutive decisions. Streaks are pair-specific and reset when geometry breaks. All kills reached on one boundary are collected before any death is applied.

The situation reward uses the specified segmented bearing, entering-angle, distance, speed and height functions with weights `0.32, 0.43, 0.10, 0.10, 0.05`. Each alive Red takes its best score over alive Blue targets. Team situation is the sum over MAV/UAV1/UAV2 divided by the fixed denominator three.

Events are +50 per Blue attack kill, -10 per UAV loss, and -100 per MAV loss. Terminal reward is +100 for Red win, -100 for Blue win/Red failure and zero for timeout draw. All Red agents receive the same team reward.

## Blue policy and termination

Blue evaluates exactly 27 normalized overload actions from `{-1,0,1}^3`. Each candidate uses the same trim mapping, dynamics, 1 s lookahead and situation reward as Red. `nearest` targets the nearest living Red; `mav_priority` targets a living MAV first; `mixed_episode` samples one of those modes with equal probability at reset and holds it for the episode.

Red wins only if both Blue aircraft were killed by Red attacks and the MAV survives. Red fails if the MAV is inactive, or if all Blue are inactive without both being Red attack kills. Timeout after 75 decisions is a draw.

