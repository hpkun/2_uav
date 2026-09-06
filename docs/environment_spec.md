# Canonical environment specification

The active contract is `heterogeneous_mavuav_4v4_v3_0`: one armed MAV and three armed UAVs versus four homogeneous fixed-rule Blue aircraft. Entity order is `MAV, UAV1, UAV2, UAV3, Blue1, Blue2, Blue3, Blue4`. MAV loss is immediate Red mission failure; UAV loss alone does not terminate an episode.

Dynamics and combat semantics are unchanged from the validated predecessor: overload-controlled 3DOF motion, trim-centred normalized actions, 1 s decisions, 0.1 s RK4 physics, 75 decisions, 1--3 km engagement distance, strict ATA below 30 degrees, strict AA below 90 degrees, three consecutive decision steps, pairwise streaks and synchronous kill resolution. Each Blue still uses the 27-candidate one-step policy with `nearest`, `mav_priority`, or seeded `mixed_episode` targeting.

Nominal starts in metres are MAV `(-4500,0,5000)`, UAV1 `(-4000,-1200,5000)`, UAV2 `(-4000,0,5000)`, UAV3 `(-4000,1200,5000)`, Blue1 `(4000,-1800,5000)`, Blue2 `(4000,-600,5000)`, Blue3 `(4000,600,5000)`, and Blue4 `(4000,1800,5000)`. Red heads 0 degrees and Blue heads 180 degrees. Speeds remain 325, 225 and 325 m/s for MAV, UAV and Blue.

## Actor observation

Each Red actor receives 100 values. Layout is self `[0:11]`; three friend blocks `[11:22]`, `[22:33]`, `[33:44]` in stable `RED_IDS` order while skipping self; and Blue blocks `[44:58]`, `[58:72]`, `[72:86]`, `[86:100]`. Self is `[x,y,h,v,theta,psi,alive,type_MAV,type_UAV,type_Blue,time]`. Each friend is `[dx,dy,dh,distance,dvx,dvy,dvh,alive,type_MAV,type_UAV,type_Blue]`. Each enemy is `[dx,dy,dh,distance,dvx,dvy,dvh,ATA,AA,alive,direct_visible,datalink_visible,own_attack_streak,killed_by_red]`.

## Centralized state

The critic receives 119 values: eight 10D entity blocks `[0:80]`; 32 normalized directed cross-team attack streaks `[80:112]` ordered as all Red-to-Blue pairs followed by all Blue-to-Red pairs; four Red-kill-history flags `[112:116]`; actual Blue episode-mode one-hot `[116:118]`; and time fraction `[118]`. The active mask order is the four Red slots.

## Reward and termination

Each alive Red contributes its maximum five-part situation score over currently team-visible alive Blue aircraft, or zero if none is visible. The sum is always divided by the four fixed Red slots. Event, terminal, safety, sensor/datalink and normalization semantics and coefficients are unchanged. A Red win requires all four Blue aircraft to have been destroyed by Red attacks while MAV remains alive.

HAPPO uses four independent actors; MAPPO uses one shared actor over four Red samples. R-HAPPO uses four independent GRUs. HRTA and Structured Uniform parse three friend blocks and four enemy blocks. All v3.0 checkpoints carry the 100D/119D contract and intentionally reject old 3v2 checkpoints.
