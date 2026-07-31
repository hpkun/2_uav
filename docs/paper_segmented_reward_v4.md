# Homogeneous 3v3 paper-segmented reward v4

`paper_segmented_team_v4` is an isolated homogeneous 3v3 reward mode for `configs/homogeneous_3v3_learnable_v7_paper_segmented.yaml`.

It adapts the MADSAC paper Eq. (25) segmented reward to the current project environment. This is not a full reproduction of the paper environment: the current 3D point-mass dynamics, attack envelope, 600-step episodes, observations, and fixed blue rule policy are kept unchanged.

Project mapping:

- The paper's 4000 m threshold is mapped to the current `attack_distance_max` of 1000 m.
- The current `attack_distance_min` is used as the lower near-range gate.
- Current 3D ATA is used for the segmented attack-angle tiers.
- Each alive aircraft uses its nearest currently alive enemy, with aircraft ID as deterministic tie-break.
- Dead own aircraft contribute zero.
- Team dense reward is divided by fixed `team_size = 3`, not alive count.

Dense terms:

- `R3`: `+0.001` when distance is at least `attack_distance_max` and ATA is within 30 degrees.
- `R41`: inside `[attack_distance_min, attack_distance_max]` and AA within 30 degrees, add `0.10 / 0.02 / 0.01` for ATA within 5 / 15 / 30 degrees, with strict non-stacking priority.
- `R42`: using reverse geometry for the same target, inside the same distance gate and reverse AA within 30 degrees, add `-0.150 / -0.025 / -0.015` for reverse ATA within 5 / 15 / 30 degrees.

Event and terminal terms:

- Attack kill: `+10` per enemy aircraft killed by attack this step.
- Own aircraft death: `-10` per own aircraft death this step, mapped to attack, boundary, or collision death components without duplicate penalties.
- Complete attack elimination success means cumulative attack kills equal 3 and at least one own aircraft survives.
- Non-success termination adds `-10 * own_survivors`.
- Mutual elimination adds an extra `-10`.
- Complete success has no additional terminal bonus; its value comes from the three attack kills minus any own losses.

The existing 14 red reward component slots are preserved. For this mode, death penalty components are signed negative values, and `red_team_total_reward` equals the sum of the previous 13 red components. v6 remains `target_consistent_team_v3`; v4/v5 remain `paper_coupled_team_v2`.
