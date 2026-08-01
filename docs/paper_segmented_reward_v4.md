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
- Dense terms are evaluated from the post-motion, post-boundary, post-collision, pre-attack state.
- Event and terminal terms are evaluated after attack resolution from the real death ledger.

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

Timing semantics:

- `R41` and an attack kill may occur in the same step. `R41` rewards the pre-attack advantageous geometry; the kill reward records the completed attack event.
- `R42` and an own attack death may occur in the same step. `R42` penalizes the pre-attack threat geometry; the death penalty records the actual aircraft loss.
- A target killed during the current step can remain the current step's `reward_targets` entry because the target was alive in the pre-attack snapshot.
- The next step recomputes reward targets from the then-current alive aircraft, so dead targets are not carried across steps.
- This timing is the current project's adaptation for step-order details not specified by the MADSAC paper; it is not a claim about the paper's original source-code execution order.

Reward accounting:

- `approach_reward`, `attack_advantage_reward`, and `threat_penalty` are diagnostic subcomponents of `dense_reward`.
- `dense_reward = R3 + R41 + R42`.
- `team_total_reward = dense_reward + kill_reward + attack_death_penalty + boundary_death_penalty + collision_death_penalty + terminal_reward`.
- The diagnostic dense subcomponents are not added again when computing `team_total_reward`.
- The 14-component vector intentionally contains both decomposition fields and aggregate fields, so callers must not unconditionally sum the first 13 columns.

The existing 14 red reward component slots are preserved. For this mode, death penalty components are signed negative values. v6 remains `target_consistent_team_v3`; v4/v5 remain `paper_coupled_team_v2`.
