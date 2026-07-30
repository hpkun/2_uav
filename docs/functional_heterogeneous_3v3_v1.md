# Functional Heterogeneous 3V3 V1

This environment is a small, isolated extension of `homogeneous_3v3_learnable_v6_task_aligned.yaml`. It keeps the same dynamics, controller, action mapping, attack envelope, battlefield, initial scenario, observation dimension, global-state dimension, and termination semantics.

## Roles

- `red_0` and `blue_0`: support.
- `red_1`, `red_2`, `blue_1`, `blue_2`: combat.
- Support and combat aircraft share the same `AircraftSpec`; heterogeneity is only role, sensor range, weapon permission, information sharing, reward contribution, and fixed rule behavior.
- In heterogeneous mode, logical `red_i` and `blue_i` are same-role physical mirror pairs with the same sampled speed, altitude, heading jitter, sensor range, weapon permission, and spec. This opt-in branch does not change v4/v5/v6 historical ID-to-slot behavior.
- V1 supports only the fixed mapping `red_0/blue_0` support and all other 3v3 slots combat.

## Visibility

Direct visibility is deterministic distance sensing:

```text
direct_visible(own, enemy) =
    own alive and enemy alive
    and own.team != enemy.team
    and distance_3d(own, enemy) <= own.sensor_range
```

Support range is 6000 m. Combat range is 3000 m. There is no FOV, noise, occlusion, probability, scan cycle, confidence, or memory.

## Information Sharing

For a combat aircraft, effective visibility is:

```text
effective_visible(combat, enemy) =
    direct_visible(combat, enemy)
    or (team support alive and direct_visible(team support, enemy))
```

For a support aircraft, effective visibility is direct visibility only. Combat aircraft do not share detections with each other.

`support_to_combat` is the only information-sharing switch retained in the v1 config. Immediate sharing, no combat-to-combat sharing, fixed enemy slots, hidden continuous fields equal to zero, and enemy status `+1/0/-1` are fixed environment semantics rather than YAML options.

## Attack Gate

An attack intent requires all of:

- attacker is alive;
- `attacker.can_attack` is true;
- target is in the attacker's effective visible enemy set;
- existing distance + ATA + AA attack model returns true.

Support aircraft never generate attack intents. Hidden targets cannot be attacked even when physically inside the geometric attack envelope.

`red_kills_with_shared_observation` and `blue_kills_with_shared_observation` count only same-step kills where the attacker could not directly detect the target but could attack because live support shared that target. With default `combat sensor range = 3000 m` and `attack distance max = 1000 m`, this diagnostic is structurally expected to be zero. The main support contribution metrics are support coverage, support survival, and the `support_to_combat` on/off ablation.

## Observation

`OBS_DIM` remains 68:

```text
self block: 8
friend blocks: 2 x 12
enemy blocks: 3 x 12
```

In heterogeneous mode, entity slots are fixed by ID. Enemy slot status uses the existing last scalar:

```text
+1.0 = alive and effective visible
 0.0 = alive but hidden
-1.0 = dead
```

Hidden and dead enemy continuous fields are zero. Teammate blocks keep the old alive/zero semantics.

## Reward

The team dense reward is fully shared. Combat reward targets are selected only from effective-visible alive enemies, by nearest 3-D distance with ID tie-break:

```text
combat_dense_raw =
    (approach_weight * sum(combat_approach)
     + attack_advantage_weight * sum(combat_attack)
     - threat_weight * sum(combat_threat)
     - combat_boundary_weight * sum(combat_boundary)) / combat_count
```

Support contributes information coverage and boundary risk:

```text
support_dense_raw =
    support_information_weight * support_coverage
    - support_boundary_weight * support_boundary_risk
```

The final dense term is:

```text
team_dense = clip(combat_dense_raw + support_dense_raw - time_penalty)
```

Event and terminal terms follow v6 semantics: attack kill, attack death, boundary death, collision death, complete attack elimination, team elimination, mutual elimination, and max-step red failure / blue success.

## Support Coverage

For team support `S` and each live enemy `j`:

```text
useful_shared_j =
    direct_visible(S, j)
    and any(live team combat cannot directly see j)
```

```text
support_coverage =
    useful_shared_enemy_count / alive_enemy_count
```

Coverage is zero if support is dead, no combat is alive, or no enemy is alive.

## Rule Policy

`functional_heterogeneous_team_v1` is deterministic:

- combat aircraft build legal `(own combat, visible enemy)` pairs and use visible-pair distance-greedy assignment with ID tie-breaks;
- support aircraft ignore enemies and follow a point `follow_distance` behind the live combat centroid along the mean combat heading;
- no state machine, prediction, evasion, formation roles, communication action, ammunition, or missile model is added.
- `rear_formation_hold_v1` is currently the only supported support rule mode; unknown modes fail during configuration or policy construction.

## Outcome Accounting

The v6 timeout convention is retained: max-step timeout is encoded as a blue environment outcome. This does not mean blue achieved complete attack elimination; complete elimination still requires three attack kills and at least one surviving attacker.

## TAM-HAPPO Alignment Boundary

This project version uses TAM-HAPPO only as motivation for functional roles: non-weapon support, weapon-bearing combat, independent actors, and a centralized critic. It does not reproduce JSBSim vehicles, missiles, GRU, attention modules, sensor uncertainty, or communication constraints.
