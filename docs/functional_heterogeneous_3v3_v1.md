# Functional Heterogeneous 3V3 V1

This environment is a small, isolated extension of `homogeneous_3v3_learnable_v6_task_aligned.yaml`. It keeps the same dynamics, controller, action mapping, attack envelope, battlefield, initial scenario, observation dimension, global-state dimension, and termination semantics.

## Roles

- `red_0` and `blue_0`: support.
- `red_1`, `red_2`, `blue_1`, `blue_2`: combat.
- Support and combat aircraft share the same `AircraftSpec`; heterogeneity is only role, sensor range, weapon permission, information sharing, reward contribution, and fixed rule behavior.

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

## Attack Gate

An attack intent requires all of:

- attacker is alive;
- `attacker.can_attack` is true;
- target is in the attacker's effective visible enemy set;
- existing distance + ATA + AA attack model returns true.

Support aircraft never generate attack intents. Hidden targets cannot be attacked even when physically inside the geometric attack envelope.

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

- combat aircraft build legal `(own combat, visible enemy)` pairs and greedily assign by distance and ID tie-break;
- support aircraft ignore enemies and follow a point `follow_distance` behind the live combat centroid along the mean combat heading;
- no state machine, prediction, evasion, formation roles, communication action, ammunition, or missile model is added.

## TAM-HAPPO Alignment Boundary

This project version uses TAM-HAPPO only as motivation for functional roles: non-weapon support, weapon-bearing combat, independent actors, and a centralized critic. It does not reproduce JSBSim vehicles, missiles, GRU, attention modules, sensor uncertainty, or communication constraints.
