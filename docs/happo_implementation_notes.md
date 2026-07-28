# Homogeneous 3V3 HAPPO implementation notes

This baseline implements a project-native HAPPO-style trainer for the existing homogeneous 3v3 fixed-blue environment. The algorithm source is Kuba et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning", ICLR 2022 / arXiv:2109.11251v2. The official PKU-MARL/HARL implementation is used only as behavioral reference for sequential HAPPO updates; no external code is copied.

## Scope and frozen environment

HAPPO uses the current homogeneous 3v3 v4 environment, reward, fixed-blue nearest-target pursuit policy, action mapping, attack/death semantics, and vector environment interface. It does not create a heterogeneous environment, alter dynamics, add weapons, or change the reward function.

## Difference from the existing MAPPO baseline

The intended comparison is narrow:

- MAPPO has one shared red actor; HAPPO has three independent red actors.
- MAPPO updates the shared actor over all red slots together; HAPPO samples a random agent ordering and updates one actor at a time.
- HAPPO multiplies later agents' advantages by the product of preceding agents' new/old policy ratios (`factor`).
- Both use a centralized value critic, GAE, PPO clipping, dead-agent active masks, and CTDE.

This is not three independent PPO trainers. The core HAPPO step is the sequential factor update implemented in `HAPPO3v3Trainer.update`.

## Networks

`IndependentHAPPOActors` owns three separate `HAPPOGaussianActor` modules and three separate optimizers. The public constructor accepts lists such as:

```text
observation_dims = [68, 68, 68]
action_dims      = [3, 3, 3]
```

The current wrapper passes homogeneous dimensions, but the core interface does not hard-code that future agents must share dimensions. Each actor sees only its own local observation. Evaluation calls Actor 0 for red slot 0, Actor 1 for red slot 1, and Actor 2 for red slot 2.

`CentralizedValueCritic` implements `V_phi(s)` over the existing 48-dimensional 3v3 global state. It is not a Q critic and does not receive joint actions.

## Buffer and GAE

`HAPPORolloutBuffer3v3` stores local observations, centralized states, per-agent actions, per-agent old log probabilities, team rewards, value predictions, done flags, and alive masks. GAE follows the existing MAPPO 3v3 semantics: `terminated OR truncated` stops bootstrap under the current environment contract, and returns are team-level.

Advantages are normalized over valid active samples. Dead-agent samples are excluded from actor losses and entropy means, and their preceding-ratio contribution to HAPPO `factor` is forced to 1.

## Sequential HAPPO update

For each rollout update:

1. collect a full rollout using the old joint policy;
2. compute team GAE once;
3. sample a random permutation of agent ids with the trainer RNG;
4. initialize `factor = 1`;
5. for each agent in that order, optimize its PPO clipped surrogate with `effective_advantage = factor * advantage`;
6. after that actor finishes its PPO epochs, recompute its new log probability over the rollout;
7. compute `preceding_ratio = exp(new_log_prob - old_log_prob)`;
8. set dead/inactive samples' preceding ratio to 1;
9. update `factor = (factor * preceding_ratio).detach()`;
10. update the centralized value critic after the actor sequence.

The factor is not clipped. PPO clipping still applies inside each actor's surrogate objective.

## Checkpointing

HAPPO checkpoints use family `homogeneous_3v3_fixed_blue_happo` and save:

- all three actor state dicts;
- all three actor optimizer states;
- centralized critic and optimizer;
- env/vector/update counters;
- RNG states;
- best evaluation metadata;
- environment YAML SHA256;
- training signature.

Signature mismatch errors report specific differing fields.

## What this baseline does not claim

This code is a baseline implementation for the current project environment. It does not claim to reproduce the original HAPPO paper's experimental results, and it does not add HARL-specific engineering modules beyond the HAPPO sequential update behavior needed here.
