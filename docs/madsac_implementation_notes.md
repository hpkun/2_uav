# Homogeneous 3V3 MADSAC implementation notes

This baseline is an independent, project-local MADSAC-style implementation for the frozen homogeneous 3V3 v4 environment. It does not claim to reproduce unpublished author code.

## Frozen project interface

The implementation treats the following as fixed:

- `configs/homogeneous_3v3_learnable_v4.yaml`
- 3 homogeneous red UAVs with a shared red policy
- 3 fixed nearest-target-pursuit blue UAVs
- local red observations `[3, 68]`
- red joint action `[3, 3]` in `[-1, 1]`
- `rate_aligned_v1` action mapping
- current `paper_coupled_team_v2` reward
- current attack envelope, death ledger, outcome semantics, and 600-step cap

No environment dynamics, controller equation, reward term, attack judgment, or fixed-rule blue behavior is changed by the MADSAC files.

## Runtime sampling defaults

The formal MADSAC training configuration defaults to 16 parallel environments and 4 worker processes, so each worker manages 4 environments by default. `total_env_steps` remains the cumulative count of environment transitions across all parallel environments: increasing `num_envs` changes sampling throughput and the number of vector steps needed to reach the same total, but it does not change the per-sample algorithm definition. CLI arguments such as `--num-envs` and `--env-workers` can still override the YAML defaults.

This is an engineering throughput setting, not a change to the paper-aligned SAC/MADSAC algorithm structure. Because checkpoint compatibility signatures include `num_envs`, old 8-environment checkpoints should be resumed with an explicit `--num-envs 8`; new experiments that omit `--num-envs` use 16 by default.

## Structures aligned with the paper description

The baseline includes these MADSAC/SAC structures:

- shared actor for homogeneous agents;
- double centralized critics;
- attention aggregation across red agents;
- uniform experience replay;
- fixed maximum-entropy coefficient;
- target actor and target critics;
- minimum of double Q values in the TD target;
- delayed actor update;
- Polyak soft update.

## Project reproduction assumptions

The following are implementation choices made for this project because the exact original code is not available:

- state-dependent `log_std_head` for the squashed Gaussian actor;
- tanh-squashed reparameterized SAC action sampling;
- `policy_delay = 2`;
- dead-agent masking semantics in critic, target, and actor losses;
- fixed `alpha = 0.1` in the first version;
- continued use of the current v2 project reward instead of adding the paper's original segmented reward.

## Actor

`SharedSquashedGaussianActor` accepts either `[batch, 3, 68]` or `[batch, 68]`.

```text
Linear(68, 256)
ReLU
Linear(256, 256)
ReLU
mean_head: Linear(256, 3)
log_std_head: Linear(256, 3)
```

Actions are sampled by reparameterization:

```text
raw = mean + std * epsilon
action = tanh(raw)
```

Log probability includes the tanh Jacobian correction and is summed over each agent's three action dimensions, producing `[batch, 3]` for 3-agent input.

## Attention critics

Each `AttentionCritic` receives:

```text
observations: [batch, 3, 68]
actions:      [batch, 3, 3]
alive_masks:  [batch, 3]
```

Each red agent encodes `concat(observation_i, action_i)` with a shared encoder. A 2-head `MultiheadAttention` layer aggregates over alive red agents. Dead agents are masked as keys/values and their final Q values are forced to zero. All-dead rows keep one zero dummy token available internally to avoid attention NaNs, then the final Q row is explicitly zeroed.

`TwinAttentionCritic` contains two fully independent `AttentionCritic` instances, Q1 and Q2.

## Replay and TD target

Replay stores:

```text
observations      [3, 68]
actions           [3, 3]
team_reward       scalar
next_observations [3, 68]
alive_masks       [3]
next_alive_masks  [3]
terminated        bool
truncated         bool
```

`done_for_bootstrap = terminated OR truncated`; max-steps transitions do not bootstrap because the current environment has explicit terminal/max-step reward semantics.

Per-agent TD target:

```text
y_i = team_reward
      + gamma * (1 - done)
        * next_alive_mask_i
        * (min(target_q1_i, target_q2_i) - alpha * next_log_prob_i)
```

Current dead agents are excluded from critic and actor losses. Next dead agents do not bootstrap.

## Checkpoint policy

Checkpoints save network and optimizer states, counters, RNG states, best-evaluation metadata, and replay metadata. The full replay array is intentionally not persisted in the first version to avoid very large checkpoint files. Loaded checkpoints report `replay_restored = false` rather than pretending to be lossless replay resumes.

Training metrics are aggregated over each logging interval instead of taking only the last `trainer.update()` result. Critic metrics are weighted by real critic updates, while actor metrics are computed only from true actor optimizer steps. Critic-only updates therefore leave actor interval fields as `None` when no actor update occurred instead of contributing artificial zeros. Reported gradient-norm metrics are the `pre_clip` values returned by `clip_grad_norm_`; clipping fractions are computed from whether those pre-clip norms exceed the configured maximum.

Resume runs preserve the historical `best_score`, `best_evaluation`, `best_checkpoint_name`, and `evaluation_history` loaded from the checkpoint. Resume is not lossless because replay contents are not restored; only replay metadata is persisted. Resume milestones for logging, evaluation, and checkpoints are scheduled at the next strict interval after the restored environment step.

Checkpoint compatibility signatures include the environment YAML content SHA256, network architecture, core SAC hyperparameters, learning rates, replay capacity, learning starts, gradient steps, gradient clipping limits, and number of vector environments. Mismatches report the specific incompatible fields. Final checkpoint reload validation uses a lightweight actor-only load path that checks the saved `online_actor` deterministic action for a fixed probe observation, rather than constructing a second full trainer, replay buffer, and vector environment.

## Correctness criteria

Tests and smoke runs check interface and numerical validity: finite actions, log probabilities, Q values, TD targets, losses, gradients, target updates, replay sampling, checkpoint reload, worker execution, and evaluation generation. Kills, collisions, boundary deaths, draw rate, or reward improvement are not used as implementation-correctness gates.
