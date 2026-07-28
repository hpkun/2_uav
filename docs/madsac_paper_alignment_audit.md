# MADSAC paper-alignment audit

This audit compares the project-local homogeneous 3v3 MADSAC baseline with Li et al., "Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic", Aerospace 2023, 10(7), 574, DOI `10.3390/aerospace10070574`.

The audit separates paper-required behavior from current-project adaptations and standard SAC reproduction assumptions. It does not use kill rate, reward trend, collision rate, boundary rate, or 5M training performance as correctness evidence.

## Audit table

| Paper component | Paper formula / figure / algorithm location | Paper requirement | Current code path | Current implementation | Conclusion | Severity | Needs change | Change basis |
|---|---|---|---|---|---|---|---|---|
| Homogeneous red policy | MADSAC cooperative multi-UAV setting and actor description | Homogeneous agents use a shared decision model during decentralized execution. | `src/uav_combat/madsac/networks.py`, `trainer_3v3.py` | `SharedSquashedGaussianActor` is one module applied to all three red observations. | Equivalent implementation | Low | No | Shared homogeneous actor matches the cooperative homogeneous setting. |
| Actor local execution | Decentralized actor / centralized critic description | Actor execution should depend on the agent's local observation, not global state. | `SharedSquashedGaussianActor.sample`, `MADSAC3v3Trainer._select_actions` | Actor input is `[num_envs, 3, 68]`; each slot passes its own 68-dim local observation through the same MLP. | Equivalent implementation | Low | No | Actor never receives `GS_DIM` global state. |
| Continuous stochastic actor | Soft actor-critic policy equations | Policy is stochastic for continuous actions. | `SharedSquashedGaussianActor.sample` | Tanh-squashed Gaussian with reparameterized sampling. | Paper unspecified, standard SAC reproduction assumption | Low | No | Paper does not specify exact squash/log-std engineering; standard SAC requires reparameterized stochastic policy. |
| Tanh log-prob Jacobian | Standard squashed SAC implementation detail | If tanh squashing is used, log probability must include the Jacobian correction. | `SharedSquashedGaussianActor.sample` | Includes `-log(1-action^2+epsilon)` summed over action dims. | Paper unspecified, standard SAC reproduction assumption | Medium | No | Required by the selected squashed Gaussian SAC assumption. |
| Actor target Q | Double-SAC actor loss | Actor update should be guided by the smaller of twin Q estimates. | `MADSAC3v3Trainer.update` | Actor loss uses `torch.minimum(q1_pi, q2_pi)`. | Equivalent implementation | Low | No | Double-SAC clipped Q target. |
| Actor joint resampling | Multi-agent actor update | Current joint policy should provide the action used for actor loss. | `MADSAC3v3Trainer.update` | All three red actions are resampled from the current actor for `actions_pi`; replay actions are not used in actor loss. | Equivalent implementation | Low | No | Avoids stale replay actions in actor objective. |
| Dead current agents | Current project terminal/death adaptation | Dead red agents should not contribute loss. | `masked_mean`, `update` | Actor and critic losses are masked by current alive mask. | Current environment necessary adaptation | Medium | No | Paper does not define this project's mid-episode death ledger; masking is required for finite valid training. |
| Attention critic self/other structure | Attention critic network formula / critic architecture figure | For each agent i: encode own `(o_i,a_i)`, query from self, keys/values from other agents, concatenate self embedding and aggregated context, output `Q_i`. | `src/uav_combat/madsac/networks.py` | Previous implementation used generic self-attention including the self token. Current implementation uses explicit query/key/value projections and excludes self from key/value attention set. | Previously deviated from paper; fixed | High | Yes, done | Paper-style attention critic requires own path plus other-agent aggregation, not ordinary Transformer self-attention. |
| Twin critics | Double-SAC / MADSAC algorithm | Q1 and Q2 must be independent. | `TwinAttentionCritic` | Two separate `AttentionCritic` instances. | Fully consistent | High | No | Independent modules and parameters. |
| TD target next action | SAC target equation | Next action is sampled from target actor. | `compute_td_target` | Uses `self.target_actor.sample(next_observations)`. | Fully consistent | High | No | Target actor is used. |
| TD target min Q | Double-SAC target | Target uses `min(target_q1,target_q2)`. | `compute_td_target` | Uses `torch.minimum(tq1,tq2)`. | Fully consistent | High | No | Clipped double-Q target. |
| Entropy term sign | SAC target and actor objective | Target subtracts alpha log-prob; actor optimizes alpha log-prob minus Q. | `compute_td_target`, `update` | `target_q - alpha * next_log_probs`; actor loss `alpha * log_probs_pi - q_pi`. | Equivalent implementation | High | No | Standard SAC sign convention. |
| Terminal bootstrap | Current environment terminal semantics | `terminated OR truncated` should stop bootstrap under current project max-step terminal reward semantics. | `replay_buffer.py`, `compute_td_target` | `done_for_bootstrap = terminated OR truncated`; target multiplied by `1-done`. | Current environment necessary adaptation | Medium | No | Project max-step transition has terminal semantics. |
| Target networks | MADSAC algorithm | Target actor and critics initialized from online networks and updated by Polyak averaging. | `trainer_3v3.py` | Target networks load online states, `requires_grad=False`, `soft_update_(target, source, tau)`. | Fully consistent | High | No | Polyak direction is target = `(1-tau) target + tau online`. |
| Delayed policy update | MADSAC algorithm | Actor and target updates are delayed relative to critic updates. | `policy_delay`, `update` | `policy_delay=2` default; actor/target update when critic count is divisible by delay. | Paper unspecified exact value; reproduction assumption | Low | No | Delay exists; exact value 2 is treated as project assumption unless directly specified. |
| Replay buffer | Off-policy SAC | Store executed transition tuples and sample uniformly. | `replay_buffer.py` | Stores local observations, executed bounded actions, team reward, next observations before reset, masks, terminal flags; samples uniformly. | Equivalent implementation with project masks | Medium | No | Meets off-policy SAC data requirements. |
| Team reward broadcasting | Cooperative current environment | Scalar team reward is used for all alive red Q targets. | `compute_td_target` | `team_rewards` is broadcast to `[batch,3]`. | Current environment necessary adaptation | Medium | No | Current reward contract is cooperative `paper_coupled_team_v2`; reward design is frozen. |
| 16 env update ratio | Engineering throughput setting | Parallel env count is not a paper algorithm parameter; update-to-data ratio should remain comparable to previous project default. | `configs/madsac_3v3_paper.yaml`, `scripts/train_madsac_3v3.py` | Formal default now uses `num_envs=16`, `num_env_workers=4`, `gradient_steps=2`; CLI can restore `--num-envs 8 --gradient-steps 1`. | Confirmed engineering issue; fixed | Medium | Yes, done | Preserves approximately one critic update per eight collected transitions from the prior project baseline. |
| Checkpoint signature | Reproducibility engineering | Resume should reject incompatible training geometry. | `training_signature`, `load_checkpoint` | Signature includes `num_envs`, `gradient_steps`, env YAML SHA256, learning rates, replay size, core SAC hyperparameters. | Project reproducibility adaptation | Medium | No | Strict mismatch reporting prevents accidental mixed-resume. |

## MADSAC fixes made from this audit

1. `AttentionCritic` now implements an explicit self path and other-agent attention set. Self tokens are excluded from the key/value set; dead teammate tokens are masked; no-other-agent context is zero; all-dead output is finite zero.
2. `configs/madsac_3v3_paper.yaml` now uses `gradient_steps=2` with the existing `num_envs=16` and `num_env_workers=4` production default.
3. `scripts/train_madsac_3v3.py` now exposes `--gradient-steps`, validates it is positive, and keeps checkpoint signature checks strict.

## Retained reproduction assumptions

- State-dependent log standard deviation is retained as a standard continuous SAC implementation choice.
- Tanh-squashed Gaussian actions and log-probability Jacobian correction are retained.
- `log_std_min`, `log_std_max`, `policy_delay=2`, and fixed `alpha=0.1` are retained as project reproduction assumptions/configuration choices, not claimed as original paper constants unless the paper explicitly states them.
- Current `paper_coupled_team_v2` reward, attack envelope, death ledger, and 3v3 v4 environment are retained as project adaptations.

## Not changed

- No reward scaling or reward terms were changed.
- No MADSAC actor/critic learning rates, batch size, replay capacity, gamma, tau, alpha, log-std bounds, attack parameters, dynamics, or fixed-blue rule policy were changed.
- No outputs or checkpoints were read or modified.
