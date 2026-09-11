# PCTA-HAPPO method specification

PCTA-HAPPO means Pursuit-Consistent Target Attention HAPPO. It is an actor-only variant for the frozen v3.5 100D/117D environment. The centralized MLP critic, vanilla HAPPO sequential update, preceding factor, GAE, PPO clipping, advantage normalization, entropy coefficient (`0.01`) and environment are unchanged. The four Red actors remain parameter-independent.

## Actor

The 100D observation is parsed without changing its contract: self plus three friend blocks `[0:44]`, followed by four 14D Blue blocks `[44:58]`, `[58:72]`, `[72:86]`, `[86:100]`. A `44 -> 64` Tanh context encoder produces the Red context. Within each actor, one shared `14 -> 32` Tanh enemy encoder processes all four Blue slots. A `64 -> 32` query scores enemy keys by scaled dot product. Only Blue slots with `alive == 1` and either direct or datalink visibility are eligible. Masked slots receive zero weight; when no valid slot exists, all weights and the enemy aggregate are exactly zero. The concatenated 96D context/enemy representation passes through a `96 -> 128 -> 128 -> 3` Tanh MLP mean head. Each actor retains an independent learned three-value `log_std`, initialized to `-0.5`.

## Pursuit consistency

For adjacent observations belonging to the same environment and Red agent, `alpha_prev` is detached and compared with `alpha_curr` by mean squared difference. A pair is valid only when the previous transition is not terminal/truncated, the agent is active at both times, a previous valid target exists, and the previous attention argmax target remains alive and team-visible at the current time. Target death or complete visibility loss therefore permits immediate reassignment without penalty. The rollout-level auxiliary update uses:

`weighted_loss = pcta_consistency_coef * mean_valid_pairs(MSE(alpha_curr, alpha_prev))`

with project default `pcta_consistency_coef = 0.05`. Each actor receives its auxiliary update after its normal PPO update and before final new log-probability recomputation. Consequently, HAPPO's next preceding factor includes both changes.

Training CSV diagnostics are `pcta_consistency_loss`, `pcta_consistency_weighted_loss`, `pcta_valid_temporal_pairs`, `pcta_attention_entropy`, and `pcta_target_switch_rate`. Switch rate uses only the same valid pairs, so target death or visibility loss is not counted as a bad switch.

## Checkpoint contract

PCTA checkpoints retain `happo_training_checkpoint_v1` and record `actor_variant=pcta`, architecture dimensions and `pcta_consistency_coef`. Vanilla and PCTA checkpoints are mutually rejected by their loaders. Resume restores the same complete trainer/vector/RNG state contract as vanilla HAPPO.

## Design provenance

The following literature facts motivate components but are not claims of reproduction:

- TAPPO (2024) uses attention for multi-UAV target assignment, establishing a target-aware attention precedent in multi-aircraft combat.
- Xu et al. (2025) uses value-attention decomposition for multi-UAV conflict/cooperation credit assignment, supporting attention as a way to expose coordination contributions.
- Chen et al. (2026) identifies target-selection chaos in heterogeneous air combat and uses temporal modelling/attention to improve coordination.
- Jiao et al. (2025) periodically updates Blue target points rather than changing them memorylessly at every instant, providing scenario-level evidence for target persistence.
- Liu et al.'s kill-web work uses sticky target state to address confusion, forgetfulness and recklessness in dynamic target selection.

The project-specific contribution is the combination of actor-internal target-aware attention with temporal pursuit-consistency regularization applied only while the prior target remains valid. Its purpose is to reduce ineffective target hopping and improve sustained pursuit and attack-geometry closure. The coefficient `0.05` is a project hyperparameter, not copied from a paper.
