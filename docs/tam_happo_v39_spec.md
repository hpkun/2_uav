# TAM-HAPPO algorithm transfer reproduction on heterogeneous_mavuav_4v4_v3_9

This implementation transfers the core TAM-HAPPO structure to the frozen v3.9 continuous-control environment. It is not an exact reproduction of the paper's simulator, weapons, action space, or reward.

## Network and training contract

Each of the four independent actors applies state memory before policy fusion: `100D observation -> GRUCell(100,128) -> Linear(128,100) -> concat(current 100D observation) -> 200 -> 256 -> 128 -> 3D tanh-squashed Gaussian`. The four actors and optimizers share no parameters.

The centralized state-only critic parses the exact 117D state as eight 10D aircraft blocks, 32 cross-team attack-streak values, four Blue-kill flags, and normalized episode progress. A `GRUCell(117,128)` supplies temporal context before a single 4-head attention layer over eight aircraft tokens plus one always-valid context token. Dead aircraft are masked. Masked pooling feeds `Linear(128,256) -> LayerNorm(256) -> Tanh -> Linear(256,128) -> LayerNorm(128) -> Tanh -> Linear(128,1)`, producing team `V(s)`. Both post-attention MLP layers use LayerNorm as required by the paper. The attention residual connection and its LayerNorm are a `PAPER_UNSPECIFIED_ADAPTATION`.

Actors and critic use ordered TBPTT chunks of length 16. Actor and critic memories persist across rollout boundaries, reset at episode boundaries, and dead-agent actor memory is cleared. Standard randomized sequential HAPPO actor updates, detached preceding factors, PPO clipping, team return, and GAE remain unchanged. TAM alone uses Huber value loss with delta 10.

## Paper-to-project mapping

| Classification | Items |
|---|---|
| `PAPER_EXPLICIT` | GRU State Memory; inactive mask; entropy regularization; multi-head attention critic; independent HAPPO policies and sequential update; GRU width 128; policy MLP `[256,128]`; learning rates `5e-4`; clip `0.2`; entropy coefficient `0.01`; GAE `0.95`; gamma `0.99`; gradient clip `10`; Huber coefficient `10` |
| `CURRENT_ENVIRONMENT_ADAPTATION` | 100D actor observations; 117D centralized state; 3D continuous tanh-Gaussian actions and learned state-independent `log_std`; v3.9 dynamics, heterogeneous reward, Blue controller, attack gate, sensing, and MAV/UAV kinematics |
| `PAPER_UNSPECIFIED_ADAPTATION` | TBPTT length 16; four attention heads; continuous-action `log_std`; tanh activations; exact eight-entity plus context-token implementation |
| `PAPER_AMBIGUITY_RESOLUTION` | A state-only centralized `V(s)` is retained instead of ambiguous state-plus-joint-action prose, preserving mathematically valid HAPPO GAE semantics |
| `NOT_APPLICABLE` | Discrete unavailable-action masking: every dimension of the current 3D continuous control action is always available |

## Masks and ablations

With `tam_inactive_mask: true`, dead agents emit zero actions, contribute neither policy loss nor entropy, have zero recurrent state, and contribute likelihood ratio one to subsequent actors' preceding factor. Dead entity tokens are excluded as critic keys/values while the context token remains valid.

The full configuration enables `tam_state_memory`, `tam_attention`, and `tam_inactive_mask`. Each can be disabled independently for future `No_State_Memory`, `No_Attention`, and `No_Mask` scientific ablations; no such ablation training is performed by this implementation task.

Two configurations are supplied: `happo_tam_v39_paper.yaml` uses the paper-explicit optimization values where transferable, while `happo_tam_v39_matched.yaml` preserves the current entropy-0.001/log-std-0.25 baseline optimization contract for structural comparison.
