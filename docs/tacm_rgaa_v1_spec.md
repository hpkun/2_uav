# TACM-RGAA v1

Tactical-context Aware Consistent Mode Routing RGAA is an experimental method on the frozen
`heterogeneous_mavuav_4v4_v3_9` environment. It retains RGAA's team and own-loss-aware role
advantages and DBM-RGAA-v1's private two-mode residual UAV actors. The MAV actor, centralized
team critic, role critics, HAPPO sequential update, and environment semantics are unchanged.

## Actor and teacher

Each UAV uses the unchanged DBM actor: a vanilla base mean plus a two-way soft router and two
private residual experts with residual scale 0.25. The fixed labels are `engagement` and
`cover_support`; they supervise routing but do not turn either expert into a rule policy.

At every pre-action state, a deterministic teacher decodes the 117D global state. It only uses
alive Blue aircraft directly visible to at least one alive Red aircraft under the frozen sensing
ranges. Engagement readiness is the unweighted mean of angle quality, distance quality,
normalized attack streak, and exact attack-gate indicator. MAV threat uses the same four terms
with Blue as attacker and MAV as target. A softmax over interception suitability allocates cover
responsibility across alive UAVs. The final two-way teacher is a temperature-softmax over
threat-modulated engagement readiness and cover responsibility. Its normalized-entropy
confidence makes the no-visible, `[0.5, 0.5]` case contribute zero loss.

Context distillation is confidence-weighted `KL(q || p)`. The router receives detached encoder
features, so this auxiliary objective directly updates only the router. PPO/RGAA gradients still
train the complete policy. The context coefficient decreases linearly from 0.05 to 0.01 over the
first 500,000 sampled environment steps.

## Temporal routing

Router consistency compares consecutive pre-action states only when the UAV remains active,
the preceding transition is not terminated/truncated, no kill or death occurred, engagement and
threat targets are unchanged, and the teacher mode is unchanged. The current router distribution
is matched to a stopped-gradient preceding distribution, weighted by the lower teacher confidence.
This router-only step occurs after PPO and before recomputing log probabilities for the HAPPO
preceding factor.

## Curriculum and reproducibility

The optional method-independent reset curriculum linearly interpolates only the five initial
randomization amplitudes from canonical `learnability` to canonical `main` during the first
400,000 sampled steps. It never changes an episode already in progress. Serial and subprocess
vector environments persist the active override in exact-continuation checkpoints. Evaluation
does not use the curriculum.

The formal 1M launcher performs exactly one deterministic final episode as an execution smoke.
It is labeled `final_evaluation_role: execution_smoke` and is not a formal result. Formal
stochastic evaluation remains a separate invocation using environment seeds 3000--3099 and
independently configured action seeds 4000--4099.

TACM stores a strict versioned teacher, temporal, DBM initialization, and curriculum contract.
The same training seed gives byte-exact initial DBM and TACM actor parameters.

## Provenance boundary

HAPPO sequential heterogeneous optimization, dynamic/specialized role ideas, hierarchical
tactical temporal abstraction, asymmetric MAV/UAV engagement-support roles, and explainable
tactical intention transitions are literature inspirations. The private two-mode residual actor,
team-visible tactical teacher, MAV-threat estimate, soft UAV cover allocation, router
distillation, event-aware temporal consistency, and randomization curriculum are project-specific
designs and are not claimed as direct reproductions of a single source.
