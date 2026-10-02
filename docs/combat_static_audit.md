# Static Combat Mechanism Audit — v3.9 / TACM-RGAA

This is a code-level audit, not a behavioral or causal conclusion. It describes the frozen
`heterogeneous_mavuav_4v4_v3_9` implementation used by TACM-RGAA.

## Attack and kill chain

For every alive aircraft, the combat resolver iterates independently over every alive opponent.
For a Red attacker and a Blue target, the pair gate is true only when distance is inclusively
within 1–3 km, ATA is strictly below 30 degrees, and AA is strictly below 90 degrees. The
pair-specific streak increments while the gate remains true and resets to zero otherwise. At a
streak of three decision steps, the pair becomes an attack-event candidate and the target is
deactivated as `red_attack`.

No explicit fire action, target lock, reward-selector match, or visibility predicate is required by
the combat resolver. Each attacker–target pair has its own streak. A Red aircraft can therefore
form valid pairs against multiple Blue aircraft in one step, and one Blue can have multiple Red
candidate attackers in the same step. The implementation retains all candidate pairs but assigns
only a target-level death cause; it does not define a unique killer.

MAV is included in the same Red attack loop and can kill Blue aircraft. This is an implementation
fact, irrespective of its intended high-value coordination role.

## Reward target versus combat target

For each alive UAV, the v3.9 process-reward target is the maximum `target_score` among alive,
team-visible Blue aircraft. The score combines angle, an in-range indicator, relative altitude,
and relative speed. This selected target controls that UAV's process reward only. Combat evaluates
all alive Blue targets independently, so `reward target = A` and `combat kill target = B` is
possible in code. MAV has no corresponding UAV reward-target selector.

The UAV process signal contains:

- angle quality: `1 - (ATA + AA)/(2*pi)`;
- distance quality: a smooth exponential quality outside/below the 1–3 km interval and 1 inside;
- coupled dense reward: `angle_quality * distance_quality - 0.5`;
- exact gate reward: `0.5 * I[distance and both angle conditions satisfy the combat gate]`.

The +100 Blue-kill event is shared. In v3.9, each Red reward is its own role-process reward plus
the same shared event/terminal/safety value. Consequently a UAV that did not form the killing pair
can still receive the kill signal through team reward.

RGAA auxiliary reward contains the corresponding process reward plus only that agent's own-loss
event. It excludes the shared Blue-kill and terminal rewards. Its actor advantage is fused with the
team advantage, however, so every actor still receives shared kill credit through the team branch.
The auxiliary branch can reduce role-credit ambiguity but cannot eliminate shared team credit.

## Visibility, Blue pursuit, and temporal persistence

The combat resolver has no explicit visibility test. Separately, UAV direct sensing is 8 km and MAV
direct sensing is 12 km. Therefore an alive Red attacker and alive Blue target satisfying the
1–3 km kill gate are necessarily direct-visible under the current range-only sensor model. These
are distinct facts: visibility is not a combat predicate, but it is implied by gate distance in the
current configuration.

Blue uses periodically refreshed nearest-Red pursuit guidance. This can bring Blue aircraft close
to Red aircraft and can create engagement opportunities; the static code alone cannot establish
that it causes the observed Red performance.

A kill cannot result from a single-step gate coincidence: the same attacker–target pair must remain
inside the gate for three consecutive one-second decision steps. Nevertheless, aircraft inertia may
allow a streak-2 geometry to persist under trim or mismatched actions, which is why the audit includes
a paired one-step streak-2 intervention.

## Classified findings

### CLEARLY INTENTIONAL BY DESIGN

- Pairwise 1–3 km / 30-degree ATA / 90-degree AA / three-step attack gate.
- Independent streak state for every cross-team attacker–target pair.
- Smooth UAV geometry reward plus an exact-gate bonus.
- Shared event and terminal reward combined with agent-specific role-process reward.
- Nearest-Red Blue pursuit and range-only direct/datalink sensing contracts.

### POTENTIAL CREDIT-ASSIGNMENT AMBIGUITY

- Blue-kill reward is shared rather than attacker-specific.
- Reward-selected UAV target is not a combat lock and need not match the killed target.
- Multiple simultaneous candidate attackers are not reduced to a unique killer.
- RGAA retains shared kill credit in the team advantage even though its auxiliary advantage excludes it.

### POTENTIAL INCIDENTAL-KILL MECHANISM

- No explicit fire decision or target lock is required after three-step gate persistence.
- One Red attacker can satisfy multiple target gates in the same step.
- Blue pursuit can move targets into favorable geometry.
- At streak two, immediate completion may sometimes persist under trim or a peer action because of
  existing geometry and inertia. This is a hypothesis tested by paired intervention, not a conclusion.

### REALISM LIMITATION BUT NOT NECESSARILY PERFORMANCE BUG

- MAV uses the same geometric attack resolver and can kill Blue aircraft.
- There is no missile model, ammunition, firing authorization, probabilistic hit model, or damage state.
- Kill is an instantaneous consequence of the third valid decision-step gate.
- Sensors are range-only; occlusion and identification uncertainty are not modeled.

None of these static findings alone demonstrates that reported kills are incidental or that learned
control is effective. The paired behavioral audit is required to measure those questions.
