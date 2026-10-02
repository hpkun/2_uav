# Static Combat Capability Audit — v3.10

This report records code-level facts for
`heterogeneous_mavuav_4v4_v3_10`. It does not replace or reinterpret the
historical v3.9 audit. The v3.10 change is a task-semantics correction, not a
performance-tuning intervention.

## Version boundary

- **v3.9 legacy:** MAV is historically attack-capable because it participates
  in the same pairwise Red-to-Blue resolver as the UAVs.
- **v3.10 formal:** MAV is unarmed. Only UAV1/UAV2/UAV3 can directly attack and
  kill Blue aircraft. Blue aircraft can still attack MAV and every UAV.

The v3.10 configuration is byte-for-value equivalent to the resolved v3.9
configuration except for the environment version and the explicit
`combat.mav_can_attack: false` capability. Dynamics, sensors, Blue pursuit,
attack geometry, hold time, rewards, boundary behavior, profiles, episode
horizon, observation dimension (100), and centralized-state dimension (117)
remain unchanged.

## Combat resolver

For v3.10 the resolver resets every MAV-to-Blue pair streak to zero and skips
the gate/event/death-candidate path. Consequently MAV-to-Blue attack events and
MAV-caused Blue deaths are impossible. The four historical MAV-to-Blue streak
slots remain in the 117D global state for network compatibility and stay zero.

UAV-to-Blue and Blue-to-Red pairs retain the frozen 1–3 km, ATA < 30 degrees,
AA < 90 degrees, three-consecutive-decision-step gate. A Blue-to-MAV kill still
ends the episode as a Blue win. Eliminating all Blue aircraft through UAV
attacks still ends it as a Red win.

## Reward semantics

MAV keeps its existing role-process reward: threat, aspect, and awareness. It
does not receive UAV attack-angle, attack-distance, exact-gate, or
attacker-specific kill shaping. No new MAV attack reward is introduced.

The shared +100 Blue-kill event remains part of the team reward received by all
Red agents, including MAV. This is team-task credit, not MAV-specific attack
credit. MAV behavior can affect awareness, team information, survivability,
Blue target allocation, and team geometry, so successful team elimination
remains a shared outcome. This release does not redesign credit assignment.

## Scientific rationale

The formal task defines MAV as a high-value command, sensing, support, and
survivability platform, while UAVs perform direct engagement. This aligns the
combat capability with the existing non-attacking MAV role reward and gives
TACM a clear heterogeneous MAV/UAV role contract. It does not claim that an
unarmed MAV improves win rate.

The following are deliberately unchanged and remain subjects for later audits:
the target score, UAV process reward, shared kill reward, and attack gate.
