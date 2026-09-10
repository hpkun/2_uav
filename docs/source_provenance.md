# Source provenance

This project deliberately combines sources and engineering choices; it is not a strict reproduction of one paper.

## A. Directly adopted from a paper

- Xiong et al. (2026), *Manned/unmanned aerial vehicle collaborative interpretable method for intelligent air combat*: overload-controlled 3DOF model; the MAV and UAV performance ranges used here; stronger and higher-value MAV versus lower-value UAV task semantics. The reference paper also defines a stronger, asymmetric Blue performance range. Locally checked against `熊威1 等 - 面向智能空战有人无人机协同可解释方法.pdf`.
- Yang Shuheng, Zhang Dong, Xiong Wei, Ren Zhi, Tang Shuo. *Air combat maneuver decision-making method based on interpretable reinforcement learning*. Acta Aeronautica et Astronautica Sinica, 2024, 45(18): 329922. DOI: `10.7527/S1000-6893.2023.29922`. Chinese citation: 杨书恒，张栋，熊威，任智，唐硕．基于可解释性强化学习的空战机动决策方法．航空学报，2024，45(18)：329922．This project directly uses its 1 s decision interval, 1-3 km engagement distance, 30-degree attacker-angle threshold, 90-degree entering-angle threshold, three consecutive decision steps, five-part situation reward, reward weights `0.32 / 0.43 / 0.10 / 0.10 / 0.05`, and maximum episode length of 75 decision steps.

## B. Combined across papers

- Xiong-style overload dynamics and heterogeneous performance are combined with Yang et al. (2024)'s geometric engagement, situation reward and decision horizon.
- Vanilla HAPPO sequential updates and vanilla MAPPO centralized-training/decentralized-execution are used as algorithm baselines, without paper-specific environment claims.

## C. Project multi-target extensions

- For each team-visible alive Blue, v3.4 takes the maximum Red-vs-Blue situation score over alive Red aircraft and averages those maxima across visible Blue aircraft.
- All cross-team attacker-target pairs maintain independent streaks and resolve kills synchronously.
- Fixed-slot multi-target observation and centralized state, using the v3.4 4v4 contract at 100D/117D; each Blue block includes three Red-relative-velocity values.
- The v3.4 scenario intentionally does not retain the reference paper's stronger asymmetric Blue platform. To construct fair same-type UAV combat, each Blue-team UAV reuses the Red UAV speed and overload limits. This equality covers dynamics and maneuver envelope only; information structure remains intentionally asymmetric.

## D. Project engineering parameters

- Physics step 0.1 s and RK4 integration inside the 1 s decision interval.
- The v3.4 4v4 formation: MAV 1 km behind the three Red UAVs, all eight nominal speeds at 275 m/s, and the existing small seeded initial jitter. The 275 m/s value is the midpoint of the common feasible MAV/UAV interval `[250,300]`, not a paper-specific copied value.
- +/-100 km horizontal volume, 1-20 km altitude, and +/-60-degree pitch guard.
- Blue's O(alive Red) nearest-alive-Red-aircraft targeting over privileged true state, with stable `RED_IDS` tie-breaking, and O(1) direct geometric pursuit. The direct controller uses the existing 3DOF equations to construct achievable overload commands and retains only analytic altitude recovery plus emergency horizontal steer-to-centre safety rules.
- Multiprocessing vector environment with deterministic per-environment auto-reset and a serial reference mode for testing.
- Distance-only heterogeneous sensor ranges (MAV 12 km, UAV 8 km).
- Instantaneous reliable Red datalink and masking of unseen enemy geometry.
- One-hot type fields and the explicit 100D actor-observation layout in v3.4. The `Blue` one-hot value denotes Blue-team UAV identity, not distinct flight performance.
- The 117D centralized state containing all 8 entities, 32 directed attack streaks, Red kill history and time fraction.
- Actor normalization scales: 30 km self x/y, 12 km relative x/y and distance, 10 km relative altitude, and 800 m/s relative velocity. Centralized-state x/y instead map the full battlefield bounds linearly to `[-1,1]`.
- Seeded `main` and `learnability` randomization profiles with team-level and slot-level offsets.
- Explicit propagation and recording of the selected `main` or `learnability` profile across training, benchmark, evaluation and checkpoints.
- The implementation choice of a once-per-step -1 team penalty below 100 m Red friendly distance, without collision physics.

The v3.4 same-type Red/Blue UAV performance contract, multi-target aggregation and Blue direct-pursuit rule controller, along with the sensor ranges, datalink assumptions, observation masking, one-hot encoding, normalization scales, randomization profiles and 100 m safety penalty, are project engineering extensions. The single-pair five-component situation reward is the literature-derived part.

## E. Externally verified literature facts used for v3.4 design

The following facts were supplied as externally verified literature findings for this revision; this record does not claim a new local PDF verification.

- Chen et al. (2026), *A deep reinforcement learning cooperative air combat method with temporal feature and attention enhancement for heterogeneous flight vehicles*, *Aerospace Science and Technology* 176, 112537: heterogeneous roles include flight performance, sensing, survivability and tactical-role differences; the MAV emphasizes coordination, survival and battlefield information/support, while UAVs emphasize cooperative engagement. Its 3v2 and 5v4 examples start all aircraft at 250 m/s, with the MAV behind or subsequently withdrawing behind UAVs. Its Blue is a fixed rule-based greedy opponent, and the 5v4 account permits remaining Blue UAVs to pursue a high-value MAV while Red UAVs still survive. Communication constraints, sensing uncertainty and control delays are left to future work. Chen's unarmed MAV is specific to that mission definition and is not copied: this project's MAV retains its existing attack capability.
- Jiao et al. (2025), *Collaborative decision-making for UAV swarm confrontation based on reinforcement learning*, *IET Control Theory & Applications* 19:e12781: the homogeneous 3v3 setting uses equal Red/Blue UAV capability, 1 s simulation sampling and a fixed rule opponent that reads Red positions for Hungarian target assignment. Its broad speed, attack-range, angle, horizon and randomization values are not copied into this differently scaled environment.
- Jiao et al. (2025) additionally provides the design precedent that rule-controlled Blue aircraft receive target points derived from current Red positions, periodically update them, directly adjust heading and retain fixed speed. v3.4 keeps only the broad principle of a simple position-driven rule; it does not copy the paper's assignment or controller.
- Chen et al. (2026) uses a fixed rule-based finite-state Blue controller composed from predefined basic maneuvers to provide a consistent benchmark. v3.4 uses this only as support for a deterministic lightweight opponent, not as a reproduction of that controller.
- Xu et al. (2025) describes using a default/rule-based low-level maneuver policy in complex air-combat training to reduce the training burden from high-order dynamics. This supports the simplification principle, not the specific equations or parameters used here.

These precedents support the v3.4 abstractions of equal nominal initial speed, a rear high-value MAV, same-dynamics Red/Blue UAVs, a deterministic privileged-state rule opponent, ideal Red information sharing and omission of detailed communication/sensing uncertainty. The replacement of candidate lookahead with direct geometric pursuit is a literature-consistent simplification. It does not make v3.4 a reproduction of any cited paper.
