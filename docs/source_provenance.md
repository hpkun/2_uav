# Source provenance

This project deliberately combines sources and engineering choices; it is not a strict reproduction of one paper.

## A. Directly adopted from a paper

- Xiong et al. (2026), *Manned/unmanned aerial vehicle collaborative interpretable method for intelligent air combat*: overload-controlled 3DOF model; MAV/UAV/Blue performance ranges; stronger and higher-value MAV versus lower-value UAV task semantics. Locally checked against `熊威1 等 - 面向智能空战有人无人机协同可解释方法.pdf`.
- Yang Shuheng, Zhang Dong, Xiong Wei, Ren Zhi, Tang Shuo. *Air combat maneuver decision-making method based on interpretable reinforcement learning*. Acta Aeronautica et Astronautica Sinica, 2024, 45(18): 329922. DOI: `10.7527/S1000-6893.2023.29922`. Chinese citation: 杨书恒，张栋，熊威，任智，唐硕．基于可解释性强化学习的空战机动决策方法．航空学报，2024，45(18)：329922．This project directly uses its 1 s decision interval, 1-3 km engagement distance, 30-degree attacker-angle threshold, 90-degree entering-angle threshold, three consecutive decision steps, five-part situation reward, reward weights `0.32 / 0.43 / 0.10 / 0.10 / 0.05`, and maximum episode length of 75 decision steps.

## B. Combined across papers

- Xiong-style overload dynamics and heterogeneous performance are combined with Yang et al. (2024)'s geometric engagement, situation reward and decision horizon.
- Vanilla HAPPO sequential updates and vanilla MAPPO centralized-training/decentralized-execution are used as algorithm baselines, without paper-specific environment claims.

## C. Project multi-target extensions

- Each Red agent uses the maximum situation score over currently Red-team-visible alive Blue aircraft, with no assigned target; no visible target contributes zero.
- Team dense reward sums the three Red slots and always divides by three.
- All cross-team attacker-target pairs maintain independent streaks and resolve kills synchronously.
- Fixed-slot multi-target observation and centralized state, using the v2.2 environment contract at 61D/67D; each Blue block includes three Red-relative-velocity values.

## D. Project engineering parameters

- Physics step 0.1 s and RK4 integration inside the 1 s decision interval.
- Exact mirrored 3v2 initial coordinates, interval-midpoint speeds and small seeded initial jitter.
- +/-100 km horizontal volume, 1-20 km altitude, and +/-60-degree pitch guard.
- Blue's 27-candidate overload lookahead and `mixed_episode` target mode.
- Synchronous Python vector environment with per-environment auto-reset.
- Distance-only heterogeneous sensor ranges (MAV 12 km, UAV 8 km).
- Instantaneous reliable Red datalink and masking of unseen enemy geometry.
- One-hot type fields and the explicit 61D actor-observation layout in v2.2.
- The 67D centralized state extension containing streaks, Red kill history, actual Blue episode mode and time fraction.
- Actor normalization scales: 30 km self x/y, 12 km relative x/y and distance, 10 km relative altitude, and 800 m/s relative velocity. Centralized-state x/y instead map the full battlefield bounds linearly to `[-1,1]`.
- Seeded `main` and `learnability` randomization profiles with team-level and slot-level offsets.
- Explicit propagation and recording of the selected `main` or `learnability` profile across training, benchmark, evaluation and checkpoints.
- The implementation choice of a once-per-step -1 team penalty below 100 m Red friendly distance, without collision physics.

The sensor ranges, datalink assumptions, observation masking, one-hot encoding, normalization scales, randomization profiles and 100 m safety penalty are project engineering extensions. They are not claimed to be parameters reproduced verbatim from the cited papers.

Chen, Luo and Guo (2026), *A deep reinforcement learning cooperative air combat method with temporal feature and attention enhancement for heterogeneous flight vehicles*, supplies only the heterogeneous 3v2 scenario-size precedent. The local paper explicitly describes TAM-HAPPO with GRU, masking and multi-head attention; none of those extensions are included here.
