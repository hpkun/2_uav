# Source provenance

This project deliberately combines sources and engineering choices; it is not a strict reproduction of one paper.

## A. Directly adopted from a paper

- Xiong et al. (2026), *Manned/unmanned aerial vehicle collaborative interpretable method for intelligent air combat*: overload-controlled 3DOF model; MAV/UAV/Blue performance ranges; stronger and higher-value MAV versus lower-value UAV task semantics. Locally checked against `熊威1 等 - 面向智能空战有人无人机协同可解释方法.pdf`.
- Yang 2024 items designated by the project research specification: 1-3 km interval, 30-degree attacker angle, 90-degree entering angle, three 1 s decision steps, segmented situation-reward equations, and weights. No standalone local PDF with unambiguous `Yang 2024` metadata was present during this refactor; these items should be linked to the final bibliography record before publication.

## B. Combined across papers

- Xiong-style overload dynamics and heterogeneous performance are combined with the Yang-designated geometric engagement and situation reward.
- Vanilla HAPPO sequential updates and vanilla MAPPO centralized-training/decentralized-execution are used as algorithm baselines, without paper-specific environment claims.

## C. Project multi-target extensions

- Each Red agent uses the maximum situation score over currently alive Blue aircraft, with no assigned target.
- Team dense reward sums the three Red slots and always divides by three.
- All cross-team attacker-target pairs maintain independent streaks and resolve kills synchronously.
- Fixed 40D entity observation and fixed 40D centralized state.

## D. Project engineering parameters

- Physics step 0.1 s and RK4 integration inside the 1 s decision interval.
- Exact mirrored 3v2 initial coordinates, interval-midpoint speeds and small seeded initial jitter.
- +/-100 km horizontal volume, 1-20 km altitude, and +/-60-degree pitch guard.
- Blue's 27-candidate overload lookahead and `mixed_episode` target mode.
- Synchronous Python vector environment with per-environment auto-reset.

Chen, Luo and Guo (2026), *A deep reinforcement learning cooperative air combat method with temporal feature and attention enhancement for heterogeneous flight vehicles*, supplies only the heterogeneous 3v2 scenario-size precedent. The local paper explicitly describes TAM-HAPPO with GRU, masking and multi-head attention; none of those extensions are included here.

