# UAV Combat Environment and MAPPO Training

**Current main experiment: Homogeneous 3v3 Red MAPPO vs Fixed-Rule Blue**

The primary goal is to validate a learnable homogeneous 3v3 air combat environment. Red uses three identical UAVs with a shared-actor MAPPO (centralised training, decentralised execution). Blue uses a paper-grounded nearest-target fixed pursuit policy and never learns.

Success is defined as red eliminating all three blue aircraft through deterministic geometric attacks. Opponent boundary or collision deaths do not count as attack success. The 3v3 scale is the project-validated verification size; it is not presented as a strict reproduction of the MADSAC paper.

The 1v1 self-play experiment is retained as an **auxiliary experiment**. It is no longer the default main experiment.

---

## Homogeneous 3v3 Red MAPPO vs Fixed-Rule Blue

### Paper grounding

- Multi-UAV cooperative Dec-POMDP with CTDE.
- Shared-policy idea: three homogeneous red UAVs share one Actor.
- Red learns against a fixed-rule blue; blue uses step-by-step nearest-alive-red selection with pure-pursuit actions (MADSAC fixed-policy design).
- Success = complete elimination of blue by red attack kills.
- The current reward mode is `paper_coupled_team_v2`.
- The main references from MADSAC are CTDE, a shared Actor for homogeneous UAVs, continuous actions, geometric observations, and a nearest-target fixed blue opponent.

### Project adaptations

- 3v3 scale (not 4v4 or 5v5); a verification scale before further scaling.
- MAPPO algorithm (not MADDPG).
- This is not a strict MADSAC reproduction.
- Deterministic geometric attack/kill logic is used.
- Automatic nearest-in-envelope target selection per aircraft per step.
- No sensor noise, missile model, ammunition count, or weapon cooldown is modelled.
- 68-dim fixed per-agent observation (including own x/y for boundary awareness).
- 48-dim centralised critic state (6 aircraft × 8 features).
- Team dense reward is summed over alive contributors and divided by fixed `team_size=3`.
- 8 parallel environments, 4 persistent CPU workers.
- Best checkpoint ordered by red complete elimination success rate.
- TAM-HAPPO and BRMA-MAPPO are reserved for later heterogeneous and variable-scale stages.

### Learnability calibration v4

`configs/homogeneous_3v3_learnable_v4.yaml` is an isolated environment-design calibration, not a replacement for historical experiments. Existing configurations keep the default `legacy_delta` action mapping for reproducibility.

The v4 configuration uses `rate_aligned_v1`, where the full normalized action range maps to the configured command-rate limits: yaw to `±yaw_rate_max`, pitch to `±pitch_rate_max`, and speed to `±acceleration_max`. This removes command-layer saturation caused by oversized target-state deltas, but it does not remove real physical limits from load factor, roll angle, speed, or pitch constraints.

The v4 initial separation is widened to 3500-5000 m so first entry into the 1000 m near-combat region is closer to the measured turn-time scale. It keeps the same reward parameters, attack model, observation, global state, MAPPO network, and MAPPO training hyperparameters as the base 3v3 configuration. It is still not a strict reproduction of the MADSAC paper environment.

Early training or exploratory policies may still show collisions, boundary deaths, zero kills, large action magnitudes, all-aircraft losses, and volatile returns. Those observations alone are not treated as environment implementation errors. Environment errors should be judged from finite-state failures, finite-reward failures, broken interfaces, non-conserved death ledgers, duplicate deaths, or reproducible logical contradictions.

### Fixed blue rule policies

The default fixed-blue rule policy remains `paper_nearest_pursuit_v1`: each aircraft independently selects the nearest alive enemy at every step and uses the existing pure-pursuit continuous action mapping. Configurations that omit `blue_rule_policy` keep this historical behavior.

`configs/homogeneous_3v3_learnable_v5_greedy_blue.yaml` keeps the v4 dynamics, reward, attack model, initial conditions, and `rate_aligned_v1` action mapping, but sets blue to `greedy_team_pursuit_v1`. This rule first performs a deterministic team-level greedy one-to-one target assignment using only current 3-D distance and aircraft-id tie-breaks, then reuses the same pure-pursuit controller. It is a simple fixed training opponent, not an optimal policy or a complex expert system.

`configs/homogeneous_3v3_learnable_v6_task_aligned.yaml` keeps the v5 dynamics, action mapping, attack parameters, initial conditions, and greedy fixed-blue opponent, but changes the task semantics and dense reward target consistency. Its `target_consistent_team_v3` reward gives each alive aircraft one nearest-alive-enemy reward target per step and uses that same target for approach progress, attack advantage, and reverse-threat terms. Max-step timeout is treated as red task failure and a blue environment outcome, while `blue_complete_elimination_success` still requires three blue attack kills. The attack envelope remains the project-defined deterministic distance + ATA + AA model; v6 is not a strict reproduction of the MADSAC paper's ATA + HA probabilistic weapon model.

### Architecture

| Component | Dim | Notes |
|---|---|---|
| Actor observation | 68 | Self (8) + 2 teammates (12 each) + 3 enemies (12 each) |
| Critic state | 48 | red_0..2 then blue_0..2, each 8 features |
| Action | 3 | yaw, pitch, speed (same as 1v1) |
| Team size | 3 | Three red agents share one Actor |

### Functional heterogeneous 3v3 v1

`configs/heterogeneous_3v3_functional_v1.yaml` is an isolated functional-heterogeneity environment derived from v6. Each side has one support aircraft (`red_0` / `blue_0`) and two combat aircraft (`red_1, red_2` / `blue_1, blue_2`). In heterogeneous mode, logical `red_i` and `blue_i` are same-role physical mirror pairs with the same speed, altitude, heading jitter, role, sensor range, weapon permission, and aircraft spec. This mirror fix is gated by `heterogeneous.enabled=true`; v4/v5/v6 keep their historical scenario ID mapping.

The support role has no weapon permission and a 6000 m deterministic distance sensor. Combat roles can attack and have a 3000 m deterministic distance sensor. A live support aircraft instantly shares only its direct detections to same-team combat aircraft when `support_to_combat=true`; combat-to-combat sharing is always disabled. Fixed enemy slots, hidden-field zeroing, immediate sharing, no combat-to-combat sharing, and enemy status `+1/0/-1` are fixed v1 semantics, not configurable switches. There is no communication delay, loss, noise, FOV, track memory, ammunition, missile entity, GRU, attention module, or type embedding in this first version.

The fixed heterogeneous rule policy `functional_heterogeneous_team_v1` uses visible-pair distance-greedy assignment for combat aircraft, while support aircraft hold a point behind the live combat centroid using the only supported support rule mode, `rear_formation_hold_v1`. This is not an optimal matching solver. HAPPO keeps three independent red actors and one centralized critic: actor 0 controls the support slot, and actors 1/2 control combat slots. This design is inspired by TAM-HAPPO's role heterogeneity idea, but it is not a strict reproduction of that paper's environment.

The `*_kills_with_shared_observation` fields count only kills where, in the same step as the attack, the attacker could not directly sense the target but obtained it through immediate support sharing. Because the default combat sensor range is 3000 m and the attack distance max is 1000 m, this diagnostic is structurally expected to be zero in default experiments. The main support contribution metrics are support coverage, support survival, and the `support_to_combat` on/off ablation. A v6-style timeout is encoded as a blue environment outcome, but it is not blue complete attack elimination.

### Functional heterogeneous red 4v3 v9

`configs/heterogeneous_4v3_main_v9.yaml` is the first main-experiment environment where only red is functionally heterogeneous: `red_0` is a non-attacking support UAV, `red_1..red_3` are combat UAVs, and `blue_0..blue_2` remain homogeneous combat UAVs controlled by a fixed nearest-target pure-pursuit rule. All seven aircraft still share the same point-mass dynamics, controller limits, action mapping, and AircraftSpec; heterogeneity is only role, sensor range, attack permission, red support-to-combat information sharing, tactical duty, and reward composition.

The red support has a 6000 m sensor and broadcasts only its direct blue detections to live red combat aircraft. Red combat aircraft have 1800 m direct sensors and may use support-shared information for early approach, but attack kills require direct observation plus the unchanged geometric attack envelope. Blue combat aircraft have 3000 m sensors, no sharing, and can attack any live red UAV through direct sensing only. `configs/happo_heterogeneous_4v3_main_v9.yaml` keeps HAPPO itself unchanged while expanding the red actor slots to four: support actor plus three combat actors.

### Reward

```
red_team_reward = team_dense_reward
    + red_kill_reward
    - red_attack_death_penalty
    - red_boundary_death_penalty
    - red_collision_death_penalty
    + red_terminal_reward
```

`red_dense_reward` is clipped after combining approach, attack-advantage, threat, soft-boundary, friendly-separation, head-on-risk, and time components. Penalty components are logged as positive magnitudes and subtracted when forming signed totals. Blue receives a symmetric diagnostic reward only (not used for training).

### Commands (3v3)

```bash
# Environment audit
python scripts/audit_3v3_combat_logic.py

# Rule baselines
python scripts/evaluate_rule_3v3.py --episodes 30

# Smoke training (16384 env steps)
python scripts/train_mappo_3v3.py --smoke --device cpu --num-envs 4 --env-workers 1

# Formal training (500k env steps — do not run for smoke validation)
# python scripts/train_mappo_3v3.py --device cuda

# Evaluate checkpoint
python scripts/evaluate_mappo_3v3.py --checkpoint outputs/mappo_3v3_fixed_blue_smoke/checkpoints/final.pt --episodes 60
```

---

## 1v1 Self-Play (auxiliary experiment)

The original 1v1 alternating-freeze competitive baseline is retained as an auxiliary experiment. The physical red and blue teams are episode roles; the learned model identities are policy A and policy B. Neither policy is permanently tied to a color.

The original 1v1 README content follows below.

## Training structure

Policy A and policy B start from numerically identical actor parameters, but own independent actors, critics, optimizers, histories, and generation metadata. Even 100,000-environment-step blocks update A and odd blocks update B. The frozen opponent is selected once per block from its history with 0.7 probability for the latest generation and 0.3 for an older generation. A single available generation is recorded separately as a forced selection.

Episode roles alternate deterministically. Across each block the active policy controls red and blue with count difference at most one. Tail chase also cycles the four active-color/active-position combinations: red/rear, red/front, blue/rear, and blue/front. Global rotation and the original three scenario geometries remain unchanged.

The rollout buffer stores only active-policy transitions:

- observations and global states: `[T, N, 14]`
- actions: `[T, N, 3]`
- log probabilities, rewards, values, dones, returns, advantages: `[T, N]`
- active color diagnostics: `[T, N]`

The critic state is ordered as the active policy's aircraft followed by its opponent. The frozen opponent acts in the environment but never enters the active policy's PPO targets.

There is one learning aircraft per team in the current 1v1 environment. Consequently, the update is mathematically a two-policy alternating-freeze PPO competition. The `mappo` package name is retained for a future same-team multi-agent extension; this 1v1 implementation is not presented as full multi-agent MAPPO.

## Evaluation and selection

Every base initial condition is run twice with the same seed, scenario, and tail rear team: A-red/B-blue, then B-red/A-blue. `--episodes` is the total number of physical episodes and must be positive and even. A formal 300-episode evaluation therefore contains 150 paired initial conditions and gives each policy 150 red and 150 blue roles.

The competitive score is compared lexicographically:

```python
(
    worst_scenario_combat_decisive_rate,
    min_policy_kill_rate,
    paired_combat_decisive_rate,
    -max(policy_a_boundary_loss_rate, policy_b_boundary_loss_rate),
    -max(policy_a_role_kill_gap, policy_b_role_kill_gap),
    -collision_rate,
)
```

Quick-score improvements are saved under `checkpoints/candidates/`. After training, `scripts/select_competitive_best.py` formally evaluates initial, final, and every candidate, writes `competitive_best.pt` and `candidate_selection.csv`, and performs no training. Zero and `PurePursuitPolicy` matchups are post-training distribution diagnostics only.

## Rewards

The default `madsac_segmented` mode adapts segmented geometric shaping and actual-kill success criteria. MADSAC is a cooperative-game reference, not the source of this competitive algorithm.

`configs/homogeneous_1v1_crdrl.yaml` enables the independent `crdrl_coupled` ablation. Its distance/ATA/AA dense formula, parameters, strict sparse thresholds, and sparse value 2 follow CR-DRL. Reusing this project's unified kill/boundary/collision terminal semantics is the project adaptation. The paper emphasizes maintaining advantageous tail-attack position, while this project measures actual kills; the ablation must be compared experimentally before it can be considered as a default.

### Homogeneous 3v3 v7: MADSAC paper-segmented reward

`configs/homogeneous_3v3_learnable_v7_paper_segmented.yaml` adds the isolated `paper_segmented_team_v4` reward mode. It keeps the v6 homogeneous 3v3 environment, dynamics, attack envelope, observations, 600-step timeout, and fixed greedy blue rule policy, but replaces the v6 target-consistent dense/terminal design with an Eq. (25)-style segmented R3/R41/R42 adaptation.

The v7 dense reward uses nearest alive target selection, fixed division by team size 3, 30/15/5 degree attack tiers, signed reverse-threat penalties, `+10` attack kills, `-10` own aircraft deaths, non-success survivor penalties, and an extra mutual-elimination penalty. See `docs/paper_segmented_reward_v4.md` for the exact mapping and the differences from v6.

## Parallel environment

Training throughput can be improved by running CPU environment steps across multiple persistent worker processes. This is a **runtime performance optimisation only**; it does not change the training algorithm, PPO targets, reward formula, or any other research behaviour.

- `num_envs` (default 32) is the total number of environment slots in the rollout.
- `num_env_workers` (default 4) is the number of CPU worker processes that advance those environments in parallel. Each worker manages `num_envs / num_env_workers` environments.
- Setting `num_env_workers = 1` runs a **sequential** `LocalCombatVectorEnv` in the main process — this is the correctness baseline and a fallback when a single process is preferred.
- Setting `num_env_workers > 1` spawns persistent worker processes via `SubprocessCombatVectorEnv`. Workers are long-lived, so process creation cost is paid once at startup, not per step.
- The Actor, Critic, and PPO update stay exclusively on the **main process / GPU**. Workers only run CPU dynamics, attack logic, and reward computation.
- `multiprocessing.get_context("spawn")` is used to avoid `fork`-related CUDA errors in the main process.
- Checkpoints **do not store** worker process state, pipes, or in-flight environments. On resume the workers are re-created fresh. Old v6 checkpoints remain fully compatible.
- Whether the parallel backend actually accelerates training depends on the machine's CPU core count and load. Run `python scripts/benchmark_parallel_env.py` to measure throughput and speedup honestly.

CLI examples:

```bash
python scripts/train_mappo.py --device cuda --env-workers 4
python scripts/train_mappo.py --device cuda --env-workers 1   # sequential baseline
python scripts/select_competitive_best.py --env-workers 4 --episodes 12 --device cuda
python scripts/benchmark_parallel_env.py --num-envs 32 --env-workers 4 --vector-steps 256
```

## Commands

Run in the `uav` Conda environment:

```bash
python -m pip install -e .
pytest -q
python scripts/train_mappo.py --smoke --device cuda
python scripts/select_competitive_best.py --output-dir outputs/mappo_v6_smoke --episodes 12 --device cuda
```

A resume may change only total environment steps, device, output directory, and `num_env_workers`. All other training-signature changes are rejected. Resume starts a new episode batch because physical simulator and worker-process state are not checkpointed, while model, optimizer, opponent, scheduler counters, candidates, best score, and RNG states are restored. v5 and earlier checkpoints are explicitly rejected. `num_env_workers` can be changed on resume (e.g. from 1 to 4 or vice-versa) without upgrading the checkpoint version.
