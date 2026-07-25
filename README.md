# Homogeneous 1v1 UAV Combat v6

This project is a deliberately small, symmetric 1v1 competitive baseline. The physical red and blue teams are episode roles; the learned model identities are policy A and policy B. Neither policy is permanently tied to a color.

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
