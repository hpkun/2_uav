# v3.13 Single-Target Weapon Lock

v3.13 (`heterogeneous_mavuav_4v4_v3_13`) is v3.12 with one change:
`combat.weapon_engagement_mode: single_target_lock`. The v3.12 all-pair
resolver and historical contracts remain unchanged.

## Fire-control contract

- UAV1–UAV3 and Blue1–Blue4 have symmetric independent weapon locks.
  MAV is unarmed and its lock is always `None`.
- Full gate is inclusive range [1000, 3000] m, strict ATA <30°, AA <90°.
  It is evaluated once after each 1 s transition, not at 0.1 s RK4 substeps.
- Retain a living target's lock while its full gate remains valid, even when
  another candidate has better geometry. On gate loss, clear the old streak
  and allow acquisition at that SAME boundary, with the new streak starting at 1.
- Acquisition considers only alive full-gate enemies. Sort by smaller ATA,
  then AA, then distance, then canonical target ID. No randomness, reward
  target, sensor eligibility, CAP assignment or learned selector is used.
- All nonlocked pair streaks are zero. A third consecutive valid boundary
  produces at most ONE candidate per attacker.
- Collect all candidates before deaths. Different attackers can kill different
  targets or produce candidates for the same victim; mutual kills remain
  synchronous. Shared kill reward still counts unique victims.
- After resolution, clear dead attackers' locks and locks targeting dead victims,
  and clear their streaks. No second acquisition occurs in this boundary.
  Reacquisition at the next boundary starts at 1 without an additional delay.

## Frozen components

Reward formulas/values, process reward target selection, sensing, CAP navigation,
initial geometry, aircraft dynamics, 100D observations, 117D global state,
3D flight actions, decision/physics intervals, episode horizon and HAPPO
optimizer/curriculum settings are unchanged. There is no ammo, cooldown,
explicit fire action or stochastic hit probability. Changed combat outcomes
can naturally change rewards, but reward definitions do not change.

## State, metadata and diagnostics

`weapon_lock_target` is environment dynamic state and is included in v3.13
vector/subprocess serialization. Restoration rejects missing/invalid locks
or nonlocked positive streaks. Reset clears all locks. Legacy states retain
their original layout and restoration behavior.

Reset/step info and episode diagnostics expose lock targets and current streaks,
without including them in policy observations or global state.
Checkpoints, resolved configuration and evaluation output identify the version,
CAP `coordinated_assignment` strategy and `single_target_lock` weapon mode.
v3.13 trainer/evaluator contracts admit only vanilla actors, MLP team critic
and baseline HAPPO. Prior frozen algorithm contracts are not extended.

## Formal seed3 run (not executed during implementation)

```bash
mkdir -p logs
RUN_NAME="happo_v313_lock_seed3_2m_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u algorithm/train_happo.py \
  --steps 2000000 --profile main --seed 3 --device cuda --num-envs 16 \
  --config configs/happo_v313_baseline.yaml --env-config configs/env_v313.yaml \
  --output-name "$RUN_NAME" --checkpoint-interval 500000 --eval-interval 0 \
  --log-interval 50000 --final-eval-episodes 200 \
  --eval-action-mode stochastic --eval-action-seed 2000 \
  2>&1 | tee "logs/${RUN_NAME}.log"
```

The training configuration is identical to `happo_v312_baseline.yaml`, including
entropy coefficient 0.001, actor initial log_std -0.25 and the 400k curriculum.
Final stochastic environment seeds remain 1000–1199 with action base seed 2000.
