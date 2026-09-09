# Episode-level combat failure audit

`tools/audit_combat_failures.py` is a read-only evaluator for canonical v3.2 Vanilla HAPPO baseline checkpoints. It validates the 100D/117D environment and vanilla/MLP/baseline method contract, then writes only `audit_episodes.csv` and `audit_summary.json` into a new or empty output directory.

Deterministic mode uses the formal evaluation action rule, Gaussian mean followed by `tanh`. Stochastic mode calls the actor's normal `sample(..., deterministic=False)` path, which uses a reparameterized sample followed by `tanh`; it never adds noise to deterministic actions. Episode `i` always resets the environment with `base_seed + i`. Stochastic Torch RNG is also reset to that episode seed for reproducible independent episodes.

Closing rate is `(previous pair distance - current pair distance) / decision_dt`, so positive values mean closing. Visibility calls `env.team_visible`, geometry calls `compute_pairwise_geometry`, and streaks come from the environment's real `_attack_streak`. A full-geometry observation requires one identical Red-Blue pair to meet distance, ATA and AA simultaneously.

Failure classes use the mutually exclusive order recorded in the JSON thresholds: late progress with a last kill at or after step 60 and near-completion evidence; target loss with at least 10 consecutive fully invisible tail decisions; cannot close when the visible tail is mostly beyond 3 km and has insufficient positive closing; bad geometry when the tail often enters the distance window but never achieves full same-pair geometry; interrupted streak when geometry or a one/two-step streak occurs without a kill; otherwise `OTHER`. The tail begins strictly after the last Red kill, or at episode start when there is no Red kill.

Example:

```bash
python -u tools/audit_combat_failures.py \
  --checkpoint outputs/RUN/checkpoint_final.pt \
  --profile main --episodes 5 --seed 1000 --device cuda \
  --policy-mode deterministic --output outputs/RUN/audit_deterministic
```
