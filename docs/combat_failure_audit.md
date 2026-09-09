# Episode-level combat failure audit

`tools/audit_combat_failures.py` is a read-only evaluator for canonical v3.2 Vanilla HAPPO baseline checkpoints. It validates the 100D/117D environment and vanilla/MLP/baseline method contract, then writes only `audit_episodes.csv` and `audit_summary.json` into a new or empty output directory.

Deterministic mode uses the formal evaluation action rule, Gaussian mean followed by `tanh`. Stochastic mode calls the actor's normal `sample(..., deterministic=False)` path, which uses a reparameterized sample followed by `tanh`; it never adds noise to deterministic actions. Episode `i` always resets the environment with `base_seed + i`. Stochastic Torch RNG is also reset to that episode seed for reproducible independent episodes.

Closing rate is `(previous pair distance - current pair distance) / decision_dt`, so positive values mean closing. A pair's first appearance has no previous distance and is omitted from the closing-rate denominator; no synthetic closing value is created. Visibility always calls `env.team_visible`, geometry calls `compute_pairwise_geometry`, and streaks come from the environment's real `_attack_streak`. A full-geometry sample requires one identical Red-Blue pair to meet distance, ATA and AA simultaneously. Geometry, ATA, AA and closing are ground-truth post-hoc diagnostics: they are not claims about what the actor observes, and target invisibility does not remove these diagnostic state measurements.

Tail visibility has two distinct layers. `tail_longest_all_invisible_streak` measures decisions when every remaining Blue is simultaneously invisible. Separately, each Blue alive at episode end is tracked across the tail; `tail_min_survivor_visible_fraction`, `tail_max_survivor_invisible_streak`, and `tail_worst_visible_blue_id` describe the least-visible final survivor (ties use canonical `BLUE_IDS` order). The tail begins strictly after the last Red kill—if a kill occurs at decision step N, post-kill records begin at N+1—or at episode start when there is no Red kill. For a 3-kill draw, the sole-survivor tail and the post-third-kill visibility metrics are checked for exact agreement.

Failure classes use this mutually exclusive priority: `LATE_PROGRESS / POSSIBLE_HORIZON`, `TARGET_LOSS`, `CANNOT_CLOSE`, `BAD_GEOMETRY`, `STREAK_INTERRUPTED`, then `OTHER`. `TARGET_LOSS` means at least one final-surviving Blue is team-invisible for 10 consecutive tail decisions; it does not require every remaining Blue to be invisible simultaneously. These labels are mechanical diagnostic buckets, not scientific causal conclusions, and `POSSIBLE_HORIZON` does not assert that the horizon is too short.

Recovery diagnostics replicate the formal 27-candidate loop without changing `BluePolicy`. `current_state_recovery_guard` is evaluated at the pre-action Blue state. `max_candidate_recovery_guard` is the maximum over predicted states that pass the one-step battlefield check. `selected_action_recovery_guard` belongs to the predicted state of the actual safe-greedy action. Candidate accounting closes as `one_step_rejected + altitude_rejected + safe = evaluated`; dead/no-target records evaluate zero candidates.

Reward-credit fractions in the summary are computed from total matching counts divided by total comparable samples, including post-last-kill and 0/1/2/3-kill-draw groups; episode fractions are never averaged equally. `draw_kill_distribution_fraction` is conditioned on draws. Completion metrics include `P(K>=2|K>=1)`, `P(K>=3|K>=2)`, and `P(K=4|K>=3)`; a zero denominator is represented by JSON `null`. For a 0-kill draw, `steps_after_last_kill` means the full episode length.

`actor_policy_distribution` records each checkpoint actor's three raw `log_std` parameters, the corresponding distribution `std = exp(clamp(log_std, -5, 2))`, per-actor mean standard deviation, and global arithmetic/geometric mean standard deviation. This metadata explains deterministic/stochastic gaps without altering the actor.

Example:

```bash
python -u tools/audit_combat_failures.py \
  --checkpoint outputs/RUN/checkpoint_final.pt \
  --profile main --episodes 5 --seed 1000 --device cuda \
  --policy-mode deterministic --output outputs/RUN/audit_deterministic
```
