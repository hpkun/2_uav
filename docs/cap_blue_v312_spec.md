# v3.12 CAP-Blue contract

Only target assignment changes from v3.11. `env_v312.yaml` differs at exactly
`environment_version` and `blue_policy.target_strategy`. Baseline training YAML
is identical to v3.11. Reward, sensing, combat, controls and all Red network
contracts remain unchanged. Old algorithms retain their original allowed versions.

CAP enumerates <=256 mappings of living Blue IDs (sorted) to living Red IDs
(sorted). All target loads differ by at most one, including zero loads. Among
legal mappings it minimizes sum of **3D Euclidean distances**, not squared
distances. Within absolute 1e-9 metres of the exact minimum it chooses the
lexicographically smallest target-ID tuple. No role, reward or combat information
enters assignment.

`prepare_step` runs before **any** Blue action. Normal steps 0/2/4 refresh the
entire cohort; intervening steps hold target/heading/pitch. A dead/invalid target,
a changed living-Blue roster or any boundary force-refresh flag causes immediate
cohort refresh on the next step. Preparation is idempotent within a decision
step; flight-control calls never assign targets individually. Reset prepares step0.

Flight control is inherited unchanged from BluePolicy. Altitude/horizontal
recovery, overload commands, inverse trim mapping and clipping are identical.
Recovery force flags affect next step's team preparation, not later aircraft
within the same step. Legacy BluePolicy.prepare_step is a no-op.

CAP state_dict includes the legacy four guidance states (target, heading, pitch,
last-refresh, force flag), prepared step, last assignment step, alive Blue roster
and distances measured at assignment time. Existing vector environment snapshots
already serialize policy state; no checkpoint format change is necessary.

Diagnostics are pure and excluded from observations/rewards/actions. Assignment
distance is the distance at the last assignment, not an instantaneous distance.
Team load diagnostics should be taken after preparation, before physics; post-kill
cached targets may refer to the just-dead target until next-step preparation.

Short rules smoke (CUDA required):

```bash
python tools/audit_cap_blue.py --rule-smoke --output outputs/audits/cap_rules_new
```

After training, the read-only verifier loads a v3.12 vanilla final checkpoint;
stochastic main-profile replay uses environment seed+i / action seed+i, restores
Torch/CUDA RNG and verifies original run SHA-256 and actor parameters unchanged.
Attack contributions are actual attacker-target event credits; simultaneous
attackers can share one unique Blue death. Load fractions use decision steps;
unnecessary duplication uses only the four-Blue/four-Red phase.

```bash
python tools/audit_cap_blue.py --run-dir outputs/YOUR_V312_RUN \
  --episodes 200 --env-seed 1000 --action-seed 2000 \
  --output outputs/audits/YOUR_CAP_AUDIT
```

This tool does not train. Existing audit directories are never overwritten.
