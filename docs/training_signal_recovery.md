# Signal-aware training exit and v3.10 seed1 audit

The flat HAPPO entry (also used by TACM) now handles SIGTERM, SIGINT and,
where available, SIGHUP. Handlers only record the first request. The next
complete rollout/update boundary writes its training CSV row, atomically
publishes `checkpoint_emergency_<actual_steps>.pt`, writes `termination.json`,
flushes the exit log, and exits with `128 + signal_number`. No final evaluation
is started after a handled request at that boundary.

The checkpoint is the ordinary complete training checkpoint: actor/team/role
critic parameters, optimizers, main/auxiliary RNG, CUDA/CPU RNG, environment
states, masks, reset counts and config. Its format and resume semantics are
unchanged. Resume using the existing training entry, original configs, original
16-env/128-step contract, `--resume <emergency-checkpoint>`, and the original
total target (not additional steps). For the currently interrupted run only
`checkpoint_501760.pt` exists; a handler added today cannot recover the unsaved
subsequent updates. No training was resumed to 2M during implementation.

SIGKILL, a killed worker/process group, interpreter crash, WSL shutdown, host
shutdown, and inaccessible disk cannot be made safe by a Python signal handler.
If collection/update fails before the boundary, use the last valid checkpoint.
Do not describe this as protection against OOM-killer SIGKILL.

`tools/run_happo_tacm_seed1.py` is the replacement for the ad-hoc serial shell
command. It requires CUDA, runs the same seed1 two-method v3.10 protocol,
records child exit codes and COMPLETE / FAILED / TERMINATED_BY_SIGNAL, forwards
handled signals to the active child, and stops on any abnormal child exit.
On Linux the child uses its own session so terminal Ctrl-C is forwarded to the
training parent rather than interrupting every environment worker.

The independent audit tool defaults to **offline-only** (`--episodes 0`). It
reads existing configs/checkpoints without modifying them. Opt-in replay is
main-profile stochastic with env seeds 1000+episode and action seeds
2000+episode. It requires CUDA and restores RNG after replay. It checks a
maximum of five episodes against the formal evaluator with exact metrics;
the HAPPO full 200-episode replay also checks the stored final evaluation.
Failed fidelity means the audit must not be used. Output must be a new directory
outside the original run. Checkpoint SHA and actor parameters are checked.

Training episode metrics are weighted by increments in completed episodes.
Update diagnostics are weighted by sampled transitions. Per-rollout death
event counters are summed, not multiplied by episode counts. Phase attribution
uses the update endpoint; episodes can span a phase boundary. These are training
data, not fixed-checkpoint evaluation. The early boundary hypothesis is VERIFIED
only if recorded boundary deaths exceed half the recorded early UAV own-loss
events; absent counters give INCONCLUSIVE.

Episode death records retain raw cause and decision step for each Red agent.
Agent-level consistency must agree with MAV/UAV survivors. Death summaries
include raw/category counts and rates, mean/median/p10/p90 death steps, and
outcome-conditioned results. Draw summaries include Blue survivor counts, Red
kill counts, and the joint `3 kills + 1 Blue survivor` group with remaining UAV
and MAV survival. For a full 200-episode audit, last-Blue predominance is VERIFIED
if this group is over half of draws, PARTIALLY_SUPPORTED if nonzero but not a
majority, REFUTED if absent, and INCONCLUSIVE when there are no draws. Tiny smoke
always remains INCONCLUSIVE. This is a descriptive endpoint decomposition,
not a causal explanation of failed pursuit.

All actor restoration follows the checkpoint: baseline uses IndependentActors;
TACM uses the actual DBM residual/router actor builder plus TACM metadata checks.
Neither this audit nor the implementation extends TACM/RGAA to v3.11.
