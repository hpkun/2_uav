# ERAM-HAPPO / v3.11 implementation contract

ERAM is an independent **baseline team-reward HAPPO** actor/critic variant,
not TAM with a relaxed environment check. It accepts only
`heterogeneous_mavuav_4v4_v3_11` and `heterogeneous_role_coupled_gate_v1`.
Existing TAM and RGAA/TACM environment checks remain unchanged.

## Entity layout and masks

`algorithm/happo/entity_layout.py` mirrors `env/mavuav.py` builders:

- observation100 = self11 + friendly3×11 + enemy4×14;
- global state117 = aircraft8×10 + context37 (32 directional streaks,
  four Blue-killed flags, normalized decision time).

Actor sees only local observations. Friendly keys require alive. Enemy keys
require alive AND (direct OR datalink); datalink-only enemies are valid.
All-masked rows use a temporary zero key and explicitly zero the returned
context and attention weights, including attention output projection bias.
Attention dropout is zero. Diagnostics are no-grad, consume no RNG, do not
update memory and never enter losses/actions.

## Actor

Four independent parameter sets and optimizers, including the MAV:
self11→64 Tanh; shared friend11→64 Tanh; shared enemy14→64 Tanh;
separate 4-head ally/enemy attention, self query; concatenate192→128 Tanh;
GRUCell128→128; policy128→128 Tanh→3. Learned state-independent log_std,
clamp[-5,2], tanh-Gaussian sampling and Jacobian correction match recurrent
HAPPO. Own death gives zero actions/log-prob/entropy/hidden; active loss
filtering and inactive likelihood-ratio=1 remain in the existing trainer.

## Centralized team critic

Shared aircraft10→128 Tanh; independent GRUCell117→128; concatenate
context37+hidden128→128 Tanh context token; aircraft8+context1 tokens;
4-head128 self-attention, residual+LayerNorm; alive/context masked mean;
128→256 LayerNorm Tanh→128 LayerNorm Tanh→1. Context always valid;
critic never uses actor visibility masks. The full state also feeds its GRU,
as specified; masking excludes dead **attention keys/pooling tokens**, not
the historical information in global memory. Baseline MSE value loss is
used, not TAM's Huber loss.

## Temporal training and continuation

`ERAMRolloutBuffer` is a thin subclass of existing dual-memory storage,
without copying rollout or GAE logic. Existing sequence HAPPO performs
ordered contiguous chunks (default16), shuffling chunks only; each actor
and critic starts with the stored chunk-initial hidden state. Multiplicative
recurrent reset masks also cut gradients across resets. Actor memory resets
on individual death/episode boundary, critic memory on episode boundary;
both persist across ordinary rollout boundaries. Bootstrap evaluation does
not advance stored critic memory.

Actor updates retain randomized sequential order, clipped PPO, team GAE,
and old/new likelihood-ratio preceding-factor correction. No role advantage,
auxiliary critic, teacher, router, reward decomposition or credit module.

## Experiment entry and metadata

`python algorithm/train_eram_happo.py` defaults to
`configs/happo_eram_v311.yaml` and `configs/env_v311.yaml`; all standard
training CLI flags remain available. Optimization and curriculum fields
exactly match `configs/happo_v310_ablation.yaml` (entropy0.001,
log_std_init-0.25, 16envs, rollout128, learnability→main400k).
Only variants and ERAM structural fields differ.

Checkpoint identifies `eram_happo` / base `happo`, variants, named entity
layout version, architectures, masks, recurrent configuration, training seed,
sequence length, standard optimizer/RNG/environment continuation and
curriculum state. Structural/semantic mismatch rejects exact resume;
legacy checkpoint loading keeps its existing paths.

`python algorithm/evaluate_happo.py CHECKPOINT --profile main --episodes 100
--device cuda --action-mode stochastic --action-seed 2000` auto-selects
recurrent evaluation for ERAM. Environment seeds start1000 as before.
Inference requires actors only; no critic is run during policy execution,
so there is no inference-time critic memory to reset. Each actor memory
starts zero per episode and resets on death. Existing metrics remain;
`eram_<agent>_<diagnostic>` fields are episode means over that agent's
active decision states, then equally averaged across episodes. They include
token counts, attention entropy/max weights and direct/datalink-only mass.
Zero-valid-target states contribute zero entropy/max/mass. Switching the
diagnostics off leaves actions, RNG ordering and episode results unchanged.
