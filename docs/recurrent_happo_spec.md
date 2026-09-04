# Recurrent HAPPO baseline

R-HAPPO is the recurrent-actor baseline for the frozen v2.2 environment contract. It does not change environment dynamics, combat, reward, sensing, the 61D actor observation, or the 67D centralized state.

- Three independent actors with no parameter sharing: `61-FC128-Tanh-GRU128-FC128-Tanh-3` plus one state-independent `log_std(3)` per actor.
- Unchanged centralized critic: `67-FC128-Tanh-FC128-Tanh-1`.
- Default TBPTT sequence length: 16. Chunks remain ordered within one environment, never cross environments, and retain a short final chunk without padding or dropping transitions.
- An episode start uses recurrent mask zero. A nonterminal UAV death clears only that UAV's hidden state; a Blue death clears no Red hidden state. Episode completion clears all Red hidden states.
- Hidden state survives a training rollout boundary. Stored chunk-start hidden states are detached, so gradients propagate only inside each TBPTT chunk.
- Active masks control policy-loss and importance-factor participation; recurrent masks control history inheritance. A new episode's first active transition is still trained.
- HAPPO's random sequential actor update and preceding factor are retained. After each actor update, new log probabilities are recomputed with stored chunk-start hidden states and ordered recurrent masks.
- Training checkpoints retain the existing `happo_training_checkpoint_v1` format and additionally store current actor hidden states and recurrent masks for exact recurrent continuation.

This is an R-HAPPO algorithm baseline implementation, not a claim about the final paper method.
