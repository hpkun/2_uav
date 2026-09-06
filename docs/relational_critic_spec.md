# RC-HAPPO relational critic specification

RC-HAPPO changes only the centralized training-time value function. Execution still uses the four independent, feed-forward vanilla HAPPO actors with their unchanged 100→128→128→3 squashed-Gaussian policy.

The canonical 4v4 global state has 119 dimensions. Dimensions 0:80 contain eight fixed-order 10D entity slots (MAV, UAV1, UAV2, UAV3, Blue1, Blue2, Blue3, Blue4). Dimensions 80:119 contain 32 bidirectional attack streaks, four Blue killed-by-Red flags, two actual Blue-mode indicators, and the episode time fraction.

The relational critic applies one shared 10→64 Tanh encoder to all entity slots, followed by one four-head 64D self-attention block, residual connection, and LayerNorm. Dead entities are masked as keys/values and their post-attention query tokens are zeroed. The eight tokens retain their fixed order and are flattened to 512 dimensions; no mean pooling or identity embedding is used.

The 39D context is encoded by a 39→64 Tanh layer. The 512D entity representation and 64D context representation are concatenated and passed through 576→128→1 with Tanh, producing one scalar team value. RC-HAPPO does not change GAE, returns, value loss, PPO clipping, the HAPPO sequential update, or the preceding factor. Recurrent actors cannot be combined with this critic in this implementation.
