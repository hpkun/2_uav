# Heterogeneous MAV/UAV 3v2 environment

`HeterogeneousMAVUAVAirCombatEnv` is the primary environment. Red exposes three
trainable agents in stable order: `MAV`, `UAV1`, and `UAV2`. The two Blue
aircraft are controlled by a fixed, finite-candidate one-step lookahead policy.

Each aircraft uses a dedicated `AircraftSpec`. A normalized three-dimensional
action maps directly to the type-specific `[nx, ny, nz]` range and is integrated
with lightweight three-degree-of-freedom point-mass equations. There are no
yaw, pitch, or speed controllers.

Observations use one self block, two friendly entity blocks, and two enemy
entity blocks. The resulting shape is fixed at 40 for every Red agent. Entity
types and alive masks are explicit; dead entities retain their block position
and carry an inactive mask.

An engagement requires 1000-3000 m range, ATA below 30 degrees, AA below 90
degrees, and three consecutive decision steps. Every aircraft type uses the
same rule. Red wins only when both Blue aircraft are destroyed while the MAV
survives; loss of a UAV does not terminate the episode, and MAV loss is a Blue
win.

The old 4v3 role-oriented experiments are legacy artifacts and are not imported
or exported by the package entry point. New research code should depend only on
`uav_combat.HeterogeneousMAVUAVAirCombatEnv` and
`configs/heterogeneous_mavuav_3v2.yaml`.

`MAVUAVVectorEnv` provides the batched trainer contract. Its action, observation,
and reward shapes are `[num_envs, 3, 3]`, `[num_envs, 3, 40]`, and
`[num_envs, 3]` respectively.
