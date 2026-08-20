# Heterogeneous MAV/UAV Air Combat

This repository's primary experiment is a lightweight three-dimensional 3v2
multi-agent air-combat environment:

- Red: one MAV and two UAVs, all trainable and all able to attack.
- Blue: two homogeneous high-performance aircraft controlled by a fixed rule.
- Dynamics: six-state, three-degree-of-freedom point mass with direct
  `[nx, ny, nz]` actions.
- Engagement: shared geometric envelope held for three decision steps.
- Observation: unified 40-value entity observation for every Red agent.

## Environment API

```python
import numpy as np
from uav_combat import HeterogeneousMAVUAVAirCombatEnv

env = HeterogeneousMAVUAVAirCombatEnv()
observations, info = env.reset(seed=0)
observations, rewards, terminated, truncated, info = env.step(
    np.zeros((3, 3), dtype=np.float32)
)
```

Agent order is always `MAV`, `UAV1`, `UAV2`. The action shape is `[3, 3]` and
each observation shape is `[40]`.

For batched MARL sampling, `MAVUAVVectorEnv` accepts actions shaped
`[num_envs, 3, 3]` and returns observations shaped `[num_envs, 3, 40]`.

Configuration lives in `configs/heterogeneous_mavuav_3v2.yaml`. Design details
are documented in `docs/heterogeneous_mavuav_3v2.md`.

## Test

Run in the project environment:

```bash
wsl -d Ubuntu
conda activate uav
cd /mnt/c/Users/HPK/Desktop/2_uav
python -m pytest -q tests/test_mavuav_3v2.py
```

Historical role-oriented 4v3 environments, scenarios, policies, configurations,
trainers, diagnostics, and tests have been removed. Remaining generic legacy
experiments are not exported by the package entry point.
