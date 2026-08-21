# MAV/UAV Heterogeneous 3v2 Air Combat

A compact research environment for multi-agent reinforcement-learning experiments:

- Red: one armed, high-performance, high-value MAV and two armed, lower-performance, lower-value UAVs.
- Blue: two homogeneous aircraft controlled by a fixed rule policy.
- Red's three aircraft are trainable agents; Blue is never trained.

The environment uses overload-controlled 3DOF point-mass dynamics. Each Red actor emits the normalized action `[ux, uy, uz]` in `[-1, 1]^3`; the environment maps it around the current trim overload to physical `[nx, ny, nz]`. One RL decision lasts 1 s and contains ten 0.1 s RK4 physics steps.

Combat is a geometric persistent engagement: 1-3 km distance, attacker target angle below 30 degrees, entering angle below 90 degrees, held at three consecutive decision boundaries. MAV, UAV and Blue use the same rule. There are no missiles, sensors, communication, target-assignment actions, recurrent networks or attention modules.

## Install and inspect

```bash
pip install -e .
python scripts/audit_mavuav_env.py --steps 1000
pytest
```

```python
import numpy as np
from uav_combat import HeterogeneousMAVUAVAirCombatEnv

env = HeterogeneousMAVUAVAirCombatEnv()
observations, info = env.reset(seed=1)
observations, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
state = env.global_state()
```

Each Red observation is 40D, the centralized state is 40D, and the Red active mask is ordered `[MAV, UAV1, UAV2]`.

## Baselines

- HAPPO: three independent feed-forward actors, sequential policy updates, one centralized critic.
- MAPPO: one shared feed-forward actor conditioned by the observation type field, one centralized critic.

```bash
python scripts/train_happo_mavuav.py --updates 10
python scripts/train_mappo_mavuav.py --updates 10
python scripts/evaluate_happo_mavuav.py outputs/happo_mavuav.pt
python scripts/evaluate_mappo_mavuav.py outputs/mappo_mavuav.pt
```

Formal evaluation reports `nearest` and `mav_priority` Blue target modes separately. This project combines published ingredients with explicit engineering adaptations; it does not claim to reproduce any single paper's environment exactly. See [environment specification](docs/environment_spec.md) and [source provenance](docs/source_provenance.md).
