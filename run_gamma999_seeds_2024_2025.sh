#!/usr/bin/env bash
set -e

python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2024 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/d999_seed2024

python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2024 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/pw999_seed2024

python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2025 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/d999_seed2025

python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2025 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/pw999_seed2025
