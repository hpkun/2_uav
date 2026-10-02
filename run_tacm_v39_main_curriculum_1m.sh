#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"

for seed in 5 7 9; do
  run_name="tacm_v39_main_curriculum_seed${seed}_1m_${timestamp}"
  log_file="logs/${run_name}.log"
  echo "[START] $(date --iso-8601=seconds) seed=${seed} run=${run_name}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u algorithm/train_tacm_rgaa.py \
    --steps 1000000 \
    --profile main \
    --seed "${seed}" \
    --device cuda \
    --num-envs 16 \
    --config configs/happo_tacm_rgaa_v39.yaml \
    --env-config configs/env_v39.yaml \
    --output-name "${run_name}" \
    --checkpoint-interval 250000 \
    --eval-interval 0 \
    --log-interval 100000 \
    2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[FAILED] $(date --iso-8601=seconds) seed=${seed} status=${status}" >&2
    exit "${status}"
  fi
  echo "[COMPLETE] $(date --iso-8601=seconds) seed=${seed} run=${run_name}"
done

echo "[ALL COMPLETE] $(date --iso-8601=seconds)"
