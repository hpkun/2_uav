#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -u tools/validate_role_guided_curriculum_configs.py
mkdir -p logs outputs/manifests
timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="outputs/manifests/role_guided_v39_main_curriculum_1m_${timestamp}.csv"
printf '%s\n' 'method,seed,run_name,output_folder,log_file,config,checkpoint_final' > "${manifest}"

methods=(rgaa rgaa_wide dbm_rgaa)
configs=(
  configs/happo_rgaa_v39_curriculum.yaml
  configs/happo_rgaa_wide_v39_curriculum.yaml
  configs/happo_dbm_rgaa_v39_curriculum.yaml
)

for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  config="${configs[$index]}"
  for seed in 5 7 9; do
    run_name="${method}_v39_main_curriculum_seed${seed}_1m_${timestamp}"
    log_file="logs/${run_name}.log"
    output_folder="outputs/${run_name}"
    checkpoint_final="${output_folder}/checkpoint_final.pt"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${method}" "${seed}" "${run_name}" "${output_folder}" "${log_file}" \
      "${config}" "${checkpoint_final}" >> "${manifest}"
    if [[ "${1:-}" == "--dry-run" ]]; then
      echo "[DRY RUN] method=${method} seed=${seed} run=${run_name}"
      continue
    fi
    echo "[START] $(date --iso-8601=seconds) method=${method} seed=${seed} run=${run_name}"
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u algorithm/train_happo.py \
      --steps 1000000 \
      --profile main \
      --seed "${seed}" \
      --device cuda \
      --num-envs 16 \
      --config "${config}" \
      --env-config configs/env_v39.yaml \
      --output-name "${run_name}" \
      --checkpoint-interval 250000 \
      --eval-interval 0 \
      --log-interval 100000 \
      --final-eval-episodes 1 \
      2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -ne 0 ]]; then
      echo "[FAILED] method=${method} seed=${seed} status=${status} run=${run_name}" >&2
      exit "${status}"
    fi
    echo "[COMPLETE] $(date --iso-8601=seconds) method=${method} seed=${seed} run=${run_name}"
  done
done

echo "[ALL COMPLETE] $(date --iso-8601=seconds) manifest=${manifest}"
