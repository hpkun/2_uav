#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -u tools/validate_role_guided_curriculum_configs.py --environment-version v3_10
mkdir -p logs outputs/manifests
timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="outputs/manifests/tacm_v310_main_curriculum_1m_${timestamp}.csv"
printf '%s\n' 'method,seed,entrypoint,config,environment_config,environment_version,run_name,output_folder,log_file,checkpoint_final' > "${manifest}"
method="tacm_rgaa"; entrypoint="algorithm/train_tacm_rgaa.py"
config="configs/happo_tacm_rgaa_v310.yaml"; environment_config="configs/env_v310.yaml"
environment_version="heterogeneous_mavuav_4v4_v3_10"

for seed in 5 7 9; do
  run_name="tacm_v310_main_curriculum_seed${seed}_1m_${timestamp}"
  output_folder="outputs/${run_name}"; log_file="logs/${run_name}.log"
  checkpoint_final="${output_folder}/checkpoint_final.pt"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${method}" "${seed}" "${entrypoint}" "${config}" "${environment_config}" \
    "${environment_version}" "${run_name}" "${output_folder}" "${log_file}" \
    "${checkpoint_final}" >> "${manifest}"
  if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY RUN] method=${method} seed=${seed} entrypoint=${entrypoint} config=${config} environment_config=${environment_config} run=${run_name} checkpoint=${checkpoint_final}"
    continue
  fi
  echo "[START] $(date --iso-8601=seconds) method=${method} seed=${seed} run=${run_name}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u "${entrypoint}" --steps 1000000 --profile main --seed "${seed}" \
    --device cuda --num-envs 16 --config "${config}" --env-config "${environment_config}" \
    --output-name "${run_name}" --checkpoint-interval 250000 --eval-interval 0 \
    --log-interval 100000 --final-eval-episodes 1 2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}; set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[FAILED] method=${method} seed=${seed} status=${status} run=${run_name}" >&2
    exit "${status}"
  fi
  echo "[COMPLETE] $(date --iso-8601=seconds) method=${method} seed=${seed} run=${run_name}"
done
echo "[ALL COMPLETE] $(date --iso-8601=seconds) manifest=${manifest}"
