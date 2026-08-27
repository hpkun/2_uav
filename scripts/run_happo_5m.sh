#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"

if [[ ! -f scripts/calibrate_mavuav_learnability.py ]]; then
    echo "ERROR: missing $project_root/scripts/calibrate_mavuav_learnability.py" >&2
    echo "Deploy the complete current project instead of the old 1_uav copy." >&2
    exit 2
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python is not available; activate the uav2 Conda environment first." >&2
    exit 2
fi

if ! python -c "import uav_combat" >/dev/null 2>&1; then
    echo "ERROR: the current project is not installed in this Python environment." >&2
    echo "Run: cd '$project_root' && python -m pip install -e ." >&2
    exit 2
fi

num_envs="${NUM_ENVS:-16}"
seed="${SEED:-1}"
sample_steps="${SAMPLE_STEPS:-5000000}"
eval_interval="${EVAL_INTERVAL:-1000000}"
eval_episodes="${EVAL_EPISODES:-50}"
final_eval_episodes="${FINAL_EVAL_EPISODES:-100}"
benchmark_steps="${BENCHMARK_STEPS:-2000}"
device="${DEVICE:-cuda}"
profile="${PROFILE:-main}"
mode="${1:---background}"

if [[ "$mode" == "--smoke-test" ]]; then
    sample_steps=16
    eval_interval=16
    eval_episodes=1
    final_eval_episodes=1
    benchmark_steps=32
    mode="--foreground"
elif [[ "$mode" != "--background" && "$mode" != "--foreground" ]]; then
    echo "Usage: bash scripts/run_happo_5m.sh [--background|--foreground|--smoke-test]" >&2
    exit 2
fi

for value in "$seed" "$num_envs" "$sample_steps" "$eval_interval" "$eval_episodes" "$final_eval_episodes" "$benchmark_steps"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: numeric training options must be positive integers, got '$value'." >&2
        exit 2
    fi
done
if (( sample_steps % num_envs != 0 || eval_interval % num_envs != 0 || benchmark_steps % num_envs != 0 )); then
    echo "ERROR: SAMPLE_STEPS, EVAL_INTERVAL and BENCHMARK_STEPS must be divisible by NUM_ENVS." >&2
    exit 2
fi

if [[ "$device" == "cuda" ]] && ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: DEVICE=cuda was requested, but PyTorch reports CUDA unavailable." >&2
    exit 2
fi
if [[ "$profile" != "learnability" && "$profile" != "main" ]]; then
    echo "ERROR: PROFILE must be 'learnability' or 'main', got '$profile'." >&2
    exit 2
fi

if [[ "$mode" == "--foreground" ]]; then
    run_root="${OUTPUT_ROOT:-/tmp/happo_seed${seed}_smoke_$(date +%Y%m%d_%H%M%S)}"
else
    run_root="${OUTPUT_ROOT:-outputs/happo_5m_seed${seed}_$(date +%Y%m%d_%H%M%S)}"
fi
run_dir="$run_root/happo_seed${seed}"
mkdir -p "$run_dir"

training_command=(
    python -u scripts/calibrate_mavuav_learnability.py
    --algorithm happo
    --profile "$profile"
    --seed "$seed"
    --sample-steps "$sample_steps"
    --eval-interval "$eval_interval"
    --eval-episodes "$eval_episodes"
    --final-eval-episodes "$final_eval_episodes"
    --device "$device"
    --num-envs "$num_envs"
    --benchmark-steps "$benchmark_steps"
    --output-dir "$run_root"
    --skip-baselines
)

echo "Project root: $project_root"
echo "Output root:  $run_root"
echo "Mode:         $mode"
echo "Profile:      $profile"

if [[ "$mode" == "--foreground" ]]; then
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "${training_command[@]}"
    echo "HAPPO smoke/foreground run completed successfully."
    exit 0
fi

echo "$run_root" > "outputs/latest_happo_5m_seed${seed}_run.txt"
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${training_command[@]}" > "$run_dir/run.log" 2>&1 &
train_pid=$!
echo "$train_pid" > "$run_dir/run.pid"

sleep 3
if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "ERROR: HAPPO exited during startup. Log follows:" >&2
    tail -n 100 "$run_dir/run.log" >&2 || true
    exit 1
fi

echo "Training PID: $train_pid"
echo "Log:          $run_dir/run.log"
echo "Follow log:   tail -f '$run_dir/run.log'"
