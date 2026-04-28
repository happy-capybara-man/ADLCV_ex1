#!/usr/bin/env bash
# =============================================================================
# Run LoRA training experiments for checkpoint, CLIP, and KID comparison.
#
# Each experiment writes:
#   - LoRA/checkpoints/TensorBoard events: experiments/<exp_name>/logs
#   - Console log captured by tee: experiment_logs/<timestamp>_<exp_name>.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-0}"
METADATA_JSONL="${METADATA_JSONL:-metadata.jsonl}"
CKPT_STEPS="${CKPT_STEPS:-200}"
LOG_ROOT="${LOG_ROOT:-experiment_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments}"
LOGGING_DIR="${LOGGING_DIR:-logs}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "${LOG_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${METADATA_JSONL}" ]; then
    echo "[ERROR] METADATA_JSONL not found: ${METADATA_JSONL}"
    exit 1
fi

if [ ! -x "./run_train.sh" ]; then
    echo "[ERROR] run_train.sh is not executable"
    echo "        Run: chmod +x run_train.sh"
    exit 1
fi

run_experiment() {
    local name="$1"
    local rank="$2"
    local lr="$3"
    local steps="$4"
    local output_dir="${OUTPUT_ROOT}/${name}_rank${rank}_lr${lr}_steps${steps}"
    local timestamp
    local log_file

    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_ROOT}/${timestamp}_${name}_rank${rank}_lr${lr}_steps${steps}.log"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${output_dir}/pytorch_lora_weights.safetensors" ]; then
        echo "[*] Skipping ${name}: final LoRA already exists at ${output_dir}"
        return 0
    fi

    echo "================================================================"
    echo "  Experiment : ${name}"
    echo "  Rank       : ${rank}"
    echo "  LR         : ${lr}"
    echo "  Steps      : ${steps}"
    echo "  CKPT steps : ${CKPT_STEPS}"
    echo "  Output     : ${output_dir}"
    echo "  TB logdir  : ${output_dir}/${LOGGING_DIR}"
    echo "  Tee log    : ${log_file}"
    echo "================================================================"

    GPU_ID="${GPU_ID}" \
    METADATA_JSONL="${METADATA_JSONL}" \
    OUTPUT_DIR="${output_dir}" \
    LOGGING_DIR="${LOGGING_DIR}" \
    RANK="${rank}" \
    LR="${lr}" \
    MAX_STEPS="${steps}" \
    CKPT_STEPS="${CKPT_STEPS}" \
    ./run_train.sh 2>&1 | tee "${log_file}"
}

# Keep max steps fixed so rank/lr comparisons are controlled.
# Use checkpoints to compare intermediate training steps within each experiment.
run_experiment "baseline" 16 "1e-4" 1000
run_experiment "exp1_rank8" 8 "1e-4" 1000
run_experiment "exp2_lr5e-5" 16 "5e-5" 1000
run_experiment "exp3_lr8e-5" 16 "8e-5" 1000

echo "================================================================"
echo "  All experiments completed."
echo "  Console logs : ${LOG_ROOT}/"
echo "  Outputs      : ${OUTPUT_ROOT}/"
echo "  TensorBoard  : tensorboard --logdir ${OUTPUT_ROOT}"
echo "================================================================"
