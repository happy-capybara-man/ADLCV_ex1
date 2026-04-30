#!/usr/bin/env bash
# =============================================================================
# Prior-preservation LoRA rank sweep for SD3.5 DreamBooth.
#
# Default plan:
#   - Train ranks 4, 8, 16, 32, 64 with the prior-class-data setup.
#   - Use 800 steps so checkpoint comparison can cover 200/400/600/800.
#   - Generate 10 prompts x 3 seeds for each checkpoint.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
RUN_TRAIN="${RUN_TRAIN:-${SCRIPT_DIR}/run_train.sh}"
PROMPTS_FILE="${PROMPTS_FILE:-old_prompt.json}"
METADATA_JSONL="${METADATA_JSONL:-metadata.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments}"
LOG_ROOT="${LOG_ROOT:-experiment_logs}"
GENERATED_ROOT="${GENERATED_ROOT:-generated_images}"
LOGGING_DIR="${LOGGING_DIR:-logs}"

GPU_ID="${GPU_ID:-0}"
RANKS="${RANKS:-4 8 16 32 64}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-800}"
CKPT_STEPS="${CKPT_STEPS:-200}"
GENERATE_CHECKPOINT_STEPS="${GENERATE_CHECKPOINT_STEPS:-200 400 600 800}"
N_PER_PROMPT="${N_PER_PROMPT:-3}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_WARMUP="${LR_WARMUP:-100}"
SEED="${SEED:-42}"

CLASS_DATA_DIR="${CLASS_DATA_DIR:-class_data}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
PRIOR_GENERATION_PRECISION="${PRIOR_GENERATION_PRECISION:-bf16}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-1}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

SWEEP_NAME="${SWEEP_NAME:-prior_rank_sweep_${MIXED_PRECISION}_lr${LR}_steps${MAX_STEPS}}"
GENERATION_OUTPUT_ROOT="${GENERATION_OUTPUT_ROOT:-${GENERATED_ROOT}/${SWEEP_NAME}_ckpts_n${MAX_PROMPTS}x${N_PER_PROMPT}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${LOG_ROOT}/${TIMESTAMP}_${SWEEP_NAME}.log}"
RECORD_FILE="${RECORD_FILE:-${LOG_ROOT}/${SWEEP_NAME}.md}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${GENERATION_OUTPUT_ROOT}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python not executable: ${PYTHON_BIN}"
    exit 1
fi

if [ ! -x "${RUN_TRAIN}" ]; then
    echo "[ERROR] Training script not executable: ${RUN_TRAIN}"
    echo "        Run: chmod +x ${RUN_TRAIN}"
    exit 1
fi

if [ ! -f "${PROMPTS_FILE}" ]; then
    echo "[ERROR] PROMPTS_FILE not found: ${PROMPTS_FILE}"
    exit 1
fi

if [ ! -f "${METADATA_JSONL}" ]; then
    echo "[ERROR] METADATA_JSONL not found: ${METADATA_JSONL}"
    exit 1
fi

write_record_header() {
    cat > "${RECORD_FILE}" <<EOF
# ${SWEEP_NAME}

Start time: ${TIMESTAMP}

## Rationale

- Baseline: experiments/prior_class_data_bf16_rank16_lr1e-4_steps600
- Sweep variable: LoRA rank only (${RANKS})
- Recommended steps: ${MAX_STEPS}, because earlier observations suggest 1000+ can overfit, while 600 may still be borderline underpowered for low ranks.
- Checkpoints for visual comparison: ${GENERATE_CHECKPOINT_STEPS}
- Generation budget: first ${MAX_PROMPTS} prompts x ${N_PER_PROMPT} seeds per checkpoint

## Fixed training settings

- LR: ${LR}
- LR warmup: ${LR_WARMUP}
- Mixed precision: ${MIXED_PRECISION}
- Resolution: ${RESOLUTION}
- Train batch size: ${TRAIN_BATCH_SIZE}
- Gradient accumulation: ${GRAD_ACCUM}
- Prior preservation: enabled
- Class data dir: ${CLASS_DATA_DIR}
- Class images: ${NUM_CLASS_IMAGES}
- Prior loss weight: ${PRIOR_LOSS_WEIGHT}
- Seed: ${SEED}

## Outputs

- Experiments: ${OUTPUT_ROOT}/prior_class_data_${MIXED_PRECISION}_rank*_lr${LR}_steps${MAX_STEPS}
- Generated images: ${GENERATION_OUTPUT_ROOT}
- Console log: ${RUN_LOG}

## Commands

EOF
}

record_command() {
    {
        echo '```bash'
        printf '%q ' "$@"
        echo
        echo '```'
        echo
    } >> "${RECORD_FILE}"
}

train_rank() {
    local rank="$1"
    local exp_name="prior_class_data_${MIXED_PRECISION}_rank${rank}_lr${LR}_steps${MAX_STEPS}"
    local output_dir="${OUTPUT_ROOT}/${exp_name}"

    echo "================================================================"
    echo "  Train rank : ${rank}"
    echo "  Steps      : ${MAX_STEPS}"
    echo "  Output     : ${output_dir}"
    echo "================================================================"

    if [ "${SKIP_EXISTING_TRAIN}" = "1" ] && [ -f "${output_dir}/pytorch_lora_weights.safetensors" ]; then
        echo "[*] Final LoRA already exists, skipping training: ${output_dir}"
        return 0
    fi

    record_command env \
        GPU_ID="${GPU_ID}" \
        METADATA_JSONL="${METADATA_JSONL}" \
        OUTPUT_DIR="${output_dir}" \
        EXPERIMENT_NAME="${exp_name}" \
        RANK="${rank}" \
        LR="${LR}" \
        MAX_STEPS="${MAX_STEPS}" \
        CKPT_STEPS="${CKPT_STEPS}" \
        LR_WARMUP="${LR_WARMUP}" \
        MIXED_PRECISION="${MIXED_PRECISION}" \
        RESOLUTION="${RESOLUTION}" \
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
        GRAD_ACCUM="${GRAD_ACCUM}" \
        SEED="${SEED}" \
        WITH_PRIOR_PRESERVATION=1 \
        CLASS_DATA_DIR="${CLASS_DATA_DIR}" \
        NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES}" \
        PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT}" \
        SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE}" \
        PRIOR_GENERATION_PRECISION="${PRIOR_GENERATION_PRECISION}" \
        "${RUN_TRAIN}"

    GPU_ID="${GPU_ID}" \
    METADATA_JSONL="${METADATA_JSONL}" \
    OUTPUT_DIR="${output_dir}" \
    EXPERIMENT_NAME="${exp_name}" \
    RANK="${rank}" \
    LR="${LR}" \
    MAX_STEPS="${MAX_STEPS}" \
    CKPT_STEPS="${CKPT_STEPS}" \
    LR_WARMUP="${LR_WARMUP}" \
    MIXED_PRECISION="${MIXED_PRECISION}" \
    RESOLUTION="${RESOLUTION}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    SEED="${SEED}" \
    WITH_PRIOR_PRESERVATION=1 \
    CLASS_DATA_DIR="${CLASS_DATA_DIR}" \
    NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES}" \
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT}" \
    SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE}" \
    PRIOR_GENERATION_PRECISION="${PRIOR_GENERATION_PRECISION}" \
    "${RUN_TRAIN}"
}

generate_rank_checkpoints() {
    local rank="$1"
    local exp_name="prior_class_data_${MIXED_PRECISION}_rank${rank}_lr${LR}_steps${MAX_STEPS}"
    local exp_dir="${OUTPUT_ROOT}/${exp_name}"
    local step

    for step in ${GENERATE_CHECKPOINT_STEPS}; do
        local lora_dir="${exp_dir}/checkpoint-${step}"
        local out_dir="${GENERATION_OUTPUT_ROOT}/${exp_name}/checkpoint-${step}"

        if [ ! -f "${lora_dir}/pytorch_lora_weights.safetensors" ]; then
            echo "[WARN] Missing checkpoint LoRA, skip generation: ${lora_dir}"
            continue
        fi

        if [ "${SKIP_GENERATION}" = "1" ] && [ -f "${out_dir}/generation_config.json" ]; then
            echo "[*] Generation config already exists, skipping: ${out_dir}"
            continue
        fi

        echo "================================================================"
        echo "  Generate rank : ${rank}"
        echo "  Checkpoint    : ${step}"
        echo "  Output        : ${out_dir}"
        echo "================================================================"

        record_command "${PYTHON_BIN}" generate.py \
            --lora_dir "${lora_dir}" \
            --output_dir "${out_dir}" \
            --prompts_file "${PROMPTS_FILE}" \
            --max_prompts "${MAX_PROMPTS}" \
            --n_per_prompt "${N_PER_PROMPT}"

        "${PYTHON_BIN}" generate.py \
            --lora_dir "${lora_dir}" \
            --output_dir "${out_dir}" \
            --prompts_file "${PROMPTS_FILE}" \
            --max_prompts "${MAX_PROMPTS}" \
            --n_per_prompt "${N_PER_PROMPT}"
    done
}

main() {
    write_record_header

    {
        echo "Run log   : ${RUN_LOG}"
        echo "Record    : ${RECORD_FILE}"
        echo "Generated : ${GENERATION_OUTPUT_ROOT}"
        echo

        local rank
        for rank in ${RANKS}; do
            train_rank "${rank}"
            generate_rank_checkpoints "${rank}"
        done

        echo
        echo "All rank sweep jobs completed."
        echo "Experiments : ${OUTPUT_ROOT}/prior_class_data_${MIXED_PRECISION}_rank*_lr${LR}_steps${MAX_STEPS}"
        echo "Generated   : ${GENERATION_OUTPUT_ROOT}"
        echo "Record      : ${RECORD_FILE}"
        echo "Log         : ${RUN_LOG}"
    } 2>&1 | tee "${RUN_LOG}"
}

main "$@"