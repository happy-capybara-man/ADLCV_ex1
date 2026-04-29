#!/usr/bin/env bash
# =============================================================================
# Generate prompt/LoRA-weight ablation images for the 600-step controllable LoRA.
#
# Each setting writes:
#   - Generated images: generated_images/controllable_rank16_lr1e-4_steps600_prompt_weight_experiments/<setting>/
#   - Per-setting config: generation_config.json inside each output folder
#   - Console log: experiment_logs/<timestamp>_controllable_rank16_lr1e-4_steps600_prompt_weight_experiments.log
#   - Markdown run record: experiment_logs/controllable_rank16_lr1e-4_steps600_prompt_weight_experiments.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
LORA_DIR="${LORA_DIR:-experiments/controllable_rank16_lr1e-4_steps600}"
PROMPTS_FILE="${PROMPTS_FILE:-prompts.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated_images/controllable_rank16_lr1e-4_steps600_prompt_weight_experiments}"
LOG_ROOT="${LOG_ROOT:-experiment_logs}"
N_PER_PROMPT="${N_PER_PROMPT:-3}"

ORDER_SUFFIX="clear and straight yellow double center lines, pristine intact asphalt surface outside the debris zone, perfectly straight road perspective, highly structured highway infrastructure"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_ROOT}/${TIMESTAMP}_controllable_rank16_lr1e-4_steps600_prompt_weight_experiments.log"
RECORD_FILE="${LOG_ROOT}/controllable_rank16_lr1e-4_steps600_prompt_weight_experiments.md"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python not executable: ${PYTHON_BIN}"
    exit 1
fi

if [ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]; then
    echo "[ERROR] Final LoRA weights not found: ${LORA_DIR}/pytorch_lora_weights.safetensors"
    exit 1
fi

write_record_header() {
    cat > "${RECORD_FILE}" <<EOF
# controllable_rank16_lr1e-4_steps600 prompt/weight experiments

Start time: ${TIMESTAMP}

## Fixed settings

- LoRA path: ${LORA_DIR}
- prompts file: ${PROMPTS_FILE}
- output root: ${OUTPUT_ROOT}
- images per prompt: ${N_PER_PROMPT}
- seed policy: seeds 0..$((N_PER_PROMPT - 1)) for every prompt in every setting
- base model: stabilityai/stable-diffusion-3.5-medium
- inference steps: 28
- guidance scale: 7.0
- resolution: 1024x1024

## Settings

| setting | prompt suffix | LoRA weight | output dir |
| --- | --- | --- | --- |
| order_only_weight1.0 | order | 1.0 | ${OUTPUT_ROOT}/order_only_weight1.0 |
| order_only_weight0.8 | order | 0.8 | ${OUTPUT_ROOT}/order_only_weight0.8 |

## Commands

EOF
}

run_setting() {
    local name="$1"
    local lora_scale="$2"
    local suffix="$3"
    local out_dir="${OUTPUT_ROOT}/${name}"

    echo "================================================================"
    echo "Setting    : ${name}"
    echo "LoRA scale : ${lora_scale}"
    echo "Output     : ${out_dir}"
    echo "================================================================"

    {
        echo "### ${name}"
        echo
        echo '```bash'
        printf '%q ' "${PYTHON_BIN}" generate.py \
            --lora_dir "${LORA_DIR}" \
            --output_dir "${out_dir}" \
            --prompts_file "${PROMPTS_FILE}" \
            --n_per_prompt "${N_PER_PROMPT}" \
            --lora_scale "${lora_scale}" \
            --prompt_suffix "${suffix}"
        echo
        echo '```'
        echo
    } >> "${RECORD_FILE}"

    "${PYTHON_BIN}" generate.py \
        --lora_dir "${LORA_DIR}" \
        --output_dir "${out_dir}" \
        --prompts_file "${PROMPTS_FILE}" \
        --n_per_prompt "${N_PER_PROMPT}" \
        --lora_scale "${lora_scale}" \
        --prompt_suffix "${suffix}"
}

main() {
    write_record_header

    {
        echo "Run log: ${RUN_LOG}"
        echo "Record : ${RECORD_FILE}"
        echo "Output : ${OUTPUT_ROOT}"
        echo

        run_setting "order_only_weight1.0" "1.0" "${ORDER_SUFFIX}"
        run_setting "order_only_weight0.8" "0.8" "${ORDER_SUFFIX}"

        echo
        echo "All prompt/weight generation settings completed."
        echo "Images : ${OUTPUT_ROOT}"
        echo "Record : ${RECORD_FILE}"
        echo "Log    : ${RUN_LOG}"
    } 2>&1 | tee "${RUN_LOG}"
}

main "$@"
