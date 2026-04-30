#!/usr/bin/env bash
# =============================================================================
# Phase 2 – SD3.5 Medium DreamBooth LoRA Training
# =============================================================================
# Prerequisites:
#   1. huggingface-cli login  (or HF_TOKEN env var) – SD3.5 is a gated model.
#      Accept the licence at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
#   2. Phase 1 must have run: python prepare_data.py
#   3. Install deps with uv: uv sync
# =============================================================================
set -euo pipefail

# Keep CUDA index mapping consistent with nvidia-smi.
# This avoids machine-specific GPU ordinal reordering.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Activate uv venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

# Run from the repository root so default relative paths resolve consistently.
cd "${SCRIPT_DIR}"

# Set MIN_DATA=1 to do a quick sanity check on a tiny subset.
# This keeps the full training path unchanged when MIN_DATA is unset or 0.
MIN_DATA="${MIN_DATA:-0}"

# Select GPU. Default uses GPU 1 (RTX 3090 on this machine) to avoid 2080 Ti OOM.
GPU_ID="${GPU_ID:-0}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -i "${GPU_ID}" >/dev/null 2>&1; then
        echo "[ERROR] GPU_ID=${GPU_ID} is not available"
        echo "        Check available GPUs: nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
        exit 1
    fi
fi

# Mask visible GPUs so only the selected physical GPU is exposed.
# After masking, accelerate should launch on logical GPU 0.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Safety check: ensure the selected logical GPU has enough VRAM (e.g., RTX 3090).
# Override MIN_GPU_MEMORY_GIB if you intentionally want to run on a smaller GPU.
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-20}"
if ! MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB}" python - <<'PY'
import os
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("[ERROR] CUDA is not available after masking GPU visibility")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
total_gib = props.total_memory / (1024 ** 3)
min_gib = float(os.environ["MIN_GPU_MEMORY_GIB"])

print(f"[*] CUDA logical GPU 0 -> {props.name} ({total_gib:.2f} GiB)")
if total_gib < min_gib:
    print(
        f"[ERROR] Selected GPU has only {total_gib:.2f} GiB VRAM, below MIN_GPU_MEMORY_GIB={min_gib:.2f}"
    )
    print("        This usually means GPU index mapping is wrong.")
    print("        Keep CUDA_DEVICE_ORDER=PCI_BUS_ID and choose GPU_ID from nvidia-smi index.")
    sys.exit(1)
PY
then
    exit 1
fi

# --------------------------------------------------------------------------
# 0. Download official DreamBooth-LoRA SD3 training script (if missing)
# --------------------------------------------------------------------------
TRAIN_SCRIPT="train_dreambooth_lora_sd3.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[*] Downloading ${TRAIN_SCRIPT} from huggingface/diffusers ..."
    wget -q \
      "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sd3.py" \
      -O "${TRAIN_SCRIPT}"
    echo "[*] Download complete."
fi

# Verify key dependencies are available in the current venv.
if ! python - <<'PY' >/dev/null 2>&1
import bitsandbytes  # noqa: F401
import tensorboard  # noqa: F401
import datasets  # noqa: F401
PY
then
    echo "[ERROR] Missing training dependencies in .venv"
    echo "        Please run: uv sync"
    exit 1
fi

# --------------------------------------------------------------------------
# 1. Accelerate single-GPU config (idempotent, non-interactive)
# --------------------------------------------------------------------------
ACCEL_CFG="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
if [ ! -f "${ACCEL_CFG}" ]; then
    echo "[*] Writing accelerate default config ..."
    mkdir -p "$(dirname "${ACCEL_CFG}")"
    cat > "${ACCEL_CFG}" << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# --------------------------------------------------------------------------
# 2. Training hyperparameters
# --------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
INSTANCE_DATA_DIR="${INSTANCE_DATA_DIR:-training_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_ROOT="${LOG_ROOT:-experiment_logs}"
LOGGING_DIR="${LOGGING_DIR:-logs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RUN_LOG_FILE="${RUN_LOG_FILE:-}"
AUTO_TEE_LOG="${AUTO_TEE_LOG:-1}"
TRIGGER_TOKEN="${TRIGGER_TOKEN:-zwxrockfall}"
METADATA_JSONL="${METADATA_JSONL:-metadata.jsonl}"
SKIP_INTERMEDIATE_VALIDATION="${SKIP_INTERMEDIATE_VALIDATION:-1}"
VALIDATION_LOSS_STEPS="${VALIDATION_LOSS_STEPS:-10}"
VALIDATION_LOSS_NUM_BATCHES="${VALIDATION_LOSS_NUM_BATCHES:-2}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-20}"

INSTANCE_PROMPT="${INSTANCE_PROMPT:-${TRIGGER_TOKEN}, road blocked by rockfall debris, real photograph, outdoor}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-${TRIGGER_TOKEN}, large boulder blocking mountain highway, real photograph}"
VALIDATION_INFER_STEPS="${VALIDATION_INFER_STEPS:-20}"

RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"            # effective batch = 4
LR="${LR:-1e-4}"
LR_WARMUP="${LR_WARMUP:-100}"
MAX_STEPS="${MAX_STEPS:-600}"
RANK="${RANK:-16}"
CKPT_STEPS="${CKPT_STEPS:-100}"           # save every 100 steps
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-1}"
CLASS_DATA_DIR="${CLASS_DATA_DIR:-class_data}"
CLASS_PROMPT="${CLASS_PROMPT:-realistic photograph of a rockfall event on an asphalt mountain road, fallen boulders and small rock fragments scattered on the driving lane, visible lane markings, rocky hillside, outdoor daylight}"
CLASS_NEGATIVE_PROMPT="${CLASS_NEGATIVE_PROMPT:-empty road, clean road, no rocks on road, rocks only on roadside, rocks only on hillside, traffic cone, construction cone, people, vehicle, cartoon, anime, painting, 3d render, cgi, blurry, low quality, text, watermark}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
PRIOR_GENERATION_PRECISION="${PRIOR_GENERATION_PRECISION:-bf16}"

if [ "${MIN_DATA}" = "1" ]; then
    INSTANCE_DATA_DIR="training_data_min"
    if [ -z "${OUTPUT_DIR}" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/min_data_smoke_test"
    fi
    MAX_STEPS=15
    CKPT_STEPS=15
    echo "[*] MIN_DATA mode enabled: using a tiny subset and short training run"
    rm -rf "${OUTPUT_DIR}"
    rm -rf "${INSTANCE_DATA_DIR}"
    mkdir -p "${INSTANCE_DATA_DIR}"
    python - <<'PY'
from pathlib import Path
import shutil

src = Path("training_data")
dst = Path("training_data_min")
dst.mkdir(exist_ok=True)

images = sorted(
    p for p in src.glob("*.jpg")
    if p.is_file() and not p.name.startswith("min_")
)[:1]
for path in images:
    shutil.copy2(path, dst / path.name)
print(f"[MIN_DATA] Copied {len(images)} images into {dst}")
PY
else
    :
fi

VALIDATION_ARGS=()
if [ "${MIN_DATA}" = "1" ]; then
    VALIDATION_EPOCHS=5
    VALIDATION_LOSS_STEPS="${VALIDATION_LOSS_STEPS:-5}"
fi

ENABLE_PRIOR=0
case "${WITH_PRIOR_PRESERVATION}" in
    1|true|TRUE|yes|YES)
        ENABLE_PRIOR=1
        ;;
    0|false|FALSE|no|NO)
        ENABLE_PRIOR=0
        ;;
    *)
        echo "[ERROR] WITH_PRIOR_PRESERVATION must be one of: 1/0/true/false/yes/no"
        exit 1
        ;;
esac

PRIOR_ARGS=()
if [ "${ENABLE_PRIOR}" = "1" ]; then
    mkdir -p "${CLASS_DATA_DIR}"
    PRIOR_ARGS=(
        --with_prior_preservation
        --class_data_dir="${CLASS_DATA_DIR}"
        --class_prompt="${CLASS_PROMPT}"
        --class_negative_prompt="${CLASS_NEGATIVE_PROMPT}"
        --num_class_images=${NUM_CLASS_IMAGES}
        --prior_loss_weight=${PRIOR_LOSS_WEIGHT}
        --sample_batch_size=${SAMPLE_BATCH_SIZE}
        --prior_generation_precision="${PRIOR_GENERATION_PRECISION}"
    )
fi

if [ -z "${EXPERIMENT_NAME}" ] && [ -n "${OUTPUT_DIR}" ]; then
    EXPERIMENT_NAME="$(basename "${OUTPUT_DIR}")"
fi

if [ -z "${EXPERIMENT_NAME}" ]; then
    if [ "${ENABLE_PRIOR}" = "1" ]; then
        EXPERIMENT_NAME="prior_$(basename "${CLASS_DATA_DIR}")_${MIXED_PRECISION}_rank${RANK}_lr${LR}_steps${MAX_STEPS}"
    else
        EXPERIMENT_NAME="baseline_${MIXED_PRECISION}_rank${RANK}_lr${LR}_steps${MAX_STEPS}"
    fi
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
if [ -z "${RUN_LOG_FILE}" ]; then
    RUN_LOG_FILE="${LOG_ROOT}/${TIMESTAMP}_${EXPERIMENT_NAME}.log"
fi

VALIDATION_ARGS=(
    --validation_prompt="${VALIDATION_PROMPT}"
    --num_validation_images=1
    --validation_num_inference_steps=${VALIDATION_INFER_STEPS}
    --validation_epochs=${VALIDATION_EPOCHS}
    --validation_loss_steps=${VALIDATION_LOSS_STEPS}
    --validation_loss_num_batches=${VALIDATION_LOSS_NUM_BATCHES}
    --final_validation_cpu_offload
)

if [ "${SKIP_INTERMEDIATE_VALIDATION}" = "1" ]; then
    VALIDATION_ARGS+=(--skip_intermediate_validation)
    echo "[*] Intermediate image validation disabled to reduce VRAM usage"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/${LOGGING_DIR}"
mkdir -p "${LOG_ROOT}"

if [ "${AUTO_TEE_LOG}" = "1" ] && [ -z "${RUN_TRAIN_TEE_ACTIVE:-}" ]; then
    export RUN_TRAIN_TEE_ACTIVE=1
    exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
fi

TRAIN_DATA_ARGS=(--instance_data_dir="${INSTANCE_DATA_DIR}")
if [ -n "${METADATA_JSONL}" ]; then
    if [ ! -f "${METADATA_JSONL}" ]; then
        echo "[ERROR] METADATA_JSONL not found: ${METADATA_JSONL}"
        exit 1
    fi

    if [ "${MIN_DATA}" = "1" ]; then
        MIN_METADATA_SRC="${METADATA_JSONL}" MIN_METADATA_DST_DIR="${INSTANCE_DATA_DIR}" python - <<'PY'
from pathlib import Path
import json
import os

src = Path(os.environ["MIN_METADATA_SRC"])
dst_dir = Path(os.environ["MIN_METADATA_DST_DIR"])
dst = dst_dir / "metadata.jsonl"

existing = {p.name for p in dst_dir.glob("*.jpg") if p.is_file()}
kept = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("file_name") in existing:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

if kept == 0:
    raise SystemExit("[ERROR] No metadata rows matched MIN_DATA images")
print(f"[*] MIN_DATA metadata rows kept: {kept}")
PY
    else
        cp "${METADATA_JSONL}" "${INSTANCE_DATA_DIR}/metadata.jsonl"
    fi

    TRAIN_DATA_ARGS=(
        --dataset_name="${INSTANCE_DATA_DIR}"
        --caption_column="text"
    )
fi

echo "================================================================"
echo "  Model  : ${MODEL_ID}"
echo "  GPU    : physical ${GPU_ID} via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "           accelerate logical device=0 (masked single GPU)"
echo "  Data   : ${INSTANCE_DATA_DIR}"
echo "  Output : ${OUTPUT_DIR}"
echo "  TB Logs: ${OUTPUT_DIR}/${LOGGING_DIR}"
echo "  Log    : ${RUN_LOG_FILE}"
echo "  Trigger: ${TRIGGER_TOKEN}"
echo "  Steps  : ${MAX_STEPS}  (LR=${LR}, Rank=${RANK}, CKPT=${CKPT_STEPS})"
if [ "${ENABLE_PRIOR}" = "1" ]; then
    echo "  Prior  : enabled (class_prompt='${CLASS_PROMPT}', class_images=${NUM_CLASS_IMAGES}, prior_w=${PRIOR_LOSS_WEIGHT})"
    echo "           class_negative_prompt='${CLASS_NEGATIVE_PROMPT}'"
    echo "           class_data_dir=${CLASS_DATA_DIR}"
    echo "           sample_batch_size=${SAMPLE_BATCH_SIZE}, prior_generation_precision=${PRIOR_GENERATION_PRECISION}"
else
    echo "  Prior  : disabled"
fi
echo "  ValLoss: every ${VALIDATION_LOSS_STEPS} steps, ${VALIDATION_LOSS_NUM_BATCHES} batch(es)"
echo "  ValImg : skip_intermediate=${SKIP_INTERMEDIATE_VALIDATION}, final_validation=enabled"
echo "================================================================"

# --------------------------------------------------------------------------
# 3. Launch training
# --------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

accelerate launch "${TRAIN_SCRIPT}" \
    --pretrained_model_name_or_path="${MODEL_ID}" \
    "${TRAIN_DATA_ARGS[@]}" \
    "${PRIOR_ARGS[@]}" \
    --output_dir="${OUTPUT_DIR}" \
    --mixed_precision="${MIXED_PRECISION}" \
    --instance_prompt="${INSTANCE_PROMPT}" \
    --resolution=${RESOLUTION} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --gradient_checkpointing \
    --learning_rate=${LR} \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=${LR_WARMUP} \
    --max_train_steps=${MAX_STEPS} \
    --rank=${RANK} \
    --checkpointing_steps=${CKPT_STEPS} \
    --seed=${SEED} \
    "${VALIDATION_ARGS[@]}" \
    --logging_dir="${LOGGING_DIR}" \
    --report_to="tensorboard"

echo "================================================================"
echo "  Training complete!  LoRA weights → ${OUTPUT_DIR}/"
echo "  To view TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/${LOGGING_DIR}"
echo "  Console log saved to: ${RUN_LOG_FILE}"
echo "================================================================"
