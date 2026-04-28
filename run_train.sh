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

# Activate uv venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

# Set MIN_DATA=1 to do a quick sanity check on a tiny subset.
# This keeps the full training path unchanged when MIN_DATA is unset or 0.
MIN_DATA="${MIN_DATA:-0}"

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
mixed_precision: fp16
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
MODEL_ID="stabilityai/stable-diffusion-3.5-medium"
INSTANCE_DATA_DIR="training_data"
OUTPUT_DIR="lora_output"
TRIGGER_TOKEN="<road_rockfall_event>"

INSTANCE_PROMPT="${TRIGGER_TOKEN}, road blocked by rockfall debris, real photograph, outdoor"
VALIDATION_PROMPT="${TRIGGER_TOKEN}, large boulder blocking mountain highway, real photograph"
VALIDATION_INFER_STEPS=20

RESOLUTION=1024
TRAIN_BATCH_SIZE=1
GRAD_ACCUM=4            # effective batch = 4
LR=1e-4
LR_WARMUP=100
MAX_STEPS=1000
RANK=16                 # LoRA rank; 16 is a good balance for ~90 images
CKPT_STEPS=200          # save every 200 steps → 5 checkpoints total
SEED=42

if [ "${MIN_DATA}" = "1" ]; then
    INSTANCE_DATA_DIR="training_data_min"
    OUTPUT_DIR="lora_output_min"
    MAX_STEPS=15
    CKPT_STEPS=15
    NUM_VALIDATION_IMAGES=1
    VALIDATION_EPOCHS_OVERRIDE="--validation_epochs=5"
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
    NUM_VALIDATION_IMAGES=2
    VALIDATION_EPOCHS_OVERRIDE=""
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "================================================================"
echo "  Model  : ${MODEL_ID}"
echo "  Data   : ${INSTANCE_DATA_DIR}"
echo "  Output : ${OUTPUT_DIR}"
echo "  Trigger: ${TRIGGER_TOKEN}"
echo "  Steps  : ${MAX_STEPS}  (LR=${LR}, Rank=${RANK})"
echo "================================================================"

# --------------------------------------------------------------------------
# 3. Launch training
# --------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

accelerate launch "${TRAIN_SCRIPT}" \
    --pretrained_model_name_or_path="${MODEL_ID}" \
    --instance_data_dir="${INSTANCE_DATA_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --mixed_precision="fp16" \
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
    --validation_prompt="${VALIDATION_PROMPT}" \
    --num_validation_images=${NUM_VALIDATION_IMAGES} \
    --validation_num_inference_steps=${VALIDATION_INFER_STEPS} \
    --final_validation_cpu_offload \
    --skip_final_validation \
    ${VALIDATION_EPOCHS_OVERRIDE} \
    --logging_dir="logs" \
    --report_to="tensorboard"

echo "================================================================"
echo "  Training complete!  LoRA weights → ${OUTPUT_DIR}/"
echo "  To view TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/logs"
echo "================================================================"
