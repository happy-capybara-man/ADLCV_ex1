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
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
INSTANCE_DATA_DIR="${INSTANCE_DATA_DIR:-training_data}"
OUTPUT_DIR="${OUTPUT_DIR:-lora_output}"
LOGGING_DIR="${LOGGING_DIR:-logs}"
TRIGGER_TOKEN="${TRIGGER_TOKEN:-zwxrockfall}"
METADATA_JSONL="${METADATA_JSONL:-}"
SKIP_INTERMEDIATE_VALIDATION="${SKIP_INTERMEDIATE_VALIDATION:-1}"
VALIDATION_LOSS_STEPS="${VALIDATION_LOSS_STEPS:-10}"
VALIDATION_LOSS_NUM_BATCHES="${VALIDATION_LOSS_NUM_BATCHES:-2}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-20}"

INSTANCE_PROMPT="${INSTANCE_PROMPT:-${TRIGGER_TOKEN}, road blocked by rockfall debris, real photograph, outdoor}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-${TRIGGER_TOKEN}, large boulder blocking mountain highway, real photograph}"
VALIDATION_INFER_STEPS="${VALIDATION_INFER_STEPS:-20}"

RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"            # effective batch = 4
LR="${LR:-1e-4}"
LR_WARMUP="${LR_WARMUP:-100}"
MAX_STEPS="${MAX_STEPS:-600}"
RANK="${RANK:-16}"                       # LoRA rank; 16 is a good balance for ~90 images
CKPT_STEPS="${CKPT_STEPS:-100}"           # save every 100 steps
SEED="${SEED:-42}"

if [ "${MIN_DATA}" = "1" ]; then
    INSTANCE_DATA_DIR="training_data_min"
    OUTPUT_DIR="lora_output_min"
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
echo "  Trigger: ${TRIGGER_TOKEN}"
echo "  Steps  : ${MAX_STEPS}  (LR=${LR}, Rank=${RANK}, CKPT=${CKPT_STEPS})"
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
    "${VALIDATION_ARGS[@]}" \
    --logging_dir="${LOGGING_DIR}" \
    --report_to="tensorboard"

echo "================================================================"
echo "  Training complete!  LoRA weights → ${OUTPUT_DIR}/"
echo "  To view TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/${LOGGING_DIR}"
echo "================================================================"
