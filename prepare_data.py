"""Phase 1 – Data Preparation
Center-crop every downloaded image to 1024×1024, convert to JPEG,
and emit a metadata.jsonl file understood by the HuggingFace imagefolder
loader (used later by train_dreambooth_lora_sd3.py via --dataset_name).

Output layout:
  training_data/
    img_0001.jpg
    img_0002.jpg
    ...
    metadata.jsonl         ← {"file_name": "img_0001.jpg", "text": "<trigger>, ..."}
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIGGER_TOKEN = "<road_rockfall_event>"
TARGET_SIZE   = 1024
OUTPUT_DIR    = Path("training_data")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Source directories to scan
SOURCE_DIRS: list[str] = [
    "rock",
    "石頭",
]

# Unified caption applied to every training image
CAPTION = (
    f"{TRIGGER_TOKEN}, large rocks and debris blocking a mountain road, "
    "rockfall event, real photograph, outdoor, natural lighting"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def process_image(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = center_crop_square(img)
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img.save(dst, "JPEG", quality=95)
        return True
    except Exception as exc:
        print(f"  [SKIP] {src.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    metadata: list[dict] = []
    idx = 1

    for src_dir_str in SOURCE_DIRS:
        src_dir = Path(src_dir_str)
        if not src_dir.exists():
            print(f"[WARN] Not found: {src_dir}")
            continue

        images = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        print(f"\n[{src_dir.name[:70]}]")
        print(f"  Found {len(images)} images")

        for src_path in images:
            out_name = f"img_{idx:04d}.jpg"
            out_path = OUTPUT_DIR / out_name

            if process_image(src_path, out_path):
                metadata.append({"file_name": out_name, "text": CAPTION})
                print(f"  [{idx:04d}] {src_path.name} → {out_name}")
                idx += 1

    # Write metadata.jsonl (HuggingFace imagefolder format)
    jsonl_path = Path("metadata.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone.  {len(metadata)} images saved to '{OUTPUT_DIR}/'")
    print(f"metadata.jsonl  →  {jsonl_path}  ({len(metadata)} lines)")


if __name__ == "__main__":
    main()
