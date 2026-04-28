"""Phase 4 – Evaluation: CLIP Score + KID

Computes for each setting:
  • CLIP Score  (torchmetrics, openai/clip-vit-large-patch14)
  • KID         (torch_fidelity, real images = training_data/)

Output:
  eval_results/results.json   ← machine-readable summary
  Console                     ← comparison table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch_fidelity import calculate_metrics
from torchmetrics.multimodal.clip_score import CLIPScore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REAL_DIR     = Path("training_data")
GEN_BASE     = Path("generated_images")
RESULTS_DIR  = Path("eval_results")
PROMPTS_FILE = "prompts.json"

CLIP_MODEL   = "openai/clip-vit-large-patch14"
CLIP_BATCH   = 16          # images per CLIP forward pass
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}

N_PER_PROMPT = 30          # must match generate.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_images(directory: Path) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    return paths


def load_uint8_tensor(path: Path) -> torch.Tensor:
    """Return C×H×W uint8 tensor (no resizing – CLIPScore handles it internally)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)          # H×W×C
    return torch.from_numpy(arr).permute(2, 0, 1)  # C×H×W


def compute_clip_score(
    image_paths: list[Path],
    prompts: list[str],
    device: str = "cuda",
) -> float:
    """Compute mean CLIP Score between each image and its corresponding prompt."""
    assert len(image_paths) == len(prompts), (
        f"Image/prompt count mismatch: {len(image_paths)} vs {len(prompts)}"
    )
    metric = CLIPScore(model_name_or_path=CLIP_MODEL).to(device)
    metric.eval()

    for i in range(0, len(image_paths), CLIP_BATCH):
        batch_paths   = image_paths[i : i + CLIP_BATCH]
        batch_prompts = prompts[i : i + CLIP_BATCH]
        batch_imgs    = torch.stack([load_uint8_tensor(p) for p in batch_paths]).to(device)
        with torch.no_grad():
            metric.update(batch_imgs, batch_prompts)

        if (i // CLIP_BATCH) % 5 == 0:
            pct = 100 * (i + len(batch_paths)) / len(image_paths)
            print(f"    CLIP [{pct:5.1f}%] processed {i + len(batch_paths)}/{len(image_paths)}")

    score = metric.compute().item()
    return float(score)


def compute_kid(real_dir: Path, fake_dir: Path) -> dict[str, float]:
    """Compute KID between real_dir and fake_dir using torch_fidelity."""
    n_real = len(list_images(real_dir))
    n_fake = len(list_images(fake_dir))
    # kid_subset_size must be ≤ min(n_real, n_fake)
    subset  = min(n_real, n_fake, 100)

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=True,
        kid=True,
        kid_subset_size=subset,
        fid=False,
        isc=False,
        verbose=False,
    )
    return {
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std":  float(metrics["kernel_inception_distance_std"]),
    }


def build_prompt_list(prompts_file: str) -> tuple[list[str], list[str]]:
    with open(prompts_file) as f:
        data = json.load(f)
    base    = data["base_prompts"]
    trigger = data["trigger_token"]
    lora    = [f"{trigger}, {p}" for p in base]
    return base, lora


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    base_prompts, lora_prompts = build_prompt_list(PROMPTS_FILE)

    settings = [
        ("setting_a", GEN_BASE / "setting_a", base_prompts),
        ("setting_b", GEN_BASE / "setting_b", lora_prompts),
    ]

    results: dict = {}

    for name, gen_dir, prompts in settings:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}  ({gen_dir})")
        print(f"{'='*60}")

        image_paths = list_images(gen_dir)
        n = len(image_paths)
        print(f"  Found {n} generated images")

        if n == 0:
            print("  [SKIP] No images found – run generate.py first.")
            results[name] = {"error": "no images"}
            continue

        # Build prompt list matching the generation order:
        #   img_{p_idx:02d}_{seed:03d}.jpg  → prompt index p_idx
        ordered_prompts: list[str] = []
        for p in image_paths:
            # filename pattern: img_<p_idx>_<seed>.jpg
            parts = p.stem.split("_")
            try:
                p_idx = int(parts[1])
            except (IndexError, ValueError):
                p_idx = 0
            ordered_prompts.append(prompts[p_idx])

        # -- CLIP Score --
        print("\n  [1/2] CLIP Score …")
        clip_score = compute_clip_score(image_paths, ordered_prompts)
        print(f"        CLIP Score = {clip_score:.4f}")

        # -- KID --
        print("\n  [2/2] KID …")
        kid = compute_kid(REAL_DIR, gen_dir)
        print(f"        KID mean   = {kid['kid_mean']:.6f}")
        print(f"        KID std    = {kid['kid_std']:.6f}")

        results[name] = {
            "n_images":   n,
            "clip_score": clip_score,
            **kid,
        }

    # Save JSON
    out_file = RESULTS_DIR / "results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_file}")

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    if "setting_a" in results and "setting_b" in results:
        a = results["setting_a"]
        b = results["setting_b"]
        print("\n" + "="*64)
        print(f"  {'Metric':<24} {'Setting A (base)':<20} {'Setting B (LoRA)':<20}")
        print("  " + "-"*62)
        for key, fmt in [("clip_score", ".4f"), ("kid_mean", ".6f"), ("kid_std", ".6f")]:
            if key in a and key in b:
                av = format(a[key], fmt)
                bv = format(b[key], fmt)
                print(f"  {key:<24} {av:<20} {bv:<20}")
        print("="*64)
        print("  Higher CLIP Score = better prompt alignment")
        print("  Lower  KID        = closer to real image distribution")


if __name__ == "__main__":
    main()
