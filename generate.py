"""Phase 3 – Dual-Setting Image Generation
Generates 300 images per setting (10 prompts × 30 seeds):

  Setting A: SD3.5 Medium base model, no LoRA
  Setting B: SD3.5 Medium + LoRA fine-tuned weights

Output:
  generated_images/setting_a/img_{prompt:02d}_{seed:03d}.jpg   (300 files)
  generated_images/setting_b/img_{prompt:02d}_{seed:03d}.jpg   (300 files)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID       = "stabilityai/stable-diffusion-3.5-medium"
LORA_DIR       = "lora_output"
OUTPUT_BASE    = Path("generated_images")
PROMPTS_FILE   = "prompts.json"

N_PER_PROMPT   = 30      # seeds 0 … N_PER_PROMPT-1 per prompt → 300 total
INFERENCE_STEPS = 28
GUIDANCE_SCALE  = 7.0
HEIGHT = WIDTH  = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompts(prompts_file: str) -> tuple[list[str], list[str]]:
    with open(prompts_file) as f:
        data = json.load(f)
    base    = data["base_prompts"]
    trigger = data["trigger_token"]
    lora    = [f"{trigger}, {p}" for p in base]
    return base, lora


def load_pipeline(use_lora: bool) -> StableDiffusion3Pipeline:
    print(f"  Loading SD3.5 Medium (LoRA={'yes' if use_lora else 'no'}) …")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    if use_lora:
        pipe.load_lora_weights(LORA_DIR)
        print(f"  LoRA weights loaded from '{LORA_DIR}'")

    return pipe


def generate_setting(
    prompts: list[str],
    out_dir: Path,
    use_lora: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many images already exist (resume support)
    existing = {p.stem for p in out_dir.glob("*.jpg")}
    total = len(prompts) * N_PER_PROMPT
    print(f"\n{'='*60}")
    print(f"  Setting: {'B  (LoRA fine-tuned)' if use_lora else 'A  (base model)'}")
    print(f"  Output : {out_dir}")
    print(f"  Target : {total} images  ({len(prompts)} prompts × {N_PER_PROMPT} seeds)")
    already_done = len(existing)
    if already_done:
        print(f"  Resume : {already_done} images already present, skipping.")
    print(f"{'='*60}")

    pipe = load_pipeline(use_lora)
    t0   = time.time()
    done = 0

    for p_idx, prompt in enumerate(prompts):
        for seed in range(N_PER_PROMPT):
            stem = f"img_{p_idx:02d}_{seed:03d}"
            if stem in existing:
                done += 1
                continue

            generator = torch.Generator("cuda").manual_seed(p_idx * N_PER_PROMPT + seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=HEIGHT,
                width=WIDTH,
            ).images[0]

            fname = out_dir / f"{stem}.jpg"
            image.save(fname, "JPEG", quality=95)
            done += 1

            elapsed = time.time() - t0
            per_img = elapsed / max(done - already_done, 1)
            remaining = (total - done) * per_img
            print(
                f"  [{done:3d}/{total}] {fname.name}  "
                f"ETA {remaining/60:.1f} min",
                flush=True,
            )

    # Clean up VRAM before the next setting
    del pipe
    torch.cuda.empty_cache()
    print(f"\n  Done! {done} images in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate images for both settings")
    ap.add_argument(
        "--setting",
        choices=["a", "b", "both"],
        default="both",
        help="Which setting(s) to generate (default: both)",
    )
    ap.add_argument(
        "--prompts_file",
        default=PROMPTS_FILE,
        help=f"Path to prompts JSON (default: {PROMPTS_FILE})",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_prompts, lora_prompts = build_prompts(args.prompts_file)

    if args.setting in ("a", "both"):
        generate_setting(
            prompts=base_prompts,
            out_dir=OUTPUT_BASE / "setting_a",
            use_lora=False,
        )

    if args.setting in ("b", "both"):
        generate_setting(
            prompts=lora_prompts,
            out_dir=OUTPUT_BASE / "setting_b",
            use_lora=True,
        )

    print("\nAll settings generated successfully.")


if __name__ == "__main__":
    main()
