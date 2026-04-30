"""Phase 3 – Image Generation
Generates images for the base SD3.5 model and/or any LoRA weights found under
the experiments directory using the same seed IDs for every prompt.

Default output examples:
    generated_images/setting_a/img_{prompt:02d}_{seed:03d}.jpg
    generated_images/experiments/<experiment_name>/img_{prompt:02d}_{seed:03d}.jpg
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
EXPERIMENTS_DIR = Path("experiments")
OUTPUT_BASE    = Path("generated_images")
PROMPTS_FILE   = "prompts.json"

N_PER_PROMPT   = 30      # seeds 0 … N_PER_PROMPT-1 per prompt → 300 total
INFERENCE_STEPS = 28
GUIDANCE_SCALE  = 7.0
HEIGHT = WIDTH  = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompts(
    prompts_file: str,
    prompt_suffix: str = "",
    max_prompts: int | None = None,
) -> tuple[list[str], list[str], str | None]:
    with open(prompts_file) as f:
        data = json.load(f)
    raw_base = data.get("base_prompts") or data["test_prompts"]
    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("--max_prompts must be at least 1")
        raw_base = raw_base[:max_prompts]
    suffix = prompt_suffix.strip()
    base = [f"{prompt}, {suffix}" if suffix else prompt for prompt in raw_base]
    trigger = str(data["trigger_token"]).strip().rstrip(",")
    lora = [f"{trigger}, {prompt}" if trigger else prompt for prompt in base]
    negative_prompt = str(data.get("negative_prompt", "")).strip() or None
    return base, lora, negative_prompt


def discover_lora_dirs(
    root_dir: Path,
    weights: str = "all",
    checkpoint_steps: list[int] | None = None,
) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"LoRA root directory not found: {root_dir}")

    candidates = []
    checkpoint_names = {
        f"checkpoint-{step}" for step in checkpoint_steps or []
    }
    for path in root_dir.rglob("pytorch_lora_weights.safetensors"):
        lora_dir = path.parent
        is_checkpoint = lora_dir.name.startswith("checkpoint-")

        if weights == "final" and is_checkpoint:
            continue
        if weights == "checkpoint-600" and lora_dir.name != "checkpoint-600":
            continue
        if weights == "final-and-checkpoint-600" and (
            is_checkpoint and lora_dir.name != "checkpoint-600"
        ):
            continue
        if weights == "checkpoints" and lora_dir.name not in checkpoint_names:
            continue
        if weights == "final-and-checkpoints" and (
            is_checkpoint and lora_dir.name not in checkpoint_names
        ):
            continue

        candidates.append(lora_dir)

    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA weight directories found under: {root_dir}"
        )
    return candidates


def load_pipeline(
    lora_dir: Path | None = None,
    lora_scale: float = 1.0,
) -> StableDiffusion3Pipeline:
    print(f"  Loading SD3.5 Medium (LoRA={'yes' if lora_dir else 'no'}) …")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir), adapter_name="rockfall")
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["rockfall"], adapter_weights=[lora_scale])
        elif lora_scale != 1.0:
            raise RuntimeError("This diffusers version cannot set LoRA adapter weights")
        print(f"  LoRA weights loaded from '{lora_dir}'")
        print(f"  LoRA scale set to {lora_scale}")

    return pipe


def write_generation_metadata(
    out_dir: Path,
    prompts: list[str],
    negative_prompt: str | None,
    n_per_prompt: int,
    height: int,
    width: int,
    lora_dir: Path | None,
    lora_scale: float,
    label: str,
    prompts_file: str,
    prompt_suffix: str,
) -> None:
    metadata = {
        "label": label,
        "model_id": MODEL_ID,
        "lora_dir": None if lora_dir is None else str(lora_dir),
        "lora_scale": lora_scale,
        "prompts_file": prompts_file,
        "prompt_suffix": prompt_suffix,
        "negative_prompt": negative_prompt,
        "n_per_prompt": n_per_prompt,
        "seed_policy": "For each prompt, generate seeds 0..n_per_prompt-1",
        "num_inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "height": height,
        "width": width,
        "prompts": prompts,
    }
    with (out_dir / "generation_config.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def generate_setting(
    prompts: list[str],
    out_dir: Path,
    n_per_prompt: int,
    negative_prompt: str | None = None,
    height: int = HEIGHT,
    width: int = WIDTH,
    lora_dir: Path | None = None,
    label: str | None = None,
    lora_scale: float = 1.0,
    prompts_file: str = PROMPTS_FILE,
    prompt_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many images already exist (resume support)
    existing = {p.stem for p in out_dir.glob("*.jpg")}
    total = len(prompts) * n_per_prompt
    setting_label = label or ("LoRA fine-tuned" if lora_dir is not None else "base model")
    print(f"\n{'='*60}")
    print(f"  Setting: {setting_label}")
    print(f"  Output : {out_dir}")
    print(f"  Target : {total} images  ({len(prompts)} prompts × {n_per_prompt} seeds)")
    already_done = len(existing)
    if already_done:
        print(f"  Resume : {already_done} images already present, skipping.")
    print(f"{'='*60}")
    write_generation_metadata(
        out_dir=out_dir,
        prompts=prompts,
        negative_prompt=negative_prompt,
        n_per_prompt=n_per_prompt,
        height=height,
        width=width,
        lora_dir=lora_dir,
        lora_scale=lora_scale,
        label=setting_label,
        prompts_file=prompts_file,
        prompt_suffix=prompt_suffix,
    )

    pipe = load_pipeline(lora_dir=lora_dir, lora_scale=lora_scale)
    t0   = time.time()
    done = 0

    for p_idx, prompt in enumerate(prompts):
        for seed in range(n_per_prompt):
            stem = f"img_{p_idx:02d}_{seed:03d}"
            if stem in existing:
                done += 1
                continue

            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=height,
                width=width,
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
    ap = argparse.ArgumentParser(description="Generate images for base and/or LoRA settings")
    ap.add_argument(
        "--setting",
        choices=["a", "b", "both", "experiments", "all"],
        default="all",
        help=(
            "Which setting(s) to generate: base only (a), default lora_output (b), "
            "both, all experiment LoRAs, or all of the above (default: all)"
        ),
    )
    ap.add_argument(
        "--prompts_file",
        default=PROMPTS_FILE,
        help=f"Path to prompts JSON (default: {PROMPTS_FILE})",
    )
    ap.add_argument(
        "--output_base",
        type=Path,
        default=OUTPUT_BASE,
        help=f"Base directory for generated images (default: {OUTPUT_BASE})",
    )
    ap.add_argument(
        "--n_per_prompt",
        type=int,
        default=N_PER_PROMPT,
        help=f"Images to generate per prompt, using seed IDs 0..N-1 (default: {N_PER_PROMPT})",
    )
    ap.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Use only the first N prompts from the prompts JSON",
    )
    ap.add_argument(
        "--experiments_dir",
        type=Path,
        default=EXPERIMENTS_DIR,
        help=f"Directory containing experiment outputs (default: {EXPERIMENTS_DIR})",
    )
    ap.add_argument(
        "--experiment_weights",
        choices=[
            "all",
            "final",
            "checkpoint-600",
            "final-and-checkpoint-600",
            "checkpoints",
            "final-and-checkpoints",
        ],
        default="all",
        help="Which experiment LoRA weights to generate (default: all)",
    )
    ap.add_argument(
        "--checkpoint_steps",
        type=int,
        nargs="*",
        default=[],
        help="Checkpoint steps used with --experiment_weights checkpoints/final-and-checkpoints",
    )
    ap.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="Generate exactly one LoRA setting from this directory",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory used with --lora_dir",
    )
    ap.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA adapter scale/weight for LoRA generation (default: 1.0)",
    )
    ap.add_argument(
        "--prompt_suffix",
        default="",
        help="Text appended to every base prompt before adding the trigger token",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=HEIGHT,
        help=f"Image height (default: {HEIGHT})",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=WIDTH,
        help=f"Image width (default: {WIDTH})",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_prompts, lora_prompts, negative_prompt = build_prompts(
        args.prompts_file,
        args.prompt_suffix,
        args.max_prompts,
    )
    output_base = args.output_base

    if args.lora_dir is not None:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when --lora_dir is used")
        generate_setting(
            prompts=lora_prompts,
            out_dir=args.output_dir,
            n_per_prompt=args.n_per_prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            lora_dir=args.lora_dir,
            label=f"LoRA  ({args.lora_dir}) scale={args.lora_scale}",
            lora_scale=args.lora_scale,
            prompts_file=args.prompts_file,
            prompt_suffix=args.prompt_suffix,
        )
        print("\nAll settings generated successfully.")
        return

    if args.setting in ("a", "both", "all"):
        generate_setting(
            prompts=base_prompts,
            out_dir=output_base / "setting_a",
            n_per_prompt=args.n_per_prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            label="A  (base model)",
            prompts_file=args.prompts_file,
            prompt_suffix=args.prompt_suffix,
        )

    if args.setting in ("b", "both", "all"):
        generate_setting(
            prompts=lora_prompts,
            out_dir=output_base / "setting_b",
            n_per_prompt=args.n_per_prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            lora_dir=Path(LORA_DIR),
            label="B  (default LoRA)",
            lora_scale=args.lora_scale,
            prompts_file=args.prompts_file,
            prompt_suffix=args.prompt_suffix,
        )

    if args.setting in ("experiments", "all"):
        lora_dirs = discover_lora_dirs(
            args.experiments_dir,
            args.experiment_weights,
            args.checkpoint_steps,
        )
        for lora_dir in lora_dirs:
            relative_name = lora_dir.relative_to(args.experiments_dir)
            generate_setting(
                prompts=lora_prompts,
                out_dir=output_base / "experiments" / relative_name,
                n_per_prompt=args.n_per_prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                lora_dir=lora_dir,
                label=f"Experiment LoRA  ({relative_name})",
                lora_scale=args.lora_scale,
                prompts_file=args.prompts_file,
                prompt_suffix=args.prompt_suffix,
            )

    print("\nAll settings generated successfully.")


if __name__ == "__main__":
    main()
