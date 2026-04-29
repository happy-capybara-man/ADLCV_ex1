"""Evaluate one generated image folder with CLIP Score and KID.

Default target:
    generated_images/controllable_rank16_lr1e-4_steps300_final

If the target folder has no images but has an experiments/ subfolder,
the script will evaluate that experiments/ folder automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch_fidelity import calculate_metrics
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_REAL_DIR = Path("training_data")
DEFAULT_GEN_DIR = Path("generated_images/controllable_rank16_lr1e-4_steps300_final")
DEFAULT_PROMPTS_FILE = Path("prompts.json")
DEFAULT_OUTPUT = Path("eval_results/controllable_rank16_lr1e-4_steps300_final_results.json")

CLIP_MODEL   = "openai/clip-vit-large-patch14"
CLIP_BATCH   = 16          # images per CLIP forward pass
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_images(directory: Path) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one folder with CLIP Score and KID")
    parser.add_argument("--generated_dir", type=Path, default=DEFAULT_GEN_DIR)
    parser.add_argument("--real_dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--prompts_file", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip_batch", type=int, default=CLIP_BATCH)
    parser.add_argument("--prompt_mode", choices=["base", "trigger", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_generated_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Generated directory not found: {path}")

    direct_images = list_images(path)
    if direct_images:
        return path

    experiments_dir = path / "experiments"
    if experiments_dir.exists() and list_images(experiments_dir):
        return experiments_dir

    raise ValueError(
        "No images found in generated_dir or generated_dir/experiments. "
        f"Target: {path}"
    )


def load_uint8_tensor(path: Path) -> torch.Tensor:
    """Return C×H×W uint8 tensor (no resizing – CLIPScore handles it internally)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)          # H×W×C
    return torch.from_numpy(arr).permute(2, 0, 1)  # C×H×W


def compute_clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    prompts: list[str],
    device: str,
    batch_size: int,
) -> float:
    """Compute mean CLIP Score between each image and its corresponding prompt."""
    assert len(image_paths) == len(prompts), (
        f"Image/prompt count mismatch: {len(image_paths)} vs {len(prompts)}"
    )
    total_score = 0.0
    total_count = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths   = image_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            vision_output = model.vision_model(pixel_values=inputs["pixel_values"])
            text_output = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = model.visual_projection(vision_output.pooler_output)
            text_features = model.text_projection(text_output.pooler_output)

            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            batch_scores = 100 * (image_features * text_features).sum(dim=-1).clamp(min=0)

            total_score += float(batch_scores.sum().item())
            total_count += int(batch_scores.numel())

        if (i // batch_size) % 5 == 0:
            pct = 100 * (i + len(batch_paths)) / len(image_paths)
            print(f"    CLIP [{pct:5.1f}%] processed {i + len(batch_paths)}/{len(image_paths)}")

    return total_score / total_count


def compute_kid(real_dir: Path, fake_dir: Path, use_cuda: bool) -> dict[str, float]:
    """Compute KID between real_dir and fake_dir using torch_fidelity."""
    n_real = len(list_images(real_dir))
    n_fake = len(list_images(fake_dir))
    if n_real == 0:
        raise ValueError(f"No real images found in {real_dir}")
    if n_fake == 0:
        raise ValueError(f"No generated images found in {fake_dir}")

    # kid_subset_size must be ≤ min(n_real, n_fake)
    subset  = min(n_real, n_fake, 100)

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=use_cuda,
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


def build_prompt_list(prompts_file: Path) -> tuple[list[str], list[str]]:
    with prompts_file.open() as f:
        data = json.load(f)
    base    = data["base_prompts"]
    trigger = str(data["trigger_token"]).strip().rstrip(", ")
    lora    = [f"{trigger}, {p}" if trigger else p for p in base]
    return base, lora


def build_ordered_prompts(image_paths: list[Path], prompt_bank: list[str]) -> list[str]:
    ordered_prompts: list[str] = []
    for path in image_paths:
        parts = path.stem.split("_")
        try:
            prompt_index = int(parts[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unexpected filename pattern: {path.name}") from exc

        if prompt_index < 0 or prompt_index >= len(prompt_bank):
            raise IndexError(
                f"Prompt index {prompt_index} out of range for {path.name}; "
                f"prompt count={len(prompt_bank)}"
            )
        ordered_prompts.append(prompt_bank[prompt_index])
    return ordered_prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    generated_dir = resolve_generated_dir(args.generated_dir)
    image_paths = list_images(generated_dir)
    n_images = len(image_paths)

    print(f"\n{'='*60}")
    print(f"Evaluating generated directory: {generated_dir}")
    print(f"Device: {device}")
    print(f"Found images: {n_images}")
    print(f"{'='*60}")

    base_prompts, trigger_prompts = build_prompt_list(args.prompts_file)
    print(f"Loading CLIP model: {CLIP_MODEL}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    results: dict[str, object] = {
        "generated_dir": str(generated_dir),
        "real_dir": str(args.real_dir),
        "prompts_file": str(args.prompts_file),
        "n_images": n_images,
        "device": device,
    }

    if args.prompt_mode in {"base", "both"}:
        print("\n[1/3] CLIP Score (base prompts)")
        ordered_base = build_ordered_prompts(image_paths, base_prompts)
        clip_base = compute_clip_score(
            model=model,
            processor=processor,
            image_paths=image_paths,
            prompts=ordered_base,
            device=device,
            batch_size=args.clip_batch,
        )
        print(f"  CLIP base    = {clip_base:.4f}")
        results["clip_score_base_prompt"] = clip_base

    if args.prompt_mode in {"trigger", "both"}:
        print("\n[2/3] CLIP Score (trigger prompts)")
        ordered_trigger = build_ordered_prompts(image_paths, trigger_prompts)
        clip_trigger = compute_clip_score(
            model=model,
            processor=processor,
            image_paths=image_paths,
            prompts=ordered_trigger,
            device=device,
            batch_size=args.clip_batch,
        )
        print(f"  CLIP trigger = {clip_trigger:.4f}")
        results["clip_score_trigger_prompt"] = clip_trigger

    print("\n[3/3] KID")
    kid = compute_kid(
        real_dir=args.real_dir,
        fake_dir=generated_dir,
        use_cuda=(device == "cuda"),
    )
    print(f"  KID mean     = {kid['kid_mean']:.6f}")
    print(f"  KID std      = {kid['kid_std']:.6f}")
    results.update(kid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
