from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-large-patch14"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preview images with CLIP score")
    parser.add_argument("--preview_dir", type=Path, default=Path("generated_images/preview_5"))
    parser.add_argument("--prompts_file", type=Path, default=Path("prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("eval_results/preview_5_clip_scores.json"))
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def load_prompts(prompts_file: Path) -> tuple[list[str], list[str]]:
    data = json.loads(prompts_file.read_text())
    base_prompts = data["base_prompts"]
    trigger_token = data["trigger_token"]
    lora_prompts = [f"{trigger_token}, {prompt}" for prompt in base_prompts]
    return base_prompts, lora_prompts


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def prompt_for_image(path: Path, prompts: list[str]) -> str:
    return prompts[int(path.stem.split("_")[1])]


def discover_output_dirs(preview_dir: Path) -> list[tuple[str, Path, bool]]:
    dirs: list[tuple[str, Path, bool]] = []
    setting_a = preview_dir / "setting_a"
    if setting_a.exists():
        dirs.append(("base_untrained", setting_a, False))

    experiments_dir = preview_dir / "experiments"
    if experiments_dir.exists():
        for directory in sorted({path.parent for path in experiments_dir.rglob("*.jpg")}):
            label = "experiments/" + str(directory.relative_to(experiments_dir))
            dirs.append((label, directory, True))
    return dirs


def get_projected_features(
    model: CLIPModel,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    vision_output = model.vision_model(pixel_values=inputs["pixel_values"])
    text_output = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    image_features = model.visual_projection(vision_output.pooler_output)
    text_features = model.text_projection(text_output.pooler_output)
    return F.normalize(image_features, p=2, dim=-1), F.normalize(text_features, p=2, dim=-1)


def compute_clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    prompt_bank: list[str],
    device: str,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        prompts = [prompt_for_image(path, prompt_bank) for path in batch_paths]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features, text_features = get_projected_features(model, inputs)
            scores = 100 * (image_features * text_features).sum(dim=-1).clamp(min=0)

        total += float(scores.sum().item())
        count += scores.numel()

    return total / count


def step_sort_key(label: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", label)
    if match:
        return int(match.group(1))
    if label == "base_untrained":
        return -1
    return 1000


def print_trends(results: dict[str, dict[str, float]]) -> None:
    by_experiment: dict[str, list[tuple[int, str, float]]] = {}
    for label, row in results.items():
        if not label.startswith("experiments/") or "clip_visual_base_prompt" not in row:
            continue
        parts = label.split("/")
        experiment = parts[1]
        step = step_sort_key(label)
        by_experiment.setdefault(experiment, []).append((step, label, row["clip_visual_base_prompt"]))

    print("\nPer-experiment trend using base prompts")
    for experiment, rows in sorted(by_experiment.items()):
        print(f"  {experiment}")
        for _, label, score in sorted(rows):
            step_name = label.split("/")[-1] if "checkpoint-" in label else "final"
            print(f"    {step_name:<14} {score:.4f}")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base_prompts, lora_prompts = load_prompts(args.prompts_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading CLIP: {CLIP_MODEL}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    results: dict[str, dict[str, float | int | str]] = {}
    for label, directory, is_lora in discover_output_dirs(args.preview_dir):
        image_paths = list_images(directory)
        print(f"\n{label}")
        print(f"  images: {len(image_paths)}")
        if not image_paths:
            results[label] = {"n_images": 0, "error": "no images"}
            continue

        visual_score = compute_clip_score(
            model, processor, image_paths, base_prompts, device, args.batch_size
        )
        row: dict[str, float | int | str] = {
            "directory": str(directory),
            "n_images": len(image_paths),
            "clip_visual_base_prompt": visual_score,
        }
        print(f"  visual/base = {visual_score:.4f}")

        if is_lora:
            trigger_score = compute_clip_score(
                model, processor, image_paths, lora_prompts, device, args.batch_size
            )
            row["clip_with_trigger_prompt"] = trigger_score
            print(f"  with trigger = {trigger_score:.4f}")

        results[label] = row

    args.output.write_text(json.dumps(results, indent=2))

    print("\nSorted by visual/base CLIP score, higher is better")
    for label, row in sorted(
        results.items(),
        key=lambda item: item[1].get("clip_visual_base_prompt", -1),
        reverse=True,
    ):
        visual = row.get("clip_visual_base_prompt")
        if not isinstance(visual, float):
            continue
        trigger = row.get("clip_with_trigger_prompt")
        suffix = "" if not isinstance(trigger, float) else f" | trigger {trigger:.4f}"
        print(f"{visual:.4f} | {label}{suffix}")

    print_trends(results)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()