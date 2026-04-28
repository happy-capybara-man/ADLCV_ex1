from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-3-27b-it"
TRIGGER_TOKEN = "<road_rockfall_event>"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_INLINE_B64_LEN = 180_000
DEFAULT_PROMPT = (
    "Step 1: Carefully observe this road rockfall disaster photo. List all explicitly visible objects and visual evidence, "
    "such as rocks, road surface, slope, guardrails, signs, cones, vehicles, machinery, trees, weather, lighting, and terrain. "
    "Do not guess or hallucinate items that are not clearly visible. "
    "Step 2: Based strictly on the visible evidence from Step 1, write one fluent, natural English dense caption, "
    "about 70 to 120 words long. Focus especially on the physical characteristics of the fallen rocks, including their size, "
    "shape, color, rough texture, dust, mud, and whether they are massive boulders, jagged blocks, scattered gravel, "
    "or wet muddy debris; their spatial distribution across the asphalt road, shoulder, lane markings, roadside, or slope; "
    "how the rockfall blocks, buries, cracks, or damages the road; and how it interacts with the surrounding terrain, "
    "weather, lighting, atmosphere, and documentary camera perspective. "
    "Do not start with filler phrases like 'The image shows', 'This is a picture of', or 'In this image'. "
    "Do not use bullet points, markdown, labels, or comma-separated tags in Step 2. "
    "Do not include the trigger token; it will be added separately."
)
FALLBACK_PROMPTS = [
    (
        "Write a single 70 to 120 word English dense caption for this road rockfall scene. "
        "Prioritize the rocks' physical size, shape, texture, color, dust or mud, their distribution over the road, "
        "blocked lanes, buried or cracked asphalt, damaged guardrails, signs, cones, vehicles, or cleanup equipment, "
        "plus the collapsed slope, cliff, mountain road, tunnel, trees, weather, lighting, atmosphere, and camera angle. "
        "Start directly with the scene description and avoid filler openings."
    ),
    (
        "Create a natural English SD3.5 training caption for this image in one paragraph. "
        "Describe the road rockfall as a physical disaster scene: rock sizes and textures, debris placement, road blockage, "
        "damage to pavement or roadside infrastructure, terrain context, weather, light, mood, and documentary camera perspective."
    ),
]
RETRY_ATTEMPTS = 2
SD35_ENRICHMENT = (
    "The scene is framed as a realistic road rockfall event, with fallen rocks and loose debris blocking or covering "
    "the travel surface and creating a clear spatial relationship between the roadway, the obstruction, and the surrounding terrain. "
    "Natural outdoor lighting, visible weather conditions, and a documentary camera perspective give the image a grounded photojournalistic style."
)
ENRICHMENTS = [
    "The scene is framed as a realistic road rockfall event, with fallen rocks blocking the travel surface. Natural outdoor lighting gives the image a documentary style.",
    "This documentary-style photograph captures a road hazard caused by a rockfall, detailing the debris scattered across the terrain and roadway.",
    "Presented from a realistic photojournalistic perspective, the image highlights the physical obstruction of a mountain road due to collapsed rocks.",
    "The outdoor daylight illuminates a hazardous road condition where loose debris and rocks have compromised the paved surface.",
]


def encode_image_for_api(path: Path, max_b64_len: int = MAX_INLINE_B64_LEN) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")

        longest_side = max(image.size)
        for side in [longest_side, 1024, 896, 768, 640, 512, 448, 384, 320]:
            if side < longest_side:
                candidate = image.copy()
                candidate.thumbnail((side, side), Image.LANCZOS)
            else:
                candidate = image

            for quality in [90, 82, 74, 66, 58, 50, 42, 34]:
                buffer = BytesIO()
                candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                if len(encoded) < max_b64_len:
                    return encoded

    raise ValueError(f"Could not compress {path} below API inline image limit")


def extract_stream_text(response: requests.Response) -> str:
    parts: list[str] = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8")
        if not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        parts.append(extract_text(payload))

    return "".join(parts).strip()


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "".join(extract_text(item) for item in payload)
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list):
        return "".join(extract_text(choice) for choice in choices)

    delta = payload.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if content:
            return extract_text(content)

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content:
            return extract_text(content)

    content = payload.get("content")
    if content:
        return extract_text(content)

    text = payload.get("text")
    if isinstance(text, str):
        return text

    return ""


def request_caption(
    image_path: Path,
    invoke_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    timeout: int,
) -> str:
    image_b64 = encode_image_for_api(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    caption = extract_stream_text(response) if stream else extract_text(response.json()).strip()
    if not caption:
        raise RuntimeError(f"Empty caption returned for {image_path}")
    return caption


def request_caption_with_fallbacks(
    image_path: Path,
    invoke_url: str,
    model: str,
    api_key: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    timeout: int,
) -> str:
    last_error = ""

    for prompt in prompts:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                candidate = request_caption(
                    image_path=image_path,
                    invoke_url=invoke_url,
                    model=model,
                    api_key=api_key,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=stream,
                    timeout=timeout,
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_error = str(exc)
                print(f"  retry {attempt}/{RETRY_ATTEMPTS} failed: {last_error}")
                time.sleep(1.0)
                continue

            if not is_bad_caption(candidate):
                return candidate

            last_error = f"unusable caption: {candidate!r}"
            print(f"  retry {attempt}/{RETRY_ATTEMPTS} failed: {last_error}")
            break

    print(f"  using generic fallback caption for {image_path.name}: {last_error}")
    return build_generic_fallback_caption(image_path)


def clean_caption(caption: str) -> str:
    if "step 2:" in caption.lower():
        caption = re.split(r"step\s*2\s*:", caption, flags=re.IGNORECASE)[-1]

    caption = re.sub(r"\s+", " ", caption).strip()
    caption = caption.strip(" \t\n\r\"'`")
    caption = re.sub(r"^(caption|description)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"^in this image (i can see|we can see)\s+", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"^in this image (there is|there are)\s+", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"^(the image shows|this image shows)\s+", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bfew\b", "several", caption, flags=re.IGNORECASE)
    caption = caption.rstrip()
    return caption


def sd35_caption(caption: str) -> str:
    caption = clean_caption(caption)
    caption = naturalize_caption(caption)
    enrichment = random.choice(ENRICHMENTS)
    if not caption:
        return enrichment
    if caption.endswith("."):
        return f"{caption} {enrichment}"
    return f"{caption}. {enrichment}"

def is_generated_fallback(caption: str) -> bool:
    normalized = clean_caption(caption).lower()
    return any(
        phrase in normalized
        for phrase in [
            "a detailed road rockfall scene shows large rugged gray rocks",
            "suitable for a detailed road hazard training caption",
        ]
    )

def naturalize_caption(caption: str) -> str:
    if not caption:
        return caption

    parts = list(dict.fromkeys(clean_list_item(part) for part in caption.split(",") if part.strip(" .")))
    parts = [part for part in parts if part]
    if len(parts) >= 3 and "." not in caption:
        return f"The image shows {join_english_list(parts)}."

    return caption[0].upper() + caption[1:] if caption[0].islower() else caption


def clean_list_item(item: str) -> str:
    item = item.strip(" .")
    item = re.sub(r"^(the image shows|this image shows)\s+", "", item, flags=re.IGNORECASE)
    item = re.sub(r"^and\s+", "", item, flags=re.IGNORECASE)
    item = re.sub(r"^there (is|are)\s+", "", item, flags=re.IGNORECASE)
    return item


def join_english_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def build_generic_fallback_caption(image_path: Path) -> str:
    brightness, color_cast, orientation = image_hints(image_path)
    return (
        f"A {orientation} realistic photograph shows a road rockfall scene with large rugged gray rocks, loose gravel, "
        f"brown dirt, and scattered debris spread across a paved mountain road. The rocks have rough fractured surfaces "
        f"and irregular edges, contrasting with the smoother roadway and the natural roadside terrain. The image has "
        f"{brightness} outdoor lighting with a subtle {color_cast} color cast, giving the scene a grounded documentary "
        f"camera perspective suitable for a detailed road hazard training caption."
    )


def image_hints(image_path: Path) -> tuple[str, str, str]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        resized = image.resize((1, 1), Image.LANCZOS)
        red, green, blue = resized.getpixel((0, 0))

    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    if luminance >= 170:
        brightness = "bright daylight"
    elif luminance >= 95:
        brightness = "diffused daylight"
    else:
        brightness = "dim, low-contrast daylight"

    strongest_channel = max((red, "warm brown"), (green, "greenish natural"), (blue, "cool gray-blue"))[1]
    orientation = "wide landscape" if width >= height else "vertical"
    return brightness, strongest_channel, orientation


def with_trigger(caption: str, trigger_token: str) -> str:
    caption = clean_caption(caption)
    if caption.startswith(trigger_token):
        return caption
    return f"{trigger_token}, {caption}"


def is_bad_caption(caption: str) -> bool:
    normalized = clean_caption(caption).lower()
    if len(normalized.split()) < 4:
        return True
    if normalized in {"yes", "no", "none", "unknown"}:
        return True
    return any(phrase in normalized for phrase in ["sorry", "not trained", "cannot answer", "can't answer"])


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return {entry["file_name"]: entry for entry in data if "file_name" in entry and "text" in entry}


def write_outputs(entries: list[dict[str, str]], json_path: Path, jsonl_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption training_data images with NVIDIA chat/completions VLM models.")
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-json", type=Path, default=Path("metadata.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("metadata.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--invoke-url", default=None, help="Override the NVIDIA VLM endpoint URL.")
    parser.add_argument("--model", default=None, help="Override the NVIDIA VLM model name.")
    parser.add_argument("--trigger-token", default=TRIGGER_TOKEN)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--no-stream", action="store_true", help="Request a normal JSON response instead of SSE streaming.")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--raw-captions", action="store_true", help="Write raw VLM captions without SD3.5 enrichment.")
    parser.add_argument("--force", action="store_true", help="Regenerate captions that already exist in output JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key. Set {args.api_key_env} before running.", file=sys.stderr)
        return 2

    invoke_url = args.invoke_url or os.environ.get("NVIDIA_VLM_INVOKE_URL", INVOKE_URL)
    model = args.model or os.environ.get("NVIDIA_VLM_MODEL", DEFAULT_MODEL)

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    images = sorted(path for path in args.data_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {args.data_dir}", file=sys.stderr)
        return 2

    existing = load_existing(args.output_json)
    entries: list[dict[str, str]] = []

    for index, image_path in enumerate(images, start=1):
        previous = existing.get(image_path.name)
        if previous and not args.force and not is_generated_fallback(previous["text"]):
            entry = {"file_name": image_path.name, "text": with_trigger(previous["text"], args.trigger_token)}
            print(f"[{index:03d}/{len(images):03d}] reuse {image_path.name}")
        else:
            print(f"[{index:03d}/{len(images):03d}] caption {image_path.name}")
            prompts = [args.prompt, *[prompt for prompt in FALLBACK_PROMPTS if prompt != args.prompt]]
            caption = request_caption_with_fallbacks(
                image_path=image_path,
                invoke_url=invoke_url,
                model=model,
                api_key=api_key,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stream=not args.no_stream,
                timeout=args.timeout,
            )
            if not args.raw_captions:
                caption = sd35_caption(caption)
            entry = {"file_name": image_path.name, "text": with_trigger(caption, args.trigger_token)}
            time.sleep(args.sleep)

        entries.append(entry)
        write_outputs(entries, args.output_json, args.output_jsonl)

    write_outputs(entries, args.output_json, args.output_jsonl)
    print(f"Wrote {len(entries)} captions to {args.output_json} and {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())