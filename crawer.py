from __future__ import annotations

import csv
import hashlib
import imghdr
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from ddgs import DDGS
from PIL import Image

SEARCH_QUERIES = [
    "rockfall blocking road real photo",
    "highway landslide debris",
    "road rockfall damage",
]

OUTPUT_FOLDER = Path("rockfall_dataset")
LIMIT_PER_QUERY = 30
MAX_RESULTS_SCAN = 240
MIN_WIDTH = 640
MIN_HEIGHT = 400

# 常見圖庫與浮水印來源，先擋掉避免髒資料進來
BLOCKED_DOMAINS = {
    "dreamstime.com",
    "alamy.com",
    "shutterstock.com",
    "istockphoto.com",
    "gettyimages.com",
    "freepik.com",
    "adobestock.com",
    "depositphotos.com",
    "123rf.com",
}

BLOCKED_URL_KEYWORDS = {
    "watermark",
    "stock-photo",
    "stockphoto",
    "stock-image",
}


def is_blocked_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in BLOCKED_DOMAINS):
        return True

    url_lower = url.lower()
    return any(keyword in url_lower for keyword in BLOCKED_URL_KEYWORDS)


def sanitize_folder_name(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", name).strip()


def detect_extension(content: bytes, content_type: str | None, url: str) -> str:
    kind = imghdr.what(None, content)
    if kind == "jpeg":
        return ".jpg"
    if kind in {"png", "gif", "webp", "bmp", "tiff"}:
        return f".{kind}"

    if content_type:
        content_type = content_type.lower()
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def valid_image_bytes(content: bytes) -> bool:
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False
            img.verify()
        return True
    except Exception:
        return False


def download_query_images(query: str, output_root: Path) -> None:
    folder = output_root / sanitize_folder_name(query)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "sources.csv"

    existing_names = {
        p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    }
    next_index = len(existing_names) + 1

    seen_hashes: set[str] = set()
    kept = 0

    print(f"準備下載關鍵字: {query}")
    print(f"[%] 目標資料夾: {folder}")

    fieldnames = ["filename", "source_url", "page_url", "title"]
    should_write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        with DDGS() as ddgs:
            results = ddgs.images(
                query,
                region="wt-wt",
                safesearch="off",
                max_results=MAX_RESULTS_SCAN,
            )

            for item in results:
                if kept >= LIMIT_PER_QUERY:
                    break

                image_url = item.get("image")
                if not image_url or is_blocked_url(image_url):
                    continue

                page_url = item.get("url", "")
                title = item.get("title", "")

                try:
                    response = requests.get(
                        image_url,
                        timeout=20,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    response.raise_for_status()
                except Exception:
                    continue

                content = response.content
                if not content or not valid_image_bytes(content):
                    continue

                digest = hashlib.sha256(content).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                ext = detect_extension(content, response.headers.get("Content-Type"), image_url)
                filename = f"Image_{next_index}{ext}"
                target = folder / filename

                with target.open("wb") as f:
                    f.write(content)

                writer.writerow(
                    {
                        "filename": filename,
                        "source_url": image_url,
                        "page_url": page_url,
                        "title": title,
                    }
                )

                kept += 1
                next_index += 1
                print(f"[%] 已下載 {kept}/{LIMIT_PER_QUERY}: {filename}")

    print(f"[%] 完成: {query}，共下載 {kept} 張")


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for query in SEARCH_QUERIES:
        download_query_images(query, OUTPUT_FOLDER)
    print("下載完成！")


if __name__ == "__main__":
    main()