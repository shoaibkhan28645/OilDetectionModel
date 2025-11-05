from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)

DEFAULT_CLASS_QUERIES: Dict[str, List[str]] = {
    "coriander_oil": ["coriander oil bottle", "dhaniya oil bottle"],
    "mustard_oil": ["mustard oil bottle", "sarso tel bottle"],
    "olive_oil": ["olive oil extra virgin bottle", "olive oil glass bottle"],
    "coconut_oil": ["coconut oil bottle", "virgin coconut oil bottle"],
    "sunflower_oil": ["sunflower oil bottle", "refined sunflower oil"],
    "sesame_oil": ["sesame oil bottle", "gingelly oil bottle"],
    "groundnut_oil": ["groundnut oil bottle", "peanut oil bottle"],
    "soybean_oil": ["soybean oil bottle", "soya oil bottle"],
    "almond_oil": ["almond oil bottle", "badam oil bottle"],
}


@dataclass
class DownloadStats:
    attempted: int = 0
    saved: int = 0
    skipped: int = 0
    failed: int = 0


def _download_image(url: str, timeout: int = 20) -> Optional[tuple[bytes, str]]:
    headers = {"User-Agent": USER_AGENT, "Referer": "https://duckduckgo.com/"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < 15_000:
            return None

        content_type = response.headers.get("Content-Type", "")

        return response.content, content_type
    except requests.RequestException:
        return None


def _validate_and_save(content: bytes, destination: Path, min_side: int = 128) -> bool:
    try:
        with Image.open(BytesIO(content)) as img:
            if min(img.size) < min_side:
                return False

            rgb = img.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination, format="JPEG", quality=92)
            return True
    except (UnidentifiedImageError, OSError):
        return False


def _image_hash(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


def _iter_image_results(query: str, max_results: int) -> Iterable[Dict]:
    with DDGS() as ddgs:
        try:
            results = ddgs.images(
                query,
                max_results=max_results,
                safesearch="moderate",
            )
        except TypeError:
            results = ddgs.images(
                keywords=query,
                max_results=max_results,
                safesearch="moderate",
                size=None,
                type_image="photo",
            )

        for result in results:
            yield result


def _build_filename(class_dir: Path, url: str, extension: str, existing_hashes: Set[str], content_hash: str) -> Path:
    filename = f"{content_hash[:16]}{extension}"
    destination = class_dir / filename
    suffix = 0
    while destination.exists() or content_hash in existing_hashes:
        suffix += 1
        destination = class_dir / f"{content_hash[:16]}_{suffix}{extension}"
    existing_hashes.add(content_hash)
    return destination


def download_images_for_class(
    class_name: str,
    queries: List[str],
    output_dir: Path,
    max_images: int,
    throttle: float,
) -> DownloadStats:
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    stats = DownloadStats()
    seen_urls: Set[str] = set()
    existing_hashes: Set[str] = set()

    for existing_file in class_dir.glob("*.jpg"):
        try:
            existing_hashes.add(_image_hash(existing_file.read_bytes()))
        except OSError:
            continue

    pbar = tqdm(total=max_images, desc=f"{class_name}", unit="img")

    for query in queries:
        if stats.saved >= max_images:
            break

        for result in _iter_image_results(query, max_results=max_images * 2):
            if stats.saved >= max_images:
                break

            url = result.get("image")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            stats.attempted += 1

            download_result = _download_image(url)
            if download_result is None:
                stats.failed += 1
                continue

            content, content_type = download_result
            content_hash = _image_hash(content)
            destination = _build_filename(
                class_dir, url, ".jpg", existing_hashes, content_hash
            )

            if _validate_and_save(content, destination):
                stats.saved += 1
                pbar.update(1)
                time.sleep(throttle)
            else:
                stats.skipped += 1
                destination.unlink(missing_ok=True)

    pbar.close()
    print(
        f"Finished {class_name}: saved={stats.saved}, skipped={stats.skipped}, "
        f"failed={stats.failed} (attempted {stats.attempted})"
    )
    return stats


def download_dataset(
    classes: Dict[str, List[str]],
    output_dir: Path,
    max_images_per_class: int,
    throttle: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, DownloadStats] = {}
    for class_name, queries in classes.items():
        stats = download_images_for_class(
            class_name=class_name,
            queries=queries,
            output_dir=output_dir,
            max_images=max_images_per_class,
            throttle=throttle,
        )
        summary[class_name] = stats

    print("\nDownload summary:")
    for class_name, stats in summary.items():
        print(
            f"  - {class_name}: saved={stats.saved}, skipped={stats.skipped}, failed={stats.failed}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Download oil images for dataset building.")
    parser.add_argument(
        "--output-dir",
        default="../data/raw",
        help="Destination directory for downloaded images.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=150,
        help="Maximum number of images to download per class.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.15,
        help="Delay (in seconds) between successful downloads to avoid rate limiting.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Subset of classes to download (defaults to all known oil classes).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_classes = DEFAULT_CLASS_QUERIES
    if args.classes:
        missing = [cls for cls in args.classes if cls not in DEFAULT_CLASS_QUERIES]
        if missing:
            raise ValueError(
                f"Unknown classes requested: {', '.join(missing)}. "
                f"Available classes: {', '.join(DEFAULT_CLASS_QUERIES.keys())}"
            )
        selected_classes = {cls: DEFAULT_CLASS_QUERIES[cls] for cls in args.classes}

    download_dataset(
        classes=selected_classes,
        output_dir=Path(args.output_dir),
        max_images_per_class=args.max_per_class,
        throttle=args.throttle,
    )


if __name__ == "__main__":
    main()
