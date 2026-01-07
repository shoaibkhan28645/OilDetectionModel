from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(directory: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in directory.glob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]
    )


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_dataset(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    test_dir: str | None = None,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
):
    """
    Split a dataset organised by folders into train/validation/(optional) test sets.

    Args:
        source_dir: Directory containing one sub-folder per oil class.
        train_dir: Destination directory for training samples.
        val_dir: Destination directory for validation samples.
        test_dir: Optional destination for test samples.
        val_split: Fraction of images to allocate to validation.
        test_split: Fraction of images to allocate to test (ignored when None).
        seed: Random seed to ensure reproducibility.
    """

    source_path = Path(source_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    test_path = Path(test_dir) if test_dir else None

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source directory {source_dir} not found. "
            "Place your raw images in this directory before splitting."
        )

    class_directories = [
        item for item in source_path.iterdir() if item.is_dir() and not item.name.startswith(".")
    ]

    if not class_directories:
        raise ValueError(
            f"No class folders detected in {source_dir}. "
            "Create one sub-folder per oil class and populate it with images."
        )

    print("🔄 Starting automatic data split...")
    print(f"📁 Found classes: {', '.join(sorted(item.name for item in class_directories))}")

    _ensure_dir(train_path)
    _ensure_dir(val_path)
    if test_path:
        _ensure_dir(test_path)

    rng = random.Random(seed)

    for class_dir in class_directories:
        images = _collect_images(class_dir)
        if not images:
            print(f"❌ No images found in {class_dir}. Skipping.")
            continue

        rng.shuffle(images)

        total = len(images)
        val_count = int(total * val_split)
        test_count = int(total * test_split) if test_path else 0
        train_count = total - val_count - test_count

        train_files = images[:train_count]
        val_files = images[train_count : train_count + val_count]
        test_files = images[train_count + val_count :] if test_path else []

        print(
            f"\n📁 {class_dir.name}: total={total} | train={len(train_files)} | "
            f"val={len(val_files)} | test={len(test_files)}"
        )

        _copy_files(train_files, train_path / class_dir.name)
        _copy_files(val_files, val_path / class_dir.name)
        if test_path:
            _copy_files(test_files, test_path / class_dir.name)

    print("\n✅ Data split completed!")
    print(f"📂 Training images: {train_path}")
    print(f"📂 Validation images: {val_path}")
    if test_path:
        print(f"📂 Test images: {test_path}")


def _copy_files(files: Iterable[Path], destination: Path):
    _ensure_dir(destination)
    for src in files:
        dst = destination / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)


def count_images(directory: str):
    """Count images in each class directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"\n📊 {directory} does not exist.")
        return

    print(f"\n📊 Image count in {directory}:")

    total = 0
    for class_dir in sorted(item for item in directory_path.iterdir() if item.is_dir()):
        count = len(_collect_images(class_dir))
        print(f"   {class_dir.name}: {count} images")
        total += count

    print(f"   Total: {total} images")


if __name__ == "__main__":
    SOURCE_DIR = "data/all_images"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/validation"
    TEST_DIR = "data/test"

    print("🤖 Oil Detection Dataset Splitter")
    print("=" * 40)

    try:
        split_dataset(
            source_dir=SOURCE_DIR,
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            val_split=0.2,
            test_split=0.1,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n❌ {exc}")
    else:
        count_images(TRAIN_DIR)
        count_images(VAL_DIR)
        count_images(TEST_DIR)
        print("\n🚀 Ready to train! Run:")
        print("   cd src")
        print("   python train.py --transfer")
