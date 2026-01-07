from __future__ import annotations

import sys
from pathlib import Path

sys.path.append("src")

from src.train import OilDetectionTrainer


def _discover_classes(directory: Path):
    """Return sorted class names discovered under a directory."""
    if not directory.exists():
        return []
    return sorted(
        [
            item.name
            for item in directory.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ]
    )


def simple_train():
    """Simple training entry point that reuses the main trainer with defaults."""

    print("🤖 Simple Oil Detection Training")
    print("=" * 40)

    train_dir = Path("data/train")
    val_dir = Path("data/validation")

    classes = _discover_classes(train_dir)

    if len(classes) < 2:
        print("❌ Need at least two oil categories in data/train to start training.")
        print("📁 Please create one folder per oil type and add images before retrying.")
        return

    print("📊 Discovered classes:", ", ".join(classes))

    for class_name in classes:
        class_dir = train_dir / class_name
        image_count = len(
            [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ]
        )
        print(f"  - {class_name}: {image_count} images")

    if not val_dir.exists():
        print("\n⚠️  data/validation not found. Using train data for validation preview.")
        val_dir = train_dir

    print("\n🔄 Starting transfer learning with default settings...")

    trainer = OilDetectionTrainer(batch_size=16)

    model, _ = trainer.train_model(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        epochs=20,
        model_type="transfer_learning",
    )

    if model is not None:
        print("\n✅ Training completed!")
        print("📁 Check the models/ directory for the saved model and metadata.")
        print("🔬 To run inference:")
        print("   python src/predict.py models/<your_model>.h5 path/to/image_or_folder")
    else:
        print("❌ Training failed. Please review the logs above for details.")


if __name__ == "__main__":
    simple_train()
