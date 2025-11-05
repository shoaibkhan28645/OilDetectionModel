from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import OilDataPreprocessor
from model import ModelMetadata, OilDetectionModel


class OilDetectionTrainer:
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        base_model: str = "efficientnet_b0",
        use_class_weights: bool = True,
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.base_model_name = base_model.lower()
        self.use_class_weights = use_class_weights
        self.history = None
        self.model_wrapper: Optional[OilDetectionModel] = None
        self.class_names: List[str] = []

        self.preprocessor = OilDataPreprocessor(
            img_size=img_size,
            batch_size=batch_size,
            preprocess_fn=self._resolve_preprocess_fn(self.base_model_name),
        )

    @staticmethod
    def _resolve_preprocess_fn(base_model: str):
        """Return the appropriate preprocess_input function for the backbone."""
        base = base_model.lower()
        try:
            if base.startswith("efficientnet"):
                from tensorflow.keras.applications.efficientnet import preprocess_input

                return preprocess_input
            if base == "mobilenet_v2":
                from tensorflow.keras.applications.mobilenet_v2 import (
                    preprocess_input,
                )

                return preprocess_input
        except ImportError:
            pass

        return None

    def _compute_class_weights(self, train_generator) -> Optional[Dict[int, float]]:
        """Compute class weights to address class imbalance."""
        if not self.use_class_weights or not hasattr(train_generator, "classes"):
            return None

        labels = train_generator.classes
        unique_classes = np.unique(labels)
        if len(unique_classes) <= 1:
            return None

        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels,
        )
        return {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}

    def _save_metadata(
        self,
        model_path: str,
        class_names: List[str],
        model_wrapper: OilDetectionModel,
    ):
        """Persist model metadata (class names, image size, backbone)."""
        metadata = ModelMetadata(
            classes=class_names,
            image_size=list(self.img_size + (3,)),
            base_model=self.base_model_name,
            loss=model_wrapper.loss or "binary_crossentropy",
            metrics=[
                metric if isinstance(metric, str) else metric.name
                for metric in (model_wrapper.metrics or [])
            ],
        )

        metadata_path = model_path.replace(".h5", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as fp:
            fp.write(metadata.to_json())

        print(f"Metadata saved to: {metadata_path}")

    def train_model(
        self,
        train_dir: str,
        val_dir: str,
        epochs: int = 50,
        model_type: str = "transfer_learning",
    ):
        """Train the oil detection model."""

        print("=" * 50)
        print("STARTING OIL DETECTION MODEL TRAINING")
        print("=" * 50)

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("Error: Training or validation directory not found!")
            print(f"Train dir: {train_dir}")
            print(f"Val dir: {val_dir}")
            return None

        print("Creating data generators...")
        train_generator, val_generator, classes = self.preprocessor.create_data_generators(
            train_dir, val_dir
        )

        if train_generator.samples == 0:
            print("Error: No training images found!")
            return None

        if val_generator.samples == 0:
            print("Error: No validation images found!")
            return None

        self.class_names = list(train_generator.class_indices.keys())
        num_classes = train_generator.num_classes

        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Detected classes: {self.class_names}")
        print(f"Using base model: {self.base_model_name}")

        if model_type == "transfer_learning":
            self.model_wrapper = OilDetectionModel(
                img_size=self.img_size,
                num_classes=num_classes,
                base_model=self.base_model_name,
            )
            model = self.model_wrapper.create_transfer_learning_model()
        else:
            self.model_wrapper = OilDetectionModel(
                img_size=self.img_size, num_classes=num_classes
            )
            model = self.model_wrapper.create_simple_cnn_model()

        print("Model created successfully!")
        print(f"Total parameters: {model.count_params():,}")

        model_save_path = (
            f"../models/oil_detection_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        os.makedirs("../models", exist_ok=True)

        callbacks = self.model_wrapper.get_callbacks(model_save_path)
        class_weights = self._compute_class_weights(train_generator)
        if class_weights:
            print("Class weights:", class_weights)

        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 50)

        try:
            steps_per_epoch = max(
                1, math.ceil(train_generator.samples / self.batch_size)
            )
            val_steps = max(1, math.ceil(val_generator.samples / self.batch_size))

            print(f"Computed steps_per_epoch: {steps_per_epoch}")
            print(f"Computed validation_steps: {val_steps}")

            self.history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=val_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
            )

            print("\n" * 2 + "=" * 50)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 50)

            self.evaluate_model(val_generator, model_save_path)
            self.model_wrapper.plot_training_history(self.history)
            self._save_metadata(model_save_path, self.class_names, self.model_wrapper)

            return model, self.history

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error during training: {exc}")
            return None

    def fine_tune_model(
        self,
        train_dir: str,
        val_dir: str,
        model_path: str,
        epochs: int = 20,
    ):
        """Fine-tune a pre-trained model."""

        print("=" * 50)
        print("STARTING FINE-TUNING")
        print("=" * 50)

        if self.model_wrapper is None:
            self.model_wrapper = OilDetectionModel(
                img_size=self.img_size, base_model=self.base_model_name
            )

        self.model_wrapper.load_model(model_path)
        self.model_wrapper.fine_tune_model()

        train_generator, val_generator, _ = self.preprocessor.create_data_generators(
            train_dir, val_dir
        )

        fine_tune_save_path = model_path.replace(".h5", "_fine_tuned.h5")
        callbacks = self.model_wrapper.get_callbacks(fine_tune_save_path)

        print(f"Fine-tuning for {epochs} epochs...")

        history_fine = self.model_wrapper.model.fit(
            train_generator,
            steps_per_epoch=max(
                1, math.ceil(train_generator.samples / self.batch_size)
            ),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=max(
                1, math.ceil(val_generator.samples / self.batch_size)
            ),
            callbacks=callbacks,
            verbose=1,
        )

        self.evaluate_model(val_generator, fine_tune_save_path)
        self.model_wrapper.plot_training_history(history_fine)

        return history_fine

    def evaluate_model(self, val_generator, model_path: str):
        """Evaluate the trained model."""

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        model = self.model_wrapper.model
        num_classes = len(self.class_names) or getattr(model, "output_shape", [None, 1])[-1]

        val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        predictions = model.predict(val_generator, verbose=0)

        if predictions.ndim == 1 or predictions.shape[-1] == 1:
            predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
            probabilities = np.hstack([1 - predictions, predictions])
        else:
            predicted_classes = np.argmax(predictions, axis=1)
            probabilities = predictions

        true_classes = val_generator.classes
        class_names = list(val_generator.class_indices.keys())

        print("\nClassification Report:")
        print(
            classification_report(
                true_classes,
                predicted_classes,
                target_names=class_names,
                digits=3,
            )
        )

        self._plot_confusion_matrix(true_classes, predicted_classes, class_names)
        self._log_sample_predictions(val_generator.filenames, probabilities, class_names)

        print(f"\nModel saved to: {model_path}")

    @staticmethod
    def _plot_confusion_matrix(true_classes, predicted_classes, class_names):
        """Plot a confusion matrix."""
        cm = confusion_matrix(true_classes, predicted_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _log_sample_predictions(filenames, probabilities, class_names, limit: int = 5):
        """Log a few sample predictions for sanity checking."""
        if probabilities.size == 0:
            return

        print("\nSample predictions:")
        for filename, probs in zip(filenames[:limit], probabilities[:limit]):
            if probs.ndim == 0:
                probs = np.array([1 - probs, probs])
            prob_dict = {
                class_names[idx]: float(prob) for idx, prob in enumerate(np.atleast_1d(probs))
            }
            top_class = max(prob_dict, key=prob_dict.get)
            print(f"  {filename} -> {top_class} ({prob_dict[top_class]:.3f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the oil detection model.")
    parser.add_argument(
        "--train-dir",
        default="../data/train",
        help="Path to the training directory.",
    )
    parser.add_argument(
        "--val-dir",
        default="../data/validation",
        help="Path to the validation directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (defaults depend on model type).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Input image size as two integers (width height).",
    )
    parser.add_argument(
        "--model-type",
        choices=["transfer_learning", "simple_cnn"],
        default="transfer_learning",
        help="Which model architecture to train.",
    )
    parser.add_argument(
        "--base-model",
        choices=["efficientnet_b0", "mobilenet_v2"],
        default="efficientnet_b0",
        help="Backbone architecture for transfer learning.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable automatic class weight computation.",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Shortcut flag for transfer learning (legacy compatibility).",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Shortcut flag for simple CNN training (legacy compatibility).",
    )

    return parser.parse_args()


def main():
    """Main training function."""

    args = parse_args()

    model_type = args.model_type
    if args.transfer:
        model_type = "transfer_learning"
    elif args.simple:
        model_type = "simple_cnn"

    epochs = args.epochs
    if epochs is None:
        epochs = 30 if model_type == "simple_cnn" else 35

    trainer = OilDetectionTrainer(
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        base_model=args.base_model,
        use_class_weights=not args.no_class_weights,
    )

    print("Oil Detection Model Trainer")
    print("=" * 30)
    print(f"Training {model_type} model for {epochs} epochs")

    if not os.path.exists(args.train_dir):
        print(f"\nError: {args.train_dir} directory not found!")
        print("Please populate your dataset before training.")
        return

    result = trainer.train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=epochs,
        model_type=model_type,
    )

    if result is not None:
        print("\nTraining completed successfully!")
        print("You can now use the trained model for predictions.")
    else:
        print("\nTraining failed. Please check your data and try again.")


if __name__ == "__main__":
    main()
