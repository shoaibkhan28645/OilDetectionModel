from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import ModelMetadata


class OilDetectionPredictor:
    def __init__(self, model_path: str, default_img_size: tuple[int, int] = (224, 224)):
        self.model_path = model_path
        self.metadata = self._load_metadata(model_path)

        if self.metadata and self.metadata.image_size:
            width, height = self.metadata.image_size[:2]
            self.img_size = (width, height)
        else:
            self.img_size = default_img_size

        self.class_names = (    
            self.metadata.classes if self.metadata and self.metadata.classes else ["coriander_oil", "mustard_oil"]
        )
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_metadata(model_path: str) -> Optional[ModelMetadata]:
        metadata_path = model_path.replace(".h5", "_metadata.json")
        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as fp:
                return ModelMetadata.from_json(fp.read())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: Failed to read metadata ({exc}). Falling back to defaults.")
            return None

    @staticmethod
    def _load_model(model_path: str):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error loading model: {exc}")
            sys.exit(1)

    def _preprocess_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def _decode_predictions(self, prediction: np.ndarray) -> Dict:
        """Convert raw model predictions into readable outputs."""
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, ...]

        if prediction.shape[-1] == 1:
            prob = float(prediction.flatten()[0])
            predicted_idx = 1 if prob > 0.5 else 0
            confidence = prob if predicted_idx == 1 else 1 - prob
            probs = [1 - prob, prob]
        else:
            probs = prediction.flatten()
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])

        class_probabilities = {
            cls_name: float(probs[idx]) for idx, cls_name in enumerate(self.class_names)
        }

        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": confidence,
            "probabilities": class_probabilities,
        }

    def predict_single_image(self, image_path: str, show_image: bool = True):
        """Predict oil type for a single image."""
        preprocessed_image = self._preprocess_image(image_path)
        if preprocessed_image is None:
            return None

        prediction = self.model.predict(preprocessed_image, verbose=0)
        decoded = self._decode_predictions(prediction[0])

        result = {
            "image_path": image_path,
            **decoded,
        }

        if show_image:
            self._display_prediction(image_path, result)

        return result

    def predict_batch_images(self, image_directory: str):
        """Predict oil type for all images in a directory."""
        supported_formats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_paths: List[str] = []

        for fmt in supported_formats:
            image_paths.extend(glob.glob(os.path.join(image_directory, fmt)))
            image_paths.extend(glob.glob(os.path.join(image_directory, fmt.upper())))

        if not image_paths:
            print(f"No images found in {image_directory}")
            return []

        results = []
        print(f"Processing {len(image_paths)} images...")

        for idx, image_path in enumerate(sorted(image_paths), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(image_path)}")
            result = self.predict_single_image(image_path, show_image=False)

            if result:
                results.append(result)
                print(
                    f"  Prediction: {result['predicted_class']} "
                    f"(confidence: {result['confidence']:.3f})"
                )

        return results

    def _display_prediction(self, image_path: str, result: Dict):
        """Display image with prediction result."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to display {image_path}; could not load image.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis("off")

        top_probs = sorted(
            result["probabilities"].items(), key=lambda item: item[1], reverse=True
        )[:3]
        probs_text = "\n".join([f"{name}: {prob:.2f}" for name, prob in top_probs])

        color = (
            "green"
            if result["confidence"] > 0.8
            else "orange"
            if result["confidence"] > 0.6
            else "red"
        )

        plt.title(
            f"Prediction: {result['predicted_class']} ({result['confidence']:.2f})\n{probs_text}",
            fontsize=14,
            color=color,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def display_batch_results(self, results: List[Dict], max_display: int = 6):
        """Display results for multiple images."""
        if not results:
            print("No results to display.")
            return

        num_images = min(len(results), max_display)
        cols = 3
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = np.array([axes])

        for i in range(rows * cols):
            row, col = divmod(i, cols)
            if i < num_images:
                result = results[i]
                image = cv2.imread(result["image_path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(image)
                axes[row, col].axis("off")

                top_prob = result["confidence"]
                color = (
                    "green"
                    if top_prob > 0.8
                    else "orange"
                    if top_prob > 0.6
                    else "red"
                )

                axes[row, col].set_title(
                    f"{result['predicted_class']} ({top_prob:.2f})",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                )
            else:
                axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

        self.print_summary_stats(results)

    def print_summary_stats(self, results: List[Dict]):
        """Print summary statistics for batch predictions."""
        if not results:
            return

        total_images = len(results)
        prediction_counts = Counter(result["predicted_class"] for result in results)

        high_confidence = sum(1 for r in results if r["confidence"] > 0.8)
        medium_confidence = sum(1 for r in results if 0.6 < r["confidence"] <= 0.8)
        low_confidence = total_images - high_confidence - medium_confidence

        print("\n" + "=" * 50)
        print("PREDICTION SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {total_images}")
        for class_name in self.class_names:
            count = prediction_counts.get(class_name, 0)
            print(f"  {class_name}: {count} ({count / total_images * 100:.1f}%)")

        print("\nConfidence distribution:")
        print(
            f"  High confidence (>80%): {high_confidence} "
            f"({high_confidence / total_images * 100:.1f}%)"
        )
        print(
            f"  Medium confidence (60-80%): {medium_confidence} "
            f"({medium_confidence / total_images * 100:.1f}%)"
        )
        print(
            f"  Low confidence (<60%): {low_confidence} "
            f"({low_confidence / total_images * 100:.1f}%)"
        )
        print("=" * 50)


def main():
    """Main prediction function."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python predict.py <model_path> <image_path>")
        print("  python predict.py <model_path> <image_directory>")
        return

    model_path = sys.argv[1]
    input_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} not found!")
        return

    predictor = OilDetectionPredictor(model_path)

    if os.path.isfile(input_path):
        print(f"Making prediction for single image: {input_path}")
        result = predictor.predict_single_image(input_path)

        if result:
            print("\nPrediction Result:")
            print(f"  Image: {result['image_path']}")
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(
                "  Probabilities: "
                + ", ".join(
                    f"{cls}={prob:.3f}" for cls, prob in result["probabilities"].items()
                )
            )

    elif os.path.isdir(input_path):
        print(f"Making predictions for all images in directory: {input_path}")
        results = predictor.predict_batch_images(input_path)

        if results:
            predictor.display_batch_results(results)
        else:
            print("No predictions were made.")
    else:
        print(f"Error: {input_path} is neither a file nor a directory!")


if __name__ == "__main__":
    main()
