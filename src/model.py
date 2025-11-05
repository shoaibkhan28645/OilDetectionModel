from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


@dataclass
class ModelMetadata:
    """Metadata saved alongside a trained model."""

    classes: List[str]
    image_size: List[int]
    base_model: str
    loss: str
    metrics: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "classes": self.classes,
                "image_size": self.image_size,
                "base_model": self.base_model,
                "loss": self.loss,
                "metrics": self.metrics,
            },
            indent=2,
        )

    @staticmethod
    def from_json(content: str) -> "ModelMetadata":
        data = json.loads(content)
        return ModelMetadata(
            classes=data.get("classes", []),
            image_size=data.get("image_size", [224, 224, 3]),
            base_model=data.get("base_model", "efficientnet_b0"),
            loss=data.get("loss", "categorical_crossentropy"),
            metrics=data.get("metrics", ["accuracy"]),
        )


class OilDetectionModel:
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        num_classes: Optional[int] = None,
        base_model: str = "efficientnet_b0",
    ):
        self.img_size = img_size + (3,)
        self.num_classes = num_classes
        self.base_model_name = base_model.lower()
        self.base_model = None
        self.model = None
        self.loss: Optional[str] = None
        self.metrics: List = []

    def create_transfer_learning_model(
        self, learning_rate: float = 5e-4, dropout_rate: float = 0.3
    ):
        """Create a transfer learning model using the configured backbone."""

        if self.num_classes is None:
            raise ValueError("num_classes must be provided before creating the model.")

        if self.base_model_name == "efficientnet_b0":
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=self.img_size,
            )
        elif self.base_model_name == "mobilenet_v2":
            base_model = MobileNetV2(
                weights="imagenet", include_top=False, input_shape=self.img_size
            )
        else:
            raise ValueError(
                f"Unsupported base model '{self.base_model_name}'. "
                "Supported options: 'efficientnet_b0', 'mobilenet_v2'."
            )

        base_model.trainable = False
        self.base_model = base_model

        inputs = tf.keras.Input(shape=self.img_size, name="image_input")
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D(name="global_average_pool")(x)
        x = BatchNormalization(name="batch_norm")(x)
        x = Dropout(dropout_rate, name="dropout_1")(x)

        if self.num_classes <= 2:
            x = Dense(128, activation="relu", name="dense_1")(x)
            x = BatchNormalization(name="batch_norm_2")(x)
            x = Dropout(dropout_rate, name="dropout_2")(x)
            outputs = Dense(1, activation="sigmoid", name="predictions")(x)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            x = Dense(256, activation="relu", name="dense_1")(x)
            x = BatchNormalization(name="batch_norm_2")(x)
            x = Dropout(dropout_rate, name="dropout_2")(x)
            outputs = Dense(self.num_classes, activation="softmax", name="predictions")(x)
            loss = "categorical_crossentropy"
            top_k = min(3, self.num_classes)
            metrics = [
                "accuracy",
                TopKCategoricalAccuracy(k=top_k, name=f"top_{top_k}_accuracy"),
            ]

        model = Model(inputs, outputs, name="oil_detection_model")

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)

        self.model = model
        self.loss = loss
        self.metrics = metrics
        return model

    def create_simple_cnn_model(self, learning_rate: float = 1e-3):
        """Create a simple CNN model from scratch."""

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.img_size
                ),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        self.loss = "binary_crossentropy"
        self.metrics = ["accuracy"]
        return model

    def get_callbacks(self, model_save_path: str = "../models/best_oil_detection_model.h5"):
        """Get training callbacks."""

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=5e-4,
            ),
            ModelCheckpoint(
                model_save_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        return callbacks

    def fine_tune_model(self, learning_rate: float = 1e-4):
        """Fine-tune the pre-trained layers."""
        if self.model is None:
            raise ValueError("Model not created yet. Create the model before fine-tuning.")

        if self.base_model is None:
            raise ValueError("Base model is missing; cannot fine-tune.")

        self.base_model.trainable = True

        fine_tune_at = max(0, len(self.base_model.layers) - 50)
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.loss if self.loss else "binary_crossentropy",
            metrics=self.metrics if self.metrics else ["accuracy"],
        )

        print(f"Fine-tuning from layer {fine_tune_at}")

    def plot_training_history(self, history):
        """Plot training history."""

        if history is None:
            print("No training history available to plot.")
            return

        acc_key = next(
            (k for k in history.history if "accuracy" in k and not k.startswith("val_")),
            None,
        )
        val_acc_key = f"val_{acc_key}" if acc_key else None

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs_range = range(len(loss))

        plt.figure(figsize=(12, 4 if acc_key else 3))

        if acc_key:
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, history.history[acc_key], label="Training")
            if val_acc_key in history.history:
                plt.plot(epochs_range, history.history[val_acc_key], label="Validation")
            plt.legend(loc="lower right")
            plt.title(f"{acc_key.replace('_', ' ').title()}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")

            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 1, 1)

        plt.plot(epochs_range, loss, label="Training")
        plt.plot(epochs_range, val_loss, label="Validation")
        plt.legend(loc="upper right")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

    def get_model_summary(self):
        """Print model summary."""
        if self.model is None:
            print("Model not created yet.")
            return

        return self.model.summary()

    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not created yet.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
