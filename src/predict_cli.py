
import sys
from predict import OilDetectionPredictor

def main():
    """Main prediction function for the CLI."""
    if len(sys.argv) != 2:
        print("Usage: python predict_cli.py <image_path>")
        return

    image_path = sys.argv[1]

    # Load the model
    model_path = '../models/oil_detection_transfer_learning_20250901_033505.h5'
    predictor = OilDetectionPredictor(model_path)

    # Predict the image
    result = predictor.predict_single_image(image_path, show_image=False)

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

if __name__ == "__main__":
    main()
