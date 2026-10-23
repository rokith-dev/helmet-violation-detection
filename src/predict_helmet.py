
from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "vgg16_helmet_classifier.keras"
)

IMAGE_SIZE = (224, 224)


def predict_helmet(image_path):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load the trained model.
    model = tf.keras.models.load_model(MODEL_PATH)

    # Read and resize the image.
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(image, IMAGE_SIZE)
    image = image.numpy().astype(np.float32)

    # Add a batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    # Apply VGG16 preprocessing.
    image = preprocess_input(image)

    # Model output is the probability of class 1:
    # class 1 = Without Helmet.
    probability_without_helmet = float(
        model.predict(image, verbose=0)[0][0]
    )

    probability_with_helmet = (
        1.0 - probability_without_helmet
    )

    if probability_without_helmet >= 0.5:
        prediction = "Without Helmet"
    else:
        prediction = "With Helmet"

    print("\n========== HELMET PREDICTION ==========")
    print(f"Image: {image_path.name}")
    print(f"Prediction: {prediction}")
    print(
        f"With Helmet probability: "
        f"{probability_with_helmet:.2%}"
    )
    print(
        f"Without Helmet probability: "
        f"{probability_without_helmet:.2%}"
    )
    print(
        "\nNote: This classifier expects a cropped rider/object "
        "image similar to its training data."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict helmet status from an image."
    )

    parser.add_argument(
        "image",
        help="Path to a JPG or PNG image.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    predict_helmet(image_path)


if __name__ == "__main__":
    main()
