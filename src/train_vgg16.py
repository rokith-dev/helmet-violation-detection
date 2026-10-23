
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MANIFEST_PATH = DATA_DIR / "object_crops_split.csv"

MODEL_DIR = PROJECT_DIR / "models"
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
HISTORY_DIR = PROJECT_DIR / "outputs" / "training_history"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

CLASS_NAMES = ["With Helmet", "Without Helmet"]

# Class 0 = With Helmet
# Class 1 = Without Helmet
LABEL_TO_ID = {
    "With Helmet": 0,
    "Without Helmet": 1,
}


# --------------------------------------------------
# 2. LOAD AND CHECK THE SPLIT MANIFEST
# --------------------------------------------------

def load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {MANIFEST_PATH}\n"
            "Run src/split_dataset.py first."
        )

    rows = []

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "label",
            "crop_filename",
            "source_image",
            "split",
        }

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "The manifest must contain these columns: "
                "label, crop_filename, source_image, split."
            )

        for row in reader:
            label = row["label"].strip()

            if label not in LABEL_TO_ID:
                continue

            image_path = PROJECT_DIR / row["crop_filename"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Crop image not found: {image_path}"
                )

            rows.append({
                "path": str(image_path),
                "label": LABEL_TO_ID[label],
                "source_image": row["source_image"],
                "split": row["split"].strip().lower(),
            })

    if not rows:
        raise ValueError("No valid crop records were found.")

    return rows


# --------------------------------------------------
# 3. LOAD AND PREPROCESS IMAGES
# --------------------------------------------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    # VGG16 requires its own preprocessing.
    image = preprocess_input(image)

    return image, label


def make_dataset(rows, training=False):
    paths = [row["path"] for row in rows]

    labels = np.array(
        [row["label"] for row in rows],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(rows),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# --------------------------------------------------
# 4. BUILD THE VGG16 CLASSIFICATION MODEL
# --------------------------------------------------

def build_model():
    # Load VGG16 pretrained on ImageNet.
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze pretrained layers initially.
    base_model.trainable = False

    inputs = Input(
        shape=(224, 224, 3)
    )

    features = base_model(
        inputs,
        training=False,
    )

    features = GlobalAveragePooling2D()(features)

    features = Dense(
        256,
        activation="relu",
    )(features)

    features = Dropout(0.4)(features)

    # Binary classification:
    # 0 = With Helmet
    # 1 = Without Helmet
    outputs = Dense(
        1,
        activation="sigmoid",
    )(features)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="VGG16_Helmet_Classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


# --------------------------------------------------
# 5. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():
    tf.keras.utils.set_random_seed(SEED)

    # Create output folders.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read the dataset manifest.
    rows = load_manifest()

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row for row in rows
        if row["split"] == "validation"
    ]

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError(
            "Training, validation, and test sets must all "
            "contain samples."
        )

    # Check that source images do not cross splits.
    split_sources = {
        "train": {
            row["source_image"] for row in train_rows
        },
        "validation": {
            row["source_image"] for row in val_rows
        },
        "test": {
            row["source_image"] for row in test_rows
        },
    }

    assert split_sources["train"].isdisjoint(
        split_sources["validation"]
    )

    assert split_sources["train"].isdisjoint(
        split_sources["test"]
    )

    assert split_sources["validation"].isdisjoint(
        split_sources["test"]
    )

    # Calculate class weights to help handle class imbalance.
    train_labels = np.array(
        [row["label"] for row in train_rows]
    )

    classes = np.unique(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print("\n========== DATASET SUMMARY ==========")
    print("Training crops:", len(train_rows))
    print("Validation crops:", len(val_rows))
    print("Test crops:", len(test_rows))
    print("Class weights:", class_weights)
    print("Source-image leakage check: PASSED")

    # Create TensorFlow datasets.
    train_ds = make_dataset(
        train_rows,
        training=True,
    )

    val_ds = make_dataset(val_rows)

    test_ds = make_dataset(test_rows)

    # Build and display the model.
    model = build_model()

    model.summary()

    best_model_path = (
        MODEL_DIR / "vgg16_helmet_classifier.keras"
    )

    # Stop when validation loss stops improving.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    # Save the best model during training.
    checkpoint = ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    print("\n========== TRAINING STARTED ==========")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            checkpoint,
        ],
        class_weight=class_weights,
    )

    # Save the model with the best validation weights.
    model.save(best_model_path)

    # Save training history for plotting later.
    history_path = (
        HISTORY_DIR / "vgg16_training_history.json"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: [
                    float(value)
                    for value in values
                ]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )

    # --------------------------------------------------
    # 6. EVALUATE ON THE HELD-OUT TEST SET
    # --------------------------------------------------

    print("\n========== TEST EVALUATION ==========")

    probabilities = model.predict(
        test_ds,
        verbose=1,
    ).ravel()

    y_true = np.array(
        [row["label"] for row in test_rows]
    )

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix:")
    print(matrix)

    # Save metrics as JSON.
    metrics_path = (
        METRICS_DIR / "vgg16_test_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "class_names": CLASS_NAMES,
                "classification_report": report_dict,
                "confusion_matrix": matrix.tolist(),
                "test_samples": len(y_true),
                "decision_threshold": 0.5,
            },
            file,
            indent=2,
        )

    print("\n========== TRAINING COMPLETE ==========")
    print("Best model:", best_model_path)
    print("Training history:", history_path)
    print("Test metrics:", metrics_path)


if __name__ == "__main__":
    main()
