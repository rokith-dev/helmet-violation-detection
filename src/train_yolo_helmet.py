
from pathlib import Path
import json
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_YAML = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "yolo_helmet_dataset"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo_training"

MODEL_NAME = "yolo11n.pt"
EPOCHS = 30
IMAGE_SIZE = 416
BATCH_SIZE = 4


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use GPU if available; otherwise use CPU.
    device = 0 if torch.cuda.is_available() else "cpu"

    print("========== YOLO HELMET TRAINING ==========")
    print("Dataset:", DATA_YAML)
    print("Base model:", MODEL_NAME)
    print("Device:", "GPU" if device == 0 else "CPU")
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=2,
        project=str(OUTPUT_DIR),
        name="helmet_yolo11n",
        exist_ok=True,
        patience=7,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
    )

    best_model = (
        OUTPUT_DIR / "helmet_yolo11n" / "weights" / "best.pt"
    )

    last_model = (
        OUTPUT_DIR / "helmet_yolo11n" / "weights" / "last.pt"
    )

    print("\n========== TRAINING FINISHED ==========")
    print("Best model exists:", best_model.exists())
    print("Best model path:", best_model)
    print("Last model exists:", last_model.exists())
    print("Last model path:", last_model)

    if best_model.exists():
        best_detector = YOLO(str(best_model))

        print("\nEvaluating best model on the test dataset...")
        metrics = best_detector.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            workers=2,
            project=str(OUTPUT_DIR),
            name="helmet_test_evaluation",
            exist_ok=True,
        )

        results = {
            "model": str(best_model),
            "test_precision": float(metrics.box.mp),
            "test_recall": float(metrics.box.mr),
            "test_map50": float(metrics.box.map50),
            "test_map50_95": float(metrics.box.map),
        }

        metrics_path = (
            PROJECT_DIR
            / "outputs"
            / "metrics"
            / "yolo_helmet_test_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print("\n========== TEST METRICS ==========")
        for name, value in results.items():
            print(f"{name}: {value}")

        print("\nMetrics saved to:", metrics_path)
    else:
        print(
            "\nNo best.pt was found. Check the training output "
            "for errors before evaluating."
        )


if __name__ == "__main__":
    main()
