
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "yolo_training"
    / "helmet_yolo11n"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "yolo_helmet_dataset"
    / "images"
    / "test"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    images = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(IMAGE_DIR.glob(extension))

    if not images:
        raise FileNotFoundError(f"No test images found in {IMAGE_DIR}")

    image_path = images[0]

    print("Model:", MODEL_PATH)
    print("Testing image:", image_path)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=416,
        save=True,
        project=str(OUTPUT_DIR),
        name="sample_test",
        exist_ok=True,
    )

    print("\n========== DETECTIONS ==========")

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            label = model.names[class_id]

            print(
                f"{label}: {confidence * 100:.2f}% confidence"
            )

    print("\nPrediction image saved under:")
    print(OUTPUT_DIR / "sample_test")


if __name__ == "__main__":
    main()
