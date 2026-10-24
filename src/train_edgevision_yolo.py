
from pathlib import Path
from ultralytics import YOLO
import torch

# Project paths
DATA_YAML = Path("data/processed/edgevision_helmet/data.yaml")
BASE_MODEL = Path(
    "outputs/yolo_training/helmet_yolo11n/weights/best.pt"
)

# New run: preserve the original trained model
PROJECT_DIR = "outputs/yolo_training"
RUN_NAME = "helmet_yolo11n_edgevision"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            f"Existing YOLO checkpoint not found: {BASE_MODEL}"
        )

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset:", DATA_YAML.resolve())
    print("Starting checkpoint:", BASE_MODEL.resolve())

    # Continue training from your existing two-class YOLO11n model.
    model = YOLO(str(BASE_MODEL))

    model.train(
        data=str(DATA_YAML.resolve()),
        epochs=15,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        patience=5,
        project=PROJECT_DIR,
        name="helmet_yolo11n_edgevision_fast",
        exist_ok=False,
        pretrained=True,
        cache=False,
        plots=True,
        seed=42,
        deterministic=True,
        verbose=True,
    )

    print("\nTraining finished.")
    print(
        "Check the best checkpoint at:",
        f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt",
    )


if __name__ == "__main__":
    main()
