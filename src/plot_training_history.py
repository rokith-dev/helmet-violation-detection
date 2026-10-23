
from pathlib import Path
import json

import matplotlib.pyplot as plt


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

HISTORY_PATH = (
    PROJECT_DIR
    / "outputs"
    / "training_history"
    / "vgg16_training_history.json"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"


def main():
    # Check that training history exists
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    # Read training history
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # 1. ACCURACY GRAPH
    # -----------------------------------------
    train_accuracy = history.get("accuracy", [])
    val_accuracy = history.get("val_accuracy", [])

    if train_accuracy and val_accuracy:
        epochs = range(1, len(train_accuracy) + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(
            epochs,
            train_accuracy,
            marker="o",
            label="Training Accuracy",
        )
        plt.plot(
            epochs,
            val_accuracy,
            marker="o",
            label="Validation Accuracy",
        )

        plt.title("VGG16 Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xticks(list(epochs))
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        accuracy_path = OUTPUT_DIR / "vgg16_accuracy.png"
        plt.savefig(accuracy_path, dpi=300)
        plt.show()
        plt.close()

        print(f"Accuracy graph saved: {accuracy_path}")
    else:
        print("Accuracy values are missing from training history.")

    # -----------------------------------------
    # 2. LOSS GRAPH
    # -----------------------------------------
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(
            epochs,
            train_loss,
            marker="o",
            label="Training Loss",
        )
        plt.plot(
            epochs,
            val_loss,
            marker="o",
            label="Validation Loss",
        )

        plt.title("VGG16 Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.xticks(list(epochs))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        loss_path = OUTPUT_DIR / "vgg16_loss.png"
        plt.savefig(loss_path, dpi=300)
        plt.show()
        plt.close()

        print(f"Loss graph saved: {loss_path}")
    else:
        print("Loss values are missing from training history.")

    print("\nGraph generation complete.")


if __name__ == "__main__":
    main()
