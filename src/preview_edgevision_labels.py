
from pathlib import Path
import random
import cv2

DATASET_DIR = Path("data/processed/edgevision_helmet")
OUTPUT_DIR = Path("outputs/edgevision_label_preview")

CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet",
}

SAMPLES = 12
SEED = 42


def main():
    image_dirs = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "images" / "test",
    ]

    image_paths = []

    for image_dir in image_dirs:
        if image_dir.exists():
            image_paths.extend(
                p for p in image_dir.iterdir()
                if p.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }
            )

    if not image_paths:
        print("No images found in the prepared dataset.")
        return

    random.Random(SEED).shuffle(image_paths)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0

    for image_path in image_paths:
        label_path = (
            DATASET_DIR / "labels" /
            image_path.parent.name / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]
        drawn = 0

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(
                    float, parts[1:]
                )
            except ValueError:
                continue

            if class_id not in CLASS_NAMES:
                continue

            x1 = round((x_center - box_width / 2) * width)
            y1 = round((y_center - box_height / 2) * height)
            x2 = round((x_center + box_width / 2) * width)
            y2 = round((y_center + box_height / 2) * height)

            # Clip drawing coordinates to the image.
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            color = (0, 200, 0) if class_id == 0 else (0, 0, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            text = CLASS_NAMES[class_id]
            text_y = max(20, y1 - 7)

            cv2.putText(
                image,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            drawn += 1

        if drawn == 0:
            continue

        output_path = OUTPUT_DIR / f"{image_path.stem}_preview.jpg"
        cv2.imwrite(str(output_path), image)
        saved += 1

        if saved >= SAMPLES:
            break

    print(f"Preview images saved: {saved}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    if saved == 0:
        print("No images with drawable annotations were found.")


if __name__ == "__main__":
    main()
