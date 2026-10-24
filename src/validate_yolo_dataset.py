
from pathlib import Path
from collections import Counter
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "processed" / "yolo_helmet_dataset"
CONFIG_PATH = DATASET_DIR / "data.yaml"

CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    total_images = 0
    total_objects = 0
    class_counts = Counter()
    errors = []

    for split in ("train", "val", "test"):
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        images = [
            path for path in image_dir.iterdir()
            if path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp"
            }
        ]

        split_objects = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                errors.append(
                    f"Missing label file: {label_path.name}"
                )
                continue

            with label_path.open("r", encoding="utf-8") as file:
                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        f"{label_path.name}, line {line_number}: "
                        "expected 5 values"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(
                        float, parts[1:]
                    )
                except ValueError:
                    errors.append(
                        f"{label_path.name}, line {line_number}: "
                        "invalid numeric value"
                    )
                    continue

                if class_id not in CLASS_NAMES:
                    errors.append(
                        f"{label_path.name}: invalid class {class_id}"
                    )
                    continue

                values = [x_center, y_center, width, height]

                if not all(0 <= value <= 1 for value in values):
                    errors.append(
                        f"{label_path.name}, line {line_number}: "
                        "coordinates outside [0, 1]"
                    )
                    continue

                if width <= 0 or height <= 0:
                    errors.append(
                        f"{label_path.name}, line {line_number}: "
                        "non-positive box dimensions"
                    )
                    continue

                class_counts[class_id] += 1
                split_objects += 1

        total_images += len(images)
        total_objects += split_objects

        print(f"\n{split.upper()}")
        print("Images:", len(images))
        print("Valid objects:", split_objects)

    print("\n========== VALIDATION SUMMARY ==========")
    print("Total images:", total_images)
    print("Total valid objects:", total_objects)

    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name}: {class_counts[class_id]}")

    print("Validation errors:", len(errors))

    for error in errors[:20]:
        print("ERROR:", error)

    if not errors:
        print("\nSUCCESS: All checked YOLO labels are valid.")
    else:
        print("\nPlease investigate the errors before training.")


if __name__ == "__main__":
    main()
