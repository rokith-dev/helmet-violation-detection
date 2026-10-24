
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(
    "data/external/edgevision/EdgeVision Dataset/"
    "EdgeVision-Dataset"
)

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels" / "yolo"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

EXPECTED_CLASS_IDS = {0, 1, 2}


# --------------------------------------------------
# Dataset validation
# --------------------------------------------------

def main():
    print("=" * 60)
    print("EDGEVISION DATASET VALIDATION")
    print("=" * 60)

    # Check required directories
    if not IMAGE_DIR.is_dir():
        print(f"ERROR: Image directory not found:\n{IMAGE_DIR}")
        return

    if not LABEL_DIR.is_dir():
        print(f"ERROR: Label directory not found:\n{LABEL_DIR}")
        return

    # Collect images and labels
    images = {
        path.stem: path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        path.stem: path
        for path in LABEL_DIR.glob("*.txt")
    }

    image_stems = set(images)
    label_stems = set(labels)

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    print(f"\nImages found: {len(images)}")
    print(f"Label files found: {len(labels)}")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

    if images_without_labels:
        print("\nImages without labels:")
        for name in images_without_labels[:20]:
            print(f"  {name}")

    if labels_without_images:
        print("\nLabels without images:")
        for name in labels_without_images[:20]:
            print(f"  {name}")

    # Validate YOLO annotation contents
    class_counts = {}
    invalid_annotations = []
    empty_label_files = 0
    valid_annotation_count = 0

    for label_path in sorted(labels.values()):
        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as error:
            invalid_annotations.append(
                f"{label_path.name}: cannot read file: {error}"
            )
            continue

        nonempty_lines = [
            (line_number, line.strip())
            for line_number, line in enumerate(lines, start=1)
            if line.strip()
        ]

        if not nonempty_lines:
            empty_label_files += 1
            continue

        for line_number, line in nonempty_lines:
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError(
                        f"Expected 5 fields, found {len(parts)}"
                    )

                class_id = int(parts[0])

                if class_id not in EXPECTED_CLASS_IDS:
                    raise ValueError(
                        f"Unexpected class ID: {class_id}"
                    )

                x_center, y_center, width, height = map(
                    float, parts[1:]
                )

                values = (
                    x_center,
                    y_center,
                    width,
                    height,
                )

                if not all(
                    value == value
                    and abs(value) != float("inf")
                    for value in values
                ):
                    raise ValueError(
                        "Coordinates contain NaN or infinity"
                    )

                if not all(
                    0 <= value <= 1
                    for value in values
                ):
                    raise ValueError(
                        "Coordinates or dimensions outside [0, 1]"
                    )

                if width <= 0 or height <= 0:
                    raise ValueError(
                        "Bounding-box width and height must be positive"
                    )

                if (
                    x_center - width / 2 < -1e-6
                    or x_center + width / 2 > 1 + 1e-6
                    or y_center - height / 2 < -1e-6
                    or y_center + height / 2 > 1 + 1e-6
                ):
                    raise ValueError(
                        "Bounding box extends outside image boundaries"
                    )

                class_counts[class_id] = (
                    class_counts.get(class_id, 0) + 1
                )
                valid_annotation_count += 1

            except (ValueError, OverflowError) as error:
                invalid_annotations.append(
                    f"{label_path.name}:{line_number}: "
                    f"{error} | {line}"
                )

    # Report
    print("\n" + "-" * 60)
    print("ANNOTATION SUMMARY")
    print("-" * 60)

    for class_id in sorted(EXPECTED_CLASS_IDS):
        print(
            f"Class {class_id}: "
            f"{class_counts.get(class_id, 0)} valid objects"
        )

    print(f"\nValid annotation lines: {valid_annotation_count}")
    print(f"Empty label files: {empty_label_files}")
    print(f"Invalid annotation lines: {len(invalid_annotations)}")

    if invalid_annotations:
        print("\nInvalid annotations:")
        for item in invalid_annotations:
            print(f"  {item}")

    # Save a report without changing the dataset
    report_path = Path("outputs/edgevision_validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report:
        report.write("EDGEVISION DATASET VALIDATION REPORT\n")
        report.write("=" * 45 + "\n")
        report.write(f"Images: {len(images)}\n")
        report.write(f"Label files: {len(labels)}\n")
        report.write(
            f"Images without labels: {len(images_without_labels)}\n"
        )
        report.write(
            f"Labels without images: {len(labels_without_images)}\n"
        )
        report.write(f"Valid annotations: {valid_annotation_count}\n")
        report.write(f"Class counts: {class_counts}\n")
        report.write(f"Empty label files: {empty_label_files}\n")
        report.write(
            f"Invalid annotation lines: {len(invalid_annotations)}\n"
        )

        for item in invalid_annotations:
            report.write(f"{item}\n")

    print(f"\nReport saved to: {report_path}")
    print("\nValidation completed.")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
