
from pathlib import Path
import json
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "detection_data"

IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

ALLOWED_CLASSES = {"With Helmet", "Without Helmet"}


def main():
    image_count = 0
    annotation_count = 0
    valid_objects = 0
    invalid_objects = 0
    missing_annotations = []
    missing_images = []
    invalid_files = []
    class_counts = {label: 0 for label in ALLOWED_CLASSES}

    image_paths = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file()
    )

    for image_path in image_paths:
        image_count += 1
        json_path = ANNOTATIONS_DIR / f"{image_path.stem}.json"

        if not json_path.is_file():
            missing_annotations.append(image_path.name)
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                record = json.load(file)

            with Image.open(image_path) as image:
                width, height = image.size

            if record["image"] != image_path.name:
                invalid_files.append(
                    f"{json_path.name}: image filename mismatch"
                )

            if record["width"] != width or record["height"] != height:
                invalid_files.append(
                    f"{json_path.name}: image dimensions mismatch"
                )

            annotation_count += 1

            for obj in record.get("objects", []):
                label = obj.get("label")
                bbox = obj.get("bbox", [])

                if (
                    label not in ALLOWED_CLASSES
                    or len(bbox) != 4
                ):
                    invalid_objects += 1
                    continue

                xmin, ymin, xmax, ymax = bbox

                valid = (
                    0 <= xmin < xmax <= width
                    and 0 <= ymin < ymax <= height
                )

                if valid:
                    valid_objects += 1
                    class_counts[label] += 1
                else:
                    invalid_objects += 1
                    invalid_files.append(
                        f"{json_path.name}: invalid box {bbox}"
                    )

        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid_files.append(f"{json_path.name}: {error}")

    image_names = {path.name for path in image_paths}

    for json_path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        annotation_name = f"{json_path.stem}{json_path.suffix}"
        expected_image = f"{json_path.stem}.png"

        # Compare annotation names against all supported image extensions.
        matching_image_exists = any(
            (IMAGES_DIR / f"{json_path.stem}{ext}").is_file()
            for ext in (".png", ".jpg", ".jpeg", ".webp")
        )

        if not matching_image_exists:
            missing_images.append(annotation_name)

    print("\n===== PREPARED DATASET VALIDATION =====")
    print(f"Images found: {image_count}")
    print(f"Annotations matched: {annotation_count}")
    print(f"Valid objects: {valid_objects}")
    print(f"Invalid objects: {invalid_objects}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Annotations missing images: {len(missing_images)}")

    print("\n===== CLASS COUNTS =====")
    for label, count in sorted(class_counts.items()):
        print(f"{label}: {count}")

    print("\n===== VALIDATION ERRORS =====")
    for error in invalid_files:
        print(error)

    if missing_annotations:
        print("\nImages without annotations:")
        for filename in missing_annotations:
            print(filename)

    print("\nValidation finished. Source dataset was not modified.")


if __name__ == "__main__":
    main()
