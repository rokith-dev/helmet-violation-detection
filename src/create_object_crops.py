
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "object_crops"
MANIFEST_PATH = PROJECT_DIR / "data" / "processed" / "object_crops_manifest.csv"

VALID_CLASSES = {"With Helmet", "Without Helmet"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    skipped_invalid = 0
    skipped_unknown = 0
    missing_images = 0
    crop_index = 0

    with MANIFEST_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "crop_filename",
                "label",
                "source_image",
                "source_annotation",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        writer.writeheader()

        for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                print(f"Skipping malformed XML: {xml_path.name}")
                continue

            filename = root.findtext("filename")
            if not filename:
                continue

            image_path = IMAGES_DIR / Path(filename).name
            if not image_path.is_file():
                missing_images += 1
                continue

            with Image.open(image_path) as source:
                image = source.convert("RGB")

            width, height = image.size

            for obj_index, obj in enumerate(
                root.findall("object"), start=1
            ):
                label = obj.findtext("name", default="Unknown")

                if label not in VALID_CLASSES:
                    skipped_unknown += 1
                    continue

                box = obj.find("bndbox")
                try:
                    xmin = float(box.findtext("xmin"))
                    ymin = float(box.findtext("ymin"))
                    xmax = float(box.findtext("xmax"))
                    ymax = float(box.findtext("ymax"))
                except (AttributeError, TypeError, ValueError):
                    skipped_invalid += 1
                    continue

                # Never guess or automatically repair coordinates.
                valid = (
                    0 <= xmin < xmax <= width
                    and 0 <= ymin < ymax <= height
                )

                if not valid:
                    skipped_invalid += 1
                    continue

                crop = image.crop(
                    (
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    )
                )

                crop_index += 1
                crop_name = (
                    f"{Path(filename).stem}_obj{obj_index:02d}.jpg"
                )

                class_dir = OUTPUT_DIR / label.replace(" ", "_")
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_path = class_dir / crop_name

                crop.save(crop_path, quality=95)

                writer.writerow({
                    "crop_filename": str(
                        crop_path.relative_to(PROJECT_DIR)
                    ),
                    "label": label,
                    "source_image": filename,
                    "source_annotation": xml_path.name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                })

                counts[label] += 1

    print("\n===== OBJECT CROP GENERATION =====")
    print(f"With Helmet crops: {counts['With Helmet']}")
    print(f"Without Helmet crops: {counts['Without Helmet']}")
    print(f"Total crops: {crop_index}")
    print(f"Invalid boxes skipped: {skipped_invalid}")
    print(f"Unknown classes skipped: {skipped_unknown}")
    print(f"Missing images: {missing_images}")
    print(f"Crop directory: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
