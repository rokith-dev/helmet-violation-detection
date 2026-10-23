
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"


def main():
    class_counts = Counter()
    invalid_boxes = []
    empty_annotations = []
    total_objects = 0
    valid_boxes = 0
    xml_count = 0
    parse_errors = 0
    missing_images = 0

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        xml_count += 1

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            parse_errors += 1
            continue

        filename = root.findtext("filename")

        if not filename:
            missing_images += 1
            continue

        image_path = IMAGES_DIR / Path(filename).name

        if not image_path.is_file():
            missing_images += 1
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        objects = root.findall("object")

        if not objects:
            empty_annotations.append(filename)

        for obj in objects:
            total_objects += 1
            label = obj.findtext("name", default="Unknown")
            class_counts[label] += 1

            box = obj.find("bndbox")

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (AttributeError, TypeError, ValueError):
                invalid_boxes.append((xml_path.name, label, "Missing/invalid values"))
                continue

            is_valid = (
                0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            )

            if is_valid:
                valid_boxes += 1
            else:
                invalid_boxes.append(
                    (xml_path.name, label, (xmin, ymin, xmax, ymax))
                )

    print("\n===== DATASET ANNOTATION SUMMARY =====")
    print(f"XML annotation files: {xml_count}")
    print(f"XML parsing errors: {parse_errors}")
    print(f"Missing image references: {missing_images}")
    print(f"Total annotated objects: {total_objects}")
    print(f"Valid bounding boxes: {valid_boxes}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")

    print("\n===== CLASS DISTRIBUTION =====")
    for label, count in sorted(class_counts.items()):
        print(f"{label}: {count}")

    print("\n===== EMPTY ANNOTATIONS =====")
    for filename in empty_annotations:
        print(filename)
    print(f"Total empty annotations: {len(empty_annotations)}")

    print("\n===== INVALID BOXES =====")
    for filename, label, coordinates in invalid_boxes:
        print(f"{filename} | {label} | {coordinates}")

    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()
