
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    print("=" * 55)
    print("HELMET DETECTION DATASET INSPECTION")
    print("=" * 55)

    if not IMAGES_DIR.is_dir() or not ANNOTATIONS_DIR.is_dir():
        print("ERROR: Dataset images or annotations folder not found.")
        return

    images = [
        p for p in IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

    image_map = {p.name.lower(): p for p in images}
    classes = Counter()
    empty_files = []
    missing_images = []
    invalid_xml = []
    corrupt_images = []
    invalid_boxes = []
    missing_boxes = 0
    total_objects = 0

    print(f"\nTotal images: {len(images)}")
    print(f"Total XML annotations: {len(xml_files)}")

    # Check whether images can be opened
    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as error:
            corrupt_images.append((image_path.name, str(error)))

    # Inspect each XML annotation
    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            invalid_xml.append(xml_file.name)
            continue

        filename = root.findtext("filename")
        image_path = image_map.get(Path(filename).name.lower()) if filename else None

        if image_path is None:
            missing_images.append((xml_file.name, filename or "No filename"))
            continue

        try:
            root_width = int(root.findtext("size/width"))
            root_height = int(root.findtext("size/height"))
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size

            if (root_width, root_height) != (actual_width, actual_height):
                print(
                    f"Dimension mismatch: {xml_file.name} "
                    f"XML={root_width}x{root_height}, "
                    f"image={actual_width}x{actual_height}"
                )
        except (TypeError, ValueError, OSError):
            print(f"Could not verify dimensions: {xml_file.name}")
            continue

        objects = root.findall("object")
        if not objects:
            empty_files.append(xml_file.name)

        for obj in objects:
            label = obj.findtext("name")
            if label and label.strip():
                classes[label.strip()] += 1
                total_objects += 1

            box = obj.find("bndbox")
            if box is None:
                missing_boxes += 1
                continue

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                if (
                    xmin < 0 or ymin < 0
                    or xmax > actual_width or ymax > actual_height
                    or xmin >= xmax or ymin >= ymax
                ):
                    invalid_boxes.append(
                        (xml_file.name, (xmin, ymin, xmax, ymax),
                         actual_width, actual_height)
                    )
            except (TypeError, ValueError):
                missing_boxes += 1

    print("\n--- CLASS COUNTS ---")
    print(f"Total annotated objects: {total_objects}")
    for label, count in sorted(classes.items()):
        print(f"{label}: {count}")

    print("\n--- DATA QUALITY ---")
    print(f"Empty annotations: {len(empty_files)}")
    print("Empty files:", empty_files)
    print(f"Invalid XML files: {len(invalid_xml)}")
    print(f"Missing image references: {len(missing_images)}")
    print(f"Unreadable images: {len(corrupt_images)}")
    print(f"Missing bounding boxes: {missing_boxes}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")

    if invalid_boxes:
        for item in invalid_boxes[:10]:
            print("Invalid box:", item)

    if missing_images:
        print("Missing image examples:", missing_images[:10])

    if corrupt_images:
        print("Unreadable image examples:", corrupt_images[:10])

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
