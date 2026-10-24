
from pathlib import Path
import csv
import shutil
import xml.etree.ElementTree as ET

import yaml


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Your original Kaggle dataset is outside the repository.
SOURCE_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = SOURCE_DIR / "images"
ANNOTATIONS_DIR = SOURCE_DIR / "annotations"

SPLIT_MANIFEST = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "object_crops_split.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "yolo_helmet_dataset"
)

CLASS_TO_ID = {
    "With Helmet": 0,
    "Without Helmet": 1,
}

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
]


def load_source_splits():
    """Reuse the existing source-image split to prevent leakage."""

    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_MANIFEST}\n"
            "Run src/split_dataset.py first."
        )

    source_splits = {}

    with SPLIT_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
            source_image = Path(row["source_image"]).name
            split = row["split"].strip().lower()

            if split not in {"train", "validation", "test"}:
                continue

            if source_image in source_splits:
                if source_splits[source_image] != split:
                    raise ValueError(
                        f"Source image appears in multiple splits: "
                        f"{source_image}"
                    )
            else:
                source_splits[source_image] = split

    return source_splits


def find_source_image(filename):
    """Find an image by its exact filename."""

    candidate = IMAGES_DIR / filename

    if candidate.is_file():
        return candidate

    # Try extensions only if XML's filename has no extension.
    if not Path(filename).suffix:
        for extension in IMAGE_EXTENSIONS:
            candidate = IMAGES_DIR / f"{filename}{extension}"
            if candidate.is_file():
                return candidate

    return None


def read_valid_objects(xml_path):
    """Read valid Pascal VOC bounding boxes from one XML file."""

    root = ET.parse(xml_path).getroot()

    objects = []

    size = root.find("size")
    if size is None:
        return None, []

    try:
        image_width = int(size.findtext("width", "0"))
        image_height = int(size.findtext("height", "0"))
    except (ValueError, TypeError):
        return None, []

    if image_width <= 0 or image_height <= 0:
        return None, []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()

        if class_name not in CLASS_TO_ID:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (ValueError, TypeError):
            continue

        # Skip invalid coordinates; never silently repair them.
        if not (
            0 <= xmin < xmax <= image_width
            and 0 <= ymin < ymax <= image_height
        ):
            continue

        # Convert Pascal VOC coordinates to normalized YOLO xywh.
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        objects.append(
            (
                CLASS_TO_ID[class_name],
                x_center,
                y_center,
                box_width,
                box_height,
            )
        )

    return (image_width, image_height), objects


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Images folder not found: {IMAGES_DIR}"
        )

    if not ANNOTATIONS_DIR.is_dir():
        raise FileNotFoundError(
            f"Annotations folder not found: {ANNOTATIONS_DIR}"
        )

    source_splits = load_source_splits()

    # Create output folders.
    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    split_folder = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    counts = {
        "train": {"images": 0, "objects": 0},
        "val": {"images": 0, "objects": 0},
        "test": {"images": 0, "objects": 0},
    }

    missing_images = 0
    skipped_xml = 0
    skipped_empty_or_invalid = 0
    seen_sources = set()

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            skipped_xml += 1
            continue

        filename = root.findtext("filename", "").strip()

        if not filename:
            skipped_xml += 1
            continue

        # Use only source images already assigned to a split.
        if filename not in source_splits:
            continue

        if filename in seen_sources:
            continue

        image_path = find_source_image(filename)

        if image_path is None:
            missing_images += 1
            continue

        try:
            image_info, objects = read_valid_objects(xml_path)
        except (ET.ParseError, OSError, ValueError):
            skipped_xml += 1
            continue

        if image_info is None:
            skipped_xml += 1
            continue

        split = split_folder[source_splits[filename]]

        destination_image = (
            OUTPUT_DIR
            / "images"
            / split
            / image_path.name
        )

        destination_label = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        shutil.copy2(image_path, destination_image)

        # An empty label file is valid for images with no valid boxes.
        with destination_label.open(
            "w",
            encoding="utf-8",
        ) as file:
            for class_id, xc, yc, bw, bh in objects:
                file.write(
                    f"{class_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}\n"
                )

        counts[split]["images"] += 1
        counts[split]["objects"] += len(objects)
        seen_sources.add(filename)

        if not objects:
            skipped_empty_or_invalid += 1

    # Write Ultralytics dataset configuration.
    yaml_path = OUTPUT_DIR / "data.yaml"

    config = {
        "path": OUTPUT_DIR.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "With Helmet",
            1: "Without Helmet",
        },
    }

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\n========== YOLO DATASET PREPARATION ==========")

    for split in ("train", "val", "test"):
        print(f"\n{split.upper()}")
        print("Images:", counts[split]["images"])
        print("Valid objects:", counts[split]["objects"])

    print("\nImages missing from source:", missing_images)
    print("XML files skipped:", skipped_xml)
    print(
        "Images with no valid boxes:",
        skipped_empty_or_invalid,
    )
    print("Unique source images copied:", len(seen_sources))
    print("Dataset configuration:", yaml_path)
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
