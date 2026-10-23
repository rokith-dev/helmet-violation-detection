
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import shutil
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "detection_data"
IMAGES_OUT = OUTPUT_DIR / "images"
ANNOTATIONS_OUT = OUTPUT_DIR / "annotations"
REPORT_PATH = OUTPUT_DIR / "preparation_report.json"


def main():
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_OUT.mkdir(parents=True, exist_ok=True)

    report = {
        "source_images_copied": 0,
        "annotations_written": 0,
        "valid_objects_retained": 0,
        "invalid_objects_skipped": 0,
        "empty_annotations": [],
        "missing_images": [],
        "unknown_classes": [],
    }

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            print(f"Skipping malformed XML: {xml_path.name}")
            continue

        filename = root.findtext("filename")
        if not filename:
            continue

        filename = Path(filename).name
        image_path = IMAGES_DIR / filename

        if not image_path.is_file():
            report["missing_images"].append(filename)
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        valid_objects = []

        for obj in root.findall("object"):
            label = obj.findtext("name", default="Unknown")
            box = obj.find("bndbox")

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (AttributeError, TypeError, ValueError):
                report["invalid_objects_skipped"] += 1
                continue

            valid = (
                0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            )

            if not valid:
                report["invalid_objects_skipped"] += 1
                continue

            if label not in {"With Helmet", "Without Helmet"}:
                report["unknown_classes"].append({
                    "file": xml_path.name,
                    "label": label,
                })
                continue

            valid_objects.append({
                "label": label,
                "bbox": [
                    int(xmin), int(ymin), int(xmax), int(ymax)
                ],
            })
            report["valid_objects_retained"] += 1

        if not valid_objects:
            report["empty_annotations"].append(filename)
            continue

        # Copy the image only if it has at least one valid object.
        shutil.copy2(image_path, IMAGES_OUT / filename)
        report["source_images_copied"] += 1

        # Store a simple JSON record for each retained image.
        record = {
            "image": filename,
            "width": width,
            "height": height,
            "objects": valid_objects,
        }

        output_json = ANNOTATIONS_OUT / f"{Path(filename).stem}.json"
        with output_json.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

        report["annotations_written"] += 1

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n===== DETECTION DATA PREPARATION =====")
    print(f"Images copied: {report['source_images_copied']}")
    print(f"Annotation files written: {report['annotations_written']}")
    print(f"Valid objects retained: {report['valid_objects_retained']}")
    print(f"Invalid objects skipped: {report['invalid_objects_skipped']}")
    print(f"Images without retained objects: {len(report['empty_annotations'])}")
    print(f"Missing images: {len(report['missing_images'])}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
