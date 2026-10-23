
from pathlib import Path
import xml.etree.ElementTree as ET
import json
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "outputs" / "metrics"
REPORT_PATH = REPORT_DIR / "annotation_cleaning_report.json"


def inspect_dataset():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "total_xml_files": 0,
        "valid_objects": 0,
        "invalid_objects": 0,
        "empty_annotations": [],
        "missing_images": [],
        "invalid_boxes": [],
        "class_counts_valid": {},
    }

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        report["total_xml_files"] += 1

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            report.setdefault("xml_errors", []).append(
                {"file": xml_path.name, "error": str(error)}
            )
            continue

        filename = root.findtext("filename")

        if not filename:
            report["missing_images"].append({
                "annotation": xml_path.name,
                "reason": "Missing filename in XML",
            })
            continue

        image_path = IMAGES_DIR / Path(filename).name

        if not image_path.is_file():
            report["missing_images"].append({
                "annotation": xml_path.name,
                "image": filename,
            })
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        objects = root.findall("object")

        if not objects:
            report["empty_annotations"].append(filename)

        for obj in objects:
            label = obj.findtext("name", default="Unknown")
            box = obj.find("bndbox")

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (AttributeError, TypeError, ValueError):
                report["invalid_objects"] += 1
                report["invalid_boxes"].append({
                    "annotation": xml_path.name,
                    "label": label,
                    "reason": "Missing or non-numeric coordinates",
                })
                continue

            valid = (
                0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            )

            if valid:
                report["valid_objects"] += 1
                report["class_counts_valid"][label] = (
                    report["class_counts_valid"].get(label, 0) + 1
                )
            else:
                report["invalid_objects"] += 1
                report["invalid_boxes"].append({
                    "annotation": xml_path.name,
                    "label": label,
                    "coordinates": [xmin, ymin, xmax, ymax],
                    "image_size": [width, height],
                })

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("DATASET INSPECTION COMPLETE")
    print(f"XML files: {report['total_xml_files']}")
    print(f"Valid objects: {report['valid_objects']}")
    print(f"Invalid objects: {report['invalid_objects']}")
    print(f"Empty annotations: {len(report['empty_annotations'])}")
    print(f"Missing images: {len(report['missing_images'])}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    inspect_dataset()
