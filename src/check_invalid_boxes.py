
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"


def main():
    print("INVALID BOUNDING BOX INVESTIGATION")
    print("-" * 60)

    invalid_count = 0

    for xml_file in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue

        filename = root.findtext("filename")
        if not filename:
            continue

        image_path = IMAGES_DIR / Path(filename).name
        if not image_path.is_file():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            invalid = (
                xmin < 0 or ymin < 0
                or xmax > width or ymax > height
                or xmin >= xmax or ymin >= ymax
            )

            if invalid:
                invalid_count += 1

                print(f"\nFile: {xml_file.name}")
                print(f"Class: {obj.findtext('name')}")
                print(f"Image size: {width} x {height}")
                print(
                    f"Original box: "
                    f"({xmin}, {ymin}, {xmax}, {ymax})"
                )
                print(
                    f"Coordinates / 1000: "
                    f"({xmin / 1000:.3f}, "
                    f"{ymin / 1000:.3f}, "
                    f"{xmax / 1000:.3f}, "
                    f"{ymax / 1000:.3f})"
                )

    print("\n" + "-" * 60)
    print(f"Invalid boxes found: {invalid_count}")
    print("Original dataset files were not modified.")


if __name__ == "__main__":
    main()
