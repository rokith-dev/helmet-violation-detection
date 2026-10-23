
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

EMPTY_FILES = [
    "BikesHelmets35.xml",
    "BikesHelmets459.xml",
    "BikesHelmets735.xml",
]


def main():
    for xml_name in EMPTY_FILES:
        xml_path = ANNOTATIONS_DIR / xml_name

        if not xml_path.is_file():
            print(f"Annotation not found: {xml_name}")
            continue

        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")

        if not filename:
            print(f"Missing image filename in {xml_name}")
            continue

        image_path = IMAGES_DIR / Path(filename).name

        print(f"\nAnnotation: {xml_name}")
        print(f"Image: {filename}")
        print(f"Objects annotated: {len(root.findall('object'))}")

        if not image_path.is_file():
            print("Image file is missing.")
            continue

        with Image.open(image_path) as image:
            print(f"Image dimensions: {image.width} x {image.height}")
            image.show()


if __name__ == "__main__":
    main()
