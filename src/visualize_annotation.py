
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

XML_FILENAME = "BikesHelmets140.xml"


def main():
    xml_path = ANNOTATIONS_DIR / XML_FILENAME
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")
    image_path = IMAGES_DIR / Path(filename).name

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    print(f"Image: {filename}")
    print(f"Image size: {width} x {height}")

    for index, obj in enumerate(root.findall("object"), start=1):
        label = obj.findtext("name", default="Unknown")
        box = obj.find("bndbox")

        if box is None:
            print(f"Object {index}: no bounding box")
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            print(f"Object {index}: invalid coordinate values")
            continue

        valid = (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        )

        color = "green" if valid else "red"

        print(
            f"Object {index}: {label} | "
            f"({xmin}, {ymin}, {xmax}, {ymax}) | "
            f"{'VALID' if valid else 'INVALID'}"
        )

        if valid:
            draw.rectangle(
                (xmin, ymin, xmax, ymax),
                outline=color,
                width=3
            )
            draw.text(
                (xmin, max(0, ymin - 18)),
                label,
                fill="yellow",
                stroke_width=1,
                stroke_fill="black"
            )
        else:
            print("  Not drawn because the coordinates are outside the image.")

    print("Original dataset files were not modified.")
    image.show()


if __name__ == "__main__":
    main()
