
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

# Change this filename later to inspect other images.
XML_FILENAME = "BikesHelmets103.xml"


def main():
    xml_path = ANNOTATIONS_DIR / XML_FILENAME
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")
    image_path = IMAGES_DIR / Path(filename).name

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    print(f"Image: {filename}")
    print(f"Image dimensions: {width} x {height}")

    for obj in root.findall("object"):
        label = obj.findtext("name", default="Unknown")
        box = obj.find("bndbox")

        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Use scaled coordinates only when the originals are invalid
        # and the scaled coordinates form a valid box in this image.
        original_valid = (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        )

        if not original_valid:
            scaled = (xmin / 1000, ymin / 1000,
                      xmax / 1000, ymax / 1000)

            sx1, sy1, sx2, sy2 = scaled
            scaled_valid = (
                0 <= sx1 < sx2 <= width
                and 0 <= sy1 < sy2 <= height
            )

            if not scaled_valid:
                print(f"Skipping invalid box for: {label}")
                continue

            xmin, ymin, xmax, ymax = scaled
            print(f"{label}: displaying coordinates divided by 1000")
        else:
            print(f"{label}: displaying original coordinates")

        draw.rectangle(
            (xmin, ymin, xmax, ymax),
            outline="red",
            width=3
        )
        draw.text(
            (xmin, max(0, ymin - 18)),
            label,
            fill="yellow",
            stroke_width=1,
            stroke_fill="black"
        )

    print("Original image and XML were not modified.")
    image.show()


if __name__ == "__main__":
    main()
