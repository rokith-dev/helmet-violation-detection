
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Dataset paths
DATASET_DIR = Path(r"D:\HelmetProject\HelmetDetectionDataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

# Change this filename to inspect another image
XML_FILENAME = "BikesHelmets103.xml"


def main():
    xml_path = ANNOTATIONS_DIR / XML_FILENAME

    # Check whether the annotation file exists
    if not xml_path.is_file():
        print(f"ERROR: Annotation file not found: {xml_path}")
        return

    # Read the XML annotation
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        print(f"ERROR: Could not parse XML: {error}")
        return

    # Find the corresponding image
    filename = root.findtext("filename")

    if not filename:
        print("ERROR: Image filename is missing from XML.")
        return

    image_path = IMAGES_DIR / Path(filename).name

    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        return

    # Open image and prepare drawing
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size

    print("=" * 60)
    print(f"Image: {filename}")
    print(f"Image dimensions: {width} x {height}")
    print("=" * 60)

    valid_count = 0
    invalid_count = 0

    # Inspect every annotated object
    for index, obj in enumerate(root.findall("object"), start=1):
        label = obj.findtext("name", default="Unknown")
        box = obj.find("bndbox")

        if box is None:
            print(f"Object {index}: {label} - NO BOUNDING BOX")
            invalid_count += 1
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            print(f"Object {index}: {label} - INVALID COORDINATE VALUES")
            invalid_count += 1
            continue

        # Check coordinates against image dimensions
        valid = (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        )

        print(
            f"Object {index}: {label}\n"
            f"  Coordinates: "
            f"({xmin}, {ymin}, {xmax}, {ymax})\n"
            f"  Status: {'VALID' if valid else 'INVALID'}"
        )

        if valid:
            valid_count += 1

            # Draw valid boxes in green
            draw.rectangle(
                (xmin, ymin, xmax, ymax),
                outline="green",
                width=3
            )

            # Add class label
            text_y = max(0, int(ymin) - 18)

            draw.text(
                (int(xmin), text_y),
                label,
                fill="yellow",
                stroke_width=1,
                stroke_fill="black"
            )
        else:
            invalid_count += 1
            print("  Box not drawn because its coordinates are invalid.")

    print("\n" + "=" * 60)
    print(f"Valid boxes:   {valid_count}")
    print(f"Invalid boxes: {invalid_count}")
    print("Original dataset files were not modified.")
    print("=" * 60)

    # Display the annotated image
    image.show()


if __name__ == "__main__":
    main()
