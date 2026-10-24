
from pathlib import Path
import cv2
import math

INPUT_DIR = Path("data/processed/hard_examples")
OUTPUT_PATH = Path("outputs/hard_examples_contact_sheet.jpg")

def main():
    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if not image_paths:
        print("No extracted frames found.")
        return

    thumbnail_width = 320
    thumbnail_height = 200
    label_height = 30
    columns = 3
    rows = math.ceil(len(image_paths) / columns)

    sheet = 255 * __import__("numpy").ones(
        (rows * (thumbnail_height + label_height),
         columns * thumbnail_width, 3),
        dtype="uint8"
    )

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))

        if image is None:
            continue

        image = cv2.resize(
            image, (thumbnail_width, thumbnail_height)
        )

        row = index // columns
        col = index % columns
        x = col * thumbnail_width
        y = row * (thumbnail_height + label_height)

        sheet[y:y + thumbnail_height, x:x + thumbnail_width] = image

        cv2.putText(
            sheet,
            path.name,
            (x + 8, y + thumbnail_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), sheet)

    print(f"Contact sheet saved: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
