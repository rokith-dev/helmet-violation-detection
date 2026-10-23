
from pathlib import Path
import random
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
CROP_DIR = PROJECT_DIR / "data" / "processed" / "object_crops"

SAMPLES_PER_CLASS = 6


def main():
    random.seed(42)

    classes = ["With_Helmet", "Without_Helmet"]

    for class_name in classes:
        class_dir = CROP_DIR / class_name

        if not class_dir.is_dir():
            print(f"Class folder not found: {class_dir}")
            continue

        images = sorted(class_dir.glob("*.jpg"))

        if not images:
            print(f"No crops found for {class_name}")
            continue

        samples = random.sample(
            images,
            min(SAMPLES_PER_CLASS, len(images))
        )

        print(f"\n{class_name}: {len(images)} crops")
        print("Sample files:")

        for path in samples:
            with Image.open(path) as image:
                print(
                    f"  {path.name} | "
                    f"size: {image.width} x {image.height}"
                )

        # Create a contact sheet for quick visual inspection.
        thumb_width = 220
        thumb_height = 170
        label_height = 28
        columns = 3
        rows = (len(samples) + columns - 1) // columns

        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width,
                rows * (thumb_height + label_height)
            ),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for index, path in enumerate(samples):
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((thumb_width - 10, thumb_height - 10))

            x = (index % columns) * thumb_width
            y = (index // columns) * (thumb_height + label_height)

            sheet.paste(thumb, (x + 5, y + 5))
            draw.text(
                (x + 5, y + thumb_height),
                path.name[:28],
                fill="black"
            )

        output_path = PROJECT_DIR / "outputs" / "figures" / (
            f"{class_name.lower()}_samples.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=90)

        print(f"Contact sheet saved to: {output_path}")
        sheet.show()


if __name__ == "__main__":
    main()
