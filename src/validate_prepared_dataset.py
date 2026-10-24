
from pathlib import Path

DATASET = Path("data/processed/edgevision_helmet")
SPLITS = ["train", "val", "test"]

errors = []
total_images = 0
total_boxes = 0

for split in SPLITS:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        errors.append(f"{split}: missing image or label directory")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    split_boxes = 0

    for image in images:
        total_images += 1
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            errors.append(f"Missing label: {image.name}")
            continue

        for line_no, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                errors.append(f"{label.name}:{line_no}: expected 5 values")
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"{label.name}:{line_no}: non-numeric values")
                continue

            if class_id not in (0, 1):
                errors.append(f"{label.name}:{line_no}: invalid class {class_id}")
                continue

            if not all(map(lambda v: 0 <= v <= 1, (x, y, w, h))):
                errors.append(f"{label.name}:{line_no}: value outside [0, 1]")
                continue

            if w <= 0 or h <= 0:
                errors.append(f"{label.name}:{line_no}: zero/negative box size")
                continue

            if (
                x - w / 2 < -1e-6 or y - h / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6
                or y + h / 2 > 1 + 1e-6
            ):
                errors.append(f"{label.name}:{line_no}: box extends outside image")
                continue

            split_boxes += 1
            total_boxes += 1

    print(f"{split}: {len(images)} images, {split_boxes} valid boxes")

print(f"\nTotal images: {total_images}")
print(f"Total valid boxes: {total_boxes}")
print(f"Validation errors: {len(errors)}")

for error in errors[:30]:
    print("ERROR:", error)

if len(errors) > 30:
    print(f"... and {len(errors) - 30} more errors")

if errors:
    print("\nFix the reported issues before training.")
else:
    print("\nDataset validation passed!")
