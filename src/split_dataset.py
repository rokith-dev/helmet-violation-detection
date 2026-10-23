
from pathlib import Path
import csv
import random
from collections import defaultdict, Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

MANIFEST_PATH = DATA_DIR / "object_crops_manifest.csv"
OUTPUT_PATH = DATA_DIR / "object_crops_split.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    # Group all crops from the same original image together.
    groups = defaultdict(list)

    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "crop_filename",
            "label",
            "source_image",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest must contain columns: {sorted(required)}"
            )

        for row in reader:
            groups[row["source_image"]].append(row)

    source_images = list(groups)
    random.Random(SEED).shuffle(source_images)

    total_images = len(source_images)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = set(source_images[:train_end])
    val_images = set(source_images[train_end:val_end])
    test_images = set(source_images[val_end:])

    def assign_split(source_image):
        if source_image in train_images:
            return "train"
        if source_image in val_images:
            return "validation"
        return "test"

    counts = Counter()
    image_counts = Counter()

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "crop_filename",
            "label",
            "source_image",
            "source_annotation",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "split",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for source_image, rows in groups.items():
            split = assign_split(source_image)
            image_counts[split] += 1

            for row in rows:
                output_row = {
                    field: row.get(field, "")
                    for field in fieldnames
                    if field != "split"
                }
                output_row["split"] = split
                writer.writerow(output_row)

                counts[(split, row["label"])] += 1

    # Confirm that source images never cross split boundaries.
    assert train_images.isdisjoint(val_images)
    assert train_images.isdisjoint(test_images)
    assert val_images.isdisjoint(test_images)

    print("\n===== DATASET SPLIT SUMMARY =====")
    print(f"Unique source images: {total_images}")

    for split in ("train", "validation", "test"):
        print(
            f"{split.title():10} | "
            f"source images: {image_counts[split]:3}"
        )
        for label in ("With Helmet", "Without Helmet"):
            print(f"  {label}: {counts[(split, label)]}")

    print(f"\nManifest saved to: {OUTPUT_PATH}")
    print(f"Random seed: {SEED}")
    print("Source-image leakage check: PASSED")


if __name__ == "__main__":
    main()
