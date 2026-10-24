
from pathlib import Path
import shutil
import random
from collections import Counter

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(
    "data/external/edgevision/EdgeVision Dataset/"
    "EdgeVision-Dataset"
)

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels" / "yolo"

OUTPUT_DIR = Path("data/processed/edgevision_helmet")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Original EdgeVision IDs:
# 1 = NoHelmet -> project ID 1 (Without Helmet)
# 2 = Helmet   -> project ID 0 (With Helmet)
CLASS_MAPPING = {
    1: 1,
    2: 0,
}


def valid_box(x, y, w, h):
    if not all(0 <= value <= 1 for value in (x, y, w, h)):
        return False

    if w <= 0 or h <= 0:
        return False

    # Reject boxes extending beyond the image.
    if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
        return False

    if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
        return False

    return True


def main():
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(
            "EdgeVision images or YOLO label directory was not found."
        )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    # Only use images that have a matching label file.
    pairs = [
        (image, LABEL_DIR / f"{image.stem}.txt")
        for image in images
        if (LABEL_DIR / f"{image.stem}.txt").is_file()
    ]

    if not pairs:
        raise RuntimeError("No matching image-label pairs found.")

    # Deterministic split.
    random.Random(SEED).shuffle(pairs)

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    counts = {
        split: Counter()
        for split in splits
    }

    invalid_skipped = 0
    empty_after_filter = 0

    for split, split_pairs in splits.items():
        image_out = OUTPUT_DIR / "images" / split
        label_out = OUTPUT_DIR / "labels" / split

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in split_pairs:
            converted = []

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                parts = line.split()

                if not parts:
                    continue

                try:
                    if len(parts) != 5:
                        invalid_skipped += 1
                        continue

                    original_id = int(parts[0])

                    # Ignore the BikeWithRider class.
                    if original_id not in CLASS_MAPPING:
                        continue

                    x, y, w, h = map(float, parts[1:])

                    if not valid_box(x, y, w, h):
                        invalid_skipped += 1
                        continue

                    new_id = CLASS_MAPPING[original_id]
                    converted.append(
                        f"{new_id} {x:.6f} {y:.6f} "
                        f"{w:.6f} {h:.6f}"
                    )
                    counts[split][new_id] += 1

                except (ValueError, OverflowError):
                    invalid_skipped += 1

            # Preserve the image even if no selected helmet object remains.
            # This represents a background image for the selected classes.
            if not converted:
                empty_after_filter += 1

            shutil.copy2(
                image_path,
                image_out / image_path.name
            )

            (label_out / f"{image_path.stem}.txt").write_text(
                "\n".join(converted) + ("\n" if converted else ""),
                encoding="utf-8"
            )

    yaml_text = (
        f"path: {OUTPUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: With Helmet\n"
        "  1: Without Helmet\n"
    )

    (OUTPUT_DIR / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8"
    )

    print("\nDataset preparation completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Matched source images: {total}")
    print(f"Invalid annotation rows skipped: {invalid_skipped}")
    print(f"Images with no selected objects: {empty_after_filter}")

    for split, split_pairs in splits.items():
        print(f"\n{split.upper()}: {len(split_pairs)} images")
        print(f"  With Helmet: {counts[split][0]}")
        print(f"  Without Helmet: {counts[split][1]}")

    print("\nConfiguration saved to data.yaml")
    print("Original dataset and model were not modified.")


if __name__ == "__main__":
    main()
