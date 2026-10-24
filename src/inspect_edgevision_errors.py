
from pathlib import Path

ROOT = Path(
    "data/external/edgevision/EdgeVision Dataset/"
    "EdgeVision-Dataset"
)
LABEL_DIR = ROOT / "labels" / "yolo"

def main():
    classes_file = LABEL_DIR / "classes.txt"

    if classes_file.exists():
        print("CLASS NAMES:")
        for index, name in enumerate(
            classes_file.read_text(encoding="utf-8").splitlines()
        ):
            print(f"{index}: {name}")

    print("\nINVALID ANNOTATION EXAMPLES:")

    count = 0

    for path in sorted(LABEL_DIR.glob("Image_*.txt")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            parts = line.split()

            try:
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                invalid = (
                    class_id not in (0, 1, 2)
                    or not all(0 <= v <= 1 for v in (x, y, w, h))
                    or w <= 0
                    or h <= 0
                    or x - w / 2 < -1e-6
                    or x + w / 2 > 1 + 1e-6
                    or y - h / 2 < -1e-6
                    or y + h / 2 > 1 + 1e-6
                )

                if invalid:
                    print(f"{path.name}:{line_number}: {line}")
                    count += 1

                    if count >= 15:
                        print("\nShowing first 15 invalid examples.")
                        return

            except ValueError:
                print(f"{path.name}:{line_number}: {line}")
                count += 1

                if count >= 15:
                    return

    if count == 0:
        print("No invalid annotations found.")

if __name__ == "__main__":
    main()

