
import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/videos/traffic_video.mp4")
OUTPUT_DIR = Path("data/processed/hard_examples")
FRAME_INTERVAL_SECONDS = 2

def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Could not determine video FPS.")
        cap.release()
        return

    interval = max(1, int(fps * FRAME_INTERVAL_SECONDS))
    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:
            output_path = OUTPUT_DIR / f"frame_{frame_number:07d}.jpg"

            if cv2.imwrite(str(output_path), frame):
                saved_count += 1

        frame_number += 1

    cap.release()

    print(f"Frames processed: {frame_number}")
    print(f"Frames saved: {saved_count}")
    print(f"Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
