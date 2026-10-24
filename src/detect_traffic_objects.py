
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolo11n.pt"

WINDOW_NAME = "Traffic Object Detection"

MAX_WIDTH = 960
MAX_HEIGHT = 540

# COCO classes:
# 0 = person
# 3 = motorcycle
TARGET_CLASSES = [0, 3]


def resize_frame(frame):
    """Resize a frame while preserving its aspect ratio."""
    height, width = frame.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0,
    )

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def open_video(source):
    """Open a webcam or a video file."""
    if source.isdigit():
        video_source = int(source)
    else:
        video_path = Path(source)

        if not video_path.is_absolute():
            video_path = PROJECT_DIR / video_path

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        video_source = str(video_path)

    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open video source: {video_source}"
        )

    return capture


def main():
    parser = argparse.ArgumentParser(
        description="Detect people and motorcycles in traffic video."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or traffic video path.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence, from 0 to 1.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    capture = open_video(args.source)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        MAX_WIDTH,
        MAX_HEIGHT,
    )

    print("Video opened successfully.")
    print("Detecting people and motorcycles.")
    print("Press Q or Esc in the video window to quit.")

    frame_count = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("Video ended or frame could not be read.")
                break

            results = model.predict(
                source=frame,
                classes=TARGET_CLASSES,
                conf=args.confidence,
                verbose=False,
            )

            # Draw detections on the original-size frame.
            annotated_frame = results[0].plot()

            display_frame = resize_frame(annotated_frame)

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("Stopped by user.")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print("Frames processed:", frame_count)
        print("Video resources released.")


if __name__ == "__main__":
    main()
