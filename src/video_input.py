
import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "Helmet Violation Detection - Video Input"

# Maximum preview size in pixels
MAX_WIDTH = 960
MAX_HEIGHT = 540


def open_video(source):
    """Open a webcam or a recorded video file."""

    # Convert numeric sources such as "0" to integer webcam indices.
    if source.isdigit():
        source = int(source)
    else:
        source_path = Path(source)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {source_path.resolve()}"
            )

        source = str(source_path)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    return capture


def resize_frame(frame):
    """Resize a frame to fit the preview while preserving aspect ratio."""

    height, width = frame.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0,
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Display webcam or traffic video frames."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or path to a video file.",
    )

    args = parser.parse_args()

    capture = open_video(args.source)

    # Create a normal, resizable window instead of fullscreen.
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
    print("Preview is limited to 960 x 540 pixels.")
    print("Press Q in the video window to quit.")

    frame_count = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("Video ended or a frame could not be read.")
                break

            display_frame = resize_frame(frame)

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            frame_count += 1

            # Q or Escape closes the preview.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("Preview stopped by user.")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print(f"Frames displayed: {frame_count}")
        print("Video resources released.")


if __name__ == "__main__":
    main()
