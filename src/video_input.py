
from pathlib import Path
import argparse

import cv2


def open_video(source):
    """
    Open a video file or webcam.

    Use source='0' for the default webcam,
    or provide the path to a recorded video.
    """

    if source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}\n"
            "Check the video path or webcam connection."
        )

    return capture


def main():
    parser = argparse.ArgumentParser(
        description="Read video frames for helmet detection."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or path to a video file.",
    )

    args = parser.parse_args()

    capture = open_video(args.source)

    print("Video opened successfully.")
    print("Press Q in the video window to quit.")

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("Video ended or a frame could not be read.")
                break

            # Display the current video frame.
            cv2.imshow(
                "Helmet Violation Detection - Video Input",
                frame,
            )

            # Press Q to exit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Video resources released.")


if __name__ == "__main__":
    main()
