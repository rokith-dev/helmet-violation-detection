
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.vgg16 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL = "yolo11n.pt"
VGG16_MODEL = (
    PROJECT_DIR / "models" / "vgg16_helmet_classifier.keras"
)

WINDOW_NAME = "Helmet Violation Detection"

MAX_WIDTH = 960
MAX_HEIGHT = 540
IMAGE_SIZE = (224, 224)

# COCO class IDs: person=0, motorcycle=3
PERSON_CLASS = 0
TARGET_CLASSES = [0, 3]

# Process one frame out of every three.
FRAME_SKIP = 3

# Smaller inference size can improve CPU speed.
YOLO_IMAGE_SIZE = 480

# Experimental threshold only.
HELMET_THRESHOLD = 0.5


def resize_frame(frame):
    height, width = frame.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0,
    )

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def open_video(source):
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


def classify_person_crop(classifier, crop):
    """Classify a full-person crop; this is experimental."""

    if crop.size == 0:
        return "Unknown", 0.0

    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, IMAGE_SIZE).numpy()
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    probability_without = float(
        classifier.predict(image, verbose=0)[0][0]
    )

    if probability_without >= HELMET_THRESHOLD:
        return "Possible No Helmet", probability_without

    return "Possible With Helmet", 1.0 - probability_without


def draw_label(frame, text, x, y, color):
    """Draw readable text on a solid background."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    y = max(y, text_height + baseline + 6)

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + 4),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experimental helmet classification on traffic video."
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
        help="YOLO confidence threshold.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    if not VGG16_MODEL.is_file():
        raise FileNotFoundError(
            f"VGG16 model not found: {VGG16_MODEL}"
        )

    print("Loading YOLO...")
    detector = YOLO(YOLO_MODEL)

    print("Loading VGG16...")
    classifier = tf.keras.models.load_model(VGG16_MODEL)

    capture = open_video(args.source)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    # We skip frames for faster CPU inference, so the display
    # delay is reduced proportionally.
    display_delay = max(1, int(1000 / fps / FRAME_SKIP))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAX_WIDTH, MAX_HEIGHT)

    print("Video opened.")
    print(f"Detection runs every {FRAME_SKIP} frames.")
    print("Orange boxes = people; blue boxes = motorcycles.")
    print("Helmet labels are experimental, not confirmed violations.")
    print("Press Q or Esc to quit.")

    frame_count = 0
    processed_count = 0
    annotated_frame = None

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("Video ended or a frame could not be read.")
                break

            frame_count += 1

            # Skip expensive inference on intermediate frames.
            if (frame_count - 1) % FRAME_SKIP == 0:
                results = detector.predict(
                    source=frame,
                    classes=TARGET_CLASSES,
                    conf=args.confidence,
                    imgsz=YOLO_IMAGE_SIZE,
                    verbose=False,
                )

                annotated_frame = frame.copy()
                boxes = results[0].boxes

                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].item())

                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        height, width = frame.shape[:2]
                        x1 = max(0, min(x1, width - 1))
                        x2 = max(0, min(x2, width))
                        y1 = max(0, min(y1, height - 1))
                        y2 = max(0, min(y2, height))

                        if class_id == PERSON_CLASS:
                            crop = frame[y1:y2, x1:x2]

                            label, score = classify_person_crop(
                                classifier, crop
                            )

                            color = (0, 165, 255)
                            text = f"TEST ONLY: {label} {score:.0%}"

                        else:
                            color = (255, 120, 0)
                            text = "Motorcycle"

                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        draw_label(
                            annotated_frame,
                            text,
                            x1,
                            y1 - 8,
                            color,
                        )

                draw_label(
                    annotated_frame,
                    "EXPERIMENTAL - NOT VERIFIED",
                    12,
                    28,
                    (0, 0, 180),
                )

                processed_count += 1

            # Keep displaying the latest processed frame.
            if annotated_frame is not None:
                cv2.imshow(
                    WINDOW_NAME,
                    resize_frame(annotated_frame),
                )

            key = cv2.waitKey(display_delay) & 0xFF

            if key in (ord("q"), 27):
                print("Stopped by user.")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print("Frames read:", frame_count)
        print("Frames analyzed:", processed_count)
        print("Video resources released.")


if __name__ == "__main__":
    main()
