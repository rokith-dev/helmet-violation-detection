
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

# This threshold is only a starting point for testing.
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
    """Classify a person crop using the trained VGG16 model."""

    if crop.size == 0:
        return "Unknown", 0.0

    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, IMAGE_SIZE).numpy()
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    probability_without_helmet = float(
        classifier.predict(image, verbose=0)[0][0]
    )

    if probability_without_helmet >= HELMET_THRESHOLD:
        label = "Possible No Helmet"
        score = probability_without_helmet
    else:
        label = "Possible With Helmet"
        score = 1.0 - probability_without_helmet

    return label, score


def main():
    parser = argparse.ArgumentParser(
        description="Detect people and test VGG16 helmet classification."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video file path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO detection confidence threshold.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    if not VGG16_MODEL.is_file():
        raise FileNotFoundError(
            f"Trained VGG16 model not found: {VGG16_MODEL}"
        )

    print("Loading YOLO model...")
    detector = YOLO(YOLO_MODEL)

    print("Loading trained VGG16 classifier...")
    classifier = tf.keras.models.load_model(VGG16_MODEL)

    capture = open_video(args.source)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAX_WIDTH, MAX_HEIGHT)

    print("Video opened. Press Q or Esc to quit.")
    print("VGG16 predictions on full person crops are experimental.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video ended or frame could not be read.")
                break

            results = detector.predict(
                source=frame,
                classes=TARGET_CLASSES,
                conf=args.confidence,
                verbose=False,
            )

            annotated = frame.copy()
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())

                    # Only classify person detections.
                    if class_id != PERSON_CLASS:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    height, width = frame.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height))

                    crop = frame[y1:y2, x1:x2]
                    label, score = classify_person_crop(
                        classifier, crop
                    )

                    # Orange indicates an experimental prediction.
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 165, 255),
                        2,
                    )

                    text = f"{label}: {score:.0%}"
                    text_y = max(y1 - 10, 25)

                    cv2.putText(
                        annotated,
                        text,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(WINDOW_NAME, resize_frame(annotated))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Video resources released.")


if __name__ == "__main__":
    main()
