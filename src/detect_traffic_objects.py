
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.vgg16 import preprocess_input


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL = "yolo11n.pt"

VGG16_MODEL = (
    PROJECT_DIR
    / "models"
    / "vgg16_helmet_classifier.keras"
)

WINDOW_NAME = "Helmet Violation Detection"

# Maximum preview dimensions
MAX_WIDTH = 1280
MAX_HEIGHT = 720

IMAGE_SIZE = (224, 224)

# COCO class IDs: person=0, motorcycle=3
PERSON_CLASS = 0
TARGET_CLASSES = [0, 3]

# Analyze one out of every three frames
FRAME_SKIP = 3

# Smaller inference size helps CPU performance
YOLO_IMAGE_SIZE = 480

# Classification threshold; experimental only
HELMET_THRESHOLD = 0.5


# --------------------------------------------------
# 2. RESIZE VIDEO FRAMES
# --------------------------------------------------

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


# --------------------------------------------------
# 3. OPEN WEBCAM OR VIDEO
# --------------------------------------------------

def open_video(source):
    """Open a webcam index or a video file."""

    if source.isdigit():
        video_source = int(source)

    else:
        video_path = Path(source)

        if not video_path.is_absolute():
            video_path = PROJECT_DIR / video_path

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path.resolve()}"
            )

        video_source = str(video_path)

    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not open video source: {video_source}"
        )

    return capture


# --------------------------------------------------
# 4. VGG16 HELMET CLASSIFICATION
# --------------------------------------------------

def classify_person_crop(classifier, crop):
    """
    Classify a detected person's crop.

    NOTE:
    The model was trained on object crops, not specifically
    head crops. Treat these predictions as experimental.
    """

    if crop.size == 0:
        return "Unknown", 0.0

    # OpenCV uses BGR; VGG16 preprocessing expects RGB input.
    image = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    ).numpy()

    image = image.astype("float32")

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0,
    )

    # Output is the predicted probability for class 1:
    # Without Helmet.
    probability_without = float(
        classifier.predict(
            image,
            verbose=0,
        )[0][0]
    )

    if probability_without >= HELMET_THRESHOLD:
        return "Possible No Helmet", probability_without

    return (
        "Possible With Helmet",
        1.0 - probability_without,
    )


# --------------------------------------------------
# 5. DRAW LARGE, READABLE LABELS
# --------------------------------------------------

def draw_label(frame, text, x, y, color):
    """Draw large white text on a solid background."""

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.9
    thickness = 2
    padding = 10

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    frame_height, frame_width = frame.shape[:2]

    # Keep label within the frame boundaries.
    x = max(
        0,
        min(
            x,
            frame_width - text_width - padding * 2,
        ),
    )

    # If the label cannot fit above the box, put it lower.
    if y - text_height - baseline - padding < 0:
        y = text_height + baseline + padding

    top = max(
        0,
        y - text_height - baseline - padding,
    )

    bottom = min(
        frame_height - 1,
        y + padding // 2,
    )

    right = min(
        frame_width - 1,
        x + text_width + padding * 2,
    )

    cv2.rectangle(
        frame,
        (x, top),
        (right, bottom),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + padding, y - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# --------------------------------------------------
# 6. MAIN VIDEO DETECTION PIPELINE
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect people and motorcycles and test "
            "the VGG16 helmet classifier."
        )
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
        help="YOLO detection confidence threshold.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error(
            "--confidence must be between 0 and 1."
        )

    if not VGG16_MODEL.is_file():
        raise FileNotFoundError(
            f"Trained VGG16 model not found: {VGG16_MODEL}"
        )

    # Load both models once, before processing video frames.
    print("Loading YOLO model...")
    detector = YOLO(YOLO_MODEL)

    print("Loading VGG16 classifier...")
    classifier = tf.keras.models.load_model(
        VGG16_MODEL
    )

    capture = open_video(args.source)

    # Create a resizable window.
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        MAX_WIDTH,
        MAX_HEIGHT,
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    # This controls display responsiveness. Because frames are
    # skipped, playback timing is approximate rather than exact.
    display_delay = max(
        1,
        int(1000 / fps / FRAME_SKIP),
    )

    print("\nVideo opened successfully.")
    print("Preview size: up to 1280 x 720")
    print(f"Detection runs every {FRAME_SKIP} frames.")
    print("Orange box: person and experimental VGG16 prediction")
    print("Blue box: motorcycle")
    print("Press Q or Esc to quit.")
    print(
        "\nWARNING: Helmet labels are experimental, "
        "not confirmed violations.\n"
    )

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

            # Run expensive inference on selected frames only.
            should_process = (
                (frame_count - 1) % FRAME_SKIP == 0
            )

            if should_process:
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

                        class_id = int(
                            box.cls[0].item()
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist(),
                        )

                        frame_height, frame_width = (
                            frame.shape[:2]
                        )

                        # Clip coordinates to the frame.
                        x1 = max(
                            0,
                            min(x1, frame_width - 1),
                        )

                        x2 = max(
                            0,
                            min(x2, frame_width),
                        )

                        y1 = max(
                            0,
                            min(y1, frame_height - 1),
                        )

                        y2 = max(
                            0,
                            min(y2, frame_height),
                        )

                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Person detection
                        if class_id == PERSON_CLASS:

                            crop = frame[
                                y1:y2,
                                x1:x2,
                            ]

                            label, score = classify_person_crop(
                                classifier,
                                crop,
                            )

                            color = (0, 140, 255)

                            # Shorter text is easier to read.
                            text = f"{label} {score:.0%}"

                        # Motorcycle detection
                        else:
                            color = (255, 120, 0)
                            text = "Motorcycle"

                        # Draw object bounding box.
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            3,
                        )

                        # Draw large, solid-background label.
                        draw_label(
                            annotated_frame,
                            text,
                            x1,
                            y1 - 8,
                            color,
                        )

                # Permanent warning banner.
                draw_label(
                    annotated_frame,
                    "EXPERIMENTAL - VERIFY RESULTS",
                    12,
                    45,
                    (0, 0, 180),
                )

                processed_count += 1

            # Show the most recently processed frame.
            if annotated_frame is not None:
                display_frame = resize_frame(
                    annotated_frame
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

            key = cv2.waitKey(
                display_delay
            ) & 0xFF

            if key in (ord("q"), 27):
                print("Stopped by user.")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print("\n========== VIDEO SUMMARY ==========")
        print("Frames read:", frame_count)
        print("Frames analyzed:", processed_count)
        print("Video resources released.")


if __name__ == "__main__":
    main()
