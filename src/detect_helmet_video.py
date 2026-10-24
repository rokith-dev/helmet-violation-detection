
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "yolo_training"
    / "helmet_yolo11n"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "videos"
    / "traffic_video.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"
OUTPUT_VIDEO = OUTPUT_DIR / "helmet_detection_video.mp4"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 416

PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 650

# Label appearance
FONT_SCALE = 1.0
FONT_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_PADDING = 8


def draw_detections(frame, result, model):
    """Draw helmet detections with large, readable labels."""

    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]

    for detection in result.boxes:
        x1, y1, x2, y2 = map(
            int, detection.xyxy[0].tolist()
        )

        # Keep coordinates inside the frame
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        class_id = int(detection.cls[0].item())
        confidence = float(detection.conf[0].item())

        class_name = model.names[class_id]
        label = f"{class_name} {confidence:.0%}"

        # Blue for With Helmet, red for Without Helmet
        if class_id == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS,
        )

        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            FONT_SCALE,
            FONT_THICKNESS,
        )

        label_width = text_width + LABEL_PADDING * 2
        label_height = (
            text_height + baseline + LABEL_PADDING * 2
        )

        # Keep label inside the image boundaries
        label_x = min(
            x1,
            max(0, frame_width - label_width),
        )

        # Prefer placing label above the box.
        # If there is insufficient space, place it inside.
        if y1 >= label_height:
            label_y = y1 - label_height
        else:
            label_y = y1

        label_bottom = min(
            frame_height - 1,
            label_y + label_height,
        )
        label_right = min(
            frame_width - 1,
            label_x + label_width,
        )

        # Solid background for readable text
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_right, label_bottom),
            color,
            -1,
        )

        # Draw large white label
        text_x = label_x + LABEL_PADDING
        text_y = min(
            frame_height - 1,
            label_y + LABEL_PADDING + text_height,
        )

        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


def main():
    # ==========================================
    # CHECK FILES
    # ==========================================

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Input video not found:\n{VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # LOAD MODEL AND VIDEO
    # ==========================================

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 45)
    print("HELMET VIOLATION DETECTION")
    print("=" * 45)
    print("Model:", MODEL_PATH)
    print("Input:", VIDEO_PATH)
    print(
        f"Resolution: {original_width} x {original_height}"
    )
    print("Total frames:", total_frames)
    print("Classes:", model.names)

    # ==========================================
    # CREATE OUTPUT VIDEO
    # ==========================================

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (original_width, original_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    window_name = "Helmet Violation Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        PREVIEW_WIDTH,
        PREVIEW_HEIGHT,
    )

    frame_number = 0
    processed_frames = 0

    # ==========================================
    # PROCESS VIDEO
    # ==========================================

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Detect helmet classes
            result = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )[0]

            # Draw custom large labels
            annotated_frame = draw_detections(
                frame,
                result,
                model,
            )

            processed_frames += 1

            # Add frame counter
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number}/{total_frames}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Save full-resolution annotated frame
            writer.write(annotated_frame)

            # Resize preview without cropping or distortion
            frame_height, frame_width = annotated_frame.shape[:2]

            scale = min(
                PREVIEW_WIDTH / frame_width,
                PREVIEW_HEIGHT / frame_height,
                1.0,
            )

            preview_width = max(
                1, int(frame_width * scale)
            )
            preview_height = max(
                1, int(frame_height * scale)
            )

            preview = cv2.resize(
                annotated_frame,
                (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(window_name, preview)

            # Press Q or Esc to stop
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("Detection stopped by user.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ==========================================
    # SUMMARY
    # ==========================================

    print("\n" + "=" * 45)
    print("VIDEO PROCESSING COMPLETED")
    print("=" * 45)
    print("Frames read:", frame_number)
    print("Frames processed:", processed_frames)
    print("Output video:", OUTPUT_VIDEO)
    print(
        f"Output resolution: {original_width} x {original_height}"
    )


if __name__ == "__main__":
    main()
