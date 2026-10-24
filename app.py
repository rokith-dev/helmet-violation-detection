from pathlib import Path
import tempfile

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

# New EdgeVision-trained model.
# This matches the actual Ultralytics output path from the training run.
MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "outputs"
    / "yolo_training"
    / "helmet_yolo11n_edgevision_fast"
    / "weights"
    / "best.pt"
)

# Keep the original model as a backup; this app uses MODEL_PATH above.
BACKUP_MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "yolo_training"
    / "helmet_yolo11n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet",
}


# =====================================================
# STREAMLIT PAGE
# =====================================================

st.set_page_config(
    page_title="Helmet Violation Detection",
    page_icon="🪖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🪖 Helmet Violation Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered helmet detection for traffic camera feeds'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload a recorded traffic video to detect helmet-related objects. "
    "The model produces predictions, not confirmed legal violations."
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model(model_path: str):
    return YOLO(str(model_path))


# =====================================================
# DRAW LARGE, READABLE DETECTION LABELS
# =====================================================

def draw_detections(frame, result):
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]

    if result.boxes is None:
        return annotated

    for detection in result.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height - 1))

        class_id = int(detection.cls[0].item())
        confidence = float(detection.conf[0].item())

        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        label = f"{class_name} {confidence:.0%}"

        # OpenCV uses BGR: green for With Helmet, red for Without Helmet.
        color = (40, 180, 40) if class_id == 0 else (40, 40, 230)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        padding = 8

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_width = text_width + padding * 2
        label_height = text_height + baseline + padding * 2

        label_x = min(x1, max(0, frame_width - label_width))
        label_y = y1 - label_height if y1 >= label_height else y1

        label_right = min(frame_width - 1, label_x + label_width)
        label_bottom = min(frame_height - 1, label_y + label_height)

        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_right, label_bottom),
            color,
            -1,
        )

        text_x = label_x + padding
        text_y = min(
            frame_height - 1,
            label_y + padding + text_height,
        )

        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# =====================================================
# VIDEO PROCESSING
# =====================================================

def process_video(
    input_path,
    output_path,
    model,
    confidence,
    image_size,
    frame_stride,
):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frames_read = 0
    frames_processed = 0

    # These are detections across analyzed frames, not unique riders.
    with_helmet = 0
    without_helmet = 0
    confidence_values = []

    progress = st.progress(0, text="Starting detection...")
    status = st.empty()
    preview = st.empty()
    last_annotated = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frames_read += 1

            if (frames_read - 1) % frame_stride == 0:
                result = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )[0]

                annotated = draw_detections(frame, result)

                if result.boxes is not None:
                    for detection in result.boxes:
                        class_id = int(detection.cls[0].item())
                        score = float(detection.conf[0].item())

                        if class_id == 0:
                            with_helmet += 1
                        elif class_id == 1:
                            without_helmet += 1

                        confidence_values.append(score)

                last_annotated = annotated
                frames_processed += 1

            else:
                # Reuse the most recent annotated frame between inference
                # frames when frame_stride is greater than one.
                if last_annotated is not None:
                    annotated = last_annotated.copy()
                else:
                    annotated = frame.copy()

            cv2.putText(
                annotated,
                f"Frame: {frames_read}/{total_frames}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            if (
                frames_read == 1
                or frames_read % max(1, int(fps)) == 0
                or frames_read == total_frames
            ):
                preview.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )

            if total_frames > 0:
                fraction = min(frames_read / total_frames, 1.0)
                progress.progress(
                    fraction,
                    text=f"Analyzing frame {frames_read}/{total_frames}",
                )

            status.caption(
                f"Frames analyzed: {frames_processed} | "
                f"Frames read: {frames_read}"
            )

    finally:
        cap.release()
        writer.release()

    progress.progress(1.0, text="Detection completed.")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Output video was not created.")

    return {
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "with_helmet": with_helmet,
        "without_helmet": without_helmet,
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        ),
        "fps": fps,
        "width": width,
        "height": height,
    }


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.10,
        step=0.05,
        help=(
            "Lower values may detect more objects but can "
            "increase false positives."
        ),
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640],
        value=640,
    )

    frame_stride = st.select_slider(
        "Process every Nth frame",
        options=[1, 2, 3, 5],
        value=1,
        help=(
            "1 analyzes every frame. Higher values are faster "
            "but can miss objects between analyzed frames."
        ),
    )

    st.divider()
    st.caption("Model: EdgeVision-trained YOLO11n")
    st.caption("Classes: With Helmet / Without Helmet")
    st.caption(f"Input size: {image_size}px")

    with st.expander("Model details"):
        st.code(str(MODEL_PATH), language="text")
        st.write("Original model backup:")
        st.code(str(BACKUP_MODEL_PATH), language="text")
        st.write(
            "Backup checkpoint exists:",
            BACKUP_MODEL_PATH.is_file(),
        )


# =====================================================
# CHECK AND LOAD MODEL
# =====================================================

if not MODEL_PATH.is_file():
    st.error(
        "The EdgeVision-trained model was not found at:\n\n"
        f"`{MODEL_PATH}`\n\n"
        "Check that the training run created `weights/best.pt` "
        "at this location."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# =====================================================
# VIDEO UPLOAD
# =====================================================

st.subheader("📤 Upload Traffic Video")

uploaded_file = st.file_uploader(
    "Choose a traffic video",
    type=["mp4", "avi", "mov", "mkv"],
    help=(
        "Use clear footage with visible riders. "
        "Actual traffic-camera footage is best for evaluation."
    ),
)

if uploaded_file is None:
    st.info("Upload a video, then click Start Helmet Detection.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("### 1. Upload\nChoose a recorded video.")
    col2.markdown("### 2. Detect\nRun your trained YOLO model.")
    col3.markdown("### 3. Review\nView and download results.")

    st.stop()

st.video(uploaded_file)


# =====================================================
# RUN DETECTION
# =====================================================

if st.button(
    "🚀 Start Helmet Detection",
    type="primary",
    width="stretch",
):
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    output_path = OUTPUT_DIR / "dashboard_helmet_detection.mp4"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / f"input_video{suffix}"
            input_path.write_bytes(uploaded_file.getvalue())

            st.subheader("🔍 Detection in Progress")

            stats = process_video(
                input_path=input_path,
                output_path=output_path,
                model=model,
                confidence=confidence,
                image_size=image_size,
                frame_stride=frame_stride,
            )

        st.session_state["helmet_output_path"] = str(output_path)
        st.session_state["helmet_stats"] = stats
        st.session_state["helmet_output_name"] = (
            f"{Path(uploaded_file.name).stem}_detected.mp4"
        )

    except Exception as exc:
        st.error(f"Video processing failed: {exc}")


# =====================================================
# DISPLAY RESULTS
# =====================================================

saved_output = st.session_state.get("helmet_output_path")
stats = st.session_state.get("helmet_stats")

if saved_output and stats:
    output_path = Path(saved_output)

    if output_path.is_file() and output_path.stat().st_size > 0:
        st.divider()
        st.subheader("📊 Detection Statistics")

        st.caption(
            "Counts represent detections across analyzed frames, "
            "not unique riders or confirmed legal violations."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("With Helmet", f"{stats['with_helmet']:,}")
        col2.metric("Without Helmet", f"{stats['without_helmet']:,}")
        col3.metric("Frames Read", f"{stats['frames_read']:,}")
        col4.metric("Frames Analyzed", f"{stats['frames_processed']:,}")

        st.subheader("📈 Detection Distribution")

        chart_data = pd.DataFrame(
            {
                "Class": ["With Helmet", "Without Helmet"],
                "Detections": [
                    stats["with_helmet"],
                    stats["without_helmet"],
                ],
            }
        )

        st.bar_chart(chart_data, x="Class", y="Detections")

        st.subheader("🎬 Annotated Output Video")
        st.video(str(output_path))

        st.caption(
            f"Resolution: {stats['width']} × {stats['height']} | "
            f"Input FPS: {stats['fps']:.2f} | "
            f"Mean confidence: {stats['mean_confidence']:.1%}"
        )

        with output_path.open("rb") as file:
            st.download_button(
                "⬇️ Download Annotated Video",
                data=file,
                file_name=st.session_state.get(
                    "helmet_output_name",
                    "helmet_detection.mp4",
                ),
                mime="video/mp4",
                width="stretch",
            )

        st.warning(
            "Review predictions before treating them as violations. "
            "The same rider may be counted in multiple frames, and "
            "incorrect detections are possible."
        )
