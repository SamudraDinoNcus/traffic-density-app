import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import time
from ultralytics import YOLO
from collections import deque, Counter


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_models():

    BASE_DIR = os.path.dirname(__file__)
    yolo_path = os.path.join(BASE_DIR, "yolov8n.pt")
    ml_path = os.path.join(BASE_DIR, "traffic_model.pkl")
    
    yolo_model = YOLO(yolo_path)
    model_ml = joblib.load(ml_path)
    
    return yolo_model, model_ml

try:
    yolo_model, model_ml = load_models()
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# ==============================
# PARAMETER
# ==============================
vehicle_classes = [2, 3, 5, 7]  # car, motor, bus, truck

# smoothing
count_buffer = deque(maxlen=10)
prediction_buffer = deque(maxlen=15)


# ==============================
# STREAMLIT UI
# ==============================
st.title("🚦 Traffic Density Monitoring System")
st.write("Deteksi kepadatan lalu lintas berbasis Computer Vision dan Machine Learning")

if "stable_label" not in st.session_state:
    st.session_state.stable_label = None

st.info("Upload video untuk mulai deteksi")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

st.markdown("### Atau coba sample video")
use_sample = st.button("Gunakan contoh video")

if uploaded_file is not None or use_sample:
     
     if uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("Video terlalu besar (max 50MB)")
            st.stop()
     
     with st.spinner("Processing video..."):

        # simpan file sementara (lebih aman)
        if use_sample:
            video_path = "sample.mp4"
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            st.error("Gagal membuka video")
            st.stop()

        stframe = st.empty()
        info_box = st.empty()

        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))

            frame_index += 1

            if total_frames > 0:
                progress.progress(min(frame_index / total_frames, 1.0))

            # skip frame biar ringan
            if frame_index % 3 != 0:
                continue

            h, w = frame.shape[:2]

            # ==============================
            # ROI
            # ==============================
            roi_points = np.array([
                [int(0.20 * w), int(0.25 * h)],
                [int(0.85 * w), int(0.25 * h)],
                [int(0.95 * w), int(0.95 * h)],
                [int(0.05 * w), int(0.95 * h)]
            ])

            # ==============================
            # DETEKSI YOLO
            # ==============================
            results = yolo_model(frame)[0]

            count = 0
            centroids = []

            for box in results.boxes:
                cls = int(box.cls[0])

                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    points = [
                        (x1, y1), (x2, y1),
                        (x1, y2), (x2, y2)
                    ]

                    inside = any(
                        cv2.pointPolygonTest(roi_points, pt, False) >= 0
                        for pt in points
                    )

                    if inside:
                        count += 1

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        centroids.append((cx, cy))

                        # gambar bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # ==============================
            # FEATURE ENGINEERING
            # ==============================
            count_buffer.append(count)
            count_smooth = np.mean(count_buffer)

            density = count / (w * h)

            if len(count_buffer) > 1:
                delta = count_buffer[-1] - count_buffer[-2]
            else:
                delta = 0

            var = np.var(count_buffer)

            flow = np.mean(np.abs(np.diff(count_buffer))) if len(count_buffer) > 1 else 0

            features = pd.DataFrame([{
                "count_smooth": count_smooth,
                "density": density,
                "delta": delta,
                "var": var,
                "flow": flow
            }])

            features = features[model_ml.feature_names_in_]

            # ==============================
            # PREDIKSI ML
            # ==============================
            pred = model_ml.predict(features)[0]

            prediction_buffer.append(pred)

            # ==============================
            # ANTI FLICKER LOGIC
            # ==============================
            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]

                if st.session_state.stable_label is None:
                    st.session_state.stable_label = most_common
                else:
                    if most_common != st.session_state.stable_label:
                        change_count = list(prediction_buffer).count(most_common)

                        if change_count > 10:
                            st.session_state.stable_label = most_common

            # ==============================
            # VISUALISASI
            # ==============================
            cv2.polylines(frame, [roi_points], True, (0,255,0), 2)

            cv2.putText(frame, f"Count: {count}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Density: {st.session_state.stable_label}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # tampilkan
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)

            # info tambahan
            confidence = Counter(prediction_buffer).most_common(1)[0][1] / len(prediction_buffer) if len(prediction_buffer) > 0 else 0

            info_box.write({
                "Raw Count": count,
                "Smoothed Count": round(count_smooth, 2),
                "Prediction (stable)": st.session_state.stable_label,
                "Confidence": round(confidence, 2)
            })

            time.sleep(0.01)

        cap.release()
        if not use_sample:
            os.remove(tfile.name)