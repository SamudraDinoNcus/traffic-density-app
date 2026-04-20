import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from collections import deque, Counter

# ==============================
# LOAD MODEL
# ==============================
yolo_model = YOLO("yolov8n.pt")  # model deteksi
model_ml = joblib.load("traffic_model.pkl")  # model klasifikasi

# ==============================
# PARAMETER
# ==============================
vehicle_classes = [2, 3, 5, 7]  # car, motor, bus, truck

# smoothing
count_buffer = deque(maxlen=10)
prediction_buffer = deque(maxlen=15)

stable_label = None

# ==============================
# STREAMLIT UI
# ==============================
st.title("🚦 Traffic Density Detection (Stable Version)")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:
    # simpan sementara
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    stframe = st.empty()
    info_box = st.empty()

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

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

            if stable_label is None:
                stable_label = most_common
            else:
                if most_common != stable_label:
                    change_count = list(prediction_buffer).count(most_common)

                    if change_count > 10:  # threshold stabil
                        stable_label = most_common

        # ==============================
        # VISUALISASI
        # ==============================
        cv2.polylines(frame, [roi_points], True, (0,255,0), 2)

        cv2.putText(frame, f"Count: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Density: {stable_label}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # tampilkan
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb)

        # info tambahan
        confidence = Counter(prediction_buffer).most_common(1)[0][1] / len(prediction_buffer) if len(prediction_buffer) > 0 else 0

        info_box.write({
            "Raw Count": count,
            "Smoothed Count": round(count_smooth, 2),
            "Prediction (stable)": stable_label,
            "Confidence": round(confidence, 2)
        })

    cap.release()