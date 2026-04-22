# 🚦 Traffic Density Detection using YOLOv8 & Machine Learning

## 📌 Overview
This project implements an end-to-end pipeline for traffic density analysis using Computer Vision and Machine Learning.

The system processes video input, detects vehicles using YOLOv8, extracts temporal features, and predicts traffic conditions (Smooth, Dense, Jammed) using a trained ML model.

🔗 **Live App**  
https://traffic-density-app-ycwjym7ab96snpm6najrp7.streamlit.app/

---

## 🎯 Problem Statement
Urban traffic congestion is a critical issue in modern cities.

Manual monitoring:
- ❌ Not scalable  
- ❌ Subjective  
- ❌ Inefficient  

This project provides a **data-driven approach** to classify traffic density automatically from video input.

---

## 🧠 System Pipeline

---

## ⚙️ How It Works

### 1. Frame Processing
- Resize frame (416x234) for efficiency
- Frame skipping to reduce computation load

### 2. Object Detection
- Model: YOLOv8 (Ultralytics)
- Classes: car, motorcycle, bus, truck

### 3. Region of Interest (ROI)
- Polygon-based ROI
- Filters irrelevant detections outside road area

### 4. Feature Engineering
Features extracted per frame:
- **count_smooth** → moving average of vehicle count
- **density** → normalized by ROI area
- **delta** → change between frames
- **var** → traffic variance
- **flow** → movement intensity

### 5. Machine Learning Model
- Input: engineered features
- Output: traffic category
  - Smooth
  - Dense
  - Jammed

### 6. Temporal Stabilization
- Prediction buffer
- Majority voting
- Reduces flickering output

---

## 🧪 Tech Stack

| Component        | Technology |
|----------------|----------|
| Language        | Python |
| CV Library      | OpenCV |
| Detection Model | YOLOv8 |
| ML Model        | Scikit-learn |
| UI              | Streamlit |

---

## 📊 Key Features
- 🎥 Video-based traffic analysis
- 🚗 Multi-vehicle detection (YOLOv8)
- 📉 ML-based classification (not rule-based)
- 🔁 Temporal smoothing (anti-flicker)
- 📊 Real-time visualization

---

## 📁 Project Structure

---

## ⚠️ Limitations
- CPU-only inference (no GPU acceleration)
- Not optimized for real-time streaming
- Performance depends on:
  - Camera angle
  - Lighting conditions
  - Video quality
- ML model is sensitive to feature distribution mismatch

---

## 📈 Engineering Challenges & Insights

This project highlights real-world ML engineering problems:

- ⚠️ Feature mismatch between training and inference
- ⚠️ Trade-off between performance vs accuracy (frame skipping)
- ⚠️ Noise in temporal data (handled with smoothing)
- ⚠️ Integration of CV + ML in a single pipeline

---

## 🚀 Future Improvements
- GPU deployment (Docker / cloud)
- Replace classical ML with temporal models (LSTM / Transformer)
- Real-time CCTV integration
- Auto ROI detection
- Retrain model with larger dataset

---

## 🧪 Model Evaluation
> ⚠️ (Add this if you have it — strongly recommended)

- Accuracy: XX%
- Dataset size: XXX samples
- Evaluation method: train/test split or cross-validation

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/username/traffic-density-app.git
cd traffic-density-app
pip install -r requirements.txt
streamlit run app.py