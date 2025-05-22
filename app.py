import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path
import gdown

# === Streamlit App Config ===
st.set_page_config(page_title="Parking Slot Occupancy Detection", layout="wide")
st.title("🚗 Parking Slot Occupancy Detection")
st.write("➡️ Upload an image with **less than 50 parking slots** for better accuracy.")

# === Constants and Paths ===
MODEL_PATH = "weights/best.pt"
Path("weights").mkdir(exist_ok=True)

# Google Drive file ID and download URL for YOLOv8 model
FILE_ID = "10hSLU7_fpGhSQMdV7eK7TJKdNUVHBJ0G"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# === Function to Download Model ===
def download_model():
    with st.spinner("📥 Downloading YOLO model from Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# === Try to Load Model ===
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
except Exception:
    st.warning("⚠️ Model not found or corrupted. Re-downloading...")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    download_model()
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        st.error("❌ Failed to load YOLO model even after re-downloading.")
        st.stop()

# === Default Images ===
default_images = {
    "Default 1": "images/P2.png",
    "Default 2": "images/Screenshot 2025-05-22 161130.png",
    "Default 3": "images/13.png",
    "Default 4": "images/26.png",
}

st.write("### 🖼 Default Images")
cols = st.columns(4)
buttons = []

for i, (label, path) in enumerate(default_images.items()):
    with cols[i]:
        if os.path.exists(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.write(f"**{label}**")
            st.image(img_rgb, use_container_width=True)
        else:
            st.error(f"❌ Image not found: {path}")
        buttons.append(st.button(f"Use {label}"))

# === Upload Custom Image ===
uploaded_file = st.file_uploader("📤 Upload a parking lot image", type=["jpg", "jpeg", "png"])

# === Determine Selected Image ===
image_path = None
if sum(buttons) + (uploaded_file is not None) > 1:
    st.warning("⚠️ Please choose only one input: a default image OR a new upload.")
elif uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
        tfile.write(uploaded_file.read())
        image_path = tfile.name
    st.info("✅ Using uploaded image.")
else:
    for i, pressed in enumerate(buttons):
        if pressed:
            image_path = list(default_images.values())[i]
            st.info(f"✅ Using {list(default_images.keys())[i]}")
            break

# === Run Detection ===
if image_path:
    img = cv2.imread(image_path)
    if img is None:
        st.error("❌ Failed to read the image.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        results = model(img_resized, conf=0.1, iou=0.4)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("⚠️ No parking slots detected.")
            st.image(img_rgb, caption="Input Image (No Detections)", use_container_width=True)
        else:
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()

            valid = confidences >= 0.3
            classes = classes[valid]
            confidences = confidences[valid]
            coordinates = coordinates[valid]

            if len(classes) == 0:
                st.warning("⚠️ All detections were low confidence (< 0.3).")
                st.image(img_rgb, caption="No valid detections", use_container_width=True)
            else:
                img_display = cv2.resize(img, (640, 640))
                for box, cls, conf in zip(coordinates, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    label = f"{'free' if cls == 0 else 'occupied'} ({conf:.2f})"
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="🟢 Detection Results", use_container_width=True)

                df = pd.DataFrame({
                    "slot_id": range(1, len(classes) + 1),
                    "status": ["occupied" if c == 1 else "free" for c in classes],
                    "confidence": confidences
                })

                st.download_button("📥 Download Results as CSV", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="parking_slots_results.csv", mime="text/csv")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Slots", len(classes))
                col2.metric("Occupied", (classes == 1).sum())
                col3.metric("Free", (classes == 0).sum())

                st.write("### 🔍 Detected Slots Summary")
                for i, (cls, conf, box) in enumerate(zip(classes, confidences, coordinates)):
                    status = "free" if cls == 0 else "occupied"
                    st.write(f"Slot {i+1}: {status}, Confidence: {conf:.2f}, Box: {box}")

    # Delete temp uploaded file after use
    if uploaded_file:
        os.unlink(image_path)
