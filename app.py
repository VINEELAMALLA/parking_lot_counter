import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path
import gdown
from ultralytics import YOLO

# === Step 1: Download the YOLO model from Google Drive if not present ===
MODEL_PATH = "weights/best.pt"
Path("weights").mkdir(exist_ok=True)

# Replace with your actual Google Drive file ID
FILE_ID = "10hSLU7_fpGhSQMdV7eK7TJKdNUVHBJ0G"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLO model..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# === Step 2: Load the model ===
model = YOLO(MODEL_PATH)

# === Step 3: Default image paths ===
default_image_1_path = "images/P2.png"
default_image_2_path = "images/Screenshot 2025-05-22 161130.png"
default_image_3_path = "images/13.png"
default_image_4_path = "images/26.png"

st.title("Parking Slot Occupancy Detection")
st.write("- Upload image with less no. of parking lots i.e. <50 for more accurate prediction by the model.")

# === Step 4: Show preview of default images ===
st.write("### Default Images")
col1, col2, col3, col4 = st.columns(4)

def show_image_column(col, path, title, caption):
    with col:
        if os.path.exists(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.write(f"**{title}**")
            st.image(img_rgb, caption=caption, use_container_width=True)
        else:
            st.error(f"Image not found: {path}")

show_image_column(col1, default_image_1_path, "Default Input Image 1", "Aerial view (many slots)")
show_image_column(col2, default_image_2_path, "Default Input Image 2", "Angled view (fewer slots)")
show_image_column(col3, default_image_3_path, "Default Image 3", "Aerial view (fewer slots)")
show_image_column(col4, default_image_4_path, "Default Input Image 4", "Aerial view (fewer slots)")

# === Step 5: Buttons for selecting default images ===
col1, col2, col3, col4 = st.columns(4)
use_default_1 = col1.button("Use Default Image 1")
use_default_2 = col2.button("Use Default Image 2")
use_default_3 = col3.button("Use Default Image 3")
use_default_4 = col4.button("Use Default Image 4")

# === Step 6: Upload custom image ===
uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

# === Step 7: Decide input image ===
image_path = None
if sum([use_default_1, use_default_2, use_default_3, use_default_4, uploaded_file is not None]) > 1:
    st.warning("Please choose only one input: either a default image or upload a new image.")
elif use_default_1:
    image_path = default_image_1_path
    st.info("Using Default Image 1 for detection.")
elif use_default_2:
    image_path = default_image_2_path
    st.info("Using Default Image 2 for detection.")
elif use_default_3:
    image_path = default_image_3_path
    st.info("Using Default Image 3 for detection.")
elif use_default_4:
    image_path = default_image_4_path
    st.info("Using Default Image 4 for detection.")
elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
        tfile.write(uploaded_file.read())
        image_path = tfile.name
    st.info("Using the uploaded image for detection.")

# === Step 8: Detection ===
if image_path is not None:
    img = cv2.imread(image_path)
    if img is None:
        st.error("Failed to read the image.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        results = model(img_resized, conf=0.1, iou=0.4)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("No parking slots detected.")
            st.image(img_rgb, caption="Input Image (No Detections)", use_container_width=True)
        else:
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()

            valid_indices = confidences >= 0.3
            classes = classes[valid_indices]
            coordinates = coordinates[valid_indices]
            confidences = confidences[valid_indices]

            if len(classes) == 0:
                st.warning("No slots detected after filtering low-confidence detections.")
                st.image(img_rgb, caption="Input Image (No Detections)", use_container_width=True)
            else:
                img_display = cv2.resize(img, (640, 640))
                for box, cls_id, conf in zip(coordinates, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                    label = f"{'free' if cls_id == 0 else 'occupied'} ({conf:.2f})"
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="Detection Results", use_container_width=True)

                df = pd.DataFrame({
                    "slot_id": range(1, len(classes) + 1),
                    "status": ["occupied" if c == 1 else "free" for c in classes],
                    "confidence": confidences
                })
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button("Download Results as CSV", data=csv_data,
                                   file_name="parking_slots_results.csv", mime="text/csv")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Slots", len(classes))
                col2.metric("Occupied Slots", (classes == 1).sum())
                col3.metric("Available Slots", (classes == 0).sum())

                st.write("### Detected Slots:")
                for i, (cls, conf, box) in enumerate(zip(classes, confidences, coordinates)):
                    status = "free" if cls == 0 else "occupied"
                    st.write(f"Slot {i+1}: {status}, Confidence: {conf:.2f}, Box: {box}")

    if uploaded_file is not None:
        os.unlink(image_path)
