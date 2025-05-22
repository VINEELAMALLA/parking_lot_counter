import streamlit as st
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# Load your trained YOLOv8 model
model_path = r"C:/Users/vinee/OneDrive/parking_space/runs/detect/train5/weights/best.pt"

# Paths to the default input images
default_image_1_path = r"C:/Users/vinee/OneDrive/parking_space/P2.png"  # First image (aerial view, 129 slots)
default_image_2_path = r"C:\Users\vinee\OneDrive\parking_space\Screenshot 2025-05-22 161130.png"  # Second image (angled view, fewer slots)
default_image_3_path = r"C:\Users\vinee\OneDrive\parking_space\images\13.png"  # Third image (newly added)
default_image_4_path = r"C:\Users\vinee\OneDrive\parking_space\images\26.png"  # Fourth image (newly added)

# Verify model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
else:
    # Verify default images exist
    if not os.path.exists(default_image_1_path):
        st.error(f"Default Image 1 not found at: {default_image_1_path}")
    elif not os.path.exists(default_image_2_path):
        st.error(f"Default Image 2 not found at: {default_image_2_path}")
    elif not os.path.exists(default_image_3_path):
        st.error(f"Default Image 3 not found at: {default_image_3_path}")
    elif not os.path.exists(default_image_4_path):
        st.error(f"Default Image 4 not found at: {default_image_4_path}")
    else:
        model = YOLO(model_path)

        st.title("Parking Slot Occupancy Detection")
        st.write("- Upload image with less no. of parking lots i.e. <50 for more accurate prediction by the model.")

        # Display previews for all default images in a single row
        st.write("### Default Images")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("**Default Input Image 1**")
            default_img_1 = cv2.imread(default_image_1_path)
            default_img_1_rgb = cv2.cvtColor(default_img_1, cv2.COLOR_BGR2RGB)
            st.image(default_img_1_rgb, caption="Aerial view (many slots)", use_container_width=True)

        with col2:
            st.write("**Default Input Image 2**")
            default_img_2 = cv2.imread(default_image_2_path)
            default_img_2_rgb = cv2.cvtColor(default_img_2, cv2.COLOR_BGR2RGB)
            st.image(default_img_2_rgb, caption="Angled view (fewer slots)", use_container_width=True)

        with col3:
            st.write("**Default Image 3**")
            default_img_3 = cv2.imread(default_image_3_path)
            default_img_3_rgb = cv2.cvtColor(default_img_3, cv2.COLOR_BGR2RGB)
            st.image(default_img_3_rgb, caption="Aerial view (fewer slots)", use_container_width=True)

        with col4:
            st.write("**Default Input Image 4**")
            default_img_4 = cv2.imread(default_image_4_path)
            default_img_4_rgb = cv2.cvtColor(default_img_4, cv2.COLOR_BGR2RGB)
            st.image(default_img_4_rgb, caption="Aerial view (fewer slots)", use_container_width=True)

        # Buttons for selecting default images (also in a single row)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            use_default_1 = st.button("Use Default Image 1")
        with col2:
            use_default_2 = st.button("Use Default Image 2")
        with col3:
            use_default_3 = st.button("Use Default Image 3")
        with col4:
            use_default_4 = st.button("Use Default Image 4")

        # File uploader for user-provided image
        uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

        # Determine which image to process
        image_path = None
        if sum([use_default_1, use_default_2, use_default_3, use_default_4, uploaded_file is not None]) > 1:
            st.warning("Please choose only one input: either a default image or upload a new image.")
        elif use_default_1:
            image_path = default_image_1_path
            st.write("Using Default Image 1 for detection.")
        elif use_default_2:
            image_path = default_image_2_path
            st.write("Using Default Image 2 for detection.")
        elif use_default_3:
            image_path = default_image_3_path
            st.write("Using Default Image 3 for detection.")
        elif use_default_4:
            image_path = default_image_4_path
            st.write("Using Default Image 4 for detection.")
        elif uploaded_file is not None:
            # Save uploaded file temporarily to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
                tfile.write(uploaded_file.read())
                image_path = tfile.name
            st.write("Using the uploaded image for detection.")

        # Process the selected image if available
        if image_path is not None:
            # Read image with OpenCV
            img = cv2.imread(image_path)

            if img is None:
                st.error("Failed to read the image. Please ensure the file is valid.")
            else:
                # Preprocess image: Convert to RGB and resize to match training input
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (640, 640))  # YOLOv8 default input size

                # Run detection with adjusted thresholds
                results = model(img_resized, conf=0.1, iou=0.4)

                # Extract boxes and confidences
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    st.warning("No parking slots detected.")
                    st.image(img_rgb, caption="Input Image (No Detections)", use_container_width=True)
                else:
                    # Extract class IDs, confidence scores, and bounding box coordinates
                    classes = boxes.cls.cpu().numpy().astype(int)  # class IDs (0=free, 1=occupied)
                    confidences = boxes.conf.cpu().numpy()  # Confidence scores
                    coordinates = boxes.xyxy.cpu().numpy()  # Bounding box coordinates

                    # Post-process: Filter low-confidence detections
                    min_confidence = 0.3  # Adjust this threshold as needed
                    valid_indices = confidences >= min_confidence
                    classes = classes[valid_indices]
                    coordinates = coordinates[valid_indices]
                    confidences = confidences[valid_indices]

                    if len(classes) == 0:
                        st.warning("No slots detected after filtering low-confidence detections.")
                        st.image(img_rgb, caption="Input Image (No Detections)", use_container_width=True)
                    else:
                        # Step 1: Display the image with detected parking spaces
                        img_display = cv2.resize(img, (640, 640))  # Resize for consistency
                        for box, cls_id, conf in zip(coordinates, classes, confidences):
                            x1, y1, x2, y2 = map(int, box)
                            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # green=free, red=occupied
                            label = f"{'free' if cls_id == 0 else 'occupied'} ({conf:.2f})"
                            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_display, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="Detection Results", use_container_width=True)

                        # Step 2: Prepare and display the downloadable Excel table (CSV)
                        df = pd.DataFrame({
                            "slot_id": range(1, len(classes) + 1),
                            "status": ["occupied" if c == 1 else "free" for c in classes],
                            "confidence": confidences
                        })

                        csv_data = df.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name="parking_slots_results.csv",
                            mime="text/csv"
                        )

                        # Step 3: Show the slot calculations (Total Slots, Occupied Slots, Available Slots)
                        total_slots = len(classes)
                        occupied_slots = (classes == 1).sum()
                        free_slots = (classes == 0).sum()

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Slots", total_slots)
                        col2.metric("Occupied Slots", occupied_slots)
                        col3.metric("Available Slots", free_slots)

                        # Step 4: Display the detailed detection information
                        st.write("### Detected Slots:")
                        for i, (cls, conf, box) in enumerate(zip(classes, confidences, coordinates)):
                            status = "free" if cls == 0 else "occupied"
                            st.write(f"Slot {i+1}: {status}, Confidence: {conf:.2f}, Box: {box}")

            # Clean up temporary file if it was created
            if uploaded_file is not None:
                os.unlink(image_path)