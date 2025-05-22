import os
import xml.etree.ElementTree as ET
from PIL import Image
from ultralytics import YOLO

# Step 1: Convert XML annotations to YOLO format
def convert_to_yolo(xml_file, img_file, output_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img = Image.open(img_file)
        img_width, img_height = img.size

        # Output label file
        txt_file = os.path.join(output_dir, os.path.basename(img_file).replace(".jpg", ".txt"))
        with open(txt_file, "w") as f:
            for space in root.findall(".//space"):
                occupied = space.get("occupied") == "1"
                class_id = 1 if occupied else 0  # 0: free, 1: occupied

                rotated_rect = space.find("rotatedRect")
                center_x = float(rotated_rect.find("center").get("x"))
                center_y = float(rotated_rect.find("center").get("y"))
                width = float(rotated_rect.find("size").get("w"))
                height = float(rotated_rect.find("size").get("h"))

                # Normalize
                x_center = center_x / img_width
                y_center = center_y / img_height
                w = width / img_width
                h = height / img_height

                f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

# Step 2: Prepare dataset structure
base_dir = r"C:\Users\vinee\OneDrive\parking space\PKLot_dataset"
splits = ["train", "valid"]

for split in splits:
    split_dir = os.path.join(base_dir, split)

    # Clear old YOLO cache (forces dataset rescan)
    cache_file = os.path.join(split_dir, f"{split}.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    # Process XML files in each split
    for file_name in os.listdir(split_dir):
        if file_name.endswith(".xml"):
            xml_path = os.path.join(split_dir, file_name)
            img_path = xml_path.replace(".xml", ".jpg")
            if os.path.exists(img_path):
                convert_to_yolo(xml_path, img_path, split_dir)
            else:
                print(f"[WARN] Image not found for {file_name}, skipping...")

# Step 3: Write YOLOv8 data YAML file
data_yaml_path = os.path.join(base_dir, "data1.yaml")
with open(data_yaml_path, "w") as f:
    f.write(f"""train: {base_dir}\\train
val: {base_dir}\\valid
nc: 2
names: ['free', 'occupied']
""")

# Step 4: Train YOLOv8 model
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # Load base model; replace with custom path if needed

    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name="parking_yolo_train"
    )
