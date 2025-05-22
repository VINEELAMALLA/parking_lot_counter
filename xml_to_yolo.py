import json
import os

# === Paths ===
json_path = r"C:\Users\vinee\OneDrive\parking space\PKLot_dataset\train\_annotations.coco.json"
images_dir = r"C:\Users\vinee\OneDrive\parking space\PKLot_dataset\train"
labels_dir = os.path.join(os.path.dirname(images_dir), "labels", "train")
os.makedirs(labels_dir, exist_ok=True)

# === Load COCO JSON ===
with open(json_path) as f:
    data = json.load(f)

# Create dict to map image_id -> file_name
image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

# Create dict: image_id -> list of annotations
image_annotations = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# === Process each image ===
for image_id, file_name in image_id_to_filename.items():
    width = next((img["width"] for img in data["images"] if img["id"] == image_id), 1)
    height = next((img["height"] for img in data["images"] if img["id"] == image_id), 1)

    label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")

    with open(label_path, "w") as f:
        anns = image_annotations.get(image_id, [])
        for ann in anns:
            # Get class_id: Roboflow uses 1 = empty, 2 = occupied
            class_id = ann["category_id"]
            if class_id == 1:
                yolo_class = 0  # space-empty → free
            elif class_id == 2:
                yolo_class = 1  # space-occupied → occupied
            else:
                continue  # skip any other classes

            # COCO bbox format: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height

            f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ Done converting COCO to YOLO format.")
