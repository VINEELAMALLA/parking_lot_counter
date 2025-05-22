import json
import os

def convert_coco_to_yolo(json_path, image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    image_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    for image_id, file_name in image_id_to_filename.items():
        width = next((img["width"] for img in data["images"] if img["id"] == image_id), 1)
        height = next((img["height"] for img in data["images"] if img["id"] == image_id), 1)

        label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(label_path, "w") as f:
            anns = image_annotations.get(image_id, [])
            for ann in anns:
                class_id = ann["category_id"]
                if class_id == 1:
                    yolo_class = 0  # free
                elif class_id == 2:
                    yolo_class = 1  # occupied
                else:
                    continue

                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height

                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Converted: {json_path}")

# === Run for train, valid, test ===
base_path = r"C:\Users\vinee\OneDrive\parking space\PKLot_dataset"
splits = ['train', 'valid', 'test']

for split in splits:
    image_dir = os.path.join(base_path, split)
    json_path = os.path.join(image_dir, "_annotations.coco.json")
    label_dir = os.path.join(base_path, "labels", split)
    if os.path.exists(json_path):
        convert_coco_to_yolo(json_path, image_dir, label_dir)
    else:
        print(f"⚠️ Skipped: No annotation file for {split}")
