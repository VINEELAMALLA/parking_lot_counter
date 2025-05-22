import os
import xml.etree.ElementTree as ET
import cv2
import random
import shutil

# Mapping labels to YOLO class IDs
label_map = {
    'free_parking_space': 0,
    'occupied_parking_space': 1
}

def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def convert_annotations(xml_path, image_dir, output_dir, split_ratio=0.8):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images = {}
    for image_tag in root.iter('image'):
        filename = image_tag.attrib['name']
        width = int(float(image_tag.attrib['width']))
        height = int(float(image_tag.attrib['height']))
        images[filename] = {
            'width': width,
            'height': height,
            'annotations': []
        }

        for polygon in image_tag.findall('polygon'):
            label = polygon.attrib['label']
            if label not in label_map:
                continue
            class_id = label_map[label]
            points = [[float(x), float(y)] for x, y in (pt.split(',') for pt in polygon.attrib['points'].split(';'))]
            x_min, y_min, x_max, y_max = polygon_to_bbox(points)

            # Normalize for YOLO format
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            images[filename]['annotations'].append((class_id, x_center, y_center, bbox_width, bbox_height))

    # Prepare directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    image_filenames = list(images.keys())
    random.shuffle(image_filenames)
    split_index = int(len(image_filenames) * split_ratio)

    for i, filename in enumerate(image_filenames):
        split = 'train' if i < split_index else 'val'
        image_info = images[filename]
        img_path = os.path.join(image_dir, os.path.basename(filename))
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Copy image
        dst_image_path = os.path.join(output_dir, 'images', split, os.path.basename(filename))
        shutil.copy(img_path, dst_image_path)

        # Write label
        label_filename = os.path.splitext(os.path.basename(filename))[0] + '.txt'
        dst_label_path = os.path.join(output_dir, 'labels', split, label_filename)
        with open(dst_label_path, 'w') as f:
            for ann in image_info['annotations']:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    print("✅ YOLO annotation conversion complete!")

# Run the converter
if __name__ == "__main__":
    convert_annotations(
        xml_path="dataset/annotations.xml",
        image_dir="dataset/images",
        output_dir="yolov8_dataset",
        split_ratio=0.8
    )
