import os
import xml.etree.ElementTree as ET
from PIL import Image

def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

def find_image_path(image_folder, target_filename):
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file == target_filename:
                return os.path.join(root, file)
    return None

def parse_cvat_polygon_annotations(xml_path, image_base_folder, label_base_folder):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Failed to parse XML: {e}")
        return

    found_images = list(root.iter('image'))
    print(f"🔍 Found {len(found_images)} image(s) in XML.")

    processed_files = set()

    for image_tag in found_images:
        filename = os.path.basename(image_tag.attrib['name'])  # e.g., 0.png

        # Skip duplicate entries
        if filename in processed_files:
            continue
        processed_files.add(filename)

        image_path = find_image_path(image_base_folder, filename)

        if not image_path:
            print(f"⚠️ Image not found: {filename}")
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️ Failed to open image {filename}: {e}")
            continue

        # Determine whether it's train or val from folder structure
        subfolder = 'train' if 'train' in image_path.lower() else 'val'

        yolo_labels = []
        for polygon in image_tag.findall('polygon'):
            label = polygon.attrib['label']
            if label not in ['free_parking_space', 'not_free_parking_space']:
                continue

            class_id = 0 if label == 'free_parking_space' else 1
            points = [
                [float(x), float(y)]
                for x, y in (pt.split(',') for pt in polygon.attrib['points'].split(';'))
            ]
            x_center, y_center, width, height = polygon_to_bbox(points)

            # Normalize
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_labels:
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_output_folder = os.path.join(label_base_folder, subfolder)
            os.makedirs(label_output_folder, exist_ok=True)

            label_path = os.path.join(label_output_folder, label_filename)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            print(f"✅ Labels written for {filename}")
        else:
            print(f"⚠️ No valid labels found for {filename}")

    print(f"\n🎉 Completed! YOLO annotation files saved under: {label_base_folder}")

# Example usage
if __name__ == "__main__":
    parse_cvat_polygon_annotations(
        xml_path=r"C:\Users\vinee\OneDrive\parking space\yolov8_dataset\annotations.xml",
        image_base_folder=r"C:\Users\vinee\OneDrive\parking space\yolov8_dataset\images",
        label_base_folder=r"C:\Users\vinee\OneDrive\parking space\yolov8_dataset\labels"
    )
