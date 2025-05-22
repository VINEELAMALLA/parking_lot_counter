import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ---------- Preprocessing Functions ----------

def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def resize_image(img, max_width=1280):
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def preprocess_pipeline(img, use_denoise=True, use_resize=True):
    if use_resize:
        img = resize_image(img)
    img = enhance_image(img)
    if use_denoise:
        img = denoise_image(img)
    return img

# ---------- Image Processing Wrapper ----------

def process_image(input_dir, output_dir, fname, use_denoise=True, use_resize=True):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)
    img = cv2.imread(input_path)

    if img is None:
        print(f"⚠️ Skipping unreadable image: {input_path}")
        return

    img_proc = preprocess_pipeline(img, use_denoise, use_resize)
    cv2.imwrite(output_path, img_proc)

# ---------- Preprocessing Execution ----------

def preprocess_and_save(input_dir, output_dir, num_threads=8, use_denoise=True, use_resize=True):
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    print(f"🔄 Processing {len(file_list)} images from {input_dir} → {output_dir} using {num_threads} threads")

    process_fn = partial(
        process_image,
        input_dir,
        output_dir,
        use_denoise=use_denoise,
        use_resize=use_resize
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_fn, file_list)

    print(f"✅ Done processing {input_dir}")

# ---------- Main Execution ----------

if __name__ == "__main__":
    base_dir = r"C:/Users/vinee/OneDrive/parking_space/PKLot_dataset"

    preprocess_and_save(
        input_dir=os.path.join(base_dir, "train"),
        output_dir=os.path.join(base_dir, "images_preprocessed/train"),
        num_threads=8,         # Adjust to your CPU (e.g., 8, 16)
        use_denoise=True,      # Set to False to skip denoising for speed
        use_resize=True        # Set to False if you don't want resizing
    )

    preprocess_and_save(
        input_dir=os.path.join(base_dir, "val"),
        output_dir=os.path.join(base_dir, "images_preprocessed/val"),
        num_threads=8,
        use_denoise=True,
        use_resize=True
    )
