import os
import json
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "images"
CORRUPT_DIR = DATA_DIR / "corruptions"
CORRUPT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "label_map.json", "r") as f:
    label_map = json.load(f)

# Define the required corruptions suite
corruptions_suite = {
    "gaussian_blur": [A.GaussianBlur(blur_limit=(3, 3), p=1), A.GaussianBlur(blur_limit=(7, 7), p=1), A.GaussianBlur(blur_limit=(11, 11), p=1)],
    "brightness": [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1), A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1), A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0, p=1)],
    "jpeg_compression": [A.ImageCompression(quality_lower=80, quality_upper=81, p=1), A.ImageCompression(quality_lower=50, quality_upper=51, p=1), A.ImageCompression(quality_lower=20, quality_upper=21, p=1)],
    "downscale_upscale": [A.Downscale(scale_min=0.7, scale_max=0.7, p=1), A.Downscale(scale_min=0.5, scale_max=0.5, p=1), A.Downscale(scale_min=0.25, scale_max=0.25, p=1)]
}

def get_breed_from_filename(filename):
    return "_".join(filename.split("_")[:-1])

def process_dataset():
    images = list(IMG_DIR.glob("*.jpg"))
    np.random.seed(42)
    
    # Split dataset into Train, Val, and Test
    trainval_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=0.2, random_state=42)
    
    split_map = {img.name: "train" for img in train_imgs}
    split_map.update({img.name: "val" for img in val_imgs})
    split_map.update({img.name: "test" for img in test_imgs})

    manifest = []
    print("Processing images, generating corruptions, and building manifest... (Please wait)")
    
    for count, img_path in enumerate(images):
        if count > 0 and count % 1000 == 0:
            print(f"Processed {count} original images...")
            
        img_name = img_path.name
        image_id = img_path.stem
        breed = get_breed_from_filename(image_id)
        class_index = label_map.get(breed, -1)
        species = "cat" if breed[0].isupper() else "dog"
        split = split_map.get(img_name, "train")
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        height, width = img.shape[:2]
        
        # Record original image data
        manifest.append({
            "image_id": image_id, "path": f"data/images/{img_name}",
            "breed": breed, "species": species, "class_index": class_index,
            "split": split, "corruption": None, "severity": 0,
            "width": width, "height": height
        })
        
        # Apply corruptions ONLY to the Test split
        if split == "test":
            for corr_name, transforms in corruptions_suite.items():
                for severity_idx, transform in enumerate(transforms):
                    severity_level = severity_idx + 1
                    augmented = transform(image=img)
                    corr_filename = f"{image_id}_{corr_name}_{severity_level}.jpg"
                    cv2.imwrite(str(CORRUPT_DIR / corr_filename), augmented["image"])
                    
                    manifest.append({
                        "image_id": f"{image_id}_{corr_name}_{severity_level}",
                        "path": f"data/corruptions/{corr_filename}",
                        "breed": breed, "species": species, "class_index": class_index,
                        "split": split, "corruption": corr_name, "severity": severity_level,
                        "width": width, "height": height
                    })

    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"✅ Process completed successfully! Total records in manifest: {len(manifest)}")

if __name__ == "__main__":
    process_dataset()
