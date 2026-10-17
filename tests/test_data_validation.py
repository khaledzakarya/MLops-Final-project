import os
import json
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def check_image(record):
    path = record['path']
    if not os.path.exists(path):
        return False, f"Missing file: {path}"
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < 32 or h < 32:
                return False, f"Image {path} is too small: {w}x{h}"
    except Exception as e:
        return False, f"Cannot open {path}: {str(e)}"
    return True, ""

def test_dataset_validity(manifest_path='data/manifest.json', label_map_path='data/label_map.json'):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    assert len(label_map) == 37, f"Expected 37 classes, found {len(label_map)}"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    train_classes_count = defaultdict(int)
    seen_trainval_ids = set()
    seen_test_ids = set()
    
    print(f"Validating {len(manifest)} images using 32 parallel threads...")
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(check_image, manifest))
        
    for success, msg in results:
        assert success, msg
        
    for record in manifest:
        if record['split'] in ['train', 'val']:
            seen_trainval_ids.add(record['image_id'])
            if record['split'] == 'train':
                train_classes_count[record['breed']] += 1
        elif record['split'] == 'test':
            seen_test_ids.add(record['image_id'])

    overlap = seen_trainval_ids.intersection(seen_test_ids)
    assert len(overlap) == 0, f"Found {len(overlap)} images overlapping across splits"
    
    for breed, count in train_classes_count.items():
        assert count >= 50, f"Class {breed} has only {count} training images (requires >= 50)"

    print("✅ All 'Validate before you train' tests passed successfully!")

if __name__ == "__main__":
    test_dataset_validity()
