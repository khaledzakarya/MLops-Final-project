import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PetDataset(Dataset):
    def __init__(self, manifest_path, label_map_path, split='train', transform=None):
        """
        Initializes the Dataset using a strictly committed label map.
        Transforms are passed externally to allow exporting the exact eval transform later.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        
        # Load the manifest
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        # Load the committed label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
            
        # Filter by split
        self.records = [item for item in manifest_data if item['split'] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Open image and ensure RGB (handles CMYK/grayscale/4-channel PNG bugs)
        image = Image.open(record['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Strictly use the committed label map for the class index
        label_idx = self.label_map[record['breed']]
        label = torch.tensor(label_idx, dtype=torch.long)
        
        return image, label
