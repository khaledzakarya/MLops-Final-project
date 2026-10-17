import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PetDataset(Dataset):
    def __init__(self, manifest_path, label_map_path, split='train', transform=None):
        super().__init__()
        self.split = split
        self.transform = transform
        
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
            
        # Robust mapping: normalizes names (lowercasing & replacing spaces with underscores)
        self.robust_label_map = {self._normalize_name(k): v for k, v in self.label_map.items()}
            
        self.records = [item for item in manifest_data if item['split'] == split]

    def _normalize_name(self, name):
        return name.lower().replace(' ', '_').strip()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        image = Image.open(record['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Look up using the normalized name
        breed_key = self._normalize_name(record['breed'])
        label_idx = self.robust_label_map[breed_key]
        label = torch.tensor(label_idx, dtype=torch.long)
        
        return image, label
