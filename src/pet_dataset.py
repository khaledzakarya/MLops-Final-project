import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PetDataset(Dataset):
    def __init__(self, manifest_path, split='train', img_size=224):
        """
        Initializes the Dataset and builds a dynamic label map to ensure 
        labels are always in the range [0, num_classes - 1].
        """
        super().__init__()
        self.split = split
        
        # Load the manifest file
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        # Self-healing: Build a foolproof breed-to-index mapping (0 to N-1)
        unique_breeds = sorted(list(set(item['breed'] for item in manifest_data)))
        self.class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
            
        # Filter data based on the split
        self.records = [item for item in manifest_data if item['split'] == split]
        self.transform = self._get_transforms(img_size)

    def _get_transforms(self, img_size):
        """Returns PyTorch transformations."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.records)

    def __getitem__(self, idx):
        """Loads an image and its dynamically mapped label on-the-fly."""
        record = self.records[idx]
        img_path = record['path']
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Get integer label dynamically from the breed name
        label_idx = self.class_to_idx[record['breed']]
        label = torch.tensor(label_idx, dtype=torch.long)
        
        return image, label
