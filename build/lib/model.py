import torch
import torch.nn as nn
from torchvision import models

class PetBreedModel(nn.Module):
    def __init__(self, backbone_name='mobilenet_v3_small', num_classes=37):
        """
        Supports the 3 backbones mandated by the MLOps Handbook:
        - resnet18
        - resnet50
        - mobilenet_v3_small
        """
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
        elif backbone_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
        elif backbone_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from resnet18, resnet50, mobilenet_v3_small")

    def forward(self, x):
        return self.model(x)
