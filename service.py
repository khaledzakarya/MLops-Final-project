import bentoml
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL.Image import Image
import json
import os

with open("data/label_map.json", "r") as f:
    idx_to_class = {int(v): k for k, v in json.load(f).items()}

CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", 
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", 
              "Siamese", "Sphynx"}

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    with open("temperature.txt", "r") as f:
        TEMPERATURE = float(f.read().strip())
except Exception:
    TEMPERATURE = 1.0

@bentoml.service(name="pet_breed_classifier")
class PetBreedService:
    def __init__(self):
        self.ort_session = ort.InferenceSession("model_resnet18.onnx")
        self.model_version = "v1.0.0"
        self.num_classes = 37
        self.abstention_threshold = 0.70  
        
    # --- NEW: Health Check Endpoint for Docker ---
    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {
            "status": "healthy",
            "model_version": self.model_version,
            "num_classes": self.num_classes
        }

    @bentoml.api
    async def predict(self, img: Image) -> dict:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        input_tensor = eval_transform(img).unsqueeze(0).numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        logits = self.ort_session.run(None, ort_inputs)[0]
        
        scaled_logits = torch.tensor(logits) / TEMPERATURE
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)[0]
        
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top_3 = []
        for i in range(3):
            breed_name = idx_to_class[top3_idx[i].item()]
            top_3.append({
                "breed": breed_name,
                "confidence": round(top3_prob[i].item(), 4)
            })
            
        best_breed = top_3[0]["breed"]
        best_conf = top_3[0]["confidence"]
        species = "cat" if best_breed in CAT_BREEDS else "dog"
        decision = "confident" if best_conf >= self.abstention_threshold else "uncertain"
        
        return {
            "breed": best_breed,
            "species": species,
            "confidence": best_conf,
            "top_3": top_3,
            "decision": decision,
            "model_version": self.model_version
        }