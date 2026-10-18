import bentoml
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL.Image import Image
import json

# تحميل الـ Label Map
with open("data/label_map.json", "r") as f:
    idx_to_class = {int(v): k for k, v in json.load(f).items()}

# التعديلات
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# الطريقة الحديثة: الكلاس
@bentoml.service(name="pet_breed_classifier")
class PetBreedService:
    def __init__(self):
        # الموديل بيتحمل مرة واحدة بس لما السيرفر يشتغل (أوفر في الرامات جداً)
        self.ort_session = ort.InferenceSession("model_resnet18.onnx")
        
    @bentoml.api
    async def predict(self, img: Image) -> dict:
        # تأمين: لو الصورة PNG شفافة نحولها لـ RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        input_tensor = eval_transform(img).unsqueeze(0).numpy()
        
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        logits = self.ort_session.run(None, ort_inputs)[0]
        
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
        max_prob, predicted_idx = torch.max(probabilities, 1)
        
        confidence = max_prob.item()
        breed = idx_to_class[predicted_idx.item()]
        
        if confidence < 0.7:
            return {"breed": "Unknown - low confidence", "confidence": round(confidence, 4)}
            
        return {"breed": breed, "confidence": round(confidence, 4)}