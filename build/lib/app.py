import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import json
from src.model import PetBreedModel
from torchvision import transforms

app = FastAPI(title="Pet Breed Classifier API")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_VERSION = "v1.0"

# 1. Load existing Model & Temperature
model = PetBreedModel(backbone_name='resnet18', num_classes=37).to(device)
model.load_state_dict(torch.load('best_model_resnet18.pth', map_location=device))
model.eval()

with open('temperature.txt', 'r') as f:
    temperature = float(f.read().strip())

with open('data/label_map.json', 'r') as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

# Oxford-IIIT Cat breeds (the rest are dogs)
CAT_BREEDS = {'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 
              'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION, "num_classes": 37}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. File Size Validation (Reject > 5MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 5 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 5MB.")

    # 2. Read and Validate Image format
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid image format.")

    # 3. Reject 4-channel PNG (RGBA)
    if image.mode == 'RGBA':
        raise HTTPException(status_code=422, detail="4-channel PNG images are not supported.")
        
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)[0]
        
    # Get Top 3
    top3_prob, top3_idx = torch.topk(probs, 3)
    top3_list = [{"breed": idx_to_class[idx.item()], "prob": round(p.item(), 4)} for p, idx in zip(top3_prob, top3_idx)]
    
    best_prob = top3_prob[0].item()
    best_breed = idx_to_class[top3_idx[0].item()]
    species = "cat" if best_breed in CAT_BREEDS else "dog"
    
    # 4. Abstention Logic matching the handbook
    decision = "confident" if best_prob >= 0.80 else "uncertain"
    
    return {
        "breed": best_breed if decision == "confident" else "Unknown",
        "species": species,
        "confidence": round(best_prob, 4),
        "top_3": top3_list,
        "decision": decision,
        "model_version": MODEL_VERSION
    }
