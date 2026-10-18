import torch
import onnxruntime as ort
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

def test_logit_agreement():
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_path = "test_dog.jpg"
    img = Image.open(img_path).convert('RGB')
    input_tensor = eval_transform(img).unsqueeze(0)

    num_classes = 37
    pytorch_model = models.resnet18(weights=None)
    pytorch_model.fc = nn.Linear(pytorch_model.fc.in_features, num_classes)
    
    # Load weights safely
    pytorch_model.load_state_dict(torch.load("best_model_resnet18.pth", map_location='cpu', weights_only=True))
    pytorch_model.eval()

    with torch.no_grad():
        pytorch_logits = pytorch_model(input_tensor).numpy()

    ort_session = ort.InferenceSession("model_resnet18.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    onnx_logits = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(pytorch_logits, onnx_logits, rtol=1e-4, atol=1e-4)
    print("✅ SUCCESS: PyTorch and ONNX logits agree to 1e-4 strictly!")

if __name__ == "__main__":
    test_logit_agreement()