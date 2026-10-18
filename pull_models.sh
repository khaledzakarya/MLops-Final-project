#!/bin/bash
echo "📥 Downloading model weights from GitHub Releases..."
wget -q -O best_model_resnet18.pth "https://github.com/khaledzakarya/MLops-Final-project/releases/download/v1.0/best_model_resnet18.pth"
wget -q -O model_resnet18.onnx "https://github.com/khaledzakarya/MLops-Final-project/releases/download/v1.0/model_resnet18.onnx"
echo "✅ Models downloaded successfully!"
