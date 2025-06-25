import torch
from torchvision import transforms
from PIL import Image
from src.models.gender_cnn import get_convnext_model
import os


class Predictor:
    def __init__(self, model_path):
        self.model = get_convnext_model()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Chuẩn hóa theo ImageNet
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.labels = ["Male", "Female"]

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.argmax(dim=1).item()
        return self.labels[pred]
