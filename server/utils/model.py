from PIL import Image
import numpy as np

import torch
import mlflow.pytorch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

mlflow.set_tracking_uri("http://nthng-quan.ddns.net:5000")
mlflow.set_experiment("fire-detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class Model(object):
    def __init__(self, num_classes=2, model_path="./model", reload=True):
        if reload:
            print(f"-> Reloading model from {model_path}...")
            self.model = ReloadModel(model_path)
        else:
            self.model = Resnet18(num_classes, freeze=True)
        self.model.eval()

    def predict(self, img_fn):
        return self.model.predict(img_fn)

class ReloadModel(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        if torch.cuda.is_available():
            self.model = mlflow.pytorch.load_model(model_path)
        else:
            self.model = mlflow.pytorch.load_model(
                model_path, map_location=torch.device('cpu'))

        self.model.eval()

    def predict(self, img_fn):
        with torch.inference_mode():
            img = Image.open(img_fn).convert('RGB')
            img = data_transform(img)
            img = img.to(device)

            result = self.model(img.unsqueeze(0))
            result = torch.argmax(result, dim=1)
        
        return result.item()

class Resnet18(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()
        self.model = models.resnet18()
        if freeze:
            for param in self.model.parameters():
                param.require_grad = False
        
        self.model.fc = nn.Linear(512, num_classes)
        self.model.to(device)
        self.model.eval()
    
    def forward(self, x):
        return self.model(x)

    def predict(self, img_fn):
        with torch.inference_mode():
            img = Image.open(img_fn).convert('RGB')
            img = data_transform(img)
            img = img.to(device)

            result = self.model(img.unsqueeze(0))
            result = torch.argmax(result, dim=1)

        return result.item()