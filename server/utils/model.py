from PIL import Image
import numpy as np

import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class FireClassifier(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()
        # self.model = models.efficientnet_b0()
        self.model = models.resnet18()
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.require_grad = False
        
        self.model.fc = nn.Linear(512, num_classes)
        self.model.to(device)
    
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