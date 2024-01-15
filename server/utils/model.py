from PIL import Image, ImageDraw
import numpy as np

import torch
import mlflow.pytorch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from ultralytics import YOLO
from PIL import Image

import json
from .helper import read_file, update_file

mlflow.set_tracking_uri("http://linux:5000")
mlflow.set_experiment("fire-detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

cfg = read_file("config.json")
server_cfg = cfg["server"]
server_url = f"http://{server_cfg['host']}:{server_cfg['port']}"    
node_red_url = f"http://{cfg['node_red']['host']}:{cfg['node_red']['port']}"
camera_url = f"http://{cfg['esp32_cam']['host']}"

class Model(object):
    def __init__(self, num_classes=2, model_path="./model", model_name='yolo'):
        if model_name == 'resnet':
            try:
                print(f"-> Reloading model from {model_path}...")
                self.model = ReloadModel(model_path)
            except:
                print(f"-> Reloading fail, using default Resnet18")
                self.model = Resnet18(num_classes, freeze=True)
            self.model.eval()
        elif model_name == 'yolo':
            self.model = YoloV8(model_path)


    def predict(self, img_fn):
        print(img_fn)
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

class YoloV8():
    def __init__(self, model_path="./model/yolo"):
        super().__init__()
        self.model = YOLO(model_path)

    def predict(self, img_fn):
        with torch.inference_mode():
            results = self.model.predict(img_fn)

        image = Image.open(img_fn)
        draw = ImageDraw.Draw(image)

        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        config = read_file("config.json")
        img_dir = f"{config['server']['image_dir']}/yolo_output"
        output_path = f"{img_dir}/{img_fn.split('/')[-1].split('.')[0]}_yolo.jpg"
        
        print('-> Saving yolo output to', output_path)
        img_url = f"{server_url}/image/{output_path.split('/')[-1]}"
        image.save(output_path)

        return int(len(results[0].boxes.cls) > 0), img_url
        